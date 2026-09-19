"""CPU tests for zero-DY evidence, launch isolation, and helper instrumentation."""

import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from scripts import repro_weighted_layer_norm_zero_dy as control


def inputs_fixture(rows=73):
    generator = torch.Generator().manual_seed(947)
    x = torch.rand((rows, 512), generator=generator).to(torch.bfloat16) * 2 - 1
    return {"x": x, "dy": torch.zeros_like(x), "weight": torch.ones(512, dtype=x.dtype),
            "bias": torch.zeros(512, dtype=x.dtype), "mean": x.float().mean(1),
            "rstd": torch.rsqrt(x.float().var(1, correction=0) + 1e-6)}


def report_fixture():
    return {"block_n": 8, "max_vgpr": None, "iteration": 7, "tile_num": 8,
            "selected_norm_configs": control.launch_configs(8), "check_before_reduction": True}


def compiled(role):
    return SimpleNamespace(hash="compiled_" + role, name="kernel_" + role,
                           metadata={"num_warps": 1}, asm={"amdgcn": "test kernel"},
                           n_regs=16, n_spills=0)


class FakeConfig:
    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1):
        self.kwargs, self.num_warps = kwargs, num_warps
        self.num_stages, self.num_ctas = num_stages, num_ctas


class FakeKernel:
    """Model autotuner dispatch while executing only caller-supplied CPU code."""
    def __init__(self, run):
        self.configs = [FakeConfig({"old": True})]
        self.cache = {"old key": "old winner"}
        self.fn = SimpleNamespace(run=run)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            config = self.configs[0]
            options = {**kwargs, **config.kwargs, "num_warps": config.num_warps,
                       "num_stages": config.num_stages, "num_ctas": config.num_ctas,
                       "warmup": False}
            options["grid"] = grid(options) if callable(grid) else grid
            return self.fn.run(*args, **options)
        return launch


FAKE_TRITON = SimpleNamespace(Config=FakeConfig, cdiv=lambda a, b: (a + b - 1) // b,
                              next_power_of_2=lambda n: 1 << (n - 1).bit_length())


def production_cpu_helper(module):
    """Execute the real public helper and implementation, replacing only kernels."""
    path = Path(control.__file__).parents[1] / "generative_recommenders/ops/triton/triton_layer_norm.py"
    source = ast.parse(path.read_text())
    names = {"_triton_weighted_layer_norm_bwd_impl", "triton_weighted_layer_norm_bwd"}
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for function in functions:
        function.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           *functions], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {"torch": torch, "triton": FAKE_TRITON,
                 "_weighted_layer_norm_bwd_dx": module._weighted_layer_norm_bwd_dx,
                 "_layer_norm_bwd_dwdb": module._layer_norm_bwd_dwdb}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["triton_weighted_layer_norm_bwd"]


class ZeroOracleTest(unittest.TestCase):
    def test_nan_inf_and_finite_nonzero_across_chunk_boundaries(self):
        value = torch.zeros((5, 7), dtype=torch.bfloat16)
        value[0, 0] = -0.0
        value.view(-1)[11] = float("nan")
        value.view(-1)[12] = float("inf")
        value.view(-1)[-1] = -3
        probe = control.zero_probe(torch, value, chunk_elements=12)
        self.assertEqual(int(probe["count"]), 3)
        self.assertEqual(int(probe["index"]), 11)
        failures = control.scalar_rows(torch, [{"iteration": 4, "outputs": {"d_x": probe}}])
        self.assertEqual(failures, [{"iteration": 4, "output": "d_x", "nonzero_elements": 3,
                                     "first_coordinate": [1, 4], "first_value": "nan"}])
        self.assertTrue(torch.isnan(probe["sample"][0, 4]))

    def test_exact_zero_and_signed_zero_pass_including_final_partial_chunk(self):
        value = torch.zeros((3, 7), dtype=torch.float32)
        value[-1, -1] = -0.0
        probe = control.zero_probe(torch, value, chunk_elements=8)
        self.assertEqual(int(probe["count"]), 0)
        self.assertEqual(int(probe["index"]), value.numel())
        self.assertEqual(control.scalar_rows(torch, [{"iteration": 1, "outputs": {"d_x": probe}}]), [])
        value[-1, -1] = float("-inf")
        failure, = control.scalar_rows(torch, [{"iteration": 2, "outputs": {
            "d_x": control.zero_probe(torch, value, chunk_elements=8)}}])
        self.assertEqual(failure["first_coordinate"], [2, 6])
        self.assertEqual(failure["first_value"], "-inf")

    def test_probe_copies_are_independent_of_later_writes(self):
        for shape in ((4, 9), (9,)):
            value = torch.zeros(shape)
            value.reshape(-1)[-2] = 7
            probe = control.zero_probe(torch, value, chunk_elements=5)
            original = weakref.ref(value)
            value.fill_(13)
            self.assertEqual(float(probe["value"]), 7)
            self.assertEqual(int((probe["sample"] == 7).sum()), 1)
            self.assertFalse(bool((probe["sample"] == 13).any()))
            del value
            self.assertIsNone(original())

    def test_noncontiguous_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            control.zero_probe(torch, torch.zeros(4, 7).t())

    def test_byte_guard_detects_signed_zero_and_tail_mutation(self):
        value = torch.zeros(9, dtype=torch.bfloat16)
        snapshot = value.clone()
        with patch.object(control, "CHUNK_ELEMENTS", 7):
            self.assertTrue(control.all_input_bytes_equal(torch, value, snapshot))
            value[-1] = -0.0
            self.assertFalse(control.all_input_bytes_equal(torch, value, snapshot))


class DirectDXTest(unittest.TestCase):
    def test_production_inner_call_has_poisoned_outputs_grid_shapes_and_strides(self):
        inputs = inputs_fixture(19)
        inputs["x"] = torch.zeros((38, 512), dtype=torch.bfloat16)[::2]
        inputs["dy"] = torch.zeros((38, 512), dtype=torch.bfloat16)[1::2]
        tiles = control.tile_count(19, 3)
        self.assertEqual(tiles, 4)
        seen = []

        def launch(*args, **kwargs):
            dx, dy, dw, db, x, weight, bias, mean, rstd = args[:9]
            for value in (dx, dw, db):
                self.assertTrue(bool(torch.isnan(value).all()))
            self.assertIs(x, inputs["x"])
            self.assertIs(dy, inputs["dy"])
            self.assertIs(weight, inputs["weight"])
            self.assertIs(bias, inputs["bias"])
            self.assertIs(mean, inputs["mean"])
            self.assertIs(rstd, inputs["rstd"])
            self.assertEqual(args[9:], (dx.stride(0), 1024, 1024, 512, 1e-6))
            self.assertEqual(tuple(dw.shape), (tiles, 512))
            self.assertEqual(dw.dtype, torch.float32)
            self.assertEqual(db.dtype, torch.float32)
            self.assertEqual(kwargs["grid"], (tiles,))
            self.assertEqual(kwargs["N"], 19)
            self.assertEqual(kwargs["BLOCK_D"], 512)
            self.assertEqual(kwargs["BLOCK_N"], 8)
            self.assertFalse(kwargs["IS_SWISH"])
            self.assertFalse(kwargs["warmup"])
            dx.zero_()
            dw.zero_()
            # Simulate one missing final write: the poison must remain observable.
            db.view(-1)[:-1].zero_()
            seen.append(True)
            return compiled("dx")

        kernel = FakeKernel(launch)
        kernel.run = Mock(side_effect=AssertionError("autotuner must be bypassed"))
        report = report_fixture()
        for _ in range(2):
            dx, dw, db = control.direct_dx(torch, kernel, inputs, control.launch_configs(8)["dx"], tiles, report)
            self.assertTrue(bool((dx == 0).all()))
            self.assertTrue(bool((dw == 0).all()))
            probe = control.zero_probe(torch, db, chunk_elements=17)
            self.assertEqual(int(probe["count"]), 1)
            self.assertEqual(int(probe["index"]), tiles * 512 - 1)
        self.assertEqual(len(seen), 2)
        kernel.run.assert_not_called()
        self.assertEqual(len(report["actual_norm_launches"]), 1)

    def test_tile_count_handles_small_rows_and_multiprocessor_cap(self):
        self.assertEqual(control.tile_count(1, 4), 1)
        self.assertEqual(control.tile_count(31, 4), 7)
        self.assertEqual(control.tile_count(2468813, 304), 304 * 8)


class HelperScopeTest(unittest.TestCase):
    def test_real_public_helper_poisoning_callback_order_and_no_retained_allocations(self):
        inputs, pending, report = inputs_fixture(), [], report_fixture()
        order, weak_outputs = [], []

        def dx_launch(*args, **kwargs):
            order.append("dx")
            self.assertEqual(kwargs["grid"], (8,))
            self.assertEqual(tuple(args[2].shape), (8, 512))
            for index in (0, 2, 3):
                self.assertTrue(bool(torch.isnan(args[index]).all()))
                weak_outputs.append(weakref.ref(args[index]))
                args[index].zero_()
            return compiled("dx")

        def after_dx(*values):
            order.append("before_reduction")
            control.append_before_reduction_probes(torch, pending, report, *values)

        def reduction_launch(*args, **kwargs):
            order.append("reduction")
            self.assertEqual(len(pending), 1)
            # With one simulated multiprocessor the production reduction uses
            # four 128-column programs, rather than the full GPU's 4-column tile.
            self.assertEqual(kwargs["grid"], (4,))
            self.assertEqual(kwargs["BLOCK_D"], 128)
            self.assertTrue(bool((args[0] == 0).all()))
            self.assertTrue(bool((args[1] == 0).all()))
            for index in (2, 3):
                self.assertTrue(bool(torch.isnan(args[index]).all()))
                args[index].zero_()
            return compiled("reduction")

        module = SimpleNamespace(_weighted_layer_norm_bwd_dx=FakeKernel(dx_launch),
                                 _layer_norm_bwd_dwdb=FakeKernel(reduction_launch))
        helper = production_cpu_helper(module)
        with patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=1)):
            with control.helper_launch_scope(FAKE_TRITON, module, control.launch_configs(8), report, after_dx=after_dx):
                outputs = helper(**inputs, learnable=True, eps=1e-6, BLOCK_D=512)
        self.assertEqual(order, ["dx", "before_reduction", "reduction"])
        self.assertEqual(control.scalar_rows(torch, pending), [])
        self.assertEqual(pending[0]["observation_point"], "before_reduction")
        self.assertEqual(report["phase"], "input_norm_helper_reduction")
        for output in outputs:
            self.assertTrue(bool((output == 0).all()))
        del output, outputs
        self.assertTrue(all(reference() is None for reference in weak_outputs))

    def test_warmup_skips_poison_and_callback_and_none_launch_result_still_probes(self):
        callback, report = Mock(), report_fixture()
        values = [torch.full((2, 4), 3.0) for _ in range(4)]

        def dx_launch(*args, **kwargs):
            for index in (0, 2, 3):
                if kwargs.get("warmup"):
                    self.assertTrue(bool((args[index] == 3).all()))
                else:
                    self.assertTrue(bool(torch.isnan(args[index]).all()))
                    args[index].zero_()
            return None

        module = SimpleNamespace(_weighted_layer_norm_bwd_dx=FakeKernel(dx_launch),
                                 _layer_norm_bwd_dwdb=FakeKernel(lambda *a, **kw: None))
        with control.helper_launch_scope(FAKE_TRITON, module, control.launch_configs(8), report, after_dx=callback):
            module._weighted_layer_norm_bwd_dx.fn.run(*values, warmup=True)
            callback.assert_not_called()
            module._weighted_layer_norm_bwd_dx.fn.run(*values, warmup=False)
        callback.assert_called_once_with(values[0], values[2], values[3])
        self.assertNotIn("actual_norm_launches", report)

    def test_wrappers_configs_and_caches_restore_after_kernel_or_callback_error(self):
        for failure_at in ("dx", "callback", "reduction"):
            with self.subTest(failure_at=failure_at):
                def dx_launch(*args, **kwargs):
                    if failure_at == "dx":
                        raise RuntimeError("injected failure")
                    for index in (0, 2, 3):
                        args[index].zero_()
                    return compiled("dx")

                def callback(*args):
                    if failure_at == "callback":
                        raise RuntimeError("injected failure")

                def reduction_launch(*args, **kwargs):
                    raise RuntimeError("injected failure")

                dx, reduction = FakeKernel(dx_launch), FakeKernel(reduction_launch)
                original_configs = (dx.configs, reduction.configs)
                module = SimpleNamespace(_weighted_layer_norm_bwd_dx=dx, _layer_norm_bwd_dwdb=reduction)
                helper = production_cpu_helper(module)
                with patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(multi_processor_count=1)):
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        with control.helper_launch_scope(FAKE_TRITON, module, control.launch_configs(8), report_fixture(), after_dx=callback):
                            self.assertEqual(dx.cache, {})
                            self.assertEqual(reduction.cache, {})
                            helper(**inputs_fixture(), learnable=True, eps=1e-6, BLOCK_D=512)
                self.assertIs(dx.fn.run, dx_launch)
                self.assertIs(reduction.fn.run, reduction_launch)
                self.assertIs(dx.configs, original_configs[0])
                self.assertIs(reduction.configs, original_configs[1])
                self.assertEqual(dx.cache, {"old key": "old winner"})
                self.assertEqual(reduction.cache, {"old key": "old winner"})


class CompactEvidenceTest(unittest.TestCase):
    def write_artifact(self, directory, pending, *, stage="helper", inputs=None, snapshots=None):
        inputs = inputs or inputs_fixture()
        snapshots = snapshots or {name: value.clone() for name, value in inputs.items()}
        args = SimpleNamespace(rows=inputs["x"].shape[0], block_n=8, seed=947, stage=stage,
                               report=Path(directory) / "report.json")
        report = report_fixture()
        report["tile_num"] = control.tile_count(args.rows, 1)
        failures = control.scalar_rows(torch, pending)
        checks = {name: bool(control.all_input_bytes_equal(torch, value, snapshots[name]))
                  for name, value in inputs.items()}
        result = control.compact_failure(torch, args, report, failures, pending, inputs, snapshots, checks)
        return result, torch.load(result["path"], map_location="cpu", weights_only=False)

    def test_same_iteration_before_and_after_evidence_stays_distinct(self):
        before, after = torch.zeros((73, 512)), torch.zeros((73, 512))
        before[2, 4], after[2, 4] = float("nan"), 17
        pending = [{"iteration": 7, "observation_point": point,
                    "outputs": {"d_x": control.zero_probe(torch, value, chunk_elements=100)}}
                   for point, value in (("before_reduction", before), ("after_helper", after))]
        before.zero_()
        after.zero_()
        with tempfile.TemporaryDirectory() as directory:
            result, artifact = self.write_artifact(directory, pending)
        self.assertLess(result["bytes"], control.MAX_ARTIFACT_BYTES)
        self.assertTrue(artifact["check_before_reduction"])
        first, second = artifact["outputs"]
        self.assertEqual(first["observation_point"], "before_reduction")
        self.assertEqual(second["observation_point"], "after_helper")
        self.assertTrue(torch.isnan(first["sample"][0, 4]))
        self.assertEqual(float(second["sample"][0, 4]), 17)
        self.assertEqual(artifact["input_sample_rows"], [1, 2, 3])
        self.assertEqual(tuple(artifact["inputs"]["x"]["at_checkpoint"].shape), (3, 512))

    def test_partial_coordinate_is_tile_and_samples_a_representative_input_row(self):
        partial = torch.zeros((8, 512))
        partial[3, 263] = float("inf")
        pending = [{"iteration": 7, "outputs": {
            "partial_d_norm_weight": control.zero_probe(torch, partial, chunk_elements=100)}}]
        with tempfile.TemporaryDirectory() as directory:
            _, artifact = self.write_artifact(directory, pending, stage="dx")
        self.assertEqual(artifact["input_sample_rows"], [23, 24, 25])
        self.assertIn("k*tile_num", artifact["partial_tile_input_mapping"])
        evidence, = artifact["outputs"]
        self.assertEqual(evidence["first_coordinate"], [3, 263])
        self.assertEqual(evidence["sample_meaning"], "first failing partial-gradient tile")
        self.assertEqual(evidence["partial_tile_provenance"], {
            "tile": 3, "has_input_rows": True, "representative_input_row": 24,
        })
        self.assertEqual(float(evidence["sample"][0, 263]), float("inf"))

    def test_inactive_partial_tile_does_not_claim_an_unrelated_input_row(self):
        # Production creates seven partial tiles for N=31, but BLOCK_N=8 only
        # assigns rows to tiles 0..3. Tile 6 must still write its zero partial.
        partial = torch.zeros((7, 512))
        partial[6, 263] = float("nan")
        pending = [{"iteration": 7, "outputs": {
            "partial_d_norm_bias": control.zero_probe(torch, partial)}}]
        with tempfile.TemporaryDirectory() as directory:
            _, artifact = self.write_artifact(directory, pending, stage="dx", inputs=inputs_fixture(31))
        self.assertEqual(artifact["tile_num"], 7)
        self.assertEqual(artifact["input_sample_rows"], [0])  # Generic fallback sample.
        self.assertNotIn(30, artifact["input_sample_rows"])
        evidence, = artifact["outputs"]
        self.assertEqual(evidence["first_coordinate"], [6, 263])
        self.assertEqual(evidence["partial_tile_provenance"], {
            "tile": 6, "has_input_rows": False, "representative_input_row": None,
        })

    def test_input_mutation_records_byte_offset_and_independent_initial_sample(self):
        inputs = inputs_fixture()
        snapshots = {name: value.clone() for name, value in inputs.items()}
        inputs["dy"][-1, -1] = 1
        pending = [{"iteration": 7, "outputs": {"d_x": control.zero_probe(torch, torch.zeros((73, 512)))}}]
        with tempfile.TemporaryDirectory() as directory:
            _, artifact = self.write_artifact(directory, pending, inputs=inputs, snapshots=snapshots)
        mutation = artifact["input_mutations"]["dy"]
        self.assertEqual(mutation["first_element_offset"], 73 * 512 - 1)
        self.assertIn(mutation["first_byte_offset"], (2 * (73 * 512 - 1), 2 * (73 * 512 - 1) + 1))
        row = artifact["input_sample_rows"].index(72)
        self.assertEqual(float(artifact["inputs"]["dy"]["initial_snapshot"][row, -1]), 0)
        self.assertEqual(float(artifact["inputs"]["dy"]["at_checkpoint"][row, -1]), 1)

    def test_tensor_budget_rejects_before_serializing(self):
        value = torch.zeros(512)
        value[0] = 1
        pending = [{"iteration": 7, "outputs": {"d_norm_weight": control.zero_probe(torch, value)}}]
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "MAX_ARTIFACT_BYTES", 1024):
            with patch.object(torch, "save") as save, self.assertRaisesRegex(RuntimeError, "tensor byte budget"):
                self.write_artifact(directory, pending)
            save.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_oversized_serialized_artifact_is_removed(self):
        value = torch.zeros(512)
        value[0] = 1
        pending = [{"iteration": 7, "outputs": {"d_norm_weight": control.zero_probe(torch, value)}}]
        def oversized_save(payload, handle):
            handle.write(b"x" * (65536 + 1))
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "MAX_ARTIFACT_BYTES", 65536):
            with patch.object(torch, "save", side_effect=oversized_save), self.assertRaisesRegex(RuntimeError, "was removed"):
                self.write_artifact(directory, pending)
            self.assertEqual(list(Path(directory).iterdir()), [])


class ArgumentsTest(unittest.TestCase):
    def test_before_reduction_is_optional_and_helper_only(self):
        args = control.arguments(["--stage", "helper", "--block-n", "8", "--report", "report.json"])
        self.assertFalse(args.check_before_reduction)
        args = control.arguments(["--stage", "helper", "--block-n", "8", "--report", "report.json", "--check-before-reduction"])
        self.assertTrue(args.check_before_reduction)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            control.arguments(["--stage", "dx", "--block-n", "8", "--report", "report.json", "--check-before-reduction"])

    def test_vgpr_cap_affects_dx_only_and_requires_hip(self):
        original = control.launch_configs(8)
        capped = control.launch_configs(8, max_vgpr=256)
        self.assertEqual(capped["dx"]["llvm_fn_attrs"], "amdgpu-num-vgpr=256")
        self.assertEqual(capped["reduction"], original["reduction"])
        with self.assertRaisesRegex(ValueError, "ROCm"):
            control.launch_configs(8, max_vgpr=256, hip=False)


def tearDownModule():
    if torch.cuda.is_initialized():
        raise AssertionError("CPU tests must not initialize a GPU")


if __name__ == "__main__":
    unittest.main()
