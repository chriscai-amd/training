"""CPU checks for the forward probe's saved-statistics path and evidence.

The production public helper performs allocations and statistic copies. Only
the Triton launch is replaced by CPU code; importing the full GPU module is
unnecessary. These tests must never initialize CUDA/ROCm.
"""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.fx._symbolic_trace import is_fx_tracing

from scripts import repro_weighted_layer_norm_zero_x as control


class FakeConfig:
    def __init__(self, block_n):
        self.kwargs = {"BLOCK_N": block_n}
        self.num_warps = 4
        self.num_stages = 1
        self.num_ctas = 1


class FakeKernel:
    """Pass a selected production-style configuration to a CPU launch stub."""

    def __init__(self, run):
        self.configs = [FakeConfig(8), FakeConfig(1)]
        self.cache = {"previous key": "previous winner"}
        self.fn = SimpleNamespace(run=run)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            config = self.configs[0]
            options = {**kwargs, **config.kwargs, "num_warps": config.num_warps,
                       "num_stages": config.num_stages, "warmup": False}
            options["grid"] = grid(options)
            return self.fn.run(*args, **options)
        return launch


def compiled():
    return SimpleNamespace(hash="cpu_forward", name="weighted_layer_norm_fwd",
                           metadata={"num_warps": 4}, n_regs=16, n_spills=0,
                           asm={"amdgcn": "CPU test launch"})


def production_cpu_helper(kernel):
    """Load the actual helper and layout helper, without GPU module imports."""
    root = Path(control.__file__).parents[1] / "generative_recommenders"
    functions = []
    for relative, wanted in (
        ("common.py", "switch_to_contiguous_if_needed"),
        ("ops/triton/triton_layer_norm.py", "triton_weighted_layer_norm_fwd"),
    ):
        source = ast.parse((root / relative).read_text())
        matches = [node for node in source.body
                   if isinstance(node, ast.FunctionDef) and node.name == wanted]
        if len(matches) != 1:
            raise AssertionError(f"expected one production helper named {wanted}")
        matches[0].decorator_list = []
        functions.extend(matches)
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *functions,
    ], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {
        "torch": torch,
        "is_fx_tracing": is_fx_tracing,
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                  next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "_weighted_layer_norm_fwd": kernel,
    }
    exec(compile(tree, str(root / "ops/triton/triton_layer_norm.py"), "exec"), namespace)
    return namespace["triton_weighted_layer_norm_fwd"]


def inputs_fixture(rows=19, *, saved=True):
    inputs = {"x": torch.zeros((rows, control.FEATURES), dtype=torch.bfloat16),
              "weight": torch.ones(control.FEATURES, dtype=torch.bfloat16),
              "bias": torch.zeros(control.FEATURES, dtype=torch.bfloat16)}
    if saved:
        inputs.update(mean=torch.zeros(rows), rstd=torch.full((rows,), 1000.0))
    return inputs


def report_fixture():
    return {"iteration": 7, "observed_launch_count": 0, "actual_compiled_launches": [],
            "diagnostic": "weighted_ln_zero_x_forward", "source": {"test": "CPU"},
            "oracle": {"y": {"expected": 0, "atol": 0},
                       "mean": {"expected": 0, "atol": 0},
                       "rstd": {"expected": 1000, "atol": 0}}}


class CPUOnlyTest(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        blocker = patch.object(torch.cuda, "_lazy_init",
                               side_effect=AssertionError("CPU test attempted GPU initialization"))
        blocker.start()
        self.addCleanup(blocker.stop)
        self.addCleanup(lambda: self.assertFalse(torch.cuda.is_initialized()))


class SavedStatisticsHelperTest(CPUOnlyTest):
    def test_real_helper_copies_statistics_before_launch_and_preserves_originals(self):
        inputs = inputs_fixture()
        # Deliberately distinct supplied statistics make accidental recomputation
        # visible to the launch stub, rather than relying only on a mode flag.
        inputs["mean"].copy_(torch.arange(19) % 3)
        inputs["rstd"].copy_(torch.arange(19) + 2)
        snapshots = {name: value.clone() for name, value in inputs.items()}
        versions = {name: value._version for name, value in inputs.items()}
        launches = []

        def launch(x, y, weight, bias, mean, rstd, *args, **kwargs):
            self.assertFalse(kwargs["COMPUTE_MEAN_AND_RSTD"])
            self.assertTrue(kwargs["TRAINING"])
            self.assertFalse(kwargs["IS_SWISH"])
            self.assertEqual(kwargs["grid"], (3,))
            self.assertTrue(bool(torch.isnan(y).all()))
            self.assertTrue(torch.equal(mean, snapshots["mean"]))
            self.assertTrue(torch.equal(rstd, snapshots["rstd"]))
            self.assertEqual(mean.dtype, torch.float32)
            self.assertEqual(rstd.dtype, torch.float32)
            original_storage = {value.untyped_storage().data_ptr() for value in inputs.values()}
            returned_storage = {value.untyped_storage().data_ptr() for value in (y, mean, rstd)}
            self.assertEqual(len(returned_storage), 3)
            self.assertTrue(original_storage.isdisjoint(returned_storage))
            y.copy_(((x.float() - mean[:, None]) * rstd[:, None]) * weight + bias)
            launches.append((y, mean, rstd))
            return compiled()

        kernel = FakeKernel(launch)
        helper = production_cpu_helper(kernel)
        report = report_fixture()
        with control.forward_launch_scope(SimpleNamespace(_weighted_layer_norm_fwd=kernel),
                                          None, report, statistics_mode="saved"):
            outputs = helper(**inputs, eps=control.EPS)
        self.assertEqual(len(launches), 1)
        for actual, observed in zip(outputs, launches[0]):
            self.assertIs(actual, observed)
        expected_y = (-snapshots["mean"][:, None] * snapshots["rstd"][:, None]).expand_as(inputs["x"])
        self.assertTrue(torch.equal(outputs[0], expected_y.to(torch.bfloat16)))
        for name, value in inputs.items():
            self.assertTrue(bool(control.input_bytes_equal(torch, value, snapshots[name])))
            self.assertEqual(value._version, versions[name])
        outputs[1].fill_(77)
        outputs[2].fill_(88)
        self.assertTrue(torch.equal(inputs["mean"], snapshots["mean"]))
        self.assertTrue(torch.equal(inputs["rstd"], snapshots["rstd"]))
        self.assertEqual(report["observed_launch_count"], 1)
        self.assertFalse(report["actual_compiled_launches"][0]["launch_config"]["COMPUTE_MEAN_AND_RSTD"])
        self.assertIs(kernel.fn.run, launch)

    def test_poison_exposes_an_unwritten_saved_mode_output_tail(self):
        inputs = inputs_fixture()

        def launch(x, y, weight, bias, mean, rstd, *args, **kwargs):
            self.assertTrue(bool(torch.isnan(y).all()))
            self.assertTrue(bool((mean == 0).all()))
            self.assertTrue(bool((rstd == 1000).all()))
            y.view(-1)[:-1].zero_()  # Simulated missing final output write.
            return compiled()

        kernel = FakeKernel(launch)
        with control.forward_launch_scope(SimpleNamespace(_weighted_layer_norm_fwd=kernel),
                                          None, report_fixture(), statistics_mode="saved"):
            outputs = production_cpu_helper(kernel)(**inputs, eps=control.EPS)
        probes = {name: control.oracle_probe(torch, value, 1000 if name == "rstd" else 0, 0,
                                             chunk_elements=700)
                  for name, value in zip(control.OUTPUTS, outputs)}
        failure, = control.scalar_failures(torch, [{"iteration": 7, "compiled_hash": "cpu_forward",
                                                    "outputs": probes}])
        self.assertEqual(failure["output"], "y")
        self.assertEqual(failure["bad_elements"], 1)
        self.assertEqual(failure["first_coordinate"], [18, 511])
        self.assertEqual(failure["first_value"], "nan")


class RecomputedStatisticsHelperTest(CPUOnlyTest):
    def test_default_mode_poisons_all_outputs_and_missing_either_stat_recomputes_both(self):
        for supplied_stat in (None, "mean", "rstd"):
            with self.subTest(supplied_stat=supplied_stat):
                inputs = inputs_fixture(saved=False)
                if supplied_stat:
                    inputs[supplied_stat] = torch.full((19,), 17.0)
                snapshots = {name: value.clone() for name, value in inputs.items()}

                def launch(x, y, weight, bias, mean, rstd, n, d, eps, *args, **kwargs):
                    self.assertTrue(kwargs["COMPUTE_MEAN_AND_RSTD"])
                    for value in (y, mean, rstd):
                        self.assertTrue(bool(torch.isnan(value).all()))
                    mean.copy_(x.float().mean(1))
                    rstd.copy_(torch.rsqrt(x.float().var(1, correction=0) + eps))
                    y.copy_((x.float() - mean[:, None]) * rstd[:, None] * weight + bias)
                    return compiled()

                kernel = FakeKernel(launch)
                report = report_fixture()
                # Omit statistics_mode to exercise the unchanged default.
                with control.forward_launch_scope(SimpleNamespace(_weighted_layer_norm_fwd=kernel), None, report):
                    y, mean, rstd = production_cpu_helper(kernel)(**inputs, eps=control.EPS)
                for value, expected, tolerance in ((y, 0, 0), (mean, 0, 0), (rstd, 1000, control.RSTD_ATOL)):
                    self.assertEqual(int(control.oracle_probe(torch, value, expected, tolerance)["count"]), 0)
                for name, value in inputs.items():
                    self.assertTrue(bool(control.input_bytes_equal(torch, value, snapshots[name])))
                self.assertTrue(report["actual_compiled_launches"][0]["launch_config"]["COMPUTE_MEAN_AND_RSTD"])


class LaunchIsolationTest(CPUOnlyTest):
    def test_wrong_mode_rejects_before_poisoning_and_restores_filtered_scope(self):
        for mode, actual_recompute in (("saved", True), ("recompute", False)):
            with self.subTest(mode=mode):
                values = [torch.full((3, 4), 17.0) for _ in range(6)]
                snapshots = [value.clone() for value in values]
                original_run = Mock(side_effect=AssertionError("wrong-mode kernel must not execute"))
                kernel = FakeKernel(original_run)
                original_configs, original_cache = kernel.configs, dict(kernel.cache)
                report = report_fixture()
                with self.assertRaisesRegex(ValueError, "expected ordinary weighted forward"):
                    with control.forward_launch_scope(SimpleNamespace(_weighted_layer_norm_fwd=kernel),
                                                      1, report, statistics_mode=mode):
                        self.assertEqual(kernel.cache, {})
                        kernel.fn.run(*values, TRAINING=True, IS_SWISH=False,
                                      COMPUTE_MEAN_AND_RSTD=actual_recompute, warmup=False)
                original_run.assert_not_called()
                self.assertIs(kernel.fn.run, original_run)
                self.assertIs(kernel.configs, original_configs)
                self.assertEqual(kernel.cache, original_cache)
                self.assertEqual(report["observed_launch_count"], 0)
                for value, snapshot in zip(values, snapshots):
                    self.assertTrue(bool(control.input_bytes_equal(torch, value, snapshot)))

    def test_warmup_does_not_poison_or_record_a_real_launch(self):
        for mode in ("saved", "recompute"):
            with self.subTest(mode=mode):
                values = [torch.full((3, 4), 17.0) for _ in range(6)]

                def launch(*args, **kwargs):
                    self.assertTrue(kwargs["warmup"])
                    self.assertTrue(all(bool((value == 17).all()) for value in args))
                    return compiled()

                kernel = FakeKernel(launch)
                report = report_fixture()
                with control.forward_launch_scope(SimpleNamespace(_weighted_layer_norm_fwd=kernel),
                                                  None, report, statistics_mode=mode):
                    kernel.fn.run(*values, TRAINING=True, IS_SWISH=False,
                                  COMPUTE_MEAN_AND_RSTD=mode == "recompute", warmup=True)
                self.assertEqual(report["observed_launch_count"], 0)
                self.assertEqual(report["actual_compiled_launches"], [])


class CompactSavedEvidenceTest(CPUOnlyTest):
    def test_statistic_mutations_include_rows_without_serializing_full_statistics(self):
        inputs = inputs_fixture(rows=4096)
        snapshots = {name: value.clone() for name, value in inputs.items()}
        inputs["mean"][2047] = 1
        inputs["rstd"][-1] = 999
        y = torch.zeros_like(inputs["x"])
        y[2, 17] = float("nan")
        pending = [{"iteration": 7, "compiled_hash": "cpu_forward", "outputs": {
            "y": control.oracle_probe(torch, y, 0, 0),
        }}]
        failures = control.scalar_failures(torch, pending)
        byte_checks = {name: bool(control.input_bytes_equal(torch, value, snapshots[name]))
                       for name, value in inputs.items()}
        # Four full FP32 statistic arrays would require 64 KiB, exceeding the
        # 50 KiB tensor budget. A correctly sampled artifact fits comfortably.
        cap = 100 * 1024
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "MAX_ARTIFACT_BYTES", cap):
            args = SimpleNamespace(rows=4096, statistics_mode="saved", report=Path(directory) / "report.json")
            result = control.compact_failure(torch, args, report_fixture(), failures, pending,
                                             inputs, snapshots, byte_checks)
            artifact = torch.load(result["path"], map_location="cpu", weights_only=False)
            self.assertLess(result["bytes"], cap)
        self.assertEqual(artifact["statistics_mode"], "saved")
        rows = artifact["input_sample_rows"]
        self.assertEqual(rows, [1, 2, 3, 2046, 2047, 2048, 4094, 4095])
        for name in ("mean", "rstd"):
            for phase in ("at_checkpoint", "initial_snapshot"):
                value = artifact["inputs"][name][phase]
                self.assertEqual(tuple(value.shape), (len(rows),))
                self.assertEqual(value.untyped_storage().nbytes(), len(rows) * 4)
        for name, changed_row, initial, changed in (("mean", 2047, 0, 1), ("rstd", 4095, 1000, 999)):
            evidence = artifact["inputs"][name]
            position = rows.index(changed_row)
            self.assertEqual(float(evidence["initial_snapshot"][position]), initial)
            self.assertEqual(float(evidence["at_checkpoint"][position]), changed)
            mutation = artifact["input_mutations"][name]
            self.assertGreater(mutation["different_bytes"], 0)
            self.assertEqual(mutation["first_byte_offset"] // 4, changed_row)
        failure, = artifact["outputs"]
        self.assertEqual(failure["first_coordinate"], [2, 17])
        self.assertTrue(torch.isnan(failure["sample"][0, 17]))
        self.assertEqual(tuple(artifact["inputs"]["x"]["at_checkpoint"].shape), (len(rows), 512))


def tearDownModule():
    if torch.cuda.is_initialized():
        raise AssertionError("CPU tests must not initialize a GPU")


if __name__ == "__main__":
    unittest.main()
