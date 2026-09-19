"""CPU mathematical and original-stage fidelity tests; never initialize a GPU."""

import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from scripts import replay_nan_tripwire as replay
from scripts import repro_preprocess_backward_stages as stages


def projection_fixture(is_y_1d=True):
    generator = torch.Generator().manual_seed(81)
    # Nonzero offsets and feature strides verify logical slicing at every axis.
    return {
        "x": torch.randn(9, 12, generator=generator, dtype=torch.float64)[1:8, 1:11:2],
        "w": torch.randn(11, 9, generator=generator, dtype=torch.float64)[1:11:2, 1:7],
        "dz": torch.randn(9, 14, generator=generator, dtype=torch.float64)[1:8, 1:13:2],
        "is_y_1d": is_y_1d,
    }


def projection_autograd(inputs):
    x, w = [inputs[name].clone().requires_grad_() for name in ("x", "w")]
    bias_shape = (w.shape[1],) if inputs["is_y_1d"] else (x.shape[0], w.shape[1])
    bias = torch.zeros(bias_shape, dtype=x.dtype, requires_grad=True)
    return torch.autograd.grad(torch.addmm(bias, x, w), (x, w, bias), inputs["dz"])


def norm_fixture(learnable=True):
    generator = torch.Generator().manual_seed(83)
    x = torch.randn(7, 9, generator=generator, dtype=torch.float64)[:, 1:8]
    eps = 1e-5
    return {
        "x": x, "dy": torch.randn(x.shape, generator=generator, dtype=torch.float64),
        "weight": torch.randn(7, generator=generator, dtype=torch.float64) if learnable else None,
        "bias": torch.randn(7, generator=generator, dtype=torch.float64) if learnable else None,
        "mean": x.mean(dim=1), "rstd": torch.rsqrt(x.var(dim=1, correction=0) + eps),
        "learnable": learnable, "eps": eps, "BLOCK_D": 8,
    }


def norm_autograd(inputs):
    values = [inputs[name].clone().requires_grad_() for name in ("x", "weight", "bias")
              if inputs[name] is not None]
    x = values[0]
    weight, bias = values[1:] if inputs["learnable"] else (None, None)
    result = F.layer_norm(x, (x.shape[1],), weight, bias, inputs["eps"])
    gradients = torch.autograd.grad(result, values, inputs["dy"])
    return gradients if inputs["learnable"] else (gradients[0], None, None)


def collect_reference(inputs, stage, rows=3, columns=2, **kwargs):
    names = stages.STAGES[stage]["outputs"]
    shapes = stages.expected_shapes(inputs, stage)
    outputs = {name: None if shape is None else torch.empty(shape, dtype=torch.float64)
               for name, shape in zip(names, shapes)}
    counts = {name: None if shape is None else torch.zeros(shape, dtype=torch.int32)
              for name, shape in zip(names, shapes)}
    for name, selection, value in stages.reference_chunks(inputs, stage, rows, columns, **kwargs):
        outputs[name][selection] = value
        counts[name][selection] += 1
    for count in counts.values():
        if count is not None and not bool((count == 1).all()):
            raise AssertionError("reference must cover every element exactly once")
    return tuple(outputs.values())


class IndependentReferenceTest(unittest.TestCase):
    def test_projection_all_gradients_match_native_autograd_with_tail_tiles(self):
        for is_y_1d in (True, False):
            inputs = projection_fixture(is_y_1d)
            with self.subTest(is_y_1d=is_y_1d):
                for actual, expected in zip(collect_reference(inputs, "input_projection"), projection_autograd(inputs)):
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_norm_matches_native_autograd_with_and_without_parameters(self):
        for learnable in (True, False):
            inputs = norm_fixture(learnable)
            with self.subTest(learnable=learnable):
                for actual, expected in zip(collect_reference(inputs, "input_norm"), norm_autograd(inputs)):
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_projection_autocast_rounds_gemm_operands_but_not_bias_reduction(self):
        inputs = {name: value.float() if isinstance(value, torch.Tensor) else value
                  for name, value in projection_fixture().items()}
        dx, dw, db = collect_reference(inputs, "input_projection", mm_input_dtype=torch.bfloat16)
        x, w, dz = [inputs[name].bfloat16().double() for name in ("x", "w", "dz")]
        torch.testing.assert_close(dx, dz @ w.t(), rtol=0, atol=0)
        torch.testing.assert_close(dw, x.t() @ dz, rtol=0, atol=0)
        torch.testing.assert_close(db, inputs["dz"].double().sum(dim=0), rtol=0, atol=0)
        self.assertFalse(torch.equal(db, dz.sum(dim=0)))

    def test_norm_uses_captured_statistics_without_recomputing(self):
        inputs = norm_fixture()
        before = collect_reference(inputs, "input_norm")
        inputs["rstd"] = inputs["rstd"] * 2
        inputs["mean"] = inputs["mean"] + 0.5
        after = collect_reference(inputs, "input_norm")
        self.assertFalse(torch.equal(before[0], after[0]))
        self.assertFalse(torch.equal(before[1], after[1]))
        torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
        inputs["eps"] = 7
        for actual, expected in zip(collect_reference(inputs, "input_norm"), after):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reference_detects_finite_tail_corruption_in_each_gradient(self):
        for stage, fixture, autograd in (
            ("input_projection", projection_fixture, projection_autograd),
            ("input_norm", norm_fixture, norm_autograd),
        ):
            inputs = fixture()
            for index, name in enumerate(stages.STAGES[stage]["outputs"]):
                with self.subTest(stage=stage, output=name):
                    outputs = [tensor.clone() for tensor in autograd(inputs)]
                    outputs[index].reshape(-1)[-1] += 1e10
                    result = stages.compare_reference(inputs, stage, outputs, rows=3, columns=2, rtol=1e-10, atol=1e-10)
                    self.assertFalse(result["passed"])
                    self.assertEqual(result["outputs"][name]["mismatched_elements"], 1)
                    self.assertEqual(result["outputs"][name]["nonfinite_elements"], 0)
                    for info in result["outputs"].values():
                        self.assertEqual(info["elements"], info["expected_elements"])

    def test_reference_rejects_nonfinite_and_wrong_output_shapes(self):
        inputs = projection_fixture()
        outputs = list(projection_autograd(inputs))
        outputs[0][0, 0] = float("inf")
        result = stages.compare_reference(inputs, "input_projection", outputs, rows=3, columns=2, rtol=0, atol=0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["outputs"]["d_normed_x"]["nonfinite_elements"], 1)
        outputs[0] = outputs[0][:1]
        with self.assertRaisesRegex(ValueError, "output shape"):
            stages.compare_reference(inputs, "input_projection", outputs, rows=3, rtol=0, atol=0)

    def test_zero_rows_still_covers_complete_parameter_gradients(self):
        inputs = projection_fixture()
        inputs["x"], inputs["dz"] = inputs["x"][:0], inputs["dz"][:0]
        outputs = collect_reference(inputs, "input_projection")
        result = stages.compare_reference(inputs, "input_projection", outputs, rows=3, rtol=0, atol=0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["outputs"]["d_normed_x"]["elements"], 0)
        self.assertTrue(bool((outputs[1] == 0).all()))


def encode_capture(inputs, outputs, stage):
    storages, indices = [], {}

    def encode(value):
        if not isinstance(value, torch.Tensor):
            return value
        storage = value.untyped_storage()
        key = storage._cdata
        if key not in indices:
            indices[key] = len(storages)
            storages.append(torch.empty(0, dtype=torch.uint8).set_(
                storage, 0, (storage.nbytes(),), (1,),
            ).clone())
        return {"__tripwire_tensor__": indices[key], "dtype": str(value.dtype), "device": "cpu",
                "shape": tuple(value.shape), "stride": tuple(value.stride()),
                "storage_offset": value.storage_offset()}

    schema = stages.STAGES[stage]
    return {"format_version": 1, "storages": storages, "payload": {
        "replay": None,
        "stage": {"format_version": 1, "name": stage,
                  "settings": {name: inputs[name] for name in schema["settings"]},
                  "parent_replay": {
                      "class": "_HSTUPreprocessAndAttentionFunction", "direction": "backward",
                      # Decoding parent tensors would fail, even without GPU access.
                      "saved_tensors": [{"__tripwire_tensor__": 999999, "device": "cuda:0"}],
                  }},
        "inputs": {name: encode(inputs[name]) for name in schema["inputs"]},
        "outputs": {name: encode(value) for name, value in zip(schema["outputs"], outputs)},
    }}


class OriginalStageFidelityTest(unittest.TestCase):
    def test_stage_decode_preserves_offsets_strides_and_skips_parent(self):
        for stage, inputs, outputs in (
            ("input_projection", projection_fixture(), projection_autograd(projection_fixture())),
            ("input_norm", norm_fixture(False), norm_autograd(norm_fixture(False))),
        ):
            capture = encode_capture(inputs, outputs, stage)
            with self.subTest(stage=stage), patch.object(replay, "replay_once", side_effect=AssertionError("parent replay forbidden")):
                fixed, original = stages.original_stage_inputs(capture, stage, "cpu")
            for name in stages.STAGES[stage]["inputs"]:
                if inputs[name] is None:
                    self.assertIsNone(fixed[name])
                else:
                    self.assertEqual(fixed[name].stride(), inputs[name].stride())
                    self.assertEqual(fixed[name].storage_offset(), inputs[name].storage_offset())
                    torch.testing.assert_close(fixed[name], inputs[name])
            for name in stages.STAGES[stage]["settings"]:
                self.assertEqual(fixed[name], inputs[name])
            result = stages.compare_reference(fixed, stage, tuple(original[name] for name in stages.STAGES[stage]["outputs"]),
                                              rows=3, columns=2, rtol=1e-10, atol=1e-10)
            self.assertTrue(result["passed"])

    def test_input_aliases_survive_but_original_output_alias_is_frozen(self):
        backing = torch.arange(48.0).reshape(8, 6)
        inputs = {"x": backing[1:5, :3], "w": torch.eye(3), "dz": backing[2:6, :3], "is_y_1d": False}
        capture = encode_capture(inputs, (inputs["x"], torch.zeros(3, 3), inputs["dz"]), "input_projection")
        fixed, original = stages.original_stage_inputs(capture, "input_projection", "cpu")
        self.assertEqual(fixed["x"].untyped_storage()._cdata, fixed["dz"].untyped_storage()._cdata)
        fixed["x"][1, 0] = -100
        self.assertEqual(float(fixed["dz"][0, 0]), -100)
        torch.testing.assert_close(original["d_normed_x"], inputs["x"])
        torch.testing.assert_close(original["d_uvqk_bias"], inputs["dz"])

    def test_wrong_parent_stage_or_schema_is_rejected(self):
        inputs = projection_fixture()
        for change in ("stage", "parent", "inputs", "settings"):
            capture = encode_capture(inputs, projection_autograd(inputs), "input_projection")
            if change == "stage":
                capture["payload"]["stage"]["name"] = "gradient_mm"
            elif change == "parent":
                capture["payload"]["stage"]["parent_replay"]["class"] = "HSTUComputeOutputFunction"
            elif change == "inputs":
                capture["payload"]["inputs"]["parent_x"] = None
            else:
                capture["payload"]["stage"]["settings"]["BLOCK_D"] = 8
            with self.subTest(change=change), self.assertRaises(ValueError):
                stages.original_stage_inputs(capture, "input_projection", "cpu")

    def test_direct_projection_invokes_actual_production_helper_body(self):
        path = Path(stages.__file__).parents[1] / "generative_recommenders/ops/triton/triton_addmm.py"
        module = ast.parse(path.read_text())
        helper = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "triton_addmm_bwd")
        source = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), helper], type_ignores=[])
        ast.fix_missing_locations(source)
        namespace = {"torch": torch}
        exec(compile(source, str(path), "exec"), namespace)
        inputs = projection_fixture()
        function = Mock(wraps=namespace["triton_addmm_bwd"])
        with patch.object(replay, "replay_once", side_effect=AssertionError("parent replay forbidden")):
            actual = stages.execute_stage("input_projection", inputs, SimpleNamespace(triton_addmm_bwd=function))
        for left, right in zip(actual, projection_autograd(inputs)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(function.call_count, 1)
        for name, value in function.call_args.kwargs.items():
            self.assertIs(value, inputs[name])

    def test_direct_norm_passes_exact_captured_tensors_and_settings(self):
        inputs = norm_fixture()
        expected = norm_autograd(inputs)
        helper = Mock(return_value=expected)
        with patch.object(replay, "replay_once", side_effect=AssertionError("parent replay forbidden")):
            actual = stages.execute_stage("input_norm", inputs, SimpleNamespace(triton_weighted_layer_norm_bwd=helper))
        self.assertIs(actual, expected)
        for name, value in helper.call_args.kwargs.items():
            self.assertIs(value, inputs[name])


class CapturedLaunchTest(unittest.TestCase):
    def fixture(self):
        descriptor = {"launch_metadata": {"format_version": 1,
            "dx": {"launch_config": {"BLOCK_N": 1, "BLOCK_D": 8, "num_warps": 1, "num_stages": 1}},
            "reduction": {"launch_config": {"BLOCK_N": 128, "BLOCK_D": 4, "num_warps": 8, "num_stages": 1}},
        }}
        args = SimpleNamespace(block_n=None, num_warps=None, num_stages=None, max_vgpr=None)
        return descriptor, args

    def test_defaults_use_captured_config_and_overrides_only_change_dx(self):
        descriptor, args = self.fixture()
        configs = stages.captured_norm_configs(descriptor, norm_fixture(), args)
        self.assertEqual(configs["dx"]["BLOCK_N"], 1)
        args.block_n, args.num_warps, args.num_stages, args.max_vgpr = 8, 2, 2, 256
        changed = stages.captured_norm_configs(descriptor, norm_fixture(), args)
        self.assertEqual(changed["dx"], {"BLOCK_N": 8, "BLOCK_D": 8, "num_warps": 2,
                                        "num_stages": 2, "llvm_fn_attrs": "amdgpu-num-vgpr=256"})
        self.assertEqual(changed["reduction"], configs["reduction"])
        self.assertEqual(descriptor["launch_metadata"]["dx"]["launch_config"]["BLOCK_N"], 1)

    def test_missing_original_config_rejected_instead_of_retuning(self):
        _, args = self.fixture()
        with self.assertRaisesRegex(ValueError, "original launch_metadata"):
            stages.captured_norm_configs({}, norm_fixture(), args)

    def test_captured_dx_width_must_match_original_helper_settings(self):
        descriptor, args = self.fixture()
        descriptor["launch_metadata"]["dx"]["launch_config"]["BLOCK_D"] = 16
        with self.assertRaisesRegex(ValueError, "BLOCK_D disagrees"):
            stages.captured_norm_configs(descriptor, norm_fixture(), args)

    def test_pin_bypasses_tuner_cache_records_actual_launch_and_restores_after_error(self):
        descriptor, args = self.fixture()
        configs = stages.captured_norm_configs(descriptor, norm_fixture(), args)

        class Config:
            def __init__(self, kwargs, num_warps=4, num_stages=2):
                self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages

        compiled = SimpleNamespace(hash="test_hash", name="actual_dx", metadata="metadata", asm={"amdgcn": "kernel"})
        original = Mock(return_value=compiled)
        dx = SimpleNamespace(configs=["old"], cache={"winner": "stale"}, fn=SimpleNamespace(run=original))
        reduction_run = Mock(return_value=compiled)
        reduction = SimpleNamespace(configs=["old_reduction"], cache={}, fn=SimpleNamespace(run=reduction_run))
        module = SimpleNamespace(_weighted_layer_norm_bwd_dx=dx, _layer_norm_bwd_dwdb=reduction)
        report = {}
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with stages.norm_launches(SimpleNamespace(Config=Config), module, norm_fixture(), configs, report):
                self.assertEqual(dx.cache, {})
                self.assertEqual(dx.configs[0].kwargs, {"BLOCK_N": 1})
                dx.fn.run(BLOCK_N=1, BLOCK_D=16, num_warps=1, num_stages=1, warmup=False)
                self.assertEqual(original.call_args.kwargs["BLOCK_D"], 8)
                reduction.fn.run(BLOCK_N=128, BLOCK_D=32, num_warps=8, num_stages=1, warmup=False)
                self.assertEqual(reduction_run.call_args.kwargs["BLOCK_D"], 4)
                self.assertEqual(report["actual_norm_launches"][0]["compiled_hash"], "test_hash")
                self.assertEqual(report["actual_norm_launches"][1]["launch_config"]["BLOCK_D"], 4)
                self.assertIn("amdgcn", report["actual_norm_launches"][0]["artifact_sha256"])
                raise RuntimeError("test failure")
        self.assertEqual(dx.configs, ["old"])
        self.assertEqual(dx.cache, {"winner": "stale"})
        self.assertIs(dx.fn.run, original)
        self.assertEqual(reduction.configs, ["old_reduction"])


class GuardTest(unittest.TestCase):
    def test_norm_overrides_rejected_for_projection(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            stages.arguments(["capture.pt", "--stage", "input_projection", "--block-n", "8", "--report", "out.json"])

    def test_report_cannot_overwrite_capture_through_hardlink(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.pt"
            capture.write_bytes(b"immutable")
            alias = Path(directory) / "alias.pt"
            alias.hardlink_to(capture)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                stages.arguments([str(capture), "--stage", "input_norm", "--report", str(alias)])
            self.assertEqual(capture.read_bytes(), b"immutable")

    def test_byte_guard_detects_untracked_write_through_decoded_alias(self):
        inputs = projection_fixture(False)
        capture = encode_capture(inputs, projection_autograd(inputs), "input_projection")
        fixed, _ = stages.original_stage_inputs(capture, "input_projection", "cpu")
        tensor = fixed["dz"]
        snapshot, version = tensor.clone(), tensor._version
        alias = torch.empty(0, dtype=tensor.dtype).set_(tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride())
        alias[-1, -1] = 0
        self.assertEqual(tensor._version, version)
        self.assertFalse(stages.all_equal(torch, tensor, snapshot, chunk_size=3, bytewise=True))

    def test_no_gpu_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
