"""CPU integration of optional fused flags with tripwire capture and replay."""

import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


from generative_recommenders.dlrm_v4.train import fused_value_checker as helper
from generative_recommenders.dlrm_v4.train import nan_tripwire as module
from generative_recommenders.dlrm_v4.train.tests.fused_value_checker_test import FakeKernel
from scripts.replay_nan_tripwire import decode_payload, replay_once


class _SavedViewsBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value, value[:, 1:])
        return value * 2

    @staticmethod
    def backward(ctx, grad_output):
        value, _ = ctx.saved_tensors
        with torch.inference_mode():
            return grad_output * 2 + value * 0


class IntegrationTest(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_default_off_preserves_torch_flags_without_loading_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            tripwire = module.NaNTripwire(directory, max_abs=1e20, finite_chunk_elements=2)
            helper_name = "generative_recommenders.dlrm_v4.train.fused_value_checker"
            with patch.dict(sys.modules, {helper_name: None}):
                finite, bound, metadata = tripwire._observe({
                    "large": torch.tensor([1e30]),
                    "bad": torch.tensor([float("nan")]),
                    "empty": torch.empty(0),
                    "integer": torch.ones(2, dtype=torch.int64),
                })
            self.assertEqual([bool(flag) for flag in finite], [True, False])
            self.assertEqual([bool(flag) for flag in bound], [False, False])
            self.assertEqual([info.get("flag_index") for info in metadata], [0, 1, None, None])
            self.assertIsNone(tripwire._fused_checker)

    def test_enabled_uses_separate_flags_without_original_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            tripwire = module.NaNTripwire(directory, max_abs=1e20, fused_value_check=True)
            finite, bound = torch.tensor(True), torch.tensor(False)
            tripwire._fused_checker = SimpleNamespace(try_check=lambda tensor, maximum: (finite, bound))
            with patch.object(module, "_all_finite", side_effect=AssertionError("unexpected Torch finite")), \
                    patch.object(module, "_all_within_bound", side_effect=AssertionError("unexpected Torch bound")):
                actual_finite, actual_bound, _ = tripwire._observe(torch.ones(5, 3))
            self.assertIs(actual_finite[0], finite)
            self.assertIs(actual_bound[0], bound)

    def test_none_result_keeps_torch_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            tripwire = module.NaNTripwire(directory, max_abs=1e20, fused_value_check=True)
            tripwire._fused_checker = SimpleNamespace(try_check=lambda tensor, maximum: None)
            finite, bound, _ = tripwire._observe(torch.tensor([1e30]))
            self.assertTrue(bool(finite[0]))
            self.assertFalse(bool(bound[0]))

    def test_enabled_cpu_and_complex_fallback_preserves_tensor_flag_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            tripwire = module.NaNTripwire(directory, max_abs=10, fused_value_check=True,
                                         finite_chunk_elements=2)
            with patch.object(helper.FusedValueChecker, "_load_kernel",
                              side_effect=AssertionError("unsupported input launched kernel")):
                finite, bound, metadata = tripwire._observe({
                    "cpu": torch.tensor([11.0]),
                    "complex": torch.tensor([complex(float("inf"), 1)]),
                    "empty": torch.empty(0),
                    "integer": torch.ones(2, dtype=torch.int64),
                })
            self.assertEqual([bool(flag) for flag in finite], [True, False])
            self.assertEqual([bool(flag) for flag in bound], [False, False])
            self.assertEqual([info.get("flag_index") for info in metadata], [0, 1, None, None])
            checker_metadata = tripwire._fused_checker.metadata()
            self.assertEqual(checker_metadata["fallbacks"], {"non_gpu": 1, "unsupported_dtype": 1})
            self.assertEqual(checker_metadata["calls"], 0)
            self.assertIsNone(checker_metadata["kernel_jit_source_sha256"])
            self.assertEqual(checker_metadata["compiled"], [])

    def test_fused_magnitude_report_keeps_nonfinite_flags_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            tripwire = module.NaNTripwire(directory, max_abs=1e20, fused_value_check=True,
                                         retain_payload=False)
            tripwire._fused_checker = helper.FusedValueChecker(enabled=True)
            tripwire._fused_checker._kernel = FakeKernel(2)
            tripwire.begin(3)
            with patch.object(helper, "unsupported_reason", return_value=None), \
                    patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")), \
                    patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")), \
                    patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronization")):
                tripwire.watch("large_output", torch.tensor([1e30]))
            with self.assertRaises(module.MagnitudeError):
                tripwire.check("backward")
            report = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(report["reason"], "magnitude")
            self.assertTrue(report["all_outputs_finite"])
            self.assertFalse(report["all_outputs_within_bound"])
            self.assertTrue(report["fused_value_check"])
            self.assertEqual(report["fused_value_checker"]["last_launch"]["compiled_hash"],
                             "exact-launch-hash")
            self.assertEqual(list(Path(directory).glob("*.pt")), [])

    def test_requested_backward_capture_preserves_aliases_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            tripwire = module.NaNTripwire(directory, fused_value_check=True, capture_step=4,
                                         capture_direction="backward", max_abs=1e20)
            checker = helper.FusedValueChecker(enabled=True)
            checker._kernel = FakeKernel(0)
            tripwire._fused_checker = checker
            tripwire.patch_function(_SavedViewsBackward)
            try:
                tripwire.begin(4)
                base = torch.arange(4 * 1024, dtype=torch.float32).reshape(4, 1024)
                value = base[:, 512:].requires_grad_()
                with patch.object(helper, "unsupported_reason", return_value=None):
                    output = _SavedViewsBackward.apply(value)
                    tripwire.check("forward")
                    output.backward(torch.full_like(output, 3))
                with self.assertRaises(module.CaptureComplete) as captured:
                    tripwire.check("backward")
            finally:
                tripwire.uninstall()
            report = json.loads(captured.exception.report_path.read_text())
            self.assertEqual(report["reason"], "requested_capture")
            self.assertEqual(report["events"][0]["direction"], "backward")
            self.assertIsNone(report["events"][0]["outputs"][0]["version"])
            self.assertEqual(report["fused_value_checker"]["compiled"][0]["hash"],
                             "exact-launch-hash")
            self.assertEqual(report["fused_value_checker"]["calls"], 6)
            capture = torch.load(captured.exception.capture_path, weights_only=False)
            self.assertEqual(capture["format_version"], 1)
            self.assertEqual(capture["report"]["fused_value_checker"], report["fused_value_checker"])
            payload = decode_payload(capture, "cpu")
            first, second = payload["replay"]["saved_tensors"]
            self.assertEqual(first.stride(), (1024, 1))
            self.assertEqual(first.storage_offset(), 512)
            self.assertEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
            self.assertTrue(torch.equal(first, value))
            result, _ = replay_once(capture, "cpu")
            self.assertTrue(result.is_inference())
            self.assertTrue(torch.equal(result, payload["outputs"]))
            self.assertTrue(torch.equal(result, torch.full_like(value, 6)))

    def test_no_bound_keeps_bound_flag_list_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            tripwire = module.NaNTripwire(directory, fused_value_check=True)
            checker = helper.FusedValueChecker(enabled=True)
            checker._kernel = FakeKernel(1)
            tripwire._fused_checker = checker
            with patch.object(helper, "unsupported_reason", return_value=None):
                finite, bound, _ = tripwire._observe(torch.tensor(float("nan")))
            self.assertEqual([bool(flag) for flag in finite], [False])
            self.assertEqual(bound, [])

    def test_env_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            for value, enabled in ((None, False), ("0", False), ("true", False), ("1", True)):
                environment = {"NAN_TRIPWIRE_DIR": directory}
                if value is not None:
                    environment["NAN_TRIPWIRE_FUSED_VALUE_CHECK"] = value
                with patch.dict(os.environ, environment, clear=True), \
                        patch.object(module.NaNTripwire, "install"):
                    tripwire = module.install_from_env()
                self.assertIs(tripwire.fused_value_check, enabled)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
