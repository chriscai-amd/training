"""CPU tests for diagnosing a bad backward without the model or a GPU."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from generative_recommenders.dlrm_v4.train.nan_tripwire import (
    CaptureComplete,
    MagnitudeError,
    NaNTripwire,
    NonFiniteError,
    _all_finite,
    _all_within_bound,
    _finite_chunks,
    install_from_env,
)
from scripts.replay_nan_tripwire import decode_payload, replay_once


class _BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, poison):
        ctx.save_for_backward(x, x[:, 1:])
        ctx.poison = poison
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors
        return grad_output * (float("nan") if ctx.poison else 2) + x * 0, None


class _BadForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * float("nan")

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _InferenceBadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        # Matches the allocation context used by _AttentionFunction.backward.
        with torch.inference_mode():
            return grad_output * float("nan")


class _SqrtForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sqrt()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _CommonStateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        from generative_recommenders.common import autotune_max_seq_len

        return grad_output * autotune_max_seq_len(2768)


def _fake_common():
    module = ModuleType("generative_recommenders.common")
    module.STATIC_MAX_SEQ_LENS = [4096]
    module.USE_RUNTIME_MAX_SEQ_LEN = False
    module.BACKEND_ALLOW_TF32 = True
    module.set_static_max_seq_lens = lambda values: setattr(module, "STATIC_MAX_SEQ_LENS", list(values))
    module.set_use_runtime_max_seq_len = lambda value: setattr(module, "USE_RUNTIME_MAX_SEQ_LEN", value)
    module.autotune_max_seq_len = lambda value: (
        2048 if module.USE_RUNTIME_MAX_SEQ_LEN else
        module.STATIC_MAX_SEQ_LENS[0] if module.STATIC_MAX_SEQ_LENS else 1
    )
    return module


class NaNTripwireTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tripwire = NaNTripwire(self.temp.name)

    def tearDown(self):
        self.tripwire.uninstall()
        self.temp.cleanup()

    def test_finite_chunk_bound_preserves_arbitrary_strided_views(self):
        base = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7)
        views = (base, base.permute(2, 0, 1), base[:, 1::2, ::2],
                 torch.ones(1, 3, 1).expand(5, 3, 7))
        for value in views:
            for bound in (1, 4, 8, 17):
                with self.subTest(shape=value.shape, stride=value.stride(), bound=bound):
                    chunks = list(_finite_chunks(value, bound))
                    self.assertEqual(sum(chunk.numel() for chunk in chunks), value.numel())
                    self.assertTrue(all(0 < chunk.numel() <= bound for chunk in chunks))
                    self.assertTrue(all(chunk.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
                                        for chunk in chunks))
                    original_isfinite = torch.isfinite
                    observed_sizes = []

                    def checked_isfinite(chunk):
                        observed_sizes.append(chunk.numel())
                        return original_isfinite(chunk)

                    with patch.object(torch, "isfinite", side_effect=checked_isfinite):
                        flag = _all_finite(value, bound)
                    self.assertTrue(flag.item())
                    self.assertEqual(flag.shape, torch.Size([]))
                    self.assertEqual(flag.dtype, torch.bool)
                    self.assertEqual(flag.device, value.device)
                    self.assertEqual(sum(observed_sizes), value.numel())
                    self.assertLessEqual(max(observed_sizes), bound)

    def test_finite_chunks_find_nan_and_infinities_across_boundaries(self):
        for poison in (float("nan"), float("inf"), -float("inf")):
            for index in (0, 6, 7, 8, 34):
                with self.subTest(poison=poison, index=index):
                    base = torch.ones(5, 14)
                    value = base[:, 1::2].t()  # Offset, row gaps, and transpose.
                    value[index // 5, index % 5] = poison
                    flag = _all_finite(value, 7)
                    self.assertFalse(flag.item())
                    self.assertEqual(flag.item(), torch.isfinite(value).all().item())
        complex_value = torch.ones(3, 7, dtype=torch.complex64).t()
        complex_value[-1, -1] = complex(1, float("inf"))
        self.assertFalse(_all_finite(complex_value, 4).item())

    def test_finite_chunk_scalars_empty_and_default_path(self):
        for value in (torch.tensor(1.0), torch.tensor(float("nan")), torch.empty(0), torch.empty(2, 0, 3)):
            for bound in (0, 1, 7):
                with self.subTest(shape=value.shape, bound=bound):
                    self.assertEqual(_all_finite(value, bound).item(), torch.isfinite(value).all().item())
        value = torch.ones(3, 11)
        original_isfinite = torch.isfinite
        with patch.object(torch, "isfinite", wraps=original_isfinite) as checked:
            self.assertTrue(_all_finite(value, 0).item())
        self.assertEqual(checked.call_count, 1)
        self.assertIs(checked.call_args.args[0], value)
        self.tripwire.finite_chunk_elements = 2
        self.tripwire.begin(1)
        self.tripwire.watch("mixed", {"empty": torch.empty(0), "scalar": torch.tensor(1.0),
                                      "integer": torch.ones(9, dtype=torch.int64)})
        self.assertEqual(len(self.tripwire.events[0]["output_flags"]), 1)
        self.tripwire.check("forward")
        self.assertEqual(self.tripwire.events, [])

    def test_finite_chunk_configuration_and_report(self):
        for invalid in (-1, 1.5, "7"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                NaNTripwire(self.temp.name, finite_chunk_elements=invalid)
        for value in (0, 7):
            with patch.dict(os.environ, {"NAN_TRIPWIRE_DIR": self.temp.name,
                                        "NAN_TRIPWIRE_FINITE_CHUNK_ELEMENTS": str(value)}), \
                 patch.object(NaNTripwire, "install"):
                tripwire = install_from_env()
            self.assertEqual(tripwire.finite_chunk_elements, value)
        self.tripwire.finite_chunk_elements = 3
        self.tripwire.retain_payload = False
        self.tripwire.begin(17)
        value = torch.ones(4, 5)
        value[-1, -1] = float("nan")
        self.tripwire.watch("late_nan", value)
        with self.assertRaises(NonFiniteError):
            self.tripwire.check("forward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertEqual(report["finite_chunk_elements"], 3)
        self.assertFalse(report["all_outputs_finite"])

    def test_magnitude_output_precedes_later_nonfinite_and_preserves_capture(self):
        self.tripwire.max_abs = 1e20
        self.tripwire.finite_chunk_elements = 3
        self.tripwire.patch_function(_BadBackward)
        self.tripwire.begin(20)
        x = torch.full((2, 3), 1e20, dtype=torch.float64)
        output = _BadBackward.apply(x, False)
        self.tripwire.watch("later_nan", torch.tensor(float("nan")))
        with self.assertRaisesRegex(MagnitudeError, "NAN_TRIPWIRE_MAX_ABS"):
            self.tripwire.check("forward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertEqual(report["reason"], "magnitude")
        self.assertEqual(report["failure_side"], "output")
        self.assertEqual(report["selected_sequence"], 0)
        self.assertEqual(report["first_bad_sequence"], 0)
        self.assertEqual(report["max_abs"], 1e20)
        self.assertTrue(report["all_inputs_finite"])
        self.assertTrue(report["all_outputs_finite"])
        self.assertTrue(report["all_inputs_within_bound"])
        self.assertFalse(report["all_outputs_within_bound"])
        event = report["events"][0]
        self.assertEqual(event["input_finite"], [True])
        self.assertEqual(event["output_finite"], [True])
        self.assertEqual(event["output_within_bound"], [False])
        self.assertNotIn("output_bound_flags", event)
        capture = torch.load(report["capture_path"], weights_only=False)
        actual, payload = replay_once(capture, "cpu")
        self.assertTrue(torch.equal(actual, output))
        self.assertTrue(torch.equal(payload["outputs"], output))

    def test_magnitude_input_only_is_reported_without_changing_finite_flags(self):
        self.tripwire.max_abs = 1e20
        self.tripwire.patch_function(_SqrtForward)
        self.tripwire.begin(21)
        _SqrtForward.apply(torch.tensor([1e30], dtype=torch.float64))
        with self.assertRaises(MagnitudeError):
            self.tripwire.check("forward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertEqual(report["reason"], "magnitude")
        self.assertEqual(report["failure_side"], "input")
        self.assertTrue(report["all_inputs_finite"])
        self.assertTrue(report["all_outputs_finite"])
        self.assertFalse(report["all_inputs_within_bound"])
        self.assertTrue(report["all_outputs_within_bound"])

    def test_magnitude_output_takes_precedence_over_earlier_input_only_failure(self):
        self.tripwire.max_abs = 1e20
        self.tripwire.patch_function(_SqrtForward)
        self.tripwire.begin(22)
        _SqrtForward.apply(torch.tensor([1e30], dtype=torch.float64))
        self.tripwire.watch("first_bad_output", torch.tensor(-1e30))
        self.tripwire.watch("second_bad_output", torch.tensor(float("inf")))
        with self.assertRaises(MagnitudeError):
            self.tripwire.check("forward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertEqual(report["selected_sequence"], 1)
        self.assertEqual(report["failure_side"], "output")
        self.assertEqual(report["events"][1]["name"], "first_bad_output")

    def test_nonfinite_output_keeps_its_reason_with_magnitude_enabled(self):
        for step, poison in enumerate((float("nan"), float("inf"), -float("inf")), 23):
            with self.subTest(poison=poison):
                self.tripwire.max_abs = 1e20
                self.tripwire.begin(step)
                self.tripwire.watch("first_nonfinite", torch.tensor(poison))
                self.tripwire.watch("later_large", torch.tensor(1e30))
                with self.assertRaises(NonFiniteError):
                    self.tripwire.check("forward")
                report_path = next(Path(self.temp.name).glob(f"step{step:06d}_*.json"))
                report = json.loads(report_path.read_text())
                self.assertEqual(report["reason"], "nonfinite")
                self.assertEqual(report["selected_sequence"], 0)
                self.assertFalse(report["all_outputs_finite"])
                self.assertFalse(report["all_outputs_within_bound"])

    def test_magnitude_chunks_cover_strides_boundaries_and_low_precision(self):
        base = torch.full((5, 14), 10.0)
        for tensor in (base[:, 1::2].t(), torch.tensor([10.0]).expand(3, 11),
                       torch.tensor(10.0), torch.empty(0), torch.tensor([6 + 8j])):
            with self.subTest(shape=tensor.shape, dtype=tensor.dtype):
                self.assertTrue(_all_within_bound(tensor, 10.0, 3).item())
        for index in (0, 2, 3, 34):
            value = base.clone()[:, 1::2].t()
            value[index // 5, index % 5] = -10.5
            self.assertFalse(_all_within_bound(value, 10.0, 3).item())
        # 1.001 rounds to 1.0 in BF16. A bound below 1.0 must still reject 1.0.
        self.assertFalse(_all_within_bound(torch.ones(3, dtype=torch.bfloat16), 0.999, 2).item())
        self.assertFalse(_all_within_bound(torch.tensor([float("nan")]), 10, 2).item())
        self.assertFalse(_all_within_bound(torch.tensor([float("inf")]), 10, 2).item())
        self.tripwire.max_abs = 10.0
        self.tripwire.finite_chunk_elements = 3
        observed_sizes = []
        original_abs = torch.Tensor.abs

        def checked_abs(value):
            observed_sizes.append(value.numel())
            return original_abs(value)

        self.tripwire.begin(26)
        with patch.object(torch.Tensor, "abs", checked_abs), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("per-op host read")), \
             patch.object(torch.Tensor, "tolist", side_effect=AssertionError("per-op host read")):
            self.tripwire.watch("bounded", base[:, 1::2].t())
        self.assertLessEqual(max(observed_sizes), 3)
        self.assertEqual(sum(observed_sizes), 35)
        self.tripwire.check("forward")

    def test_magnitude_threshold_validation_and_disabled_default(self):
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), True, "1e20"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive finite"):
                NaNTripwire(self.temp.name, max_abs=value)
        for value in ("0", "-1", "nan", "inf", "-inf", "invalid"):
            with self.subTest(value=value), \
                 patch.dict(os.environ, {"NAN_TRIPWIRE_DIR": self.temp.name, "NAN_TRIPWIRE_MAX_ABS": value}), \
                 patch.object(NaNTripwire, "install"), self.assertRaises(ValueError):
                install_from_env()
        for value, expected in (("", None), ("1e20", 1e20)):
            with patch.dict(os.environ, {"NAN_TRIPWIRE_DIR": self.temp.name, "NAN_TRIPWIRE_MAX_ABS": value}), \
                 patch.object(NaNTripwire, "install"):
                tripwire = install_from_env()
            self.assertEqual(tripwire.max_abs, expected)
        self.assertIsNone(self.tripwire.max_abs)
        self.tripwire.begin(27)
        with patch("generative_recommenders.dlrm_v4.train.nan_tripwire._all_within_bound",
                   side_effect=AssertionError("disabled magnitude check ran")):
            self.tripwire.watch("large_but_finite", torch.tensor(1e30))
        self.tripwire.check("forward")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_clean_step_releases_references_and_does_not_write(self):
        self.tripwire.patch_function(_BadBackward)
        self.tripwire.begin(4)
        x = torch.ones(2, 3, requires_grad=True)
        y = _BadBackward.apply(x, False)
        self.tripwire.check("forward")
        self.assertEqual(self.tripwire.events, [])
        y.sum().backward()
        self.tripwire.watch("dense_grad", x.grad)
        self.tripwire.check("backward")
        self.assertEqual(self.tripwire.events, [])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 2)))

    def test_bad_backward_captures_inputs_and_replays_with_aliases(self):
        self.tripwire.patch_function(_BadBackward)
        self.tripwire.begin(7, train_ts=13)
        # This has a nonzero storage offset and a noncontiguous row stride.
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        x = base[1:3, 1:4].requires_grad_()
        y = _BadBackward.apply(x, True)
        self.tripwire.check("forward")
        y.backward(torch.full_like(y, 3))
        with self.assertRaises(NonFiniteError):
            self.tripwire.check("backward")
        reports = list(Path(self.temp.name).glob("*.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertTrue(report["all_inputs_finite"])
        self.assertEqual(report["metadata"], {"train_ts": 13})
        capture = torch.load(report["capture_path"], weights_only=False)
        payload = decode_payload(capture, "cpu")
        first, second = payload["replay"]["saved_tensors"]
        self.assertEqual(first.stride(), x.stride())
        self.assertEqual(first.storage_offset(), x.storage_offset())
        self.assertEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(first, x))
        self.assertTrue(torch.equal(payload["replay"]["args"][0], torch.full_like(y, 3)))
        result, _ = replay_once(capture, "cpu")
        self.assertTrue(torch.isnan(result[0]).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_bad_forward_and_capture_budget(self):
        self.tripwire.capture_limit_bytes = 0
        self.tripwire.patch_function(_BadForward)
        self.tripwire.begin(0)
        _BadForward.apply(torch.ones(3))
        with self.assertRaises(NonFiniteError):
            self.tripwire.check("forward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertTrue(report["all_inputs_finite"])
        self.assertIn("exceeds", report["capture_error"])
        self.assertEqual(list(Path(self.temp.name).glob("*.pt")), [])

    def test_inference_backward_captures_and_replays_without_version_counter(self):
        self.tripwire.patch_function(_InferenceBadBackward)
        self.tripwire.begin(8)
        x = torch.ones(2, 3, requires_grad=True)
        y = _InferenceBadBackward.apply(x)
        self.tripwire.check("forward")
        y.sum().backward()
        with self.assertRaises(NonFiniteError):
            self.tripwire.check("backward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertTrue(report["all_inputs_finite"])
        output_info = report["events"][0]["outputs"][0]
        self.assertIsNone(output_info["version"])
        self.assertIsNone(output_info["version_at_capture"])
        self.assertIsNone(output_info["version_changed"])
        capture = torch.load(report["capture_path"], weights_only=False)
        self.assertIsNone(capture["payload"]["outputs"]["version"])
        result, _ = replay_once(capture, "cpu")
        self.assertTrue(result.is_inference())
        self.assertTrue(torch.isnan(result).all())

    def test_preexisting_bad_input_is_not_attributed_as_finite(self):
        self.tripwire.patch_function(_BadForward)
        self.tripwire.begin(0)
        _BadForward.apply(torch.tensor([float("nan")]))
        with self.assertRaises(NonFiniteError):
            self.tripwire.check("forward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertFalse(report["all_inputs_finite"])

    def test_changed_input_cannot_turn_bad_observation_into_valid_finite_replay(self):
        self.tripwire.patch_function(_SqrtForward)
        self.tripwire.begin(9)
        x = torch.tensor([-1.0])
        _SqrtForward.apply(x)
        # Deferred capture would now replay sqrt(4) and falsely appear clean.
        x.fill_(4)
        with self.assertRaises(NonFiniteError):
            self.tripwire.check("forward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertTrue(report["all_inputs_finite"])
        self.assertTrue(report["events"][0]["inputs"][0]["version_changed"])
        capture = torch.load(report["capture_path"], weights_only=False)
        with self.assertRaisesRegex(ValueError, "Inputs changed between observation and capture"):
            replay_once(capture, "cpu")
        with self.assertWarnsRegex(RuntimeWarning, "cannot validate"):
            result, _ = replay_once(capture, "cpu", allow_mutated_inputs=True)
        self.assertTrue(torch.equal(result, torch.tensor([2.0])))

    def test_active_jagged_tensor_module_is_instrumented_by_default(self):
        module_name = "generative_recommenders.ops.triton.triton_jagged_tensors"
        module = ModuleType(module_name)
        cls = type("JaggedProbe", (_BadForward,), {"__module__": module_name})
        module.JaggedProbe = cls
        with patch.dict(sys.modules, {module_name: module}):
            self.tripwire.install()
            self.tripwire.begin(10)
            cls.apply(torch.ones(3))
            with self.assertRaisesRegex(NonFiniteError, "triton_jagged_tensors.JaggedProbe"):
                self.tripwire.check("forward")

    def test_metadata_only_releases_storage_before_check_and_writes_no_payload(self):
        self.tripwire.retain_payload = False
        self.tripwire.patch_function(_BadForward)
        self.tripwire.begin(11)
        x = torch.ones(1024)
        storage_weak_ref = x.untyped_storage()._weak_ref()
        try:
            _BadForward.apply(x)
            del x
            self.assertTrue(torch.UntypedStorage._expired(storage_weak_ref))
        finally:
            torch.UntypedStorage._free_weak_ref(storage_weak_ref)
        self.assertIsNone(self.tripwire.events[0]["payload"])
        self.assertEqual(
            self.tripwire.events[0]["replay_metadata"]["args"][0]["shape"], [1024]
        )
        with self.assertRaises(NonFiniteError):
            self.tripwire.check("backward")
        report = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
        self.assertFalse(report["retain_payload"])
        self.assertEqual(list(Path(self.temp.name).glob("*.pt")), [])

    def test_deferred_check_preserves_first_bad_step_and_metadata(self):
        self.tripwire.retain_payload = False
        self.tripwire.defer_steps = 3
        self.tripwire.end_step = 49
        self.tripwire.begin(48, train_ts=12, batch_idx=3)
        self.tripwire.watch("loss", torch.tensor(1.0))
        self.tripwire.begin(49, train_ts=13, batch_idx=0)
        self.tripwire.watch("loss", torch.tensor(float("nan")))
        self.tripwire.begin(50, train_ts=13, batch_idx=1)
        # The pending interval must still be checked after leaving the active
        # step range; new observations at step 50 should not be included.
        self.tripwire.watch("ignored", torch.tensor(0.0))
        self.assertEqual([event["step"] for event in self.tripwire.events], [48, 49])
        with self.assertRaisesRegex(NonFiniteError, "step=49 check_step=50"):
            self.tripwire.check("metrics")
        report_path = next(Path(self.temp.name).glob("*.json"))
        self.assertTrue(report_path.name.startswith("step000049_"))
        report = json.loads(report_path.read_text())
        self.assertEqual(report["step"], 49)
        self.assertEqual(report["check_step"], 50)
        self.assertEqual(report["metadata"], {"train_ts": 13, "batch_idx": 0})
        self.assertEqual(report["check_metadata"], {"train_ts": 13, "batch_idx": 1})
        self.assertEqual(report["first_bad_sequence"], 1)

    def test_observation_outside_step_does_nothing(self):
        self.tripwire.watch("ignored", torch.tensor(float("nan")))
        self.tripwire.check("forward")
        self.assertEqual(self.tripwire.events, [])

    def test_requested_finite_backward_capture_replays_without_consuming_rng(self):
        self.tripwire = NaNTripwire(
            self.temp.name,
            capture_step=13,
            capture_function_pattern=r"\._BadBackward$",
        )
        self.tripwire.patch_function(_BadBackward)
        rng_before = torch.get_rng_state()
        self.tripwire.begin(12)
        x = torch.ones(2, 3, requires_grad=True)
        _BadBackward.apply(x, False).sum().backward()
        self.tripwire.check("backward")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.tripwire.begin(13, train_ts=13, batch_idx=2)
        y = _BadBackward.apply(x, False)
        self.tripwire.check("forward")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        y.backward(torch.full_like(y, 3))
        with self.assertRaises(CaptureComplete) as caught:
            self.tripwire.check("backward")
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        report = json.loads(caught.exception.report_path.read_text())
        self.assertEqual(report["reason"], "requested_capture")
        self.assertIsNone(report["first_bad_sequence"])
        self.assertTrue(report["all_inputs_finite"])
        self.assertTrue(report["all_outputs_finite"])
        self.assertEqual(len(list(Path(self.temp.name).glob("*.pt"))), 1)
        capture = torch.load(caught.exception.capture_path, weights_only=False)
        output, payload = replay_once(capture, "cpu")
        self.assertEqual(payload["replay"]["direction"], "backward")
        self.assertTrue(torch.equal(output[0], torch.full_like(x, 6)))

    def test_requested_capture_failure_does_not_signal_completion(self):
        self.tripwire = NaNTripwire(
            self.temp.name,
            capture_step=14,
            capture_direction="forward",
            capture_limit_bytes=0,
        )
        self.tripwire.patch_function(_BadBackward)
        self.tripwire.begin(14)
        _BadBackward.apply(torch.ones(2, 3), False)
        with self.assertRaisesRegex(RuntimeError, "Requested diagnostic capture failed") as caught:
            self.tripwire.check("forward")
        self.assertNotIsInstance(caught.exception, CaptureComplete)

    def test_replay_restores_common_tuning_state(self):
        common = _fake_common()
        with patch.dict(sys.modules, {common.__name__: common}):
            self.tripwire = NaNTripwire(self.temp.name, capture_step=15)
            self.tripwire.patch_function(_CommonStateBackward)
            self.tripwire.begin(15)
            _CommonStateBackward.apply(torch.ones(3, requires_grad=True)).sum().backward()
            with self.assertRaises(CaptureComplete) as caught:
                self.tripwire.check("backward")
            capture = torch.load(caught.exception.capture_path, weights_only=False)
            common.STATIC_MAX_SEQ_LENS = []
            common.USE_RUNTIME_MAX_SEQ_LEN = True
            common.BACKEND_ALLOW_TF32 = False
            result, _ = replay_once(capture, "cpu")
            self.assertTrue(torch.equal(result, torch.full((3,), 4096.0)))
            self.assertEqual(common.STATIC_MAX_SEQ_LENS, [4096])
            self.assertFalse(common.USE_RUNTIME_MAX_SEQ_LEN)
            self.assertTrue(common.BACKEND_ALLOW_TF32)
            self.assertFalse(torch.cuda.is_initialized())

    def test_old_capture_warns_and_preserves_explicit_current_tuning_state(self):
        common = _fake_common()
        with patch.dict(sys.modules, {common.__name__: common}):
            self.tripwire = NaNTripwire(self.temp.name, capture_step=16)
            self.tripwire.patch_function(_CommonStateBackward)
            self.tripwire.begin(16)
            _CommonStateBackward.apply(torch.ones(3, requires_grad=True)).sum().backward()
            with self.assertRaises(CaptureComplete) as caught:
                self.tripwire.check("backward")
            capture = torch.load(caught.exception.capture_path, weights_only=False)
            del capture["payload"]["replay"]["runtime_state"]
            common.STATIC_MAX_SEQ_LENS = [1024]
            with self.assertWarnsRegex(RuntimeWarning, "missing runtime tuning state"):
                result, _ = replay_once(capture, "cpu")
            self.assertTrue(torch.equal(result, torch.full((3,), 1024.0)))
            self.assertEqual(common.STATIC_MAX_SEQ_LENS, [1024])


if __name__ == "__main__":
    unittest.main()
