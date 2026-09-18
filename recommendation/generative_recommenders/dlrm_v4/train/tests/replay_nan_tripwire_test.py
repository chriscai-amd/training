"""CPU coverage for persistent-input replay and captured-output comparison."""

import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from generative_recommenders.dlrm_v4.train.nan_tripwire import (
    CaptureComplete,
    NaNTripwire,
)
from scripts import replay_nan_tripwire as replay
from scripts.replay_nan_tripwire import decode_payload, stress_replay


class _FailsAfterThree(torch.autograd.Function):
    calls = 0

    @staticmethod
    def forward(ctx, x):
        _FailsAfterThree.calls += 1
        scale = float("nan") if _FailsAfterThree.calls > 3 else 2
        return x * scale

    @staticmethod
    def backward(ctx, dy):
        return dy


class _RandomForward(torch.autograd.Function):
    draws = []
    input_ptrs = []

    @staticmethod
    def forward(ctx, x):
        if getattr(ctx, "used", False):
            raise RuntimeError("context reused")
        ctx.used = True
        draw = torch.rand_like(x) + random.random()
        _RandomForward.draws.append(draw.clone())
        _RandomForward.input_ptrs.append(x.data_ptr())
        return x + draw

    @staticmethod
    def backward(ctx, dy):
        return dy


class _MutatingForward(torch.autograd.Function):
    mutate = False
    calls = 0

    @staticmethod
    def forward(ctx, x):
        _MutatingForward.calls += 1
        if _MutatingForward.mutate:
            x.add_(1)
        return x * 2

    @staticmethod
    def backward(ctx, dy):
        return dy


class _AliasingForward(torch.autograd.Function):
    mutate = False

    @staticmethod
    def forward(ctx, x):
        if _AliasingForward.mutate:
            x.add_(1)
        return x

    @staticmethod
    def backward(ctx, dy):
        return dy


def _capture_forward(cls, *, mutate_before_capture=False):
    with tempfile.TemporaryDirectory() as directory:
        tripwire = NaNTripwire(
            directory, capture_step=1, capture_direction="forward"
        )
        tripwire.patch_function(cls)
        try:
            tripwire.begin(1)
            x = torch.ones(2, 3)
            cls.apply(x)
            if mutate_before_capture:
                x.zero_()
            try:
                tripwire.check("forward")
            except CaptureComplete as captured:
                return torch.load(captured.capture_path, weights_only=False)
            raise AssertionError("Tripwire did not capture the forward operation")
        finally:
            tripwire.uninstall()


class StressReplayTest(unittest.TestCase):

    def test_delayed_check_reports_first_bad_iteration_and_decodes_once(self):
        _FailsAfterThree.calls = 0
        capture = _capture_forward(_FailsAfterThree)
        _FailsAfterThree.calls = 0
        with patch("scripts.replay_nan_tripwire.decode_payload", wraps=decode_payload) as decode:
            report = stress_replay(capture, "cpu", repeat=12, check_every=5)
        self.assertEqual(decode.call_count, 1)
        self.assertTrue(decode.call_args.kwargs["replay_only"])
        self.assertEqual(report["iterations_completed"], 5)
        self.assertEqual(report["first_nonfinite_iteration"], 4)
        self.assertEqual(len(report["checks"]), 1)
        self.assertEqual(
            report["checks"][0]["failures"],
            [{"iteration": 4, "output": ""}, {"iteration": 5, "output": ""}],
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_rng_and_context_reset_while_input_storage_is_reused(self):
        capture = _capture_forward(_RandomForward)
        _RandomForward.draws.clear()
        _RandomForward.input_ptrs.clear()
        report = stress_replay(capture, "cpu", repeat=7, check_every=3)
        self.assertIsNone(report["first_nonfinite_iteration"])
        self.assertEqual(report["iterations_completed"], 7)
        self.assertEqual(
            [entry["checked_through_iteration"] for entry in report["checks"]],
            [3, 6, 7],
        )
        self.assertEqual(len(set(_RandomForward.input_ptrs)), 1)
        self.assertTrue(all(
            torch.equal(draw, _RandomForward.draws[0])
            for draw in _RandomForward.draws
        ))
        _RandomForward.draws.clear()

    def test_tracked_argument_mutation_stops_before_second_iteration(self):
        _MutatingForward.mutate = False
        capture = _capture_forward(_MutatingForward)
        _MutatingForward.calls = 0
        _MutatingForward.mutate = True
        try:
            with self.assertRaisesRegex(ValueError, "input mutated at iteration 1: args"):
                stress_replay(capture, "cpu", repeat=5, check_every=5)
            self.assertEqual(_MutatingForward.calls, 1)
        finally:
            _MutatingForward.mutate = False

    def test_known_mutated_capture_is_rejected_before_execution(self):
        _MutatingForward.mutate = False
        capture = _capture_forward(_MutatingForward, mutate_before_capture=True)
        _MutatingForward.calls = 0
        with self.assertRaisesRegex(ValueError, "changed between observation and capture"):
            stress_replay(capture, "cpu", repeat=5)
        self.assertEqual(_MutatingForward.calls, 0)


class OriginalOutputComparisonTest(unittest.TestCase):
    def run_main(self, actual_outputs, original, *options):
        payload = {
            "replay": {"module": "cpu_test", "class": "Operation", "direction": "forward"},
            "inputs": torch.ones(2),
            "outputs": original,
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            argv = [
                "replay_nan_tripwire.py", "capture.pt", "--device", "cpu",
                "--report", str(report_path), "--repeat", str(len(actual_outputs)),
                *options,
            ]
            with (
                patch("sys.argv", argv),
                patch.object(replay.torch, "load", return_value={"format_version": 1}),
                patch.object(replay, "changed_capture_inputs", return_value=[]),
                patch.object(
                    replay, "replay_once",
                    side_effect=[(output, payload) for output in actual_outputs],
                ) as execute,
                redirect_stdout(io.StringIO()),
            ):
                status = replay.main()
            report = json.loads(report_path.read_text())
        self.assertFalse(torch.cuda.is_initialized())
        return status, report, execute.call_args_list

    def test_finite_wrong_outputs_report_count_and_absolute_error(self):
        comparison = replay.compare_original_outputs(
            {"gradients": (torch.tensor([1.0, 4.0, -2.0]), None)},
            {"gradients": (torch.tensor([1.0, 2.0, 2.0]), None)},
        )
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["structure_errors"], [])
        self.assertEqual(len(comparison["tensors"]), 1)
        tensor = comparison["tensors"][0]
        self.assertEqual(tensor["name"], "gradients[0]")
        self.assertFalse(tensor["allclose"])
        self.assertEqual(tensor["mismatched_elements"], 2)
        self.assertEqual(tensor["max_abs_diff"], 4.0)

    def test_nested_structure_mismatches_are_failures(self):
        tensor = torch.ones(2)
        cases = [
            ({"out": tensor.reshape(1, 2)}, {"out": tensor}, "shape differs"),
            ({"out": tensor.double()}, {"out": tensor}, "dtype differs"),
            ({"out": [tensor]}, {"out": (tensor,)}, "tuple/list mismatch"),
            ({"out": (tensor,)}, {"out": (tensor, None)}, "sequence lengths differ"),
            ({"out": tensor}, {"out": tensor, "extra": None}, "dictionary keys differ"),
            ({"out": None}, {"out": tensor}, "tensor/non-tensor mismatch"),
            ({"out": (tensor, 0)}, {"out": (tensor, None)}, "non-tensor values or types differ"),
        ]
        for actual, original, error in cases:
            with self.subTest(error=error):
                comparison = replay.compare_original_outputs(actual, original)
                self.assertFalse(comparison["passed"])
                self.assertTrue(any(error in item for item in comparison["structure_errors"]))

    def test_integer_outputs_compare_exactly_without_float_precision_loss(self):
        original = torch.tensor([2**60, 2**60 + 1], dtype=torch.int64)
        actual = torch.tensor([2**60, 2**60 + 2], dtype=torch.int64)
        comparison = replay.compare_original_outputs(actual, original, rtol=1, atol=1)
        self.assertFalse(comparison["passed"])
        tensor = comparison["tensors"][0]
        self.assertEqual(tensor["comparison"], "exact")
        self.assertEqual(tensor["mismatched_elements"], 1)
        self.assertIsNone(tensor["max_abs_diff"])

    def test_matching_noncontiguous_scalar_and_empty_tensors_pass(self):
        original = (torch.arange(12.0).reshape(3, 4).t(), torch.tensor(2.0), torch.empty(0, 2))
        actual = tuple(tensor.clone() for tensor in original)
        self.assertFalse(original[0].is_contiguous())
        comparison = replay.compare_original_outputs(actual, original, rtol=0, atol=0)
        self.assertTrue(comparison["passed"])
        self.assertEqual(len(comparison["tensors"]), 3)
        self.assertTrue(all(item["mismatched_elements"] == 0 for item in comparison["tensors"]))

    def test_comparison_is_opt_in_for_finite_wrong_outputs(self):
        actual, original = torch.zeros(2), torch.ones(2)
        status, report, calls = self.run_main([actual], original)
        self.assertEqual(status, 0)
        self.assertNotIn("comparison", report["trials"][0])
        self.assertFalse(calls[0].kwargs["snapshot_original_outputs"])

        status, report, calls = self.run_main([actual], original, "--compare-original")
        self.assertEqual(status, 1)
        self.assertEqual(report["trials"][0]["nonfinite"], 0)
        self.assertFalse(report["comparisons_passed"])
        self.assertFalse(report["trials"][0]["comparison"]["passed"])
        self.assertTrue(calls[0].kwargs["snapshot_original_outputs"])

    def test_cli_default_tolerance_passes_and_strict_tolerance_fails(self):
        actual, original = torch.tensor([1.005, 1e-13]), torch.tensor([1.0, 0.0])
        status, report, _ = self.run_main([actual], original, "--compare-original")
        self.assertEqual(status, 0)
        self.assertTrue(report["comparisons_passed"])
        self.assertEqual(report["rtol"], 0.01)
        self.assertEqual(report["atol"], 1e-12)

        status, report, _ = self.run_main(
            [actual], original, "--compare-original", "--rtol", "0", "--atol", "0"
        )
        self.assertEqual(status, 1)
        comparison = report["trials"][0]["comparison"]
        self.assertEqual(comparison["tensors"][0]["mismatched_elements"], 2)

    def test_default_tolerance_detects_zeroed_small_gradients(self):
        original = torch.tensor([2e-9, -2e-9])
        zeroed = torch.zeros_like(original)
        self.assertFalse(replay.compare_original_outputs(zeroed, original)["passed"])
        status, report, _ = self.run_main([zeroed], original, "--compare-original")
        self.assertEqual(status, 1)
        self.assertEqual(report["trials"][0]["nonfinite"], 0)
        self.assertEqual(
            report["trials"][0]["comparison"]["tensors"][0]["mismatched_elements"], 2
        )

        rounded = original * 1.005
        self.assertTrue(torch.all((rounded - original).abs() > 1e-12))
        self.assertTrue(replay.compare_original_outputs(rounded, original)["passed"])
        status, report, _ = self.run_main([rounded], original, "--compare-original")
        self.assertEqual(status, 0)
        self.assertTrue(report["comparisons_passed"])

    def test_later_passing_trial_does_not_hide_comparison_failure(self):
        original = torch.ones(2)
        status, report, _ = self.run_main(
            [torch.zeros(2), original.clone()], original, "--compare-original"
        )
        self.assertEqual(status, 1)
        self.assertFalse(report["comparisons_passed"])
        self.assertEqual(
            [trial["comparison"]["passed"] for trial in report["trials"]],
            [False, True],
        )

    def test_nonfinite_expectation_does_not_bypass_comparison(self):
        same_nan = torch.tensor([float("nan")])
        status, report, _ = self.run_main(
            [same_nan], same_nan, "--compare-original", "--expect-nonfinite"
        )
        self.assertEqual(status, 1)
        comparison = report["trials"][0]["comparison"]
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["tensors"][0]["mismatched_elements"], 1)

        same_inf = torch.tensor([float("inf")])
        status, report, _ = self.run_main([same_inf], same_inf, "--compare-original")
        self.assertEqual(status, 1)
        self.assertTrue(report["comparisons_passed"])
        self.assertEqual(report["trials"][0]["nonfinite"], 1)

    def test_invalid_tolerances_are_rejected(self):
        tensor = torch.ones(1)
        for option in ("rtol", "atol"):
            for value in (-1.0, float("inf"), float("nan")):
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                        replay.compare_original_outputs(tensor, tensor, **{option: value})
                    with (
                        patch("sys.argv", ["replay_nan_tripwire.py", "capture.pt", f"--{option}", str(value)]),
                        patch.object(replay.torch, "load") as load,
                        redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as error,
                    ):
                        replay.main()
                    self.assertEqual(error.exception.code, 2)
                    load.assert_not_called()

    def test_stress_comparison_is_rejected_before_loading_capture(self):
        with (
            patch("sys.argv", ["replay_nan_tripwire.py", "capture.pt", "--stress", "--compare-original"]),
            patch.object(replay.torch, "load") as load,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            replay.main()
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()

    def test_aliased_original_output_is_frozen_before_in_place_replay(self):
        _AliasingForward.mutate = False
        capture = _capture_forward(_AliasingForward)
        decoded = decode_payload(capture, "cpu")
        self.assertEqual(
            decoded["outputs"].untyped_storage().data_ptr(),
            decoded["replay"]["args"][0].untyped_storage().data_ptr(),
        )
        _AliasingForward.mutate = True
        try:
            actual, payload = replay.replay_once(
                capture, "cpu", snapshot_original_outputs=True
            )
        finally:
            _AliasingForward.mutate = False
        self.assertTrue(torch.equal(actual, torch.full((2, 3), 2.0)))
        self.assertTrue(torch.equal(payload["outputs"], torch.ones(2, 3)))
        self.assertFalse(replay.compare_original_outputs(actual, payload["outputs"])["passed"])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
