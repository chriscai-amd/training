"""CPU checks of the second GEMM in the actual output autograd class.

The shared fixtures extract production Python autograd classes and replace only
their numerical helpers. No GPU operation imports or launches are needed.
"""

import functools
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from capture_training_backward import _rng_equal
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from nan_backward_training_extended import (
    ExtendedTrainingBackwardBoundaryProbe, OUTPUT_WEIGHT_OPERATION,
)
from nan_replay_state import capture_rng
from replay_backward_boundary import require_pristine_input_snapshot
from test_nan_backward_output import Model, make_modules
from test_nan_backward_training_extended import Layer as FullLayer, make_full_modules


OUTPUT_OPERATIONS = (
    "hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd", OUTPUT_WEIGHT_OPERATION,
)
FOUR_OPERATIONS = (
    "hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd",
    "triton_addmm_bwd", "triton_weighted_layer_norm_bwd",
)
SIX_OPERATIONS = (
    "hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd",
    "hstu_attention_bwd", "hstu_silu_bwd", "triton_addmm_bwd",
    "triton_weighted_layer_norm_bwd",
)
SEVEN_OPERATIONS = (*OUTPUT_OPERATIONS, *SIX_OPERATIONS[2:])


class OutputWeightBoundaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = 0
        environment = patch.dict(os.environ, {
            "NAN_BACKWARD_TARGET": "_stu_layers.1.", "NAN_BACKWARD_SAVE_ALL": "0",
            "NAN_BACKWARD_ABS_THRESHOLD": "1e20", "NAN_BACKWARD_CHUNK_MIB": "1",
            "NAN_BACKWARD_MAX_CAPTURE_GIB": "32",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def setup_probe(self, *, mode="monitor_on_anomaly", recompute=False,
                    operations=(OUTPUT_WEIGHT_OPERATION,), full=False,
                    save_all=False, clone_returned_gradient=False):
        if full:
            compute, preprocess, output, preprocessing = make_full_modules()
            model = torch.nn.Module()
            model._stu_layers = torch.nn.ModuleList([FullLayer(compute), FullLayer(compute)])
        else:
            compute, preprocess, output = make_modules()
            model = Model(compute, recompute=recompute)
            preprocessing = None
        state = types.SimpleNamespace(
            calls=[], norm_y=None, forward_y=None, extreme_second=False,
            mutate_second=False, extreme_saved_y=False, extreme_recomputed_y=False,
        )
        original_torch = output.torch
        numerical_mm = original_torch.mm
        numerical_forward = output.triton_layer_norm_mul_dropout_fwd
        numerical_norm = output.triton_layer_norm_mul_dropout_bwd

        @functools.wraps(numerical_forward)
        def forward(*args, **kwargs):
            result = numerical_forward(*args, **kwargs)
            state.forward_y = result[0]
            if state.extreme_saved_y:
                result[0][0, 1] = 1e25
            return result

        @functools.wraps(numerical_norm)
        def norm(*args, **kwargs):
            result = numerical_norm(*args, **kwargs)
            state.norm_y = result[4]
            if state.extreme_recomputed_y:
                result[4][0, 1] = 1e25
            return result

        def mm(left, right):
            call = {"left": left, "right": right}
            state.calls.append(call)
            if len(state.calls) == 2:
                call.update(y_before=left.t().clone(), dout_before=right.clone(),
                            expected_y=state.probe._local.output_backward.get("expected_y"))
            result = numerical_mm(left, right)
            call["result"] = result
            if len(state.calls) == 2:
                if state.mutate_second:
                    left.fill_(42.)
                    right.fill_(43.)
                if state.extreme_second:
                    result[0, 1] = 1e25
            return result

        output.triton_layer_norm_mul_dropout_fwd = forward
        output.triton_layer_norm_mul_dropout_bwd = norm
        original_torch.mm = mm
        if clone_returned_gradient:
            numerical_backward = output.HSTUComputeOutputFunction.backward

            @functools.wraps(numerical_backward)
            def backward(ctx, dout):
                result = list(numerical_backward(ctx, dout))
                result[5] = result[5].clone()
                return tuple(result)

            output.HSTUComputeOutputFunction.backward = staticmethod(backward)
        original_backward = inspect.getattr_static(output.HSTUComputeOutputFunction, "backward")
        self.case += 1
        directory = self.root / str(self.case)
        with patch.dict(os.environ, {"NAN_BACKWARD_SAVE_ALL": "1" if save_all else "0"}):
            probe = ExtendedTrainingBackwardBoundaryProbe(
                model, directory, mode=mode, selected_operations=operations,
                compute_module=compute, preprocess_module=preprocess, output_module=output,
            )
        self.addCleanup(probe.close)
        state.probe = probe
        probe.set_attempt("training", 0, 12)
        return types.SimpleNamespace(
            model=model, probe=probe, output=output, state=state, directory=directory,
            preprocessing=preprocessing, original_torch=original_torch,
            original_backward=original_backward, original_norm=norm,
        )

    def execute(self, fixture, *, layer=1, group_norm=False):
        x = (torch.arange(12, dtype=torch.float64).reshape(3, 4) / 16).requires_grad_()
        result = fixture.model._stu_layers[layer](x)
        if group_norm:
            result.grad_fn.group_norm = True
        # This zero-stride input checks that replay retains actual GEMM layout.
        fixture.dout = torch.tensor(0.5, dtype=x.dtype).expand_as(result)
        fixture.before_backward_pointers = fixture.probe._pointers.copy()
        result.backward(fixture.dout)
        return x

    def dumps(self, fixture):
        return [torch.load(path, map_location="cpu", weights_only=False)
                for path in sorted(fixture.directory.glob("*.pt"))]

    def events(self, fixture, kind=None):
        events = [json.loads(line) for line in
                  (fixture.directory / "boundaries.jsonl").read_text().splitlines()]
        return events if kind is None else [event for event in events if event["event"] == kind]

    def assert_scope_clean(self, fixture):
        self.assertIsNone(fixture.probe._local.output_backward)
        self.assertEqual(fixture.probe._pointers, fixture.before_backward_pointers)
        self.assertFalse(torch.cuda.is_initialized())

    def test_saved_and_recomputed_y_capture_actual_inputs_with_norm_filtered(self):
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                fixture = self.setup_probe(mode="pristine", recompute=recompute, save_all=True)
                self.execute(fixture)
                dump, = self.dumps(fixture)
                self.assertEqual(dump["operation"], OUTPUT_WEIGHT_OPERATION)
                self.assertEqual(dump["stage"], "finite")
                self.assertEqual(set(dump["kwargs"]), {"y", "dout"})
                self.assertEqual(dump["args"], ())
                self.assertEqual(dump["kwargs"]["dout"].stride(), (0, 0))
                self.assertEqual(dump["event"]["parameter"], "model._stu_layers.1._output_weight")
                expected_y = fixture.state.norm_y if recompute else fixture.state.forward_y
                second = fixture.state.calls[1]
                self.assertIs(second["right"], fixture.dout)
                self.assertEqual(second["expected_y"].untyped_storage()._cdata,
                                 expected_y.untyped_storage()._cdata)
                self.assertEqual(second["left"].untyped_storage()._cdata,
                                 expected_y.untyped_storage()._cdata)
                self.assertTrue(torch.equal(dump["kwargs"]["y"], expected_y))
                self.assertTrue(torch.equal(dump["outputs"][0], expected_y.t() @ fixture.dout))
                self.assertEqual(fixture.original_torch.mm_calls, 2)
                self.assertEqual(len(fixture.output.norm_calls), 1)
                if not recompute:
                    self.assertIsNone(fixture.state.norm_y)
                self.assert_scope_clean(fixture)

    def test_seven_boundaries_preserve_order_rng_and_healthy_no_copy_policy(self):
        fixture = self.setup_probe(full=True, operations=SEVEN_OPERATIONS)
        before_rng = capture_rng()
        original_mm = torch.mm
        with patch("nan_backward_boundaries._cpu_copy_tree", side_effect=AssertionError("healthy copy")), \
             patch("nan_backward_training_probe._cpu_copy_tree", side_effect=AssertionError("healthy copy")), \
             patch("nan_backward_training_extended._cpu_copy_tree", side_effect=AssertionError("healthy copy")):
            self.execute(fixture)
        after = self.events(fixture, "after")
        self.assertEqual([event["operation"] for event in after], list(SEVEN_OPERATIONS))
        self.assertEqual([event["call"] for event in after], list(range(1, 8)))
        weight_event = after[2]
        self.assertEqual({item["name"] for item in weight_event["inputs"]}, {"y", "dout"})
        self.assertTrue(_rng_equal(before_rng, capture_rng()))
        self.assertEqual(fixture.original_torch.mm_calls, 2)
        self.assertEqual((fixture.preprocessing.attention_calls, fixture.preprocessing.silu_calls), (1, 1))
        self.assertFalse(self.dumps(fixture))
        self.assertIs(torch.mm, original_mm)
        self.assert_scope_clean(fixture)

    def test_second_gemm_fault_stops_before_attention_and_silu(self):
        fixture = self.setup_probe(full=True, operations=SEVEN_OPERATIONS)
        fixture.state.extreme_second = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(fixture)
        dump, = self.dumps(fixture)
        self.assertEqual((dump["operation"], dump["stage"]), (OUTPUT_WEIGHT_OPERATION, "output"))
        self.assertEqual(dump["event"]["extreme_outputs"], ["d_output_weight"])
        self.assertEqual(dump["outputs"][0][0, 1].item(), 1e25)
        after = self.events(fixture, "after")
        self.assertEqual([event["operation"] for event in after], list(OUTPUT_OPERATIONS))
        for event in after[:2]:
            self.assertTrue(all(item["finite"] and not item["extreme"] for item in event["outputs"]))
        self.assertEqual(fixture.original_torch.mm_calls, 2)
        self.assertEqual((fixture.preprocessing.attention_calls, fixture.preprocessing.silu_calls), (0, 0))
        self.assertTrue(fixture.probe.failed)
        self.assert_scope_clean(fixture)

    def test_saved_or_recomputed_extreme_y_stops_before_weight_gemm(self):
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                fixture = self.setup_probe(recompute=recompute)
                fixture.state.extreme_saved_y = not recompute
                fixture.state.extreme_recomputed_y = recompute
                with self.assertRaises(BoundaryAnomalyError):
                    self.execute(fixture)
                dump, = self.dumps(fixture)
                self.assertEqual((dump["operation"], dump["stage"]), (OUTPUT_WEIGHT_OPERATION, "input"))
                self.assertEqual(dump["event"]["extreme_inputs"], ["y"])
                self.assertTrue(dump["input_snapshot_pristine"])
                self.assertIsNone(dump["outputs"])
                self.assertEqual(fixture.original_torch.mm_calls, 1)
                require_pristine_input_snapshot(dump)
                self.assert_scope_clean(fixture)

    def test_second_gemm_input_mutation_distinguishes_pristine_and_monitor(self):
        for mode in ("pristine", "monitor_on_anomaly"):
            with self.subTest(mode=mode):
                fixture = self.setup_probe(mode=mode, recompute=True)
                fixture.state.mutate_second = fixture.state.extreme_second = True
                with self.assertRaises(BoundaryAnomalyError):
                    self.execute(fixture)
                dump, = self.dumps(fixture)
                before = fixture.state.calls[1]
                self.assertEqual(set(dump["kwargs"]), {"y", "dout"})
                self.assertTrue(torch.equal(before["left"], torch.full_like(before["left"], 42.)))
                self.assertTrue(torch.equal(before["right"], torch.full_like(before["right"], 43.)))
                if mode == "pristine":
                    self.assertTrue(torch.equal(dump["kwargs"]["y"], before["y_before"]))
                    self.assertTrue(torch.equal(dump["kwargs"]["dout"], before["dout_before"]))
                    require_pristine_input_snapshot(dump)
                else:
                    self.assertFalse(dump["input_snapshot_pristine"])
                    self.assertTrue(torch.equal(dump["kwargs"]["y"], before["left"].t()))
                    self.assertTrue(torch.equal(dump["kwargs"]["dout"], before["right"]))
                    inputs = {entry["name"]: entry for entry in dump["event"]["input_device_monitor"]}
                    self.assertLess(inputs["y"]["max_abs_finite"], 42.)
                    self.assertEqual(inputs["dout"]["max_abs_finite"], 0.5)
                    with self.assertRaisesRegex(ValueError, "Post-call input snapshots"):
                        require_pristine_input_snapshot(dump)
                self.assertEqual(fixture.original_torch.mm_calls, 2)
                self.assert_scope_clean(fixture)

    def test_wrong_second_operands_are_rejected_before_numerical_mm(self):
        for wrong in ("y_copy", "dout_copy", "y_negative_view", "y_stride"):
            with self.subTest(wrong=wrong):
                fixture = self.setup_probe(recompute=True)
                intercept = fixture.probe._output_mm

                def tampered(function, args, kwargs):
                    scope = fixture.probe._local.output_backward
                    if scope["grad_mm_calls"]:
                        left, right = args
                        if wrong == "y_copy":
                            left = left.clone()
                        elif wrong == "dout_copy":
                            right = right.clone()
                        elif wrong == "y_negative_view":
                            left = torch._neg_view(left)
                        else:
                            left = left.as_strided(left.shape, (3, 1))
                        args = (left, right)
                    return intercept(function, args, kwargs)

                with patch.object(fixture.probe, "_output_mm", side_effect=tampered):
                    with self.assertRaises(BoundaryProbeError):
                        self.execute(fixture)
                self.assertEqual(fixture.original_torch.mm_calls, 1)
                self.assertFalse(self.dumps(fixture))
                self.assert_scope_clean(fixture)

    def test_missing_duplicate_or_replaced_second_result_is_rejected(self):
        for change in ("missing", "duplicate", "returned_copy"):
            with self.subTest(change=change):
                fixture = self.setup_probe(clone_returned_gradient=change == "returned_copy")
                intercept = fixture.probe._output_mm

                def changed(function, args, kwargs):
                    scope = fixture.probe._local.output_backward
                    is_second = bool(scope["grad_mm_calls"])
                    if is_second and change == "missing":
                        return function(*args, **kwargs)
                    result = intercept(function, args, kwargs)
                    if is_second and change == "duplicate":
                        intercept(function, args, kwargs)
                    return result

                with patch.object(fixture.probe, "_output_mm", side_effect=changed):
                    with self.assertRaises(BoundaryProbeError):
                        self.execute(fixture)
                self.assertEqual(fixture.original_torch.mm_calls, 2)
                self.assertFalse(self.dumps(fixture))
                self.assert_scope_clean(fixture)

    def test_existing_four_and_six_selections_keep_their_original_events(self):
        for operations in (FOUR_OPERATIONS, SIX_OPERATIONS):
            with self.subTest(operations=operations):
                fixture = self.setup_probe(full=True, operations=operations)
                self.execute(fixture)
                self.assertEqual([event["operation"] for event in self.events(fixture, "after")],
                                 list(operations))
                self.assertFalse(any(event.get("operation") == OUTPUT_WEIGHT_OPERATION
                                     for event in self.events(fixture)))
                self.assertEqual(fixture.original_torch.mm_calls, 2)
                self.assert_scope_clean(fixture)

    def test_optional_group_norm_rejected_before_backward_arithmetic(self):
        fixture = self.setup_probe()
        with self.assertRaises(BoundaryProbeError):
            self.execute(fixture, group_norm=True)
        self.assertEqual(fixture.original_torch.mm_calls, 0)
        self.assertFalse(fixture.output.norm_calls)
        self.assertFalse(self.dumps(fixture))
        self.assertIsNone(getattr(fixture.probe._local, "output_backward", None))
        self.assertEqual(fixture.probe._pointers, fixture.before_backward_pointers)

    def test_failed_probe_restores_local_proxy_and_original_descriptors(self):
        fixture = self.setup_probe()
        original_global_mm = torch.mm
        fixture.state.extreme_second = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(fixture)
        self.assert_scope_clean(fixture)
        fixture.probe.close()
        self.assertIs(fixture.output.torch, fixture.original_torch)
        self.assertIs(fixture.output.triton_layer_norm_mul_dropout_bwd, fixture.original_norm)
        self.assertIs(inspect.getattr_static(fixture.output.HSTUComputeOutputFunction, "backward"),
                      fixture.original_backward)
        self.assertIs(torch.mm, original_global_mm)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    program = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU tests unexpectedly initialized CUDA"
    sys.exit(not program.result.wasSuccessful())
