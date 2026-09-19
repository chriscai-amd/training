"""CPU integration checks with real preprocessing/output autograd classes."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import nan_backward_training_batched as batched
import nan_backward_training_extended as extended
import nan_backward_training_probe as regular
import nan_backward_training_upstream as upstream
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from test_nan_backward_training_extended import Layer, make_full_modules


class Stack(torch.nn.Module):
    def __init__(self, compute, count=3):
        super().__init__()
        self._stu_layers = torch.nn.ModuleList([Layer(compute) for _ in range(count)])
        self.corrupt_lower_incoming = False

    def forward(self, x):
        for index, layer in enumerate(self._stu_layers):
            x = layer(x)
            if index == 1 and self.corrupt_lower_incoming:
                def corrupt(gradient):
                    result = gradient.clone()
                    result[1, 2] = 1e25
                    return result
                x.register_hook(corrupt)
        return x


class UpstreamBoundaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        environment = patch.dict(os.environ, {
            "NAN_BACKWARD_TARGET": "*", "NAN_BACKWARD_SAVE_ALL": "0",
            "NAN_BACKWARD_ABS_THRESHOLD": "1e20", "NAN_BACKWARD_CHUNK_MIB": "1",
            "NAN_BACKWARD_MAX_CAPTURE_GIB": "32"})
        environment.start()
        self.addCleanup(environment.stop)

    def setup_probe(self):
        compute, preprocess, output, state = make_full_modules()
        model = Stack(compute)
        probe = upstream.UpstreamBatchedTrainingBackwardBoundaryProbe(
            model, self.directory, compute_module=compute, preprocess_module=preprocess,
            output_module=output)
        self.addCleanup(probe.close)
        probe.set_attempt("training", 0, 1)
        return model, probe, state, output

    def execute(self, model, *, only_top=False):
        x = (torch.arange(12, dtype=torch.float64).reshape(3, 4) / 64).requires_grad_()
        y = model._stu_layers[2](x) if only_top else model(x)
        y.backward(torch.full_like(y, 0.5))
        return x

    def events(self):
        return [json.loads(line) for line in (self.directory / "boundaries.jsonl").read_text().splitlines()]

    def dumps(self):
        return [torch.load(path, map_location="cpu", weights_only=False)
                for path in self.directory.glob("*.pt")]

    def test_full_stack_has_exact_upper7_before_after_lower_input_only_and15_batches(self):
        model, probe, state, output = self.setup_probe()
        initial_rng = torch.get_rng_state().clone()
        original_reader = batched._read_endpoint_batch
        with patch.object(batched, "_read_endpoint_batch", wraps=original_reader) as reader, \
                patch.object(regular, "monitor_summaries", side_effect=AssertionError("old regular scanner")), \
                patch.object(extended, "monitor_summaries", side_effect=AssertionError("old readwrite scanner")), \
                patch.object(regular, "_cpu_copy_tree", side_effect=AssertionError("healthy capture")), \
                patch.object(extended, "_cpu_copy_tree", side_effect=AssertionError("healthy capture")):
            self.execute(model)
        before = [event for event in self.events() if event["event"] == "before"]
        after = [event for event in self.events() if event["event"] == "after"]
        self.assertEqual({(event["layer"], event["operation"]) for event in before},
                         {("model._stu_layers.2", operation) for operation in upstream.ALL_OPERATIONS}
                         | {("model._stu_layers.1", upstream.LOWER_OPERATION)})
        self.assertEqual(len(before), 8)
        self.assertEqual({(event["layer"], event["operation"]) for event in after},
                         {("model._stu_layers.2", operation) for operation in upstream.ALL_OPERATIONS})
        self.assertEqual(len(after), 7)
        self.assertEqual(reader.call_count, 15)
        skipped = [event for event in self.events() if event["event"] == "skipped"]
        self.assertEqual(len(skipped), 13)
        self.assertTrue(all(event["reason"] == "layer_operation_filter" for event in skipped))
        self.assertEqual(probe.call, 21)
        self.assertEqual((state.attention_calls, state.silu_calls), (3, 3))
        self.assertEqual(output.torch._original.mm_calls, 6)
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        lower = before[-1]
        self.assertTrue(lower["input_only_witness"])
        self.assertFalse(lower["output_values_monitored"])
        witnesses = lower["residual_flow_witnesses"]
        self.assertEqual(set(witnesses), {"upper_original_dout", "upper_preprocessing_d_x", "lower_incoming_dout"})
        upper_bound = sum(witnesses[name]["summary"]["max_abs_finite"]
                          for name in ("upper_original_dout", "upper_preprocessing_d_x"))
        self.assertLessEqual(witnesses["lower_incoming_dout"]["summary"]["max_abs_finite"], upper_bound)
        self.assertFalse(self.dumps())
        self.assertFalse(torch.cuda.is_initialized())

    def test_bad_lower_accumulated_gradient_stops_before_its_gemm_with_upper_witnesses(self):
        model, probe, state, output = self.setup_probe()
        model.corrupt_lower_incoming = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertEqual((dump["layer"], dump["operation"], dump["stage"]),
                         ("model._stu_layers.1", upstream.LOWER_OPERATION, "input"))
        self.assertTrue(dump["input_snapshot_pristine"])
        self.assertFalse(dump["event"]["operation_executed"])
        self.assertIsNone(dump["outputs"])
        self.assertEqual(dump["kwargs"]["dout"][1, 2].item(), 1e25)
        self.assertEqual(output.torch._original.mm_calls, 2)
        self.assertEqual((state.attention_calls, state.silu_calls), (1, 1))
        witnesses = dump["event"]["residual_flow_witnesses"]
        self.assertEqual(set(witnesses), {"upper_original_dout", "upper_preprocessing_d_x", "lower_incoming_dout"})
        self.assertTrue(all(witnesses[name]["summary"]["max_abs_finite"] < 1e20
                            for name in ("upper_original_dout", "upper_preprocessing_d_x")))
        self.assertEqual(witnesses["lower_incoming_dout"]["summary"]["max_abs_finite"], 1e25)
        self.assertEqual(dump["event"]["boundary_scan_policy"], upstream.UPSTREAM_POLICY)
        self.assertTrue(probe.failed)

    def test_recomputed_q_fault_stops_before_attention_and_retains_pristine_policy(self):
        model, probe, state, _ = self.setup_probe()
        state.corrupt_recomputed_q = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model, only_top=True)
        dump, = self.dumps()
        self.assertEqual((dump["operation"], dump["stage"]), ("hstu_attention_bwd", "input"))
        self.assertTrue(dump["input_snapshot_pristine"])
        self.assertTrue(dump["initial_destination_bytes_retained"])
        self.assertTrue(torch.isnan(dump["kwargs"]["dq"]).all())
        self.assertEqual((state.attention_calls, state.silu_calls), (0, 0))
        self.assertEqual(dump["event"]["boundary_scan_policy"], upstream.UPSTREAM_POLICY)
        self.assertEqual(set(dump["event"]["residual_flow_witnesses"]), {"upper_original_dout"})

    def test_attention_output_fault_retains_postcall_mutation_and_batched_readwrite_policy(self):
        model, _, state, _ = self.setup_probe()
        state.extreme_attention = state.mutate_attention_inputs = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model, only_top=True)
        dump, = self.dumps()
        self.assertEqual((dump["operation"], dump["stage"]), ("hstu_attention_bwd", "output"))
        self.assertFalse(dump["input_snapshot_pristine"])
        self.assertFalse(dump["initial_destination_bytes_retained"])
        self.assertTrue(torch.equal(dump["kwargs"]["q"], torch.full_like(dump["kwargs"]["q"], 42.)))
        self.assertEqual(dump["event"]["boundary_scan_policy"], upstream.UPSTREAM_POLICY)
        self.assertEqual(dump["outputs"][0].untyped_storage()._cdata,
                         dump["outputs"][1].untyped_storage()._cdata)
        self.assertEqual((state.attention_calls, state.silu_calls), (1, 0))

    def test_silu_write_only_input_is_not_scanned_and_output_fault_is_localized(self):
        model, _, state, _ = self.setup_probe()
        state.extreme_silu = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model, only_top=True)
        dump, = self.dumps()
        self.assertEqual((dump["operation"], dump["stage"]), ("hstu_silu_bwd", "output"))
        self.assertEqual(dump["event"]["extreme_outputs"], ["du"])
        self.assertNotIn("grad_input", {item["name"] for item in dump["event"]["input_device_monitor"]})
        self.assertEqual(dump["event"]["boundary_scan_policy"], upstream.UPSTREAM_POLICY)

    def test_attempt_reset_drops_previous_scalar_witnesses(self):
        model, probe, _, _ = self.setup_probe()
        self.execute(model)
        self.assertEqual(len(probe._residual_summaries), 3)
        probe.set_attempt("training", 0, 2)
        self.assertEqual(probe._residual_summaries, {})
        self.execute(model)
        self.assertEqual(len(probe._residual_summaries), 3)

    def test_missing_or_ambiguous_layers_fail_before_any_wrappers_are_installed(self):
        for ambiguous in (False, True):
            with self.subTest(ambiguous=ambiguous):
                compute, preprocess, output, _ = make_full_modules()
                if ambiguous:
                    model = torch.nn.Module()
                    model.left, model.right = Stack(compute), Stack(compute)
                else:
                    model = Stack(compute, count=2)
                original_output = output.HSTUComputeOutputFunction.backward
                original_preprocess = preprocess._HSTUPreprocessAndAttentionFunction.backward
                original_torch = output.torch
                with self.assertRaises(BoundaryProbeError):
                    upstream.UpstreamBatchedTrainingBackwardBoundaryProbe(
                        model, self.directory / str(ambiguous), compute_module=compute,
                        preprocess_module=preprocess, output_module=output)
                self.assertIs(output.HSTUComputeOutputFunction.backward, original_output)
                self.assertIs(preprocess._HSTUPreprocessAndAttentionFunction.backward, original_preprocess)
                self.assertIs(output.torch, original_torch)
                self.assertTrue(all(not layer._forward_hooks and not layer._forward_pre_hooks
                                    for layer in model.modules()))

    def test_scoped_methods_leave_both_original_global_namespaces_unchanged(self):
        regular_summary = regular.TrainingBackwardBoundaryProbe._run_monitor.__globals__["monitor_summaries"]
        readwrite_summary = extended.ExtendedTrainingBackwardBoundaryProbe._run_readwrite.__globals__["monitor_summaries"]
        model, probe, _, _ = self.setup_probe()
        self.execute(model)
        self.assertIs(regular.TrainingBackwardBoundaryProbe._run_monitor.__globals__["monitor_summaries"], regular_summary)
        self.assertIs(extended.ExtendedTrainingBackwardBoundaryProbe._run_readwrite.__globals__["monitor_summaries"], readwrite_summary)
        self.assertIsNot(probe._batched_regular_monitor.__globals__, regular.TrainingBackwardBoundaryProbe._run_monitor.__globals__)
        self.assertIsNot(probe._batched_readwrite_monitor.__globals__, extended.ExtendedTrainingBackwardBoundaryProbe._run_readwrite.__globals__)


if __name__ == "__main__":
    unittest.main()
