"""CPU integration with real autograd classes and injected propagation faults."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import nan_backward_training_all_layers as all_layers
import nan_backward_training_batched as batched
import nan_backward_training_extended as extended
import nan_backward_training_probe as regular
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from test_nan_backward_training_extended import make_full_modules
from test_nan_backward_training_upstream import Stack


class AllLayerBoundaryTest(unittest.TestCase):
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

    def setup_probe(self, *, sentinel=False):
        compute, preprocess, output, state = make_full_modules()
        model = Stack(compute)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) if sentinel else None
        probe = all_layers.AllLayersBatchedTrainingBackwardBoundaryProbe(
            model, self.directory, optimizer=optimizer, compute_module=compute,
            preprocess_module=preprocess, output_module=output)
        self.addCleanup(probe.close)
        return model, probe, state, output, optimizer

    def execute(self, model):
        x = (torch.arange(12, dtype=torch.float64).reshape(3, 4) / 64).requires_grad_()
        y = model(x)
        y.backward(torch.full_like(y, 0.5))
        return x

    def events(self):
        return [json.loads(line) for line in (self.directory / "boundaries.jsonl").read_text().splitlines()]

    def dumps(self):
        return [torch.load(path, map_location="cpu", weights_only=False)
                for path in self.directory.glob("*.pt")]

    def test_exact42_endpoints_scalars_and_unchanged_original_scanners(self):
        model, probe, state, output, _ = self.setup_probe()
        regular_summary = regular.TrainingBackwardBoundaryProbe._run_monitor.__globals__["monitor_summaries"]
        readwrite_summary = extended.ExtendedTrainingBackwardBoundaryProbe._run_readwrite.__globals__["monitor_summaries"]
        initial_rng = torch.get_rng_state().clone()
        with patch.object(batched, "_read_endpoint_batch", wraps=batched._read_endpoint_batch) as reader, \
                patch.object(regular, "monitor_summaries", side_effect=AssertionError("old scanner")), \
                patch.object(extended, "monitor_summaries", side_effect=AssertionError("old scanner")):
            probe.set_attempt("training", 0, 1)
            self.execute(model)
        probe.assert_complete_operation_scans()
        self.assertEqual(reader.call_count, 42)
        self.assertEqual(probe.call, 21)
        self.assertFalse([event for event in self.events() if event["event"] == "skipped"])
        self.assertEqual((state.attention_calls, state.silu_calls), (3, 3))
        self.assertEqual(output.torch._original.mm_calls, 6)
        attention = [event for event in self.events() if event.get("operation") == "hstu_attention_bwd"]
        self.assertEqual(len(attention), 6)
        self.assertTrue(all(event["scalar_arguments"] == {"N": 3, "alpha": 0.5, "max_attn_len": 0,
                             "contextual_seq_len": 0, "enable_tma": False, "num_softmax_heads": 0}
                            for event in attention))
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        self.assertIs(regular.TrainingBackwardBoundaryProbe._run_monitor.__globals__["monitor_summaries"], regular_summary)
        self.assertIs(extended.ExtendedTrainingBackwardBoundaryProbe._run_readwrite.__globals__["monitor_summaries"], readwrite_summary)
        self.assertFalse(self.dumps())
        self.assertFalse(torch.cuda.is_initialized())

    def test_lower_layer_attention_fault_is_first_bad_output_and_retains_scalars(self):
        model, probe, state, output, _ = self.setup_probe()
        # A hook at layer1's backward input turns the fault on only after layer2
        # has completed. All layer1 operation inputs are still bounded.
        model._stu_layers[1].register_full_backward_pre_hook(
            lambda module, gradient: setattr(state, "extreme_attention", True))
        probe.set_attempt("training", 0, 1)
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertEqual((dump["layer"], dump["operation"], dump["stage"]),
                         ("model._stu_layers.1", "hstu_attention_bwd", "output"))
        self.assertFalse(dump["input_snapshot_pristine"])
        self.assertEqual(dump["event"]["scalar_arguments"]["N"], dump["kwargs"]["N"])
        self.assertEqual(dump["event"]["boundary_scan_policy"], all_layers.ALL_LAYER_POLICY)
        self.assertEqual((state.attention_calls, state.silu_calls), (2, 1))

    def test_recomputed_input_fault_retains_pristine_readwrite_storage(self):
        model, probe, state, _, _ = self.setup_probe()
        # Forward performs three projections; first recompute is projection4.
        original = probe.preprocess.maybe_triton_addmm_fwd
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            if state.projection_calls == 4:
                result[:, 8:12] = 1e25
            return result
        probe.preprocess.maybe_triton_addmm_fwd = corrupt
        probe.set_attempt("training", 0, 1)
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertEqual((dump["operation"], dump["stage"]), ("hstu_attention_bwd", "input"))
        self.assertTrue(dump["input_snapshot_pristine"])
        self.assertTrue(dump["initial_destination_bytes_retained"])
        self.assertEqual(state.attention_calls, 0)
        self.assertTrue(torch.isnan(dump["kwargs"]["dq"]).all())

    def test_dense_phase_callback_runs_after_all_grads_before_clip_and_proxy_preserves_optimizer(self):
        model, probe, _, _, optimizer = self.setup_probe(sentinel=True)
        old_step = optimizer.step
        probe.set_attempt("training", 0, 1)
        self.execute(model)
        self.assertTrue(probe.sentinel.backward_complete)
        self.assertFalse(probe.sentinel.optimizer_complete)
        self.assertTrue(all(parameter.grad is not None for parameter in probe.sentinel.parameters.values()))
        before = [parameter.detach().clone() for parameter in model.parameters()]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        probe.sentinel.optimizer_proxy.step()
        self.assertEqual(optimizer.step, old_step)
        self.assertIs(probe.sentinel.optimizer_proxy.param_groups, optimizer.param_groups)
        self.assertTrue(probe.sentinel.optimizer_complete)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        phases = [event["phase"] for event in self.events() if event["event"] == "dense_phase"]
        self.assertEqual(phases, all_layers.DENSE_PHASE_POLICY["phases"])
        optimizer.zero_grad()
        probe.set_attempt("training", 0, 2)
        self.assertFalse(probe.sentinel.backward_complete)
        self.assertFalse(probe.sentinel.optimizer_complete)
        probe.close()
        self.assertTrue(all(not parameter._backward_hooks for parameter in model.parameters()))

    def test_accumulated_nan_stops_before_clip_or_optimizer_and_freezes_dense_state(self):
        model, probe, _, _, optimizer = self.setup_probe(sentinel=True)
        parameter = model._stu_layers[2]._output_weight
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        self.addCleanup(handle.remove)
        original_parameter = parameter.detach().clone()
        probe.set_attempt("training", 0, 1)
        with patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=AssertionError("clipping reached")), \
                patch.object(optimizer, "step", side_effect=AssertionError("optimizer reached")), \
                self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            probe.sentinel.optimizer_proxy.step()
        dump, = self.dumps()
        self.assertEqual(dump["phase"], "after_backward_before_clip")
        self.assertEqual(dump["format"], "training_dense_phase_capture_v1")
        self.assertTrue(dump["trigger_persists_in_cpu_snapshot"])
        self.assertTrue(torch.equal(parameter, original_parameter))
        probe.assert_complete_operation_scans()

    def test_postclip_bad_gradient_stops_before_optimizer(self):
        model, probe, _, _, optimizer = self.setup_probe(sentinel=True)
        probe.set_attempt("training", 0, 1)
        self.execute(model)
        model._stu_layers[0]._output_weight.grad.fill_(float("nan"))
        with patch.object(optimizer, "step", side_effect=AssertionError("optimizer reached")), \
                self.assertRaises(BoundaryAnomalyError):
            probe.sentinel.optimizer_proxy.step()
        dump, = self.dumps()
        self.assertEqual(dump["phase"], "before_optimizer_step")
        self.assertFalse(dump["event"]["explicit_optimizer_step_executed"])

    def test_bad_optimizer_output_is_distinguished_from_clean_gradients(self):
        model, probe, _, _, optimizer = self.setup_probe(sentinel=True)
        probe.set_attempt("training", 0, 1)
        self.execute(model)
        original = optimizer.step
        def poison():
            result = original()
            with torch.no_grad():
                model._stu_layers[0]._output_weight.fill_(float("nan"))
            return result
        with patch.object(optimizer, "step", side_effect=poison), self.assertRaises(BoundaryAnomalyError):
            probe.sentinel.optimizer_proxy.step()
        dump, = self.dumps()
        self.assertEqual(dump["phase"], "after_optimizer_step")
        self.assertTrue(dump["event"]["explicit_optimizer_step_executed"])
        self.assertTrue(all(name.startswith("parameters/") for name in dump["event"]["bad"]))

    def test_layer_count_and_owner_validation_precedes_wrappers(self):
        for count in (2, 4):
            with self.subTest(count=count):
                compute, preprocess, output, _ = make_full_modules()
                model = Stack(compute, count=count)
                original = output.HSTUComputeOutputFunction.backward
                with self.assertRaises(BoundaryProbeError):
                    all_layers.AllLayersBatchedTrainingBackwardBoundaryProbe(
                        model, self.directory / str(count), compute_module=compute,
                        preprocess_module=preprocess, output_module=output)
                self.assertIs(output.HSTUComputeOutputFunction.backward, original)
                self.assertTrue(all(not layer._forward_hooks for layer in model.modules()))

    def test_partial_dense_hook_installation_and_install_event_failure_clean_up(self):
        for failing_stage in ("registration", "installed_event"):
            with self.subTest(failing_stage=failing_stage):
                compute, preprocess, output, _ = make_full_modules()
                model = Stack(compute)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
                original_backward = output.HSTUComputeOutputFunction.backward
                original_hook = torch.Tensor.register_hook
                original_emit = all_layers.AllLayersBatchedTrainingBackwardBoundaryProbe._emit
                count = 0
                def register(parameter, hook):
                    nonlocal count
                    count += 1
                    if failing_stage == "registration" and count == 3:
                        raise RuntimeError("injected install fault")
                    return original_hook(parameter, hook)
                def emit(probe, event):
                    if failing_stage == "installed_event" and event["event"] == "dense_sentinel_installed":
                        raise RuntimeError("injected install fault")
                    return original_emit(probe, event)
                with patch.object(torch.Tensor, "register_hook", register), \
                        patch.object(all_layers.AllLayersBatchedTrainingBackwardBoundaryProbe, "_emit", emit), \
                        self.assertRaisesRegex(RuntimeError, "injected install fault"):
                    all_layers.AllLayersBatchedTrainingBackwardBoundaryProbe(
                        model, self.directory / failing_stage, optimizer=optimizer,
                        compute_module=compute, preprocess_module=preprocess, output_module=output)
                self.assertIs(output.HSTUComputeOutputFunction.backward, original_backward)
                self.assertTrue(all(not parameter._backward_hooks for parameter in model.parameters()))
                self.assertTrue(all(not layer._forward_hooks for layer in model.modules()))


if __name__ == "__main__":
    unittest.main()
