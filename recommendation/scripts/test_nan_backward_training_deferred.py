"""CPU behavior and mocked stream-join checks for the deferred scalar trace."""
from contextlib import nullcontext
import gc
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

import nan_backward_training_deferred as deferred
import nan_backward_training_probe as regular
import nan_backward_training_extended as extended
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from test_nan_backward_training_extended import make_full_modules
from test_nan_backward_training_upstream import Stack


class DeferredTraceTest(unittest.TestCase):
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
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        probe = deferred.DeferredAllLayersTrainingBackwardProbe(
            model, self.directory, optimizer=optimizer, compute_module=compute,
            preprocess_module=preprocess, output_module=output)
        self.addCleanup(probe.close)
        return model, probe, state, optimizer

    def execute(self, model):
        x = (torch.arange(12, dtype=torch.float64).reshape(3, 4) / 64).requires_grad_()
        y = model(x)
        y.backward(torch.full_like(y, 0.5))

    def events(self):
        return [json.loads(line) for line in (self.directory / "boundaries.jsonl").read_text().splitlines()]

    def test_queue_retains_only_scalars_and_preserves_enqueue_time_values(self):
        queue = deferred.DeferredScalarQueue(chunk_bytes=4, abs_threshold=10.)
        value = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]
        original = value.clone()
        reference = weakref.ref(value)
        with patch.object(deferred, "_read_endpoint_batch", wraps=deferred._read_endpoint_batch) as reader:
            queued = queue.enqueue({"x": value}, {"phase": "before"})
            self.assertEqual(reader.call_count, 0)
            self.assertFalse(queued["values_resolved_on_host"])
            self.assertNotIn("finite", queued["summaries"][0])
            value.fill_(100.)
            del value
            gc.collect()
            self.assertIsNone(reference())
            observed, = queue.flush()
            self.assertEqual(reader.call_count, 1)
        summary, = observed["summaries"]
        self.assertEqual(summary["max"], original.max().item())
        self.assertFalse(summary["extreme"])
        self.assertEqual(summary["stride"], [4, 2])
        self.assertFalse(queue.pending)
        with self.assertRaises(BoundaryProbeError):
            queue.flush()

    def test_queue_tracks_nan_inf_finite_extreme_and_ignores_integer_values(self):
        queue = deferred.DeferredScalarQueue(chunk_bytes=4, abs_threshold=10.)
        queue.enqueue({"nan": torch.tensor([float("nan")]), "inf": torch.tensor([float("inf")]),
                       "large": torch.tensor([11.]), "integer": torch.tensor([999])}, {"phase": "test"})
        scan, = queue.flush()
        self.assertEqual(scan["flagged_names"], ["nan", "inf", "large"])
        summaries = {entry["name"]: entry for entry in scan["summaries"]}
        self.assertFalse(summaries["integer"]["scanned"])
        self.assertTrue(summaries["large"]["finite"])
        self.assertFalse(summaries["nan"]["finite"])

    def test_mocked_multi_stream_join_occurs_before_the_single_readback(self):
        queue = deferred.DeferredScalarQueue(chunk_bytes=32, abs_threshold=10.)
        order = []
        class Stream:
            def wait_event(self, event):
                order.append(("wait", event.producer))
        class Event:
            def record(self, producer):
                self.producer = producer
                order.append(("record", producer))
        entry = {"name": "x", "scanned": True}
        queue.scans.append({"summaries": [entry]})
        queue.pending['cuda:0'] = [(entry, torch.tensor([-2., 3.], dtype=torch.float64))]
        queue.streams['cuda:0'] = {11: 'stream11', 12: 'stream12'}
        def read(batch):
            order.append(("read", len(batch)))
            return batch.tolist()
        with patch.object(torch.cuda, "current_stream", return_value=Stream()), \
                patch.object(torch.cuda, "Event", Event), patch.object(torch.cuda, "stream", return_value=nullcontext()), \
                patch.object(deferred, "_read_endpoint_batch", side_effect=read):
            queue.flush()
        self.assertEqual(order, [("record", "stream11"), ("wait", "stream11"),
                                 ("record", "stream12"), ("wait", "stream12"), ("read", 1)])
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_backward_has42_queued_endpoints_two_dense_phases_and_one_flush(self):
        model, probe, state, optimizer = self.setup_probe()
        initial_rng = torch.get_rng_state().clone()
        with patch.object(deferred, "_read_endpoint_batch", wraps=deferred._read_endpoint_batch) as reader, \
                patch.object(regular, "monitor_summaries", side_effect=AssertionError("synchronous scan")), \
                patch.object(extended, "monitor_summaries", side_effect=AssertionError("synchronous scan")), \
                patch.object(regular, "_cpu_copy_tree", side_effect=AssertionError("raw capture")), \
                patch.object(extended, "_cpu_copy_tree", side_effect=AssertionError("raw capture")):
            probe.set_attempt("training", 0, 1)
            self.assertEqual(reader.call_count, 0)
            self.execute(model)
            self.assertEqual(reader.call_count, 1)
            probe.sentinel.optimizer_proxy.step()
            self.assertEqual(reader.call_count, 1)
        probe.assert_complete_operation_scans()
        self.assertEqual((state.attention_calls, state.silu_calls), (3, 3))
        events = self.events()
        endpoints = [event for event in events if event["event"] in ("before", "after")]
        self.assertEqual(len(endpoints), 42)
        self.assertTrue(all(not event["values_resolved_on_host"] for event in endpoints))
        self.assertTrue(all(not event["boundary_scan_policy"]["raw_snapshot_present"] for event in endpoints))
        self.assertTrue(all("finite" not in item for event in endpoints for item in event["summaries"]))
        phases = [event["phase"] for event in events if event["event"] == "dense_phase_queued"]
        self.assertEqual(phases, ["before_forward", "after_backward_before_clip"])
        flush, = [event for event in events if event["event"] == "deferred_flush"]
        self.assertEqual(flush["scan_count"], 44)
        self.assertIsNone(flush["first_flagged_enqueue_index"])
        attention = [scan for scan in flush["scans"] if scan.get("operation") == "hstu_attention_bwd"]
        self.assertEqual(len(attention), 6)
        self.assertTrue(all(scan["scalar_arguments"]["N"] == 3 for scan in attention))
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        self.assertFalse(list(self.directory.glob("*.pt")))
        optimizer.zero_grad()
        probe.set_attempt("training", 0, 2)
        self.assertFalse(probe.scalar_queue.flushed)

    def test_attention_fault_is_resolved_after_all_operations_but_before_clipping(self):
        model, probe, state, optimizer = self.setup_probe()
        state.extreme_attention = True
        probe.set_attempt("training", 0, 1)
        with patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=AssertionError("clip reached")), \
                patch.object(optimizer, "step", side_effect=AssertionError("optimizer reached")), \
                self.assertRaises(BoundaryAnomalyError) as raised:
            self.execute(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            probe.sentinel.optimizer_proxy.step()
        self.assertEqual((state.attention_calls, state.silu_calls), (3, 3))
        probe.assert_complete_operation_scans()
        payload = json.loads(raised.exception.dump_path.read_text())
        first = payload["first_flagged_observation"]
        self.assertEqual((first["operation"], first["phase"]), ("hstu_attention_bwd", "after"))
        self.assertEqual(first["flagged_names"], ["dq"])
        self.assertEqual(payload["scan_count"], 44)
        self.assertFalse(payload["raw_snapshot_present"])
        self.assertFalse(payload["boundary_scan_policy"]["pristine_input_replay_claim"])
        self.assertTrue(any(scan["phase"] == "before" and scan["flagged_names"] for scan in payload["scans"]))
        self.assertFalse(list(self.directory.glob("*.pt")))

    def test_accumulated_gradient_fault_is_localized_to_final_dense_phase(self):
        model, probe, _, _ = self.setup_probe()
        handle = model._stu_layers[2]._output_weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("nan")))
        self.addCleanup(handle.remove)
        probe.set_attempt("training", 0, 1)
        with self.assertRaises(BoundaryAnomalyError) as raised:
            self.execute(model)
        payload = json.loads(raised.exception.dump_path.read_text())
        self.assertEqual(payload["first_flagged_observation"]["enqueue_index"], 43)
        self.assertEqual(payload["first_flagged_observation"]["phase"], "after_backward_before_clip")
        self.assertTrue(all(not scan["flagged_names"] for scan in payload["scans"][:-1]))

    def test_incomplete_backward_fails_before_readback_or_optimizer(self):
        model, probe, _, optimizer = self.setup_probe()
        probe.set_attempt("training", 0, 1)
        with patch.object(deferred, "_read_endpoint_batch", side_effect=AssertionError("incomplete flush")), \
                self.assertRaisesRegex(BoundaryProbeError, "Incomplete all-layer"):
            x = torch.ones(3, 4, dtype=torch.float64, requires_grad=True)
            model._stu_layers[2](x).sum().backward()
        with self.assertRaises(BoundaryProbeError):
            probe.sentinel.optimizer_proxy.step()
        probe.close()
        self.assertTrue(all(not parameter._backward_hooks for parameter in model.parameters()))

    def test_interleaved_counter_advance_preserves_regular_and_readwrite_call_identity(self):
        model, probe, _, _ = self.setup_probe()
        probe.set_attempt("training", 0, 1)
        def regular_call(dout, output_weight):
            probe.call += 1  # Another wrapper advances the shared host counter.
            return (dout @ output_weight.T,)
        probe._run_monitor("hstu_output_grad_mm", regular_call, inspect.signature(regular_call), (),
                           {"dout": torch.ones(3, 4), "output_weight": model._stu_layers[2]._output_weight})
        def silu_call(grad_output, self, *, grad_input):
            probe.call += 1
            grad_input.copy_(grad_output)
            return grad_input
        probe._batched_readwrite_monitor(
            "hstu_silu_bwd", silu_call, inspect.signature(silu_call), (),
            {"grad_output": torch.ones(3, 4), "self": torch.ones(3, 4), "grad_input": torch.empty(3, 4)},
            ("grad_input",), {"layer": "model._stu_layers.2", "parameter": "test_parameter"})
        endpoints = [event for event in self.events() if event["event"] in ("before", "after")]
        self.assertEqual([(event["event"], event["call"]) for event in endpoints],
                         [("before", 1), ("after", 1), ("before", 3), ("after", 3)])
        self.assertEqual(probe.call, 4)


if __name__ == "__main__":
    unittest.main()
