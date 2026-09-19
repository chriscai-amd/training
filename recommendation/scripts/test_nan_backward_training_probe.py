"""CPU tests for opt-in training monitor provenance and exact call semantics."""

import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import nan_backward_training_probe as training
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from test_nan_backward_output import Model, make_modules


class MonitorReductionsTest(unittest.TestCase):
    def test_strided_nested_empty_and_metadata_scan_all_logical_elements(self):
        backing = torch.arange(96, dtype=torch.float64).reshape(8, 12)
        value = backing[:, 1:10:3]
        value[-1, -1] = -1e25
        tree = {"nested": [value, (torch.empty(0), None)],
                "expanded": torch.tensor(2.).expand(3, 7),
                "mask": torch.tensor([255], dtype=torch.uint8), "seed": 7}
        observed_sizes = []
        original = torch.aminmax

        def bounded(tensor):
            observed_sizes.append(tensor.numel() * tensor.element_size())
            return original(tensor)

        with patch.object(torch, "aminmax", side_effect=bounded):
            result = {item["name"]: item for item in training.monitor_summaries(
                tree, chunk_bytes=16, abs_threshold=1e20)}
        self.assertTrue(observed_sizes)
        self.assertLessEqual(max(observed_sizes), 16)
        self.assertTrue(result["nested/0"]["extreme"])
        self.assertEqual(result["nested/0"]["max_abs_finite"], 1e25)
        self.assertEqual(result["nested/0"]["stride"], list(value.stride()))
        self.assertTrue(result["nested/1/0"]["finite"])
        self.assertIsNone(result["nested/1/0"]["max_abs_finite"])
        self.assertEqual(result["expanded"]["max_abs_finite"], 2.)
        self.assertFalse(result["mask"]["scanned"])

    def test_nonfinite_and_bfloat_threshold_not_rounded(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            result = training.monitor_summaries(
                torch.tensor([1., bad, 2.]), chunk_bytes=4, abs_threshold=1e20)[0]
            self.assertFalse(result["finite"])
            json.dumps(result, allow_nan=False)
        result = training.monitor_summaries(
            torch.tensor([256.], dtype=torch.bfloat16), chunk_bytes=2,
            abs_threshold=255.5)[0]
        self.assertTrue(result["extreme"])
        self.assertEqual(result["max_abs_finite"], 256.)

    def test_autocast_and_grad_state_unchanged_and_complex_rejected(self):
        tensor = torch.tensor([1., -2.], requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            result = training.monitor_summaries(tensor, chunk_bytes=4, abs_threshold=1e20)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(result[0]["dtype"], "torch.float32")
        self.assertIsNone(tensor.grad)
        with self.assertRaisesRegex(BoundaryProbeError, "complex"):
            training.monitor_summaries(torch.tensor([1j]), chunk_bytes=8,
                                        abs_threshold=1e20)


class TrainingProbeTest(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "NAN_BACKWARD_TARGET": "_stu_layers.1.", "NAN_BACKWARD_SAVE_ALL": "0",
            "NAN_BACKWARD_ABS_THRESHOLD": "1e20", "NAN_BACKWARD_CHUNK_MIB": "1",
            "NAN_BACKWARD_MAX_CAPTURE_GIB": "32",
        })
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def setup_probe(self, *, mode="monitor_on_anomaly", selected_operations=None,
                    preprocess_function=None):
        compute, preprocess, output = make_modules()
        if preprocess_function is not None:
            preprocess.triton_addmm_bwd = preprocess_function
        model = Model(compute)
        probe = training.TrainingBackwardBoundaryProbe(
            model, self.directory, mode=mode, selected_operations=selected_operations,
            compute_module=compute, preprocess_module=preprocess, output_module=output)
        self.addCleanup(probe.close)
        probe.set_attempt("training", 0, 17)
        return model, probe, output

    def execute(self, model, *, layer=1, extreme_dout=False):
        x = (torch.arange(12, dtype=torch.float64).reshape(3, 4) / 16).requires_grad_()
        result = model(x, layer=layer)
        result.backward(torch.tensor(1e25 if extreme_dout else 0.5,
                                     dtype=x.dtype).expand_as(result))
        return x

    def dumps(self):
        return [torch.load(path, map_location="cpu", weights_only=False)
                for path in sorted(self.directory.glob("*.pt"))]

    def events(self):
        return [json.loads(line) for line in
                (self.directory / "boundaries.jsonl").read_text().splitlines()]

    def test_healthy_monitor_never_copies_full_storage_and_preserves_rng(self):
        model, _, output = self.setup_probe()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        with patch.object(training, "_cpu_copy_tree", side_effect=AssertionError("unexpected copy")):
            self.execute(model)
        self.assertEqual(output.torch._original.mm_calls, 2)
        self.assertEqual(len(output.norm_calls), 1)
        self.assertFalse(self.dumps())
        self.assertEqual(random.getstate(), python_state)
        current = np.random.get_state()
        self.assertEqual(current[0], numpy_state[0])
        self.assertTrue(np.array_equal(current[1], numpy_state[1]))
        self.assertEqual(current[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        after = [event for event in self.events() if event["event"] == "after"]
        self.assertEqual(len(after), 2)
        self.assertTrue(all(event["attempt"]["step"] == 17 for event in after))
        self.assertTrue(all(event["probe_mode"] == "monitor_on_anomaly" for event in after))
        self.assertFalse(torch.cuda.is_initialized())

    def test_bad_input_snapshot_is_pristine_and_operation_never_executes(self):
        model, probe, output = self.setup_probe()
        with self.assertRaises(BoundaryAnomalyError) as caught:
            self.execute(model, extreme_dout=True)
        dump, = self.dumps()
        self.assertEqual(dump["format"], training.MONITOR_FORMAT)
        self.assertTrue(dump["input_snapshot_pristine"])
        self.assertEqual(dump["stage"], "input")
        self.assertIsNone(dump["outputs"])
        self.assertFalse(dump["event"]["operation_executed"])
        self.assertEqual(dump["kwargs"]["dout"].stride(), (0, 0))
        self.assertEqual(dump["event"]["extreme_inputs"], ["dout"])
        self.assertEqual(output.torch._original.mm_calls, 0)
        self.assertFalse(output.norm_calls)
        self.assertTrue(caught.exception.dump_path.is_file())
        self.assertIsNone(probe._local.output_backward)
        with self.assertRaises(BoundaryProbeError):
            probe.set_attempt("training", 0, 18)

    def test_output_fault_keeps_precheck_but_does_not_claim_pristine_inputs(self):
        model, _, output = self.setup_probe()
        output.extreme_norm = output.mutate_norm_inputs = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertFalse(dump["input_snapshot_pristine"])
        self.assertEqual(dump["stage"], "output")
        self.assertEqual(dump["event"]["extreme_outputs"], ["d_attn"])
        self.assertEqual(len(dump["outputs"]), 5)
        self.assertIsNone(dump["outputs"][4])
        self.assertTrue(torch.equal(dump["kwargs"]["dy"], torch.full((3, 12), 42., dtype=torch.float64)))
        self.assertTrue(torch.equal(dump["kwargs"]["random_mask"],
                                    output.norm_calls[0]["random_mask"]))
        before = {item["name"]: item for item in dump["event"]["input_device_monitor"]}
        after = {item["name"]: item for item in dump["event"]["inputs"]}
        self.assertLess(before["dy"]["max_abs_finite"], 42.)
        self.assertEqual(after["dy"]["max_abs_finite"], 42.)
        self.assertIn("may have changed", dump["note"])
        self.assertTrue(dump["event"]["device_trigger_persists_in_cpu_snapshot"])
        self.assertEqual(output.torch._original.mm_calls, 1)

    def test_gpu_trigger_still_stops_when_later_cpu_snapshot_is_clean(self):
        model, _, output = self.setup_probe()
        output.torch._original.extreme_first_mm = True
        original = training._cpu_copy_tree

        def changed_before_copy(tree, chunk_bytes, max_bytes):
            tree["outputs"][0].zero_()
            return original(tree, chunk_bytes, max_bytes)

        with patch.object(training, "_cpu_copy_tree", side_effect=changed_before_copy):
            with self.assertRaises(BoundaryAnomalyError):
                self.execute(model)
        dump, = self.dumps()
        self.assertFalse(dump["event"]["device_trigger_persists_in_cpu_snapshot"])
        self.assertEqual(dump["event"]["extreme_outputs"], ["dy"])
        self.assertEqual(dump["event"]["snapshot_extreme_outputs"], [])
        self.assertEqual(output.torch._original.mm_calls, 1)
        self.assertFalse(output.norm_calls)

    def test_operation_and_layer_filters_do_not_scan_unselected_calls(self):
        model, probe, output = self.setup_probe(selected_operations=(training.DEFAULT_OPERATIONS[1],))
        seen = []
        original = training.monitor_summaries

        def observed(value, **kwargs):
            seen.append(tuple(value))
            return original(value, **kwargs)

        with patch.object(training, "monitor_summaries", side_effect=observed):
            self.execute(model)
            probe.set_attempt("training", 0, 18)
            self.execute(model, layer=0)
        self.assertEqual(len(seen), 2)
        self.assertIn("dy", seen[0])
        self.assertEqual(output.torch._original.mm_calls, 4)
        skipped = [event for event in self.events() if event["event"] == "skipped"]
        self.assertEqual([event["reason"] for event in skipped],
                         ["operation_filter", "operation_filter", "target_filter"])

    def test_pristine_mode_delegates_existing_capture_with_original_input_bytes(self):
        model, _, output = self.setup_probe(mode="pristine")
        output.extreme_norm = output.mutate_norm_inputs = True
        with patch.object(training, "monitor_summaries", side_effect=AssertionError("unexpected monitor")):
            with self.assertRaises(BoundaryAnomalyError):
                self.execute(model)
        dump, = self.dumps()
        self.assertNotEqual(dump.get("format"), training.MONITOR_FORMAT)
        self.assertTrue(torch.equal(dump["kwargs"]["dy"], output.norm_calls[0]["before_dy"]))
        self.assertTrue(torch.equal(dump["kwargs"]["random_mask"], output.norm_calls[0]["before_mask"]))
        self.assertEqual(dump["event"]["input_observation"]["source"], "pristine_pre_call_cpu_snapshot")

    def test_fault_copy_preserves_cross_boundary_aliases_and_union_storage_limit(self):
        def mutate_and_alias(x, w, dz, is_y_1d):
            x[0, 0] = 1e25
            return x, dz, dz.sum(dim=0)

        model, probe, _ = self.setup_probe(selected_operations=("triton_addmm_bwd",),
                                           preprocess_function=mutate_and_alias)
        x, dz = torch.ones(2, 4), torch.ones(2, 4)
        weight = model._stu_layers[1]._uvqk_weight
        # 32-byte x + 32-byte dz + 64-byte w + 16-byte sum. Counting
        # inputs and outputs independently would incorrectly exceed this cap.
        probe.max_bytes = 144
        with self.assertRaises(BoundaryAnomalyError):
            probe.preprocess.triton_addmm_bwd(x, weight, dz, True)
        dump, = self.dumps()
        saved_x = dump["args"][0]
        self.assertEqual(saved_x.untyped_storage()._cdata, dump["outputs"][0].untyped_storage()._cdata)
        self.assertEqual(dump["event"]["capture_storage_bytes"], 144)
        self.assertGreater(dump["event"]["input_bytes"] + dump["event"]["output_bytes"], 144)

    def test_fault_capture_limit_is_hard_and_does_not_write_partial_dump(self):
        model, probe, output = self.setup_probe()
        probe.max_bytes = 1
        with self.assertRaisesRegex(BoundaryProbeError, "exceeds remaining capture limit"):
            self.execute(model, extreme_dout=True)
        self.assertFalse(self.dumps())
        self.assertEqual(output.torch._original.mm_calls, 0)
        self.assertIsNone(probe._local.output_backward)
        with self.assertRaises(BoundaryProbeError):
            probe.set_attempt("training", 0, 18)

    def test_invalid_options_fail_before_installation(self):
        for options in ({"mode": "unknown"}, {"selected_operations": ()},
                        {"selected_operations": "hstu_output_grad_mm"},
                        {"selected_operations": ("wrong",)},
                        {"selected_operations": ("hstu_output_grad_mm",) * 2}):
            with self.assertRaises(ValueError):
                training.TrainingBackwardBoundaryProbe(None, self.directory, **options)
        with patch.dict(os.environ, {"NAN_BACKWARD_SAVE_ALL": "1"}):
            with self.assertRaisesRegex(ValueError, "SAVE_ALL=0"):
                training.TrainingBackwardBoundaryProbe(None, self.directory,
                                                        mode="monitor_on_anomaly")


if __name__ == "__main__":
    unittest.main()
