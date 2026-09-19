"""CPU regressions for batched monitor transfers and output-only capture.

The real output autograd class is exercised with the numerical helper fixtures
from test_nan_backward_output. No Triton import or GPU execution is needed.
"""

import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import nan_backward_training_batched as batched
import nan_backward_training_probe as training
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from test_nan_backward_output import Model, make_modules


class BatchedSummariesTest(unittest.TestCase):
    def scan(self, value, *, chunk_bytes=32, abs_threshold=1e20):
        return batched.batched_monitor_summaries(
            value, chunk_bytes=chunk_bytes, abs_threshold=abs_threshold)

    def assert_matches_original(self, value, **options):
        parameters = {"chunk_bytes": 32, "abs_threshold": 1e20, **options}
        expected = training.monitor_summaries(value, **parameters)
        actual = batched.batched_monitor_summaries(value, **parameters)
        self.assertEqual(actual, expected)
        json.dumps(actual, allow_nan=False)
        return actual

    def test_all_tensor_chunks_finish_before_one_float64_endpoint_batch(self):
        tree = {
            "half": torch.arange(13, dtype=torch.float16),
            "nested": [torch.arange(24, dtype=torch.float64).reshape(4, 6).t(),
                       torch.tensor([-256., 3.], dtype=torch.bfloat16)],
            "single": torch.tensor(7., dtype=torch.float32),
            "empty": torch.empty(0),
            "integer": torch.tensor([9], dtype=torch.int64),
        }
        expected = training.monitor_summaries(tree, chunk_bytes=16, abs_threshold=1e20)
        order, chunk_sizes, endpoint_shapes = [], [], []
        original_reduction = torch.aminmax
        original_reader = batched._read_endpoint_batch

        def reduce(chunk):
            order.append("reduce")
            chunk_sizes.append(chunk.numel() * chunk.element_size())
            return original_reduction(chunk)

        def read(endpoints):
            order.append("read")
            self.assertEqual(endpoints.dtype, torch.float64)
            self.assertEqual(endpoints.device.type, "cpu")
            endpoint_shapes.append(tuple(endpoints.shape))
            return original_reader(endpoints)

        with patch.object(batched, "_synchronize", side_effect=lambda value: order.append("sync")) as sync:
            with patch.object(torch, "aminmax", side_effect=reduce):
                with patch.object(batched, "_read_endpoint_batch", side_effect=read) as reader:
                    result = self.scan(tree, chunk_bytes=16)
        self.assertEqual(result, expected)
        self.assertEqual(sync.call_count, 1)
        self.assertIs(sync.call_args.args[0], tree)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(endpoint_shapes, [(4, 2)])
        self.assertGreater(len(chunk_sizes), 4)
        self.assertLessEqual(max(chunk_sizes), 16)
        self.assertEqual(order, ["sync"] + ["reduce"] * len(chunk_sizes) + ["read"])

    def test_mixed_dtypes_compare_threshold_in_float64_and_strictly(self):
        values = {str(dtype): torch.tensor([-256., 255.], dtype=dtype)
                  for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
        for threshold, expected_extreme in ((255.5, True), (256., False), (256.5, False)):
            with self.subTest(threshold=threshold):
                result = self.assert_matches_original(values, abs_threshold=threshold)
                self.assertTrue(all(item["extreme"] == expected_extreme for item in result))
                self.assertTrue(all(item["max_abs_finite"] == 256. for item in result))

    def test_nonfinite_endpoints_are_json_safe_and_match_unbatched(self):
        values = {"nan": torch.tensor([1., float("nan"), 1e25]),
                  "positive": torch.tensor([float("inf"), 1.]),
                  "negative": torch.tensor([2., -float("inf")]),
                  "both": torch.tensor([-float("inf"), float("inf")])}
        result = self.assert_matches_original(values, chunk_bytes=4)
        self.assertTrue(all(not item["finite"] for item in result))
        self.assertTrue(all(not item["extreme"] for item in result))
        self.assertTrue(all(item["max_abs_finite"] is None for item in result))

    def test_float64_huge_values_do_not_overflow_during_batch_assembly(self):
        values = {"double": torch.tensor([-1e300, 1e299], dtype=torch.float64),
                  "single": torch.tensor([1e30], dtype=torch.float32),
                  "small": torch.tensor([1.], dtype=torch.float16)}
        result = {item["name"]: item for item in self.assert_matches_original(values)}
        self.assertTrue(result["double"]["finite"])
        self.assertEqual(result["double"]["max_abs_finite"], 1e300)
        self.assertTrue(result["double"]["extreme"])
        self.assertFalse(result["small"]["extreme"])

    def test_only_logical_elements_of_offset_strided_and_broadcast_views_scanned(self):
        backing = torch.full((6, 12), float("nan"), dtype=torch.float64)
        view = backing[1:6:2, 2:11:3]
        view.copy_(torch.arange(view.numel(), dtype=view.dtype).reshape_as(view))
        view[-1, -1] = -1e25
        values = {"offset": view, "transposed": view.t(),
                  "broadcast": torch.tensor(2.).expand(4, 9),
                  "scalar": torch.tensor(-3., dtype=torch.float64)}
        result = {item["name"]: item for item in self.assert_matches_original(values, chunk_bytes=8)}
        self.assertTrue(all(item["finite"] for item in result.values()))
        self.assertEqual(result["offset"]["stride"], list(view.stride()))
        self.assertEqual(result["offset"]["max_abs_finite"], 1e25)
        self.assertEqual(result["broadcast"]["max_abs_finite"], 2.)

    def test_empty_and_metadata_only_tree_does_not_read_an_endpoint_batch(self):
        values = {"empty": torch.empty(0, dtype=torch.float64),
                  "integers": torch.tensor([2**62], dtype=torch.int64),
                  "mask": torch.tensor([False, True]), "none": None, "seed": 23}
        expected = training.monitor_summaries(values, chunk_bytes=8, abs_threshold=1e20)
        with patch.object(batched, "_read_endpoint_batch", side_effect=AssertionError("unexpected read")):
            with patch.object(batched, "_synchronize") as sync:
                result = self.scan(values, chunk_bytes=8)
        self.assertEqual(result, expected)
        sync.assert_called_once_with(values)
        self.assertTrue(all(item["max_abs_finite"] is None for item in result))

    def test_invalid_options_and_tensor_kinds_fail_before_synchronization(self):
        invalid_options = ({"chunk_bytes": 0}, {"chunk_bytes": -1},
                           *({"abs_threshold": value} for value in
                             (0, -1, float("nan"), float("inf"))))
        with patch.object(batched, "_synchronize", side_effect=AssertionError("unexpected sync")):
            for options in invalid_options:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.scan(torch.ones(2), **options)
            unsupported = (torch.tensor([1j]), torch.ones(2).to_sparse(),
                           torch.quantize_per_tensor(torch.ones(2), 0.5, 0, torch.qint8),
                           torch.empty(2, device="meta"))
            for tensor in unsupported:
                with self.subTest(dtype=tensor.dtype, device=tensor.device, layout=tensor.layout):
                    with self.assertRaises(BoundaryProbeError):
                        self.scan(tensor)

    def test_scanning_preserves_autocast_grad_rng_and_input_versions(self):
        tensor = torch.tensor([1., -2.], requires_grad=True)
        snapshot, version, rng = tensor.detach().clone(), tensor._version, torch.get_rng_state()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assert_matches_original(tensor, chunk_bytes=4)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(torch.equal(tensor, snapshot))
        self.assertEqual(tensor._version, version)
        self.assertIsNone(tensor.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(torch.cuda.is_initialized())


class BatchedProbeTest(unittest.TestCase):
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

    def setup_probe(self, *, directory=None, cls=None, **options):
        compute, preprocess, output = make_modules()
        model = Model(compute)
        cls = cls or batched.BatchedTrainingBackwardBoundaryProbe
        probe = cls(model, directory or self.directory,
                    compute_module=compute, preprocess_module=preprocess,
                    output_module=output, **options)
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

    def test_healthy_output_stages_use_four_batches_without_full_storage_copy(self):
        model, probe, output = self.setup_probe()
        original_reader = batched._read_endpoint_batch
        with patch.object(training, "monitor_summaries", side_effect=AssertionError("old scanner used")):
            with patch.object(training, "_cpu_copy_tree", side_effect=AssertionError("unexpected copy")):
                with patch.object(batched, "_read_endpoint_batch", wraps=original_reader) as reader:
                    self.execute(model)
        self.assertEqual(reader.call_count, 4)
        self.assertEqual(probe.mode, "monitor_on_anomaly")
        self.assertEqual(tuple(probe.selected_operations), training.DEFAULT_OPERATIONS)
        self.assertEqual(output.torch._original.mm_calls, 2)
        self.assertEqual(len(output.norm_calls), 1)
        self.assertFalse(self.dumps())
        after = [event for event in self.events() if event["event"] == "after"]
        self.assertEqual([event["operation"] for event in after], list(training.DEFAULT_OPERATIONS))
        self.assertTrue(all(event["attempt"]["step"] == 17 for event in after))
        self.assertFalse(torch.cuda.is_initialized())

    def test_input_anomaly_stops_before_execution_and_retains_pristine_snapshot(self):
        model, probe, output = self.setup_probe()
        with self.assertRaises(BoundaryAnomalyError) as caught:
            self.execute(model, extreme_dout=True)
        dump, = self.dumps()
        self.assertTrue(dump["input_snapshot_pristine"])
        self.assertEqual(dump["event"]["boundary_scan_policy"], batched.BATCHED_POLICY)
        self.assertFalse(dump["event"]["boundary_scan_policy"]["output_fault_inputs_pristine"])
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

    def test_output_anomaly_preserves_precheck_and_marks_mutated_inputs_nonpristine(self):
        model, probe, output = self.setup_probe()
        output.extreme_norm = output.mutate_norm_inputs = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertFalse(dump["input_snapshot_pristine"])
        self.assertEqual(dump["event"]["boundary_scan_policy"], batched.BATCHED_POLICY)
        self.assertFalse(dump["event"]["boundary_scan_policy"]["output_fault_inputs_pristine"])
        self.assertEqual(dump["stage"], "output")
        self.assertEqual(dump["event"]["extreme_outputs"], ["d_attn"])
        before = {item["name"]: item for item in dump["event"]["input_device_monitor"]}
        copied = {item["name"]: item for item in dump["event"]["inputs"]}
        self.assertLess(before["dy"]["max_abs_finite"], 42.)
        self.assertEqual(copied["dy"]["max_abs_finite"], 42.)
        self.assertTrue(torch.equal(dump["kwargs"]["dy"], torch.full((3, 12), 42., dtype=torch.float64)))
        self.assertTrue(torch.equal(dump["kwargs"]["random_mask"], output.norm_calls[0]["random_mask"]))
        self.assertFalse(dump["event"]["input_snapshot_observation"]["input_mutation_during_operation_excluded"])
        self.assertIn("may have changed", dump["note"])
        self.assertIsNone(probe._local.output_backward)
        self.assertEqual(output.torch._original.mm_calls, 1)

    def test_actual_gemm_anomaly_stops_before_norm_and_saves_the_same_dy(self):
        model, _, output = self.setup_probe()
        output.torch._original.extreme_first_mm = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertEqual(dump["operation"], "hstu_output_grad_mm")
        self.assertEqual(dump["stage"], "output")
        self.assertEqual(dump["event"]["extreme_outputs"], ["dy"])
        self.assertEqual(dump["outputs"][0][0, 1].item(), 1e25)
        self.assertFalse(dump["input_snapshot_pristine"])
        self.assertEqual(output.torch._original.mm_calls, 1)
        self.assertFalse(output.norm_calls)

    def test_simultaneous_probes_do_not_change_original_scanner_or_cross_route(self):
        original_scanner = training.monitor_summaries
        model, _, _ = self.setup_probe()
        ordinary_model, _, _ = self.setup_probe(
            directory=self.directory / "ordinary", cls=training.TrainingBackwardBoundaryProbe,
            mode="monitor_on_anomaly")
        self.assertIs(training.monitor_summaries, original_scanner)
        with patch.object(training, "monitor_summaries", wraps=original_scanner) as old_scanner:
            with patch.object(batched, "_read_endpoint_batch", wraps=batched._read_endpoint_batch) as reader:
                self.execute(model)
                self.assertEqual(old_scanner.call_count, 0)
                self.assertEqual(reader.call_count, 4)
                self.execute(ordinary_model)
                self.assertEqual(old_scanner.call_count, 4)
                self.assertEqual(reader.call_count, 4)
        self.assertIs(training.monitor_summaries, original_scanner)

    def test_close_restores_exact_wrappers_and_descriptors_after_success_or_fault(self):
        for fault in (False, True):
            with self.subTest(fault=fault):
                compute, preprocess, output = make_modules()
                model = Model(compute)
                targets = [(compute, "triton_hstu_preprocess_and_attention"),
                           (compute, "triton_hstu_compute_output"),
                           (preprocess, "triton_addmm_bwd"),
                           (preprocess, "triton_weighted_layer_norm_bwd"),
                           (output, "torch"), (output, "triton_layer_norm_mul_dropout_bwd"),
                           (output.HSTUComputeOutputFunction, "backward")]
                originals = [inspect.getattr_static(module, name) for module, name in targets]
                probe = batched.BatchedTrainingBackwardBoundaryProbe(
                    model, self.directory / str(fault), compute_module=compute,
                    preprocess_module=preprocess, output_module=output)
                self.addCleanup(probe.close)
                probe.set_attempt("training", 0, 17)
                if fault:
                    with self.assertRaises(BoundaryAnomalyError):
                        self.execute(model, extreme_dout=True)
                else:
                    self.execute(model)
                probe.close()
                probe.close()
                for (module, name), original in zip(targets, originals):
                    self.assertIs(inspect.getattr_static(module, name), original)
                for layer in model._stu_layers:
                    self.assertFalse(layer._forward_pre_hooks)
                    self.assertFalse(layer._forward_hooks)

    def test_unselected_layer_passes_through_without_endpoint_readback(self):
        model, _, output = self.setup_probe()
        with patch.object(batched, "_read_endpoint_batch", side_effect=AssertionError("unexpected scan")):
            self.execute(model, layer=0)
        self.assertEqual(output.torch._original.mm_calls, 2)
        self.assertEqual(len(output.norm_calls), 1)
        skipped = [event for event in self.events() if event["event"] == "skipped"]
        self.assertEqual([event["reason"] for event in skipped], ["target_filter", "target_filter"])

    def test_unsupported_configuration_fails_before_wrapper_installation(self):
        invalid = ({"mode": "pristine"}, {"mode": "unknown"},
                   {"selected_operations": ("triton_addmm_bwd",)},
                   {"selected_operations": ()},
                   {"selected_operations": "hstu_output_grad_mm"},
                   {"selected_operations": ("hstu_output_grad_mm",) * 2})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                batched.BatchedTrainingBackwardBoundaryProbe(None, self.directory, **options)
        with patch.dict(os.environ, {"NAN_BACKWARD_SAVE_ALL": "1"}):
            with self.assertRaises(ValueError):
                batched.BatchedTrainingBackwardBoundaryProbe(None, self.directory)


if __name__ == "__main__":
    unittest.main()
