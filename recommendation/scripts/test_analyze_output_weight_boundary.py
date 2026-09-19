"""CPU-only reference/provenance checks for the actual output-weight GEMM."""

import copy
import math
import unittest

import torch

import analyze_output_weight_boundary as analyzer
import replay_hstu_output_boundary as common
from capture_training_backward import _rng_equal
from nan_replay_state import capture_rng


OPERATION = "hstu_output_weight_mm"


def fixture(y=None, dout=None, observed=None):
    if y is None:
        y = torch.arange(35, dtype=torch.float64).reshape(7, 5) / 8
    if dout is None:
        dout = torch.arange(y.shape[0] * 3, dtype=y.dtype).reshape(y.shape[0], 3) / 16
    if observed is None:
        observed = (y.to(torch.float64).T @ dout.to(torch.float64)).to(y.dtype)
    controls = common.execution_controls()
    return {
        "operation": OPERATION, "stage": "output", "layer": "model._stu_layers.1",
        "args": (), "kwargs": {"y": y, "dout": dout}, "outputs": (observed,),
        "execution_controls": controls,
        "event": {
            "execution_controls": copy.deepcopy(controls),
            "input_observation": {"source": "pristine_pre_call_cpu_snapshot"},
        },
    }


def tensor_metadata(name, value):
    return {"name": name, "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "storage_offset": value.storage_offset(),
            "source_device": "cpu", "requires_grad": value.requires_grad}


class OutputWeightAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.rng = capture_rng()
        self.controls = common.execution_controls()

    def tearDown(self):
        self.assertTrue(_rng_equal(self.rng, capture_rng()))
        self.assertEqual(common.execution_controls(), self.controls)
        self.assertFalse(torch.cuda.is_initialized())

    def test_zero_oracle_finds_wrong_finite_bfloat16_weight_row(self):
        y = torch.zeros((11, 5), dtype=torch.bfloat16)
        dout = torch.arange(33, dtype=torch.bfloat16).reshape(11, 3)
        observed = torch.zeros((5, 3), dtype=torch.bfloat16)
        observed[3, 1] = 1e25
        payload = fixture(y, dout, observed)
        report = analyzer.analyze(payload, reduction_chunk=3, column_chunk=2, chunk_bytes=8192)
        reference = report["reference"]
        self.assertEqual(reference["reference_max_abs_finite"], 0.)
        self.assertEqual(reference["reference_nonfinite"], 0)
        self.assertEqual(reference["elements"], 3)
        self.assertEqual(reference["compared_elements"], 3)
        self.assertEqual(reference["max_abs_error_vs_fp64_finite_pairs"], float(observed[3, 1]))
        self.assertTrue(report["zero_oracle"]["applicable"])
        self.assertEqual(report["zero_oracle"]["observed_nonzero_count"], 1)

    def test_zero_oracle_requires_finite_other_operand(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                y = torch.zeros((5, 2), dtype=torch.float64)
                dout = torch.ones((5, 3), dtype=torch.float64)
                dout[4, 2] = bad
                payload = fixture(y, dout, torch.zeros((2, 3), dtype=torch.float64))
                report = analyzer.analyze(payload, rows="all", max_rows=0,
                                          reduction_chunk=2, chunk_bytes=8192)
                self.assertFalse(report["zero_oracle"]["applicable"])
                self.assertEqual(report["reference"]["reference_nonfinite"], 2)

    def test_reference_includes_all_reduction_chunks_and_preserves_selected_row_order(self):
        y = torch.zeros((17, 3), dtype=torch.bfloat16)
        y[0, 0], y[4, 0], y[8, 0], y[16, 0] = 256., 1., -256., 2.
        y[:, 1] = torch.arange(17, dtype=torch.bfloat16)
        y[:, 2] = torch.arange(17, dtype=torch.bfloat16) - 8
        y[16, 2] = 11.
        dout = torch.tensor([1., 2., 4., 0.], dtype=torch.bfloat16).expand(17, 4)
        rows = torch.tensor([2, 0, 1])
        # Python scalar fsum is independent of the analyzer's tiled matmuls.
        expected = torch.tensor([
            [math.fsum(float(y[n, row]) * float(dout[n, col]) for n in range(17))
             for col in range(4)] for row in rows.tolist()
        ], dtype=torch.float64)
        self.assertEqual(expected[1, 0].item(), 3.)
        for chunk, budget in ((1, 8192), (3, 8192), (17, 8192), (32, 8192), (32, 512)):
            with self.subTest(reduction_chunk=chunk, budget=budget):
                result, metadata = analyzer.reference_rows(
                    {"y": y, "dout": dout}, rows, reduction_chunk=chunk,
                    feature_chunk=1, column_chunk=2, chunk_bytes=budget)
                self.assertEqual(result.dtype, torch.float64)
                self.assertEqual(tuple(result.shape), (3, 4))
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
                self.assertEqual(metadata["reduction_rows"], 17)
                self.assertEqual(metadata["output_columns"], 4)
                self.assertTrue(metadata["all_reduction_rows_included"])
                self.assertTrue(metadata["all_output_columns_included"])
                self.assertEqual(metadata["retained_reference_bytes"], 3 * 4 * 8)
                self.assertLessEqual(metadata["estimated_workspace_bound_bytes"], budget)
                if budget == 512:
                    self.assertLess(metadata["effective_reduction_chunk"], chunk)
        with self.assertRaises(ValueError):
            analyzer.reference_rows({"y": y, "dout": dout}, rows,
                reduction_chunk=3, feature_chunk=1, column_chunk=2, chunk_bytes=1)

    def test_strided_shared_storage_padding_and_negative_view_are_unchanged(self):
        for negative in (False, True):
            with self.subTest(negative=negative):
                backing = torch.full((13, 17), float("nan"), dtype=torch.float64)
                y = backing[1:12, 1:10:3]
                dout = backing[1:12, 11:17:2]
                y.copy_(torch.arange(33, dtype=torch.float64).reshape(11, 3) / 8)
                dout.copy_(torch.arange(33, dtype=torch.float64).reshape(11, 3) / 16)
                if negative:
                    y = torch._neg_view(y)
                payload = fixture(y, dout)
                before = common.input_digests(payload["kwargs"], chunk_bytes=128, threshold=1e6)
                values, provenance, controls = analyzer.bind_inputs(payload)
                self.assertIs(values["y"], y)
                self.assertIs(values["dout"], dout)
                self.assertEqual(values["y"].untyped_storage()._cdata,
                                 values["dout"].untyped_storage()._cdata)
                self.assertEqual(values["y"].is_neg(), negative)
                self.assertTrue(provenance["input_snapshot_pristine"])
                self.assertEqual(controls, payload["execution_controls"])
                report = analyzer.analyze(payload, rows="all", max_rows=0,
                    reduction_chunk=3, feature_chunk=1, column_chunk=2, chunk_bytes=8192)
                self.assertEqual(report["reference"]["compared_elements"], 9)
                self.assertLess(report["reference"]["max_abs_error_vs_fp64_finite_pairs"], 1e-12)
                self.assertEqual(before, common.input_digests(
                    payload["kwargs"], chunk_bytes=128, threshold=1e6))
                self.assertEqual(before["inputs"]["y"]["nonfinite_count"], 0)
                self.assertEqual(len(before["raw_storages"]), 1)
                self.assertTrue(torch.isnan(backing[0]).all())

    def test_postcall_inputs_default_reject_and_explicit_reference_remains_conditional(self):
        payload = fixture()
        payload.update(format="training_backward_monitor_v1", input_snapshot_pristine=False)
        payload["event"].update(input_snapshot_pristine=False,
            input_snapshot_observation={"source": "post_call_fault_cpu_snapshot"})
        payload["event"]["input_observation"] = {"source": "pre_call_device_scalar_reductions"}
        with self.assertRaises(ValueError):
            analyzer.bind_inputs(payload)
        with self.assertRaises(ValueError):
            analyzer.analyze(payload)
        _, provenance, _ = analyzer.bind_inputs(payload, allow_postcall_inputs=True)
        self.assertFalse(provenance["input_snapshot_pristine"])
        self.assertTrue(provenance["conditional_reference"])
        self.assertFalse(provenance["original_call_replay_valid"])
        report = analyzer.analyze(payload, rows="all", max_rows=0,
                                  allow_postcall_inputs=True, chunk_bytes=8192)
        self.assertEqual(report["provenance"], provenance)
        self.assertLess(report["reference"]["max_abs_error_vs_fp64_finite_pairs"], 1e-12)
        self.assertIs(payload["input_snapshot_pristine"], False)
        self.assertIs(payload["event"]["input_snapshot_pristine"], False)

    def test_matching_metadata_accepted_and_layout_contradictions_rejected(self):
        payload = fixture()
        payload["event"]["inputs"] = [tensor_metadata(name, value)
                                      for name, value in payload["kwargs"].items()]
        payload["event"]["outputs"] = [tensor_metadata("d_output_weight", payload["outputs"][0])]
        analyzer.bind_inputs(payload)
        for location, field, replacement in (
            ("inputs", "shape", [999, 5]),
            ("inputs", "stride", [1, 7]),
            ("inputs", "dtype", "torch.bfloat16"),
            ("inputs", "storage_offset", 9),
            ("inputs", "source_device", "cuda:1"),
            ("outputs", "shape", [3, 5]),
            ("outputs", "dtype", "torch.int64"),
            ("outputs", "source_device", "cuda:0"),
        ):
            with self.subTest(location=location, field=field):
                invalid = copy.deepcopy(payload)
                invalid["event"][location][0][field] = replacement
                with self.assertRaises(ValueError):
                    analyzer.bind_inputs(invalid)

    def test_autocast_overflow_disables_zero_oracle_and_missing_device_is_ambiguous(self):
        y = torch.zeros((5, 2), dtype=torch.float32)
        dout = torch.full((5, 3), 1e10, dtype=torch.float32)
        payload = fixture(y, dout, torch.full((2, 3), float("nan"), dtype=torch.float16))
        for controls in (payload["execution_controls"], payload["event"]["execution_controls"]):
            controls["autocast"]["cuda"].update(enabled=True, dtype="torch.float16")
        payload["event"]["inputs"] = [dict(tensor_metadata(name, value), source_device="cuda:0")
                                      for name, value in payload["kwargs"].items()]
        report = analyzer.analyze(payload, rows="all", max_rows=0, chunk_bytes=8192)
        self.assertTrue(report["zero_oracle"]["saved_inputs_finite"])
        self.assertFalse(report["zero_oracle"]["effective_input_cast_finiteness_certified"])
        self.assertFalse(report["zero_oracle"]["applicable"])
        self.assertEqual(report["reference"]["reference_max_abs_finite"], 0.)
        # The source metadata rejects different GPU indices even with no GPU present.
        invalid = copy.deepcopy(payload)
        invalid["event"]["inputs"][1]["source_device"] = "cuda:1"
        with self.assertRaises(ValueError):
            analyzer.bind_inputs(invalid)
        payload.update(stage="input", outputs=None)
        payload["event"].pop("inputs")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            analyzer.bind_inputs(payload)

    def test_controls_missing_conflicting_or_malformed_are_rejected(self):
        for problem in ("top_missing", "event_missing", "disagree", "bad_bool", "bad_precision", "bad_autocast"):
            with self.subTest(problem=problem):
                payload = fixture()
                if problem == "top_missing":
                    payload.pop("execution_controls")
                elif problem == "event_missing":
                    payload["event"].pop("execution_controls")
                elif problem == "disagree":
                    payload["event"]["execution_controls"]["allow_tf32"] = not payload["execution_controls"]["allow_tf32"]
                else:
                    for controls in (payload["execution_controls"], payload["event"]["execution_controls"]):
                        if problem == "bad_bool":
                            controls["allow_tf32"] = "false"
                        elif problem == "bad_precision":
                            controls["float32_matmul_precision"] = "unknown"
                        else:
                            controls["autocast"]["cuda"]["enabled"] = "false"
                with self.assertRaises(ValueError):
                    analyzer.bind_inputs(payload)

    def test_invalid_operation_arguments_outputs_and_provenance_are_rejected(self):
        for problem in ("operation", "extra_input", "missing_input", "bad_shape", "integer_input",
                        "wrong_output_shape", "integer_output", "conflicting_provenance"):
            with self.subTest(problem=problem):
                payload = fixture()
                if problem == "operation":
                    payload["operation"] = "hstu_output_grad_mm"
                elif problem == "extra_input":
                    payload["kwargs"]["output_weight"] = torch.ones((5, 3))
                elif problem == "missing_input":
                    payload["kwargs"].pop("dout")
                elif problem == "bad_shape":
                    payload["kwargs"]["dout"] = torch.ones((8, 3), dtype=torch.float64)
                elif problem == "integer_input":
                    payload["kwargs"]["y"] = payload["kwargs"]["y"].to(torch.int64)
                elif problem == "wrong_output_shape":
                    payload["outputs"] = (torch.ones((3, 5), dtype=torch.float64),)
                elif problem == "integer_output":
                    payload["outputs"] = (payload["outputs"][0].to(torch.int64),)
                else:
                    payload["input_snapshot_pristine"] = True
                    payload["event"]["input_snapshot_pristine"] = False
                with self.assertRaises((ValueError, TypeError)):
                    analyzer.bind_inputs(payload, allow_postcall_inputs=True)

    def test_input_trigger_has_reference_without_observed_comparison(self):
        payload = fixture()
        payload.update(stage="input", outputs=None)
        report = analyzer.analyze(payload, rows="all", max_rows=0, chunk_bytes=8192)
        self.assertEqual(report["reference"]["elements"], 15)
        self.assertEqual(report["reference"]["compared_elements"], 0)


if __name__ == "__main__":
    unittest.main()
