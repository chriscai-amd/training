#!/usr/bin/env python3
"""CPU-only contracts for full-shape resident projection stress and failure evidence."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import stress_addmm_backward_boundary as stress


def public_formula(**values):
    dy = values["dz"].sum(0) if values["is_y_1d"] else values["dz"]
    dw = values["x"].T @ values["dz"]
    dx = values["dz"] @ values["w"].T
    return dx, dw, dy


def fixture(*, strided=False, is_y_1d=True):
    if strided:
        xraw = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 64
        wraw = torch.arange(30, dtype=torch.float32).reshape(3, 10) / 128
        zraw = torch.arange(60, dtype=torch.float32).reshape(6, 10) / 1024
        x, w, dz = xraw[:, 1:7:2], wraw[:, 1:9:2], zraw[:, 2:10:2]
    else:
        x = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 64
        w = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 128
        dz = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 1024
    values = {"x": x, "w": w, "dz": dz, "is_y_1d": is_y_1d}
    outputs = tuple(value.clone() for value in public_formula(**values))
    controls = stress.helpers.execution_controls()
    return {"format_version": 1, "operation": stress.OPERATION, "stage": "finite",
            "session": "CPU-test", "attempt": {"step": 101, "repeat": 0, "mode": "current"},
            "args": (), "kwargs": values, "outputs": outputs, "execution_controls": controls,
            "event": {"operation_executed": True, "execution_controls": copy.deepcopy(controls),
                      "input_observation": {"source": "pristine_pre_call_cpu_snapshot",
                                            "operation_started": False,
                                            "source_devices_synchronized_before_copy": True}}}


def raw(tensor):
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,)).clone()


class AddmmStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()
        cls.controls = stress.helpers.execution_controls()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert stress.helpers.execution_controls() == cls.controls
        assert not torch.cuda.is_initialized()

    def prepare(self, payload, zero=False):
        return stress.prepare_capture(payload, zero_oracle=zero, chunk_bytes=1024, max_clone_bytes=1 << 20)

    def run_capture(self, payload, prepared, path, function=public_formula, **kwargs):
        return stress.run_stress(payload, prepared, function=function, failure_path=path,
                                 configuration={"cpu_test": True}, device="cpu", chunk_bytes=1024,
                                 chunk_elements=5, max_resident_bytes=1 << 20,
                                 max_capture_bytes=2 << 20, **kwargs)

    def test_pristine_evidence_and_three_references_required(self):
        cases = []
        p = fixture(); p["input_snapshot_pristine"] = False; cases.append(p)
        p = fixture(); p["outputs"] = None; cases.append(p)
        p = fixture(); p["stage"] = "output"; cases.append(p)
        p = fixture(); p["event"]["input_observation"]["operation_started"] = True; cases.append(p)
        p = fixture(); p["event"]["operation_executed"] = False; cases.append(p)
        p = fixture(); p["kwargs"]["is_y_1d"] = 1; cases.append(p)
        p = fixture(); p["outputs"] = (*p["outputs"][:2], torch.zeros(5)); cases.append(p)
        p = fixture(); p["outputs"] = (p["outputs"][0].double(), *p["outputs"][1:]); cases.append(p)
        p = fixture(); p["kwargs"]["x"] = p["kwargs"]["x"][:1].expand(6, 3); cases.append(p)
        for payload in cases:
            with self.subTest(payload=payload.get("stage")), self.assertRaises(ValueError):
                self.prepare(payload)

    def test_controls_are_complete_and_agree_with_event(self):
        for change in ("missing", "mismatch", "type"):
            payload = fixture()
            if change == "missing":
                del payload["execution_controls"]
            elif change == "mismatch":
                payload["event"]["execution_controls"]["allow_tf32"] = not payload["execution_controls"]["allow_tf32"]
            else:
                payload["execution_controls"]["allow_tf32"] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare(payload)

    def test_all_original_inputs_and_outputs_must_be_finite(self):
        for zero in (False, True):
            for name in ("x", "w", "dz", *stress.NAMES):
                for bad in (float("nan"), float("inf")):
                    payload = fixture()
                    value = payload["kwargs"][name] if name in payload["kwargs"] else payload["outputs"][stress.NAMES.index(name)]
                    value.reshape(-1)[-1] = bad
                    with self.subTest(zero=zero, name=name, bad=bad), self.assertRaises(ValueError):
                        self.prepare(payload, zero)

    def test_zero_transform_preserves_original_xw_and_unused_dz_backing(self):
        payload = fixture(strided=True)
        original = {name: raw(value) for name, value in payload["kwargs"].items() if isinstance(value, torch.Tensor)}
        original_outputs = [raw(value) for value in payload["outputs"]]
        prepared = self.prepare(payload, True)
        transformed = prepared["prepared"]["inputs"]
        for name in ("x", "w", "dz"):
            self.assertTrue(torch.equal(raw(payload["kwargs"][name]), original[name]))
            self.assertNotEqual(transformed[name].untyped_storage()._cdata, payload["kwargs"][name].untyped_storage()._cdata)
            self.assertEqual(transformed[name].stride(), payload["kwargs"][name].stride())
            self.assertEqual(transformed[name].storage_offset(), payload["kwargs"][name].storage_offset())
        for name in ("x", "w"):
            self.assertTrue(torch.equal(raw(transformed[name]), original[name]))
        expected, _ = stress.common.restore_raw_tree(payload["kwargs"], device="cpu", chunk_bytes=13)
        expected["dz"].zero_()
        self.assertTrue(torch.equal(raw(transformed["dz"]), raw(expected["dz"])))
        full = raw(transformed["dz"]).view(torch.float32).reshape(6, 10)
        before = original["dz"].view(torch.float32).reshape(6, 10)
        self.assertTrue(torch.equal(full[:, :2], before[:, :2]))
        self.assertTrue(torch.equal(full[:, 3::2], before[:, 3::2]))
        self.assertEqual(int(torch.count_nonzero(transformed["dz"])), 0)
        for index, value in enumerate(prepared["prepared"]["references"]):
            self.assertEqual(int(torch.count_nonzero(value)), 0)
            self.assertTrue(torch.equal(raw(payload["outputs"][index]), original_outputs[index]))
        self.assertFalse(prepared["original_input_digests"] == prepared["prepared_input_digests"])

    def test_zero_transform_rejects_alias_to_protected_xw_and_autocast_narrowing(self):
        payload = fixture()
        payload["outputs"] = (payload["kwargs"]["x"], *payload["outputs"][1:])
        with self.assertRaisesRegex(ValueError, "protected"):
            self.prepare(payload, True)
        payload = fixture()
        payload["execution_controls"]["autocast"]["cuda"]["enabled"] = True
        payload["event"]["execution_controls"] = copy.deepcopy(payload["execution_controls"])
        with self.assertRaisesRegex(ValueError, "narrowing"):
            self.prepare(payload, True)

    def test_zero_oracle_all_three_tiny_nonzero_nonfinite_and_signed_zero(self):
        references = (torch.zeros(6, 3), torch.zeros(3, 4), torch.zeros(4))
        smallest = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
        for output_index in range(3):
            for bad in (smallest, -1e-25, float("nan"), float("inf")):
                outputs = tuple(value.clone() for value in references)
                outputs[output_index].reshape(-1)[-1] = bad
                checks, classification = stress.compare_outputs(outputs, references, zero_oracle=True,
                                                                 chunk_elements=2, threshold=1e6, atol=1.)
                with self.subTest(index=output_index, bad=bad):
                    self.assertTrue(classification["failed"])
                    self.assertEqual(checks[stress.NAMES[output_index]]["nonzero_elements_including_nonfinite"], 1)
        signed = tuple(torch.full_like(value, -0.) for value in references)
        checks, classification = stress.compare_outputs(signed, references, zero_oracle=True, strict_exact=True)
        self.assertFalse(classification["failed"])
        self.assertTrue(all(item["different_bytes"] for item in checks.values()))

    def test_mutated_resident_zero_reference_cannot_hide_nonzero_output(self):
        values = (torch.ones(6, 3), torch.ones(3, 4), torch.ones(4))
        _, classification = stress.compare_outputs(values, values, zero_oracle=True)
        self.assertTrue(classification["failed"])

    def test_integer_oracle_detects_all_supported_subnormal_bit_patterns(self):
        for dtype, integer in ((torch.bfloat16, torch.int16), (torch.float16, torch.int16),
                               (torch.float32, torch.int32), (torch.float64, torch.int64)):
            references = tuple(torch.zeros(1, dtype=dtype) for _ in stress.NAMES)
            for word in (1, torch.iinfo(integer).min + 1):
                outputs = tuple(torch.tensor([word], dtype=integer).view(dtype) for _ in stress.NAMES)
                checked, result = stress.compare_outputs(outputs, references, zero_oracle=True, chunk_elements=1, atol=1e6)
                with self.subTest(dtype=dtype, word=word):
                    self.assertTrue(result["failed"])
                    self.assertTrue(all(value["nonzero_elements_including_nonfinite"] == 1 for value in checked.values()))

    def test_original_mode_preserves_finite_roundoff_classification(self):
        references = (torch.ones(6, 3), torch.ones(3, 4), torch.ones(4))
        outputs = tuple(value.clone() for value in references)
        outputs[1][-1, -1] += .001
        _, result = stress.compare_outputs(outputs, references, chunk_elements=2)
        self.assertFalse(result["failed"])
        self.assertEqual(result["classifications"]["d_uvqk_weight"], "finite_difference_within_tolerance")
        _, strict = stress.compare_outputs(outputs, references, strict_exact=True)
        self.assertTrue(strict["failed"])
        outputs[2][-1] = 1e10
        _, result = stress.compare_outputs(outputs, references)
        self.assertEqual(result["classifications"]["d_uvqk_bias"], "extreme_finite")

    def test_hundred_calls_reuse_inputs_and_restore_controls(self):
        for zero in (False, True):
            payload = fixture()
            prepared = self.prepare(payload, zero)
            pointers, calls = [], []
            def function(**values):
                calls.append(stress.helpers.execution_controls())
                pointers.append(tuple(values[name].data_ptr() for name in ("x", "w", "dz")))
                return public_formula(**values)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "failure.pt"
                result = self.run_capture(payload, prepared, path, function, repeats=100)
                self.assertFalse(path.exists())
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["exact_iterations"], 100)
            self.assertEqual(len(set(pointers)), 1)
            self.assertTrue(all(control == prepared["controls"] for control in calls))
            self.assertTrue(all(result["final_integrity"].values()))

    def test_each_output_first_failure_retains_original_prepared_current_without_rerun(self):
        for zero in (False, True):
            for index, name in enumerate(stress.NAMES):
                payload = fixture(strided=True)
                prepared = self.prepare(payload, zero)
                calls, returned = [], []
                def function(**values):
                    calls.append(len(calls) + 1)
                    outputs = public_formula(**values)
                    if len(calls) == 3:
                        outputs[index].reshape(-1)[-1] = 1e-30 if zero else 1e30
                        returned.append(tuple(value.clone() for value in outputs))
                    return outputs
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "failure.pt"
                    result = self.run_capture(payload, prepared, path, function, repeats=100)
                    saved = torch.load(path, weights_only=False, map_location="cpu")
                    self.assertEqual(calls, [1, 2, 3])
                    self.assertEqual(saved["iteration"], 3)
                    self.assertEqual(result["status"], "FAIL")
                    self.assertTrue(saved["input_and_reference_integrity"]["inputs_match_prepared_logical_and_raw_bytes"])
                    self.assertEqual(saved["transformation"]["zero_oracle"], zero)
                    self.assertTrue(torch.equal(saved["current_at_failure"]["outputs"][index], returned[0][index]))
                    self.assertTrue(torch.equal(saved["original_capture"]["references"][index], payload["outputs"][index]))
                    self.assertEqual(saved["prepared_pre_call"]["inputs"]["dz"].stride(), payload["kwargs"]["dz"].stride())
                    self.assertTrue(saved["saved_output_classification"]["failed"])
                    self.assertEqual(saved["failure_locations"][name][0]["index"], [size - 1 for size in returned[0][index].shape])
                    self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_coincident_input_mutation_is_preserved_before_attribution(self):
        payload = fixture()
        prepared = self.prepare(payload)
        calls = []
        def function(**values):
            calls.append(1)
            outputs = public_formula(**values)
            values["w"].add_(1)
            outputs[0][0, 0] = float("nan")
            return outputs
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.pt"
            result = self.run_capture(payload, prepared, path, function, repeats=5)
            saved = torch.load(path, weights_only=False, map_location="cpu")
            self.assertEqual(calls, [1])
            self.assertFalse(result["failure_capture"]["inputs_match_prepared_logical_and_raw_bytes"])
            self.assertTrue(torch.equal(saved["current_at_failure"]["inputs"]["w"], payload["kwargs"]["w"] + 1))
            self.assertTrue(torch.equal(saved["prepared_pre_call"]["inputs"]["w"], payload["kwargs"]["w"]))

    def test_final_raw_mutation_audit_retains_last_outputs(self):
        payload = fixture(strided=True)
        prepared = self.prepare(payload)
        calls = []
        def function(**values):
            calls.append(1)
            outputs = public_formula(**values)
            # Change only unused backing bytes, bypassing the logical tensor version.
            storage = values["dz"].untyped_storage()
            raw_view = torch.empty(0, dtype=torch.float32).set_(storage, 0, (60,), (1,))
            raw_view[0] = 123.
            return outputs
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.pt"
            result = self.run_capture(payload, prepared, path, function, repeats=2)
            saved = torch.load(path, weights_only=False, map_location="cpu")
            self.assertEqual(len(calls), 2)
            self.assertEqual(saved["observed_record"]["changed_input_versions"], [])
            self.assertIn("mutation start time unknown", saved["observed_record"]["failure_detection"])
            self.assertEqual(len(saved["current_at_failure"]["outputs"]), 3)
            self.assertFalse(result["failure_capture"]["inputs_match_prepared_logical_and_raw_bytes"])

    def test_is_y_1d_false_output_alias_retained_and_budgeted_once(self):
        payload = fixture(is_y_1d=False)
        prepared = self.prepare(payload, True)
        def function(**values):
            outputs = public_formula(**values)
            outputs[0][0, 0] = 1e-30
            return outputs
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.pt"
            result = self.run_capture(payload, prepared, path, function, repeats=2)
            saved = torch.load(path, weights_only=False, map_location="cpu")
            current = saved["current_at_failure"]
            self.assertEqual(current["inputs"]["dz"].untyped_storage()._cdata, current["outputs"][2].untyped_storage()._cdata)
            self.assertEqual(result["resource_plan"]["expected_complete_failure_storage_bytes"], saved["storage_bytes"])

    def test_budgets_and_paths_reject_before_dispatch(self):
        payload = fixture()
        prepared = self.prepare(payload, True)
        function = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.pt"
            for key in ("max_resident_bytes", "max_capture_bytes"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    stress.run_stress(payload, prepared, function=function, failure_path=path,
                                      configuration={}, device="cpu", **{key: 1})
            path.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                self.run_capture(payload, prepared, path, function)
            self.assertEqual(path.read_bytes(), b"keep")
        function.assert_not_called()

    def test_atomic_save_cleanup_and_race_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.pt"
            def broken(stream):
                stream.write(b"partial")
                raise RuntimeError("write failure")
            with self.assertRaisesRegex(RuntimeError, "write failure"):
                stress.atomic_new_save(path, broken)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            def race(stream):
                stream.write(b"new")
                path.write_bytes(b"other writer")
            with self.assertRaises(FileExistsError):
                stress.atomic_new_save(path, race)
            self.assertEqual(path.read_bytes(), b"other writer")
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_atomic_save_fsyncs_file_and_directory_and_preserves_foreign_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.pt"
            with mock.patch.object(stress.os, "fsync", wraps=os.fsync) as fsync:
                stress.atomic_new_save(path, lambda stream: stream.write(b"complete"))
            self.assertEqual(path.read_bytes(), b"complete")
            self.assertEqual(fsync.call_count, 2)
            path.unlink()
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(b"foreign")
            with self.assertRaises(FileExistsError):
                stress.atomic_new_save(path, lambda stream: stream.write(b"new"))
            self.assertEqual(temporary.read_bytes(), b"foreign")

    def test_arguments_cap_calls_and_reject_source_temp_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            base = [str(folder / "capture.pt"), "--context", str(folder / "context.json"),
                    "--report", str(folder / "report.json"), "--failure-dump", str(folder / "failure.pt")]
            args = stress.arguments(base + ["--zero-oracle", "--expect-rows", "1959558"])
            self.assertEqual(args.repeats, 1000)
            self.assertTrue(args.zero_oracle)
            for count in (0, 1001):
                with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    stress.arguments(base + ["--repeats", str(count)])
            base[0] = str(folder / "failure.pt.tmp")
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                stress.arguments(base)

    def test_environment_mandatory_overrides_follow_context_and_remove_allocator_alias(self):
        context = {"environment": {"AMDGCN_USE_BUFFER_OPS": "1", "TRITON_FULL_AUTOTUNE": None,
                                   "TRITON_ALLOW_PIPELINING": None, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                   "HIPBLASLT_TENSILE_LIBPATH": "/recorded/tensile"}}
        with mock.patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": "incorrect", "TRITON_ALLOW_PIPELINING": "1"}, clear=True):
            report = stress.restore_environment(context)
            for name in ("AMDGCN_USE_BUFFER_OPS", "TRITON_FULL_AUTOTUNE", "TRITON_ALLOW_PIPELINING"):
                self.assertEqual(os.environ[name], "0")
            self.assertNotIn("PYTORCH_ALLOC_CONF", os.environ)
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
            self.assertEqual(report["mandatory_environment_differences"]["TRITON_ALLOW_PIPELINING"], {"captured": None, "effective": "0"})
            self.assertEqual(report["effective_HIPBLASLT_TENSILE_LIBPATH"], "/recorded/tensile")


if __name__ == "__main__":
    unittest.main()
