#!/usr/bin/env python3
"""CPU mocks for stage stress checking and preserving the actual failing call."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import stress_hstu_output_boundary as stress


class StressStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()

    def test_full_strided_checks_and_classifications(self):
        expected = torch.arange(48, dtype=torch.float32).reshape(6, 8)[:, 1::2] + 1
        actual = expected.clone()
        actual[0, 0] += .001
        checks = stress.compare_device_outputs((actual,), (expected,), ("dy",), chunk_elements=5)
        self.assertEqual(checks["dy"]["different_numeric_elements"], 1)
        self.assertGreater(checks["dy"]["different_bytes"], 0)
        self.assertEqual(stress.classify(checks)["classifications"]["dy"], "finite_difference_within_tolerance")
        self.assertFalse(stress.classify(checks)["failed"])
        self.assertTrue(stress.classify(checks, strict_exact=True)["failed"])
        actual[-1, -1] += 10
        checks = stress.compare_device_outputs((actual,), (expected,), ("dy",), chunk_elements=5)
        self.assertEqual(checks["dy"]["outside_tolerance_elements"], 1)
        self.assertEqual(stress.classify(checks)["classifications"]["dy"], "finite_mismatch_outside_tolerance")
        actual[1, 1] = 1e30
        checks = stress.compare_device_outputs((actual,), (expected,), ("dy",), chunk_elements=5)
        self.assertEqual(checks["dy"]["extreme_elements"], 1)
        self.assertEqual(stress.classify(checks)["classifications"]["dy"], "extreme_finite")
        actual[2, 1] = float("nan")
        checks = stress.compare_device_outputs((actual,), (expected,), ("dy",), chunk_elements=5)
        self.assertEqual(checks["dy"]["nonfinite_elements"], 1)
        self.assertEqual(stress.classify(checks)["classifications"]["dy"], "nonfinite")

    def test_exact_bytes_detect_signed_zero(self):
        actual, reference = torch.tensor([-0.]), torch.tensor([0.])
        checks = stress.compare_device_outputs((actual, None), (reference, None), ("value", "optional"))
        self.assertEqual(checks["value"]["different_numeric_elements"], 0)
        self.assertGreater(checks["value"]["different_bytes"], 0)
        self.assertIsNone(checks["optional"])
        self.assertTrue(stress.classify(checks, strict_exact=True)["failed"])

    def test_first_failed_outputs_preserved_without_rerun(self):
        pristine = {"dout": torch.ones(2, 3), "output_weight": torch.ones(4, 3)}
        expected = (torch.full((2, 4), 3.),)
        calls, captured, pointers, progress = [], [], [], []

        def function(**inputs):
            calls.append(len(calls) + 1)
            pointers.append(inputs["dout"].data_ptr())
            output = expected[0].clone()
            if len(calls) == 2:
                output[0, 0] += .001
            if len(calls) == 4:
                output[1, 3] = 1e30
            return (output,)

        def on_failure(iteration, inputs, outputs, record, controls):
            captured.append((iteration, outputs[0].clone(), record))
            return {"iteration": iteration, "saved": True}

        report = stress.stress_loop(function, pristine, expected, operation=stress.replay.MM,
                                    repeats=100, controls=None, failure_callback=on_failure,
                                    progress=progress.append, chunk_elements=3)
        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(len(set(pointers)), 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["rounding_deviation_iterations"], 1)
        self.assertEqual(report["exact_iterations"], 2)
        self.assertEqual(report["failure_capture"]["iteration"], 4)
        self.assertEqual(len(captured), 1)
        self.assertGreater(float(captured[0][1][1, 3]), 1e20)
        self.assertEqual(len(progress), 4)

    def test_hundred_exact_calls_and_input_version_stop(self):
        values = {"dout": torch.ones(2, 3), "output_weight": torch.ones(4, 3)}
        reference = (torch.full((2, 4), 3.),)
        function = mock.Mock(side_effect=lambda **kw: (torch.mm(kw["dout"], kw["output_weight"].T),))
        saved = mock.Mock()
        report = stress.stress_loop(function, values, reference, operation=stress.replay.MM,
                                    repeats=100, controls=None, failure_callback=saved)
        self.assertEqual(function.call_count, 100)
        self.assertEqual(report["exact_iterations"], 100)
        self.assertEqual(report["status"], "PASS")
        saved.assert_not_called()

        def mutating(**kw):
            output = torch.mm(kw["dout"], kw["output_weight"].T)
            kw["dout"].add_(1)
            return (output,)

        report = stress.stress_loop(mutating, values, reference, operation=stress.replay.MM,
                                    repeats=100, controls=None, failure_callback=lambda *args: {"saved": True})
        self.assertEqual(report["iterations_completed"], 1)
        self.assertEqual(report["iterations"][0]["changed_input_versions"], ["dout"])

    def test_failure_artifact_keeps_pristine_current_outputs_and_reference(self):
        raw = torch.arange(60, dtype=torch.float32).reshape(6, 10)
        values = {"dout": raw[:3, 1:7:2], "output_weight": raw[3:, 2:8:2]}
        reference = (values["dout"] @ values["output_weight"].T,)
        inputs, _ = stress.replay.common.restore_raw_tree(values, device="cpu", chunk_bytes=13)
        outputs = (reference[0].clone(),)
        outputs[0][1, 2] = 1e30
        resident_reference = (reference[0].clone(),)
        resident_reference[0][0, 0] += 1
        controls = stress.replay.execution_controls()
        payload = {"operation": stress.replay.MM, "args": (), "kwargs": values, "outputs": reference,
                   "event": {"execution_controls": controls}}
        checks = stress.compare_device_outputs(outputs, reference, ("dy",))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.pt"
            summary = stress.save_failed_call(path, payload, values, inputs, outputs, iteration=7,
                                              checks=checks, configuration={}, chunk_bytes=4096,
                                              gpu_references=resident_reference)
            self.assertTrue(summary["input_bytes_unchanged"])
            self.assertFalse(summary["resident_reference_bytes_unchanged"])
            loaded = torch.load(path, weights_only=False, map_location="cpu")
            self.assertEqual(loaded["iteration"], 7)
            self.assertEqual(loaded["execution_controls"], controls)
            self.assertEqual(loaded["source_execution_controls"], controls)
            self.assertGreater(float(loaded["outputs"][0][1, 2]), 1e20)
            self.assertEqual(loaded["kwargs"]["dout"].stride(), values["dout"].stride())
            self.assertEqual(loaded["kwargs"]["dout"].untyped_storage()._cdata,
                             loaded["kwargs"]["output_weight"].untyped_storage()._cdata)
            self.assertEqual(loaded["failure_locations"]["dy"][0]["index"], [1, 2])
            with self.assertRaises(FileExistsError):
                stress.save_failed_call(path, payload, values, inputs, outputs, iteration=7,
                                         checks=checks, configuration={})

    def test_arguments_preserve_explicit_empty_allocator_and_new_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base = [str(directory / "source.pt"), "--context", str(directory / "context.json"),
                    "--report", str(directory / "report.json"), "--failure-dump", str(directory / "failure.pt")]
            self.assertIsNone(stress.arguments(base).allocator_config)
            self.assertEqual(stress.arguments(base + ["--allocator-config", ""]).allocator_config, "")
            (directory / "report.json").touch()
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                stress.arguments(base)

    def test_norm_optional_output_stress_and_metadata_no_launch(self):
        values = {"x": torch.ones(2, 3)}
        references = (torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3), torch.zeros(3), None)
        function = mock.Mock(side_effect=lambda **kw: tuple(t.clone() if t is not None else None for t in references))
        report = stress.stress_loop(function, values, references, operation=stress.replay.NORM,
                                    repeats=3, controls=None, failure_callback=mock.Mock())
        self.assertEqual(report["exact_iterations"], 3)
        self.assertEqual(report["iterations"][0]["classifications"]["y"], "not_returned")
        self.assertEqual(stress.norm_kernel_metadata(None)["operation_kind"], "torch.mm")
        from types import SimpleNamespace
        config = SimpleNamespace(kwargs={"BLOCK_N": 16}, num_warps=4, num_stages=1, num_ctas=1)
        kernel = SimpleNamespace(device_caches={}, cache={(512,): config}, best_config=config)
        module = SimpleNamespace(_ln_mul_dropout_bwd_dwdb=kernel)
        metadata = stress.norm_kernel_metadata(module)
        self.assertEqual(metadata["triton_kernels"]["_ln_mul_dropout_bwd_dwdb"]["best_config"]["kwargs"], {"BLOCK_N": 16})


if __name__ == "__main__":
    unittest.main()
