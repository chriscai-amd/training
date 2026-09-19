"""CPU-only validation of exact-iteration attention failure persistence."""
from pathlib import Path
import tempfile
import unittest

import torch

import repro_hstu_attention_capture as runner


class FailureDumpTests(unittest.TestCase):
    def test_failure_dump_requires_one_call_checks(self):
        for options in (["--failure-dump", "failure.pt"],
                        ["--failure-dump", "failure.pt", "--check-every", "2"],
                        ["--failure-dump", "failure.pt", "--check-every", "1", "--report", "failure.pt"],
                        ["--failure-dump", "failure.pt", "--check-every", "1", "--export-prefix", "prefix.pt"]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                runner.arguments(["capture.pt", *options])
        args = runner.arguments(["capture.pt", "--failure-dump", "failure.pt", "--check-every", "1"])
        self.assertEqual(args.failure_dump, Path("failure.pt"))
        self.assertEqual(runner.arguments(["capture.pt"]).check_every, 5)
        self.assertIsNone(runner.arguments(["capture.pt"]).failure_dump)

    def fixture(self):
        packed = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 16)
        q, k, v = [packed[:, index:index + 4].view(6, 2, 2) for index in (8, 12, 4)]
        inputs = dict(q=q, k=k, v=v, dout=torch.zeros(6, 2, 2),
                      seq_offsets=torch.tensor([0, 3, 6]), num_targets=torch.ones(2, dtype=torch.int64),
                      sort_by_length_indices=torch.tensor([0, 1]))
        inputs["dout"][[2, 5]] = 0.125
        grad_storage = torch.zeros_like(packed)
        outputs = {name: grad_storage[:, index:index + 4].view(6, 2, 2)
                   for name, index in (("dq", 8), ("dk", 12), ("dv", 4))}
        outputs["dq"][1, 0, 1] = 1e30
        outputs["dq"][2, 0, 0] = .01
        outputs["dk"][4, 1, 1] = float("inf")
        return packed, inputs, outputs

    def test_roundtrip_retains_original_outputs_and_pristine_inputs(self):
        before_rng = torch.get_rng_state().clone()
        packed, inputs, outputs = self.fixture()
        pristine = runner.snapshot_failure_inputs(inputs, chunk_bytes=4096, max_bytes=1 << 20)
        saved_q = pristine["inputs"]["q"].clone()
        # Corrupt a logical input and an unused U backing byte after snapshot.
        inputs["q"][0, 0, 0] += 7
        packed[0, 0] = -19
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.pt"
            summary = runner.save_failure_dump(
                path, pristine, inputs, outputs, iteration=27,
                failures=[{"iteration": 27, "check": "dq_history_zero"}],
                configuration={"context": {"max_seq_len": 2963}}, compiled_kernels=[],
                report_path="report.json", chunk_bytes=4096, max_bytes=1 << 20)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["format"], "hstu_attention_backward_failure")
            self.assertEqual(loaded["format_version"], 1)
            self.assertEqual(loaded["iteration"], 27)
            self.assertTrue(torch.equal(loaded["pristine_inputs"]["q"], saved_q))
            self.assertTrue(torch.equal(loaded["inputs_at_failure"]["q"], inputs["q"]))
            for name in outputs:
                self.assertTrue(torch.equal(loaded["outputs"][name], outputs[name]))
                self.assertEqual(loaded["outputs"][name].stride(), outputs[name].stride())
                self.assertEqual(loaded["outputs"][name].storage_offset(), outputs[name].storage_offset())
            self.assertEqual(loaded["outputs"]["dq"].untyped_storage()._cdata,
                             loaded["outputs"]["dk"].untyped_storage()._cdata)
            comparison = summary["input_comparison"]
            self.assertFalse(comparison["all_logical_input_bytes_unchanged"])
            self.assertFalse(comparison["all_input_backing_bytes_unchanged"])
            self.assertEqual(comparison["logical_tensors"]["q"]["mismatched_elements"], 1)
            self.assertEqual(summary["dq_zero_oracle"]["history_nonzero_elements"], 1)
            sample = summary["dq_zero_oracle"]["sample_locations"][0]
            self.assertEqual((sample["row"], sample["head"], sample["feature"]), (1, 0, 1))
            self.assertEqual(sample["sequence"], 0)
            self.assertFalse(Path(str(path) + ".tmp").exists())
            with self.assertRaises(FileExistsError):
                runner.save_failure_dump(path, pristine, inputs, outputs, iteration=27,
                                         failures=[{"iteration": 27}], configuration={}, compiled_kernels=[])
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_clean_inputs_and_reject_delayed_failure(self):
        _, inputs, outputs = self.fixture()
        pristine = runner.snapshot_failure_inputs(inputs, chunk_bytes=4096, max_bytes=1 << 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.pt"
            with self.assertRaisesRegex(ValueError, "current positive iteration"):
                runner.save_failure_dump(path, pristine, inputs, outputs, iteration=30,
                                         failures=[{"iteration": 27}], configuration={}, compiled_kernels=[])
            summary = runner.save_failure_dump(path, pristine, inputs, outputs, iteration=27,
                                             failures=[{"iteration": 27}], configuration={}, compiled_kernels=[],
                                             chunk_bytes=4096, max_bytes=1 << 20)
            self.assertTrue(summary["input_comparison"]["all_logical_input_bytes_unchanged"])
            self.assertTrue(summary["input_comparison"]["all_input_backing_bytes_unchanged"])


if __name__ == "__main__":
    unittest.main()
