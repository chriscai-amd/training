"""Small CPU artifacts exercise pairing, bit identity and rejected stage slots."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

import analyze_narrow_attention_pair as analysis


def fixture(directory, selected="old_dq", *, forced=False, count_error=False,
            global_error=False, unclaimed_write=False, empty=False, ordinary=False,
            dot_value=0.375, negative_zero=False, mutate_k=False):
    n, heads, dim, bm, bn = 7, 2, 2, 4, 4
    inputs = {name: torch.zeros(n, heads, dim, dtype=torch.bfloat16) for name in ("q", "k", "v", "dout")}
    inputs["k"] = torch.arange(n * heads * dim).reshape(n, heads, dim).to(torch.bfloat16) / 32
    inputs.update(seq_offsets=torch.tensor([0, 3, 7]), num_targets=torch.ones(2, dtype=torch.int64),
                  sort_by_length_indices=torch.tensor([1, 0]))
    current = {name: value.clone() for name, value in inputs.items()}
    if mutate_k:
        current["k"][3, 0, 0] = -4
    outputs = {name: torch.zeros(n, heads, dim, dtype=torch.bfloat16) for name in ("dq", "dk", "dv")}
    if negative_zero:
        outputs["dq"][0, 0, 0] = -0.0
    stride = 16 + bn * dim + dim * bm
    spec = {"programs": 4, "block_m": bm, "block_n": bn, "dim": dim, "selected_stage": selected,
            "global_words": 8, "header_words": 16, "program_stride_words": stride,
            "total_words": 8 + 4 * stride, "bytes": 4 * (8 + 4 * stride), "panels": [
                {"name": "k", "shape": [bn, dim], "dtype": "torch.bfloat16", "offset_words": 16, "words": 8},
                {"name": selected, "shape": [dim, bm], "dtype": "torch.float32" if selected == "dot" else "torch.bfloat16",
                 "offset_words": 24, "words": 8}]}
    words = torch.zeros(spec["total_words"], dtype=torch.int32)
    if not empty:
        words[:2] = torch.tensor([1, 2 if global_error else 1])
        row = words[8:8 + stride]
        row[:9] = torch.tensor([3 if selected == "dot" else 2, 0, 0, 0, 4, 4, analysis.MAGIC, int(forced), 2 if count_error else 1])
        row[16:24] = inputs["k"][3:7, 0].contiguous().view(torch.int16).reshape(-1).to(torch.int32) & 65535
        panel = torch.zeros(dim, bm, dtype=torch.float32 if selected == "dot" else torch.bfloat16)
        panel[1, 2] = inputs["k"][3, 0, 0] if selected == "old_dq" else dot_value
        if forced:
            panel.zero_()
            panel[0, 0] = 1
        elif selected == "old_dq":
            outputs["dq"][5, 0, 1] = panel[1, 2]
        else:
            outputs["dq"][5, 0, 1] = panel[1, 2] * 0.5
        row[24:32] = panel.view(torch.int32).reshape(-1) if selected == "dot" else panel.view(torch.int16).reshape(-1).to(torch.int32) & 65535
    else:
        outputs["dq"][0, 1, 1] = 0.5
    if unclaimed_write:
        words[8 + stride + 20] = 99
    stage_path, raw_path = directory / "stage.pt", directory / "raw.pt"
    pair_path = Path(str(stage_path) + ".pair.json")
    stage = {"format": "hstu_dq_narrow_raw_failure_workspace_v2" if ordinary else "hstu_dq_narrow_stage_probe_v1",
             "format_version": 2 if ordinary else 1, "iteration": 4, "selected_stage": selected,
             "force_positive_control": forced, "workspace_spec": spec, "workspace_raw_int32": words,
             "compiled_kernels": [], "decode_error": "saved count mismatch" if count_error else None}
    corr = {"iteration": 4, "narrow_launch_count": 4, "workspace_path": str(stage_path),
            "workspace_sha256": analysis.hashlib.sha256(analysis.tensor_bytes(words)).hexdigest()}
    if ordinary:
        corr.update(raw_failure_path=str(raw_path), first_raw_failure=True)
        stage["correlation"] = copy.deepcopy(corr)
    torch.save(stage, stage_path)
    if not ordinary:
        corr.update(workspace_file_sha256=analysis.file_identity(stage_path)["sha256"], raw_output_path=str(raw_path),
                    pair_path=str(pair_path), selected_stage=selected, force_positive_control=forced,
                    decode_error=stage["decode_error"], actual_kernel_views_match_retained_callers=True,
                    capture_kind="forced_diagnostic_control" if forced else "positive_instrumented_stage")
    raw = {"format": "hstu_attention_backward_failure", "format_version": 1, "iteration": 4,
           "failures": [{"iteration": 4, "check": "forced_narrow_stage_control" if forced else "narrow_stage_positive"}],
           "compiled_kernels": [], "pristine_inputs": inputs, "inputs_at_failure": current, "outputs": outputs,
           "configuration": {"context": {"attn_alpha": 0.5},
                "narrow_stage_workspace_capture" if ordinary else "narrow_positive_stage_capture": corr}}
    torch.save(raw, raw_path)
    if not ordinary:
        pair_path.write_text(json.dumps({"format": "hstu_narrow_positive_pair_v3", "correlation": corr,
            "raw_dump_status": "complete", "raw_file_sha256": analysis.file_identity(raw_path)["sha256"]}))
    return stage_path, raw_path, pair_path


class PairAnalysisTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.rng = torch.get_rng_state().clone()

    def tearDown(self):
        self.assertTrue(torch.equal(self.rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def run_case(self, **kwargs):
        stage, raw, _ = fixture(self.directory, **kwargs)
        return analysis.analyze(stage, raw, max_records=32, max_candidates=4, row_chunk=2)

    def test_old_bits_map_sorted_sequence_to_complete_final_coordinate(self):
        result = self.run_case()
        self.assertTrue(result["workspace"]["workspace_strictly_consistent"])
        item = result["workspace"]["observed_samples"][0]
        self.assertEqual(item["DQ_global_index"], [5, 0, 1])
        self.assertEqual(item["sequence"], 1)
        self.assertTrue(item["stage_BF16_bits_equal_final"])
        self.assertEqual(item["stage_K_bit_candidates"]["sample_global_K_indices"], [[3, 0, 0]])
        self.assertEqual(result["output_summaries"]["dq"]["numeric_nonzero"], 1)
        self.assertTrue(result["input_comparison"]["all_backing_bits_unchanged"])

    def test_count_mismatch_retains_partial_comparison_without_accepting_slot(self):
        result = self.run_case(count_error=True)
        self.assertFalse(result["workspace"]["workspace_strictly_consistent"])
        self.assertEqual(result["workspace"]["recorded_decoder_error"], "saved count mismatch")
        self.assertIn("count_panel_inconsistency", result["workspace"]["slots"][0]["errors"])
        self.assertFalse(result["workspace"]["observed_samples"][0]["slot_locally_consistent"])
        self.assertEqual(result["workspace"]["final_DQ_nonzero_in_locally_consistent_regions"], 0)

    def test_global_inconsistency_does_not_invalidate_local_measurement_or_pass_strictly(self):
        result = self.run_case(global_error=True)
        self.assertFalse(result["workspace"]["workspace_strictly_consistent"])
        self.assertTrue(result["workspace"]["slots"][0]["locally_consistent"])

    def test_all_unclaimed_words_are_scanned(self):
        result = self.run_case(unclaimed_write=True)
        self.assertFalse(result["workspace"]["workspace_strictly_consistent"])
        self.assertIn("unclaimed_slot_contains_writes", result["workspace"]["slots"][1]["errors"])

    def test_negative_zero_is_numeric_zero_but_has_nonzero_bits(self):
        result = self.run_case(negative_zero=True)
        summary = result["output_summaries"]["dq"]
        self.assertEqual(summary["numeric_nonzero"], 1)
        self.assertEqual(summary["nonzero_bits"], 2)
        self.assertEqual(summary["negative_zero"], 1)

    def test_dot_conditional_scale_is_separate_from_direct_identity(self):
        result = self.run_case(selected="dot")
        item = result["workspace"]["observed_samples"][0]
        self.assertTrue(item["dot_exactly_representable_as_BF16"])
        self.assertTrue(item["scaled_dot_candidate_bits_equal_final"])
        self.assertFalse(item["stage_numeric_equals_final"])
        self.assertNotIn("stage_BF16_bits_equal_final", item)

    def test_dot_rounding_is_not_exact_K_correspondence(self):
        result = self.run_case(selected="dot", dot_value=0.37501)
        item = result["workspace"]["observed_samples"][0]
        self.assertFalse(item["dot_exactly_representable_as_BF16"])
        self.assertNotIn("stage_K_bit_candidates", item)

    def test_forced_control_is_labeled_and_sentinel_has_no_K_provenance_search(self):
        result = self.run_case(forced=True)
        self.assertTrue(result["forced_control"])
        self.assertEqual(result["output_summaries"]["dq"]["numeric_nonzero"], 0)
        self.assertNotIn("stage_K_bit_candidates", result["workspace"]["observed_samples"][0])

    def test_mutated_current_K_is_reported_independently(self):
        result = self.run_case(mutate_k=True)
        self.assertFalse(result["input_comparison"]["all_logical_bits_unchanged"])
        slot = result["workspace"]["slots"][0]
        self.assertTrue(slot["K_panel_bits_match_pristine"])
        self.assertFalse(slot["K_panel_bits_match_final_input"])

    def test_empty_inherited_workspace_keeps_ordinary_wrong_final_output(self):
        result = self.run_case(ordinary=True, empty=True)
        self.assertEqual(result["pair_identity"]["capture_kind"], "ordinary_raw_failure")
        self.assertTrue(result["workspace"]["all_workspace_words_zero"])
        self.assertEqual(result["workspace"]["final_DQ_nonzero_outside_decodable_regions"], 1)

    def test_incomplete_pair_is_rejected(self):
        stage, raw, pair = fixture(self.directory)
        value = json.loads(pair.read_text())
        value["raw_dump_status"] = "pending"
        pair.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "not marked complete"):
            analysis.analyze(stage, raw)

    def test_changed_raw_bytes_are_rejected(self):
        stage, raw, _ = fixture(self.directory)
        with raw.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            analysis.analyze(stage, raw)

    def test_control_expectation_cannot_be_silently_dropped(self):
        stage, raw, _ = fixture(self.directory, forced=True)
        with self.assertRaisesRegex(ValueError, "expectation mismatch"):
            analysis.analyze(stage, raw, expect="natural")

    def test_actual_frozen_v3_saver_pair_is_accepted_without_rebuilding_schema(self):
        from test_attention_dq_narrow_stage_probe_v3 import PositiveBuffersV3Test
        from attention_dq_narrow_stage_probe import StageProbePositive
        original = PositiveBuffersV3Test()
        original.setUp()
        try:
            bundle = original.bundle()
            with self.assertRaises(StageProbePositive):
                original.fire(bundle)
            result = analysis.analyze(bundle.controller.destination, bundle.args.failure_dump,
                                      max_records=32, max_candidates=4, row_chunk=2)
            self.assertTrue(result["pair_identity"]["verified"])
            self.assertEqual(result["output_summaries"]["dq"]["numeric_nonzero"], bundle.outputs["dq"].numel())
        finally:
            original.tearDown()
            original.doCleanups()


if __name__ == "__main__":
    unittest.main()
