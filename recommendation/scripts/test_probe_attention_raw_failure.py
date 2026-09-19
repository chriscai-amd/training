"""CPU-only checks of raw attention replay inputs and experiment controls."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import probe_attention_raw_failure as runner


def fixture():
    packed = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 16) / 128
    inputs = {name: packed[:, start:start + 4].view(6, 2, 2)
              for name, start in (("q", 8), ("k", 12), ("v", 4))}
    inputs.update(dout=torch.zeros(6, 2, 2), seq_offsets=torch.tensor([0, 3, 6]),
                  num_targets=torch.ones(2, dtype=torch.int64), sort_by_length_indices=torch.tensor([0, 1]))
    inputs["dout"][[2, 5]] = 0.125
    grad = torch.zeros_like(packed)
    outputs = {name: grad[:, start:start + 4].view(6, 2, 2)
               for name, start in (("dq", 8), ("dk", 12), ("dv", 4))}
    context = {"has_multiple_targets": True, "num_softmax_heads": 0, "enable_tma": False,
               "sort_by_length": True, "num_heads": 2, "attn_dim": 2, "hidden_dim": 2,
               "max_seq_len": 3, "attn_alpha": 0.5, "max_attn_len": 0, "contextual_seq_len": 0}
    config = {"context": context, "pre_hook": "_bwd_pre_hook", "config": {"SEQUENCE_PARALLEL": False}}
    failure = {"format": "hstu_attention_backward_failure", "format_version": 1,
               "pristine_inputs": inputs, "outputs": outputs, "configuration": config}
    return failure


class FakeAttention:
    def __init__(self, config, *, failures=(), skip_zero=False, mutate_backing=False):
        self.config = config
        self.failures = set(failures)
        self.skip_zero = skip_zero
        self.mutate_backing = mutate_backing
        self.calls = self.launched = 0
        self._hstu_attn_bwd = SimpleNamespace(configs=[config])

    def triton_hstu_attention_bwd(self, **values):
        self.calls += 1
        self.config.pre_hook({"DQ": values["dq"], "SEQUENCE_PARALLEL": False})
        self.launched += 1
        values["dk"].zero_()
        values["dv"].zero_()
        if torch.count_nonzero(values["dout"]) and any(torch.count_nonzero(values[name]) for name in ("q", "k", "v")):
            values["dq"][[2, 5]] = 0.1
        if self.calls in self.failures:
            values["dq"][1, 0, 1] = 0.25
        if self.mutate_backing:
            raw = torch.empty(0, dtype=torch.float32).set_(values["q"].untyped_storage(), 0, (96,), (1,))
            raw[0] += 1  # Unused U column; separate version counter.


class RawFailureTests(unittest.TestCase):
    def setup_loop(self, options=(), *, failures=(), zero_hook=True, mutate=False):
        source = fixture()
        history, digests = runner.validate_capture(source, chunk_bytes=4096)
        args = runner.arguments(["nonexistent-capture.pt", "--repeat", "4", *options])
        expected, _ = runner.prepare_inputs(source["pristine_inputs"], qkv_zero=args.qkv_zero,
                                             dout_zero=args.dout_zero, zero_inputs=args.zero_input,
                                             chunk_bytes=4096, max_bytes=1 << 20)
        digests = runner.input_digests(expected, chunk_bytes=4096, threshold=1e6)
        inputs, size = runner.common.restore_raw_tree(expected, device="cpu", chunk_bytes=4096)
        outputs, _ = runner.allocate_output_views(source["outputs"], device="cpu", max_bytes=1 << 20)
        original = (lambda values: values["DQ"].zero_()) if zero_hook else (lambda values: None)
        config = SimpleNamespace(pre_hook=original)
        attention = FakeAttention(config, failures=failures, mutate_backing=mutate)
        from nan_backward_boundaries import _source_specs
        pristine = {"inputs": expected, "storage_bytes": size,
                    "source_specs": _source_specs(inputs)}
        report = {"configuration": source["configuration"]}
        result = runner.run_loop(args, report, lambda _: None, attention, config, inputs, outputs,
                                 source["configuration"]["context"], history, pristine, digests, chunk_bytes=4096)
        self.assertIs(config.pre_hook, original)
        return result, report, attention, outputs

    def test_raw_restore_keeps_aliases_offsets_and_bytes(self):
        source = fixture()
        history, digests = runner.validate_capture(source, chunk_bytes=4096)
        copied, _ = runner.common.restore_raw_tree(source["pristine_inputs"], device="cpu", chunk_bytes=64)
        self.assertEqual(runner.input_digests(copied, chunk_bytes=4096, threshold=1e6), digests)
        self.assertEqual(copied["q"].storage_offset(), 8)
        self.assertEqual(copied["q"].stride(), (16, 2, 1))
        self.assertEqual(copied["q"].untyped_storage()._cdata, copied["v"].untyped_storage()._cdata)
        self.assertNotEqual(copied["q"].untyped_storage()._cdata, source["pristine_inputs"]["q"].untyped_storage()._cdata)
        self.assertEqual(history.tolist(), [True, True, False, True, True, False])

    def test_output_allocation_and_dq_fill_leave_other_views_alone(self):
        source = fixture()
        outputs, _ = runner.allocate_output_views(source["outputs"], device="cpu", max_bytes=1 << 20)
        outputs["dk"].fill_(3)
        outputs["dv"].fill_(4)
        runner.initialize_dq(outputs["dq"], "sentinel", 7)
        self.assertTrue(bool((outputs["dq"] == 7).all()))
        self.assertTrue(bool((outputs["dk"] == 3).all()))
        self.assertTrue(bool((outputs["dv"] == 4).all()))
        self.assertTrue(bool((source["outputs"]["dq"] == 0).all()))
        self.assertEqual(outputs["dq"].untyped_storage()._cdata, outputs["dv"].untyped_storage()._cdata)

    def test_qkv_zero_preserves_complete_raw_tree_and_source_evidence(self):
        source = fixture()["pristine_inputs"]
        before = runner.input_digests(source, chunk_bytes=4096, threshold=1e6)
        for dout_zero in (False, True):
            with self.subTest(dout_zero=dout_zero):
                changed, preparation = runner.prepare_inputs(source, qkv_zero=True, dout_zero=dout_zero,
                                                              chunk_bytes=64, max_bytes=1 << 20)
                self.assertTrue(preparation["cloned_complete_input_backings"])
                self.assertIn("unused packed U", preparation["qkv_zero_scope"])
                for name, tensor in changed.items():
                    self.assertEqual(tensor.shape, source[name].shape)
                    self.assertEqual(tensor.stride(), source[name].stride())
                    self.assertEqual(tensor.storage_offset(), source[name].storage_offset())
                    self.assertEqual(tensor.untyped_storage().nbytes(), source[name].untyped_storage().nbytes())
                    self.assertNotEqual(tensor.untyped_storage()._cdata, source[name].untyped_storage()._cdata)
                self.assertEqual(len({changed[name].untyped_storage()._cdata for name in ("q", "k", "v")}), 1)
                raw = torch.empty(0, dtype=torch.float32).set_(changed["q"].untyped_storage(), 0, (6, 16), (16, 1))
                original_raw = torch.empty(0, dtype=torch.float32).set_(source["q"].untyped_storage(), 0, (6, 16), (16, 1))
                expected_raw = original_raw.clone()
                expected_raw[:, 4:].zero_()
                self.assertTrue(torch.equal(raw, expected_raw))
                self.assertTrue(torch.equal(changed["dout"], torch.zeros_like(source["dout"]) if dout_zero else source["dout"]))
                for name in ("seq_offsets", "num_targets", "sort_by_length_indices"):
                    self.assertTrue(torch.equal(changed[name], source[name]))
                self.assertEqual(runner.input_digests(source, chunk_bytes=4096, threshold=1e6), before)

    def test_default_input_preparation_reuses_source_without_copy_or_write(self):
        source = fixture()["pristine_inputs"]
        unchanged, preparation = runner.prepare_inputs(source, qkv_zero=False, dout_zero=False,
                                                       chunk_bytes=64, max_bytes=1 << 20)
        self.assertTrue(all(unchanged[name] is tensor for name, tensor in source.items()))
        self.assertFalse(preparation["cloned_complete_input_backings"])
        self.assertEqual(preparation["logical_inputs_zeroed"], [])

    def test_selected_zero_input_changes_only_selected_logical_bytes(self):
        source = fixture()["pristine_inputs"]
        before = runner.input_digests(source, chunk_bytes=4096, threshold=1e6)
        original_raw = torch.empty(0, dtype=torch.float32).set_(source["q"].untyped_storage(), 0, (6, 16), (16, 1))
        for selected in ("q", "k", "v"):
            with self.subTest(selected=selected):
                changed, preparation = runner.prepare_inputs(
                    source, qkv_zero=False, dout_zero=True, zero_inputs=[selected, selected],
                    chunk_bytes=64, max_bytes=1 << 20)
                self.assertEqual(preparation["selected_zero_inputs"], [selected])
                self.assertEqual(preparation["logical_inputs_zeroed"], [selected, "dout"])
                self.assertEqual(len({changed[name].untyped_storage()._cdata for name in ("q", "k", "v")}), 1)
                for name, tensor in changed.items():
                    self.assertEqual(tensor.shape, source[name].shape)
                    self.assertEqual(tensor.stride(), source[name].stride())
                    self.assertEqual(tensor.storage_offset(), source[name].storage_offset())
                    self.assertNotEqual(tensor.untyped_storage()._cdata, source[name].untyped_storage()._cdata)
                    self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor) if name in (selected, "dout") else source[name]))
                raw = torch.empty(0, dtype=torch.float32).set_(changed["q"].untyped_storage(), 0, (6, 16), (16, 1))
                expected_raw = original_raw.clone()
                start = {"q": 8, "k": 12, "v": 4}[selected]
                expected_raw[:, start:start + 4].zero_()
                self.assertTrue(torch.equal(raw, expected_raw))
                self.assertEqual(runner.input_digests(source, chunk_bytes=4096, threshold=1e6), before)

    def test_repeatable_zero_input_parser_and_full_convenience(self):
        args = runner.arguments(["capture.pt", "--zero-input", "v", "--zero-input", "q", "--dout-zero"])
        _, preparation = runner.prepare_inputs(fixture()["pristine_inputs"], qkv_zero=args.qkv_zero,
                                                dout_zero=args.dout_zero, zero_inputs=args.zero_input,
                                                chunk_bytes=64, max_bytes=1 << 20)
        self.assertEqual(preparation["selected_zero_inputs"], ["q", "v"])
        _, preparation = runner.prepare_inputs(fixture()["pristine_inputs"], qkv_zero=True, dout_zero=False,
                                                zero_inputs=["k"], chunk_bytes=64, max_bytes=1 << 20)
        self.assertEqual(preparation["selected_zero_inputs"], ["q", "k", "v"])

    def test_all_initialization_modes_pass_actual_zero_hook(self):
        for mode in ("production", "zero", "sentinel", "nan"):
            with self.subTest(mode=mode):
                result, report, _, _ = self.setup_loop(["--dq-init", mode, "--prehook-check"])
                self.assertEqual(result, 0)
                self.assertEqual(report["pre_hook_totals"]["calls"], 4)
                self.assertEqual(report["pre_hook_totals"]["zero_checks"], 4)
                self.assertEqual(report["failed_iterations"], 0)

    def test_keep_going_counts_intermittent_failures(self):
        result, report, attention, _ = self.setup_loop(["--keep-going"], failures=(2, 4))
        self.assertEqual(result, 1)
        self.assertEqual(attention.launched, 4)
        self.assertEqual(report["failed_iterations"], 2)
        self.assertEqual(report["failure_counts"]["dq_zero_oracle"], 2)
        sample = report["checks"][1]["outputs"]["dq"]["sample_bad_rows"][0]
        self.assertEqual((sample["row"], sample["sequence"], sample["sequence_position"]), (1, 0, 1))
        self.assertTrue(report["checks"][1]["input_bytes"]["all_logical_and_backing_bytes_unchanged"])

    def test_default_stops_at_first_bad_call(self):
        _, report, attention, _ = self.setup_loop(failures=(2, 4))
        self.assertEqual(attention.launched, 2)
        self.assertEqual(report["failed_iterations"], 1)

    def test_prehook_failure_is_separate_and_prevents_kernel(self):
        result, report, attention, _ = self.setup_loop(
            ["--keep-going", "--dq-init", "sentinel", "--prehook-check"], zero_hook=False)
        self.assertEqual(result, 1)
        self.assertEqual(attention.launched, 0)
        self.assertEqual(report["failed_iterations"], 4)
        self.assertEqual(report["pre_hook_totals"]["zero_failures"], 4)
        self.assertTrue(all(not c["attention_call_completed"] for c in report["checks"]))
        self.assertTrue(all("dq_zero_oracle" not in c["failures"] for c in report["checks"]))

    def test_raw_input_check_finds_unused_backing_mutation(self):
        result, report, _, _ = self.setup_loop(["--input-check", "every"], mutate=True)
        self.assertEqual(result, 1)
        self.assertIn("input_bytes_changed", report["checks"][0]["failures"])
        self.assertFalse(any("version_changed" in name for name in report["checks"][0]["failures"]))

    def test_zero_dout_checks_all_three_gradients(self):
        result, report, _, _ = self.setup_loop(["--dout-zero", "--prehook-check"])
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "PASS")

    def test_qkv_zero_does_not_require_nonzero_target_dq(self):
        for options in (["--qkv-zero"], ["--qkv-zero", "--dout-zero"]):
            with self.subTest(options=options):
                result, report, _, _ = self.setup_loop([*options, "--prehook-check"])
                self.assertEqual(result, 0)
                self.assertEqual(report["status"], "PASS")

    def test_keep_going_saves_only_first_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first.pt"
            _, report, _, _ = self.setup_loop(["--keep-going", "--failure-dump", str(path)], failures=(2, 4))
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["iteration"], 2)
            self.assertEqual(report["iterations_completed"], 4)
            self.assertEqual(report["first_failure_dump"]["iteration"], 2)

    def test_invalid_history_dout_is_rejected(self):
        source = fixture()
        source["pristine_inputs"]["dout"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "history dOut"):
            runner.validate_capture(source, chunk_bytes=4096)

    def test_diagnostic_kernel_scope_restores_raw_function_on_error(self):
        original = SimpleNamespace(arg_names=["Q", "DQ"])
        replacement = SimpleNamespace(arg_names=["Q", "DQ"], diagnostic_metadata={"variant": "baseline"})
        attention = SimpleNamespace(_hstu_attn_bwd=SimpleNamespace(fn=original))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with runner.selected_kernel(attention, "baseline", lambda *_: replacement) as metadata:
                self.assertIs(attention._hstu_attn_bwd.fn, replacement)
                self.assertEqual(metadata, {"variant": "baseline"})
                raise RuntimeError("injected")
        self.assertIs(attention._hstu_attn_bwd.fn, original)

    def test_changed_math_kernel_requires_zero_dout_intervention(self):
        for variant in ("load_only", "dot_only", "store_zero"):
            with self.subTest(variant=variant), self.assertRaises(SystemExit):
                runner.arguments(["capture.pt", "--kernel-variant", variant])
            options = runner.arguments(["capture.pt", "--kernel-variant", variant, "--dout-zero"])
            self.assertEqual(options.kernel_variant, variant)


if __name__ == "__main__":
    program = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU tests unexpectedly initialized CUDA"
    sys.exit(not program.result.wasSuccessful())
