"""CPU checks of complete workspace capture at the first raw failing launch."""

import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import attention_dq_narrow_stage_probe as base
import attention_dq_narrow_stage_probe_v2 as v2
import probe_attention_raw_failure as runner
from nan_backward_boundaries import _source_specs
from test_attention_dq_narrow_stage_probe import fixture as workspace_fixture
from test_probe_attention_raw_failure import fixture as raw_fixture


class WorkspaceFailureV2Test(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.rng = torch.get_rng_state().clone()

    def tearDown(self):
        self.assertTrue(torch.equal(self.rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def controller(self, *, raw=None, calls=2):
        positive, spec, pristine, _ = workspace_fixture("dot")
        controller = v2.NarrowStageControllerV2(self.directory / "workspace.pt", 1 << 20, "dot")
        controller.spec = spec
        controller.workspace = torch.zeros_like(positive) if raw is None else raw.clone()
        controller.pristine = pristine
        controller.calls = calls
        controller.source_capture = "source.pt"
        controller.expected_digests = {}
        controller.configuration = {"config": {"SEQUENCE_PARALLEL": False}}
        controller.metadata = {"variant": "unchanged_v1_kernel"}
        controller.launch_metadata = {"launch_grid": [4, 1], "workspace_spec": spec}
        controller.report = {}
        return controller

    def capture(self, controller, original_save, *, iteration=2, failures=None):
        return controller.save_raw_failure(
            original_save, self.directory / "raw.pt", {"inputs": controller.pristine}, {},
            {"dq": torch.ones((7, 2, 2), dtype=torch.bfloat16)}, iteration=iteration,
            failures=failures or [{"iteration": iteration, "check": "dq_zero_oracle"}],
            configuration=controller.configuration, compiled_kernels=[],
        )

    def simple_saver(self, path, pristine, inputs, outputs, **kwargs):
        torch.save({"outputs": outputs, **kwargs}, path)
        return {"path": str(path), "iteration": kwargs["iteration"]}

    def test_complete_scan_includes_word_zero_reserved_words_and_unclaimed_panels(self):
        raw, spec, _, _ = workspace_fixture("dot")
        raw.zero_()
        scan = v2.summarize_workspace(raw, spec)
        self.assertTrue(scan["all_workspace_words_zero"])
        self.assertEqual(scan["scanned_words"], spec["total_words"])
        self.assertEqual(scan["scanned_bytes"], spec["bytes"])
        self.assertEqual(scan["global_words"], [0] * base.GLOBAL_WORDS)
        self.assertIn("not proof", scan["interpretation"])
        for offset in (0, 1, 7, base.GLOBAL_WORDS + 9, spec["total_words"] - 1):
            with self.subTest(offset=offset):
                raw.zero_()
                raw[offset] = -2147483648
                scan = v2.summarize_workspace(raw, spec)
                self.assertFalse(scan["all_workspace_words_zero"])
                self.assertEqual(scan["nonzero_words"], 1)
                self.assertEqual(scan["global_first_claim_word"], -2147483648 if offset == 0 else 0)
                self.assertEqual(scan["global_words"], raw[:base.GLOBAL_WORDS].tolist())
                self.assertEqual(scan["first_nonzero_words"], [
                    {"offset_words": offset, "int32": -2147483648, "hex_uint32": "0x80000000"}])
                self.assertEqual(scan["unclaimed_program_nonzero_words"], int(offset >= base.GLOBAL_WORDS))
                self.assertEqual(len(scan["per_program"]), spec["programs"])

    def test_workspace_saved_before_raw_dump_and_links_same_iteration(self):
        controller = self.controller()
        controller.workspace[-1] = 187
        before = controller.workspace.clone()
        original_configuration = copy.deepcopy(controller.configuration)
        calls = []

        def save(path, pristine, inputs, outputs, **kwargs):
            calls.append(kwargs["iteration"])
            self.assertTrue(controller.destination.exists())
            workspace = torch.load(controller.destination, weights_only=False)
            self.assertTrue(torch.equal(workspace["workspace_raw_int32"], before))
            self.assertEqual(workspace["trigger"], "raw_output_failure")
            self.assertEqual(workspace["iteration"], 2)
            self.assertEqual(workspace["correlation"]["raw_failure_path"], str(Path(path).resolve()))
            self.assertEqual(workspace["correlation"]["workspace_sha256"],
                             hashlib.sha256(before.numpy().tobytes()).hexdigest())
            self.assertEqual(kwargs["configuration"]["narrow_stage_workspace_capture"],
                             workspace["correlation"])
            self.assertFalse(workspace["workspace_scan"]["all_workspace_words_zero"])
            self.assertEqual(workspace["workspace_scan"]["global_first_claim_word"], 0)
            self.assertEqual(workspace["workspace_scan"]["unclaimed_program_nonzero_words"], 1)
            self.assertEqual(workspace["records"], [])
            controller.workspace.fill_(999)  # Must not alter already saved evidence.
            return self.simple_saver(path, pristine, inputs, outputs, **kwargs)

        result = self.capture(controller, save)
        self.assertEqual(calls, [2])
        self.assertEqual(result["narrow_stage_workspace"]["iteration"], 2)
        self.assertEqual(controller.configuration, original_configuration)
        self.assertEqual(controller.raw_failure_capture["raw_failure_dump_status"], "complete")
        self.assertTrue(torch.equal(torch.load(controller.destination, weights_only=False)["workspace_raw_int32"], before))
        self.assertFalse(controller.destination.with_suffix(".pt.tmp").exists())

    def test_inconsistent_global_claim_preserves_raw_bits_and_decode_error(self):
        controller = self.controller()
        controller.workspace[0] = 1
        controller.workspace[7] = 91
        before = controller.workspace.clone()
        self.capture(controller, self.simple_saver)
        payload = torch.load(controller.destination, weights_only=False)
        self.assertTrue(torch.equal(payload["workspace_raw_int32"], before))
        self.assertIn("decode_error", payload)
        self.assertEqual(payload["workspace_scan"]["nonzero_words"], 2)
        self.assertFalse(payload["workspace_scan"]["all_workspace_words_zero"])

    def test_all_zero_workspace_is_saved_without_claiming_arithmetic_correctness(self):
        controller = self.controller()
        self.capture(controller, self.simple_saver)
        payload = torch.load(controller.destination, weights_only=False)
        self.assertTrue(payload["workspace_scan"]["all_workspace_words_zero"])
        self.assertEqual(payload["workspace_scan"]["nonzero_words"], 0)
        self.assertEqual(payload["records"], [])
        self.assertIn("not proof", payload["scope"])
        self.assertEqual(payload["trigger"], "raw_output_failure")

    def test_iteration_mismatch_missing_workspace_and_failure_record_mismatch_reject(self):
        for problem in ("old_workspace", "no_launch", "missing_workspace", "failure_iteration"):
            with self.subTest(problem=problem):
                controller = self.controller()
                if problem == "old_workspace":
                    controller.calls = 1
                elif problem == "no_launch":
                    controller.calls = 0
                elif problem == "missing_workspace":
                    controller.workspace = None
                failures = [{"iteration": 1 if problem == "failure_iteration" else 2,
                             "check": "dq_zero_oracle"}]
                with self.assertRaises(ValueError):
                    self.capture(controller, lambda *_args, **_kwargs: self.fail("must not save stale outputs"),
                                 failures=failures)
                self.assertFalse(controller.destination.exists())

    def test_first_failure_forbids_repeated_capture_or_subsequent_launch(self):
        controller = self.controller()
        self.capture(controller, self.simple_saver)
        artifact = controller.destination.read_bytes()
        with self.assertRaises(ValueError):
            self.capture(controller, lambda *_args, **_kwargs: self.fail("second save"))
        with self.assertRaises(ValueError):
            controller.launch(lambda **_kwargs: self.fail("post-failure launch"), {}, grid=(4, 1))
        self.assertEqual(controller.destination.read_bytes(), artifact)
        self.assertEqual(controller.calls, 2)

    def test_raw_dump_error_leaves_workspace_and_fatal_capture_state(self):
        controller = self.controller()

        def fail(*_args, **_kwargs):
            raise OSError("raw dump unavailable")

        with self.assertRaisesRegex(OSError, "raw dump unavailable"):
            self.capture(controller, fail)
        self.assertTrue(controller.destination.exists())
        self.assertEqual(controller.raw_failure_capture["raw_failure_dump_status"], "failed")
        with self.assertRaises(ValueError):
            controller.launch(lambda **_kwargs: self.fail("post-failure launch"), {}, grid=(4, 1))

    def cli(self, *raw_options):
        return ["--stage", "dot", "--stage-dump", str(self.directory / "workspace.pt"), "--",
                str(self.directory / "source.pt"), "--dout-zero", *raw_options]

    def test_cli_rejects_keep_going_abbreviations_missing_raw_dump_and_path_collision(self):
        raw_path = self.directory / "raw.pt"
        for option in ("--keep-going", "--keep-g"):
            with self.subTest(option=option):
                with patch.object(v2.base, "main", side_effect=AssertionError("GPU runner reached")):
                    with self.assertRaisesRegex(ValueError, "keep-going"):
                        v2.main(self.cli("--failure-dump", str(raw_path), option))
        with self.assertRaisesRegex(ValueError, "failure-dump"):
            v2.arguments(self.cli())
        with self.assertRaisesRegex(ValueError, "distinct"):
            v2.arguments(self.cli("--failure-dump", str(self.directory / "workspace.pt")))
        options, raw_args = v2.arguments(self.cli("--failure-dump", str(raw_path)))
        self.assertEqual(options.stage, "dot")
        self.assertEqual(raw_args.failure_dump, raw_path)
        self.assertFalse(raw_args.keep_going)

    def test_scoped_host_hooks_restore_on_error(self):
        original_class, original_save = base.NarrowStageController, runner.save_failure_dump
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with v2.capture_hooks():
                controller = base.NarrowStageController(self.directory / "stage.pt", 1 << 20, "dot")
                self.assertIsInstance(controller, v2.NarrowStageControllerV2)
                self.assertIsNot(runner.save_failure_dump, original_save)
                raise RuntimeError("stop")
        self.assertIs(base.NarrowStageController, original_class)
        self.assertIs(runner.save_failure_dump, original_save)

    def test_real_raw_loop_stops_and_snapshots_second_launch_before_any_third_launch(self):
        source = raw_fixture()
        # Use actual shared BF16 layouts accepted by the narrow controller.
        packed = torch.arange(6 * 16).reshape(6, 16).to(torch.bfloat16) / 128
        source["pristine_inputs"] = {name: packed[:, start:start + 4].view(6, 2, 2)
            for name, start in (("q", 8), ("k", 12), ("v", 4))}
        source["pristine_inputs"].update(
            dout=torch.zeros((6, 2, 2), dtype=torch.bfloat16), seq_offsets=torch.tensor([0, 3, 6]),
            num_targets=torch.ones(2, dtype=torch.int64), sort_by_length_indices=torch.tensor([0, 1]))
        source["pristine_inputs"]["dout"][[2, 5]] = 0.125
        grad = torch.zeros_like(packed)
        source["outputs"] = {name: grad[:, start:start + 4].view(6, 2, 2)
            for name, start in (("dq", 8), ("dk", 12), ("dv", 4))}
        history, _ = runner.validate_capture(source, chunk_bytes=4096)
        args = runner.arguments([str(self.directory / "source.pt"), "--repeat", "4", "--dout-zero",
            "--zero-input", "q", "--zero-input", "v", "--failure-dump", str(self.directory / "raw.pt")])
        expected, _ = runner.prepare_inputs(source["pristine_inputs"], dout_zero=True,
            qkv_zero=False, zero_inputs=["q", "v"], chunk_bytes=4096, max_bytes=1 << 20)
        inputs, size = runner.common.restore_raw_tree(expected, device="cpu", chunk_bytes=4096)
        outputs, _ = runner.allocate_output_views(source["outputs"], device="cpu", max_bytes=1 << 20)
        digests = runner.input_digests(expected, chunk_bytes=4096, threshold=1e6)
        pristine = {"inputs": expected, "storage_bytes": size, "source_specs": _source_specs(inputs)}
        config = SimpleNamespace(pre_hook=lambda values: values["DQ"].zero_(),
            kwargs={"BLOCK_M": 4, "BLOCK_N": 4, "SEQUENCE_PARALLEL": False})
        controller = v2.NarrowStageControllerV2(self.directory / "workspace.pt", 1 << 20, "dot")
        controller.pristine, controller.config = expected, config
        controller.source_capture, controller.expected_digests = str(args.capture), digests
        controller.configuration = source["configuration"]
        report = {"configuration": source["configuration"]}
        controller.report = report
        launches = []

        def attention_backward(**values):
            config.pre_hook({"DQ": values["dq"], "SEQUENCE_PARALLEL": False})

            def kernel(**actual):
                launches.append(controller.calls)
                self.assertLessEqual(len(launches), 2, "third launch would destroy first-failure workspace")
                for output in outputs.values():
                    output.zero_()
                # Global claim stays zero; retained unclaimed bytes differ by launch.
                actual["LOCK"][-1] = 100 + len(launches)
                if len(launches) == 2:
                    outputs["dq"][1, 0, 1] = 1.3828125

            controller.launch(kernel, {"Q": values["q"], "Z": 2, "H": 2, "DimQ": 2}, grid=(4, 1))

        attention = SimpleNamespace(triton_hstu_attention_bwd=attention_backward,
                                    _hstu_attn_bwd=SimpleNamespace(device_caches={}))
        controller.attention = attention
        original_save = runner.save_failure_dump

        def save(*args, **kwargs):
            return controller.save_raw_failure(original_save, *args, **kwargs)

        with patch.object(runner, "save_failure_dump", side_effect=save):
            result = runner.run_loop(args, report, lambda _event: None, attention, config, inputs,
                outputs, source["configuration"]["context"], history, pristine, digests, chunk_bytes=4096)
        self.assertEqual(result, 1)
        self.assertEqual(launches, [1, 2])
        self.assertEqual(report["iterations_completed"], 2)
        self.assertEqual([check["status"] for check in report["checks"]], ["PASS", "FAIL"])
        raw_saved = torch.load(args.failure_dump, weights_only=False)
        stage_saved = torch.load(controller.destination, weights_only=False)
        self.assertEqual(raw_saved["iteration"], stage_saved["iteration"])
        self.assertEqual(stage_saved["iteration"], 2)
        self.assertEqual(stage_saved["workspace_raw_int32"][-1].item(), 102)
        self.assertEqual(raw_saved["outputs"]["dq"][1, 0, 1].item(), 1.3828125)
        self.assertEqual(raw_saved["configuration"]["narrow_stage_workspace_capture"], stage_saved["correlation"])
        self.assertEqual(stage_saved["workspace_scan"]["global_first_claim_word"], 0)
        self.assertFalse(stage_saved["workspace_scan"]["all_workspace_words_zero"])
        self.assertTrue(raw_saved["input_comparison"]["all_input_backing_bytes_unchanged"])


if __name__ == "__main__":
    unittest.main()
