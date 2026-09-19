"""CPU checks of complete final buffers captured at a positive narrow stage.

These tests use the unchanged v1 launch/snapshot logic and the actual raw
saver/loop with tiny CPU tensors. They never compile or launch GPU code.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import attention_dq_narrow_stage_probe as base
import attention_dq_narrow_stage_probe_v2 as v2
import attention_dq_narrow_stage_probe_v3 as v3
import probe_attention_raw_failure as runner
from nan_backward_boundaries import _source_specs
from test_attention_dq_narrow_stage_probe import fixture as workspace_fixture
from test_probe_attention_raw_failure import fixture as raw_fixture


ARGUMENTS = {"Q": "q", "K": "k", "V": "v", "DOut": "dout",
             "DQ": "dq", "DK": "dk", "DV": "dv",
             "seq_offsets": "seq_offsets", "num_targets": "num_targets",
             "sort_by_length_indices": "sort_by_length_indices"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def storage_bytes(tensor):
    storage = tensor.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
    return raw.numpy().tobytes()


class PositiveBuffersV3Test(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cases = 0
        self.rng = torch.get_rng_state().clone()

    def tearDown(self):
        self.assertTrue(torch.equal(self.rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def bundle(self, *, forced=False, corrupt=False, controller_factory=None, bind=True, save=None):
        self.cases += 1
        directory = self.directory / str(self.cases)
        directory.mkdir()
        raw, spec, workspace_pristine, _ = workspace_fixture("old_dq", forced=forced)
        if corrupt:
            raw[1] = 2
        source = raw_fixture()
        packed = torch.arange(7 * 16).reshape(7, 16).to(torch.bfloat16) / 64
        source["pristine_inputs"] = {name: packed[:, start:start + 4].view(7, 2, 2)
            for name, start in (("q", 8), ("k", 12), ("v", 4))}
        source["pristine_inputs"]["k"].copy_(workspace_pristine["k"])
        source["pristine_inputs"].update(
            dout=torch.zeros((7, 2, 2), dtype=torch.bfloat16),
            seq_offsets=workspace_pristine["seq_offsets"],
            num_targets=torch.ones(2, dtype=torch.int64),
            sort_by_length_indices=workspace_pristine["sort_by_length_indices"])
        source["pristine_inputs"]["dout"][[2, 6]] = 0.125
        grad = torch.full_like(packed, -0.375)
        source["outputs"] = {name: grad[:, start:start + 4].view(7, 2, 2)
            for name, start in (("dq", 8), ("dk", 12), ("dv", 4))}
        source["configuration"]["context"]["max_seq_len"] = 4
        history, _ = runner.validate_capture(source, chunk_bytes=4096)
        args = runner.arguments([str(directory / "source.pt"), "--repeat", "4", "--dout-zero",
            "--zero-input", "q", "--zero-input", "v", "--failure-dump", str(directory / "raw.pt")])
        expected, _ = runner.prepare_inputs(source["pristine_inputs"], dout_zero=True,
            qkv_zero=False, zero_inputs=["q", "v"], chunk_bytes=4096, max_bytes=1 << 20)
        inputs, size = runner.common.restore_raw_tree(expected, device="cpu", chunk_bytes=4096)
        outputs, _ = runner.allocate_output_views(source["outputs"], device="cpu", max_bytes=1 << 20)
        digests = runner.input_digests(expected, chunk_bytes=4096, threshold=1e6)
        pristine = {"inputs": expected, "storage_bytes": size, "source_specs": _source_specs(inputs)}
        config = SimpleNamespace(pre_hook=lambda values: values["DQ"].zero_(),
            kwargs={"BLOCK_M": 4, "BLOCK_N": 4, "SEQUENCE_PARALLEL": False})
        cls = controller_factory or v3.NarrowStageControllerV3
        controller = cls(directory / "workspace.pt", 1 << 20, "old_dq", force_positive_control=forced)
        controller.pristine, controller.config = expected, config
        controller.source_capture, controller.expected_digests = str(args.capture), digests
        controller.configuration = source["configuration"]
        controller.metadata = {"variant": "unchanged_v1_kernel"}
        controller.attention = SimpleNamespace(_hstu_attn_bwd=SimpleNamespace(device_caches={}))
        report = {"configuration": source["configuration"], "iterations_completed": 0}
        controller.report = report
        kwargs = {argument: (inputs if name in inputs else outputs)[name]
                  for argument, name in ARGUMENTS.items()}
        kwargs.update(Z=2, H=2, DimQ=2)
        result = SimpleNamespace(directory=directory, controller=controller, raw=raw, spec=spec,
            source=source, args=args, inputs=inputs, outputs=outputs, pristine=pristine,
            config=config, report=report, history=history, digests=digests, kwargs=kwargs)
        if bind:
            controller.bind_raw_context(args, report, inputs, outputs, pristine, chunk_bytes=4096, save=save)
        return result

    def fill_outputs(self, bundle):
        for index, output in enumerate(bundle.outputs.values(), 1):
            output.copy_(torch.arange(output.numel()).reshape(output.shape).to(output.dtype) / 16 + index)

    def fire(self, bundle, *, mutate_input=False):
        self.fill_outputs(bundle)

        def kernel(**actual):
            if mutate_input:
                bundle.inputs["k"][6, 1, 1] = -4
            actual["LOCK"].copy_(bundle.raw)

        return bundle.controller.launch(kernel, bundle.kwargs, grid=(4, 1))

    def assert_full_saved_buffers(self, bundle, artifact):
        self.assertEqual(set(artifact["inputs_at_failure"]), set(bundle.inputs))
        self.assertEqual(set(artifact["pristine_inputs"]), set(bundle.pristine["inputs"]))
        self.assertEqual(set(artifact["outputs"]), {"dq", "dk", "dv"})
        for saved_name, current in (("inputs_at_failure", bundle.inputs),
                                    ("pristine_inputs", bundle.pristine["inputs"]),
                                    ("outputs", bundle.outputs)):
            for name, tensor in current.items():
                with self.subTest(saved_name=saved_name, tensor=name):
                    saved = artifact[saved_name][name]
                    self.assertEqual(saved.shape, tensor.shape)
                    self.assertEqual(saved.stride(), tensor.stride())
                    self.assertEqual(saved.storage_offset(), tensor.storage_offset())
                    self.assertEqual(saved.dtype, tensor.dtype)
                    self.assertEqual(storage_bytes(saved), storage_bytes(tensor))
                    self.assertNotEqual(saved.untyped_storage()._cdata, tensor.untyped_storage()._cdata)
        for tree, names in ((artifact["inputs_at_failure"], ("q", "k", "v")),
                            (artifact["outputs"], ("dq", "dk", "dv"))):
            self.assertEqual(len({tree[name].untyped_storage()._cdata for name in names}), 1)

    def assert_pair(self, bundle, *, expected_iteration=1, forced=False, corrupt=False):
        controller = bundle.controller
        self.assertEqual(controller.pair_path, Path(str(controller.destination) + ".pair.json"))
        pair = json.loads(controller.pair_path.read_text())
        self.assertEqual(pair["format"], "hstu_narrow_positive_pair_v3")
        self.assertEqual(pair["raw_dump_status"], "complete")
        self.assertEqual(pair["raw_file_sha256"], file_hash(bundle.args.failure_dump))
        corr = pair["correlation"]
        self.assertEqual(corr["iteration"], expected_iteration)
        self.assertEqual(Path(corr["workspace_path"]).resolve(), controller.destination.resolve())
        self.assertEqual(Path(corr["raw_output_path"]).resolve(), bundle.args.failure_dump.resolve())
        self.assertEqual(corr["workspace_sha256"], hashlib.sha256(bundle.raw.numpy().tobytes()).hexdigest())
        self.assertEqual(corr["workspace_file_sha256"], file_hash(controller.destination))
        self.assertEqual(corr["force_positive_control"], forced)
        self.assertEqual(bool(corr["decode_error"]), corrupt)
        artifact = torch.load(bundle.args.failure_dump, weights_only=False)
        stage = torch.load(controller.destination, weights_only=False)
        self.assertEqual(artifact["configuration"]["narrow_positive_stage_capture"], corr)
        self.assertEqual(stage["format"], "hstu_dq_narrow_stage_probe_v1")
        self.assertEqual(artifact["iteration"], stage["iteration"])
        self.assertEqual(stage["iteration"], expected_iteration)
        self.assertTrue(torch.equal(stage["workspace_raw_int32"], bundle.raw))
        self.assertEqual(controller.positive_output_capture["raw_dump_status"], "complete")
        self.assertEqual(controller.positive_output_capture["iteration"], expected_iteration)
        return artifact, stage, pair

    def test_positive_launch_saves_complete_original_buffers_before_reraise_and_preserves_v1_bytes(self):
        saved_stage_bytes = []
        saver_calls = []
        original_save = runner.save_failure_dump
        bundle = None

        def observer(path, pristine, inputs, outputs, **kwargs):
            self.assertIs(inputs, bundle.inputs)
            self.assertIs(outputs, bundle.outputs)
            self.assertIs(pristine, bundle.pristine)
            saved_stage_bytes.append(bundle.controller.destination.read_bytes())
            pair = json.loads(bundle.controller.pair_path.read_text())
            self.assertEqual(pair["raw_dump_status"], "pending")
            self.assertEqual(pair["correlation"]["workspace_file_sha256"], file_hash(bundle.controller.destination))
            saver_calls.append(kwargs["iteration"])
            return original_save(path, pristine, inputs, outputs, **kwargs)

        bundle = self.bundle(save=observer)
        original_configuration = copy.deepcopy(bundle.report["configuration"])
        with self.assertRaises(base.StageProbePositive):
            self.fire(bundle)
        artifact, stage, _ = self.assert_pair(bundle)
        self.assertEqual(saver_calls, [1])
        self.assertEqual(bundle.controller.destination.read_bytes(), saved_stage_bytes[0])
        self.assertNotIn("narrow_positive_stage_capture", stage["configuration"])
        self.assertEqual(bundle.report["configuration"], original_configuration)
        self.assert_full_saved_buffers(bundle, artifact)
        self.assertTrue(artifact["input_comparison"]["all_input_backing_bytes_unchanged"])
        self.assertGreater(artifact["dq_zero_oracle"]["history_nonzero_elements"], 2)

    def test_positive_current_input_mutation_is_saved_separately_from_pristine(self):
        bundle = self.bundle()
        pristine_bits = storage_bytes(bundle.pristine["inputs"]["k"])
        with self.assertRaises(base.StageProbePositive):
            self.fire(bundle, mutate_input=True)
        artifact, _, _ = self.assert_pair(bundle)
        self.assert_full_saved_buffers(bundle, artifact)
        self.assertEqual(storage_bytes(artifact["pristine_inputs"]["k"]), pristine_bits)
        self.assertEqual(artifact["inputs_at_failure"]["k"][6, 1, 1].item(), -4)
        self.assertFalse(artifact["input_comparison"]["all_input_backing_bytes_unchanged"])

    def test_decode_error_and_forced_control_each_capture_full_outputs_with_explicit_labels(self):
        for forced, corrupt in ((False, True), (True, False), (True, True)):
            with self.subTest(forced=forced, corrupt=corrupt):
                bundle = self.bundle(forced=forced, corrupt=corrupt)
                with self.assertRaises(base.StageProbePositive):
                    self.fire(bundle)
                artifact, stage, pair = self.assert_pair(bundle, forced=forced, corrupt=corrupt)
                self.assert_full_saved_buffers(bundle, artifact)
                self.assertEqual("decode_error" in stage, corrupt)
                self.assertEqual(stage["force_positive_control"], forced)
                if forced:
                    self.assertIn("FORCED DIAGNOSTIC CONTROL", stage["scope"])
                self.assertTrue(artifact["failures"])
                self.assertTrue(all(item["iteration"] == 1 for item in artifact["failures"]))
                self.assertEqual({item["check"] for item in artifact["failures"]},
                                 {"forced_narrow_stage_control" if forced else "narrow_stage_positive"})
                self.assertEqual(pair["correlation"]["capture_kind"],
                                 "forced_diagnostic_control" if forced else "positive_instrumented_stage")
                self.assertEqual(bool(pair["correlation"]["decode_error"]), corrupt)

    def test_no_recapture_or_future_launch_after_first_positive_pair(self):
        bundle = self.bundle()
        with self.assertRaises(base.StageProbePositive):
            self.fire(bundle)
        paths = (bundle.controller.destination, bundle.args.failure_dump, bundle.controller.pair_path)
        hashes = [file_hash(path) for path in paths]
        with self.assertRaises(ValueError):
            bundle.controller.capture_positive_outputs()
        with self.assertRaises(ValueError):
            bundle.controller.launch(lambda **_: self.fail("post-positive launch"), bundle.kwargs, grid=(4, 1))
        self.assertEqual(bundle.controller.calls, 1)
        self.assertEqual([file_hash(path) for path in paths], hashes)

    def prepare_v1_positive(self, bundle):
        # Establish the actual launch-view verification while delaying only
        # the new host saver, so metadata guards see genuine v1 stage bytes.
        with patch.object(bundle.controller, "capture_positive_outputs", return_value=None):
            with self.assertRaises(base.StageProbePositive):
                bundle.controller.launch(lambda **actual: actual["LOCK"].copy_(bundle.raw),
                                         bundle.kwargs, grid=(4, 1))
        self.assertFalse(bundle.args.failure_dump.exists())
        self.assertIsNone(bundle.controller.positive_output_capture)

    def test_positive_capture_rejects_stale_iteration_or_nonfirst_capture(self):
        for case in ("no_launch", "newer_calls", "completed_iteration", "summary_iteration",
                     "payload_iteration", "previous_raw_failure"):
            with self.subTest(case=case):
                bundle = self.bundle(save=lambda *_a, **_k: self.fail("invalid pair was saved"))
                self.prepare_v1_positive(bundle)
                if case == "no_launch":
                    bundle.controller.calls = 0
                elif case == "newer_calls":
                    bundle.controller.calls = 2
                    bundle.report["iterations_completed"] = 1
                elif case == "completed_iteration":
                    bundle.report["iterations_completed"] = 1
                elif case == "summary_iteration":
                    bundle.controller.positive["iteration"] = 2
                elif case == "payload_iteration":
                    payload = torch.load(bundle.controller.destination, weights_only=False)
                    payload["iteration"] = 2
                    torch.save(payload, bundle.controller.destination)
                else:
                    bundle.report["first_failure_dump"] = {"iteration": 1, "path": "previous.pt"}
                before = bundle.controller.destination.read_bytes()
                with self.assertRaises(ValueError):
                    bundle.controller.capture_positive_outputs()
                self.assertEqual(bundle.controller.destination.read_bytes(), before)
                self.assertFalse(bundle.args.failure_dump.exists())

    def test_all_kernel_tensor_buffer_mismatches_reject_before_any_launch(self):
        for argument in ARGUMENTS:
            with self.subTest(argument=argument):
                bundle = self.bundle()
                wrong = dict(bundle.kwargs)
                wrong[argument] = wrong[argument].clone()
                with self.assertRaises(ValueError):
                    bundle.controller.launch(lambda **_: self.fail("mismatched launch"), wrong, grid=(4, 1))
                self.assertEqual(bundle.controller.calls, 0)
                self.assertIsNone(bundle.controller.workspace)
                self.assertFalse(bundle.controller.destination.exists())

    def test_same_storage_changed_view_and_changed_caller_mapping_reject_before_launch(self):
        for case in ("stride", "offset", "missing_argument", "caller_input", "caller_output", "unbound"):
            with self.subTest(case=case):
                bundle = self.bundle(bind=case != "unbound")
                actual = dict(bundle.kwargs)
                if case == "stride":
                    actual["K"] = actual["K"].transpose(1, 2)
                    self.assertEqual(actual["K"].data_ptr(), bundle.inputs["k"].data_ptr())
                elif case == "offset":
                    q = actual["Q"]
                    actual["Q"] = q.as_strided(q.shape, q.stride(), q.storage_offset() - 4)
                    self.assertEqual(actual["Q"].untyped_storage()._cdata, q.untyped_storage()._cdata)
                elif case == "missing_argument":
                    actual.pop("DV")
                elif case == "caller_input":
                    bundle.inputs["k"] = bundle.inputs["k"].clone()
                elif case == "caller_output":
                    bundle.outputs["dq"] = bundle.outputs["dq"].clone()
                with self.assertRaises(ValueError):
                    bundle.controller.launch(lambda **_: self.fail("unbound/mismatched launch"), actual, grid=(4, 1))
                self.assertEqual(bundle.controller.calls, 0)
                self.assertFalse(bundle.controller.destination.exists())

    def test_raw_saver_error_leaves_immutable_stage_and_failed_pair_and_forbids_next_launch(self):
        captured = []
        bundle = None

        def fail(*_args, **_kwargs):
            captured.append(bundle.controller.destination.read_bytes())
            self.assertEqual(json.loads(bundle.controller.pair_path.read_text())["raw_dump_status"], "pending")
            raise OSError("raw dump unavailable")

        bundle = self.bundle(save=fail)
        with self.assertRaisesRegex(OSError, "raw dump unavailable"):
            self.fire(bundle)
        self.assertEqual(bundle.controller.destination.read_bytes(), captured[0])
        self.assertFalse(bundle.args.failure_dump.exists())
        pair = json.loads(bundle.controller.pair_path.read_text())
        self.assertEqual(pair["raw_dump_status"], "failed")
        self.assertIn("raw dump unavailable", str(pair))
        self.assertEqual(bundle.controller.positive_output_capture["raw_dump_status"], "failed")
        with self.assertRaises(ValueError):
            bundle.controller.launch(lambda **_: self.fail("post-error launch"), bundle.kwargs, grid=(4, 1))

    def test_pair_creation_error_after_stage_save_still_forbids_a_later_launch(self):
        bundle = self.bundle()
        # Simulate a path appearing after validation. Its bytes must not be
        # replaced, and the newly saved v1 stage must remain the first one.
        bundle.controller.pair_path.write_text('{"owner":"existing"}\n')
        pair_before = bundle.controller.pair_path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.fire(bundle)
        stage_before = bundle.controller.destination.read_bytes()
        self.assertIsNotNone(bundle.controller.positive)
        self.assertIsNone(bundle.controller.positive_output_capture)
        self.assertFalse(bundle.args.failure_dump.exists())
        with self.assertRaises(ValueError):
            bundle.controller.launch(lambda **_: self.fail("launch after saved positive stage"),
                                     bundle.kwargs, grid=(4, 1))
        self.assertEqual(bundle.controller.calls, 1)
        self.assertEqual(bundle.controller.destination.read_bytes(), stage_before)
        self.assertEqual(bundle.controller.pair_path.read_bytes(), pair_before)

    def test_raw_saver_wrong_iteration_or_path_cannot_mark_pair_complete(self):
        original_save = runner.save_failure_dump
        for case in ("iteration", "path"):
            with self.subTest(case=case):
                def wrong(path, pristine, inputs, outputs, **kwargs):
                    result = original_save(path, pristine, inputs, outputs, **kwargs)
                    if case == "iteration":
                        result["iteration"] += 1
                    else:
                        result["path"] = str(Path(path).with_name("different.pt"))
                    return result

                bundle = self.bundle(save=wrong)
                with self.assertRaisesRegex(ValueError, "iteration or path"):
                    self.fire(bundle)
                pair = json.loads(bundle.controller.pair_path.read_text())
                self.assertEqual(pair["raw_dump_status"], "failed")
                self.assertNotIn("raw_file_sha256", pair)
                self.assertEqual(pair["correlation"]["workspace_file_sha256"], file_hash(bundle.controller.destination))
                self.assertEqual(torch.load(bundle.args.failure_dump, weights_only=False)["iteration"], 1)
                with self.assertRaises(ValueError):
                    bundle.controller.launch(lambda **_: self.fail("post-invalid-result launch"),
                                             bundle.kwargs, grid=(4, 1))

    def run_hooked_loop(self, *, positive):
        original_factory, original_save, original_loop = base.NarrowStageController, runner.save_failure_dump, runner.run_loop
        launches = []
        with v3.capture_hooks():
            bundle = self.bundle(controller_factory=base.NarrowStageController, bind=False)
            self.assertIsInstance(bundle.controller, v3.NarrowStageControllerV3)
            self.assertIsNot(runner.save_failure_dump, original_save)
            self.assertIsNot(runner.run_loop, original_loop)
            original_pre_hook = bundle.config.pre_hook

            def attention_backward(**values):
                bundle.config.pre_hook({"DQ": values["dq"], "SEQUENCE_PARALLEL": False})
                kwargs = {argument: values[name] for argument, name in ARGUMENTS.items()}
                kwargs.update(Z=2, H=2, DimQ=2)

                def kernel(**actual):
                    launches.append(bundle.controller.calls)
                    self.assertLessEqual(len(launches), 2, "third launch would replace first-failure state")
                    for output in bundle.outputs.values():
                        output.zero_()
                    actual["LOCK"][-1] = 100 + len(launches)
                    if len(launches) == 2:
                        if positive:
                            self.fill_outputs(bundle)
                            actual["LOCK"].copy_(bundle.raw)
                        else:
                            bundle.outputs["dq"][1, 0, 1] = 1.3828125

                return bundle.controller.launch(kernel, kwargs, grid=(4, 1))

            attention = SimpleNamespace(triton_hstu_attention_bwd=attention_backward,
                                        _hstu_attn_bwd=SimpleNamespace(device_caches={}))
            bundle.controller.attention = attention
            call = lambda: runner.run_loop(bundle.args, bundle.report, lambda _event: None, attention,
                bundle.config, bundle.inputs, bundle.outputs, bundle.source["configuration"]["context"],
                bundle.history, bundle.pristine, bundle.digests, chunk_bytes=4096)
            if positive:
                with self.assertRaises(base.StageProbePositive):
                    call()
            else:
                self.assertEqual(call(), 1)
            self.assertIs(bundle.config.pre_hook, original_pre_hook)
        self.assertIs(base.NarrowStageController, original_factory)
        self.assertIs(runner.save_failure_dump, original_save)
        self.assertIs(runner.run_loop, original_loop)
        self.assertEqual(launches, [1, 2])
        return bundle

    def test_real_raw_loop_auto_binds_context_and_saves_second_positive_before_standard_checks(self):
        bundle = self.run_hooked_loop(positive=True)
        artifact, stage, _ = self.assert_pair(bundle, expected_iteration=2)
        self.assert_full_saved_buffers(bundle, artifact)
        self.assertEqual(bundle.report["iterations_completed"], 1)
        self.assertEqual([check["status"] for check in bundle.report["checks"]], ["PASS"])
        self.assertEqual(stage["iteration"], 2)
        self.assertEqual(bundle.controller.calls, 2)
        with self.assertRaises(ValueError):
            bundle.controller.launch(lambda **_: self.fail("third launch"), bundle.kwargs, grid=(4, 1))

    def test_real_raw_failure_keeps_v2_workspace_and_raw_dump_behavior(self):
        bundle = self.run_hooked_loop(positive=False)
        raw_saved = torch.load(bundle.args.failure_dump, weights_only=False)
        stage_saved = torch.load(bundle.controller.destination, weights_only=False)
        self.assertEqual(stage_saved["format"], "hstu_dq_narrow_raw_failure_workspace_v2")
        self.assertEqual(stage_saved["iteration"], 2)
        self.assertEqual(raw_saved["iteration"], 2)
        self.assertEqual(stage_saved["workspace_raw_int32"][-1].item(), 102)
        self.assertEqual(raw_saved["outputs"]["dq"][1, 0, 1].item(), 1.3828125)
        self.assertEqual(raw_saved["configuration"]["narrow_stage_workspace_capture"], stage_saved["correlation"])
        self.assertEqual(bundle.controller.raw_failure_capture["raw_failure_dump_status"], "complete")
        self.assertIsNone(bundle.controller.positive_output_capture)
        self.assertFalse(bundle.controller.pair_path.exists())
        self.assertEqual([check["status"] for check in bundle.report["checks"]], ["PASS", "FAIL"])
        self.assertFalse(stage_saved["workspace_scan"]["all_workspace_words_zero"])

    def test_hooks_restore_factory_saver_and_loop_on_external_exception(self):
        originals = (base.NarrowStageController, runner.save_failure_dump, runner.run_loop)
        with self.assertRaisesRegex(RuntimeError, "external stop"):
            with v3.capture_hooks():
                self.assertIsNot(base.NarrowStageController, originals[0])
                self.assertIsNot(runner.save_failure_dump, originals[1])
                self.assertIsNot(runner.run_loop, originals[2])
                raise RuntimeError("external stop")
        self.assertIs(base.NarrowStageController, originals[0])
        self.assertIs(runner.save_failure_dump, originals[1])
        self.assertIs(runner.run_loop, originals[2])

    def cli(self, stage_path, raw_path, *extra, capture_path=None):
        return ["--stage", "old_dq", "--stage-dump", str(stage_path), "--",
                str(capture_path or self.directory / "source.pt"), "--dout-zero",
                "--failure-dump", str(raw_path), *extra]

    def test_cli_requires_new_pair_path_distinct_from_capture_raw_and_report_and_keeps_v2_guards(self):
        stage, raw = self.directory / "stage.pt", self.directory / "raw.pt"
        pair = Path(str(stage) + ".pair.json")
        options, raw_args = v3.arguments(self.cli(stage, raw))
        self.assertEqual(options.stage_dump, stage)
        self.assertEqual(raw_args.failure_dump, raw)
        for argv in (self.cli(stage, pair), self.cli(stage, raw, "--report", str(pair)),
                     self.cli(stage, raw, capture_path=pair), self.cli(stage, raw, "--keep-g")):
            with self.subTest(argv=argv):
                with self.assertRaises((ValueError, FileExistsError, SystemExit)):
                    v3.arguments(argv)
        pair.write_text('{"existing":"do not change"}\n')
        before = pair.read_bytes()
        with self.assertRaises((ValueError, FileExistsError, SystemExit)):
            v3.arguments(self.cli(stage, raw))
        self.assertEqual(pair.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
