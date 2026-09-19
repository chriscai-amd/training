import builtins
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from probe_attention_code_object import digest, install_code_object, main, prepared_plan, public_plan


class FakeKernel:
    def __init__(self, name="_hstu_attn_bwd", key="key", binary=b"original",
                 assembly="original assembly\n", initialized=False):
        self.name, self.hash, self.kernel = name, key, binary
        self.asm = {"hsaco": binary, "amdgcn": assembly}
        self.module = self.function = self._run = self._module_pid = None
        self.loaded = []
        self.calls = []
        if initialized:
            self._init_handles()

    def _init_handles(self):
        if self.module is None:
            self.loaded.append(self.kernel)
            self.module, self.function, self._module_pid = object(), object(), 123
            self._run = lambda *args: self.calls.append(args)
            self.n_regs, self.n_spills, self.n_max_threads = 654, 0, 1024

    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run


def plan():
    return {"binary": b"modified", "assembly": "modified assembly\n", "source_kernel_name": "_hstu_attn_bwd",
            "source_kernel_hash": "key", "original_hsaco_sha256": digest(b"original"),
            "loaded_hsaco_sha256": digest(b"modified"),
            "original_assembly_sha256": digest(b"original assembly\n"),
            "loaded_assembly_sha256": digest(b"modified assembly\n")}


def preparation(root):
    (root / "baseline.hsaco").write_bytes(b"original")
    (root / "nop7.hsaco").write_bytes(b"modified")
    (root / "baseline.amdgcn").write_text("original assembly\n")
    (root / "nop7.amdgcn").write_text("modified assembly\n")
    record = {"format": "attention_post_msb_nop_preparation_v1",
              "unmodified_roundtrip_hsaco_byte_identical": True,
              "base_hsaco_sha256": digest(b"original"), "base_kernel_hash": "key",
              "base_assembly_sha256": digest(b"original assembly\n"),
              "arms": [{"name": "nop7", "hsaco_sha256": digest(b"modified"),
                        "assembly_sha256": digest(b"modified assembly\n")}]}
    (root / "preparation.json").write_text(json.dumps(record))
    return record


class CodeObjectTest(unittest.TestCase):
    def test_only_modified_bytes_reach_loader_and_repeated_load_not_recounted(self):
        original_methods = FakeKernel.__init__, FakeKernel._init_handles, FakeKernel.run
        events = []
        with install_code_object(FakeKernel, plan(), lambda event: events.append(event["event"])) as state:
            k = FakeKernel()
            self.assertIsNone(k.module)
            launcher = k.run
            self.assertIs(k.run, launcher)
            k.run("first")
            k.run("second")
            k._init_handles()
            self.assertEqual(k.loaded, [b"modified"])
            self.assertEqual(k.calls, [("first",), ("second",)])
            self.assertEqual(k.asm["amdgcn"], "modified assembly\n")
            self.assertEqual(state["installed_instances"], 1)
            self.assertEqual(state["loaded_instances"], 1)
            self.assertEqual(events, ["attention_code_object_installed_before_load", "attention_code_object_loaded"])
            self.assertNotIn("binary", state)
            self.assertNotIn("assembly", state)
            json.dumps(state)
        self.assertEqual((FakeKernel.__init__, FakeKernel._init_handles, FakeKernel.run), original_methods)

    def test_other_kernels_unchanged(self):
        with install_code_object(FakeKernel, plan()) as state:
            k = FakeKernel(name="unrelated", binary=b"other")
            k.run("unrelated")
            self.assertEqual(k.loaded, [b"other"])
            self.assertEqual(k.calls, [("unrelated",)])
            self.assertEqual(state["installed_instances"], 0)

    def test_mismatched_source_or_original_binary_rejected(self):
        with install_code_object(FakeKernel, plan()):
            for kwargs in ({"key": "other"}, {"binary": b"other"},
                           {"assembly": "other"}, {"initialized": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    FakeKernel(**kwargs)

    def test_unintercepted_cached_instance_rejected(self):
        stale = FakeKernel()
        with install_code_object(FakeKernel, plan()), self.assertRaises(ValueError):
            stale._init_handles()
        self.assertEqual(stale.loaded, [])

    def test_already_loaded_cached_run_cannot_bypass_validation(self):
        stale = FakeKernel(initialized=True)
        with install_code_object(FakeKernel, plan()), self.assertRaises(ValueError):
            stale.run("must not execute")
        self.assertEqual(stale.calls, [])
        self.assertEqual(stale.loaded, [b"original"])

    def test_earlier_context_loaded_instance_rejected_even_with_same_plan(self):
        shared = plan()
        with install_code_object(FakeKernel, shared):
            stale = FakeKernel()
            stale.run("first context")
        with install_code_object(FakeKernel, shared), self.assertRaises(ValueError):
            stale.run("must not execute")
        self.assertEqual(stale.calls, [("first context",)])

    def test_caller_cannot_change_active_plan(self):
        shared = plan()
        with install_code_object(FakeKernel, shared):
            shared["binary"] = b"changed"
            shared["assembly"] = "changed"
            k = FakeKernel()
            k.run()
            self.assertEqual(k.loaded, [b"modified"])

    def test_identity_binary_and_assembly_changes_rejected_before_and_after_load(self):
        for loaded in (False, True):
            for field in ("name", "hash", "kernel", "hsaco", "amdgcn"):
                with self.subTest(loaded=loaded, field=field), install_code_object(FakeKernel, plan()):
                    k = FakeKernel()
                    if loaded:
                        k._init_handles()
                    if field in ("hsaco", "amdgcn"):
                        k.asm[field] = "bad" if field == "amdgcn" else b"bad"
                    else:
                        setattr(k, field, b"bad" if field == "kernel" else "bad")
                    with self.assertRaises(ValueError):
                        k.run("must not execute")
                    with self.assertRaises(ValueError):
                        k._init_handles()
                    self.assertEqual(k.calls, [])
                    self.assertEqual(k.loaded, [b"modified"] if loaded else [])

    def test_unobserved_or_changed_runtime_handles_rejected(self):
        original_handles = FakeKernel._init_handles
        for observed in (False, True):
            for field in ("module", "function", "_run", "_module_pid"):
                with self.subTest(observed=observed, field=field), install_code_object(FakeKernel, plan()):
                    k = FakeKernel()
                    if observed:
                        k._init_handles()
                    else:
                        original_handles(k)
                    setattr(k, field, (lambda: None) if field == "_run" else object())
                    with self.assertRaises(ValueError):
                        k.run("must not execute")
                    self.assertEqual(k.calls, [])

    def test_external_load_with_unchanged_bytes_still_rejected(self):
        original_handles = FakeKernel._init_handles
        with install_code_object(FakeKernel, plan()):
            k = FakeKernel()
            original_handles(k)
            with self.assertRaises(ValueError):
                k.run("must not execute")

    def test_incomplete_runtime_load_is_rejected(self):
        class IncompleteKernel(FakeKernel):
            def _init_handles(self):
                self.module, self.function = object(), object()

        with install_code_object(IncompleteKernel, plan()) as state:
            with self.assertRaises(ValueError):
                IncompleteKernel().run()
            self.assertEqual(state["loaded_instances"], 0)

    def test_methods_restored_on_error(self):
        before = FakeKernel.__init__, FakeKernel._init_handles, FakeKernel.run
        with self.assertRaises(RuntimeError):
            with install_code_object(FakeKernel, plan()):
                raise RuntimeError("stop")
        self.assertEqual((FakeKernel.__init__, FakeKernel._init_handles, FakeKernel.run), before)

    def test_hooks_restored_when_loader_raises(self):
        class FailedKernel(FakeKernel):
            def _init_handles(self):
                self._run = lambda: None
                raise RuntimeError("runtime failed")

        before = FailedKernel.__init__, FailedKernel._init_handles, FailedKernel.run
        with self.assertRaisesRegex(RuntimeError, "runtime failed"):
            with install_code_object(FailedKernel, plan()) as state:
                FailedKernel().run()
        self.assertEqual(state["loaded_instances"], 0)
        self.assertEqual((FailedKernel.__init__, FailedKernel._init_handles, FailedKernel.run), before)

    def test_manifest_validation_and_binary_hashes(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            record = preparation(root)
            path = root / "preparation.json"
            path.write_text(json.dumps(record))
            self.assertEqual(prepared_plan(root, "nop7")["binary"], b"modified")
            self.assertEqual(prepared_plan(root, "baseline")["binary"], b"original")
            result = prepared_plan(root, "nop7")
            self.assertEqual(result["assembly"], "modified assembly\n")
            self.assertEqual(result["original_assembly_path"], str(root / "baseline.amdgcn"))
            self.assertEqual(result["loaded_hsaco_path"], str(root / "nop7.hsaco"))
            self.assertEqual(result["loader_sha256"], digest(Path(result["loader_path"]).read_bytes()))
            self.assertEqual(result["preparation_sha256"], digest(path.read_bytes()))
            self.assertNotIn("binary", public_plan(result))
            self.assertNotIn("assembly", public_plan(result))
            (root / "nop7.hsaco").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                prepared_plan(root, "nop7")
            with self.assertRaises(ValueError):
                prepared_plan(root, "unknown")
            record["unmodified_roundtrip_hsaco_byte_identical"] = False
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                prepared_plan(root, "baseline")

    def test_each_frozen_binary_and_assembly_is_verified(self):
        for filename in ("baseline.hsaco", "nop7.hsaco", "baseline.amdgcn", "nop7.amdgcn"):
            with self.subTest(filename=filename), TemporaryDirectory() as name:
                root = Path(name)
                preparation(root)
                (root / filename).write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    prepared_plan(root, "nop7")

    def test_preparation_hash_uses_the_single_parsed_read(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            preparation(root)
            path = root / "preparation.json"
            original_bytes = path.read_bytes()
            original_read = Path.read_bytes
            reads = []

            def changing_read(current):
                data = original_read(current)
                if current == path:
                    reads.append(current)
                    current.write_text("changed after the first read")
                return data

            with patch.object(Path, "read_bytes", changing_read):
                result = prepared_plan(root, "nop7")
            self.assertEqual(reads, [path])
            self.assertEqual(result["preparation_sha256"], digest(original_bytes))

    def test_invalid_cli_fails_before_runtime_import(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            preparation(root)
            capture = root / "capture.pt"
            capture.write_bytes(b"trusted fixture path")
            existing = root / "existing.json"
            existing.write_text("existing evidence")
            prefix = ["--prepared", str(root), "--arm", "nop7", "--"]
            required = [str(capture), "--report", str(root / "new.json"),
                        "--failure-dump", str(root / "new.pt")]
            bad = [required + ["--keep-going"], required + ["--validate-only"],
                   required + ["--kernel-variant", "baseline"], required + ["--repeat", "0"],
                   [str(capture)], [str(capture), "--report", str(root / "new.json")],
                   [str(capture), "--failure-dump", str(root / "new.pt")],
                   [str(root / "missing.pt"), *required[1:]], [str(root), *required[1:]],
                   required + ["--report", str(capture)], required + ["--report", str(existing)],
                   required + ["--failure-dump", str(root / "new.json")]]
            original_import = builtins.__import__

            def no_runtime(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch.") or name == "triton" or name.startswith("triton."):
                    raise AssertionError(f"Runtime import before validation: {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", no_runtime), redirect_stderr(io.StringIO()):
                for args in bad:
                    with self.subTest(args=args), self.assertRaises((ValueError, SystemExit)):
                        main(prefix + args)
                with self.assertRaises(FileNotFoundError):
                    main(["--prepared", str(root / "missing"), "--arm", "nop7", "--", *required])
                for args in ([], ["--prepared", str(root)], ["--prepared", str(root), "--arm", "bad"]):
                    with self.subTest(args=args), self.assertRaises(SystemExit):
                        main(args)

    def test_main_records_public_metadata_and_restores_runner(self):
        import probe_attention_raw_failure as runner

        with TemporaryDirectory() as name:
            root = Path(name)
            preparation(root)
            capture = root / "capture.pt"
            capture.write_bytes(b"trusted fixture path")
            module = ModuleType("triton.compiler.compiler")
            module.CompiledKernel = FakeKernel
            reports = []

            def raw_loop(args, report, publish):
                reports.append(report)
                FakeKernel().run()
                report["status"] = "PASS"
                return 0

            def raw_run(args, report, publish):
                report["configuration"] = {"source_sha256": {}}
                return runner.run_loop(args, report, publish)

            previous_argv = sys.argv
            with patch.dict(sys.modules, {"triton.compiler.compiler": module}), \
                    patch.object(runner, "run", raw_run), patch.object(runner, "run_loop", raw_loop), \
                    patch("builtins.print"):
                self.assertEqual(main(["--prepared", str(root), "--arm", "nop7", "--", str(capture),
                                       "--report", str(root / "report.json"),
                                       "--failure-dump", str(root / "failure.pt")]), 0)
                self.assertIs(runner.run, raw_run)
                self.assertIs(runner.run_loop, raw_loop)
            self.assertIs(sys.argv, previous_argv)
            config = reports[0]["configuration"]
            metadata = config["post_assembly_override"]
            self.assertNotIn("binary", metadata)
            self.assertNotIn("assembly", metadata)
            self.assertEqual(config["source_sha256"][metadata["loader_path"]], metadata["loader_sha256"])
            self.assertEqual(reports[0]["post_assembly_override"]["loaded_instances"], 1)
            json.dumps(reports[0])


if __name__ == "__main__":
    unittest.main()
