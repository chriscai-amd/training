import builtins
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import probe_attention_wmma_subset as subset
from probe_attention_code_object import digest, public_plan
from test_probe_attention_code_object import FakeKernel


def preparation(root):
    record = {
        "format": "attention_post_wmma_subset35_preparation_v1",
        "status": "CPU_PREPARED_AND_ELF_VERIFIED_GPU_NOT_RUN",
        "unmodified_roundtrip_hsaco_byte_identical": True,
        "original_delay_instructions_unchanged": True,
        "original_wmma_count": 40, "selected_wmma_count": 35,
        "selected_original_assembly_lines": list(subset.SELECTED_LINES),
        "excluded_original_assembly_lines": list(subset.EXCLUDED_LINES),
        "base_kernel_hash": "key", "base_hsaco_sha256": digest(b"original"),
        "base_assembly_sha256": digest(b"original assembly\n"), "arms": [],
    }
    (root / "baseline.hsaco").write_bytes(b"original")
    (root / "baseline.amdgcn").write_bytes(b"original assembly\n")
    for arm, opcode in (("v5", "v_nop"), ("s5", "s_nop 0")):
        binary, assembly = arm.encode(), (arm + " assembly\n").encode()
        (root / (arm + ".hsaco")).write_bytes(binary)
        (root / (arm + ".amdgcn")).write_bytes(assembly)
        record["arms"].append({"name": arm, "hsaco_sha256": digest(binary),
                               "assembly_sha256": digest(assembly), "padding_instruction": opcode,
                               "padding_per_site": 5, "injected_instructions": 175,
                               "removing_marked_padding_recovers_baseline": True})
    (root / "preparation.json").write_text(json.dumps(record))
    return record


class WMMASubsetTest(unittest.TestCase):
    def test_all_arms_use_selected_bytes_with_both_loader_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preparation(root)
            for arm in ("baseline", "v5", "s5"):
                with self.subTest(arm=arm):
                    plan = subset.prepared_plan(root, arm)
                    self.assertEqual(plan["binary"], (root / (arm + ".hsaco")).read_bytes())
                    self.assertEqual(plan["assembly"], (root / (arm + ".amdgcn")).read_text())
                    self.assertEqual(plan["selected_original_assembly_lines"], list(subset.SELECTED_LINES))
                    for prefix in ("loader", "shared_loader"):
                        self.assertEqual(plan[prefix + "_sha256"], digest(Path(plan[prefix + "_path"]).read_bytes()))
                    self.assertNotIn("binary", public_plan(plan))
                    self.assertNotIn("assembly", public_plan(plan))
                    json.dumps(public_plan(plan))

    def test_unsupported_scope_or_missing_preparation_proof_rejected(self):
        changes = (("format", "attention_post_msb_nop_preparation_v1"),
                   ("status", "partial"), ("unmodified_roundtrip_hsaco_byte_identical", False),
                   ("original_delay_instructions_unchanged", False), ("original_wmma_count", 39),
                   ("selected_wmma_count", 40), ("selected_original_assembly_lines", []),
                   ("excluded_original_assembly_lines", []))
        for key, value in changes:
            with self.subTest(key=key), TemporaryDirectory() as directory:
                root = Path(directory)
                record = preparation(root)
                record[key] = value
                (root / "preparation.json").write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    subset.prepared_plan(root, "v5")

    def test_unmatched_or_mislabeled_padding_rejected(self):
        for change in ("duplicate", "count", "opcode", "padding", "recovery"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                record = preparation(root)
                if change == "duplicate":
                    record["arms"][1] = dict(record["arms"][0])
                else:
                    key, value = {"count": ("injected_instructions", 200),
                                  "opcode": ("padding_instruction", "s_nop 7"),
                                  "padding": ("padding_per_site", 4),
                                  "recovery": ("removing_marked_padding_recovers_baseline", False)}[change]
                    record["arms"][0][key] = value
                (root / "preparation.json").write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    subset.prepared_plan(root, "v5")

    def test_baseline_and_selected_object_or_assembly_tampering_rejected(self):
        for arm in ("v5", "s5"):
            for name in ("baseline", arm):
                for extension in ("hsaco", "amdgcn"):
                    with self.subTest(arm=arm, name=name, extension=extension), TemporaryDirectory() as directory:
                        root = Path(directory)
                        preparation(root)
                        (root / (name + "." + extension)).write_bytes(b"tampered")
                        with self.assertRaises(ValueError):
                            subset.prepared_plan(root, arm)

    def test_old_arm_name_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preparation(root)
            with self.assertRaises(ValueError):
                subset.prepared_plan(root, "nop0")

    def test_real_plan_reaches_shared_installation_hooks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preparation(root)
            for arm in ("baseline", "v5", "s5"):
                with self.subTest(arm=arm):
                    plan = subset.prepared_plan(root, arm)
                    with subset.shared.install_code_object(FakeKernel, plan) as state:
                        kernel = FakeKernel()
                        kernel.run("one")
                        kernel.run("two")
                        self.assertEqual(kernel.loaded, [plan["binary"]])
                        self.assertEqual(kernel.asm["amdgcn"], plan["assembly"])
                        self.assertEqual(state["loaded_instances"], 1)

    def test_invalid_cli_is_rejected_before_gpu_runtime_import(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preparation(root)
            capture = root / "capture.pt"
            capture.write_bytes(b"fixture")
            prefix = ["--prepared", str(root), "--arm", "v5", "--", str(capture)]
            required = ["--report", str(root / "new.json"), "--failure-dump", str(root / "new.pt")]
            original_import = builtins.__import__

            def no_runtime(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch.") or name == "triton" or name.startswith("triton."):
                    raise AssertionError("GPU runtime imported during invalid argument validation")
                return original_import(name, *args, **kwargs)

            for arguments in (prefix, prefix + required + ["--keep-going"],
                              prefix + required + ["--validate-only"],
                              prefix + required + ["--kernel-variant", "dot_only"]):
                with self.subTest(arguments=arguments), patch("builtins.__import__", side_effect=no_runtime):
                    with redirect_stderr(io.StringIO()), self.assertRaises((ValueError, SystemExit)):
                        subset.main(arguments)

    def test_runner_bridge_records_wrapper_and_shared_hook_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preparation(root)
            capture = root / "capture.pt"
            capture.write_bytes(b"fixture")
            report = {"configuration": {"source_sha256": {}}}
            runner = ModuleType("probe_attention_raw_failure")
            runner.__file__ = "fake_raw_runner.py"
            runner.arguments = lambda raw: SimpleNamespace(
                keep_going=False, validate_only=False, kernel_variant="production",
                capture=capture, report=root / "report.json", failure_dump=root / "failure.pt")
            runner.run_loop = lambda args, report, publish: 17

            def run(args, report, publish):
                kernel = FakeKernel()
                kernel.run("replay")
                self.assertEqual(kernel.loaded, [b"v5"])
                return runner.run_loop(args, report, publish)

            runner.run = run
            runner.main = lambda: runner.run(None, report, None)
            before = runner.run, runner.run_loop, sys.argv
            compiler = ModuleType("triton.compiler.compiler")
            compiler.CompiledKernel = FakeKernel
            with patch.dict(sys.modules, {"probe_attention_raw_failure": runner,
                                          "triton.compiler.compiler": compiler}):
                result = subset.main(["--prepared", str(root), "--arm", "v5", "--", str(capture)])
            self.assertEqual(result, 17)
            self.assertEqual((runner.run, runner.run_loop, sys.argv), before)
            state = report["post_assembly_override"]
            self.assertEqual(state["loaded_instances"], 1)
            config = report["configuration"]
            self.assertEqual(config["post_assembly_override"]["experiment"], "wmma_subset35_v1")
            self.assertEqual(len(config["source_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
