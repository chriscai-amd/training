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

import probe_attention_wmma_all40 as full
from probe_attention_code_object import digest, public_plan
from test_probe_attention_code_object import FakeKernel


def preparation(root):
    record = {
        "format": "attention_post_wmma_all40_waits_preparation_v1",
        "status": "CPU_PREPARED_AND_ELF_VERIFIED_GPU_NOT_RUN",
        "unmodified_roundtrip_hsaco_byte_identical": True,
        "original_wmma_count": 40, "selected_original_assembly_lines": list(full.SELECTED_LINES),
        "split_delay_original_lines": [1579, 1584, 1589],
        "split_second_consumer_original_lines": [1582, 1587, 1592],
        "stronger_wait_original_lines": [2766, 2776],
        "stronger_wait_instruction": "s_wait_alu depctr_va_vdst(0)",
        "skeleton_added_wait_instructions": 3, "skeleton_replaced_wait_instructions": 2,
        "reversing_declared_skeleton_changes_recovers_baseline": True,
        "requires_positive_skeleton_control": True,
        "base_kernel_hash": "key", "base_hsaco_sha256": digest(b"original"),
        "base_assembly_sha256": digest(b"original assembly\n"), "arms": [],
    }
    (root / "baseline.hsaco").write_bytes(b"original")
    (root / "baseline.amdgcn").write_bytes(b"original assembly\n")
    for arm, opcode in (("skeleton", None), ("v5", "v_nop"), ("s5", "s_nop 0")):
        binary, assembly = arm.encode(), (arm + " assembly\n").encode()
        (root / (arm + ".hsaco")).write_bytes(binary)
        (root / (arm + ".amdgcn")).write_bytes(assembly)
        record["arms"].append({"name": arm, "hsaco_sha256": digest(binary),
                               "assembly_sha256": digest(assembly), "padding_instruction": opcode,
                               "padding_per_site": 0 if arm == "skeleton" else 5,
                               "injected_padding_instructions": 0 if arm == "skeleton" else 200,
                               "removing_marked_padding_recovers_skeleton": True})
    (root / "preparation.json").write_text(json.dumps(record))
    return record


class WMMAFullTest(unittest.TestCase):
    def test_every_arm_installs_exact_selected_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preparation(root)
            for arm in ("baseline", "skeleton", "v5", "s5"):
                with self.subTest(arm=arm):
                    plan = full.prepared_plan(root, arm)
                    with full.shared.install_code_object(FakeKernel, plan) as state:
                        kernel = FakeKernel()
                        kernel.run("one")
                        kernel.run("two")
                        self.assertEqual(kernel.loaded, [(root / (arm + ".hsaco")).read_bytes()])
                        self.assertEqual(state["loaded_instances"], 1)
                    self.assertEqual(plan["skeleton_hsaco_sha256"], digest(b"skeleton"))
                    self.assertTrue(plan["requires_positive_skeleton_control"])
                    self.assertNotIn("binary", public_plan(plan))
                    self.assertNotIn("assembly", public_plan(plan))
                    json.dumps(public_plan(plan))

    def test_skeleton_tampering_rejected_for_every_arm(self):
        for arm in ("baseline", "skeleton", "v5", "s5"):
            for extension in ("hsaco", "amdgcn"):
                with self.subTest(arm=arm, extension=extension), TemporaryDirectory() as directory:
                    root = Path(directory)
                    preparation(root)
                    (root / ("skeleton." + extension)).write_bytes(b"changed")
                    with self.assertRaises(ValueError):
                        full.prepared_plan(root, arm)

    def test_baseline_and_selected_bytes_are_hashed(self):
        for filename in ("baseline.hsaco", "baseline.amdgcn", "v5.hsaco", "v5.amdgcn"):
            with self.subTest(filename=filename), TemporaryDirectory() as directory:
                root = Path(directory)
                preparation(root)
                (root / filename).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    full.prepared_plan(root, "v5")

    def test_wrong_wait_skeleton_or_subset_scope_rejected(self):
        changes = (("format", "attention_post_wmma_subset35_preparation_v1"),
                   ("selected_original_assembly_lines", list(full.SELECTED_LINES[:-1])),
                   ("stronger_wait_instruction", "s_wait_alu depctr_va_vdst(1)"),
                   ("stronger_wait_original_lines", [2766]),
                   ("split_delay_original_lines", []),
                   ("split_second_consumer_original_lines", [1580, 1585, 1590]),
                   ("skeleton_added_wait_instructions", 2),
                   ("requires_positive_skeleton_control", False))
        for key, value in changes:
            with self.subTest(key=key), TemporaryDirectory() as directory:
                root = Path(directory)
                record = preparation(root)
                record[key] = value
                (root / "preparation.json").write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    full.prepared_plan(root, "v5")

    def test_missing_or_mislabeled_control_rejected(self):
        for change in ("missing", "padded_skeleton", "wrong_vector_count", "duplicate"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                record = preparation(root)
                if change == "missing":
                    record["arms"].pop(0)
                elif change == "padded_skeleton":
                    record["arms"][0]["padding_instruction"] = "s_nop 0"
                elif change == "wrong_vector_count":
                    record["arms"][1]["injected_padding_instructions"] = 175
                else:
                    record["arms"][2] = dict(record["arms"][1])
                (root / "preparation.json").write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    full.prepared_plan(root, "v5")

    def test_invalid_cli_precedes_runtime_import(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preparation(root)
            capture = root / "capture.pt"
            capture.write_bytes(b"fixture")
            prefix = ["--prepared", str(root), "--arm", "skeleton", "--", str(capture)]
            original_import = builtins.__import__

            def no_runtime(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch.") or name == "triton" or name.startswith("triton."):
                    raise AssertionError("GPU runtime imported for invalid arguments")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=no_runtime):
                with redirect_stderr(io.StringIO()), self.assertRaises((ValueError, SystemExit)):
                    full.main(prefix)

    def test_skeleton_cli_uses_verified_hook_and_records_provenance(self):
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
            runner.run_loop = lambda args, report, publish: 19

            def run(args, report, publish):
                kernel = FakeKernel()
                kernel.run("replay")
                self.assertEqual(kernel.loaded, [b"skeleton"])
                return runner.run_loop(args, report, publish)

            runner.run = run
            runner.main = lambda: runner.run(None, report, None)
            before = runner.run, runner.run_loop, sys.argv
            compiler = ModuleType("triton.compiler.compiler")
            compiler.CompiledKernel = FakeKernel
            with patch.dict(sys.modules, {"probe_attention_raw_failure": runner,
                                          "triton.compiler.compiler": compiler}):
                result = full.main(["--prepared", str(root), "--arm", "skeleton", "--", str(capture)])
            self.assertEqual(result, 19)
            self.assertEqual((runner.run, runner.run_loop, sys.argv), before)
            self.assertEqual(report["post_assembly_override"]["loaded_instances"], 1)
            plan = report["configuration"]["post_assembly_override"]
            self.assertEqual(plan["arm"], "skeleton")
            self.assertEqual(plan["experiment"], "wmma_all40_waits_v1")
            self.assertEqual(len(report["configuration"]["source_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
