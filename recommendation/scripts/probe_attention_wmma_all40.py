#!/usr/bin/env python3
"""Replay full-WMMA padding over a separately prepared strengthened-wait skeleton.

Run the unpadded skeleton as a positive control before interpreting V5/S5.
All three arms split three merged waits and strengthen two VALU waits. V5/S5
then add five V_NOP/S_NOP0 instructions at every original WMMA instruction.
The original baseline is also available. Use a fresh process for each arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import probe_attention_code_object as shared

SELECTED_LINES = (1574, 1576, 1580, 1582, 1585, 1587, 1590, 1592, 2079, 2093,
                  2108, 2152, 2222, 2242, 2263, 2281, 2475, 2485, 2487, 2489,
                  2491, 2493, 2496, 2498, 2501, 2502, 2503, 2504, 2506, 2507,
                  2508, 2509, 2700, 2714, 2734, 2750, 2752, 2764, 2774, 2781)


def prepared_plan(directory, arm):
    directory = Path(directory)
    preparation_path = directory / "preparation.json"
    preparation_bytes = preparation_path.read_bytes()
    record = json.loads(preparation_bytes)
    expected = {
        "format": "attention_post_wmma_all40_waits_preparation_v1",
        "status": "CPU_PREPARED_AND_ELF_VERIFIED_GPU_NOT_RUN",
        "unmodified_roundtrip_hsaco_byte_identical": True,
        "original_wmma_count": 40, "selected_original_assembly_lines": list(SELECTED_LINES),
        "split_delay_original_lines": [1579, 1584, 1589],
        "split_second_consumer_original_lines": [1582, 1587, 1592],
        "stronger_wait_original_lines": [2766, 2776],
        "stronger_wait_instruction": "s_wait_alu depctr_va_vdst(0)",
        "skeleton_added_wait_instructions": 3, "skeleton_replaced_wait_instructions": 2,
        "reversing_declared_skeleton_changes_recovers_baseline": True,
        "requires_positive_skeleton_control": True,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Invalid or unsupported full-WMMA strengthened-wait preparation")
    entries = record.get("arms", [])
    if len(entries) != 3 or {item["name"] for item in entries} != {"skeleton", "v5", "s5"}:
        raise ValueError("Preparation requires skeleton and matched v5/s5 arms")
    for item in entries:
        name = item["name"]
        if (item.get("injected_padding_instructions") != (0 if name == "skeleton" else 200)
                or item.get("padding_per_site") != (0 if name == "skeleton" else 5)
                or item.get("padding_instruction") != {"skeleton": None, "v5": "v_nop", "s5": "s_nop 0"}[name]
                or not item.get("removing_marked_padding_recovers_skeleton")):
            raise ValueError("Invalid full-WMMA padding specification")
    choices = {item["name"]: item for item in entries}
    choices["baseline"] = {"hsaco_sha256": record["base_hsaco_sha256"],
                           "assembly_sha256": record["base_assembly_sha256"]}
    if arm not in choices:
        raise ValueError("Select baseline, skeleton, v5 or s5")
    loaded = {}
    # The declared unpadded control is also hashed for every selected arm.
    for name in dict.fromkeys(("baseline", "skeleton", arm)):
        for extension, field in (("hsaco", "hsaco_sha256"), ("amdgcn", "assembly_sha256")):
            path = directory / (name + "." + extension)
            data = path.read_bytes()
            if shared.digest(data) != choices[name][field]:
                raise ValueError("Code-object or assembly bytes disagree with preparation")
            loaded[name, extension] = (path, data)
    loader_path, shared_path = Path(__file__).resolve(), Path(shared.__file__).resolve()
    plan = {
        "format": "attention_post_assembly_override_v1", "experiment": "wmma_all40_waits_v1", "arm": arm,
        "preparation_path": str(preparation_path.resolve()),
        "preparation_sha256": shared.digest(preparation_bytes),
        "loader_path": str(loader_path), "loader_sha256": shared.digest(loader_path.read_bytes()),
        "shared_loader_path": str(shared_path), "shared_loader_sha256": shared.digest(shared_path.read_bytes()),
        "source_kernel_hash": record["base_kernel_hash"], "source_kernel_name": "_hstu_attn_bwd",
        "selected_original_assembly_lines": list(SELECTED_LINES),
        "split_delay_original_lines": [1579, 1584, 1589],
        "split_second_consumer_original_lines": [1582, 1587, 1592],
        "stronger_wait_original_lines": [2766, 2776],
        "stronger_wait_instruction": "s_wait_alu depctr_va_vdst(0)",
        "requires_positive_skeleton_control": True,
        "binary": loaded[arm, "hsaco"][1], "assembly": loaded[arm, "amdgcn"][1].decode("utf-8"),
        "loaded_hsaco_bytes": len(loaded[arm, "hsaco"][1]),
        "scope": "All40 WMMA sites over an identical strengthened-wait skeleton. The unpadded skeleton must reproduce failure before padding comparisons support interpretation. This changes wait timing deliberately; it does not preserve original timing or prove a hardware mechanism. Actual loaded bytes are identified by loaded_hsaco_sha256. Only in-memory objects change; fresh process required per arm.",
    }
    for prefix, name in (("original", "baseline"), ("skeleton", "skeleton"), ("loaded", arm)):
        for extension, field in (("hsaco", "hsaco"), ("amdgcn", "assembly")):
            path, data = loaded[name, extension]
            plan[f"{prefix}_{field}_path"] = str(path.resolve())
            plan[f"{prefix}_{field}_sha256"] = shared.digest(data)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "skeleton", "v5", "s5"), required=True)
    parser.add_argument("raw_args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    raw_args = options.raw_args[1:] if options.raw_args[:1] == ["--"] else options.raw_args
    plan = prepared_plan(options.prepared, options.arm)
    import probe_attention_raw_failure as runner
    shared.execution_arguments(runner, raw_args)
    from triton.compiler.compiler import CompiledKernel
    original_run, original_loop, previous_argv = runner.run, runner.run_loop, sys.argv

    def run(args, report, publish):
        with shared.install_code_object(CompiledKernel, plan, publish) as state:
            report["post_assembly_override"] = state
            result = original_run(args, report, publish)
            if not state["loaded_instances"]:
                raise ValueError("No verified attention code object was loaded")
            return result

    def loop(args, report, publish, *args_rest, **kwargs):
        report["configuration"]["post_assembly_override"] = shared.public_plan(plan)
        for prefix in ("loader", "shared_loader"):
            report["configuration"]["source_sha256"][plan[prefix + "_path"]] = plan[prefix + "_sha256"]
        return original_loop(args, report, publish, *args_rest, **kwargs)

    runner.run, runner.run_loop = run, loop
    sys.argv = [runner.__file__, *raw_args]
    try:
        return runner.main()
    finally:
        runner.run, runner.run_loop, sys.argv = original_run, original_loop, previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
