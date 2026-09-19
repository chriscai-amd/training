#!/usr/bin/env python3
"""Run the fixed attention oracle with a separately prepared 35-site WMMA subset.

The matched v5/s5 arms add five V_NOP/S_NOP0 instructions at the same 35 sites.
Five of the 40 original WMMA sites are deliberately excluded. This wrapper
reuses the verified installation hooks without modifying the existing loader.
Use a fresh Python process for every arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import probe_attention_code_object as shared

SELECTED_LINES = (1574, 1576, 1582, 1587, 1592, 2079, 2093, 2108, 2152, 2222,
                  2242, 2263, 2281, 2475, 2485, 2487, 2489, 2491, 2493, 2496,
                  2498, 2501, 2502, 2503, 2504, 2506, 2507, 2508, 2509, 2700,
                  2714, 2734, 2750, 2752, 2781)
EXCLUDED_LINES = (1580, 1585, 1590, 2764, 2774)


def prepared_plan(directory, arm):
    directory = Path(directory)
    preparation_path = directory / "preparation.json"
    preparation_bytes = preparation_path.read_bytes()
    record = json.loads(preparation_bytes)
    if (record.get("format") != "attention_post_wmma_subset35_preparation_v1"
            or record.get("status") != "CPU_PREPARED_AND_ELF_VERIFIED_GPU_NOT_RUN"
            or not record.get("unmodified_roundtrip_hsaco_byte_identical")
            or not record.get("original_delay_instructions_unchanged")
            or record.get("original_wmma_count") != 40
            or record.get("selected_wmma_count") != 35
            or record.get("selected_original_assembly_lines") != list(SELECTED_LINES)
            or record.get("excluded_original_assembly_lines") != list(EXCLUDED_LINES)):
        raise ValueError("Invalid or unsupported 35-site WMMA subset preparation")
    entries = record.get("arms", [])
    if len(entries) != 2 or {item["name"] for item in entries} != {"v5", "s5"}:
        raise ValueError("Preparation must contain exactly matched v5/s5 arms")
    for item in entries:
        if (item.get("injected_instructions") != 175 or item.get("padding_per_site") != 5
                or item.get("padding_instruction") != {"v5": "v_nop", "s5": "s_nop 0"}[item["name"]]
                or not item.get("removing_marked_padding_recovers_baseline")):
            raise ValueError("Invalid subset padding specification")
    choices = {item["name"]: item for item in entries}
    choices["baseline"] = {"hsaco_sha256": record["base_hsaco_sha256"],
                           "assembly_sha256": record["base_assembly_sha256"]}
    if arm not in choices:
        raise ValueError("Select baseline, v5 or s5")
    loaded = {}
    for name in dict.fromkeys(("baseline", arm)):
        for extension, field in (("hsaco", "hsaco_sha256"), ("amdgcn", "assembly_sha256")):
            path = directory / (name + "." + extension)
            data = path.read_bytes()
            if shared.digest(data) != choices[name][field]:
                raise ValueError("Code-object or assembly bytes disagree with preparation")
            loaded[name, extension] = (path, data)
    loader_path, shared_path = Path(__file__).resolve(), Path(shared.__file__).resolve()
    plan = {
        "format": "attention_post_assembly_override_v1", "experiment": "wmma_subset35_v1", "arm": arm,
        "preparation_path": str(preparation_path.resolve()),
        "preparation_sha256": shared.digest(preparation_bytes),
        "loader_path": str(loader_path), "loader_sha256": shared.digest(loader_path.read_bytes()),
        "shared_loader_path": str(shared_path), "shared_loader_sha256": shared.digest(shared_path.read_bytes()),
        "source_kernel_hash": record["base_kernel_hash"], "source_kernel_name": "_hstu_attn_bwd",
        "selected_original_assembly_lines": list(SELECTED_LINES),
        "excluded_original_assembly_lines": list(EXCLUDED_LINES),
        "original_delay_instructions_unchanged": True,
        "binary": loaded[arm, "hsaco"][1], "assembly": loaded[arm, "amdgcn"][1].decode("utf-8"),
        "loaded_hsaco_bytes": len(loaded[arm, "hsaco"][1]),
        "scope": "35-site WMMA subset only; five sites excluded. V5/S5 have matched layout. Actual loaded bytes are identified by loaded_hsaco_sha256; source hash remains the original cache key. Reuses verified in-memory installation hooks. Fresh process required per arm.",
    }
    for prefix, name in (("original", "baseline"), ("loaded", arm)):
        for extension, field in (("hsaco", "hsaco"), ("amdgcn", "assembly")):
            path, data = loaded[name, extension]
            plan[f"{prefix}_{field}_path"] = str(path.resolve())
            plan[f"{prefix}_{field}_sha256"] = shared.digest(data)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "v5", "s5"), required=True)
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
