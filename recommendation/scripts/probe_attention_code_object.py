#!/usr/bin/env python3
"""Run the fixed attention oracle with a verified, prepared code object.

The on-disk JIT cache and source key remain unchanged. The in-memory HSACO and
assembly are replaced before the first runtime load; both hashes are recorded.
Use a fresh Python process per arm: modified cached objects persist in memory
after the class hooks are restored. Every run-property access is verified.
This is an opt-in post-assembly experiment, not a production kernel change.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def public_plan(plan):
    return {key: value for key, value in plan.items() if key not in ("binary", "assembly")}


def prepared_plan(directory, arm):
    directory = Path(directory)
    preparation_path = directory / "preparation.json"
    preparation_bytes = preparation_path.read_bytes()
    record = json.loads(preparation_bytes)
    if record.get("format") != "attention_post_msb_nop_preparation_v1":
        raise ValueError("Unsupported code-object preparation manifest")
    if not record.get("unmodified_roundtrip_hsaco_byte_identical"):
        raise ValueError("Preparation lacks an exact unmodified assembly roundtrip")
    original_path = directory / "baseline.hsaco"
    original = original_path.read_bytes()
    if digest(original) != record["base_hsaco_sha256"]:
        raise ValueError("Original code-object bytes disagree with preparation")
    choices = {item["name"]: item for item in record["arms"]}
    choices["baseline"] = {"hsaco_sha256": record["base_hsaco_sha256"],
                           "assembly_sha256": record["base_assembly_sha256"]}
    if arm not in choices or arm not in ("baseline", "nop0", "nop7"):
        raise ValueError("Select a recorded baseline, nop0 or nop7 arm")
    selected_path = directory / (arm + ".hsaco")
    selected = selected_path.read_bytes()
    if digest(selected) != choices[arm]["hsaco_sha256"]:
        raise ValueError("Selected code-object bytes disagree with preparation")
    original_assembly_path = directory / "baseline.amdgcn"
    selected_assembly_path = directory / (arm + ".amdgcn")
    original_assembly = original_assembly_path.read_bytes()
    selected_assembly = selected_assembly_path.read_bytes()
    if digest(original_assembly) != record["base_assembly_sha256"]:
        raise ValueError("Original assembly bytes disagree with preparation")
    if digest(selected_assembly) != choices[arm]["assembly_sha256"]:
        raise ValueError("Selected assembly bytes disagree with preparation")
    loader_path = Path(__file__).resolve()
    return {
        "format": "attention_post_assembly_override_v1", "arm": arm,
        "preparation_path": str(preparation_path.resolve()),
        "preparation_sha256": digest(preparation_bytes),
        "loader_path": str(loader_path), "loader_sha256": digest(loader_path.read_bytes()),
        "source_kernel_hash": record["base_kernel_hash"],
        "original_hsaco_path": str(original_path.resolve()),
        "original_hsaco_sha256": digest(original),
        "loaded_hsaco_path": str(selected_path.resolve()),
        "loaded_hsaco_sha256": digest(selected), "loaded_hsaco_bytes": len(selected),
        "original_assembly_path": str(original_assembly_path.resolve()),
        "original_assembly_sha256": digest(original_assembly),
        "loaded_assembly_path": str(selected_assembly_path.resolve()),
        "loaded_assembly_sha256": digest(selected_assembly),
        "assembly": selected_assembly.decode("utf-8"),
        "source_kernel_name": "_hstu_attn_bwd", "binary": selected,
        "scope": "Compiled kernel hash remains the original source/cache key; loaded_hsaco_sha256 identifies the actual in-memory code object. Only in-memory HSACO/assembly change. Hooks are restored, but modified cached instances persist: use a fresh Python process per arm.",
    }


@contextmanager
def install_code_object(kernel_class, plan, publish=None):
    """Replace only the pinned target before lazy module initialization."""
    # A fresh private copy is also the ownership marker for this context. A
    # loaded object from an earlier context must not be accepted by this one.
    plan = dict(plan)
    original_init, original_handles = kernel_class.__init__, kernel_class._init_handles
    original_run = kernel_class.run
    if not isinstance(original_run, property) or original_run.fget is None:
        raise ValueError("Expected the lazy CompiledKernel.run property")
    state = public_plan(plan)
    state.update(installed_instances=0, loaded_instances=0, loaded_resources=[])
    observed_loads = {}
    handle_fields = ("module", "function", "_run", "_module_pid")

    def emit(event):
        if publish is not None:
            publish({"event": event, "post_assembly_override": state})

    def is_target(instance):
        return (instance.name == plan["source_kernel_name"]
                or hasattr(instance, "_attention_override_plan"))

    def verify(instance):
        if (instance.name != plan["source_kernel_name"]
                or instance.hash != plan["source_kernel_hash"]
                or getattr(instance, "_attention_override_plan", None) is not plan
                or digest(instance.kernel) != plan["loaded_hsaco_sha256"]
                or digest(instance.asm["hsaco"]) != plan["loaded_hsaco_sha256"]
                or digest(instance.asm["amdgcn"].encode("utf-8")) != plan["loaded_assembly_sha256"]):
            raise ValueError("Unverified or changed attention source key, code object or assembly")

    def check_handles(instance, *, require_loaded=False):
        snapshot = observed_loads.get(instance)
        if snapshot is None:
            if require_loaded or any(getattr(instance, key, None) is not None for key in handle_fields):
                raise ValueError("Attention handles were initialized outside this verified load")
        elif any(getattr(instance, key, None) is not value for key, value in snapshot.items()):
            raise ValueError("Attention handles changed after the verified load")

    def initialize(instance, *args, **kwargs):
        original_init(instance, *args, **kwargs)
        if instance.name != plan["source_kernel_name"]:
            return
        if (instance.hash != plan["source_kernel_hash"]
                or digest(instance.kernel) != plan["original_hsaco_sha256"]
                or digest(instance.asm["hsaco"]) != plan["original_hsaco_sha256"]
                or digest(instance.asm["amdgcn"].encode("utf-8")) != plan["original_assembly_sha256"]):
            raise ValueError("Attention source key or original binary/assembly differs from the pinned code object")
        if any(getattr(instance, key, None) is not None for key in handle_fields):
            raise ValueError("Attention module was initialized before code-object replacement")
        instance.kernel = plan["binary"]
        instance.asm["hsaco"] = plan["binary"]
        instance.asm["amdgcn"] = plan["assembly"]
        instance._attention_override_plan = plan
        state["installed_instances"] += 1
        emit("attention_code_object_installed_before_load")

    def handles(instance):
        if not is_target(instance):
            return original_handles(instance)
        verify(instance)
        check_handles(instance)
        first = instance not in observed_loads
        result = original_handles(instance)
        verify(instance)
        if first:
            if instance.module is None or instance.function is None or not callable(instance._run):
                raise ValueError("Attention runtime did not finish loading the selected code object")
            observed_loads[instance] = {key: getattr(instance, key, None) for key in handle_fields}
            state["loaded_instances"] += 1
            state["loaded_resources"].append({key: getattr(instance, key, None)
                                               for key in ("n_regs", "n_spills", "n_max_threads")})
            emit("attention_code_object_loaded")
        check_handles(instance, require_loaded=True)
        return result

    def run(instance):
        if not is_target(instance):
            return original_run.fget(instance)
        verify(instance)
        check_handles(instance)
        launcher = original_run.fget(instance)
        verify(instance)
        check_handles(instance, require_loaded=True)
        if launcher is not instance._run:
            raise ValueError("Attention run property returned an unverified launcher")
        return launcher

    kernel_class.__init__, kernel_class._init_handles = initialize, handles
    kernel_class.run = property(run, original_run.fset, original_run.fdel, original_run.__doc__)
    try:
        yield state
    finally:
        kernel_class.__init__, kernel_class._init_handles = original_init, original_handles
        kernel_class.run = original_run


def execution_arguments(runner, raw_args):
    """Reject invalid execution options before importing Triton/runtime code."""
    raw = runner.arguments(raw_args)
    if (raw.keep_going or raw.validate_only or raw.kernel_variant != "production"
            or raw.failure_dump is None or raw.report is None):
        raise ValueError("Code-object experiment requires execution, production variant, new report/failure dump, and no keep-going")
    paths = [raw.capture, raw.report, raw.failure_dump]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Capture, report and failure dump must be distinct")
    if not raw.capture.is_file():
        raise ValueError("Capture must be an existing regular file")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "nop0", "nop7"), required=True)
    parser.add_argument("raw_args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    raw_args = options.raw_args[1:] if options.raw_args[:1] == ["--"] else options.raw_args
    plan = prepared_plan(options.prepared, options.arm)
    import probe_attention_raw_failure as runner
    execution_arguments(runner, raw_args)
    from triton.compiler.compiler import CompiledKernel
    original_run, original_loop, previous_argv = runner.run, runner.run_loop, sys.argv

    def run(args, report, publish):
        with install_code_object(CompiledKernel, plan, publish) as state:
            report["post_assembly_override"] = state
            result = original_run(args, report, publish)
            if not state["loaded_instances"]:
                raise ValueError("No verified attention code object was loaded")
            return result

    def loop(args, report, publish, *args_rest, **kwargs):
        report["configuration"]["post_assembly_override"] = public_plan(plan)
        report["configuration"]["source_sha256"][plan["loader_path"]] = plan["loader_sha256"]
        return original_loop(args, report, publish, *args_rest, **kwargs)

    runner.run, runner.run_loop = run, loop
    sys.argv = [runner.__file__, *raw_args]
    try:
        return runner.main()
    finally:
        runner.run, runner.run_loop, sys.argv = original_run, original_loop, previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
