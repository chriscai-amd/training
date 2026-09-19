#!/usr/bin/env python3
"""Replay attention directly from a trusted failure's pristine raw input bytes.

No normalization, UVQK projection, sorting, or random input preparation occurs.
The default arm retains captured input/output layouts and the production DQ
zeroing prehook. DQ fills, contiguous DQ, zero dOut/QKV, and synchronous prehook
checking are explicit interventions. Every call checks the exact history-DQ
zero oracle; checks/readback change timing even with --synchronize none.
--qkv-zero clones complete CPU input backings and zeros logical Q/K/V only;
unused packed U columns and padding retain their original bytes.
Repeatable --zero-input q|k|v selects individual fields for the same raw-clone
intervention and requires --dout-zero to retain the all-gradient zero oracle.
Captured environment is restored after Torch loads the CPU artifact, before
CUDA initialization and attention import. Existing shell values are recorded
as experiment overrides; this includes the VGPR cap and allocator settings.

Example:
  HSTU_BWD_MAX_VGPR=0 python scripts/probe_attention_raw_failure.py failure.pt \
      --repeat 100 --keep-going --report raw.json
Only --validate-only is CPU-only. Load trusted local artifacts only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import replay_backward_boundary as common
from replay_hstu_output_boundary import input_digests
from repro_hstu_attention_capture import all_chunks, compiled_metadata, history_zero, save_failure_dump
from repro_hstu_attention_zero_grad import check_zero


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--dq-init", choices=("production", "zero", "sentinel", "nan"), default="production")
    parser.add_argument("--sentinel", type=float, default=7.0)
    parser.add_argument("--dq-layout", choices=("captured", "contiguous"), default="captured")
    parser.add_argument("--dout-zero", action="store_true", help="clone captured CPU dOut then zero it before raw upload")
    parser.add_argument("--qkv-zero", action="store_true",
                        help="clone all raw CPU input backings, then zero logical Q/K/V; unused U/padding bytes remain pristine")
    parser.add_argument("--zero-input", action="append", choices=("q", "k", "v"), default=[],
                        help="repeat to zero selected logical Q/K/V fields in one raw CPU clone; requires --dout-zero")
    parser.add_argument("--kernel-variant", choices=("production", "baseline", "barrier_before_load",
                        "barrier_after_store", "barrier_both", "load_only", "dot_only", "store_zero"),
                        default="production", help="isolated accumulator diagnostic; production wrapper/prehook retained")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--prehook-check", action="store_true", help="synchronously check actual DQ after production prehook zero, before launch")
    parser.add_argument("--synchronize", choices=("none", "before", "after", "both"), default="none")
    parser.add_argument("--input-check", choices=("failure", "every"), default="failure",
                        help="verify all logical/backing input bytes after failures or every call; setup/final always checked")
    parser.add_argument("--failure-dump", type=Path, help="save only the first current failure, without replaying")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--check-chunk-rows", type=int, default=16384)
    parser.add_argument("--sample-bad-rows", type=int, default=8)
    parser.add_argument("--max-storage-gib", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-source-change", action="store_true")
    args = parser.parse_args(argv)
    if min(args.repeat, args.chunk_mib, args.check_chunk_rows, args.sample_bad_rows, args.max_storage_gib) < 1:
        parser.error("repeat and all limits must be positive")
    if not math.isfinite(args.sentinel) or args.sentinel == 0:
        parser.error("sentinel must be finite and nonzero")
    if args.kernel_variant in ("load_only", "dot_only", "store_zero") and not args.dout_zero:
        parser.error("load_only/dot_only/store_zero deliberately change DQ and require --dout-zero")
    if args.zero_input and not args.dout_zero and not args.qkv_zero:
        parser.error("selective --zero-input probes require --dout-zero; --qkv-zero retains the existing all-QKV control")
    for output in (args.report, args.failure_dump):
        if output and (output.exists() or output.resolve() == args.capture.resolve()):
            parser.error("report and failure-dump must name new files distinct from the capture")
    if args.report and args.failure_dump and args.report.resolve() == args.failure_dump.resolve():
        parser.error("report and failure-dump must differ")
    return args


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_capture(failure, *, chunk_bytes):
    torch = common._torch()
    if failure.get("format") != "hstu_attention_backward_failure" or failure.get("format_version") != 1:
        raise ValueError("Requires hstu_attention_backward_failure version 1")
    inputs, outputs, config = failure["pristine_inputs"], failure["outputs"], failure["configuration"]
    names = {"q", "k", "v", "dout", "seq_offsets", "num_targets", "sort_by_length_indices"}
    if set(inputs) != names or set(outputs) != {"dq", "dk", "dv"}:
        raise ValueError("Unexpected captured input/output fields")
    for name, tensor in {**inputs, **outputs}.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError(f"{name} is not a CPU strided tensor")
    ctx = config["context"]
    if (ctx.get("has_multiple_targets") is not True or ctx.get("num_softmax_heads") != 0
            or ctx.get("enable_tma") is not False or ctx.get("sort_by_length") is not True):
        raise ValueError("Only the captured sorted, target-masked, non-TMA SiLU attention path is supported")
    if config.get("pre_hook") != "_bwd_pre_hook" or config["config"].get("SEQUENCE_PARALLEL") is not False:
        raise ValueError("Requires the original production zeroing prehook and nonparallel sequence config")
    n, h, d = inputs["q"].shape
    expected = {"q": (n, ctx["num_heads"], ctx["attn_dim"]), "k": (n, h, d),
                "v": (n, h, ctx["hidden_dim"]), "dout": (n, h, ctx["hidden_dim"])}
    for name, shape in expected.items():
        if tuple(inputs[name].shape) != shape or not inputs[name].is_floating_point():
            raise ValueError(f"Invalid {name} shape/dtype")
    offsets, targets, order = (inputs[name] for name in ("seq_offsets", "num_targets", "sort_by_length_indices"))
    if any(t.dtype != torch.int64 or t.ndim != 1 for t in (offsets, targets, order)):
        raise ValueError("Sequence metadata must be one-dimensional int64")
    lengths = offsets[1:] - offsets[:-1]
    if (offsets.numel() < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != n
            or not bool((lengths > 0).all()) or int(lengths.max()) > ctx["max_seq_len"]
            or targets.shape != lengths.shape or not bool((targets == 1).all())
            or order.shape != lengths.shape or not torch.equal(torch.sort(order).values, torch.arange(len(lengths)))):
        raise ValueError("Invalid sequence offsets, target counts, or sort permutation")
    if not bool((lengths[order][1:] <= lengths[order][:-1]).all()):
        raise ValueError("Captured order is not descending sequence length")
    for name, input_name in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        if outputs[name].shape != inputs[input_name].shape or outputs[name].dtype != inputs[input_name].dtype:
            raise ValueError(f"Invalid captured {name} layout template")
    digests = input_digests(inputs, chunk_bytes=chunk_bytes, threshold=1e6)
    for name in expected:
        summary = digests["inputs"][name]
        if summary["nonfinite_count"] or summary["extreme_count"]:
            raise ValueError(f"Pristine {name} is nonfinite or exceeds the diagnostic 1e6 bound")
    history = torch.ones(n, dtype=torch.bool)
    history[offsets[1:] - 1] = False
    if not bool(history_zero(torch, inputs["dout"], history)):
        raise ValueError("The exact history-DQ zero oracle requires exactly zero history dOut")
    if not bool((inputs["dout"][offsets[1:] - 1] != 0).flatten(1).any(1).all()):
        raise ValueError("Captured target dOut must be nonzero in every sequence")
    return history, digests


def allocate_output_views(templates, *, device, max_bytes):
    """Fresh uninitialized complete backings, preserving captured output aliases."""
    torch = common._torch()
    storages = {tensor.untyped_storage()._cdata: tensor.untyped_storage() for tensor in templates.values()}
    size = sum(storage.nbytes() for storage in storages.values())
    if size > max_bytes:
        raise ValueError(f"Output storage allocation {size} exceeds {max_bytes} bytes")
    backing = {key: torch.empty(storage.nbytes(), dtype=torch.uint8, device=device)
               for key, storage in storages.items()}
    result = {name: torch.empty(0, dtype=tensor.dtype, device=device).set_(
        backing[tensor.untyped_storage()._cdata].untyped_storage(), tensor.storage_offset(),
        tensor.shape, tensor.stride()) for name, tensor in templates.items()}
    return result, size


def prepare_inputs(pristine_inputs, *, qkv_zero, dout_zero, chunk_bytes, max_bytes, zero_inputs=()):
    """Keep source evidence untouched; QKV intervention preserves raw layout."""
    if set(zero_inputs) - {"q", "k", "v"}:
        raise ValueError("Only q, k, v may be selected for raw input zeroing")
    selected = [name for name in ("q", "k", "v") if qkv_zero or name in zero_inputs]
    cloned_bytes = 0
    if selected:
        expected, cloned_bytes = common.restore_raw_tree(
            pristine_inputs, device="cpu", chunk_bytes=chunk_bytes, max_bytes=max_bytes)
        for name in selected:
            expected[name].zero_()
    else:
        expected = dict(pristine_inputs)
    if dout_zero:
        if selected:
            # Already independent; retain this view's exact stride/offset and
            # its relationships to the rest of the cloned raw input tree.
            expected["dout"].zero_()
        else:
            # Preserve the established dOut-only intervention unchanged.
            expected["dout"] = expected["dout"].clone().zero_()
    return expected, {
        "cloned_complete_input_backings": bool(selected),
        "cloned_input_storage_bytes": cloned_bytes,
        "selected_zero_inputs": selected,
        "logical_inputs_zeroed": selected + (["dout"] if dout_zero else []),
        "qkv_zero_scope": "selected logical Q/K/V fields only; unselected fields, unused packed U columns and padding retain pristine bytes" if selected else None,
        "original_capture_tensors_modified": False,
    }


def initialize_dq(dq, mode, sentinel):
    if mode == "zero":
        dq.zero_()
    elif mode != "production":
        dq.fill_(float("nan") if mode == "nan" else sentinel)


class PrehookViolation(RuntimeError):
    pass


@contextmanager
def count_pre_hook(config, expected_dq, *, check_zero_after):
    torch = common._torch()
    original = config.pre_hook
    state = {"calls": 0, "zero_checks": 0, "zero_failures": 0, "checked_elements": 0}

    def counted(nargs):
        state["calls"] += 1
        if nargs["SEQUENCE_PARALLEL"] is not False:
            raise PrehookViolation("Unexpected sequence-parallel prehook")
        actual = nargs["DQ"]
        if (actual.data_ptr() != expected_dq.data_ptr() or actual.shape != expected_dq.shape
                or actual.stride() != expected_dq.stride()):
            raise PrehookViolation("Production wrapper changed the DQ destination")
        original(nargs)
        if check_zero_after:
            state["zero_checks"] += 1
            nonzero = int(torch.count_nonzero(actual).item())
            state["checked_elements"] += actual.numel()
            if nonzero:
                state["zero_failures"] += 1
                raise PrehookViolation(f"Actual DQ has {nonzero} nonzero elements after production zero, before launch")

    config.pre_hook = counted
    try:
        yield state
    finally:
        config.pre_hook = original


def addresses(tensors):
    return {name: {"data_ptr": t.data_ptr(), "storage_data_ptr": t.untyped_storage().data_ptr(),
                   "storage_bytes": t.untyped_storage().nbytes(), "storage_offset": t.storage_offset(),
                   "shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)}
            for name, t in tensors.items()}


@contextmanager
def selected_kernel(attention, variant, factory=None):
    """Scope just the autotuner's raw JIT dependency, retaining its config."""
    if variant == "production":
        yield {"variant": "production", "preserves_dq_calculation": True}
        return
    if factory is None:
        from attention_acc_dq_diagnostics import make_diagnostic_kernel
        factory = make_diagnostic_kernel
    autotuner = attention._hstu_attn_bwd
    original = autotuner.fn
    replacement = factory(attention, variant)
    if replacement.arg_names != original.arg_names:
        raise ValueError("Diagnostic raw kernel changed the production argument signature")
    autotuner.fn = replacement
    try:
        yield replacement.diagnostic_metadata
    finally:
        autotuner.fn = original


def failing_output_details(outputs, history, cpu_offsets, args):
    torch = common._torch()
    all_zero = args.dout_zero or args.qkv_zero
    details = {}
    for name, tensor in outputs.items():
        result = check_zero(torch, tensor, chunk_rows=args.check_chunk_rows,
                            sample_limit=args.sample_bad_rows,
                            zero_row_mask=history if name == "dq" and not all_zero else None,
                            require_zero=all_zero or name == "dq")
        for sample in result["sample_bad_rows"]:
            row = sample["row"]
            seq = int(torch.searchsorted(cpu_offsets[1:], row, right=True))
            sample.update(sequence=seq, sequence_position=row - int(cpu_offsets[seq]))
        details[name] = result
    return details


def run_loop(args, report, publish, attention, config, inputs, outputs, ctx, history,
             pristine, expected_digests, *, chunk_bytes):
    """The same loop supports CPU fake kernels for control-flow tests."""
    torch = common._torch()
    device = inputs["q"].device
    offsets_cpu = pristine["inputs"]["seq_offsets"]
    versions = {name: tensor._version for name, tensor in inputs.items()}
    history = history.to(device)
    all_zero = args.dout_zero or args.qkv_zero
    failed_iterations = 0
    report.update(iterations_completed=0, failed_iterations=0, failure_counts={}, checks=[])
    started = time.monotonic()
    with count_pre_hook(config, outputs["dq"], check_zero_after=args.prehook_check) as hooks:
        for iteration in range(1, args.repeat + 1):
            initialize_dq(outputs["dq"], args.dq_init, args.sentinel)
            before_calls, before_checks = hooks["calls"], hooks["zero_checks"]
            if args.synchronize in ("before", "both") and device.type == "cuda":
                torch.cuda.synchronize(device)
            errors = []
            prehook_error = None
            try:
                attention.triton_hstu_attention_bwd(
                    **inputs, **outputs, N=ctx["max_seq_len"], alpha=ctx["attn_alpha"],
                    max_attn_len=ctx["max_attn_len"], contextual_seq_len=ctx["contextual_seq_len"],
                    enable_tma=ctx["enable_tma"], num_softmax_heads=ctx["num_softmax_heads"])
            except PrehookViolation as error:
                prehook_error = str(error)
                errors.append("prehook_zero_or_destination")
            if args.synchronize in ("after", "both") and device.type == "cuda":
                torch.cuda.synchronize(device)
            if hooks["calls"] - before_calls != 1:
                errors.append("prehook_count_not_one")
            flags = {}
            if prehook_error is None:
                for name, tensor in outputs.items():
                    flags[name + "_finite"] = all_chunks(torch, tensor, lambda tile, _: torch.isfinite(tile).all(),
                                                          args.check_chunk_rows)
                    if all_zero or name == "dq":
                        mask = torch.ones_like(history) if all_zero else history
                        flags[name + "_zero_oracle"] = history_zero(torch, tensor, mask)
                if not all_zero:
                    flags["dq_target_nonzero"] = (outputs["dq"][inputs["seq_offsets"][1:] - 1] != 0).any()
                values = torch.stack(list(flags.values())).cpu().tolist()
                errors.extend(name for name, passed in zip(flags, values) if not passed)
            errors.extend(name + "_version_changed" for name, tensor in inputs.items() if tensor._version != versions[name])
            check = {"iteration": iteration, "pre_hook_calls": hooks["calls"] - before_calls,
                     "pre_hook_zero_checks": hooks["zero_checks"] - before_checks,
                     "pre_hook_error": prehook_error, "attention_call_completed": prehook_error is None,
                     "elapsed_seconds": time.monotonic() - started}
            if errors or args.input_check == "every":
                after = input_digests(inputs, chunk_bytes=chunk_bytes, threshold=1e6)
                unchanged = after == expected_digests
                check["input_bytes"] = {"all_logical_and_backing_bytes_unchanged": unchanged,
                                        "digests": after if not unchanged else None}
                if not unchanged:
                    errors.append("input_bytes_changed")
            if errors and prehook_error is None:
                check["outputs"] = failing_output_details(outputs, history, offsets_cpu, args)
            if errors:
                failed_iterations += 1
                for name in errors:
                    report["failure_counts"][name] = report["failure_counts"].get(name, 0) + 1
                if args.failure_dump and "first_failure_dump" not in report:
                    report["first_failure_dump"] = save_failure_dump(
                        args.failure_dump, pristine, inputs, outputs, iteration=iteration,
                        failures=[{"iteration": iteration, "check": name} for name in errors],
                        configuration=report["configuration"], compiled_kernels=compiled_metadata(attention._hstu_attn_bwd),
                        report_path=args.report, chunk_bytes=chunk_bytes, max_bytes=args.max_storage_gib << 30)
            check.update(status="FAIL" if errors else "PASS", failures=errors)
            report["checks"].append(check)
            report.update(iterations_completed=iteration, failed_iterations=failed_iterations,
                          status="FAIL" if failed_iterations else "RUNNING",
                          pre_hook_totals=dict(hooks), compiled_kernels=compiled_metadata(attention._hstu_attn_bwd))
            publish({"event": "check", **check})
            if errors and not args.keep_going:
                break
    final = input_digests(inputs, chunk_bytes=chunk_bytes, threshold=1e6)
    report["final_input_bytes"] = {"all_logical_and_backing_bytes_unchanged": final == expected_digests,
                                    "digests": final if final != expected_digests else None}
    report["status"] = "FAIL" if failed_iterations or final != expected_digests else "PASS"
    return int(report["status"] != "PASS")


def restore_environment(captured):
    # Existing shell values are explicit experiment overrides, especially VGPR
    # cap, allocator, and serialization controls. Record every difference.
    overrides = {name: os.environ[name] for name, value in captured.items()
                 if name in os.environ and os.environ[name] != value}
    for name, value in captured.items():
        if name not in os.environ and value is not None:
            os.environ[name] = value
    os.environ.setdefault("HSA_ENABLE_COREDUMP", "0")
    return {"captured": captured, "explicit_environment_overrides": overrides,
            "effective": {name: os.environ.get(name) for name in captured}}


def run(args, report, publish):
    torch = common._torch()
    chunk_bytes = args.chunk_mib << 20
    failure = torch.load(args.capture, map_location="cpu", mmap=True, weights_only=False)
    history, original_digests = validate_capture(failure, chunk_bytes=chunk_bytes)
    report.update(source_artifact_sha256=file_sha256(args.capture), source_iteration=failure["iteration"],
                  captured_compiled_kernels=failure.get("compiled_kernels"), original_input_digests=original_digests,
                  validation={"pristine_input_bytes_used_directly": True, "history_dout_exact_zero": True,
                              "source_output_bytes_used_for_initialization": False})
    if args.validate_only:
        report.update(status="VALIDATED", cuda_initialized=torch.cuda.is_initialized())
        return 0
    config_source = failure["configuration"]
    report["environment"] = restore_environment(config_source["env"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from replay_nan_tripwire import _restore_runtime_state
    runtime = config_source.get("effective_runtime_state", config_source.get("captured_runtime_state"))
    if runtime is None:
        raise ValueError("Failure lacks captured runtime state")
    _restore_runtime_state(runtime)
    torch.cuda.set_device(0)
    from generative_recommenders.ops.triton import triton_hstu_attention as attention
    from generative_recommenders import common as model_common
    configs = attention._hstu_attn_bwd.configs
    if len(configs) != 1 or configs[0].pre_hook is not attention._bwd_pre_hook:
        raise ValueError("Requires one pinned config with unchanged production zeroing prehook")
    config = configs[0]
    actual_kwargs = {name: value for name, value in config.kwargs.items() if name != "llvm_fn_attrs"}
    expected_kwargs = {name: value for name, value in config_source["config"].items() if name != "llvm_fn_attrs"}
    if (actual_kwargs != expected_kwargs or config.num_warps != config_source["num_warps"]
            or config.num_stages != config_source["num_stages"]):
        raise ValueError("Production launch config differs from capture beyond the permitted VGPR cap")
    current_source_hash = file_sha256(attention.__file__)
    captured_source_hashes = config_source.get("source_sha256", {})
    matching = [value for path, value in captured_source_hashes.items() if Path(path).name == Path(attention.__file__).name]
    if matching and matching != [current_source_hash] and not args.allow_source_change:
        raise ValueError("Attention source hash differs; use --allow-source-change for an explicit comparison")
    expected_inputs, preparation = prepare_inputs(
        failure["pristine_inputs"], qkv_zero=args.qkv_zero, dout_zero=args.dout_zero,
        zero_inputs=args.zero_input, chunk_bytes=chunk_bytes, max_bytes=args.max_storage_gib << 30)
    expected_digests = input_digests(expected_inputs, chunk_bytes=chunk_bytes, threshold=1e6)
    inputs, input_bytes = common.restore_raw_tree(expected_inputs, device="cuda:0", chunk_bytes=chunk_bytes,
                                                  max_bytes=args.max_storage_gib << 30)
    outputs, output_bytes = allocate_output_views(failure["outputs"], device="cuda:0", max_bytes=args.max_storage_gib << 30)
    if args.dq_layout == "contiguous":
        outputs["dq"] = torch.empty_like(outputs["dq"], memory_format=torch.contiguous_format)
    restored_digests = input_digests(inputs, chunk_bytes=chunk_bytes, threshold=1e6)
    if restored_digests != expected_digests:
        raise ValueError("Uploaded raw input bytes/layouts differ from intended inputs")
    properties = torch.cuda.get_device_properties(0)
    effective_runtime = {
        "common": {"STATIC_MAX_SEQ_LENS": list(model_common.STATIC_MAX_SEQ_LENS),
                   "USE_RUNTIME_MAX_SEQ_LEN": model_common.USE_RUNTIME_MAX_SEQ_LEN,
                   "BACKEND_ALLOW_TF32": model_common.BACKEND_ALLOW_TF32},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul": {name: getattr(torch.backends.cuda.matmul, name)
                        for name in runtime.get("cuda_matmul", {})},
    }
    if effective_runtime != runtime:
        raise ValueError(f"Runtime state did not restore exactly: expected={runtime}, actual={effective_runtime}")
    report["configuration"] = {**config_source,
        "context": config_source["context"], "raw_input_source": str(args.capture.resolve()),
        "raw_input_source_sha256": report["source_artifact_sha256"],
        "source_sha256": {str(Path(__file__).resolve()): file_sha256(__file__), attention.__file__: current_source_hash},
        "source_matches_capture": matching == [current_source_hash] if matching else None,
        "runtime_state": runtime, "effective_runtime_state": effective_runtime,
        "env": report["environment"]["effective"],
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "actual_static_max_seq_lens": model_common.STATIC_MAX_SEQ_LENS,
        "actual_use_runtime_max_seq_len": model_common.USE_RUNTIME_MAX_SEQ_LEN,
        "config": dict(config.kwargs), "num_warps": config.num_warps, "num_stages": config.num_stages,
        "arch": getattr(properties, "gcnArchName", properties.name),
        "torch": str(torch.__version__), "hip": torch.version.hip,
        "raw_input_storage_bytes": input_bytes, "output_template_storage_bytes": output_bytes,
        "input_addresses": addresses(inputs), "output_addresses": addresses(outputs),
        "interventions": {"dq_init": args.dq_init, "sentinel": args.sentinel, "dq_layout": args.dq_layout,
                          "dout_zero": args.dout_zero, "qkv_zero": args.qkv_zero,
                          "zero_input": preparation["selected_zero_inputs"],
                          "input_preparation": preparation, "prehook_check": args.prehook_check,
                          "synchronize": args.synchronize, "input_check": args.input_check},
        "effective_input_digests": expected_digests, "raw_restore_verified": True,
        "scope": "Repeated raw input/output allocation reuse; all history DQ must be exactly zero. Nonzero gradients are not numerically validated."}
    pristine = {"inputs": expected_inputs, "storage_bytes": input_bytes, "source_specs": {
        name: {"shape": list(t.shape), "stride": list(t.stride()), "storage_offset": t.storage_offset(),
               "dtype": str(t.dtype), "source_device": str(t.device), "requires_grad": False}
        for name, t in inputs.items()}}
    amp = config_source["autocast"]
    with selected_kernel(attention, args.kernel_variant) as diagnostic, torch.no_grad(), torch.autocast(
            "cuda", enabled=amp["enabled"], dtype=getattr(torch, amp["dtype"].removeprefix("torch."))):
        report["configuration"]["diagnostic_kernel"] = diagnostic
        if args.kernel_variant != "production":
            import attention_acc_dq_diagnostics
            path = attention_acc_dq_diagnostics.__file__
            report["configuration"]["source_sha256"][path] = file_sha256(path)
        publish({"event": "configuration", **report["configuration"]})
        return run_loop(args, report, publish, attention, config, inputs, outputs, config_source["context"],
                        history, pristine, expected_digests, chunk_bytes=chunk_bytes)


def main():
    args = arguments()
    report = {"format": "hstu_attention_raw_failure_probe_v1", "capture": str(args.capture),
              "repeat": args.repeat, "keep_going": args.keep_going, "status": "RUNNING",
              "timing_note": "Scalar per-call oracle readback synchronizes. Prehook checks, fills, input hashing, and optional failure copying add traffic."}

    def publish(record):
        print(json.dumps(record, default=str, allow_nan=False), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
            os.replace(temporary, args.report)

    try:
        result = run(args, report, publish)
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        publish({"event": "error", "error": report["error"]})
        raise
    publish({"event": "summary", "status": report["status"], "iterations_completed": report.get("iterations_completed", 0),
             "failed_iterations": report.get("failed_iterations", 0), "failure_counts": report.get("failure_counts", {})})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
