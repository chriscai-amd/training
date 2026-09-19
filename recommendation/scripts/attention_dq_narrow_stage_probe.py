#!/usr/bin/env python3
"""Capture one selected DQ stage with a narrow diagnostic workspace.

  python scripts/attention_dq_narrow_stage_probe.py --stage dqk_trans \
      --stage-dump first-dqk.pt -- capture.pt --dout-zero --zero-input q \
      --zero-input v --repeat 1000 --report dqk.json

Stages are dqk_trans (alias dqk), old_dq, and dot. Q, V and dOut must be exactly
zero. The selected stage and K are saved as exact bits. Production arithmetic
continues unchanged; instrumentation changes compilation and scheduling.

--force-positive-control injects 1 only into the diagnostic observed panel at
its first valid element. It does not change production arithmetic. The control
must capture on the first launch and is labeled CONTROL_CAPTURED, never a
spontaneous computation failure. Run it separately before interpreting a
negative run: negative runs do not exercise the diagnostic store branch.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

from attention_acc_dq_diagnostics import _clone_jit, _raw_jit
from attention_dq_stage_probe import _nonzero_count, _store_panel, StageProbePositive


STAGES = ("dqk_trans", "old_dq", "dot")
GLOBAL_WORDS, HEADER_WORDS, MAGIC = 8, 16, 0x44514E31


def canonical_stage(stage):
    stage = "dqk_trans" if stage == "dqk" else stage
    if stage not in STAGES:
        raise ValueError(f"Unknown narrow DQ stage: {stage!r}")
    return stage


@triton.jit
def _record_narrow_stage(LOCK, k, value, start_m, key_start, seq_len, mask_m,
                         STAGE: tl.constexpr, FORCE: tl.constexpr):
    if FORCE:
        rows = tl.arange(0, value.shape[0])
        cols = tl.arange(0, value.shape[1])
        inject = (rows[:, None] == 0) & (cols[None, :] == 0) & mask_m[None, :]
        # Modify only the diagnostic panel, never a production operand/result.
        observed = tl.where(inject, 1.0, value).to(value.dtype)
    else:
        observed = value
    count = _nonzero_count(observed, mask_m)
    if count != 0:
        pid = tl.program_id(0)
        K_WORDS: tl.constexpr = k.shape[0] * k.shape[1]
        VALUE_WORDS: tl.constexpr = value.shape[0] * value.shape[1]
        STRIDE: tl.constexpr = 16 + K_WORDS + VALUE_WORDS
        slot = LOCK + 8 + pid * STRIDE
        claimed = tl.atomic_cas(slot, 0, -1, sem="acq_rel")
        if claimed == 0:
            tl.atomic_cas(LOCK, 0, pid + 1, sem="acq_rel")
            tl.atomic_add(LOCK + 1, 1, sem="relaxed")
            tl.store(slot + 1, pid)
            tl.store(slot + 2, start_m)
            tl.store(slot + 3, key_start)
            tl.store(slot + 4, seq_len)
            tl.store(slot + 5, tl.sum(mask_m.to(tl.int32), 0))
            tl.store(slot + 6, 0x44514E31)
            tl.store(slot + 7, 1 if FORCE else 0)
            tl.store(slot + 8, count)
            _store_panel(slot + 16, k)
            _store_panel(slot + 16 + K_WORDS, observed)
            tl.store(slot, STAGE + 1)


@triton.jit
def _narrow_acc_dq(dq_ptrs_trans, start_m, stride_dqm, k, dqk_trans, alpha,
                   mask_m, MAX_SEQ_LEN, LOCK, BLOCK_M: tl.constexpr,
                   ATOMIC_ADD: tl.constexpr, ALLOW_TF32: tl.constexpr,
                   key_start, seq_len, STAGE: tl.constexpr, FORCE: tl.constexpr):
    tl.static_assert(not ATOMIC_ADD, "Narrow stage probe requires nonparallel ownership")
    tl.static_assert(k.dtype == tl.bfloat16, "Narrow stage probe requires BF16 inputs")
    if STAGE == 0:
        _record_narrow_stage(LOCK, k, dqk_trans, start_m, key_start, seq_len, mask_m, STAGE, FORCE)
    old = tl.load(dq_ptrs_trans + start_m * stride_dqm, mask=mask_m[None, :],
                  other=0.0, eviction_policy="evict_last")
    if STAGE == 1:
        _record_narrow_stage(LOCK, k, old, start_m, key_start, seq_len, mask_m, STAGE, FORCE)
    dot = tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32)
    if STAGE == 2:
        _record_narrow_stage(LOCK, k, dot, start_m, key_start, seq_len, mask_m, STAGE, FORCE)
    result = (old + dot * alpha).to(k.dtype)
    tl.store(dq_ptrs_trans + start_m * stride_dqm, result, mask=mask_m[None, :],
             eviction_policy="evict_last")


def workspace_spec(programs, block_m, block_n, dim, stage):
    stage = canonical_stage(stage)
    if min(programs, block_m, block_n, dim) < 1:
        raise ValueError("Workspace dimensions must be positive")
    shape = [block_n if stage == "dqk_trans" else dim, block_m]
    k_words, value_words = block_n * dim, shape[0] * shape[1]
    stride = HEADER_WORDS + k_words + value_words
    panels = [
        {"name": "k", "shape": [block_n, dim], "dtype": "torch.bfloat16",
         "offset_words": HEADER_WORDS, "words": k_words},
        {"name": stage, "shape": shape,
         "dtype": "torch.float32" if stage == "dot" else "torch.bfloat16",
         "offset_words": HEADER_WORDS + k_words, "words": value_words},
    ]
    return {"programs": programs, "block_m": block_m, "block_n": block_n, "dim": dim,
            "selected_stage": stage, "global_words": GLOBAL_WORDS, "header_words": HEADER_WORDS,
            "program_stride_words": stride, "total_words": GLOBAL_WORDS + programs * stride,
            "bytes": 4 * (GLOBAL_WORDS + programs * stride), "panels": panels}


def make_narrow_probe_kernel(attention, stage, *, force_positive_control=False):
    stage = canonical_stage(stage)
    original_block = attention._hstu_attn_bwd_one_block
    tree = ast.parse(original_block.src)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "acc_dq"]
    if len(calls) != 1:
        raise ValueError("Expected exactly one production acc_dq callsite")
    calls[0].keywords.extend([
        ast.keyword(arg="key_start", value=ast.parse("tl.min(offs_n, 0)", mode="eval").body),
        ast.keyword(arg="seq_len", value=ast.Name(id="seq_len", ctx=ast.Load())),
        ast.keyword(arg="STAGE", value=ast.Constant(STAGES.index(stage))),
        ast.keyword(arg="FORCE", value=ast.Constant(bool(force_positive_control))),
    ])
    ast.fix_missing_locations(tree)
    variant = f"narrow_{stage}_control_{int(force_positive_control)}"
    acc = _clone_jit(_narrow_acc_dq, {}, variant)
    block = _clone_jit(original_block, {"acc_dq": acc}, variant)
    block._unsafe_update_src(ast.unparse(tree))
    col = _clone_jit(attention._hstu_attn_bwd_one_col_block,
                     {"_hstu_attn_bwd_one_block": block}, variant)
    raw = _raw_jit(attention._hstu_attn_bwd)
    kernel = _clone_jit(raw, {"_hstu_attn_bwd_one_col_block": col}, variant)
    kernel.diagnostic_metadata = {
        "variant": "dq_narrow_stage_probe", "selected_stage": stage,
        "force_positive_control": bool(force_positive_control),
        "production_jit_cache_key": raw.cache_key, "diagnostic_jit_cache_key": kernel.cache_key,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acc_source_sha256": hashlib.sha256(acc.src.encode()).hexdigest(),
        "block_source_sha256": hashlib.sha256(block.src.encode()).hexdigest(),
        "requires_zero_logical_inputs": ["q", "v", "dout"],
        "record_scope": "First selected-stage nonzero tile per program; global first claim is a race winner, not causal order.",
        "math_scope": "Original load/dot/scale/add/cast/store algebra; instrumentation changes compilation and scheduling.",
        "control_scope": ("Diagnostic observed panel[0,0]=1 only; production operands and results unchanged."
                          if force_positive_control else "No sentinel injected."),
    }
    return kernel


def decode_workspace(raw, spec, pristine, *, force_positive_control=False):
    if (raw.device.type != "cpu" or raw.dtype != torch.int32 or raw.ndim != 1
            or raw.numel() != spec["total_words"]):
        raise ValueError("Unexpected narrow stage workspace layout")
    stage = canonical_stage(spec["selected_stage"])
    headers = raw[GLOBAL_WORDS:].view(spec["programs"], spec["program_stride_words"])
    heads = pristine["k"].shape[1]
    order, offsets = pristine["sort_by_length_indices"], pristine["seq_offsets"]
    records = []
    for pid in torch.nonzero(headers[:, 0]).flatten().tolist():
        row = headers[pid]
        if (int(row[0]) != STAGES.index(stage) + 1 or int(row[1]) != pid
                or int(row[6]) != MAGIC or int(row[7]) != int(force_positive_control)):
            raise ValueError(f"Invalid/incomplete narrow stage header for program {pid}")
        query, key, length = map(int, row[2:5])
        seq = pid // heads if order is None else int(order[pid // heads])
        head, seq_start = pid % heads, int(offsets[seq])
        if length != int(offsets[seq + 1]) - seq_start or not 0 <= query < length or not 0 <= key < length:
            raise ValueError("Narrow stage sequence coordinates disagree with pristine metadata")
        mask = torch.arange(spec["block_m"]) + query < length
        if int(row[5]) != int(mask.sum()):
            raise ValueError("Narrow stage query mask disagrees with sequence length")
        panels = {}
        for panel in spec["panels"]:
            data = row[panel["offset_words"]:panel["offset_words"] + panel["words"]]
            if panel["dtype"] == "torch.bfloat16":
                if bool(((data < 0) | (data > 65535)).any()):
                    raise ValueError("BF16 narrow panel contains invalid high bits")
                value = data.to(torch.int16).view(torch.bfloat16)
            else:
                value = data.clone().view(torch.float32)
            panels[panel["name"]] = value.reshape(panel["shape"])
        selected = panels[stage]
        bad = (selected != 0) & mask[None, :]
        count = int(bad.sum())
        if count == 0 or count != int(row[8]):
            raise ValueError("Narrow stage count disagrees with exact panel bits")
        expected_k = torch.zeros_like(panels["k"])
        valid_k = min(spec["block_n"], length - key)
        expected_k[:valid_k] = pristine["k"][seq_start + key:seq_start + key + valid_k, head]
        records.append({
            "program_id": pid, "sequence": seq, "head": head, "query_start": query,
            "global_query_start": seq_start + query, "key_start": key,
            "global_key_start": seq_start + key, "seq_len": length, "selected_stage": stage,
            "nonzero_count": count, "count_payload_consistent": True,
            "k_tile_matches_pristine_global_input": torch.equal(
                expected_k.view(torch.int16), panels["k"].view(torch.int16)),
            "force_positive_control": bool(force_positive_control),
            "control_sentinel_only": bool(force_positive_control and count == 1 and selected[0, 0] == 1),
            "samples": [{"tile_index": index, "value": str(selected[tuple(index)].item())}
                        for index in torch.nonzero(bad)[:32].tolist()],
            "panels": panels,
        })
    first = int(raw[0]) - 1
    if (len(records) != int(raw[1])
            or (records and first not in {record["program_id"] for record in records})
            or (not records and int(raw[0]) != 0)):
        raise ValueError("Global and per-program narrow stage claims disagree")
    return {"first_claimed_program": first if records else None, "records": records}


def addresses(tensors):
    return {name: {"data_ptr": value.data_ptr(),
                   "storage_data_ptr": value.untyped_storage().data_ptr(),
                   "storage_bytes": value.untyped_storage().nbytes(),
                   "storage_offset": value.storage_offset(), "shape": list(value.shape),
                   "stride": list(value.stride()), "dtype": str(value.dtype),
                   "device": str(value.device)}
            for name, value in tensors.items() if isinstance(value, torch.Tensor)}


def valid_control_capture(positive, *, expected_programs, calls):
    return (calls == 1 and not positive.get("decode_error") and expected_programs > 0
            and len(positive["records"]) == expected_programs
            and all(record["control_sentinel_only"]
                    and record["k_tile_matches_pristine_global_input"]
                    for record in positive["records"]))


class NarrowLaunchProxy:
    def __init__(self, original, controller):
        self.original, self.controller = original, controller

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        function = self.original[grid]

        def launch(*args, **kwargs):
            if args:
                raise ValueError("Narrow proxy requires production named launch arguments")
            return self.controller.launch(function, kwargs, grid=grid)
        return launch


class NarrowStageController:
    def __init__(self, destination, max_workspace_bytes, stage, *, force_positive_control=False):
        self.destination = Path(destination)
        self.max_workspace_bytes = max_workspace_bytes
        self.stage = canonical_stage(stage)
        self.force_positive_control = bool(force_positive_control)
        self.workspace = self.spec = self.pristine = self.metadata = None
        self.report = self.publish = None
        self.calls, self.positive = 0, None
        self.launch_metadata = None

    @contextmanager
    def install(self, attention, variant):
        if variant != "production":
            raise ValueError("Narrow stage probe cannot combine a separate --kernel-variant")
        original = attention._hstu_attn_bwd
        original_fn = original.fn
        kernel = make_narrow_probe_kernel(attention, self.stage,
                                          force_positive_control=self.force_positive_control)
        if kernel.arg_names != original_fn.arg_names:
            raise ValueError("Narrow stage clone changed the raw launch signature")
        self.metadata = kernel.diagnostic_metadata
        original.fn = kernel
        attention._hstu_attn_bwd = NarrowLaunchProxy(original, self)
        try:
            yield self.metadata
        finally:
            attention._hstu_attn_bwd = original
            original.fn = original_fn

    def launch(self, function, kwargs, *, grid):
        if self.pristine is None:
            raise ValueError("Narrow probe lacks effective pristine inputs")
        config = self.config
        if kwargs["Q"].dtype != torch.bfloat16 or config.kwargs["SEQUENCE_PARALLEL"] is not False:
            raise ValueError("Narrow probe requires BF16, SEQUENCE_PARALLEL=False")
        spec = workspace_spec(kwargs["Z"] * kwargs["H"], config.kwargs["BLOCK_M"],
                              config.kwargs["BLOCK_N"], kwargs["DimQ"], self.stage)
        actual_grid = tuple(grid({**kwargs, **config.kwargs}) if callable(grid) else grid)
        if actual_grid not in ((spec["programs"], 1), (spec["programs"], 1, 1)):
            raise ValueError(f"Narrow probe launch grid exceeds declared program ownership: {actual_grid}")
        if spec["bytes"] > self.max_workspace_bytes:
            raise ValueError(f"Narrow workspace needs {spec['bytes']} bytes; increase --max-workspace-mib")
        if self.workspace is None:
            self.spec = spec
            self.workspace = torch.empty(spec["total_words"], dtype=torch.int32, device=kwargs["Q"].device)
        elif spec != self.spec:
            raise ValueError("Launch dimensions changed during narrow probe")
        self.calls += 1
        self.workspace.zero_()
        kwargs = {**kwargs, "LOCK": self.workspace}
        if self.calls == 1:
            self.launch_metadata = {
                "selected_stage": self.stage, "force_positive_control": self.force_positive_control,
                "workspace_spec": self.spec, "launch_grid": list(actual_grid),
                "kernel_arg_addresses": addresses(kwargs), "config": dict(config.kwargs),
                "num_warps": getattr(config, "num_warps", None),
                "num_stages": getattr(config, "num_stages", None),
            }
            if self.report is not None:
                self.report["narrow_stage_allocation"] = self.launch_metadata
            if self.publish is not None:
                self.publish({"event": "narrow_stage_allocation", "before_launch": 1,
                              **self.launch_metadata})
        result = function(**kwargs)
        if int(self.workspace[0].item()) == 0:
            return result
        raw = self.workspace.detach().cpu().clone()
        try:
            decoded = decode_workspace(raw, self.spec, self.pristine,
                                        force_positive_control=self.force_positive_control)
        except ValueError as error:
            decoded = {"first_claimed_program": int(raw[0]) - 1, "records": [],
                       "decode_error": str(error)}
        from repro_hstu_attention_capture import compiled_metadata
        payload = {
            "format": "hstu_dq_narrow_stage_probe_v1", "format_version": 1,
            "iteration": self.calls, "selected_stage": self.stage,
            "force_positive_control": self.force_positive_control,
            "workspace_spec": self.spec, "workspace_raw_int32": raw,
            "launch_metadata": self.launch_metadata, "diagnostic_metadata": self.metadata,
            "compiled_kernels": compiled_metadata(self.attention._hstu_attn_bwd),
            "source_capture": self.source_capture, "effective_input_digests": self.expected_digests,
            "configuration": self.configuration, **decoded,
            "scope": ("FORCED DIAGNOSTIC CONTROL; observed panel sentinel is not a computation failure."
                      if self.force_positive_control else
                      "First detected nonzero selected stage in an instrumented launch; not production-kernel proof."),
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_suffix(self.destination.suffix + ".tmp")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.destination)
        fd = os.open(self.destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.positive = {"path": str(self.destination), "iteration": self.calls,
                         "selected_stage": self.stage, "force_positive_control": self.force_positive_control,
                         "first_claimed_program": decoded["first_claimed_program"],
                         "workspace_bytes": self.spec["bytes"],
                         "records": [{key: value for key, value in record.items() if key != "panels"}
                                     for record in decoded["records"]]}
        if "decode_error" in decoded:
            self.positive["decode_error"] = decoded["decode_error"]
        raise StageProbePositive("Saved first positive narrow stage workspace")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=(*STAGES, "dqk"), required=True)
    parser.add_argument("--stage-dump", type=Path, required=True)
    parser.add_argument("--max-workspace-mib", type=int, default=64)
    parser.add_argument("--force-positive-control", action="store_true")
    parser.add_argument("raw_args", nargs=argparse.REMAINDER)
    result = parser.parse_args(argv)
    if result.stage_dump.exists() or result.max_workspace_mib < 1:
        parser.error("Stage dump must be new and workspace limit positive")
    result.stage = canonical_stage(result.stage)
    result.raw_args = result.raw_args[1:] if result.raw_args[:1] == ["--"] else result.raw_args
    if not result.raw_args:
        parser.error("Pass raw runner arguments after --")
    return result


def main(argv=None):
    options = arguments(argv)
    import probe_attention_raw_failure as runner
    controller = NarrowStageController(options.stage_dump, options.max_workspace_mib << 20,
                                        options.stage, force_positive_control=options.force_positive_control)
    original_selected, original_loop = runner.selected_kernel, runner.run_loop

    def loop(args, report, publish, attention, config, inputs, outputs, ctx, history,
             pristine, expected_digests, *, chunk_bytes):
        for name in ("q", "v", "dout"):
            summary = expected_digests["inputs"][name]
            if summary["max_abs_finite"] != 0 or summary["nonfinite_count"]:
                raise ValueError("Narrow probe requires exactly zero logical Q, V and dOut")
        if expected_digests["inputs"]["k"]["nonfinite_count"]:
            raise ValueError("Narrow probe requires finite K for the zero oracle")
        controller.pristine = pristine["inputs"]
        controller.config, controller.attention = config, attention
        controller.source_capture, controller.expected_digests = str(args.capture), expected_digests
        controller.configuration = report["configuration"]
        controller.report, controller.publish = report, publish
        report["narrow_stage_control"] = options.force_positive_control
        try:
            result = original_loop(args, report, publish, attention, config, inputs, outputs, ctx,
                                   history, pristine, expected_digests, chunk_bytes=chunk_bytes)
            if options.force_positive_control:
                raise RuntimeError("Forced narrow control finished without exercising the capture branch")
            return result
        except StageProbePositive:
            positive = controller.positive
            valid_control = False
            if options.force_positive_control:
                expected_programs = int((controller.pristine["seq_offsets"].diff() > 0).sum()) * inputs["k"].shape[1]
                valid_control = valid_control_capture(
                    positive, expected_programs=expected_programs, calls=controller.calls)
            status = ("CONTROL_CAPTURED" if valid_control else "CONTROL_INVALID") if options.force_positive_control else "FAIL"
            report.update(status=status, iterations_completed=controller.calls,
                          failed_iterations=0 if valid_control else 1,
                          failure_counts={} if valid_control else {"narrow_stage_positive": 1},
                          first_narrow_stage_capture=positive,
                          stopped_before_standard_output_checks=True)
            after = runner.input_digests(inputs, chunk_bytes=chunk_bytes, threshold=1e6)
            report["narrow_stage_input_bytes_unchanged"] = after == expected_digests
            report["narrow_stage_final_outputs"] = runner.failing_output_details(
                outputs, history, pristine["inputs"]["seq_offsets"], args)
            outputs_zero = all(item["passed"] and item["exact_zero"] is True
                               for item in report["narrow_stage_final_outputs"].values())
            if valid_control and (after != expected_digests or not outputs_zero):
                valid_control = False
                report.update(status="CONTROL_INVALID", failed_iterations=1,
                              failure_counts={"control_input_or_output_changed": 1})
            publish({"event": "narrow_stage_positive", "status": report["status"],
                     **positive, "input_bytes_unchanged": after == expected_digests})
            return 0 if valid_control else 1
        finally:
            report["narrow_stage_launches"] = controller.calls
            report["narrow_stage_workspace_spec"] = controller.spec

    previous_argv = sys.argv
    runner.selected_kernel, runner.run_loop = controller.install, loop
    sys.argv = [str(Path(runner.__file__)), *options.raw_args]
    try:
        return runner.main()
    finally:
        sys.argv = previous_argv
        runner.selected_kernel, runner.run_loop = original_selected, original_loop


if __name__ == "__main__":
    raise SystemExit(main())
