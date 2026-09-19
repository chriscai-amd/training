#!/usr/bin/env python3
"""Capture the first checked nonzero DQ intermediate in a diagnostic kernel.

Delegate arguments to the frozen raw runner after ``--``. Logical Q, V and
dOut must be zero. Example (GPU execution is explicit in this command):

  python scripts/attention_dq_stage_probe.py --stage-dump first-stage.pt -- \
      capture.pt --dout-zero --zero-input q --zero-input v --repeat 1000 --report stages.json

The production wrapper and zeroing prehook remain active. A cloned accumulator
observes dqk, old DQ, raw/scaled dot, sum, BF16 cast and volatile post-store
reload. Each program atomically retains its first bad tile. A global claim
records the first claiming program, not a causal ordering between programs.
Extra live values, reductions, workspace writes and readback change compilation
and scheduling. A passing diagnostic does not establish a production fix.
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


STAGES = ("dqk_trans", "old_dq", "dot", "scaled_dot", "sum", "cast", "stored_reload")
MAGIC = 0x44515031
GLOBAL_WORDS = 8
HEADER_WORDS = 16


@triton.jit
def _nonzero_count(value, mask_m):
    wrong = (value != 0) & mask_m[None, :]
    return tl.sum(tl.sum(wrong.to(tl.int32), 0), 0)


@triton.jit
def _store_panel(pointer, value):
    rows = tl.arange(0, value.shape[0])
    cols = tl.arange(0, value.shape[1])
    if value.dtype == tl.bfloat16:
        bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
    else:
        tl.static_assert(value.dtype == tl.float32)
        bits = value.to(tl.int32, bitcast=True)
    tl.store(pointer + rows[:, None] * value.shape[1] + cols[None, :], bits)


@triton.jit
def _stage_acc_dq(
    dq_ptrs_trans, start_m, stride_dqm, k, dqk_trans, alpha, mask_m,
    MAX_SEQ_LEN, LOCK, BLOCK_M: tl.constexpr, ATOMIC_ADD: tl.constexpr,
    ALLOW_TF32: tl.constexpr, key_start, seq_len,
):
    tl.static_assert(not ATOMIC_ADD, "Stage probe requires nonparallel sequence ownership")
    tl.static_assert(k.dtype == tl.bfloat16, "Initial stage probe supports BF16 inputs")
    count0 = _nonzero_count(dqk_trans, mask_m)
    old = tl.load(dq_ptrs_trans + start_m * stride_dqm, mask=mask_m[None, :],
                  other=0.0, eviction_policy="evict_last")
    count1 = _nonzero_count(old, mask_m)
    dot = tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32)
    count2 = _nonzero_count(dot, mask_m)
    scaled = dot * alpha
    count3 = _nonzero_count(scaled, mask_m)
    summed = old + scaled
    count4 = _nonzero_count(summed, mask_m)
    cast = summed.to(k.dtype)
    count5 = _nonzero_count(cast, mask_m)
    tl.store(dq_ptrs_trans + start_m * stride_dqm, cast, mask=mask_m[None, :],
             eviction_policy="evict_last")
    stored = tl.load(dq_ptrs_trans + start_m * stride_dqm, mask=mask_m[None, :],
                     other=0.0, volatile=True)
    count6 = _nonzero_count(stored, mask_m)
    stage = tl.where(count6 != 0, 7, 0)
    stage = tl.where(count5 != 0, 6, stage)
    stage = tl.where(count4 != 0, 5, stage)
    stage = tl.where(count3 != 0, 4, stage)
    stage = tl.where(count2 != 0, 3, stage)
    stage = tl.where(count1 != 0, 2, stage)
    stage = tl.where(count0 != 0, 1, stage)
    if stage != 0:
        pid = tl.program_id(0)
        K_WORDS: tl.constexpr = k.shape[0] * k.shape[1]
        G_WORDS: tl.constexpr = dqk_trans.shape[0] * dqk_trans.shape[1]
        D_WORDS: tl.constexpr = k.shape[1] * BLOCK_M
        STRIDE: tl.constexpr = 16 + K_WORDS + G_WORDS + 6 * D_WORDS
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
            tl.store(slot + 6, 0x44515031)
            tl.store(slot + 8, count0)
            tl.store(slot + 9, count1)
            tl.store(slot + 10, count2)
            tl.store(slot + 11, count3)
            tl.store(slot + 12, count4)
            tl.store(slot + 13, count5)
            tl.store(slot + 14, count6)
            data = slot + 16
            _store_panel(data, k)
            data += K_WORDS
            _store_panel(data, dqk_trans)
            data += G_WORDS
            _store_panel(data, old)
            data += D_WORDS
            _store_panel(data, dot)
            data += D_WORDS
            _store_panel(data, scaled)
            data += D_WORDS
            _store_panel(data, summed)
            data += D_WORDS
            _store_panel(data, cast)
            data += D_WORDS
            _store_panel(data, stored)
            # Completion marker is written only after this program's panels.
            tl.store(slot, stage)


def workspace_spec(programs, block_m, block_n, dim):
    if min(programs, block_m, block_n, dim) < 1:
        raise ValueError("Workspace dimensions must be positive")
    panels, offset = [], HEADER_WORDS
    for name, shape, dtype in (
        ("k", (block_n, dim), "torch.bfloat16"),
        ("dqk_trans", (block_n, block_m), "torch.bfloat16"),
        ("old_dq", (dim, block_m), "torch.bfloat16"),
        ("dot", (dim, block_m), "torch.float32"),
        ("scaled_dot", (dim, block_m), "torch.float32"),
        ("sum", (dim, block_m), "torch.float32"),
        ("cast", (dim, block_m), "torch.bfloat16"),
        ("stored_reload", (dim, block_m), "torch.bfloat16"),
    ):
        words = shape[0] * shape[1]
        panels.append({"name": name, "shape": list(shape), "dtype": dtype, "offset_words": offset, "words": words})
        offset += words
    return {"programs": programs, "block_m": block_m, "block_n": block_n, "dim": dim,
            "global_words": GLOBAL_WORDS, "header_words": HEADER_WORDS,
            "program_stride_words": offset, "total_words": GLOBAL_WORDS + programs * offset,
            "bytes": 4 * (GLOBAL_WORDS + programs * offset), "panels": panels}


def make_stage_probe_kernel(attention):
    """Build independent JIT state without a GPU launch or source-file edit."""
    original_block = attention._hstu_attn_bwd_one_block
    tree = ast.parse(original_block.src)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "acc_dq"]
    if len(calls) != 1:
        raise ValueError("Expected exactly one production acc_dq callsite")
    calls[0].keywords.extend([
        ast.keyword(arg="key_start", value=ast.parse("tl.min(offs_n, 0)", mode="eval").body),
        ast.keyword(arg="seq_len", value=ast.Name(id="seq_len", ctx=ast.Load())),
    ])
    ast.fix_missing_locations(tree)
    acc = _clone_jit(_stage_acc_dq, {}, "stage_probe")
    block = _clone_jit(original_block, {"acc_dq": acc}, "stage_probe")
    block._unsafe_update_src(ast.unparse(tree))
    col = _clone_jit(attention._hstu_attn_bwd_one_col_block,
                     {"_hstu_attn_bwd_one_block": block}, "stage_probe")
    original_raw = _raw_jit(attention._hstu_attn_bwd)
    kernel = _clone_jit(original_raw, {"_hstu_attn_bwd_one_col_block": col}, "stage_probe")
    kernel.diagnostic_metadata = {
        "variant": "dq_stage_probe", "stages_in_priority_order": list(STAGES),
        "production_jit_cache_key": original_raw.cache_key, "diagnostic_jit_cache_key": kernel.cache_key,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acc_source_sha256": hashlib.sha256(acc.src.encode()).hexdigest(),
        "block_source_sha256": hashlib.sha256(block.src.encode()).hexdigest(),
        "requires_zero_logical_inputs": ["q", "v", "dout"],
        "record_scope": "First bad tile per program; global first claim is a race winner, not a causal order.",
        "math_scope": "Original load/dot/scale/add/cast/store algebra; exposing stages can change fusion, register allocation and scheduling.",
        "stored_observation": "Volatile post-store reload; no explicit extra barrier.",
    }
    return kernel


def decode_workspace(raw, spec, pristine):
    """Decode exact BF16/FP32 bits and validate header/count/payload agreement."""
    if (raw.device.type != "cpu" or raw.dtype != torch.int32 or raw.ndim != 1
            or raw.numel() != spec["total_words"]):
        raise ValueError("Unexpected stage workspace layout")
    headers = raw[GLOBAL_WORDS:].view(spec["programs"], spec["program_stride_words"])
    records = []
    heads = pristine["k"].shape[1]
    order, offsets = pristine["sort_by_length_indices"], pristine["seq_offsets"]
    for pid in torch.nonzero(headers[:, 0]).flatten().tolist():
        row = headers[pid]
        stage = int(row[0])
        if not 1 <= stage <= len(STAGES) or int(row[1]) != pid or int(row[6]) != MAGIC:
            raise ValueError(f"Invalid/incomplete stage header for program {pid}")
        query, key, length = map(int, row[2:5])
        seq = pid // heads if order is None else int(order[pid // heads])
        head = pid % heads
        seq_start = int(offsets[seq])
        if length != int(offsets[seq + 1]) - seq_start or not 0 <= query < length or not 0 <= key < length:
            raise ValueError("Stage header sequence coordinates disagree with pristine metadata")
        mask = torch.arange(spec["block_m"]) + query < length
        if int(row[5]) != int(mask.sum()):
            raise ValueError("Stage header query mask disagrees with sequence length")
        panels = {}
        for panel in spec["panels"]:
            data = row[panel["offset_words"]:panel["offset_words"] + panel["words"]]
            if panel["dtype"] == "torch.bfloat16":
                if bool(((data < 0) | (data > 65535)).any()):
                    raise ValueError("BF16 panel contains invalid high bits")
                tensor = data.to(torch.int16).view(torch.bfloat16)
            else:
                tensor = data.clone().view(torch.float32)
            panels[panel["name"]] = tensor.reshape(panel["shape"])
        observed = [int(((panels[name] != 0) & mask[None, :]).sum()) for name in STAGES]
        counts = row[8:15].tolist()
        if observed != counts or next((i + 1 for i, count in enumerate(counts) if count), 0) != stage:
            raise ValueError("Recorded stage counts disagree with saved exact panel values")
        expected_k = torch.zeros_like(panels["k"])
        valid_k = min(spec["block_n"], length - key)
        expected_k[:valid_k] = pristine["k"][seq_start + key:seq_start + key + valid_k, head]
        k_equal = torch.equal(expected_k.view(torch.int16), panels["k"].view(torch.int16))
        failing = panels[STAGES[stage - 1]]
        bad = (failing != 0) & mask[None, :]
        samples = [{"tile_index": index, "value": str(failing[tuple(index)].item())}
                   for index in torch.nonzero(bad)[:32].tolist()]
        records.append({"program_id": pid, "sequence": seq, "head": head,
                        "query_start": query, "global_query_start": seq_start + query,
                        "key_start": key, "global_key_start": seq_start + key, "seq_len": length,
                        "first_checked_nonzero_stage": STAGES[stage - 1], "stage_counts": dict(zip(STAGES, counts)),
                        "k_tile_matches_pristine_global_input": k_equal, "count_payload_consistent": True,
                        "samples": samples, "panels": panels})
    first = int(raw[0]) - 1
    if (len(records) != int(raw[1])
            or (records and first not in {r["program_id"] for r in records})
            or (not records and int(raw[0]) != 0)):
        raise ValueError("Global and per-program stage claims disagree")
    return {"first_claimed_program": first if records else None, "records": records}


class StageProbePositive(RuntimeError):
    pass


class LaunchProxy:
    def __init__(self, original, controller):
        self.original, self.controller = original, controller

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]

        def run(*args, **kwargs):
            if args:
                raise ValueError("Stage proxy requires the production named launch arguments")
            return self.controller.launch(launch, kwargs)
        return run


class StageController:
    def __init__(self, destination, max_workspace_bytes):
        self.destination = Path(destination)
        self.max_workspace_bytes = max_workspace_bytes
        self.workspace = self.spec = self.pristine = self.metadata = None
        self.calls = 0
        self.positive = None

    @contextmanager
    def install(self, attention, variant):
        if variant != "production":
            raise ValueError("Stage probe cannot be combined with a separate --kernel-variant")
        original = attention._hstu_attn_bwd
        kernel = make_stage_probe_kernel(attention)
        original_fn = original.fn
        if kernel.arg_names != original_fn.arg_names:
            raise ValueError("Stage clone changed the raw launch signature")
        self.metadata = kernel.diagnostic_metadata
        original.fn = kernel
        attention._hstu_attn_bwd = LaunchProxy(original, self)
        try:
            yield self.metadata
        finally:
            attention._hstu_attn_bwd = original
            original.fn = original_fn

    def launch(self, function, kwargs):
        if self.pristine is None:
            raise ValueError("Stage probe lacks effective pristine inputs")
        config = self.config
        if kwargs["Q"].dtype != torch.bfloat16 or config.kwargs["SEQUENCE_PARALLEL"] is not False:
            raise ValueError("Stage probe requires BF16, SEQUENCE_PARALLEL=False")
        spec = workspace_spec(kwargs["Z"] * kwargs["H"], config.kwargs["BLOCK_M"],
                              config.kwargs["BLOCK_N"], kwargs["DimQ"])
        if spec["bytes"] > self.max_workspace_bytes:
            raise ValueError(f"Stage workspace needs {spec['bytes']} bytes; increase --max-workspace-mib")
        if self.workspace is None:
            self.spec = spec
            self.workspace = torch.empty(spec["total_words"], dtype=torch.int32, device=kwargs["Q"].device)
        elif spec != self.spec:
            raise ValueError("Launch dimensions changed during stage probe")
        self.calls += 1
        self.workspace.zero_()
        kwargs = {**kwargs, "LOCK": self.workspace}
        result = function(**kwargs)
        # One scalar readback, after the entire kernel. Negative calls retain
        # no tensor copies; a positive call freezes this exact workspace.
        if int(self.workspace[0].item()) == 0:
            return result
        raw = self.workspace.detach().cpu().clone()
        # Even an inconsistent workspace is evidence: persist its exact bits
        # before stopping, rather than losing it to a decoder exception.
        try:
            decoded = decode_workspace(raw, self.spec, self.pristine)
        except ValueError as error:
            decoded = {"first_claimed_program": int(raw[0]) - 1, "records": [],
                       "decode_error": str(error)}
        from repro_hstu_attention_capture import compiled_metadata
        payload = {"format": "hstu_dq_stage_probe_v1", "iteration": self.calls,
                   "workspace_spec": self.spec, "workspace_raw_int32": raw,
                   "diagnostic_metadata": self.metadata, "compiled_kernels": compiled_metadata(self.attention._hstu_attn_bwd),
                   "source_capture": self.source_capture, "effective_input_digests": self.expected_digests,
                   "configuration": self.configuration, **decoded,
                   "scope": "First detected nonzero intermediate in this instrumented launch, not production-kernel proof."}
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        self.positive = {"path": str(self.destination), "iteration": self.calls,
                         "first_claimed_program": decoded["first_claimed_program"],
                         "workspace_bytes": self.spec["bytes"],
                         "records": [{key: value for key, value in record.items() if key != "panels"}
                                     for record in decoded["records"]]}
        if "decode_error" in decoded:
            self.positive["decode_error"] = decoded["decode_error"]
        raise StageProbePositive("Saved first positive stage workspace")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-dump", type=Path, required=True)
    parser.add_argument("--max-workspace-mib", type=int, default=256)
    parser.add_argument("raw_args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if options.stage_dump.exists() or options.max_workspace_mib < 1:
        parser.error("Stage dump must be new and workspace limit positive")
    delegated = options.raw_args[1:] if options.raw_args[:1] == ["--"] else options.raw_args
    if not delegated:
        parser.error("Pass raw runner arguments after --")
    import probe_attention_raw_failure as runner
    controller = StageController(options.stage_dump, options.max_workspace_mib << 20)
    original_selected, original_loop = runner.selected_kernel, runner.run_loop

    def loop(args, report, publish, attention, config, inputs, outputs, ctx, history,
             pristine, expected_digests, *, chunk_bytes):
        for name in ("q", "v", "dout"):
            summary = expected_digests["inputs"][name]
            if summary["max_abs_finite"] != 0 or summary["nonfinite_count"]:
                raise ValueError("Stage probe requires exactly zero logical Q, V and dOut")
        if expected_digests["inputs"]["k"]["nonfinite_count"]:
            raise ValueError("Stage probe requires finite K for the zero oracle")
        controller.pristine = pristine["inputs"]
        controller.config, controller.attention = config, attention
        controller.source_capture = str(args.capture)
        controller.expected_digests = expected_digests
        controller.configuration = report["configuration"]
        try:
            return original_loop(args, report, publish, attention, config, inputs, outputs, ctx,
                                 history, pristine, expected_digests, chunk_bytes=chunk_bytes)
        except StageProbePositive:
            report.update(status="FAIL", iterations_completed=controller.calls, failed_iterations=1,
                          failure_counts={"stage_probe_positive": 1}, first_stage_capture=controller.positive,
                          stopped_before_standard_output_checks=True,
                          stage_probe_prehook_note="pre_hook_totals, when present, cover completed standard iterations before this positive launch")
            after = runner.input_digests(inputs, chunk_bytes=chunk_bytes, threshold=1e6)
            report["stage_probe_input_bytes_unchanged"] = after == expected_digests
            report["stage_probe_final_outputs"] = runner.failing_output_details(outputs, history, pristine["inputs"]["seq_offsets"], args)
            publish({"event": "stage_probe_positive", **controller.positive,
                     "input_bytes_unchanged": after == expected_digests})
            return 1
        finally:
            report["stage_probe_launches"] = controller.calls
            report["stage_probe_workspace_spec"] = controller.spec

    previous_argv = sys.argv
    runner.selected_kernel, runner.run_loop = controller.install, loop
    sys.argv = [str(Path(runner.__file__)), *delegated]
    try:
        return runner.main()
    finally:
        sys.argv = previous_argv
        runner.selected_kernel, runner.run_loop = original_selected, original_loop


if __name__ == "__main__":
    raise SystemExit(main())
