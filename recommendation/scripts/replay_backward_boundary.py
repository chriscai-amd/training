#!/usr/bin/env python3
"""Analyze a trusted backward-boundary .pt dump with bounded CPU FP64 references.

Default: inspect at most 4096 suspicious sample rows, retaining every feature
column. --rows all --max-rows 0 includes every row and computes complete weight
and bias gradients. Otherwise their references are partial row contributions.
Finite dumps are supported; without suspicious rows, the first 16 control rows
are selected. Row-selection bookkeeping is O(total rows), outside the tile budget.
Optional --gpu replays the complete operation (three fresh attempts by default)
and requires the original full-state capture's context.json. This is a numerical
diagnostic, not a claim of bitwise equivalence between GPU kernels and FP64.

Examples:
  python scripts/replay_backward_boundary.py boundary.pt --report analysis.json
  python scripts/replay_backward_boundary.py boundary.pt --rows 17,31:35
  python scripts/replay_backward_boundary.py repeat0.pt --compare-other repeat1.pt --compare-only
  python scripts/replay_backward_boundary.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


_TORCH = None
_ARGUMENTS = {
    "triton_addmm_bwd": ("x", "w", "dz", "is_y_1d"),
    "triton_weighted_layer_norm_bwd":
        ("dy", "x", "weight", "bias", "mean", "rstd", "learnable", "eps", "BLOCK_D"),
}
_OUTPUTS = {
    "triton_addmm_bwd": ("d_normed_x", "d_uvqk_weight", "d_uvqk_bias"),
    "triton_weighted_layer_norm_bwd": ("d_x", "d_norm_weight", "d_norm_bias"),
}


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def _number(value):
    return value if math.isfinite(value) else str(value)


def _tensors(tree):
    torch = _torch()
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _tensors(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _tensors(value)
    elif tree is not None and not isinstance(tree, (bool, int, float, str)):
        raise TypeError(f"Unsupported dump argument: {type(tree)}")


def require_pristine_input_snapshot(payload):
    """Reject fault-time inputs that cannot reproduce the original operation.

    Legacy boundary dumps predate this marker and contain pre-call snapshots.
    The training monitor also permits pre-call input faults: it stops before
    execution and explicitly certifies that snapshot as pristine. Post-call
    snapshots remain useful raw evidence but are not exact replay inputs.
    """
    event = payload.get("event", {})
    pristine = payload.get("input_snapshot_pristine")
    monitor = payload.get("format") == "training_backward_monitor_v1"
    if (pristine is False or event.get("input_snapshot_pristine") is False
            or (monitor and (payload.get("stage") != "input" or pristine is not True))):
        raise ValueError(
            "Post-call input snapshots cannot establish pristine inputs for original-call "
            "replay; use a pristine pre-call boundary capture. Pre-call scalar checks "
            "do not prove that input bytes were unchanged during execution."
        )


def bind_inputs(payload):
    """Bind the two supported public APIs without importing any GPU kernels."""
    require_pristine_input_snapshot(payload)
    operation = payload.get("operation")
    if operation not in _ARGUMENTS:
        raise ValueError(f"Unsupported operation: {operation!r}")
    signature = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in _ARGUMENTS[operation]
    ])
    values = dict(signature.bind(*payload["args"], **payload["kwargs"]).arguments)
    torch = _torch()
    for tensor in _tensors(values):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("Load a CPU, strided-tensor dump before analysis")
        if not tensor.is_floating_point() or tensor.is_quantized:
            raise ValueError("Only real floating-point operation tensors are supported")
    x = values["x"]
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("Expected x with shape [sample rows, positive feature width]")
    m, d = x.shape
    if operation == "triton_addmm_bwd":
        w, dz = values["w"], values["dz"]
        if w.ndim != 2 or dz.ndim != 2 or w.shape != (d, dz.shape[1]) or dz.shape[0] != m:
            raise ValueError("Incompatible x, w, dz GEMM shapes")
        if len({x.dtype, w.dtype, dz.dtype}) != 1:
            raise ValueError("GEMM input dtypes differ")
    else:
        if values["dy"].shape != x.shape:
            raise ValueError("Layer-norm dy and x shapes differ")
        if values["mean"].shape != (m,) or values["rstd"].shape != (m,):
            raise ValueError("Layer-norm mean/rstd must have one value per sample row")
        if values["learnable"] and any(
                values[name] is None or values[name].shape != (d,)
                for name in ("weight", "bias")):
            raise ValueError("Learnable layer norm requires feature-length weight and bias")
    outputs = payload.get("outputs")
    if outputs is not None and (not isinstance(outputs, (tuple, list)) or len(outputs) != 3):
        raise ValueError("Expected three captured outputs, or outputs=None for an input trigger")
    return values


def _row_mask(tensor, threshold, chunk_bytes):
    """Scan real sample rows; parameter feature indices are never sample rows."""
    torch = _torch()
    width = math.prod(tensor.shape[1:])
    rows_per_chunk = max(1, chunk_bytes // max(1, width * 8 * 4))
    if width * 8 * 4 > chunk_bytes:
        raise ValueError("One row exceeds --chunk-mib; increase the analysis chunk budget")
    result = torch.zeros(tensor.shape[0], dtype=torch.bool)
    for start in range(0, tensor.shape[0], rows_per_chunk):
        value = tensor[start:start + rows_per_chunk].to(torch.float64)
        bad = ~torch.isfinite(value) | (value.abs() > threshold)
        result[start:start + value.shape[0]] = bad.reshape(value.shape[0], -1).any(dim=1)
    return result


def select_rows(payload, values, rows, max_rows, threshold, chunk_bytes):
    torch = _torch()
    count = values["x"].shape[0]
    reason = rows
    if rows == "all":
        selected = torch.arange(count)
    elif rows == "suspicious":
        names = ("x", "dz") if payload["operation"] == "triton_addmm_bwd" else (
            "x", "dy", "mean", "rstd")
        bad = torch.zeros(count, dtype=torch.bool)
        for name in names:
            bad |= _row_mask(values[name], threshold, chunk_bytes)
        outputs = payload.get("outputs")
        if outputs is not None and outputs[0] is not None:
            if outputs[0].shape != values["x"].shape:
                raise ValueError("Captured dx shape does not match x")
            bad |= _row_mask(outputs[0], threshold, chunk_bytes)
        selected = torch.nonzero(bad).flatten()
        if selected.numel() == 0:
            selected = torch.arange(min(16, count))
            reason = "no suspicious sample rows; first 16 control rows (parameter anomalies do not identify sample rows)"
    else:
        mask = torch.zeros(count, dtype=torch.bool)
        for part in rows.split(","):
            if ":" in part:
                bounds = part.split(":")
                if len(bounds) != 2:
                    raise ValueError("Rows use comma-separated indices or start:stop ranges")
                start, stop = map(int, bounds)
            else:
                start = int(part)
                stop = start + 1
            if not 0 <= start <= stop <= count:
                raise ValueError(f"Row selection {part!r} is outside [0, {count})")
            mask[start:stop] = True
        selected = torch.nonzero(mask).flatten()
    eligible = selected.numel()
    if max_rows:
        selected = selected[:max_rows]
    return selected, {
        "requested": rows, "reason": reason, "total_sample_rows": count,
        "eligible_rows": eligible, "analyzed_rows": selected.numel(),
        "truncated": selected.numel() < eligible,
        "all_rows_included": selected.numel() == count,
        "first_sample_rows": selected[:32].tolist(),
        "last_sample_rows": selected[-8:].tolist(),
        "columns": "all; no feature/reduction columns sampled",
    }


def _paired_chunks(left, right, elements, origin=None):
    if origin is None:
        origin = (0,) * left.ndim
    if left.numel() <= elements:
        yield left, right, origin
        return
    axis = next(i for i, size in enumerate(left.shape) if size > 1)
    width = max(1, elements // (left.numel() // left.shape[axis]))
    for start in range(0, left.shape[axis], width):
        length = min(width, left.shape[axis] - start)
        offset = list(origin)
        offset[axis] += start
        yield from _paired_chunks(left.narrow(axis, start, length), right.narrow(axis, start, length),
                                  elements, tuple(offset))


def _coordinate(flat, shape, origin):
    result = []
    for size in reversed(shape):
        result.append(flat % size)
        flat //= size
    return [index + offset for index, offset in zip(reversed(result), origin)]


def _spec(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
            "storage_offset": tensor.storage_offset(), "storage_bytes": tensor.untyped_storage().nbytes(),
            "is_conj": tensor.is_conj(), "is_neg": tensor.is_neg()}


def compare_tensors(left, right, *, chunk_bytes, axis0):
    """Compare every logical byte, preserving NaN payload/signed-zero differences."""
    torch = _torch()
    result = {"left": _spec(left), "right": _spec(right), "axis0_semantics": axis0}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {**result, "comparable": False, "equal_logical_bytes": False,
                "reason": "shape or dtype differs"}
    if left.device.type != "cpu" or right.device.type != "cpu":
        raise ValueError("Boundary comparisons require CPU tensors")
    result.update(comparable=True, compared_bytes=left.numel() * left.element_size(),
                  mismatched_bytes=0, mismatched_elements=0, nonfinite_left=0, nonfinite_right=0,
                  max_abs_finite_left=0.0, max_abs_finite_right=0.0,
                  max_abs_difference_finite_pairs=0.0, max_difference_location=None, samples=[])
    changed_rows = torch.zeros(left.shape[0], dtype=torch.bool) if left.ndim else None
    for lhs, rhs, origin in _paired_chunks(left, right, max(1, chunk_bytes // 128)):
        if not lhs.numel():
            continue
        lc = lhs.resolve_conj().resolve_neg().contiguous()
        rc = rhs.resolve_conj().resolve_neg().contiguous()
        byte_diff = lc.reshape(-1).view(torch.uint8) != rc.reshape(-1).view(torch.uint8)
        element_diff = byte_diff.reshape(-1, left.element_size()).any(dim=1)
        result["mismatched_bytes"] += int(byte_diff.sum())
        result["mismatched_elements"] += int(element_diff.sum())
        if changed_rows is not None:
            rows = element_diff.reshape(lhs.shape[0], -1).any(dim=1)
            changed_rows[origin[0]:origin[0] + lhs.shape[0]] |= rows
        if len(result["samples"]) < 16:
            for flat in torch.nonzero(element_diff).flatten()[:16 - len(result["samples"])].tolist():
                result["samples"].append({"index": _coordinate(flat, lhs.shape, origin),
                                          "left": str(lc.reshape(-1)[flat].item()),
                                          "right": str(rc.reshape(-1)[flat].item())})
        l64, r64 = lc.to(torch.float64), rc.to(torch.float64)
        lf, rf = torch.isfinite(l64), torch.isfinite(r64)
        result["nonfinite_left"] += int((~lf).sum())
        result["nonfinite_right"] += int((~rf).sum())
        for value, finite, key in ((l64, lf, "max_abs_finite_left"), (r64, rf, "max_abs_finite_right")):
            if bool(finite.any()):
                result[key] = max(result[key], float(value[finite].abs().max()))
        both = lf & rf
        if bool(both.any()):
            difference = torch.where(both, (l64 - r64).abs(), -1.)
            flat = int(difference.reshape(-1).argmax())
            maximum = float(difference.reshape(-1)[flat])
            if maximum > result["max_abs_difference_finite_pairs"]:
                result["max_abs_difference_finite_pairs"] = maximum
                result["max_difference_location"] = {
                    "index": _coordinate(flat, lhs.shape, origin),
                    "left": str(l64.reshape(-1)[flat].item()), "right": str(r64.reshape(-1)[flat].item()),
                }
    result["equal_logical_bytes"] = result["mismatched_bytes"] == 0
    result["layout_equal"] = result["left"] == result["right"]
    indices = torch.nonzero(changed_rows).flatten() if changed_rows is not None else None
    result["affected_axis0_count"] = int(indices.numel()) if indices is not None else None
    result["affected_axis0_range"] = [int(indices[0]), int(indices[-1])] if indices is not None and indices.numel() else None
    return {key: _number(value) if isinstance(value, float) else value for key, value in result.items()}


def _compare_storages(left, right, chunk_bytes):
    torch = _torch()
    ls, rs = left.untyped_storage(), right.untyped_storage()
    lb = torch.empty(0, dtype=torch.uint8).set_(ls, 0, (ls.nbytes(),), (1,))
    rb = torch.empty(0, dtype=torch.uint8).set_(rs, 0, (rs.nbytes(),), (1,))
    common = min(lb.numel(), rb.numel())
    changed = 0
    samples = []
    for start in range(0, common, chunk_bytes):
        different = lb[start:start + min(chunk_bytes, common - start)] != rb[start:start + min(chunk_bytes, common - start)]
        changed += int(different.sum())
        if len(samples) < 16:
            # Sampling remains bounded even if every storage byte differs.
            for offset in range(0, different.numel(), 4096):
                samples.extend((start + offset + torch.nonzero(different[offset:offset + 4096]).flatten()[:16 - len(samples)]).tolist())
                if len(samples) == 16:
                    break
    return {"left_bytes": lb.numel(), "right_bytes": rb.numel(), "compared_common_bytes": common,
            "mismatched_common_bytes": changed, "unpaired_bytes": abs(lb.numel() - rb.numel()),
            "equal": changed == 0 and lb.numel() == rb.numel(), "first_mismatch_byte_offsets": samples}


def compare_boundaries(left, right, *, chunk_bytes=64 << 20):
    """Compare all inputs/outputs and complete backing storages in CPU chunks."""
    torch = _torch()
    if chunk_bytes < 1024:
        raise ValueError("Comparison chunk must be at least 1024 bytes")
    if left.get("operation") != right.get("operation"):
        raise ValueError("Compare two dumps of the same backward operation")
    lv, rv = bind_inputs(left), bind_inputs(right)
    operation = left["operation"]
    lvalues = {"input/" + key: value for key, value in lv.items()}
    rvalues = {"input/" + key: value for key, value in rv.items()}
    for source, dest in ((left, lvalues), (right, rvalues)):
        output = source.get("outputs")
        for i, name in enumerate(_OUTPUTS[operation]):
            dest["output/" + name] = output[i] if output is not None else None
    tensors, scalars, storages, pairs = {}, {}, {}, {}
    aliases_left, aliases_right = {}, {}
    started = time.monotonic()
    for name in lvalues:
        lhs, rhs = lvalues[name], rvalues[name]
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            sample_axis = name in {"input/x", "input/dz", "input/dy", "input/mean", "input/rstd",
                                   "output/d_normed_x", "output/d_x"}
            if name == "output/d_uvqk_bias" and not lv["is_y_1d"]:
                sample_axis = True
            tensors[name] = compare_tensors(lhs, rhs, chunk_bytes=chunk_bytes,
                                             axis0="sample row" if sample_axis else "parameter feature")
            key = (lhs.untyped_storage()._cdata, rhs.untyped_storage()._cdata)
            if key not in pairs:
                sid = str(len(pairs))
                pairs[key] = sid
                storages[sid] = _compare_storages(lhs, rhs, chunk_bytes)
            tensors[name]["storage_pair"] = pairs[key]
            aliases_left.setdefault(key[0], []).append(name)
            aliases_right.setdefault(key[1], []).append(name)
        else:
            scalars[name] = {"left": repr(lhs) if not isinstance(lhs, torch.Tensor) else _spec(lhs),
                             "right": repr(rhs) if not isinstance(rhs, torch.Tensor) else _spec(rhs),
                             "equal": lhs is rhs if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor) else lhs == rhs}
    return {"operation": operation, "left_attempt": left.get("attempt"), "right_attempt": right.get("attempt"),
            "left_stage": left.get("stage"), "right_stage": right.get("stage"),
            "scope": "Every logical tensor byte, all rows/columns; complete backing storages separately include unused bytes.",
            "tensors": tensors, "scalars_or_absent_outputs": scalars, "storage_pairs": storages,
            "left_storage_alias_groups": list(aliases_left.values()),
            "right_storage_alias_groups": list(aliases_right.values()),
            "seconds": time.monotonic() - started}


class ReferenceSummary:
    """Accumulate reductions over bounded FP64 tiles, without retaining them."""
    def __init__(self, dtype, *, scope, observed_available):
        self.dtype = dtype
        self.data = {
            "reference_dtype": "torch.float64", "output_dtype": str(dtype),
            "scope": scope, "elements": 0, "reference_nonfinite": 0,
            "reference_max_abs_finite": 0.0, "outside_output_finite_range": 0,
            "cast_nonfinite": 0, "overflow_samples": [],
            "observed_comparison": "available" if observed_available else "not available for this scope",
            "compared_elements": 0, "observed_nonfinite": 0,
            "observed_nonfinite_with_finite_cast_reference": 0,
            "observed_finite_with_nonfinite_cast_reference": 0,
            "max_abs_error_vs_fp64_finite_pairs": 0.0,
            "max_abs_error_vs_cast_reference_finite_pairs": 0.0,
        }

    def add(self, reference, observed=None, coordinate=None):
        torch = _torch()
        d = self.data
        finite = torch.isfinite(reference)
        magnitude = reference.abs()
        exceeds = finite & (magnitude > torch.finfo(self.dtype).max)
        cast = reference.to(self.dtype).to(torch.float64)
        cast_finite = torch.isfinite(cast)
        d["elements"] += reference.numel()
        d["reference_nonfinite"] += int((~finite).sum())
        d["outside_output_finite_range"] += int(exceeds.sum())
        d["cast_nonfinite"] += int((~cast_finite).sum())
        if bool(finite.any()):
            d["reference_max_abs_finite"] = max(
                d["reference_max_abs_finite"], float(magnitude[finite].max()))
        if len(d["overflow_samples"]) < 16:
            for index in torch.nonzero(exceeds)[:16 - len(d["overflow_samples"])].tolist():
                d["overflow_samples"].append({
                    "index": coordinate(index) if coordinate else index,
                    "reference": str(reference[tuple(index)].item()),
                    "cast": str(cast[tuple(index)].item()),
                })
        if observed is not None:
            observed = observed.to(device="cpu", dtype=torch.float64)
            if observed.shape != reference.shape:
                raise ValueError("Observed/reference output tile shapes differ")
            good = torch.isfinite(observed)
            d["compared_elements"] += observed.numel()
            d["observed_nonfinite"] += int((~good).sum())
            d["observed_nonfinite_with_finite_cast_reference"] += int((~good & cast_finite).sum())
            d["observed_finite_with_nonfinite_cast_reference"] += int((good & ~cast_finite).sum())
            for source, source_finite, field in (
                (reference, finite, "max_abs_error_vs_fp64_finite_pairs"),
                (cast, cast_finite, "max_abs_error_vs_cast_reference_finite_pairs"),
            ):
                both = good & source_finite
                if bool(both.any()):
                    d[field] = max(d[field], float((observed[both] - source[both]).abs().max()))

    def result(self):
        return {key: _number(value) if isinstance(value, float) else value
                for key, value in self.data.items()}


def analyze(payload, *, rows="suspicious", max_rows=4096, threshold=1e20,
            chunk_bytes=64 << 20, row_chunk=64, selection=None,
            observation_origin="captured dump"):
    """CPU-only reference API. Inputs are never mutated or cast in full."""
    torch = _torch()
    if max_rows < 0 or row_chunk < 1 or chunk_bytes < 1024 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid reference limits or threshold")
    values = bind_inputs(payload)
    selected, coverage = selection if selection is not None else select_rows(
        payload, values, rows, max_rows, threshold, chunk_bytes)
    operation = payload["operation"]
    x = values["x"]
    m, d = x.shape
    width = max(d, values["dz"].shape[1] if operation == "triton_addmm_bwd" else d)
    if width * 8 * 16 > chunk_bytes:
        raise ValueError("One full-feature reference row exceeds --chunk-mib")
    block_rows = max(1, min(row_chunk, chunk_bytes // (width * 8 * 16)))
    tile = max(1, min(128, int(math.sqrt(chunk_bytes // (8 * 16)))))
    outputs = payload.get("outputs")
    full = coverage["all_rows_included"]
    reduction_scope = "all sample rows" if full else "partial contribution from selected sample rows; not the full parameter gradient"
    names = _OUTPUTS[operation]
    weight_dtype = values["w"].dtype if operation == "triton_addmm_bwd" else (
        values["weight"].dtype if values["learnable"] else x.dtype)
    summaries = [ReferenceSummary(x.dtype, scope="selected sample rows, all columns", observed_available=outputs is not None),
                 ReferenceSummary(weight_dtype, scope=reduction_scope, observed_available=outputs is not None and full),
                 ReferenceSummary(weight_dtype, scope=reduction_scope, observed_available=outputs is not None and full)]

    def batches():
        for start in range(0, selected.numel(), block_rows):
            yield selected[start:start + block_rows]

    start_time = time.monotonic()
    with torch.no_grad():
        if operation == "triton_addmm_bwd":
            w, dz = values["w"], values["dz"]
            p = dz.shape[1]
            for ids in batches():
                # All reduction columns participate, even for one selected row.
                for a in range(0, d, tile):
                    stop_a = min(a + tile, d)
                    ref = torch.zeros((ids.numel(), stop_a - a), dtype=torch.float64)
                    for b in range(0, p, tile):
                        ref += dz[ids, b:b + tile].to(torch.float64) @ w[a:stop_a, b:b + tile].to(torch.float64).T
                    observed = outputs[0][ids, a:stop_a] if outputs is not None else None
                    summaries[0].add(ref, observed, lambda ij: [int(ids[ij[0]]), a + ij[1]])
            for a in range(0, d, tile):
                for b in range(0, p, tile):
                    ref = torch.zeros((min(tile, d - a), min(tile, p - b)), dtype=torch.float64)
                    for ids in batches():
                        ref += x[ids, a:a + tile].to(torch.float64).T @ dz[ids, b:b + tile].to(torch.float64)
                    observed = outputs[1][a:a + tile, b:b + tile] if outputs is not None and full else None
                    summaries[1].add(ref, observed, lambda ij: [a + ij[0], b + ij[1]])
            if values["is_y_1d"]:
                for b in range(0, p, tile):
                    ref = torch.zeros(min(tile, p - b), dtype=torch.float64)
                    for ids in batches():
                        ref += dz[ids, b:b + tile].to(torch.float64).sum(dim=0)
                    summaries[2].add(ref, outputs[2][b:b + tile] if outputs is not None and full else None,
                                     lambda ij: [b + ij[0]])
            else:
                summaries[2] = ReferenceSummary(dz.dtype, scope="selected sample rows, all columns", observed_available=outputs is not None)
                for ids in batches():
                    summaries[2].add(dz[ids].to(torch.float64), outputs[2][ids] if outputs is not None else None,
                                     lambda ij: [int(ids[ij[0]]), ij[1]])
        else:
            learnable = values["learnable"]
            dw = torch.zeros(d, dtype=torch.float64) if learnable else None
            db = torch.zeros(d, dtype=torch.float64) if learnable else None
            for ids in batches():
                x64 = x[ids].to(torch.float64)
                dy64 = values["dy"][ids].to(torch.float64)
                rstd64 = values["rstd"][ids].to(torch.float64).reshape(-1, 1)
                h = (x64 - values["mean"][ids].to(torch.float64).reshape(-1, 1)) * rstd64
                v = dy64 * values["weight"].to(torch.float64) if learnable else dy64
                ref = (v - h * (h * v).mean(dim=1, keepdim=True) - v.mean(dim=1, keepdim=True)) * rstd64
                summaries[0].add(ref, outputs[0][ids] if outputs is not None else None,
                                 lambda ij: [int(ids[ij[0]]), ij[1]])
                if learnable:
                    dw += (dy64 * h).sum(dim=0)
                    db += dy64.sum(dim=0)
            if learnable:
                summaries[1].add(dw, outputs[1] if outputs is not None and full else None)
                summaries[2].add(db, outputs[2] if outputs is not None and full else None)
            else:
                summaries[1:] = [None, None]
    return {
        "operation": operation, "stage": payload.get("stage"), "attempt": payload.get("attempt"),
        "layer": payload.get("layer"), "observed_outputs_available": outputs is not None,
        "observed_outputs_source": observation_origin,
        "selection": coverage, "threshold": threshold, "chunk_bytes": chunk_bytes,
        "row_chunk": block_rows, "seconds": time.monotonic() - start_time,
        "references": {name: summary.result() if summary is not None else None
                       for name, summary in zip(names, summaries)},
        "interpretation": [
            "FP64 references use captured input values; they are not bitwise GPU oracles.",
            "Layer norm uses captured mean/rstd; eps does not recompute those statistics.",
            "FP32 GPU intermediates/reduction order can overflow or round differently from FP64.",
            "Outside finite dtype range and nonfinite after casting are separate measurements.",
            "Only all-row parameter-gradient references may be compared with captured parameter gradients.",
        ],
    }


def restore_raw_tree(tree, *, device, chunk_bytes=64 << 20, max_bytes=32 << 30):
    """Fresh complete backing storages; preserve aliases, strides and offsets."""
    torch = _torch()
    storages = {}
    for tensor in _tensors(tree):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("Raw restore requires CPU strided, nonquantized source tensors")
        storage = tensor.untyped_storage()
        storages.setdefault(storage._cdata, storage)
    total = sum(storage.nbytes() for storage in storages.values())
    if chunk_bytes <= 0 or total > max_bytes:
        raise ValueError(f"Input storages need {total} bytes; restore limit is {max_bytes}")
    copies = {}
    for key, storage in storages.items():
        source = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
        target = torch.empty(storage.nbytes(), dtype=torch.uint8, device=device)
        for start in range(0, storage.nbytes(), chunk_bytes):
            target[start:start + chunk_bytes].copy_(source[start:start + chunk_bytes])
        copies[key] = target

    def rebuild(value):
        if isinstance(value, torch.Tensor):
            backing = copies[value.untyped_storage()._cdata]
            result = torch.empty(0, dtype=value.dtype, device=device).set_(
                backing.untyped_storage(), value.storage_offset(), value.shape, value.stride())
            if value.is_conj():
                result = result.conj()
            if value.is_neg():
                result = torch._neg_view(result)
            return result
        if isinstance(value, dict):
            return {key: rebuild(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(rebuild(item) for item in value)
        if isinstance(value, list):
            return [rebuild(item) for item in value]
        return value
    return rebuild(tree), total


def _logical_chunks(tensor, elements):
    if tensor.numel() <= elements:
        yield tensor
    else:
        axis = next(i for i, size in enumerate(tensor.shape) if size > 1)
        width = max(1, elements // (tensor.numel() // tensor.shape[axis]))
        for start in range(0, tensor.shape[axis], width):
            yield from _logical_chunks(tensor.narrow(axis, start, min(width, tensor.shape[axis] - start)), elements)


def output_digest(tensor, chunk_bytes, threshold):
    if tensor is None:
        return None
    torch = _torch()
    digest = hashlib.sha256()
    nonfinite = extreme = 0
    largest = 0.0
    for chunk in _logical_chunks(tensor, max(1, chunk_bytes // 32)):
        cpu = chunk.detach().to("cpu").resolve_conj().resolve_neg().contiguous()
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        value = cpu.to(torch.float64)
        finite = torch.isfinite(value)
        nonfinite += int((~finite).sum())
        extreme += int((finite & (value.abs() > threshold)).sum())
        if bool(finite.any()):
            largest = max(largest, float(value[finite].abs().max()))
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "logical_sha256": digest.hexdigest(),
            "nonfinite_count": nonfinite, "extreme_count": extreme,
            "max_abs_finite": _number(largest)}


def replay_gpu(payload, *, repeats=3, chunk_bytes=64 << 20, max_bytes=32 << 30,
               threshold=1e20, rows="suspicious", max_rows=4096, row_chunk=64, progress=None):
    """Replay full operations only. Caller restores stack environment first."""
    torch = _torch()
    values = bind_inputs(payload)
    operation = payload["operation"]
    module_name = "triton_addmm" if operation == "triton_addmm_bwd" else "triton_layer_norm"
    module = importlib.import_module("generative_recommenders.ops.triton." + module_name)
    function = getattr(module, operation)
    captured = payload.get("outputs")
    expected = [output_digest(t, chunk_bytes, threshold) for t in captured] if captured is not None else None
    # Select from pristine CPU evidence once. Fresh GPU outputs are compared on
    # those exact rows, without scanning/FP64-casting a full GPU tensor.
    selection = select_rows(payload, values, rows, max_rows, threshold, chunk_bytes)
    result = []
    first = None

    def phase(name, repeat):
        if progress is not None:
            progress({"phase": name, "repeat": repeat, "attempts": list(result)})

    for repeat in range(repeats):
        phase("restore_inputs", repeat)
        fresh, copied_bytes = restore_raw_tree({"args": payload["args"], "kwargs": payload["kwargs"]},
                                              device="cuda:0", chunk_bytes=chunk_bytes, max_bytes=max_bytes)
        torch.cuda.synchronize()
        phase("execute_operation", repeat)
        started = time.monotonic()
        with torch.no_grad():
            outputs = function(*fresh["args"], **fresh["kwargs"])
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        phase("summarize_outputs", repeat)
        observed = [output_digest(t, chunk_bytes, threshold) for t in outputs]
        phase("compare_cpu_reference", repeat)
        reference = analyze({**payload, "outputs": outputs}, selection=selection,
                            threshold=threshold, chunk_bytes=chunk_bytes, row_chunk=row_chunk,
                            observation_origin=f"fresh GPU replay repeat {repeat}")

        def same(left, right):
            if left is None or right is None:
                return left is right
            return all(left[k] == right[k] for k in ("shape", "dtype", "logical_sha256"))

        result.append({"repeat": repeat, "fresh_input_storage_bytes": copied_bytes,
                       "operation_seconds": seconds, "outputs": dict(zip(_OUTPUTS[operation], observed)),
                       "fp64_reference_comparison": reference,
                       "matches_captured_logical_bytes": [same(a, b) for a, b in zip(observed, expected)] if expected is not None else None,
                       "matches_first_repeat_logical_bytes": [same(a, b) for a, b in zip(observed, first)] if first is not None else None})
        if first is None:
            first = observed
        del fresh, outputs
        phase("complete_repeat", repeat)
    return result


def self_test():
    """Persisted CPU mocks; importing GPU operator modules is unnecessary."""
    torch = _torch()
    rng = torch.get_rng_state().clone()
    raw = torch.arange(72, dtype=torch.float64).reshape(6, 12) / 16
    x, dz = raw[:, 1:7:2], raw[:, 2:10:2]
    w = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 8
    outputs = (dz @ w.T, x.T @ dz, dz.sum(0))
    payload = {"operation": "triton_addmm_bwd", "args": (),
               "kwargs": {"x": x, "w": w, "dz": dz, "is_y_1d": True},
               "outputs": outputs, "stage": "output"}
    report = analyze(payload, rows="all", max_rows=0, chunk_bytes=4096, row_chunk=2)
    for value in report["references"].values():
        assert value["max_abs_error_vs_fp64_finite_pairs"] < 1e-12
        assert value["compared_elements"] == value["elements"]
    finite_control = analyze(payload, chunk_bytes=4096, row_chunk=2)
    assert finite_control["selection"]["analyzed_rows"] == 6
    assert "control rows" in finite_control["selection"]["reason"]
    copied, _ = restore_raw_tree(payload["kwargs"], device="cpu", chunk_bytes=13)
    assert copied["x"].stride() == x.stride() and copied["x"].storage_offset() == x.storage_offset()
    assert copied["x"].untyped_storage()._cdata == copied["dz"].untyped_storage()._cdata
    copied["x"].fill_(99)
    fresh, _ = restore_raw_tree(payload["kwargs"], device="cpu", chunk_bytes=17)
    assert torch.equal(fresh["x"], x) and not torch.equal(copied["x"], x)
    fresh["dz"][2, 1] += 5
    complete_backing = torch.empty(0, dtype=torch.float64).set_(fresh["x"].untyped_storage(), 0, (72,), (1,))
    complete_backing[0] += 7  # Outside both exposed input views.
    comparison = compare_boundaries(payload, {**payload, "kwargs": fresh, "stage": "finite"}, chunk_bytes=4096)
    delta = comparison["tensors"]["input/dz"]
    assert delta["mismatched_elements"] == 1 and delta["affected_axis0_range"] == [2, 2]
    assert delta["max_abs_difference_finite_pairs"] == 5
    unchanged_view = comparison["tensors"]["input/x"]
    assert unchanged_view["equal_logical_bytes"]
    assert not comparison["storage_pairs"][unchanged_view["storage_pair"]]["equal"]
    bit_left = torch.tensor([0x7fc00001, 0, -2147483648], dtype=torch.int32).view(torch.float32)
    bit_right = torch.tensor([0x7fc00002, -2147483648, -2147483648], dtype=torch.int32).view(torch.float32)
    bits = compare_tensors(bit_left, bit_right, chunk_bytes=1024, axis0="test")
    assert bits["mismatched_elements"] == 2 and bits["mismatched_bytes"] == 2
    assert bits["nonfinite_left"] == bits["nonfinite_right"] == 1
    assert bits["max_abs_difference_finite_pairs"] == 0
    try:
        restore_raw_tree(payload["kwargs"], device="cpu", max_bytes=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Restore byte cap ignored")

    x = torch.arange(24, dtype=torch.float64).reshape(6, 4) / 8
    dy = torch.arange(24, dtype=torch.float64).reshape(6, 4) / 16
    weight = torch.tensor([0.5, 1., -1., 2.], dtype=torch.float64)
    xx = x.clone().requires_grad_()
    ww = weight.clone().requires_grad_()
    bb = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    y = torch.nn.functional.layer_norm(xx, (4,), ww, bb, 1e-6)
    expected = torch.autograd.grad(y, (xx, ww, bb), dy)
    mean = x.mean(1)
    rstd = torch.rsqrt(x.var(1, unbiased=False) + 1e-6)
    ln = {"operation": "triton_weighted_layer_norm_bwd", "args": (), "stage": "output",
          "kwargs": dict(dy=dy, x=x, weight=weight, bias=bb.detach(), mean=mean,
                         rstd=rstd, learnable=True, eps=1e-6, BLOCK_D=4), "outputs": expected}
    report = analyze(ln, rows="all", max_rows=0, chunk_bytes=4096, row_chunk=2)
    assert all(value["max_abs_error_vs_fp64_finite_pairs"] < 1e-12 for value in report["references"].values())
    partial = analyze(ln, rows="1,3:5", chunk_bytes=4096, row_chunk=1)
    assert partial["selection"]["analyzed_rows"] == 3
    assert partial["references"]["d_norm_weight"]["compared_elements"] == 0

    ln["outputs"] = None
    ln["stage"] = "input"
    ln["kwargs"] = dict(dy=torch.tensor([[1e37, -1e37, 1e37, -1e37]], dtype=torch.bfloat16),
                         x=torch.zeros(1, 4, dtype=torch.bfloat16), weight=torch.full((4,), 100., dtype=torch.bfloat16),
                         bias=torch.zeros(4, dtype=torch.bfloat16), mean=torch.zeros(1), rstd=torch.full((1,), 100.),
                         learnable=True, eps=1e-6, BLOCK_D=4)
    extreme = analyze(ln, chunk_bytes=4096)
    assert extreme["selection"]["first_sample_rows"] == [0]
    assert extreme["references"]["d_x"]["outside_output_finite_range"] == 4
    assert extreme["references"]["d_x"]["cast_nonfinite"] == 4
    check = ReferenceSummary(torch.bfloat16, scope="test", observed_available=False)
    check.add(torch.tensor([torch.finfo(torch.bfloat16).max * 1.001], dtype=torch.float64))
    assert check.data["outside_output_finite_range"] == 1 and check.data["cast_nonfinite"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    print("PASS: CPU GEMM/LN references, finite/input-only dumps, partial scope, BF16 overflow rounding, byte/NaN/signed-zero and unused-storage comparisons, raw aliases/fresh restore/cap, unchanged RNG; CUDA uninitialized.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", nargs="?", type=Path, help="Trusted local .pt produced by nan_backward_boundaries.py")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare-other", type=Path, help="Compare all input/output bytes with another dump of this operation")
    parser.add_argument("--compare-only", action="store_true", help="Only compare the two dumps; skip FP64 references")
    parser.add_argument("--rows", default="suspicious", help="suspicious, all, or indices/ranges such as 7,20:30")
    parser.add_argument("--max-rows", type=int, default=4096, help="Reference row limit; 0 means unlimited")
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--threshold", type=float, help="Suspicious magnitude; defaults to dump threshold or 1e20")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--gpu", action="store_true", help="Additionally replay the complete original GPU operation")
    parser.add_argument("--context", type=Path, help="Original full-state capture context.json; required for --gpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-input-gib", type=int, default=32,
                        help="Cap restored input backing storages only; GPU outputs and kernel scratch are additional")
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run only embedded CPU mocks")
    args = parser.parse_args()
    if args.self_test:
        if args.gpu:
            parser.error("--self-test never executes GPU operations")
        self_test()
        return
    if args.dump is None:
        parser.error("a trusted dump path is required")
    if min(args.chunk_mib, args.row_chunk, args.cpu_threads, args.repeats, args.max_input_gib) < 1 or args.max_rows < 0:
        parser.error("limits must be positive (--max-rows may be zero)")
    if args.gpu and args.context is None:
        parser.error("--gpu requires the original capture --context for stack/environment checks")
    if args.compare_only and (args.compare_other is None or args.gpu):
        parser.error("--compare-only requires --compare-other and cannot be combined with --gpu")
    context = json.loads(args.context.read_text()) if args.context else None
    changed = []
    root = Path(__file__).resolve().parents[1]
    if args.gpu:
        for key, value in context["environment"].items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, expected in context["source_sha256"].items():
            if name.startswith("generative_recommenders/"):
                path = root / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    changed.append(name)
        if changed and not args.allow_source_change:
            parser.error(f"captured model source differs: {changed}; intentional bisection needs --allow-source-change")
        sys.path.insert(0, str(root))
    torch = _torch()
    torch.set_num_threads(args.cpu_threads)
    versions = {"python": platform.python_version(), "torch": torch.__version__, "hip": torch.version.hip}
    for package in ("triton", "numpy"):
        try:
            versions[package + "_distribution"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package + "_distribution"] = None
    if args.gpu:
        import triton
        versions["triton"] = triton.__version__
        for name in ("torch", "hip", "triton", "triton_distribution"):
            if name in context and context[name] != versions[name]:
                parser.error(f"stack mismatch for {name}: captured {context[name]!r}, current {versions[name]!r}")
    # mmap preserves full backing storages/aliases without eagerly copying a
    # multi-GB dump. No tensor is modified, including when replaying on GPU.
    payload = torch.load(args.dump, map_location="cpu", weights_only=False, mmap=True)
    thresholds = [item["abs_threshold"] for item in payload.get("event", {}).get("inputs", []) if "abs_threshold" in item]
    threshold = args.threshold if args.threshold is not None else (thresholds[0] if thresholds else 1e20)
    report = {"dump": str(args.dump), "versions": versions,
              "original_context": str(args.context) if args.context else None,
              "original_versions": {key: context.get(key) for key in ("torch", "hip", "triton", "triton_distribution")} if context else None,
              "restored_environment": context["environment"] if args.gpu else None,
              "changed_model_source": changed, "gpu_validation": "not requested"}
    report_path = args.report or args.dump.with_suffix(".analysis.json")

    def save():
        temporary = report_path.with_name(report_path.name + ".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)

    if args.compare_other:
        other = torch.load(args.compare_other, map_location="cpu", weights_only=False, mmap=True)
        report["compare_other"] = str(args.compare_other)
        report["boundary_comparison"] = compare_boundaries(payload, other, chunk_bytes=args.chunk_mib << 20)
        save()
    if not args.compare_only:
        report["cpu_reference"] = analyze(payload, rows=args.rows, max_rows=args.max_rows, threshold=threshold,
                                          chunk_bytes=args.chunk_mib << 20, row_chunk=args.row_chunk)
    save()  # Preserve CPU findings even if the optional GPU operation hangs.
    if args.gpu:
        report["gpu_validation"] = "in progress"
        report["gpu_input_storage_limit_bytes"] = args.max_input_gib << 30
        report["gpu_memory_limit_note"] = "Input storages only; full operation outputs and kernel scratch are additional allocations."
        save()

        def progress(state):
            report["gpu_progress"] = {key: value for key, value in state.items() if key != "attempts"}
            report["gpu_attempts"] = state["attempts"]
            save()

        try:
            report["gpu_attempts"] = replay_gpu(payload, repeats=args.repeats, threshold=threshold,
                                                chunk_bytes=args.chunk_mib << 20, max_bytes=args.max_input_gib << 30,
                                                rows=args.rows, max_rows=args.max_rows, row_chunk=args.row_chunk,
                                                progress=progress)
            report["gpu_validation"] = "completed; compare summaries and hashes, not merely process exit"
        except Exception as error:
            report["gpu_validation"] = "failed"
            report["gpu_error"] = {"type": type(error).__name__, "message": str(error)}
            save()
            raise
        save()
    brief = {"report": str(report_path), "gpu_validation": report["gpu_validation"]}
    if "cpu_reference" in report:
        brief.update(selection=report["cpu_reference"]["selection"], references=report["cpu_reference"]["references"])
    if "boundary_comparison" in report:
        brief["comparison"] = {
            name: {key: value.get(key) for key in ("equal_logical_bytes", "mismatched_bytes", "mismatched_elements",
                                                  "max_abs_difference_finite_pairs", "max_difference_location", "affected_axis0_range")}
            for name, value in report["boundary_comparison"]["tensors"].items()}
    print(json.dumps(brief, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
