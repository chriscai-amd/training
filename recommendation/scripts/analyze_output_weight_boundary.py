#!/usr/bin/env python3
"""CPU FP64 reference for trusted hstu_output_weight_mm captures.

Weight row i is a Y feature: dW[i,c] = sum_n Y[n,i] * dout[n,c].
Selected weight rows always include every N reduction row and every output
column. This is an independent mathematical reference, not the original GPU
kernel/precision replay. --allow-postcall-inputs permits only explicitly labeled
conditional analysis of fault-time inputs; it never recertifies them pristine.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import replay_backward_boundary as common
from replay_hstu_output_boundary import ReferenceSummary, input_digests


OPERATION = "hstu_output_weight_mm"


def _provenance(payload, allow_postcall_inputs):
    event = payload.get("event", {})
    markers = [source["input_snapshot_pristine"] for source in (payload, event)
               if "input_snapshot_pristine" in source]
    if any(type(value) is not bool for value in markers) or len(set(markers)) > 1:
        raise ValueError("Conflicting or malformed pristine input provenance")
    snapshot = event.get("input_snapshot_observation", {}).get("source")
    observation = event.get("input_observation", {}).get("source")
    postcall = bool(markers and markers[0] is False) or snapshot == "post_call_fault_cpu_snapshot"
    if postcall and ((markers and markers[0] is True) or payload.get("stage") == "input"
                     or event.get("operation_executed") is False or observation == "pristine_pre_call_cpu_snapshot"):
        raise ValueError("Conflicting post-call and pristine input provenance")
    if postcall:
        if not allow_postcall_inputs:
            common.require_pristine_input_snapshot(payload)
            raise ValueError("Post-call input snapshots require --allow-postcall-inputs")
    else:
        common.require_pristine_input_snapshot(payload)
        if not (markers and markers[0] is True) and observation != "pristine_pre_call_cpu_snapshot":
            raise ValueError("Input snapshot has no explicit pristine or post-call provenance")
    return {
        "input_snapshot_pristine": not postcall,
        "conditional_reference": postcall,
        "reference_basis": "saved_postcall_inputs" if postcall else "pristine_pre_call_inputs",
        "original_call_replay_valid": False,
        "original_call_inputs_retained": not postcall,
        "interpretation": ("Conditional mathematical reference of post-call inputs. Relating it to the original output assumes input bytes did not change during that call; this is unverified."
                           if postcall else "Mathematical FP64 reference of captured pre-call inputs; the GPU kernel, reduction order and arithmetic controls are not reproduced."),
    }


def _controls(payload):
    first = payload.get("execution_controls")
    second = payload.get("event", {}).get("execution_controls")
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("Output-weight capture requires top-level and event execution controls")
    if first != second:
        raise ValueError("Conflicting captured execution controls")
    controls = first
    if controls.get("float32_matmul_precision") not in ("highest", "high", "medium"):
        raise ValueError("Invalid captured float32 matmul precision")
    for name in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        if type(controls.get(name)) is not bool:
            raise ValueError(f"Invalid captured execution control {name}")
    autocast = controls.get("autocast")
    if not isinstance(autocast, dict):
        raise ValueError("Missing captured autocast controls")
    for device in ("cpu", "cuda"):
        options = autocast.get(device, {})
        if (type(options.get("enabled")) is not bool
                or options.get("dtype") not in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64")
                or options["enabled"] and options["dtype"] not in ("torch.float16", "torch.bfloat16")):
            raise ValueError(f"Invalid captured {device} autocast controls")
    return controls


def _tensor(value, name):
    torch = common._torch()
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.layout != torch.strided or value.is_quantized
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError(f"{name} must be a CPU strided real floating-point tensor")


def _validate_metadata(entries, values):
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise ValueError("Captured tensor metadata must be a list")
    if any(not isinstance(entry, dict) or not isinstance(entry.get("name"), str) for entry in entries):
        raise ValueError("Malformed captured tensor metadata")
    by_name = {entry["name"]: entry for entry in entries}
    if len(by_name) != len(entries) or set(by_name) != set(values):
        raise ValueError("Captured tensor metadata names disagree with operation arguments")
    for name, tensor in values.items():
        actual = common._spec(tensor)
        for key in ("shape", "stride", "dtype", "storage_offset", "storage_bytes", "is_conj", "is_neg"):
            if key in by_name[name] and by_name[name][key] != actual[key]:
                raise ValueError(f"Captured {name} metadata disagrees with tensor {key}")
    return by_name


def bind_inputs(payload, *, allow_postcall_inputs=False):
    torch = common._torch()
    if payload.get("operation") != OPERATION or payload.get("format_version", 1) != 1:
        raise ValueError("Expected hstu_output_weight_mm capture format version 1")
    provenance = _provenance(payload, allow_postcall_inputs)
    controls = _controls(payload)
    signature = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                   for name in ("y", "dout")])
    values = dict(signature.bind(*payload.get("args", ()), **payload.get("kwargs", {})).arguments)
    for name, tensor in values.items():
        _tensor(tensor, name)
    y, dout = values["y"], values["dout"]
    if y.ndim != 2 or dout.ndim != 2 or y.shape[0] != dout.shape[0] or min(y.shape[1], dout.shape[1]) <= 0:
        raise ValueError("Expected y [N,D] and dout [N,C] with positive feature dimensions")
    if y.dtype != dout.dtype:
        raise ValueError("This production-path reference requires matching input dtypes")
    metadata = _validate_metadata(payload.get("event", {}).get("inputs"), values)
    source_devices = {str(entry["source_device"])
                      for entry in metadata.values() if "source_device" in entry}
    device_types = {device.split(":", 1)[0] for device in source_devices}
    if len(source_devices) > 1 or device_types - {"cpu", "cuda"}:
        raise ValueError("Captured input source devices are incompatible")
    for device in source_devices:
        try:
            torch.device(device)
        except (ValueError, RuntimeError) as error:
            raise ValueError("Malformed captured source device") from error
    devices = device_types or {"cpu", "cuda"}
    possible_output_dtypes = {
        controls["autocast"][device]["dtype"]
        if y.dtype != torch.float64 and controls["autocast"][device]["enabled"] else str(y.dtype)
        for device in devices
    }
    outputs = payload.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
            raise ValueError("Expected one d_output_weight tensor or outputs=None")
        _tensor(outputs[0], "d_output_weight")
        if outputs[0].shape != (y.shape[1], dout.shape[1]) or str(outputs[0].dtype) not in possible_output_dtypes:
            raise ValueError("Captured d_output_weight shape/dtype disagrees with inputs and autocast controls")
        output_metadata = _validate_metadata(payload.get("event", {}).get("outputs"), {"d_output_weight": outputs[0]})
        output_device = output_metadata.get("d_output_weight", {}).get("source_device")
        if output_device is not None and source_devices and str(output_device) not in source_devices:
            raise ValueError("Captured output source device disagrees with inputs")
    if payload.get("stage") == "input" and outputs is not None:
        raise ValueError("Input-stage capture must not contain executed outputs")
    if outputs is None and len(possible_output_dtypes) != 1:
        raise ValueError("Missing source device makes the input-stage output dtype ambiguous")
    provenance.update(original_source_device=next(iter(source_devices)) if source_devices else None,
                      possible_output_dtypes=sorted(possible_output_dtypes))
    return values, provenance, controls


def select_rows(payload, values, *, rows, max_rows, threshold, chunk_bytes):
    torch = common._torch()
    count = values["y"].shape[1]
    reason = rows
    if rows == "all":
        selected = torch.arange(count)
    elif rows == "suspicious":
        observed = payload.get("outputs")
        selected = (torch.nonzero(common._row_mask(observed[0], threshold, chunk_bytes)).flatten()
                    if observed is not None else torch.empty(0, dtype=torch.int64))
        if not selected.numel():
            selected = torch.arange(min(16, count))
            reason = "No suspicious output-weight rows available; first 16 feature rows are controls"
    else:
        ids = set()
        for item in rows.split(","):
            fields = item.split(":")
            if len(fields) == 1:
                start, end = int(item), int(item) + 1
            elif len(fields) == 2:
                start, end = map(int, fields)
            else:
                raise ValueError("Weight rows use indices or start:stop ranges")
            if not 0 <= start < end <= count:
                raise ValueError("Weight-row selection outside Y feature dimensions")
            ids.update(range(start, end))
        selected = torch.tensor(sorted(ids), dtype=torch.int64)
    eligible = selected.numel()
    if max_rows:
        selected = selected[:max_rows]
    if not selected.numel():
        raise ValueError("Select at least one output-weight row")
    return selected, {"requested": rows, "reason": reason, "axis0_semantics": "Y features / output-weight rows",
                      "total_weight_rows": count, "eligible_rows": eligible,
                      "analyzed_rows": selected.numel(), "selected_rows": selected.tolist(),
                      "truncated": selected.numel() < eligible, "all_weight_rows_included": selected.numel() == count}


def reference_rows(values, rows, *, reduction_chunk=8192, feature_chunk=16,
                   column_chunk=128, chunk_bytes=64 << 20):
    torch = common._torch()
    y, dout = values["y"], values["dout"]
    if min(reduction_chunk, feature_chunk, column_chunk, chunk_bytes) <= 0:
        raise ValueError("Reference tile sizes and chunk bytes must be positive")
    if (rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 1 or not rows.numel()
            or bool((rows < 0).any()) or bool((rows >= y.shape[1]).any())
            or rows.unique().numel() != rows.numel()):
        raise ValueError("Reference rows must be unique valid CPU int64 Y-feature indices")
    n, c, r = y.shape[0], dout.shape[1], rows.numel()
    f, cc = min(feature_chunk, r), min(column_chunk, c)
    reference_bytes = r * c * 8
    # Conservative tensor bound includes native gather/conversion overlap,
    # partial matmul, tiled summary temporaries and the retained final reference.
    # It excludes input storage, Python/BLAS internal workspaces and allocator
    # caching. No whole N-by-D conversion is made.
    available = chunk_bytes - reference_bytes - 128 * f * cc
    effective_n = min(reduction_chunk, available // (16 * (f + cc)))
    if effective_n < 1:
        raise ValueError("Reference output/tile exceeds chunk memory budget")
    result = torch.zeros((r, c), dtype=torch.float64)
    zero_y = torch.ones(r, dtype=torch.bool)
    zero_dout = torch.ones(c, dtype=torch.bool)
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        for first in range(0, r, feature_chunk):
            ids = rows[first:first + feature_chunk]
            for col in range(0, c, column_chunk):
                width = min(column_chunk, c - col)
                total = result[first:first + ids.numel(), col:col + width]
                for start in range(0, n, effective_n):
                    end = min(start + effective_n, n)
                    # Narrow N before gathering selected D indices, and narrow
                    # C before dtype conversion; preserve saved view semantics.
                    left = y[start:end].index_select(1, ids).to(torch.float64)
                    right = dout[start:end, col:col + width].to(torch.float64)
                    zero_y[first:first + ids.numel()] &= ~(left != 0).any(dim=0)
                    zero_dout[col:col + width] &= ~(right != 0).any(dim=0)
                    total.add_(left.t() @ right)
                    del left, right
    return result, {"reduction_rows": n, "all_reduction_rows_included": True,
                    "output_columns": c, "all_output_columns_included": True,
                    "effective_reduction_chunk": effective_n,
                    "reduction_chunks_per_output_tile": math.ceil(n / effective_n),
                    "feature_chunk": feature_chunk, "column_chunk": column_chunk,
                    "retained_reference_bytes": reference_bytes,
                    "estimated_workspace_bound_bytes": reference_bytes + 128 * f * cc + effective_n * 16 * (f + cc),
                    "workspace_bound_scope": "Explicit reference/summary tensors; excludes captured input storage, Python/BLAS internals and allocator caching",
                    "selected_y_features_identically_zero": zero_y.tolist(),
                    "dout_columns_identically_zero": zero_dout.tolist(),
                    "accumulation": "FP64 partial GEMMs plus FP64 sum across every N chunk; final dtype cast occurs only after the complete reduction"}


def analyze(payload, *, rows="suspicious", max_rows=16, threshold=1e6, chunk_bytes=64 << 20,
            reduction_chunk=8192, feature_chunk=16, column_chunk=128, allow_postcall_inputs=False):
    torch = common._torch()
    if max_rows < 0 or chunk_bytes <= 0 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid row limit, chunk budget or finite threshold")
    started = time.monotonic()
    values, provenance, controls = bind_inputs(payload, allow_postcall_inputs=allow_postcall_inputs)
    observed = payload.get("outputs")
    tree = {**values, **({"d_output_weight": observed[0]} if observed is not None else {})}
    before = input_digests(tree, chunk_bytes=chunk_bytes, threshold=threshold)
    selected, selection = select_rows(payload, values, rows=rows, max_rows=max_rows,
                                      threshold=threshold, chunk_bytes=chunk_bytes)
    reference, reduction = reference_rows(values, selected, reduction_chunk=reduction_chunk,
        feature_chunk=feature_chunk, column_chunk=column_chunk, chunk_bytes=chunk_bytes)
    dtype = observed[0].dtype if observed is not None else getattr(torch, provenance["possible_output_dtypes"][0].split(".")[-1])
    summary = ReferenceSummary(dtype, scope="selected output-weight rows; all N reduction rows and all C columns",
                               observed_available=observed is not None)
    inputs_finite = all(before["inputs"][name]["nonfinite_count"] == 0 for name in ("y", "dout"))
    # Finite FP32 inputs can overflow when autocast converts them to FP16/BF16.
    # The saved-input FP64 reference still has meaning, but a zero product is
    # only a GPU arithmetic oracle when the other operand stays finite.
    input_cast_finite = inputs_finite and all(
        before["inputs"][name]["max_abs_finite"] <= torch.finfo(dtype).max for name in ("y", "dout"))
    zero_y = torch.tensor(reduction["selected_y_features_identically_zero"])
    zero_dout = torch.tensor(reduction["dout_columns_identically_zero"])
    zero_count, violations = 0, 0 if observed is not None else None
    for first in range(0, selected.numel(), feature_chunk):
        ids = selected[first:first + feature_chunk]
        for col in range(0, reference.shape[1], column_chunk):
            stop = min(col + column_chunk, reference.shape[1])
            tile = reference[first:first + ids.numel(), col:stop]
            observed_tile = observed[0][:, col:stop].index_select(0, ids) if observed is not None else None
            summary.add(tile, observed_tile, lambda ij: [int(ids[ij[0]]), col + ij[1]])
            if input_cast_finite:
                zero_mask = zero_y[first:first + ids.numel(), None] | zero_dout[None, col:stop]
                zero_count += int(zero_mask.sum())
                if observed_tile is not None:
                    violations += int(((observed_tile != 0) & zero_mask).sum())
            del tile, observed_tile
    after = input_digests(tree, chunk_bytes=chunk_bytes, threshold=threshold)
    if before != after:
        raise RuntimeError("CPU analysis changed captured logical or backing-storage bytes")
    return {"format": "output_weight_cpu_reference_v1", "operation": OPERATION,
            "stage": payload.get("stage"), "layer": payload.get("layer"), "attempt": payload.get("attempt"),
            "provenance": provenance, "execution_controls": controls,
            "controls_policy": "Captured controls validated and recorded; only CPU autocast is disabled locally. No GPU flags are changed or GPU kernel replay claimed.",
            "selection": selection, "reduction": reduction, "reference": summary.result(),
            "zero_oracle": {"applicable": zero_count > 0, "certified_elements": zero_count,
                            "observed_nonzero_count": violations, "conditional_reference": provenance["conditional_reference"],
                            "saved_inputs_finite": inputs_finite,
                            "effective_input_cast_finiteness_certified": input_cast_finite,
                            "effective_arithmetic_dtype": str(dtype),
                            "rule": "Inputs finite within the effective arithmetic dtype range and an identically zero selected Y feature or dout column imply exact zero for that entire dot product. The range check is conservative for autocast."},
            "tensor_digests": before, "captured_bytes_unchanged_by_analysis": True,
            "immutability_scope": "Before/after CPU analysis only; no assertion that GPU-call inputs were unchanged during execution.",
            "limitations": ["FP64 chunked reduction order differs from the GPU; finite cast-reference differences alone are not a kernel fault certificate.",
                            "Only selected weight rows are compared; every reduction row and output column is included for each selected weight row."],
            "seconds": time.monotonic() - started, "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", default="suspicious", help="Output-weight feature indices, ranges, all, or suspicious")
    parser.add_argument("--max-rows", type=int, default=16, help="0 includes all selected weight rows")
    parser.add_argument("--threshold", type=float, default=1e6)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--reduction-chunk", type=int, default=8192)
    parser.add_argument("--feature-chunk", type=int, default=16)
    parser.add_argument("--column-chunk", type=int, default=128)
    parser.add_argument("--allow-postcall-inputs", action="store_true",
                        help="Explicit conditional reference of saved post-call inputs; never original-call replay")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("--report must be a new path")
    torch = common._torch()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False, mmap=True)
    result = analyze(payload, rows=args.rows, max_rows=args.max_rows, threshold=args.threshold,
                     chunk_bytes=args.chunk_mib << 20, reduction_chunk=args.reduction_chunk,
                     feature_chunk=args.feature_chunk, column_chunk=args.column_chunk,
                     allow_postcall_inputs=args.allow_postcall_inputs)
    result["capture"] = str(args.capture.resolve())
    result["capture_sha256"] = _sha256(args.capture)
    result["analysis_source_sha256"] = {Path(path).name: _sha256(path)
        for path in (__file__, common.__file__, Path(__file__).with_name("replay_hstu_output_boundary.py"))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.report)
    print(json.dumps({"report": str(args.report), "provenance": result["provenance"],
                      "selected_weight_rows": result["selection"]["analyzed_rows"],
                      "reduction_rows": result["reduction"]["reduction_rows"],
                      "zero_oracle": result["zero_oracle"]}, allow_nan=False))


if __name__ == "__main__":
    main()
