#!/usr/bin/env python3
"""Replay trusted HSTU output-backward stage dumps and compare CPU FP64 references.

The default is CPU-only analysis of selected rows, with every feature/reduction
column included. Norm parameter gradients are partial unless all rows are used.
Training norm references require the captured packed dropout mask: this tool
never substitutes regenerated RNG. --gpu independently replays the whole saved
stage with fresh complete backing storages and requires capture context.json.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
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

import replay_backward_boundary as common


MM = "hstu_output_grad_mm"
NORM = "triton_layer_norm_mul_dropout_bwd"
ARGUMENTS = {
    MM: ("dout", "output_weight"),
    NORM: ("dy", "x", "u", "weight", "bias", "mean", "rstd", "BLOCK_D",
           "num_warps", "eps", "training", "dropout_ratio", "seed", "silu_u",
           "concat_u", "concat_x", "mul_u_activation_type", "compute_y", "random_mask"),
}
OUTPUTS = {MM: ("dy",), NORM: ("d_attn", "d_u", "d_norm_weight", "d_norm_bias", "y")}
DEFAULTS = {"seed": None, "silu_u": False, "concat_u": False, "concat_x": False,
            "mul_u_activation_type": "none", "compute_y": False, "random_mask": None}


class UnsupportedReference(ValueError):
    """Saved inputs do not support the independent CPU formula."""


class ReferenceSummary(common.ReferenceSummary):
    """Also distinguish opposite-signed infinity and NaN-position mismatches."""

    def add(self, reference, observed=None, coordinate=None):
        super().add(reference, observed, coordinate)
        self.data.setdefault("nonfinite_pattern_mismatches_vs_cast_reference", 0)
        self.data.setdefault("nonfinite_pattern_mismatch_samples", [])
        if observed is None:
            return
        torch = common._torch()
        observed = observed.to(device="cpu", dtype=torch.float64)
        cast = reference.to(self.dtype).to(torch.float64)
        bad = ((torch.isnan(cast) != torch.isnan(observed)) |
               (torch.isposinf(cast) != torch.isposinf(observed)) |
               (torch.isneginf(cast) != torch.isneginf(observed)))
        self.data["nonfinite_pattern_mismatches_vs_cast_reference"] += int(bad.sum())
        samples = self.data["nonfinite_pattern_mismatch_samples"]
        for index in torch.nonzero(bad)[:16 - len(samples)].tolist():
            samples.append({"index": coordinate(index) if coordinate else index,
                            "cast_reference": str(cast[tuple(index)].item()),
                            "observed": str(observed[tuple(index)].item())})


def bind_inputs(payload):
    """Bind without importing Triton or initializing a GPU."""
    common.require_pristine_input_snapshot(payload)
    torch = common._torch()
    operation = payload.get("operation")
    if operation not in ARGUMENTS:
        raise ValueError(f"Unsupported output-backward operation: {operation!r}")
    signature = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          default=DEFAULTS.get(name, inspect.Parameter.empty))
        for name in ARGUMENTS[operation]
    ])
    bound = signature.bind(*payload.get("args", ()), **payload.get("kwargs", {}))
    bound.apply_defaults()
    values = dict(bound.arguments)
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
                raise ValueError("Operation evidence must be CPU strided nonquantized tensors")
            if name != "random_mask" and not value.is_floating_point():
                raise ValueError(f"{name} must be a real floating-point tensor")
    if operation == MM:
        dout, weight = values["dout"], values["output_weight"]
        if dout.ndim != 2 or weight.ndim != 2 or dout.shape[1] != weight.shape[1]:
            raise ValueError("Expected dout [N,C], output_weight [D,C]")
        if dout.dtype != weight.dtype or min(dout.shape[1], weight.shape[0]) < 1:
            raise ValueError("GEMM requires matching dtypes and positive feature dimensions")
        shapes = [(dout.shape[0], weight.shape[0])]
    else:
        x = values["x"]
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("Expected x [N,D] with positive feature dimension")
        n, d = x.shape
        p = d * (1 + int(values["concat_u"]) + int(values["concat_x"]))
        for name, shape in (("dy", (n, p)), ("u", (n, d)), ("weight", (d,)),
                            ("bias", (d,)), ("mean", (n,)), ("rstd", (n,))):
            if not isinstance(values[name], torch.Tensor) or tuple(values[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if values["mul_u_activation_type"] not in ("none", "silu", "sigmoid"):
            raise ValueError("Unsupported multiplication activation")
        if not 0 <= values["dropout_ratio"] < 1:
            raise ValueError("dropout_ratio must be in [0,1)")
        mask = values["random_mask"]
        if mask is not None and mask.numel():
            if tuple(mask.shape) != (n, d) or mask.dtype not in (torch.int8, torch.uint8, torch.bool):
                raise ValueError("Saved packed random_mask must be int8/uint8/bool [N,D]")
        shapes = [(n, d), (n, d), (d,), (d,), (n, p) if values["compute_y"] else None]
    outputs = payload.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(shapes):
            raise ValueError(f"Expected {len(shapes)} stage outputs, or outputs=None")
        for name, value, shape in zip(OUTPUTS[operation], outputs, shapes):
            if shape is None:
                if value is not None:
                    raise ValueError(f"{name} must be None when compute_y=False")
            elif not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
    return values


def select_rows(payload, values, rows, max_rows, threshold, chunk_bytes):
    torch = common._torch()
    operation = payload["operation"]
    names = ("dout",) if operation == MM else ("x", "u", "dy", "mean", "rstd")
    count = values[names[0]].shape[0]
    reason = rows
    if rows == "all":
        selected = torch.arange(count)
    elif rows == "suspicious":
        bad = torch.zeros(count, dtype=torch.bool)
        for name in names:
            bad |= common._row_mask(values[name], threshold, chunk_bytes)
        for index, output in enumerate(payload.get("outputs") or ()):
            if output is not None and (operation == MM or index in (0, 1, 4)):
                bad |= common._row_mask(output, threshold, chunk_bytes)
        selected = torch.nonzero(bad).flatten()
        if not selected.numel():
            selected = torch.arange(min(16, count))
            reason = "no suspicious rows; first 16 control rows"
    else:
        mask = torch.zeros(count, dtype=torch.bool)
        for item in rows.split(","):
            bounds = item.split(":")
            if len(bounds) == 1:
                start, stop = int(item), int(item) + 1
            elif len(bounds) == 2:
                start, stop = map(int, bounds)
            else:
                raise ValueError("Rows use indices or start:stop ranges")
            if not 0 <= start <= stop <= count:
                raise ValueError(f"Row selection {item!r} outside [0,{count})")
            mask[start:stop] = True
        selected = torch.nonzero(mask).flatten()
    eligible = selected.numel()
    if max_rows:
        selected = selected[:max_rows]
    return selected, {"requested": rows, "reason": reason, "total_sample_rows": count,
                      "eligible_rows": eligible, "analyzed_rows": selected.numel(),
                      "truncated": selected.numel() < eligible,
                      "all_rows_included": selected.numel() == count,
                      "first_sample_rows": selected[:32].tolist(),
                      "last_sample_rows": selected[-8:].tolist(),
                      "columns": "all; no feature or reduction columns sampled"}


def norm_reference_rows(values, ids):
    """FP64 rowwise formula; parameter results are this tile's contributions."""
    torch = common._torch()
    if values["training"] and (values["random_mask"] is None or not values["random_mask"].numel()):
        raise UnsupportedReference("CPU training reference requires saved packed random_mask; fused RNG is unsupported")
    if values["training"]:
        packed = values["random_mask"][ids].to(torch.int64)
        valid_bits = (1 << (1 + int(values["concat_u"]) + int(values["concat_x"]))) - 1
        if bool(((packed < 0) | ((packed & ~valid_bits) != 0)).any()):
            raise UnsupportedReference(f"Saved random_mask has invalid bits in selected rows (allowed bitmask {valid_bits}); CPU reference is unsupported for malformed masks")
    x, u, dy = (values[name][ids].to(torch.float64) for name in ("x", "u", "dy"))
    weight, bias = (values[name].to(torch.float64) for name in ("weight", "bias"))
    mean = values["mean"][ids].to(torch.float64)[:, None]
    rstd = values["rstd"][ids].to(torch.float64)[:, None]
    h = (x - mean) * rstd
    ln = h * weight + bias
    sigmoid = torch.sigmoid(u)
    silu = u * sigmoid
    dsilu = sigmoid + silu * (1 - sigmoid)
    activation = values["mul_u_activation_type"]
    if activation == "silu":
        mul, derivative = silu, dsilu
    elif activation == "sigmoid":
        mul, derivative = sigmoid, sigmoid * (1 - sigmoid)
    else:
        mul, derivative = u, torch.ones_like(u)
    d = x.shape[1]
    offset = 0
    du_skip = torch.zeros_like(x)
    dx_skip = torch.zeros_like(x)
    if values["concat_u"]:
        du_skip = dy[:, offset:offset + d]
        offset += d
    if values["concat_x"]:
        dx_skip = dy[:, offset:offset + d]
        offset += d
    dmul = dy[:, offset:offset + d]
    u_value = silu if values["silu_u"] else u
    x_value, y_value = x, ln * mul
    if values["training"]:
        scale = 1.0 / (1.0 - values["dropout_ratio"])
        y_keep = (packed & 1) != 0
        dmul = torch.where(y_keep, dmul * scale, 0.)
        y_value = torch.where(y_keep, y_value * scale, 0.)
        if values["concat_u"]:
            bit = 4 if values["concat_x"] else 2
            keep = (packed & bit) != 0
            du_skip = torch.where(keep, du_skip * scale, 0.)
            u_value = torch.where(keep, u_value * scale, 0.)
        if values["concat_x"]:
            keep = (packed & 2) != 0
            dx_skip = torch.where(keep, dx_skip * scale, 0.)
            x_value = torch.where(keep, x_value * scale, 0.)
    dln = dmul * mul
    du = dmul * ln * derivative + du_skip * (dsilu if values["silu_u"] else 1.)
    v = dln * weight
    dx = dx_skip + (v - h * (h * v).mean(1, keepdim=True) - v.mean(1, keepdim=True)) * rstd
    y = None
    if values["compute_y"]:
        parts = ([u_value] if values["concat_u"] else []) + ([x_value] if values["concat_x"] else []) + [y_value]
        y = torch.cat(parts, dim=1)
    return dx, du, (dln * h).sum(0), dln.sum(0), y


def analyze(payload, *, rows="suspicious", max_rows=4096, threshold=1e20,
            chunk_bytes=64 << 20, row_chunk=64, selection=None,
            observation_origin="captured stage outputs"):
    torch = common._torch()
    if max_rows < 0 or min(row_chunk, chunk_bytes) < 1 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid reference limits or threshold")
    values = bind_inputs(payload)
    selected, coverage = selection if selection is not None else select_rows(
        payload, values, rows, max_rows, threshold, chunk_bytes)
    operation = payload["operation"]
    report = {"operation": operation, "stage": payload.get("stage"), "layer": payload.get("layer"),
              "attempt": payload.get("attempt"), "selection": coverage,
              "observed_outputs_source": observation_origin, "supported": True,
              "references": {}, "interpretation": [
                  "FP64 references use original captured inputs and all columns; GPU roundoff is not bitwise modeled.",
                  "Norm uses saved mean/rstd and saved dropout masks; no forward statistics or RNG are regenerated.",
                  "Packed-mask validity is checked on analyzed rows; unselected mask rows are not validated by the CPU reference.",
                  "Only all-row reductions are comparable with complete parameter gradients.",
                  "Finite dtype-range overflow and infinity after dtype conversion are reported separately."]}
    if operation == NORM and values["training"] and (values["random_mask"] is None or not values["random_mask"].numel()):
        report.update(supported=False, limitation="CPU training norm reference requires the saved packed random_mask; fused RNG is unsupported. Full GPU stage replay remains available.")
        return report
    outputs = payload.get("outputs")
    started = time.monotonic()
    full = coverage["all_rows_included"]
    if operation == MM:
        dout, weight = values["dout"], values["output_weight"]
        width = max(dout.shape[1], weight.shape[0])
        dtype = dout.dtype
        dtypes = [dtype]
    else:
        width = max(values["dy"].shape[1], values["x"].shape[1])
        dtypes = [values["x"].dtype, values["u"].dtype, values["weight"].dtype,
                  values["weight"].dtype, values["x"].dtype]
    if width * 8 * 32 > chunk_bytes:
        raise ValueError("One full-column reference row exceeds chunk budget")
    block = max(1, min(row_chunk, chunk_bytes // (width * 8 * 32)))
    summaries = []
    for index, dtype in enumerate(dtypes):
        reduction = operation == NORM and index in (2, 3)
        if operation == NORM and index == 4 and not values["compute_y"]:
            summaries.append(None)
            continue
        scope = ("all sample rows" if full else "partial selected-row contribution; not the full parameter gradient") if reduction else "selected rows, all columns"
        summaries.append(ReferenceSummary(dtype, scope=scope,
                         observed_available=outputs is not None and (full or not reduction)))
    with torch.no_grad():
        if operation == MM:
            # Tile both matrix dimensions to keep parameter conversion bounded.
            tile = min(128, max(1, int(math.sqrt(chunk_bytes // (8 * 32)))))
            for start in range(0, selected.numel(), block):
                ids = selected[start:start + block]
                for col in range(0, weight.shape[0], tile):
                    stop = min(col + tile, weight.shape[0])
                    reference = torch.zeros((ids.numel(), stop - col), dtype=torch.float64)
                    for k in range(0, dout.shape[1], tile):
                        reference += dout[ids, k:k + tile].to(torch.float64) @ weight[col:stop, k:k + tile].to(torch.float64).T
                    observed = outputs[0][ids, col:stop] if outputs is not None else None
                    summaries[0].add(reference, observed, lambda ij: [int(ids[ij[0]]), col + ij[1]])
        else:
            dw = torch.zeros(values["x"].shape[1], dtype=torch.float64)
            db = torch.zeros_like(dw)
            for start in range(0, selected.numel(), block):
                ids = selected[start:start + block]
                try:
                    reference = norm_reference_rows(values, ids)
                except UnsupportedReference as error:
                    report.update(supported=False, limitation=str(error), references={})
                    return report
                dw += reference[2]
                db += reference[3]
                for index in (0, 1, 4):
                    if summaries[index] is not None:
                        observed = outputs[index][ids] if outputs is not None else None
                        summaries[index].add(reference[index], observed, lambda ij: [int(ids[ij[0]]), ij[1]])
            for index, reference in ((2, dw), (3, db)):
                summaries[index].add(reference, outputs[index] if outputs is not None and full else None)
    report.update(seconds=time.monotonic() - started, row_chunk=block, chunk_bytes=chunk_bytes,
                  references={name: summary.result() if summary is not None else None
                              for name, summary in zip(OUTPUTS[operation], summaries)})
    return report


def input_digests(values, *, chunk_bytes, threshold):
    """Logical input hashes plus every backing byte, including unused padding."""
    torch = common._torch()
    logical, storage_keys, raw = {}, {}, []
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            logical[name] = {"scalar": value}
            continue
        storage = value.untyped_storage()
        key = storage._cdata
        if key not in storage_keys:
            storage_keys[key] = len(raw)
            bytes_view = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
            digest = hashlib.sha256()
            for start in range(0, storage.nbytes(), chunk_bytes):
                digest.update(bytes_view[start:start + chunk_bytes].to("cpu").numpy().tobytes())
            raw.append({"bytes": storage.nbytes(), "sha256": digest.hexdigest()})
        logical[name] = {**common.output_digest(value, chunk_bytes, threshold),
                         "storage_offset": value.storage_offset(), "storage_index": storage_keys[key],
                         "is_conj": value.is_conj(), "is_neg": value.is_neg()}
    return {"inputs": logical, "raw_storages": raw}


def same_digest(left, right):
    if left is None or right is None:
        return left is right
    return all(left[key] == right[key] for key in ("shape", "dtype", "logical_sha256"))


def execution_controls():
    """Query flags without initializing CUDA."""
    torch = common._torch()
    return {"autocast": {device: {"enabled": torch.is_autocast_enabled(device),
                                  "dtype": str(torch.get_autocast_dtype(device))}
                         for device in ("cpu", "cuda")},
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            **{key: getattr(torch.backends.cuda.matmul, key) for key in (
                "allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")}}


@contextmanager
def restored_execution_controls(captured):
    """Restore operation-local autocast and matmul flags for just the call."""
    torch = common._torch()
    previous = execution_controls()
    if captured is None:
        yield {"captured_controls_available": False, "effective": previous,
               "limitation": "Legacy dump lacks operation-local controls; context.json alone cannot recover them."}
        return

    def set_flags(values):
        torch.set_float32_matmul_precision(values["float32_matmul_precision"])
        for key in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
            setattr(torch.backends.cuda.matmul, key, values[key])

    try:
        set_flags(captured)
        with ExitStack() as stack:
            for device in ("cpu", "cuda"):
                options = captured["autocast"][device]
                dtype_name = options["dtype"].removeprefix("torch.")
                dtype = getattr(torch, dtype_name)
                stack.enter_context(torch.autocast(device, enabled=options["enabled"], dtype=dtype))
            effective = execution_controls()
            if effective != captured:
                raise ValueError(f"Operation controls could not be restored exactly: captured={captured}, effective={effective}")
            yield {"captured_controls_available": True, "captured": captured, "effective": effective,
                   "verified": True}
    finally:
        set_flags(previous)


def replay_gpu(payload, *, repeats=3, chunk_bytes=64 << 20, max_bytes=32 << 30,
               threshold=1e20, rows="suspicious", max_rows=4096, row_chunk=64, progress=None):
    """Whole-stage replay, independent of the other stage or full model."""
    torch = common._torch()
    values = bind_inputs(payload)
    operation = payload["operation"]
    if operation == MM:
        def function(**inputs):
            return (torch.mm(inputs["dout"], inputs["output_weight"].t()),)
    else:
        module = importlib.import_module("generative_recommenders.ops.triton.triton_hstu_linear")
        function = getattr(module, NORM)
    pristine = input_digests(values, chunk_bytes=chunk_bytes, threshold=threshold)
    captured = payload.get("outputs")
    expected = [common.output_digest(tensor, chunk_bytes, threshold) for tensor in captured] if captured is not None else None
    selection = select_rows(payload, values, rows, max_rows, threshold, chunk_bytes)
    attempts, first = [], None

    def phase(name, repeat, **extra):
        if progress:
            progress({"phase": name, "repeat": repeat, "attempts": list(attempts), **extra})

    for repeat in range(repeats):
        phase("restore_inputs", repeat, pristine_inputs=pristine, captured_outputs=expected)
        fresh, size = common.restore_raw_tree(values, device="cuda:0", chunk_bytes=chunk_bytes, max_bytes=max_bytes)
        torch.cuda.synchronize()
        restored = input_digests(fresh, chunk_bytes=chunk_bytes, threshold=threshold)
        verified = restored == pristine
        phase("verify_inputs", repeat, input_digests=restored, matches_pristine=verified)
        if not verified:
            raise RuntimeError("Restored GPU input logical or backing-storage hashes differ from pristine dump")
        phase("execute_operation", repeat)
        start = time.monotonic()
        captured_controls = payload.get("execution_controls", payload.get("event", {}).get("execution_controls"))
        with restored_execution_controls(captured_controls) as controls, torch.no_grad():
            outputs = function(**fresh)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        reduction_config = None
        if operation == NORM:
            chosen = getattr(module._ln_mul_dropout_bwd_dwdb, "best_config", None)
            if chosen is not None:
                reduction_config = {"kwargs": dict(chosen.kwargs),
                                    "num_warps": chosen.num_warps,
                                    "num_stages": chosen.num_stages}
        phase("verify_post_operation_inputs", repeat)
        after = input_digests(fresh, chunk_bytes=chunk_bytes, threshold=threshold)
        phase("summarize_outputs", repeat)
        observed = [common.output_digest(tensor, chunk_bytes, threshold) for tensor in outputs]
        reference = analyze({**payload, "outputs": outputs}, selection=selection, threshold=threshold,
                            chunk_bytes=chunk_bytes, row_chunk=row_chunk,
                            observation_origin=f"fresh GPU repeat {repeat}; reference uses pristine CPU inputs")
        attempts.append({"repeat": repeat, "fresh_input_storage_bytes": size,
                         "pristine_input_digests": pristine,
                         "input_digests": restored, "matches_pristine_inputs": verified,
                         "input_digests_after_operation": after,
                         "inputs_unchanged_by_operation": after == restored,
                         "execution_controls": controls,
                         "operation_seconds": elapsed, "outputs": dict(zip(OUTPUTS[operation], observed)),
                         "norm_reduction_config": reduction_config,
                         "captured_output_digests": expected,
                         "matches_captured_logical_bytes": [same_digest(a, b) for a, b in zip(observed, expected)] if expected is not None else None,
                         "matches_first_repeat_logical_bytes": [same_digest(a, b) for a, b in zip(observed, first)] if first is not None else None,
                         "fp64_reference_comparison": reference})
        if first is None:
            first = observed
        del fresh, outputs
        phase("complete_repeat", repeat)
    return attempts


def restore_replay_environment(captured_environment, allocator_config=None):
    """Restore context, then apply an explicit allocator arm before Torch import."""
    for key, value in captured_environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    names = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    overrides = {}
    if allocator_config is not None:
        overrides = dict.fromkeys(names, allocator_config)
        os.environ.update(overrides)
    return {"environment_overrides": overrides,
            "effective_allocator_environment": {name: os.environ.get(name) for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path, help="Trusted local nan_backward_boundaries.py stage dump")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rows", default="suspicious")
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=1e20)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-input-gib", type=int, default=32)
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument("--allocator-config", default=None,
                        help="GPU-only explicit override for both PyTorch allocator variables; an empty string clears them")
    args = parser.parse_args()
    if min(args.row_chunk, args.chunk_mib, args.cpu_threads, args.repeats, args.max_input_gib) < 1 or args.max_rows < 0:
        parser.error("limits must be positive; --max-rows may be zero")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be finite and positive")
    if args.gpu and args.context is None:
        parser.error("--gpu requires the original capture --context")
    if args.allocator_config is not None and not args.gpu:
        parser.error("--allocator-config is only valid with --gpu")
    context = json.loads(args.context.read_text()) if args.context else None
    changed = []
    environment_report = {"environment_overrides": {}, "effective_allocator_environment": None}
    root = Path(__file__).resolve().parents[1]
    if args.gpu:
        environment_report = restore_replay_environment(context["environment"], args.allocator_config)
        for name, expected in context["source_sha256"].items():
            if name.startswith("generative_recommenders/"):
                path = root / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    changed.append(name)
        if changed and not args.allow_source_change:
            parser.error(f"captured model source differs: {changed}; intentional comparison requires --allow-source-change")
        sys.path.insert(0, str(root))
    torch = common._torch()
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
        for key in ("torch", "hip", "triton", "triton_distribution"):
            if key in context and context[key] != versions[key]:
                parser.error(f"stack mismatch for {key}: capture {context[key]!r}, current {versions[key]!r}")
    payload = torch.load(args.dump, map_location="cpu", weights_only=False, mmap=True)
    report = {"dump": str(args.dump), "versions": versions, "original_context": str(args.context) if args.context else None,
              "restored_environment": context["environment"] if args.gpu else None,
              **environment_report,
              "changed_model_source": changed, "gpu_validation": "not requested"}
    report_path = args.report or args.dump.with_suffix(".output_analysis.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = report_path.with_name(report_path.name + ".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)

    report["cpu_reference"] = analyze(payload, rows=args.rows, max_rows=args.max_rows,
                                       threshold=args.threshold, chunk_bytes=args.chunk_mib << 20,
                                       row_chunk=args.row_chunk)
    save()
    if args.gpu:
        report["gpu_validation"] = "in progress"
        report["gpu_input_storage_limit_bytes"] = args.max_input_gib << 30
        report["gpu_memory_limit_note"] = "Input storage limit only; outputs and kernel scratch are additional allocations."
        save()

        def progress(state):
            report["gpu_progress"] = {key: value for key, value in state.items() if key != "attempts"}
            report["gpu_attempts"] = state["attempts"]
            save()

        try:
            report["gpu_attempts"] = replay_gpu(payload, repeats=args.repeats, rows=args.rows,
                                                max_rows=args.max_rows, row_chunk=args.row_chunk,
                                                threshold=args.threshold, chunk_bytes=args.chunk_mib << 20,
                                                max_bytes=args.max_input_gib << 30, progress=progress)
            report["gpu_validation"] = "completed; evaluate numerical references and hashes"
        except Exception as error:
            report["gpu_validation"] = "failed"
            report["gpu_error"] = {"type": type(error).__name__, "message": str(error)}
            save()
            raise
        save()
    print(json.dumps({"report": str(report_path), "gpu_validation": report["gpu_validation"],
                      "cpu_reference": report["cpu_reference"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
