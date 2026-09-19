#!/usr/bin/env python3
"""Replay HSTU output backward, then isolate its GEMM or LN/multiply/dropout.

Requires a trusted HSTUComputeOutputFunction.backward tripwire capture, the
matching training checkout on PYTHONPATH, and its prepared Torch/Triton stack.
Original gradient_mm/norm stage captures from that function are also accepted;
those decode the original stage tensors directly and skip full preparation.
No dataset or model is loaded. For full backward captures, preparation executes
the whole captured backward once and retains the actual GEMM-produced dy passed
to the normalization helper.
The complete result is compared diagnostically with the original captured result.
The retained dy is produced by this new preparation execution; it is NOT the
internal dy from the original captured invocation. Later preparation kernels may
also affect those retained tensors through untracked writes. Only a boundary
capture made inside the original invocation can establish its internal dy.
Original stage captures retain tensors until the tripwire copies them. Their
scalar flags describe boundary observation time, but version checks cannot rule
out later untracked writes to retained tensor storage.

--stage mm repeats only torch.mm(dout, output_weight.t()). --stage norm repeats
only triton_layer_norm_mul_dropout_bwd with the fixed preparation inputs, including
the captured dropout mask. All returned floating tensors must be finite, within
--max-abs, and exactly equal to the first result under the selected configuration.
That baseline is a repeatability oracle, not an independent mathematical oracle.
--reference-cpu additionally compares the baseline with a chunked CPU float64
reference. The norm reference uses captured mean/rstd and the packed dropout mask;
it does not validate the forward statistics or reconstruct a fused RNG mask.
Reference agreement is judged with --reference-rtol/--reference-atol, which may
need tuning for cancellation and production reduction/rounding differences.
Preparation failures are reported separately from stress failures. A corrupt
fixed input can cause every isolated invocation to fail numerically without
establishing that the isolated operation created the corruption.

--num-stages, --num-warps and --max-vgpr alter only the isolated norm dx/du launch;
the norm stage also executes the production weight/bias reduction kernel.
Default settings preserve production dispatch. Outputs/partial sums are allocated
normally; inputs remain fixed. --check-input-contents adds byte comparisons to
version guards. Scalar checkpoints change timing and memory traffic.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NORM_OUTPUT_NAMES = ("dattn", "du", "d_norm_weight", "d_norm_bias", "y")
NORM_TENSOR_INPUT_NAMES = {"dy", "x", "u", "weight", "bias", "mean", "rstd", "random_mask"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--stage", choices=("mm", "norm"), required=True)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--max-abs", type=float, default=1e20)
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--max-vgpr", type=int)
    parser.add_argument("--check-input-contents", action="store_true")
    parser.add_argument("--reference-cpu", action="store_true")
    parser.add_argument("--reference-rows", type=int, default=256)
    parser.add_argument("--reference-rtol", type=float, default=1e-2)
    parser.add_argument("--reference-atol", type=float, default=1e-6)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.check_every < 1 or not math.isfinite(args.max_abs) or args.max_abs <= 0:
        parser.error("repeat, check-every and max-abs must be positive and finite")
    if any(v is not None and v < 1 for v in (args.num_stages, args.max_vgpr)):
        parser.error("num-stages and max-vgpr must be positive")
    if args.stage == "mm" and any(v is not None for v in (args.num_stages, args.num_warps, args.max_vgpr)):
        parser.error("norm launch overrides do not apply to the mm stage")
    if args.reference_rows < 1 or any(
        not math.isfinite(v) or v < 0 for v in (args.reference_rtol, args.reference_atol)
    ):
        parser.error("reference-rows must be positive; reference tolerances must be finite and nonnegative")
    if args.report.resolve() == args.capture.resolve() or (
        args.report.exists() and args.capture.exists() and args.report.samefile(args.capture)
    ):
        parser.error("report must not overwrite the capture")
    return args


def prepare(capture, linear):
    """Retain new preparation tensors, never claim they are original internal dy."""
    from scripts.replay_nan_tripwire import compare_original_outputs, replay_once

    original = linear.triton_layer_norm_mul_dropout_bwd
    calls = []

    def retain(*args, **kwargs):
        if calls:
            raise ValueError("requires exactly one non-group-normalization backward call")
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        return original(*args, **kwargs)

    linear.triton_layer_norm_mul_dropout_bwd = retain
    try:
        actual, payload = replay_once(capture, "cuda", snapshot_original_outputs=True)
    finally:
        linear.triton_layer_norm_mul_dropout_bwd = original
    if len(calls) != 1:
        raise ValueError("requires exactly one non-group-normalization backward call")
    comparison = compare_original_outputs(actual, payload["outputs"], rtol=0, atol=0)
    return calls[0], payload["replay"], comparison


def original_stage_inputs(capture, stage, device):
    """Decode only original stage tensors, without executing any parent kernel."""
    import torch
    from scripts.replay_nan_tripwire import decode_payload, iter_tensors

    payload = capture["payload"]
    descriptor = payload["stage"]
    expected_name = "gradient_mm" if stage == "mm" else "norm"
    if descriptor.get("format_version") != 1 or descriptor.get("name") != expected_name:
        raise ValueError("capture does not contain requested original stage: " + expected_name)
    # Parent tensors are provenance only. Avoid uploading every saved tensor in
    # a full HSTU backward when replaying just the original gradient GEMM.
    selected = {**capture, "payload": {name: payload[name] for name in ("inputs", "outputs")}}
    decoded = decode_payload(selected, device)
    values, outputs = decoded["inputs"], decoded["outputs"]
    names = {"dout", "output_weight"} if stage == "mm" else NORM_TENSOR_INPUT_NAMES
    if values.keys() != names:
        raise ValueError("unexpected original stage input names: " + str(sorted(values)))
    output_names = {"dy"} if stage == "mm" else set(NORM_OUTPUT_NAMES)
    if outputs.keys() != output_names:
        raise ValueError("unexpected original stage output names: " + str(sorted(outputs)))
    if stage == "mm":
        fixed = {"dout": values["dout"], "weight": values["output_weight"]}
    else:
        settings = descriptor["settings"]
        if settings.keys() & names:
            raise ValueError("stage settings must contain only scalar helper arguments")
        fixed = {**values, **settings}
    # Freeze expected outputs if a diagnostic hook captured an alias of a stage
    # input. Normal MM/norm output allocations are independent and need no copy.
    input_storages = {(str(t.device), t.untyped_storage().data_ptr()) for _, t in iter_tensors(fixed)}
    outputs = {
        name: value.detach().clone() if isinstance(value, torch.Tensor) and
        (str(value.device), value.untyped_storage().data_ptr()) in input_storages else value
        for name, value in outputs.items()
    }
    return fixed, outputs


def all_equal(torch, a, b, *, chunk_size=16 * 1024**2, bytewise=False):
    """Bounded logical-value comparison, including noncontiguous tensor views."""
    from generative_recommenders.dlrm_v4.train.nan_tripwire import _finite_chunks

    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("repeat output shape/dtype/device changed")
    result = torch.ones((), dtype=torch.bool, device=a.device)
    for left, right in zip(_finite_chunks(a, chunk_size), _finite_chunks(b, chunk_size)):
        if bytewise:
            # contiguous() may retain a non-unit feature stride on a singleton
            # slice because it is already logically contiguous. An explicit
            # contiguous-format clone gives byte reinterpretation a unit stride.
            left, right = (t.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
                           for t in (left, right))
        result = result & (left == right).all()
    return result


def reference_chunks(inputs, stage, rows, *, mm_input_dtype=None):
    """Yield named CPU FP64 outputs in row chunks; parameter sums span all rows.

    The LN Jacobian uses the supplied forward mean/rstd. This isolates backward
    arithmetic from recomputing forward statistics, including any bad statistics
    already present in the capture. It is independent of the Triton kernels and
    GPU GEMM, but float64 algebra intentionally differs from their rounding.
    """
    import torch

    if rows < 1:
        raise ValueError("reference rows must be positive")

    def cpu(value):
        return value.detach().to(device="cpu", dtype=torch.float64)

    if stage == "mm":
        def mm_cpu(value):
            # CUDA autocast applies its input rounding before the GEMM. Keep
            # that rounding while moving all multiply/accumulate work to FP64.
            if mm_input_dtype is not None and value.dtype != torch.float64:
                value = value.detach().to(device="cpu", dtype=mm_input_dtype)
            return cpu(value)

        weight = mm_cpu(inputs["weight"])
        for start in range(0, inputs["dout"].shape[0], rows):
            yield "dy", start, mm_cpu(inputs["dout"][start:start + rows]) @ weight.t()
        return
    if stage != "norm":
        raise ValueError("unknown stage: " + stage)
    x_all = inputs["x"]
    n, d = x_all.shape
    concat_u, concat_x = inputs["concat_u"], inputs["concat_x"]
    u_mask_bit = 4 if concat_x else 2
    columns = d * (1 + int(concat_u) + int(concat_x))
    if tuple(inputs["dy"].shape) != (n, columns):
        raise ValueError("norm dy shape disagrees with concat flags")
    if tuple(inputs["u"].shape) != (n, d):
        raise ValueError("norm u shape disagrees with x")
    p = inputs["dropout_ratio"]
    training = inputs["training"]
    if not 0 <= p < 1:
        raise ValueError("reference requires 0 <= dropout_ratio < 1")
    mask_all = inputs["random_mask"]
    if training and (mask_all is None or mask_all.numel() == 0):
        raise ValueError("CPU norm reference requires a captured packed dropout mask when training")
    if training and (tuple(mask_all.shape) != (n, d) or mask_all.dtype != torch.int8):
        raise ValueError("CPU norm reference requires an (N, D) int8 packed dropout mask")
    activation = inputs["mul_u_activation_type"]
    if activation not in ("none", "silu", "sigmoid"):
        raise ValueError("unsupported multiplication activation: " + activation)
    weight, bias = cpu(inputs["weight"]), cpu(inputs["bias"])
    dw, db = torch.zeros_like(weight), torch.zeros_like(bias)
    for start in range(0, n, rows):
        sl = slice(start, start + rows)
        x, u, upstream = cpu(x_all[sl]), cpu(inputs["u"][sl]), cpu(inputs["dy"][sl])
        mean, rstd = cpu(inputs["mean"][sl])[:, None], cpu(inputs["rstd"][sl])[:, None]
        mask = mask_all[sl].detach().cpu() if training else None

        def dropout(value, bit):
            return torch.where((mask & bit) != 0, value / (1 - p), 0.0) if training else value

        offset = 0
        du_direct = torch.zeros_like(u)
        dx_direct = torch.zeros_like(x)
        if concat_u:
            du_direct = dropout(upstream[:, offset:offset + d], u_mask_bit)
            offset += d
        if concat_x:
            dx_direct = dropout(upstream[:, offset:offset + d], 2)
            offset += d
        dy = dropout(upstream[:, offset:offset + d], 1)
        xhat = (x - mean) * rstd
        affine = xhat * weight + bias
        sigmoid = torch.sigmoid(u)
        silu_derivative = sigmoid * (1 + u * (1 - sigmoid))
        if activation == "silu":
            multiplier, derivative = u * sigmoid, silu_derivative
        elif activation == "sigmoid":
            multiplier, derivative = sigmoid, sigmoid * (1 - sigmoid)
        else:
            multiplier, derivative = u, 1.0
        du = dy * affine * derivative
        du += du_direct * silu_derivative if inputs["silu_u"] else du_direct
        d_affine = dy * multiplier
        wdy = d_affine * weight
        dx = dx_direct + rstd * (
            wdy - wdy.mean(dim=1, keepdim=True) - xhat * (xhat * wdy).mean(dim=1, keepdim=True)
        )
        dw += (d_affine * xhat).sum(dim=0)
        db += d_affine.sum(dim=0)
        yield "dattn", start, dx
        yield "du", start, du
        if inputs["compute_y"]:
            parts = []
            if concat_u:
                parts.append(dropout(u * sigmoid if inputs["silu_u"] else u, u_mask_bit))
            if concat_x:
                parts.append(dropout(x, 2))
            parts.append(dropout(affine * multiplier, 1))
            yield "y", start, torch.cat(parts, dim=1)
    yield "d_norm_weight", None, dw
    yield "d_norm_bias", None, db


def compare_reference(inputs, stage, outputs, *, rows, rtol, atol, mm_input_dtype=None):
    import torch

    actual = dict(zip(("dy",) if stage == "mm" else NORM_OUTPUT_NAMES, outputs))
    if len(outputs) != (1 if stage == "mm" else len(NORM_OUTPUT_NAMES)):
        raise ValueError("unexpected isolated output structure")
    checks = {}
    for name, start, expected in reference_chunks(inputs, stage, rows, mm_input_dtype=mm_input_dtype):
        tensor = actual[name]
        value = tensor if start is None else tensor[start:start + expected.shape[0]]
        value = value.detach().to(device="cpu", dtype=torch.float64)
        if value.shape != expected.shape:
            raise ValueError("reference output shape changed: " + name)
        finite = torch.isfinite(value) & torch.isfinite(expected)
        close = torch.isclose(value, expected, rtol=rtol, atol=atol) & finite
        info = checks.setdefault(name, {"elements": 0, "mismatched_elements": 0,
                                       "nonfinite_elements": 0, "max_abs_diff": 0.0,
                                       "max_abs_actual": 0.0, "max_abs_reference": 0.0})
        info["elements"] += value.numel()
        info["mismatched_elements"] += int((~close).sum())
        info["nonfinite_elements"] += int((~finite).sum())
        for key, data in (("max_abs_diff", value - expected),
                          ("max_abs_actual", value), ("max_abs_reference", expected)):
            if data.numel():
                maximum = float(torch.nan_to_num(data.abs(), nan=float("inf"), posinf=float("inf")).max())
                info[key] = max(info[key], maximum)
    return {"passed": all(info["mismatched_elements"] == 0 for info in checks.values()),
            "device": "cpu", "dtype": "torch.float64", "rtol": rtol, "atol": atol,
            "rows_per_chunk": rows, "uses_saved_forward_statistics": stage == "norm",
            "mm_autocast_input_dtype": str(mm_input_dtype) if mm_input_dtype is not None else None,
            "outputs": checks}


def run(args, report, publish):
    import torch
    import triton
    from scripts.replay_nan_tripwire import changed_capture_inputs, compare_original_outputs, iter_tensors, _restore_runtime_state
    from scripts.repro_hstu_attention_capture import compiled_metadata
    from generative_recommenders.dlrm_v4.train.nan_tripwire import (
        _all_finite, _all_within_bound,
    )

    chunk_size = 16 * 1024**2

    def flags(value):
        out = {}
        for name, tensor in iter_tensors(value):
            if tensor.is_floating_point():
                out[name + ":finite"] = _all_finite(tensor, chunk_size)
                out[name + ":within_bound"] = _all_within_bound(tensor, args.max_abs, chunk_size)
        return out

    def resolve(checks):
        return dict(zip(checks, torch.stack(list(checks.values())).cpu().tolist())) if checks else {}

    def equal(a, b, bytewise=False):
        return all_equal(torch, a, b, chunk_size=chunk_size, bytewise=bytewise)

    capture = torch.load(args.capture, map_location="cpu", mmap=True, weights_only=False)
    if capture.get("format_version") != 1:
        raise ValueError("requires tripwire format_version=1")
    stage_capture = capture["payload"].get("stage")
    replay = stage_capture["parent_replay"] if stage_capture is not None else capture["payload"]["replay"]
    if replay is None:
        raise ValueError("requires an HSTU output backward or original stage capture")
    if replay["class"] != "HSTUComputeOutputFunction" or replay["direction"] != "backward":
        raise ValueError("requires HSTUComputeOutputFunction.backward capture")
    if replay["ctx"].get("group_norm"):
        raise ValueError("requires group_norm=False")
    changed = changed_capture_inputs(capture)
    if changed:
        raise ValueError("tracked input mutation before capture: " + str(changed))
    for name, value in capture["report"]["environment"].items():
        if not name.startswith("NAN_TRIPWIRE_"):
            os.environ.setdefault(name, value)
    torch.cuda.set_device(0)
    from generative_recommenders.ops.triton import triton_hstu_linear as linear

    report["source"] = {
        "path": str(args.capture.resolve()), "bytes": args.capture.stat().st_size,
        "step": capture["report"]["step"], "metadata": capture["report"].get("metadata"),
        "torch": str(torch.__version__), "hip": torch.version.hip, "triton": triton.__version__,
        "runtime_state": replay.get("runtime_state"),
        "cuda_autocast_enabled": replay.get("cuda_autocast_enabled", False),
        "cuda_autocast_dtype": replay.get("cuda_autocast_dtype"),
        "operation": f"{replay['module']}.{replay['class']}.{replay['direction']}",
        "selected_sequence": capture["report"].get("selected_sequence", capture["report"].get("first_bad_sequence")),
        "stage_identity": stage_capture.get("identity") if stage_capture is not None else None,
    }
    report["provenance"] = {
        "dout_and_output_weight": "decoded_original_boundary_inputs",
        "norm_inputs": "retained_after_full_preparation_replay",
        "norm_dy": "regenerated_by_preparation_gradient_mm; original_internal_dy_unavailable",
        "preparation_comparison": "diagnostic_comparison_to_original_full_backward_outputs",
        "baseline": "first_isolated_stage_execution_under_selected_configuration",
        "exact_repeat": "repeatability_only",
    }
    report["stage_operation"] = (
        "torch.mm(dout, output_weight.t())" if args.stage == "mm"
        else "triton_layer_norm_mul_dropout_bwd (dx/du and weight/bias reduction)"
    )
    report["stage_output_names"] = ["dy"] if args.stage == "mm" else list(NORM_OUTPUT_NAMES)
    publish("stage_decode_start" if stage_capture is not None else "preparation_start")
    with torch.no_grad():
        original_stage_outputs = None
        if stage_capture is not None:
            fixed, original_stage_outputs = original_stage_inputs(capture, args.stage, "cuda")
            _restore_runtime_state(replay.get("runtime_state"))
            decoded_replay = replay
            inputs = fixed if args.stage == "norm" else None
            report["provenance"].update(
                norm_inputs="original_invocation_stage_capture" if args.stage == "norm" else "not_used",
                norm_dy="original_invocation_norm_input" if args.stage == "norm" else "not_used",
                preparation_comparison="not_applicable; no_preparation_execution",
                original_stage_outputs="original_invocation_stage_capture",
                capture_semantics="retained_tensors_copied_at_tripwire_check; later_untracked_writes_not_excluded",
            )
        else:
            inputs, decoded_replay, comparison = prepare(capture, linear)
            report["preparation_comparison"] = comparison
            fixed = inputs if args.stage == "norm" else {
                "dout": decoded_replay["args"][0], "weight": decoded_replay["saved_tensors"][6],
            }
        if inputs is not None:
            report["norm_input_checks"] = resolve(flags(inputs))
            report["norm_settings"] = {k: v for k, v in inputs.items() if not isinstance(v, torch.Tensor)}
        report["input_layouts"] = {
            name: {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
                   "storage_offset": t.storage_offset()}
            for name, t in iter_tensors(fixed)
        }
        del capture, replay
        if args.stage == "norm":
            operation = lambda: linear.triton_layer_norm_mul_dropout_bwd(**fixed)
        else:
            operation = lambda: (torch.mm(fixed["dout"], fixed["weight"].t()),)
        report["stage_input_checks"] = resolve(flags(fixed))
        if not all(report["stage_input_checks"].values()):
            report["status"] = "INPUT_FAILURE"
            publish("input_failure")
            return 1
        amp_enabled = decoded_replay.get("cuda_autocast_enabled", False)
        amp_dtype = decoded_replay.get("cuda_autocast_dtype", torch.bfloat16)
        del decoded_replay
        kernel = None
        if args.stage == "norm":
            mask = fixed["random_mask"]
            kernel = linear._ln_mul_dropout_bwd_dx_du_rng if mask is not None and mask.numel() > 0 else linear._ln_mul_dropout_bwd_dx_du
        original_run = kernel.run if kernel is not None else None
        overrides = {}
        if args.num_stages is not None:
            overrides["num_stages"] = args.num_stages
        if args.num_warps is not None:
            overrides["num_warps"] = args.num_warps
        if args.max_vgpr is not None:
            overrides["llvm_fn_attrs"] = f"amdgpu-num-vgpr={args.max_vgpr}"

        launched = {}

        def launch(*positional, **kwargs):
            compiled = original_run(*positional, **{**kwargs, **overrides})
            if compiled.hash not in launched:
                metadata = compiled.metadata._asdict() if hasattr(compiled.metadata, "_asdict") else str(compiled.metadata)
                launched[compiled.hash] = {
                    "hash": compiled.hash, "name": compiled.name, "metadata": metadata,
                    "artifact_sha256": {
                        key: hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()
                        for key, value in compiled.asm.items() if key in ("llir", "amdgcn", "hsaco")
                    },
                }
            report["selected_norm_launches"] = list(launched.values())
            return compiled

        if args.stage == "norm":
            kernel.run = launch
        report["launch_overrides"] = overrides
        try:
            with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                tensors = dict(iter_tensors(fixed))
                versions = {name: t._version for name, t in tensors.items()}
                snapshots = {name: t.clone() for name, t in tensors.items()} if args.check_input_contents else {}

                def input_checks():
                    checks = {}
                    for name, t in tensors.items():
                        if t._version != versions[name]:
                            raise RuntimeError("tracked input mutation: " + name)
                        if name in snapshots:
                            checks[name + ":input_unchanged"] = equal(t, snapshots[name], bytewise=True)
                    return checks

                baseline = operation()
                report["baseline_checks"] = resolve({**flags(baseline), **input_checks()})
                if original_stage_outputs is not None:
                    report["original_stage_comparison"] = compare_original_outputs(
                        dict(zip(report["stage_output_names"], baseline)), original_stage_outputs, rtol=0, atol=0,
                    )
                    del original_stage_outputs
                if args.stage == "norm":
                    report["cached_norm_reduction_kernels"] = compiled_metadata(linear._ln_mul_dropout_bwd_dwdb)
                inputs_unchanged = all(value for name, value in report["baseline_checks"].items()
                                       if name.endswith(":input_unchanged"))
                if args.reference_cpu and inputs_unchanged:
                    publish("reference_start")
                    report["cpu_reference"] = compare_reference(
                        fixed, args.stage, baseline, rows=args.reference_rows,
                        rtol=args.reference_rtol, atol=args.reference_atol,
                        mm_input_dtype=amp_dtype if args.stage == "mm" and amp_enabled else None,
                    )
                    report["reference_input_checks"] = resolve(input_checks())
                if not all(report["baseline_checks"].values()):
                    report["status"] = "BASELINE_FAILURE"
                    publish("baseline_failure")
                    return 1
                if args.reference_cpu:
                    if not report["cpu_reference"]["passed"] or not all(report["reference_input_checks"].values()):
                        report["status"] = "REFERENCE_FAILURE"
                        publish("reference_failure")
                        return 1
                references = dict(iter_tensors(baseline))
                publish("stress_start")
                pending = []
                started = time.monotonic()
                for iteration in range(1, args.repeat + 1):
                    outputs = operation()
                    checks = flags(outputs)
                    actual = dict(iter_tensors(outputs))
                    if actual.keys() != references.keys():
                        raise ValueError("repeat output structure changed")
                    for name, t in actual.items():
                        checks[name + ":exact_repeat"] = equal(t, references[name])
                    checks.update(input_checks())
                    pending.extend((iteration, name, flag) for name, flag in checks.items())
                    del outputs, actual
                    if iteration % args.check_every and iteration != args.repeat:
                        continue
                    values = torch.stack([flag for _, _, flag in pending]).cpu().tolist()
                    failures = [{"iteration": i, "check": name} for (i, name, _), passed in zip(pending, values) if not passed]
                    report["checks"].append({"through": iteration, "failures": failures, "seconds": time.monotonic() - started})
                    report["iterations_completed"] = iteration
                    report["status"] = "FAIL" if failures else ("PASS" if iteration == args.repeat else "RUNNING")
                    if args.stage == "norm":
                        report["compiled_kernels"] = compiled_metadata(kernel)
                    publish("check")
                    pending.clear()
                    if failures:
                        return 1
        finally:
            if kernel is not None:
                kernel.run = original_run
    return 0


def main():
    args = arguments()
    for key, value in {"AMDGCN_USE_BUFFER_OPS": "0", "PYTORCH_CUDA_ALLOC_CONF": "",
                       "PYTORCH_ALLOC_CONF": "", "HSA_ENABLE_COREDUMP": "0"}.items():
        os.environ.setdefault(key, value)
    report = {"status": "RUNNING", "stage": args.stage, "repeat": args.repeat,
              "max_abs": args.max_abs, "iterations_completed": 0, "checks": []}

    def publish(event):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({"event": event, "status": report["status"],
                          "iterations_completed": report["iterations_completed"],
                          "last_check": report["checks"][-1] if report["checks"] else None,
                          "report": str(args.report)}, default=str), flush=True)

    try:
        return run(args, report, publish)
    except Exception as error:
        report.update(status="ERROR", error=repr(error))
        publish("error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
