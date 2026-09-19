#!/usr/bin/env python3
"""Stress ORIGINAL preprocess-backward input_projection or input_norm captures.

Only generic tripwire stage payloads (version 1) from
_HSTUPreprocessAndAttentionFunction.backward are accepted. Decode the stage's
inputs/outputs directly, preserving input storage aliases, offsets and strides;
no attention, normalization forward, or parent backward preparation is run.
input_projection invokes the production triton_addmm_bwd (bias sum and two
torch.mm calls). input_norm invokes triton_weighted_layer_norm_bwd, including
its parameter-gradient reduction. Parent runtime and autocast state are restored.

For nonempty norm captures, default replay pins the actual captured DX and
reduction launch configurations. --block-n, --num-warps, --num-stages and
--max-vgpr override the DX kernel only. Actual compiled launches are recorded.
Every output must be finite, within --max-abs and exactly repeat the baseline.
The exact original-output comparison is diagnostic; exact repetition alone is
not a mathematical oracle. --reference-cpu independently checks EVERY output
element with chunked CPU FP64 arithmetic and configurable tolerances. Projection
references apply captured autocast rounding to GEMM operands, but not bias sums.
Norm references use captured mean/rstd; they do not verify forward statistics.

Input version guards always run; --check-input-contents also compares every
logical input byte. Original stage tensors were retained until capture, so
untracked writes between observation and capture remain possible. Scalar checks
alter timing and memory traffic. Only load trusted local pickle captures.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.repro_output_backward_stages import all_equal

STAGES = {
    "input_projection": {
        "inputs": {"x", "w", "dz"}, "settings": {"is_y_1d"},
        "outputs": ("d_normed_x", "d_uvqk_weight", "d_uvqk_bias"),
    },
    "input_norm": {
        "inputs": {"dy", "x", "weight", "bias", "mean", "rstd"},
        "settings": {"learnable", "eps", "BLOCK_D"},
        "outputs": ("d_x", "d_norm_weight", "d_norm_bias"),
    },
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--max-abs", type=float, default=1e20)
    parser.add_argument("--block-n", type=int, choices=(1, 8))
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--max-vgpr", type=int)
    parser.add_argument("--check-input-contents", action="store_true")
    parser.add_argument("--reference-cpu", action="store_true")
    parser.add_argument("--reference-rows", type=int, default=256)
    parser.add_argument("--reference-columns", type=int, default=256,
                        help="projection feature/reduction tile width")
    parser.add_argument("--reference-rtol", type=float, default=1e-2)
    parser.add_argument("--reference-atol", type=float, default=1e-6)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.repeat, args.check_every, args.reference_rows, args.reference_columns) < 1:
        parser.error("repeat, check-every and reference chunk dimensions must be positive")
    if not math.isfinite(args.max_abs) or args.max_abs <= 0:
        parser.error("max-abs must be positive and finite")
    if any(v is not None and v < 1 for v in (args.num_stages, args.max_vgpr)):
        parser.error("num-stages and max-vgpr must be positive")
    if args.stage == "input_projection" and any(v is not None for v in (
        args.block_n, args.num_warps, args.num_stages, args.max_vgpr,
    )):
        parser.error("norm launch overrides do not apply to input_projection")
    if any(not math.isfinite(v) or v < 0 for v in (args.reference_rtol, args.reference_atol)):
        parser.error("reference tolerances must be finite and nonnegative")
    if args.report.resolve() == args.capture.resolve() or (
        args.report.exists() and args.capture.exists() and args.report.samefile(args.capture)
    ):
        parser.error("report must not overwrite the capture")
    return args


def stage_descriptor(capture, stage):
    if capture.get("format_version") != 1:
        raise ValueError("requires tripwire format_version=1")
    descriptor = capture["payload"].get("stage")
    if not descriptor or descriptor.get("format_version") != 1 or descriptor.get("name") != stage:
        raise ValueError("requires requested original stage: " + stage)
    parent = descriptor.get("parent_replay") or {}
    if parent.get("class") != "_HSTUPreprocessAndAttentionFunction" or parent.get("direction") != "backward":
        raise ValueError("requires _HSTUPreprocessAndAttentionFunction.backward parent")
    return descriptor


def original_stage_inputs(capture, stage, device):
    """Decode only stage storages; parent descriptors never reach the decoder."""
    import torch
    from scripts.replay_nan_tripwire import decode_payload, iter_tensors

    descriptor = stage_descriptor(capture, stage)
    schema = STAGES[stage]
    payload = capture["payload"]
    if payload["inputs"].keys() != schema["inputs"]:
        raise ValueError("unexpected original stage input names")
    if payload["outputs"].keys() != set(schema["outputs"]):
        raise ValueError("unexpected original stage output names")
    if descriptor["settings"].keys() != schema["settings"]:
        raise ValueError("unexpected original stage scalar settings")
    if list(iter_tensors(descriptor["settings"])):
        raise ValueError("stage settings must be scalar")
    selected = {**capture, "payload": {name: payload[name] for name in ("inputs", "outputs")}}
    decoded = decode_payload(selected, device)
    fixed = {**decoded["inputs"], **descriptor["settings"]}
    outputs = decoded["outputs"]
    input_storages = {(str(t.device), t.untyped_storage()._cdata) for _, t in iter_tensors(fixed)}
    # Freeze output aliases before any helper can write their shared input.
    outputs = {
        name: value.detach().clone() if isinstance(value, torch.Tensor) and
        (str(value.device), value.untyped_storage()._cdata) in input_storages else value
        for name, value in outputs.items()
    }
    return fixed, outputs


def expected_shapes(inputs, stage):
    n, d = inputs["x"].shape
    if stage == "input_projection":
        p = inputs["w"].shape[1]
        if tuple(inputs["w"].shape) != (d, p) or tuple(inputs["dz"].shape) != (n, p):
            raise ValueError("inconsistent projection matrix shapes")
        return ((n, d), (d, p), (p,) if inputs["is_y_1d"] else (n, p))
    if stage != "input_norm":
        raise ValueError("unknown stage: " + stage)
    if tuple(inputs["dy"].shape) != (n, d) or any(
        tuple(inputs[name].shape) != (n,) for name in ("mean", "rstd")
    ):
        raise ValueError("inconsistent layer-norm input shapes")
    if inputs["learnable"] and any(
        inputs[name] is None or tuple(inputs[name].shape) != (d,) for name in ("weight", "bias")
    ):
        raise ValueError("learnable layer norm requires feature weight and bias")
    return ((n, d), (d,) if inputs["learnable"] else None, (d,) if inputs["learnable"] else None)


def reference_chunks(inputs, stage, rows, columns=256, *, mm_input_dtype=None):
    """Yield (name, slice tuple, FP64 tile), covering each output element once.

    Projection tiles bound both feature axes and every reduction dimension;
    neither a full FP64 weight nor a full FP64 parameter gradient is allocated.
    Norm requires complete feature rows for its two per-row reductions. All
    parameter-gradient reductions include all captured rows, including tail rows.
    """
    import torch

    if min(rows, columns) < 1:
        raise ValueError("reference chunk dimensions must be positive")
    expected_shapes(inputs, stage)

    def cpu(value, *, mm=False):
        value = value.detach().to(device="cpu")
        if mm and mm_input_dtype is not None and value.dtype != torch.float64:
            value = value.to(dtype=mm_input_dtype)
        return value.to(dtype=torch.float64)

    n, d = inputs["x"].shape
    names = STAGES[stage]["outputs"]
    if stage == "input_projection":
        x, w, dz = (inputs[name] for name in ("x", "w", "dz"))
        p = w.shape[1]
        for start in range(0, n, rows):
            rs = slice(start, min(start + rows, n))
            for a in range(0, d, columns):
                cs = slice(a, min(a + columns, d))
                dx = torch.zeros((rs.stop - start, cs.stop - a), dtype=torch.float64)
                for b in range(0, p, columns):
                    ps = slice(b, min(b + columns, p))
                    dx += cpu(dz[rs, ps], mm=True) @ cpu(w[cs, ps], mm=True).t()
                yield names[0], (rs, cs), dx
        for a in range(0, d, columns):
            ds = slice(a, min(a + columns, d))
            for b in range(0, p, columns):
                ps = slice(b, min(b + columns, p))
                dw = torch.zeros((ds.stop - a, ps.stop - b), dtype=torch.float64)
                for start in range(0, n, rows):
                    rs = slice(start, min(start + rows, n))
                    dw += cpu(x[rs, ds], mm=True).t() @ cpu(dz[rs, ps], mm=True)
                yield names[1], (ds, ps), dw
        for b in range(0, p, columns):
            ps = slice(b, min(b + columns, p))
            if inputs["is_y_1d"]:
                db = torch.zeros(ps.stop - b, dtype=torch.float64)
                for start in range(0, n, rows):
                    db += cpu(dz[start:start + rows, ps]).sum(dim=0)
                yield names[2], (ps,), db
            else:
                for start in range(0, n, rows):
                    rs = slice(start, min(start + rows, n))
                    yield names[2], (rs, ps), cpu(dz[rs, ps])
        return

    learnable = inputs["learnable"]
    weight = cpu(inputs["weight"]) if learnable else None
    dw = torch.zeros(d, dtype=torch.float64) if learnable else None
    db = torch.zeros(d, dtype=torch.float64) if learnable else None
    for start in range(0, n, rows):
        rs = slice(start, min(start + rows, n))
        x, dy = cpu(inputs["x"][rs]), cpu(inputs["dy"][rs])
        rstd = cpu(inputs["rstd"][rs])[:, None]
        xhat = (x - cpu(inputs["mean"][rs])[:, None]) * rstd
        wdy = dy * weight if learnable else dy
        dx = rstd * (wdy - wdy.mean(dim=1, keepdim=True)
                     - xhat * (xhat * wdy).mean(dim=1, keepdim=True))
        yield names[0], (rs, slice(None)), dx
        if learnable:
            dw += (dy * xhat).sum(dim=0)
            db += dy.sum(dim=0)
    if learnable:
        yield names[1], (slice(None),), dw
        yield names[2], (slice(None),), db


def compare_reference(inputs, stage, outputs, *, rows, columns=256, rtol, atol, mm_input_dtype=None):
    import torch

    names = STAGES[stage]["outputs"]
    if len(outputs) != len(names):
        raise ValueError("unexpected isolated output structure")
    shapes = expected_shapes(inputs, stage)
    actual = dict(zip(names, outputs))
    checks = {}
    for name, shape in zip(names, shapes):
        value = actual[name]
        if shape is None:
            if value is not None:
                raise ValueError("expected absent parameter gradient: " + name)
            continue
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError("reference output shape changed: " + name)
        checks[name] = {"elements": 0, "expected_elements": math.prod(shape),
                        "mismatched_elements": 0, "nonfinite_elements": 0,
                        "max_abs_diff": 0.0, "max_abs_actual": 0.0, "max_abs_reference": 0.0}
    for name, selection, expected in reference_chunks(
        inputs, stage, rows, columns, mm_input_dtype=mm_input_dtype,
    ):
        value = actual[name][selection].detach().to(device="cpu", dtype=torch.float64)
        finite = torch.isfinite(value) & torch.isfinite(expected)
        close = torch.isclose(value, expected, rtol=rtol, atol=atol) & finite
        info = checks[name]
        info["elements"] += value.numel()
        info["mismatched_elements"] += int((~close).sum())
        info["nonfinite_elements"] += int((~finite).sum())
        for key, tensor in (("max_abs_diff", value - expected),
                            ("max_abs_actual", value), ("max_abs_reference", expected)):
            if tensor.numel():
                maximum = float(torch.nan_to_num(tensor.abs(), nan=float("inf"), posinf=float("inf")).max())
                info[key] = max(info[key], maximum)
    for info in checks.values():
        if info["elements"] != info["expected_elements"]:
            raise RuntimeError("CPU reference did not cover every output element")
    return {"passed": all(info["mismatched_elements"] == 0 for info in checks.values()),
            "coverage": "all output elements; all reduction rows and columns",
            "device": "cpu", "dtype": "torch.float64", "rtol": rtol, "atol": atol,
            "rows_per_chunk": rows, "projection_columns_per_chunk": columns,
            "uses_saved_forward_statistics": stage == "input_norm",
            "mm_autocast_input_dtype": str(mm_input_dtype) if mm_input_dtype is not None else None,
            "outputs": checks}


def execute_stage(stage, fixed, helper_module):
    """Use the production helper unchanged; no parent operation is reachable."""
    helper = (helper_module.triton_addmm_bwd if stage == "input_projection"
              else helper_module.triton_weighted_layer_norm_bwd)
    return helper(**fixed)


def captured_norm_configs(descriptor, inputs, args):
    if not inputs["learnable"] and args.block_n is not None:
        raise ValueError("block-n requires learnable input_norm")
    if inputs["x"].shape[0] == 0:
        return {}
    metadata = descriptor.get("launch_metadata") or {}
    if metadata.get("format_version") != 1:
        raise ValueError("nonempty norm stage requires original launch_metadata version 1")
    configs = {}
    for role in (("dx", "reduction") if inputs["learnable"] else ("dx",)):
        config = (metadata.get(role) or {}).get("launch_config")
        if not config or any(key not in config for key in ("BLOCK_D", "num_warps", "num_stages")):
            raise ValueError("missing captured norm launch configuration: " + role)
        if inputs["learnable"] and "BLOCK_N" not in config:
            raise ValueError("missing captured norm BLOCK_N: " + role)
        configs[role] = dict(config)
    if configs["dx"]["BLOCK_D"] != inputs["BLOCK_D"]:
        raise ValueError("captured DX BLOCK_D disagrees with original helper settings")
    for name, value in (("BLOCK_N", args.block_n), ("num_warps", args.num_warps),
                        ("num_stages", args.num_stages)):
        if value is not None:
            configs["dx"][name] = value
    if args.max_vgpr is not None:
        configs["dx"]["llvm_fn_attrs"] = f"amdgpu-num-vgpr={args.max_vgpr}"
    return configs


@contextmanager
def norm_launches(triton, module, inputs, configs, report):
    """Pin autotuners and record actual inner JIT launches, restoring on error."""
    kernels = ({"dx": module._weighted_layer_norm_bwd_dx, "reduction": module._layer_norm_bwd_dwdb}
               if inputs["learnable"] else {"dx": module._layer_norm_bwd_dx})
    restore = []
    observed = {}
    try:
        for role, config in configs.items():
            kernel = kernels[role]
            # BLOCK_D is selected by the public helper separately for each
            # kernel; never duplicate it in an autotuner's configuration kwargs.
            caller_config = {"BLOCK_D": config["BLOCK_D"]}
            config = {k: v for k, v in config.items() if k not in ("BLOCK_D", "IS_SWISH", "N")}
            if hasattr(kernel, "configs"):
                fields = set(inspect.signature(triton.Config).parameters) - {"kwargs"}
                tuning = {k: v for k, v in config.items() if k in fields}
                kwargs = {k: v for k, v in config.items() if k not in fields}
                pinned = triton.Config(kwargs, **tuning)
                old_configs, old_cache = kernel.configs, dict(kernel.cache)
                restore.append((kernel, "configs", old_configs))
                restore.append((kernel, "cache", old_cache))
                kernel.configs = [pinned]
                kernel.cache.clear()
                jit = kernel.fn
                overrides = caller_config
            else:
                jit, overrides = kernel, {**config, **caller_config}
            original_run = jit.run
            restore.append((jit, "run", original_run))

            def launch(*positional, _original=original_run, _role=role, _overrides=overrides, **kwargs):
                kwargs = {**kwargs, **_overrides}
                compiled = _original(*positional, **kwargs)
                if compiled is not None and not kwargs.get("warmup", False):
                    key = (_role, compiled.hash)
                    if key not in observed:
                        metadata = compiled.metadata
                        metadata = metadata._asdict() if hasattr(metadata, "_asdict") else str(metadata)
                        observed[key] = {
                            "role": _role, "compiled_hash": compiled.hash, "kernel_name": compiled.name,
                            "registers": getattr(compiled, "n_regs", None),
                            "spills": getattr(compiled, "n_spills", None),
                            "compiled_metadata": metadata,
                            "launch_config": {k: v for k, v in kwargs.items()
                                              if isinstance(v, (bool, int, float, str)) and k != "warmup"},
                            "artifact_sha256": {
                                k: hashlib.sha256(v if isinstance(v, bytes) else v.encode()).hexdigest()
                                for k, v in compiled.asm.items() if k in ("llir", "amdgcn", "hsaco")
                            },
                        }
                    report["actual_norm_launches"] = list(observed.values())
                return compiled

            jit.run = launch
        yield
    finally:
        for obj, name, value in reversed(restore):
            setattr(obj, name, value)


def run(args, report, publish):
    import torch
    import triton
    from contextlib import nullcontext
    from scripts.replay_nan_tripwire import (
        changed_capture_inputs, compare_original_outputs, iter_tensors, _restore_runtime_state,
    )
    from generative_recommenders.dlrm_v4.train.nan_tripwire import _all_finite, _all_within_bound

    chunk_size = 16 * 1024**2

    def flags(value):
        checks = {}
        for name, tensor in iter_tensors(value):
            if tensor.is_floating_point():
                checks[name + ":finite"] = _all_finite(tensor, chunk_size)
                checks[name + ":within_bound"] = _all_within_bound(tensor, args.max_abs, chunk_size)
        return checks

    def resolve(checks):
        return dict(zip(checks, torch.stack(list(checks.values())).cpu().tolist())) if checks else {}

    capture = torch.load(args.capture, map_location="cpu", mmap=True, weights_only=False)
    descriptor = stage_descriptor(capture, args.stage)
    parent = descriptor["parent_replay"]
    changed = changed_capture_inputs(capture)
    if changed:
        raise ValueError("tracked input mutation before capture: " + str(changed))
    for name, value in capture["report"]["environment"].items():
        if not name.startswith("NAN_TRIPWIRE_"):
            os.environ.setdefault(name, value)
    torch.cuda.set_device(0)
    if args.stage == "input_projection":
        from generative_recommenders.ops.triton import triton_addmm as module
    else:
        from generative_recommenders.ops.triton import triton_layer_norm as module
    _restore_runtime_state(parent.get("runtime_state"))
    amp_enabled = parent.get("cuda_autocast_enabled", False)
    amp_dtype = parent.get("cuda_autocast_dtype", torch.bfloat16)
    report["source"] = {
        "path": str(args.capture.resolve()), "bytes": args.capture.stat().st_size,
        "step": capture["report"]["step"], "metadata": capture["report"].get("metadata"),
        "torch": str(torch.__version__), "hip": torch.version.hip, "triton": triton.__version__,
        "runtime_state": parent.get("runtime_state"),
        "cuda_autocast_enabled": amp_enabled, "cuda_autocast_dtype": str(amp_dtype),
        "stage_identity": descriptor.get("identity"),
        "selected_sequence": capture["report"].get("selected_sequence", capture["report"].get("first_bad_sequence")),
        "original_launch_metadata": descriptor.get("launch_metadata"),
        "helper_source": module.__file__,
        "helper_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "effective_environment": {name: os.environ.get(name) for name in capture["report"]["environment"]
                                  if not name.startswith("NAN_TRIPWIRE_")},
    }
    report["provenance"] = {
        "stage_inputs": "original_invocation_stage_capture",
        "parent_preparation": "none; parent tensors are not decoded",
        "original_comparison": "diagnostic exact comparison to original stage outputs",
        "baseline": "first isolated execution; exact repetition is only a repeatability oracle",
        "capture_semantics": "retained tensors copied at tripwire check; later untracked writes not excluded",
    }
    report["stage_operation"] = ("triton_addmm_bwd (bias sum and two torch.mm calls)"
                                 if args.stage == "input_projection"
                                 else "triton_weighted_layer_norm_bwd (DX and parameter reduction)")
    names = STAGES[args.stage]["outputs"]
    report["stage_output_names"] = list(names)
    report["launch_overrides"] = {name: value for name, value in (
        ("BLOCK_N", args.block_n), ("num_warps", args.num_warps),
        ("num_stages", args.num_stages), ("max_vgpr", args.max_vgpr),
    ) if value is not None}
    publish("stage_decode_start")
    with torch.no_grad():
        fixed, original_outputs = original_stage_inputs(capture, args.stage, "cuda")
        expected_shapes(fixed, args.stage)
        configs = captured_norm_configs(descriptor, fixed, args) if args.stage == "input_norm" else {}
        report["selected_norm_configs"] = configs
        report["input_layouts"] = {
            name: {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
                   "storage_offset": t.storage_offset()}
            for name, t in iter_tensors(fixed)
        }
        report["settings"] = descriptor["settings"]
        del capture, descriptor, parent
        report["stage_input_checks"] = resolve(flags(fixed))
        if not all(report["stage_input_checks"].values()):
            report["status"] = "INPUT_FAILURE"
            publish("input_failure")
            return 1
        tensors = dict(iter_tensors(fixed))
        versions = {name: None if t.is_inference() else t._version for name, t in tensors.items()}
        snapshots = {name: t.clone() for name, t in tensors.items()} if args.check_input_contents else {}

        def input_checks():
            checks = {}
            for name, tensor in tensors.items():
                if versions[name] is not None and tensor._version != versions[name]:
                    raise RuntimeError("tracked input mutation: " + name)
                if name in snapshots:
                    checks[name + ":input_unchanged"] = all_equal(
                        torch, tensor, snapshots[name], chunk_size=chunk_size, bytewise=True,
                    )
            return checks

        launch_context = norm_launches(triton, module, fixed, configs, report) if configs else nullcontext()
        with launch_context, torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            baseline = execute_stage(args.stage, fixed, module)
            if len(baseline) != len(names):
                raise ValueError("unexpected isolated output structure")
            baseline = dict(zip(names, baseline))
            report["baseline_checks"] = resolve({**flags(baseline), **input_checks()})
            report["original_stage_comparison"] = compare_original_outputs(baseline, original_outputs, rtol=0, atol=0)
            del original_outputs
            if not all(report["baseline_checks"].values()):
                report["status"] = "BASELINE_FAILURE"
                publish("baseline_failure")
                return 1
            if args.reference_cpu:
                publish("reference_start")
                report["cpu_reference"] = compare_reference(
                    fixed, args.stage, tuple(baseline.values()), rows=args.reference_rows,
                    columns=args.reference_columns, rtol=args.reference_rtol, atol=args.reference_atol,
                    mm_input_dtype=amp_dtype if amp_enabled and args.stage == "input_projection" else None,
                )
                report["reference_input_checks"] = resolve(input_checks())
                if not report["cpu_reference"]["passed"] or not all(report["reference_input_checks"].values()):
                    report["status"] = "REFERENCE_FAILURE"
                    publish("reference_failure")
                    return 1
            references = dict(iter_tensors(baseline))
            pending = []
            started = time.monotonic()
            publish("stress_start")
            for iteration in range(1, args.repeat + 1):
                outputs = execute_stage(args.stage, fixed, module)
                if len(outputs) != len(names):
                    raise ValueError("repeat output structure changed")
                outputs = dict(zip(names, outputs))
                actual = dict(iter_tensors(outputs))
                if actual.keys() != references.keys():
                    raise ValueError("repeat output tensor structure changed")
                checks = flags(outputs)
                for name, tensor in actual.items():
                    checks[name + ":exact_repeat"] = all_equal(torch, tensor, references[name], chunk_size=chunk_size)
                checks.update(input_checks())
                pending.extend((iteration, name, flag) for name, flag in checks.items())
                del outputs, actual
                if iteration % args.check_every and iteration != args.repeat:
                    continue
                values = torch.stack([flag for _, _, flag in pending]).cpu().tolist()
                failures = [{"iteration": i, "check": name}
                            for (i, name, _), passed in zip(pending, values) if not passed]
                report["checks"].append({"through": iteration, "failures": failures,
                                         "seconds": time.monotonic() - started})
                report["iterations_completed"] = iteration
                report["status"] = "FAIL" if failures else ("PASS" if iteration == args.repeat else "RUNNING")
                publish("check")
                pending.clear()
                if failures:
                    return 1
    return 0


def main():
    args = arguments()
    for key, value in {"AMDGCN_USE_BUFFER_OPS": "0", "PYTORCH_CUDA_ALLOC_CONF": "",
                       "PYTORCH_ALLOC_CONF": "", "HSA_ENABLE_COREDUMP": "0"}.items():
        os.environ.setdefault(key, value)
    report = {"status": "RUNNING", "stage": args.stage, "repeat": args.repeat,
              "max_abs": args.max_abs, "check_every": args.check_every,
              "check_input_contents": args.check_input_contents,
              "iterations_completed": 0, "checks": []}

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
