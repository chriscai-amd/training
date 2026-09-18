#!/usr/bin/env python3
"""Stress LN-multiply-dropout forward from a real HSTU output forward capture.

Restore the captured RNG and run HSTUComputeOutputFunction.forward once with one
pinned LN configuration. Retain the actual kernel inputs and generated dropout
mask, then repeat ONLY _ln_mul_dropout_fwd_rng with those inputs and one reused
output allocation. No mask generation, GEMM, or RNG restoration occurs in the
stress loop. Y/mean/rstd must be finite and exactly equal to the preparation
result produced with the same configuration. This detects repeat nondeterminism;
it does not establish independent numerical correctness of that first result.

The complete preparation output is compared to the originally captured output
for diagnosis. This does not gate the run unless BOTH --original-rtol and
--original-atol are explicitly supplied: changing the configuration can change
floating-point rounding. A preparation failure is distinct from a repeat failure.

Examples inside the prepared training container with custom Triton 3.8:
  python scripts/repro_hstu_output_capture.py capture.pt --block-n 8 --num-warps 1
  python scripts/repro_hstu_output_capture.py capture.pt --block-n 8 --num-warps 4
  python scripts/repro_hstu_output_capture.py capture.pt --max-vgpr 256 --report capped.json

Inputs and dropout mask come from the captured forward, including its restored
CPU RNG or explicit seed. The replayed seed is checked against ctx_after_forward
when available. Only the separated-mask training path is supported. Mask contents
are always checked for mutation; --check-input-contents additionally checks all
floating input bytes. Version counters guard all inputs by default. Finite and
comparison scans use bounded chunks, and scalar readback occurs every
--check-every iterations. Checks change timing and memory traffic.
Only load trusted local captures: torch.load uses Python pickle.
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


CHUNK_ELEMENTS = 16 * 1024**2
OUTPUT_NAMES = ("Y", "Mean", "Rstd")
INPUT_NAMES = ("X", "U", "W", "B", "RANDOM_MASK")
KERNEL_ARGUMENTS = {
    *INPUT_NAMES, *OUTPUT_NAMES, "N", "D", "eps", "dropout_ratio", "stride_x",
    "stride_u", "stride_y", "stride_mask", "SILU_U", "BLOCK_D", "TRAINING",
    "CONCAT_U", "CONCAT_X", "MUL_U_ACTIVATION_TYPE",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--block-n", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--max-vgpr", type=int)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--check-input-contents", action="store_true")
    parser.add_argument("--original-rtol", type=float, help="with original-atol, require preparation agreement")
    parser.add_argument("--original-atol", type=float, help="with original-rtol, require preparation agreement")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.check_every < 1 or (args.max_vgpr is not None and args.max_vgpr < 1):
        parser.error("repeat, check-every and max-vgpr must be positive")
    tolerances = (args.original_rtol, args.original_atol)
    if (tolerances[0] is None) != (tolerances[1] is None):
        parser.error("original-rtol and original-atol must be supplied together")
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in tolerances):
        parser.error("original tolerances must be finite and nonnegative")
    return args


def forward_arguments(replay, forward):
    bound = inspect.signature(forward).bind(None, *replay["args"], **replay["kwargs"])
    bound.apply_defaults()
    values = dict(bound.arguments)
    values.pop(next(iter(inspect.signature(forward).parameters)))
    if values["group_norm"] or not values["training"] or not 0 < values["dropout_ratio"] < 1:
        raise ValueError("requires training with nonzero dropout and group_norm=False")
    return values


def force_config(triton, kernel, block_n, num_warps, max_vgpr):
    kwargs = {"BLOCK_N": block_n}
    if max_vgpr is not None:
        from triton.backends.amd.compiler import HIPOptions

        if "llvm_fn_attrs" not in HIPOptions.__dataclass_fields__:
            raise RuntimeError("this Triton build does not support llvm_fn_attrs")
        kwargs["llvm_fn_attrs"] = f"amdgpu-num-vgpr={max_vgpr}"
    config = triton.Config(kwargs, num_warps=num_warps, num_stages=1)
    kernel.configs = [config]
    kernel.cache.clear()
    return config


def bind_kernel_arguments(kernel, args, kwargs):
    if len(args) > len(kernel.arg_names):
        raise ValueError("too many positional LN kernel arguments")
    values = dict(zip(kernel.arg_names, args))
    for name, value in kwargs.items():
        if name in ("grid", "warmup"):
            continue
        if name in values:
            raise ValueError(f"duplicate LN kernel argument: {name}")
        values[name] = value
    # BLOCK_N is supplied by the singleton autotuner config at launch.
    values.pop("BLOCK_N", None)
    if values.keys() != KERNEL_ARGUMENTS:
        raise ValueError(f"unexpected LN kernel argument layout: {sorted(values)}")
    return values


def prepare(capture, linear, *, rtol, atol):
    """Capture exact launch inputs after layout conversion and mask generation."""
    from scripts.replay_nan_tripwire import compare_original_outputs, replay_once

    kernel = linear._ln_mul_dropout_fwd_rng
    original_run = kernel.run
    original_helper = linear.triton_layer_norm_mul_dropout_fwd
    calls, helper_results = [], []

    def retain_launch(*args, **kwargs):
        if calls:
            raise RuntimeError("expected exactly one separated-RNG LN forward launch")
        calls.append(bind_kernel_arguments(kernel, args, kwargs))
        return original_run(*args, **kwargs)

    def retain_helper(*args, **kwargs):
        if helper_results:
            raise RuntimeError("expected exactly one LN forward helper call")
        result = original_helper(*args, **kwargs)
        helper_results.append(result)
        return result

    kernel.run = retain_launch
    linear.triton_layer_norm_mul_dropout_fwd = retain_helper
    try:
        output, payload = replay_once(capture, "cuda", snapshot_original_outputs=True)
    finally:
        kernel.run = original_run
        linear.triton_layer_norm_mul_dropout_fwd = original_helper
    if len(calls) != 1 or len(helper_results) != 1:
        raise RuntimeError("forward did not use exactly one separated-mask LN call")
    helper = helper_results[0]
    if len(helper) != 7 or helper[6] is not calls[0]["RANDOM_MASK"]:
        raise RuntimeError("helper output mask differs from the intercepted kernel input")
    for name, tensor in zip(OUTPUT_NAMES, helper[:3]):
        if tensor is not calls[0][name]:
            raise RuntimeError(f"helper output {name} differs from the intercepted kernel output")
    comparison = compare_original_outputs(output, payload["outputs"], rtol=rtol, atol=atol)
    return calls[0], helper[5], comparison


def all_equal(torch, left, right, *, bytewise=False):
    from generative_recommenders.dlrm_v4.train.nan_tripwire import _finite_chunks

    if any(getattr(left, name) != getattr(right, name) for name in ("shape", "dtype", "device")):
        raise ValueError("comparison tensors differ in shape, dtype or device")
    if bytewise:
        left, right = (value.reshape(1) if value.ndim == 0 else value for value in (left, right))
        left, right = left.view(torch.uint8), right.view(torch.uint8)
    result = None
    for a, b in zip(_finite_chunks(left, CHUNK_ELEMENTS), _finite_chunks(right, CHUNK_ELEMENTS)):
        flag = (a == b).all()
        result = flag if result is None else torch.logical_and(result, flag)
    return result


def validate_launch(torch, launch):
    n, d = launch["N"], launch["D"]
    if n < 1 or d < 1 or not launch["TRAINING"]:
        raise ValueError("requires a nonempty separated-mask training launch")
    columns = d * (1 + int(launch["CONCAT_U"]) + int(launch["CONCAT_X"]))
    shapes = {"X": (n, d), "U": (n, d), "W": (d,), "B": (d,),
              "RANDOM_MASK": (n, d), "Y": (n, columns), "Mean": (n,), "Rstd": (n,)}
    for name, shape in shapes.items():
        tensor = launch[name]
        if tuple(tensor.shape) != shape or tensor.stride(-1) != 1:
            raise ValueError(f"unexpected {name} shape or feature stride")
    if launch["RANDOM_MASK"].dtype != torch.int8:
        raise ValueError("requires the production packed int8 dropout mask")
    for name, stride in (("X", "stride_x"), ("U", "stride_u"), ("Y", "stride_y"),
                         ("RANDOM_MASK", "stride_mask")):
        if launch[name].stride(0) != launch[stride]:
            raise ValueError(f"captured {stride} disagrees with {name}")


def run(args, report, publish):
    for name, value in {"AMDGCN_USE_BUFFER_OPS": "0", "TRITON_FULL_AUTOTUNE": "0",
                        "TRITON_ALLOW_PIPELINING": "0", "PYTORCH_CUDA_ALLOC_CONF": "",
                        "PYTORCH_ALLOC_CONF": "", "HSA_ENABLE_COREDUMP": "0"}.items():
        os.environ.setdefault(name, value)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import triton
    from scripts.replay_nan_tripwire import changed_capture_inputs
    from scripts.repro_hstu_attention_capture import compiled_metadata
    from generative_recommenders.dlrm_v4.train.nan_tripwire import _all_finite, _finite_chunks

    capture = torch.load(args.capture, map_location="cpu", mmap=True, weights_only=False)
    if capture.get("format_version") != 1:
        raise ValueError("requires an original tripwire capture with reference outputs")
    changed = changed_capture_inputs(capture)
    if changed:
        raise ValueError("tracked input mutation before capture: " + ", ".join(changed))
    replay = capture["payload"]["replay"]
    if replay is None or replay["class"] != "HSTUComputeOutputFunction" or replay["direction"] != "forward":
        raise ValueError("requires an HSTUComputeOutputFunction forward capture")
    for name, value in capture["report"]["environment"].items():
        if not name.startswith("NAN_TRIPWIRE_"):
            os.environ.setdefault(name, value)
    torch.cuda.set_device(0)
    from generative_recommenders.ops.triton import triton_hstu_linear as linear

    named = forward_arguments(replay, linear.HSTUComputeOutputFunction.forward)
    kernel = linear._ln_mul_dropout_fwd_rng
    config = force_config(triton, kernel, args.block_n, args.num_warps, args.max_vgpr)
    config_record = dict(kwargs=dict(config.kwargs), num_warps=config.num_warps, num_stages=config.num_stages)
    amp_enabled = replay.get("cuda_autocast_enabled", False)
    amp_dtype = replay.get("cuda_autocast_dtype", torch.bfloat16)
    after_forward = replay.get("ctx_after_forward", {})
    if after_forward.get("has_random_mask") is False:
        raise ValueError("capture used the fused-RNG path; a separated-mask capture is required")
    report["source"] = dict(capture=str(args.capture.resolve()), bytes=args.capture.stat().st_size,
                          step=capture["report"].get("step"), metadata=capture["report"].get("metadata"),
                          captured_runtime_state=replay.get("runtime_state"),
                          captured_seed=after_forward.get("seed"),
                          captured_has_random_mask=after_forward.get("has_random_mask"))
    require_original = args.original_rtol is not None
    rtol, atol = (args.original_rtol, args.original_atol) if require_original else (0.0, 0.0)
    report["preparation"] = dict(status="RUNNING", config=config_record,
                                 original_comparison_required=require_original, rtol=rtol, atol=atol)
    publish({"event": "preparation", **report["preparation"]})
    with torch.no_grad():
        launch, seed, comparison = prepare(capture, linear, rtol=rtol, atol=atol)
        validate_launch(torch, launch)
        seed_matches = "seed" not in after_forward or seed == after_forward["seed"]
        inputs = {name: launch[name] for name in INPUT_NAMES}
        references = {name: launch[name] for name in OUTPUT_NAMES}
        setup = {name + "_finite": _all_finite(tensor, CHUNK_ELEMENTS)
                 for name, tensor in {**inputs, **references}.items() if name != "RANDOM_MASK"}
        mask_max = (1 << (1 + int(launch["CONCAT_U"]) + int(launch["CONCAT_X"]))) - 1
        mask_flags = [(chunk >= 0).all() & (chunk <= mask_max).all()
                      for chunk in _finite_chunks(inputs["RANDOM_MASK"], CHUNK_ELEMENTS)]
        setup["mask_bits_valid"] = torch.stack(mask_flags).all()
        setup_values = torch.stack(list(setup.values())).cpu().tolist()
        report["preparation"].update(
            original_comparison=comparison, replayed_seed=seed, seed_matches_capture=seed_matches,
            checks=dict(zip(setup, setup_values)), compiled_kernels=compiled_metadata(kernel),
        )
        if not seed_matches or not all(setup_values):
            report["status"] = report["preparation"]["status"] = "PREPARATION_FAIL"
            publish({"event": "preparation", **report["preparation"]})
            return 1
        report["preparation"]["status"] = "PASS" if not require_original or comparison["passed"] else "ORIGINAL_MISMATCH"
        publish({"event": "preparation", **report["preparation"]})
        if require_original and not comparison["passed"]:
            report["status"] = "ORIGINAL_MISMATCH"
            return 1
        del named, replay, capture
        versions = {name: tensor._version for name, tensor in inputs.items()}
        snapshots = {name: tensor.clone() for name, tensor in inputs.items()
                     if args.check_input_contents or name == "RANDOM_MASK"}
        # Preserve the preparation results; stress writes a separate allocation.
        outputs = {name: torch.empty_like(tensor) for name, tensor in references.items()}
        launch.update(outputs)
        launch["stride_y"] = outputs["Y"].stride(0)
        validate_launch(torch, launch)
        properties = torch.cuda.get_device_properties(0)
        report["configuration"] = dict(
            config=config_record, torch=str(torch.__version__), hip=torch.version.hip, triton=triton.__version__,
            arch=getattr(properties, "gcnArchName", properties.name), source=linear.__file__,
            source_sha256=hashlib.sha256(Path(linear.__file__).read_bytes()).hexdigest(),
            kernel_parameters={name: value for name, value in launch.items() if not isinstance(value, torch.Tensor)},
            autocast=dict(enabled=amp_enabled, dtype=str(amp_dtype)),
            FUSE_OUTPUT_LN_RNG_BLACKWELL=linear.FUSE_OUTPUT_LN_RNG_BLACKWELL,
            COMPUTE_OUTPUT_LN_FAST_DROPOUT=linear.COMPUTE_OUTPUT_LN_FAST_DROPOUT,
            input_layouts={name: dict(shape=list(t.shape), stride=list(t.stride()),
                                      storage_offset=t.storage_offset(), dtype=str(t.dtype))
                           for name, t in inputs.items()},
            output_layouts={name: dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype))
                            for name, t in outputs.items()},
            finite_chunk_elements=CHUNK_ELEMENTS, reference="preparation output at the same pinned configuration",
            mask_policy="one mask generated by full replay with restored RNG; exact mask bytes reused and checked",
            env={name: os.environ.get(name) for name in ("AMDGCN_USE_BUFFER_OPS", "TRITON_ALLOW_PIPELINING",
                 "TRITON_FULL_AUTOTUNE", "AMD_SERIALIZE_KERNEL", "TRITON_HIP_USE_EXPERT_SCHEDULING",
                 "TRITON_HIP_USE_COEXEC_SCHEDULER", "AMDGCN_SCALARIZE_PACKED_FOPS",
                 "TRITON_HIP_USE_IN_THREAD_TRANSPOSE")},
        )
        publish({"event": "configuration", **report["configuration"]})
        pending = []
        started = time.monotonic()
        with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            for iteration in range(1, args.repeat + 1):
                if (len(kernel.configs) != 1 or kernel.configs[0] is not config
                        or config.kwargs != config_record["kwargs"]
                        or config.num_warps != config_record["num_warps"]
                        or config.num_stages != config_record["num_stages"]):
                    raise RuntimeError("pinned LN configuration changed during stress")
                kernel[(triton.cdiv(launch["N"], args.block_n),)](**launch)
                changed = [name for name, tensor in inputs.items() if tensor._version != versions[name]]
                if changed:
                    raise RuntimeError(f"tracked input mutation at iteration {iteration}: {changed}")
                flags = {}
                for name, tensor in outputs.items():
                    flags[name + "_finite"] = _all_finite(tensor, CHUNK_ELEMENTS)
                    flags[name + "_matches_preparation"] = all_equal(torch, tensor, references[name])
                for name, snapshot in snapshots.items():
                    flags[name + "_unchanged"] = all_equal(torch, inputs[name], snapshot, bytewise=True)
                pending.extend((iteration, name, flag) for name, flag in flags.items())
                report["iterations_completed"] = iteration
                if iteration % args.check_every and iteration != args.repeat:
                    continue
                values = torch.stack([flag for _, _, flag in pending]).cpu().tolist()
                failures = [{"iteration": step, "check": name}
                            for (step, name, _), passed in zip(pending, values) if not passed]
                check = dict(checked_through_iteration=iteration, failures=failures,
                             elapsed_seconds=time.monotonic() - started)
                report["checks"].append(check)
                report["compiled_kernels"] = compiled_metadata(kernel)
                pending.clear()
                report["status"] = "REPEAT_FAIL" if failures else ("PASS" if iteration == args.repeat else "RUNNING")
                publish({"event": "check", **check, "status": report["status"]})
                if failures:
                    return 1
    return 0


def main():
    args = arguments()
    report = dict(capture=str(args.capture), block_n=args.block_n, num_warps=args.num_warps,
                  max_vgpr=args.max_vgpr, repeat=args.repeat, check_every=args.check_every,
                  check_input_contents=args.check_input_contents, status="RUNNING",
                  iterations_completed=0, checks=[])

    def publish(record):
        print(json.dumps(record, default=str, allow_nan=False), flush=True)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")

    try:
        result = run(args, report, publish)
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        publish({"event": "error", "error": report["error"]})
        raise
    publish({"event": "summary", "status": report["status"],
             "iterations_completed": report["iterations_completed"],
             "compiled_kernels": report.get("compiled_kernels", [])})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
