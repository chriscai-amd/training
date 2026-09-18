#!/usr/bin/env python3
"""Stress weighted layer-norm backward using the real step-55 tripwire capture.

Run the complete captured backward once with attention capped at 256 VGPRs,
hook its layer-norm call, and require all five preparation outputs to match the
captured outputs exactly. Then retain only the layer-norm inputs and references
and repeatedly call the public layer-norm backward. Every DX must match the
captured DX with rtol=atol=0, and DX/dweight/dbias must all remain finite.
The parameter gradients are compared diagnostically on the first isolated call:
changing BLOCK_N changes their reduction order and may change rounding.

Examples inside the matching training image:
  python scripts/repro_hstu_layer_norm_capture.py capture.pt --block-n 8 --report ln8.json
  python scripts/repro_hstu_layer_norm_capture.py capture.pt --block-n 1 --report ln1.json
  python scripts/repro_hstu_layer_norm_capture.py capture.pt --block-n 8 --max-vgpr 256

Preparation always uses LN BLOCK_N=8, num_warps=1, without an LN VGPR cap.
--max-vgpr affects only the isolated weighted DX kernel. Both LN configurations
use num_warps=1 and num_stages=1. Inputs are reused; the public wrapper allocates
outputs and partial-reduction buffers normally. The first output is retained
until the first diagnostic comparison at a scalar checkpoint. Scalar checks are
copied to the host every --check-every iterations. Checks alter timing and memory
traffic.
Default input guards use version counters; --check-input-contents additionally
compares every input byte each iteration to detect untracked kernel writes.
An exact mismatch establishes disagreement with the capture, not its cause.
Only load trusted local captures: torch.load uses Python pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--block-n", type=int, choices=(1, 8), default=8)
    parser.add_argument("--max-vgpr", type=int, help="cap VGPRs for the isolated LN DX kernel")
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--check-input-contents", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.check_every < 1 or (args.max_vgpr is not None and args.max_vgpr < 1):
        parser.error("repeat, check-every and max-vgpr must be positive")
    return args


def force_config(triton, kernel, block_n, max_vgpr=None):
    """A singleton skips autotuning, including any previously cached winner."""
    kwargs = {"BLOCK_N": block_n}
    if max_vgpr is not None:
        from triton.backends.amd.compiler import HIPOptions

        if "llvm_fn_attrs" not in HIPOptions.__dataclass_fields__:
            raise RuntimeError("this Triton build does not support llvm_fn_attrs")
        kwargs["llvm_fn_attrs"] = f"amdgpu-num-vgpr={max_vgpr}"
    # Setting configs directly bypasses common.clamp_num_stages.
    config = triton.Config(kwargs, num_warps=1, num_stages=1)
    kernel.configs = [config]
    kernel.cache.clear()
    return dict(kwargs=config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)


def output_flags(torch, outputs, expected_dx):
    from scripts.repro_hstu_attention_capture import all_chunks

    if len(outputs) != 3:
        raise ValueError("weighted layer norm must return three tensors")
    dx = outputs[0]
    if any(getattr(dx, key) != getattr(expected_dx, key) for key in ("shape", "dtype", "device")):
        raise ValueError("DX shape, dtype or device differs from the captured reference")
    flags = {
        name + "_finite": all_chunks(torch, tensor, lambda chunk, _: torch.isfinite(chunk).all())
        for name, tensor in zip(("dx", "dweight", "dbias"), outputs)
    }
    # Exact equality has isclose(rtol=0, atol=0, equal_nan=False) semantics here;
    # the separate finite check rejects even matching signed infinities.
    flags["dx_matches_original"] = all_chunks(
        torch, dx, lambda chunk, start: (chunk == expected_dx[start:start + len(chunk)]).all()
    )
    return flags


def prepare(torch, capture, preproc):
    """Capture the actual GEMM-produced dy without duplicating preprocessing."""
    from scripts.replay_nan_tripwire import compare_original_outputs, replay_once

    calls = []
    original = preproc.triton_weighted_layer_norm_bwd

    def retain_inputs(**kwargs):
        if calls:
            raise RuntimeError("expected exactly one weighted layer-norm backward call")
        calls.append(dict(kwargs))
        return original(**kwargs)

    preproc.triton_weighted_layer_norm_bwd = retain_inputs
    try:
        outputs, payload = replay_once(capture, "cuda", snapshot_original_outputs=True)
    finally:
        preproc.triton_weighted_layer_norm_bwd = original
    if len(calls) != 1:
        raise RuntimeError("preprocessing did not call weighted layer-norm backward")
    comparison = compare_original_outputs(outputs, payload["outputs"], rtol=0, atol=0)
    return calls[0], tuple(payload["outputs"][:3]), comparison


def run(args, report, publish):
    # Preparation must not inherit an uncapped attention setting from a previous arm.
    os.environ["HSTU_BWD_MAX_VGPR"] = "256"
    for name, value in {"AMDGCN_USE_BUFFER_OPS": "0", "TRITON_FULL_AUTOTUNE": "0",
                        "TRITON_ALLOW_PIPELINING": "0", "PYTORCH_CUDA_ALLOC_CONF": "",
                        "PYTORCH_ALLOC_CONF": "", "HSA_ENABLE_COREDUMP": "0"}.items():
        os.environ.setdefault(name, value)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import triton
    from scripts.replay_nan_tripwire import changed_capture_inputs, compare_original_outputs, iter_tensors
    from scripts.repro_hstu_attention_capture import all_chunks, compiled_metadata

    capture = torch.load(args.capture, map_location="cpu", mmap=True, weights_only=False)
    if capture.get("format_version") != 1:
        raise ValueError("requires an original tripwire capture with reference outputs")
    changed = changed_capture_inputs(capture)
    if changed:
        raise ValueError("tracked input mutation before capture: " + ", ".join(changed))
    replay = capture["payload"]["replay"]
    if replay is None or replay["class"] != "_HSTUPreprocessAndAttentionFunction" or replay["direction"] != "backward":
        raise ValueError("requires a preprocess-and-attention backward capture")
    for name, value in capture["report"]["environment"].items():
        if not name.startswith("NAN_TRIPWIRE_"):
            os.environ.setdefault(name, value)

    torch.cuda.set_device(0)
    from generative_recommenders import common
    from generative_recommenders.ops.triton import triton_hstu_attention as attention
    from generative_recommenders.ops.triton import triton_hstu_preprocess_and_attention as preproc
    from generative_recommenders.ops.triton import triton_layer_norm as layer_norm

    # Known runtime state for step 55; replay_once restores captured state if present.
    common.set_static_max_seq_lens([4096])
    common.set_use_runtime_max_seq_len(False)
    attention_configs = attention._hstu_attn_bwd.configs
    if len(attention_configs) != 1 or attention_configs[0].kwargs.get("llvm_fn_attrs") != "amdgpu-num-vgpr=256":
        raise RuntimeError("preparation requires the pinned attention config capped at 256 VGPRs")
    kernel = layer_norm._weighted_layer_norm_bwd_dx
    preparation_config = force_config(triton, kernel, 8)
    report["source"] = dict(capture=str(args.capture.resolve()), bytes=args.capture.stat().st_size,
                          step=capture["report"].get("step"), metadata=capture["report"].get("metadata"),
                          captured_runtime_state=replay.get("runtime_state"))
    amp_enabled = replay.get("cuda_autocast_enabled", False)
    amp_dtype = replay.get("cuda_autocast_dtype", torch.bfloat16)
    report["preparation"] = dict(status="RUNNING", layer_norm_config=preparation_config,
                                 attention_config=attention_configs[0].kwargs)
    publish({"event": "preparation", **report["preparation"]})
    with torch.no_grad():
        inputs, references, comparison = prepare(torch, capture, preproc)
        report["preparation"].update(
            status="PASS" if comparison["passed"] else "FAIL", comparison=comparison,
            compiled_ln_kernels=compiled_metadata(kernel),
            compiled_attention_kernels=compiled_metadata(attention._hstu_attn_bwd),
        )
        publish({"event": "preparation", **report["preparation"]})
        if not comparison["passed"]:
            report["status"] = "PREPARATION_FAIL"
            return 1
        del replay, capture
        if not inputs["learnable"] or inputs["x"].ndim != 2 or inputs["x"].stride(-1) != 1 or inputs["dy"].stride(-1) != 1:
            raise ValueError("requires learnable layer norm with contiguous feature columns")
        tensors = dict(iter_tensors(inputs))
        finite_inputs = {name: all_chunks(torch, tensor, lambda chunk, _: torch.isfinite(chunk).all())
                         for name, tensor in tensors.items()}
        finite_values = torch.stack(list(finite_inputs.values())).cpu().tolist()
        if not all(finite_values):
            raise ValueError("nonfinite layer-norm input: " + str(dict(zip(finite_inputs, finite_values))))
        versions = {name: tensor._version for name, tensor in tensors.items()}
        snapshots = {name: tensor.clone() for name, tensor in tensors.items()} if args.check_input_contents else {}
        config = force_config(triton, kernel, args.block_n, args.max_vgpr)
        properties = torch.cuda.get_device_properties(0)
        report["configuration"] = dict(
            torch=str(torch.__version__), hip=torch.version.hip, triton=triton.__version__,
            arch=getattr(properties, "gcnArchName", properties.name),
            layer_norm_source=layer_norm.__file__,
            layer_norm_source_sha256=hashlib.sha256(Path(layer_norm.__file__).read_bytes()).hexdigest(),
            config=config, num_rows=inputs["x"].shape[0], feature_dim=inputs["x"].shape[1],
            BLOCK_D=inputs["BLOCK_D"], eps=inputs["eps"],
            autocast=dict(enabled=amp_enabled, dtype=str(amp_dtype)),
            static_max_seq_lens=common.STATIC_MAX_SEQ_LENS,
            use_runtime_max_seq_len=common.USE_RUNTIME_MAX_SEQ_LEN,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            input_layouts={name: dict(shape=list(t.shape), stride=list(t.stride()),
                                      storage_offset=t.storage_offset(), dtype=str(t.dtype))
                           for name, t in tensors.items()},
            env={name: os.environ.get(name) for name in ("HSTU_BWD_MAX_VGPR", "HSTU_BWD_BLOCK_N",
                 "AMDGCN_USE_BUFFER_OPS", "TRITON_ALLOW_PIPELINING", "TRITON_FULL_AUTOTUNE",
                 "AMD_SERIALIZE_KERNEL", "TRITON_HIP_USE_EXPERT_SCHEDULING",
                 "TRITON_HIP_USE_COEXEC_SCHEDULER", "AMDGCN_SCALARIZE_PACKED_FOPS",
                 "TRITON_HIP_USE_IN_THREAD_TRANSPOSE")},
        )
        publish({"event": "configuration", **report["configuration"]})
        report["compiled_kernels_scope"] = "all LN DX variants cached in this process, including preparation BN8"
        pending = []
        first_outputs = None
        started = time.monotonic()
        with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            for iteration in range(1, args.repeat + 1):
                outputs = layer_norm.triton_weighted_layer_norm_bwd(**inputs)
                changed = [name for name, tensor in tensors.items() if tensor._version != versions[name]]
                if changed:
                    raise RuntimeError(f"tracked input mutation at iteration {iteration}: {changed}")
                flags = output_flags(torch, outputs, references[0])
                for name, snapshot in snapshots.items():
                    flags[name + "_unchanged"] = all_chunks(
                        torch, tensors[name], lambda chunk, start, snapshot=snapshot:
                        (chunk.view(torch.uint8) == snapshot[start:start + len(chunk)].view(torch.uint8)).all()
                    )
                pending.extend((iteration, name, flag) for name, flag in flags.items())
                report["iterations_completed"] = iteration
                if iteration == 1:
                    first_outputs = outputs
                del outputs
                if iteration % args.check_every and iteration != args.repeat:
                    continue
                values = torch.stack([flag for _, _, flag in pending]).cpu().tolist()
                failures = [{"iteration": step, "check": name}
                            for (step, name, _), passed in zip(pending, values) if not passed]
                if first_outputs is not None:
                    # Diagnostic only for DW/DB: their reduction order can change.
                    report["first_iteration_comparison"] = compare_original_outputs(first_outputs, references, rtol=0, atol=0)
                    first_outputs = None
                    publish({"event": "first_iteration_comparison", **report["first_iteration_comparison"]})
                check = dict(checked_through_iteration=iteration, failures=failures,
                             elapsed_seconds=time.monotonic() - started)
                report["checks"].append(check)
                report["compiled_kernels"] = compiled_metadata(kernel)
                report["compiled_reduction_kernels"] = compiled_metadata(layer_norm._layer_norm_bwd_dwdb)
                pending.clear()
                report["status"] = "FAIL" if failures else ("PASS" if iteration == args.repeat else "RUNNING")
                publish({"event": "check", **check, "status": report["status"]})
                if failures:
                    return 1
    return 0


def main():
    args = arguments()
    report = dict(capture=str(args.capture), block_n=args.block_n, max_vgpr=args.max_vgpr,
                  repeat=args.repeat, check_every=args.check_every,
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
