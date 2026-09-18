#!/usr/bin/env python3
"""Replay a captured custom autograd boundary without the dataset or model.

Example (inside the matching training image, with the repository on PYTHONPATH):
  AMDGCN_USE_BUFFER_OPS=0 python scripts/replay_nan_tripwire.py capture.pt --repeat 5
  AMDGCN_USE_BUFFER_OPS=0 python scripts/replay_nan_tripwire.py capture.pt --compare-original
  AMDGCN_USE_BUFFER_OPS=0 python scripts/replay_nan_tripwire.py capture.pt --stress --repeat 1000

Stress mode uploads once, reuses inputs, restores RNG and context for each call,
and checks output finiteness every --check-every iterations (default 10). It
rejects tracked input mutations; untracked kernel writes remain undetectable.
Normal mode can also compare every returned tensor against the captured outputs
with --compare-original. Finite mismatches then fail the replay. Float/complex
tensors use allclose semantics (default rtol=0.01, atol=1e-12); --rtol 0 --atol 0
requires exact values. Container structure, shapes, dtypes and nonfloat values
must match. NaNs are not treated as equal. Comparison is unavailable in stress
mode because that mode avoids decoding and retaining captured outputs.

Backward captures replay the original backward directly from saved tensors and
context, including real dropout masks/seeds. Forward captures restore RNG state.
Tensor storage aliases, shapes, strides and storage offsets are preserved. Use
--restore-env to restore captured Triton/HSTU environment before importing ops.
Inputs known to have changed before capture are rejected unless explicitly
allowed; such a replay cannot validate the originally observed failure or a fix.
Only load captures produced locally by the trusted diagnostic: torch.load uses
Python pickle because context attributes can include torch dtype objects.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def iter_tensors(value: Any, path: str = ""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from iter_tensors(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from iter_tensors(child, f"{path}[{index}]")


def decode_payload(
    capture: dict[str, Any], device: str, *, replay_only: bool = False
) -> Any:
    buffers: dict[tuple[int, str], torch.Tensor] = {}

    def decode(value: Any) -> Any:
        if isinstance(value, dict) and "__tripwire_tensor__" in value:
            index = value["__tripwire_tensor__"]
            # Host scalars/RNG states must stay on CPU; CUDA maps to requested GPU.
            target = device if value["device"].startswith("cuda") else "cpu"
            key = (index, target)
            if key not in buffers:
                buffers[key] = capture["storages"][index].to(target).clone()
            dtype = getattr(torch, value["dtype"].removeprefix("torch."))
            tensor = torch.empty(0, dtype=dtype, device=target)
            return tensor.set_(
                buffers[key].untyped_storage(),
                value["storage_offset"],
                tuple(value["shape"]),
                tuple(value["stride"]),
            )
        if isinstance(value, dict):
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return tuple(decode(child) for child in value)
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value

    selected = capture["payload"]
    if replay_only:
        # Stress needs the operation inputs, not the potentially huge original
        # outputs saved for detailed one-shot comparison.
        selected = {"replay": selected["replay"]}
    return decode(selected)


class ReplayContext(SimpleNamespace):
    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors: torch.Tensor) -> None:
        pass

    def mark_dirty(self, *tensors: torch.Tensor) -> None:
        pass

    def set_materialize_grads(self, value: bool) -> None:
        pass


def summarize(value: Any) -> list[dict[str, Any]]:
    summaries = []
    for name, tensor in iter_tensors(value):
        info = {
            "name": name,
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }
        if (tensor.is_floating_point() or tensor.is_complex()) and tensor.numel():
            finite = torch.isfinite(tensor)
            info["nonfinite"] = int((~finite).sum().item())
            info["max_abs"] = float(torch.where(finite, tensor.abs(), 0).max().item())
        summaries.append(info)
    return summaries


def compare_original_outputs(
    actual: Any, original: Any, *, rtol: float = 0.01, atol: float = 1e-12
) -> dict[str, Any]:
    """Compare nested outputs without broadcasting or large whole-tensor copies.

    Chunked isclose(...).all() has torch.allclose semantics, including unequal
    NaNs. Float differences are evaluated in float64 to avoid bf16 subtraction
    rounding; nonfloating tensors use exact equality and omit max_abs_diff.
    """
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise ValueError("rtol and atol must be finite and nonnegative")
    tensors = []
    errors = []

    def compare(left: Any, right: Any, name: str) -> None:
        where = name or "<root>"
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            info = {"name": name, "shape": list(left.shape), "dtype": str(left.dtype),
                    "original_shape": list(right.shape), "original_dtype": str(right.dtype),
                    "numel": left.numel(), "allclose": False,
                    "mismatched_elements": None, "max_abs_diff": None}
            tensors.append(info)
            for field in ("shape", "dtype", "layout", "device"):
                if getattr(left, field) != getattr(right, field):
                    errors.append(f"{where}: {field} differs ({getattr(left, field)} vs {getattr(right, field)})")
            if any(getattr(left, field) != getattr(right, field) for field in ("shape", "dtype", "layout", "device")):
                return
            if left.layout != torch.strided:
                errors.append(f"{where}: comparison requires strided tensors")
                return
            floating = left.is_floating_point() or left.is_complex()
            info["comparison"] = "allclose" if floating else "exact"
            count = torch.zeros((), dtype=torch.int64, device=left.device)
            maximum = torch.zeros((), dtype=torch.float64, device=left.device)
            # Slice rows before conversion, including for noncontiguous tensors.
            left, right = (t.reshape(1) if t.ndim == 0 else t for t in (left, right))
            row_elements = math.prod(left.shape[1:])
            chunk_rows = max(1, 1024 * 1024 // max(1, row_elements))
            for start in range(0, left.shape[0], chunk_rows):
                a, b = left[start:start + chunk_rows], right[start:start + chunk_rows]
                if not a.numel():
                    continue
                close = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False) if floating else a == b
                count += (~close).sum()
                if floating:
                    dtype = torch.complex128 if a.is_complex() else torch.float64
                    delta = (a.to(dtype) - b.to(dtype)).abs()
                    delta = torch.where(a == b, 0.0, delta)  # Equal signed infinities have zero difference.
                    delta.nan_to_num_(nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
                    maximum = torch.maximum(maximum, delta.max())
            info["mismatched_elements"] = int(count.item())
            info["allclose"] = info["mismatched_elements"] == 0
            if floating:
                difference = float(maximum.item())
                info["max_abs_diff"] = difference if math.isfinite(difference) else "inf"
            return
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            errors.append(f"{where}: tensor/non-tensor mismatch")
            return
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                errors.append(f"{where}: dictionary keys differ")
            for key in right:
                if key in left:
                    compare(left[key], right[key], f"{name}.{key}" if name else str(key))
            return
        if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
            if isinstance(left, tuple) != isinstance(right, tuple):
                errors.append(f"{where}: tuple/list mismatch")
            if len(left) != len(right):
                errors.append(f"{where}: sequence lengths differ ({len(left)} vs {len(right)})")
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{name}[{index}]")
            return
        if type(left) is not type(right) or left != right:
            errors.append(f"{where}: non-tensor values or types differ")

    with torch.no_grad():
        compare(actual, original, "")
    return {"passed": not errors and all(info["allclose"] for info in tensors),
            "structure_errors": errors, "tensors": tensors}


def changed_capture_inputs(capture: dict[str, Any]) -> list[str]:
    """Find tracked mutations between observing inputs and copying their storage.

    Version agreement cannot exclude untracked custom-kernel writes or hardware
    corruption. Inference tensors have no version counter, so remain unknown.
    """
    report = capture["report"]
    selected_sequence = report.get("selected_sequence", report.get("first_bad_sequence"))
    event = next(
        entry for entry in report["events"]
        if entry["sequence"] == selected_sequence
    )
    encoded_inputs = {}

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict) and "__tripwire_tensor__" in value:
            encoded_inputs[path] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(capture["payload"]["inputs"])
    changed = []
    for info in event["inputs"]:
        before = info.get("version")
        # The payload fallback also checks captures made before the JSON report
        # gained version_at_capture/version_changed fields.
        after = info.get("version_at_capture", encoded_inputs.get(info["name"], {}).get("version"))
        if before is not None and after is not None and before != after:
            changed.append(f"{info['name']} (version {before} -> {after})")
    return changed


def replay_once(
    capture: dict[str, Any], device: str, *, allow_mutated_inputs: bool = False,
    snapshot_original_outputs: bool = False,
) -> tuple[Any, Any]:
    changed = changed_capture_inputs(capture)
    if changed:
        message = (
            "Inputs changed between observation and capture: " + ", ".join(changed)
            + ". This replay cannot validate the originally observed failure or a fix."
        )
        if not allow_mutated_inputs:
            raise ValueError(message + " Use --allow-mutated-inputs only for exploratory replay.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    payload = decode_payload(capture, device)
    replay = payload["replay"]
    if replay is None:
        raise ValueError("This is a phase observation; no custom operation was captured")
    if snapshot_original_outputs:
        # A captured output can alias an input to an in-place operation. Freeze
        # those references before replay so it cannot change its own expected
        # result. Independent original-output storages need no additional copy.
        input_storages = {
            (str(tensor.device), tensor.untyped_storage().data_ptr())
            for _, tensor in iter_tensors(replay)
        }

        def snapshot(value):
            if isinstance(value, torch.Tensor):
                key = (str(value.device), value.untyped_storage().data_ptr())
                return value.detach().clone() if key in input_storages else value
            if isinstance(value, dict):
                return {key: snapshot(child) for key, child in value.items()}
            if isinstance(value, tuple):
                return tuple(snapshot(child) for child in value)
            if isinstance(value, list):
                return [snapshot(child) for child in value]
            return value

        payload["outputs"] = snapshot(payload["outputs"])
    return _execute_replay(replay, device), payload


def _fresh_containers(value: Any) -> Any:
    """Reset mutable context containers while preserving the decoded tensors."""
    if isinstance(value, dict):
        return {key: _fresh_containers(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_fresh_containers(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_fresh_containers(child) for child in value)
    return value


def _restore_runtime_state(state: dict[str, Any] | None) -> None:
    if state is None:
        warnings.warn(
            "Capture is missing runtime tuning state; keeping current common and "
            "matmul settings. Older captures cannot establish these original settings.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    common_state = state.get("common")
    if common_state is not None:
        common = importlib.import_module("generative_recommenders.common")
        common.set_static_max_seq_lens(common_state["STATIC_MAX_SEQ_LENS"])
        common.set_use_runtime_max_seq_len(common_state["USE_RUNTIME_MAX_SEQ_LEN"])
        if "BACKEND_ALLOW_TF32" in common_state:
            common.BACKEND_ALLOW_TF32 = common_state["BACKEND_ALLOW_TF32"]
    if "float32_matmul_precision" in state:
        torch.set_float32_matmul_precision(state["float32_matmul_precision"])
    for name, value in state.get("cuda_matmul", {}).items():
        if hasattr(torch.backends.cuda.matmul, name):
            setattr(torch.backends.cuda.matmul, name, value)
        else:
            warnings.warn(
                f"This torch version cannot restore cuda.matmul.{name}",
                RuntimeWarning,
                stacklevel=2,
            )


def _execute_replay(replay: dict[str, Any], device: str, operation=None) -> Any:
    _restore_runtime_state(replay.get("runtime_state"))
    if operation is None:
        module = importlib.import_module(replay["module"])
        operation = getattr(getattr(module, replay["class"]), replay["direction"])
    ctx = ReplayContext(**_fresh_containers(replay["ctx"]))
    ctx.saved_tensors = replay["saved_tensors"]
    ctx.needs_input_grad = replay["needs_input_grad"]
    torch.set_rng_state(replay["torch_rng_state"])
    random.setstate(replay["python_rng_state"])
    if "cuda_rng_state" in replay and device.startswith("cuda"):
        torch.cuda.set_rng_state(replay["cuda_rng_state"], device=device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        enabled=replay.get("cuda_autocast_enabled", False) and device.startswith("cuda"),
        dtype=replay.get("cuda_autocast_dtype", torch.bfloat16),
    ):
        return operation(ctx, *replay["args"], **replay["kwargs"])


def stress_replay(
    capture: dict[str, Any],
    device: str,
    *,
    repeat: int,
    check_every: int = 10,
    on_check: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Reuse one decoded payload and batch scalar checks without retaining outputs.

    Versions are checked on the actual arguments, saved tensors and context
    tensors passed to the operation. Decoded aliases can have independent
    version counters, so inspecting only payload['inputs'] would miss mutations.
    """
    if repeat < 1 or check_every < 1:
        raise ValueError("repeat and check_every must be positive")
    changed = changed_capture_inputs(capture)
    if changed:
        raise ValueError(
            "Stress replay rejects inputs changed between observation and capture: "
            + ", ".join(changed)
        )
    payload = decode_payload(capture, device, replay_only=True)
    replay = payload["replay"]
    del payload
    if replay is None:
        raise ValueError("This is a phase observation; no custom operation was captured")
    module = importlib.import_module(replay["module"])
    cls = getattr(module, replay["class"])
    operation = getattr(cls, replay["direction"])
    versions = [
        (name, tensor, None if tensor.is_inference() else tensor._version)
        for name, tensor in iter_tensors({
            key: replay[key] for key in ("args", "kwargs", "saved_tensors", "ctx")
        })
    ]
    report = {
        "mode": "stress",
        "operation": f"{replay['module']}.{replay['class']}.{replay['direction']}",
        "repeat_requested": repeat,
        "check_every": check_every,
        "iterations_completed": 0,
        "first_nonfinite_iteration": None,
        "checks": [],
    }
    # Only scalar flags survive each iteration; no full output tensor is kept.
    pending: list[tuple[int, str, torch.Tensor]] = []
    start = time.monotonic()
    for iteration in range(1, repeat + 1):
        output = _execute_replay(replay, device, operation)
        mutations = [
            f"{name} (version {before} -> {tensor._version})"
            for name, tensor, before in versions
            if before is not None and tensor._version != before
        ]
        if mutations:
            raise ValueError(
                f"Stress replay input mutated at iteration {iteration}: "
                + ", ".join(mutations)
                + ". Refusing cumulative replay on changed inputs."
            )
        checked_outputs = 0
        with torch.no_grad():
            for name, tensor in iter_tensors(output):
                if (tensor.is_floating_point() or tensor.is_complex()) and tensor.numel():
                    pending.append((iteration, name, torch.isfinite(tensor).all()))
                    checked_outputs += 1
        if not checked_outputs:
            raise RuntimeError("Replay returned no nonempty floating tensors to check")
        # The loop variable otherwise retains the last full output until next call.
        del output, tensor
        report["iterations_completed"] = iteration
        if iteration % check_every and iteration != repeat:
            continue
        by_device: dict[torch.device, list[tuple[int, str, torch.Tensor]]] = {}
        for item in pending:
            by_device.setdefault(item[2].device, []).append(item)
        failures = []
        for entries in by_device.values():
            finite_flags = torch.stack([item[2] for item in entries]).cpu().tolist()
            for (observed_iteration, name, _), finite in zip(entries, finite_flags):
                if not finite:
                    failures.append({"iteration": observed_iteration, "output": name})
        failures.sort(key=lambda item: item["iteration"])
        entry = {
            "checked_through_iteration": iteration,
            "outputs_checked": len(pending),
            "failures": failures,
            "elapsed_seconds": time.monotonic() - start,
        }
        report["checks"].append(entry)
        pending.clear()
        by_device.clear()
        if on_check is not None:
            on_check(entry)
        if failures:
            report["first_nonfinite_iteration"] = failures[0]["iteration"]
            break
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--restore-env", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expect-nonfinite", action="store_true")
    parser.add_argument("--allow-mutated-inputs", action="store_true")
    parser.add_argument("--compare-original", action="store_true",
                        help="fail on output structure/type differences or values outside allclose tolerances")
    parser.add_argument("--rtol", type=float, default=0.01, help="comparison relative tolerance (default: 0.01)")
    parser.add_argument("--atol", type=float, default=1e-12, help="comparison absolute tolerance (default: 1e-12)")
    parser.add_argument("--stress", action="store_true", help="reuse one decoded payload and batch finite checks")
    parser.add_argument("--check-every", type=int, default=10, help="stress-mode finite-check interval (default: 10)")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.check_every < 1:
        parser.error("--check-every must be positive")
    if args.stress and args.allow_mutated_inputs:
        parser.error("--stress does not permit --allow-mutated-inputs")
    if args.stress and args.compare_original:
        parser.error("--stress does not permit --compare-original")
    if not all(math.isfinite(value) and value >= 0 for value in (args.rtol, args.atol)):
        parser.error("--rtol and --atol must be finite and nonnegative")
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if capture.get("format_version") != 1:
        raise ValueError("Unsupported capture format")
    if args.restore_env:
        for name, value in capture["report"]["environment"].items():
            if not name.startswith("NAN_TRIPWIRE_"):
                os.environ[name] = value
    report = {
        "capture": str(args.capture),
        "torch_version": torch.__version__,
        "changed_inputs": changed_capture_inputs(capture),
        "allow_mutated_inputs": args.allow_mutated_inputs,
        "trials": [],
    }
    seen_nonfinite = False
    comparisons_passed = True
    if args.compare_original:
        report.update(compare_original=True, rtol=args.rtol, atol=args.atol)
    if args.stress:
        report.update(stress_replay(
            capture, args.device, repeat=args.repeat, check_every=args.check_every,
            on_check=lambda entry: print(json.dumps(entry), flush=True),
        ))
        seen_nonfinite = report["first_nonfinite_iteration"] is not None
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        return 0 if seen_nonfinite == args.expect_nonfinite else 1
    for trial in range(args.repeat):
        output, payload = replay_once(
            capture, args.device, allow_mutated_inputs=args.allow_mutated_inputs,
            snapshot_original_outputs=args.compare_original,
        )
        output_summary = summarize(output)
        if not output_summary:
            raise RuntimeError("Replay returned no tensors; cannot claim a finite result")
        nonfinite = sum(item.get("nonfinite", 0) for item in output_summary)
        seen_nonfinite |= nonfinite > 0
        entry = {
            "trial": trial,
            "operation": f"{payload['replay']['module']}.{payload['replay']['class']}.{payload['replay']['direction']}",
            "inputs": summarize(payload["inputs"]),
            "outputs": output_summary,
            "original_outputs": summarize(payload["outputs"]),
            "nonfinite": nonfinite,
        }
        if args.compare_original:
            entry["comparison"] = compare_original_outputs(output, payload["outputs"], rtol=args.rtol, atol=args.atol)
            comparisons_passed &= entry["comparison"]["passed"]
            report["comparisons_passed"] = comparisons_passed
        report["trials"].append(entry)
        print(json.dumps(entry), flush=True)
        del output, payload
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    # By default any replayed non-finite is a failure; the flag inverts the
    # expected result for confirming a captured failure before testing a fix.
    return 0 if seen_nonfinite == args.expect_nonfinite and comparisons_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
