"""Synchronous, bounded-input diagnostic for a saved HSTU backward replay.

Install after constructing the model and before its next forward. The default
target is layer 1; NAN_BACKWARD_TARGET is a parameter-name substring (``*`` means
all layers). NAN_BACKWARD_ABS_THRESHOLD defaults to 1e20.
NAN_BACKWARD_SAVE_ALL=1 also persists finite selected calls for comparisons.
NAN_BACKWARD_CHUNK_MIB defaults to 64 and
NAN_BACKWARD_MAX_CAPTURE_GIB to 32. The latter is a hard per-operation host-copy
limit, never a truncation limit.

Each selected operation owns pristine CPU input bytes before executing. Finite
calls release their copies immediately. The first nonfinite or extreme input or
output produces a trusted local .pt dump and raises BoundaryAnomalyError. Input-stage
failures stop before calling the operation and therefore have outputs=None.
Copies, scans, and file names do not consume Python, NumPy, or Torch RNG state.
The synchronization and extra memory traffic deliberately change scheduling.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import torch


class BoundaryProbeError(RuntimeError):
    """The requested diagnostic could not be performed completely."""


class BoundaryAnomalyError(BoundaryProbeError):
    def __init__(self, dump_path: Path, operation: str, layer: str, stage: str):
        self.dump_path = dump_path
        super().__init__(
            f"Nonfinite or extreme {stage} at {layer} / {operation}; completed dump: {dump_path}"
        )


# Retain the original integration name while extending the trigger to finite
# extremes. Both names denote the same exception type and expose dump_path.
BoundaryNonFiniteError = BoundaryAnomalyError


def _tensors(value: Any, path: str = ""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensors(item, f"{path}/{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{path}/{index}" if path else str(index))
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"Unsupported operation argument at {path}: {type(value)}")


def _synchronize(value: Any) -> None:
    devices = {str(tensor.device) for _, tensor in _tensors(value)
               if tensor.device.type == "cuda"}
    for device in sorted(devices):
        torch.cuda.synchronize(torch.device(device))


def _source_specs(value: Any) -> dict[str, Any]:
    return {name: {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                   "dtype": str(tensor.dtype), "source_device": str(tensor.device),
                   "storage_offset": tensor.storage_offset(),
                   "requires_grad": tensor.requires_grad}
            for name, tensor in _tensors(value)}


def _cpu_copy_tree(value: Any, chunk_bytes: int, max_bytes: int):
    """Copy each complete backing storage once, with no full GPU temporary.

    CPU views preserve strides, offsets, dtype, and alias relationships. Tensor
    requires_grad is recorded separately; the diagnostic copies need no graph.
    """
    storages: dict[tuple[str, int], tuple[Any, torch.device]] = {}
    for _, tensor in _tensors(value):
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise BoundaryProbeError("Boundary snapshots require strided, nonquantized tensors")
        if tensor.device.type not in ("cpu", "cuda"):
            raise BoundaryProbeError(f"Unsupported boundary tensor device: {tensor.device}")
        storage = tensor.untyped_storage()
        storages.setdefault((str(tensor.device), storage._cdata), (storage, tensor.device))
    total_bytes = sum(storage.nbytes() for storage, _ in storages.values())
    if total_bytes > max_bytes:
        raise BoundaryProbeError(
            f"Boundary input/output copy needs {total_bytes} bytes, exceeds remaining "
            f"capture limit {max_bytes}; raise NAN_BACKWARD_MAX_CAPTURE_GIB explicitly"
        )
    _synchronize(value)
    copies = {}
    for key, (storage, device) in storages.items():
        nbytes = storage.nbytes()
        source = torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (nbytes,), (1,))
        target = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
        for start in range(0, nbytes, chunk_bytes):
            target[start:start + chunk_bytes].copy_(source[start:start + chunk_bytes])
        copies[key] = target

    def rebuild(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            backing = copies[(str(item.device), storage._cdata)]
            result = torch.empty(0, dtype=item.dtype, device="cpu").set_(
                backing.untyped_storage(), item.storage_offset(), item.shape, item.stride())
            if item.is_conj():
                result = result.conj()
            if item.is_neg():
                result = torch._neg_view(result)
            return result
        if isinstance(item, dict):
            return {key: rebuild(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(rebuild(child) for child in item)
        if isinstance(item, list):
            return [rebuild(child) for child in item]
        return item

    return rebuild(value), total_bytes


def _logical_chunks(tensor: torch.Tensor, limit: int, origin=None):
    # Split logical slices before any contiguous operation. A large strided
    # tensor must not materialize a full contiguous copy merely for a scan.
    if origin is None:
        origin = (0,) * tensor.ndim
    if tensor.numel() <= limit:
        yield tensor, origin
        return
    dimension = next(index for index, size in enumerate(tensor.shape) if size > 1)
    other_elements = tensor.numel() // tensor.shape[dimension]
    width = max(1, limit // other_elements)
    for start in range(0, tensor.shape[dimension], width):
        part = tensor.narrow(dimension, start, min(width, tensor.shape[dimension] - start))
        next_origin = list(origin)
        next_origin[dimension] += start
        yield from _logical_chunks(part, limit, tuple(next_origin))


def _edge_true(mask: torch.Tensor, *, last=False) -> int:
    """Locate one set bit without an unbounded nonzero result."""
    if last:
        for end in range(mask.numel(), 0, -4096):
            start = max(0, end - 4096)
            indices = torch.nonzero(mask[start:end]).flatten()
            if indices.numel():
                return start + int(indices[-1].item())
    else:
        for start in range(0, mask.numel(), 4096):
            indices = torch.nonzero(mask[start:start + 4096]).flatten()
            if indices.numel():
                return start + int(indices[0].item())
    raise ValueError("Expected at least one anomalous element")


def _location_samples(chunk, bad_mask, finite_mask, origin, full_shape, maximum):
    samples = []
    flattened = bad_mask.reshape(-1)
    for start in range(0, flattened.numel(), 4096):
        indices = torch.nonzero(flattened[start:start + 4096]).flatten()
        for index in indices[:maximum - len(samples)].tolist():
            flat = start + index
            coordinate = []
            for size in reversed(chunk.shape):
                coordinate.append(flat % size)
                flat //= size
            coordinate.reverse()
            absolute = [index + offset for index, offset in zip(coordinate, origin)]
            flat_absolute = 0
            for index, size in zip(absolute, full_shape):
                flat_absolute = flat_absolute * size + index
            item = tuple(coordinate)
            samples.append({
                "index": absolute, "flat_index": flat_absolute,
                "row": absolute[0] if absolute else None,
                "kind": "extreme" if bool(finite_mask[item].item()) else "nonfinite",
                # Strings represent NaN/Inf without invalid JSON numeric values.
                "value": str(chunk[item].item()),
            })
        if len(samples) >= maximum:
            break
    return samples


def _summaries(value: Any, chunk_bytes: int, specs: dict[str, Any],
               abs_threshold: float = 1e20):
    result = []
    for name, tensor in _tensors(value):
        count = extreme_count = 0
        max_abs = None
        row_min = row_max = None
        locations = []
        if tensor.is_floating_point() or tensor.is_complex():
            # All additional masks/reductions remain bounded by this chunk.
            limit = max(1, chunk_bytes // max(8, tensor.element_size()))
            for chunk, origin in _logical_chunks(tensor, limit):
                if not chunk.numel():
                    continue
                finite = torch.isfinite(chunk)
                magnitude = torch.abs(chunk)
                # A bounded float64 conversion makes threshold comparisons
                # independent of half/bfloat scalar-rounding conventions.
                comparable = magnitude.to(torch.float64)
                extreme = finite & (comparable > abs_threshold)
                nonfinite = ~finite
                n_bad = int(nonfinite.sum().item())
                n_extreme = int(extreme.sum().item())
                count += n_bad
                extreme_count += n_extreme
                finite_magnitude = finite & torch.isfinite(comparable)
                if bool(finite_magnitude.any().item()):
                    largest = torch.where(finite_magnitude, comparable, 0.0).max().item()
                    max_abs = largest if max_abs is None else max(max_abs, largest)
                if n_bad or n_extreme:
                    bad = nonfinite | extreme
                    if tensor.ndim:
                        rows = bad.reshape(chunk.shape[0], -1).any(dim=1)
                        first = origin[0] + _edge_true(rows)
                        last = origin[0] + _edge_true(rows, last=True)
                        row_min = first if row_min is None else min(row_min, first)
                        row_max = last if row_max is None else max(row_max, last)
                    if len(locations) < 16:
                        locations.extend(_location_samples(
                            chunk, bad, finite, origin, tensor.shape, 16 - len(locations)))
        result.append({"name": name, **specs[name], "numel": tensor.numel(),
                       "finite": count == 0, "nonfinite_count": count,
                       "max_abs_finite": max_abs, "extreme_count": extreme_count,
                       "abs_threshold": abs_threshold,
                       "affected_row_range": None if row_min is None else [row_min, row_max],
                       "sample_rows": sorted({item["row"] for item in locations
                                              if item["row"] is not None}),
                       "sample_locations": locations})
    return result


def _pointer(tensor: torch.Tensor):
    return (str(tensor.device), tensor.data_ptr(), str(tensor.dtype),
            tuple(tensor.shape), tuple(tensor.stride()))


class BackwardBoundaryProbe:
    _OPERATIONS = {
        "triton_addmm_bwd": ("w", ("d_normed_x", "d_uvqk_weight", "d_uvqk_bias")),
        "triton_weighted_layer_norm_bwd":
            ("weight", ("d_x", "d_norm_weight", "d_norm_bias")),
    }

    def __init__(self, model: torch.nn.Module, directory: str | Path,
                 *, preprocess_module=None, compute_module=None):
        # Injectable modules support CPU mock tests without importing GPU ops.
        self.preprocess = preprocess_module or importlib.import_module(
            "generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention")
        self.compute = compute_module or importlib.import_module(
            "generative_recommenders.ops.hstu_compute")
        from nan_replay_state import _walk_modules

        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.target = os.environ.get("NAN_BACKWARD_TARGET", "_stu_layers.1.")
        self.abs_threshold = float(os.environ.get("NAN_BACKWARD_ABS_THRESHOLD", "1e20"))
        self.save_all = os.environ.get("NAN_BACKWARD_SAVE_ALL") == "1"
        self.chunk_bytes = int(os.environ.get("NAN_BACKWARD_CHUNK_MIB", "64")) << 20
        self.max_bytes = int(os.environ.get("NAN_BACKWARD_MAX_CAPTURE_GIB", "32")) << 30
        if self.chunk_bytes <= 0 or self.max_bytes <= 0:
            raise ValueError("Boundary copy chunk and capture limits must be positive")
        if not math.isfinite(self.abs_threshold) or self.abs_threshold <= 0:
            raise ValueError("NAN_BACKWARD_ABS_THRESHOLD must be finite and positive")
        self.session = f"{os.getpid()}-{time.time_ns()}"
        self.attempt = None
        self.call = 0
        self.failed = False
        self.closed = False
        self._local = threading.local()
        self._pointers = {}
        self._handles = []
        self._patches = []
        self._layers = {
            path.replace("/", "."): module
            for path, module in _walk_modules(model).items()
            if hasattr(module, "_input_norm_weight") and hasattr(module, "_uvqk_weight")
        }
        if not self._layers:
            raise BoundaryProbeError("No STU parameter owners found for layer attribution")
        if self.target not in ("*", "all") and not any(
                self.target in name + "." for name in self._layers):
            raise BoundaryProbeError(f"No model layer matches NAN_BACKWARD_TARGET={self.target!r}")
        self._events = (self.directory / "boundaries.jsonl").open("a", buffering=1)
        try:
            self._install_hooks()
            self._install_wrappers()
            self._emit({"event": "installed", "target": self.target,
                        "abs_threshold": self.abs_threshold,
                        "save_all": self.save_all,
                        "layers": list(self._layers), "chunk_bytes": self.chunk_bytes,
                        "max_capture_bytes": self.max_bytes})
        except BaseException:
            self.close()
            raise

    def _emit(self, event: dict[str, Any]) -> None:
        payload = {"session": self.session, "attempt": self.attempt,
                   "unix_time": time.time(), **event}
        self._events.write(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
        self._events.flush()

    def _install_hooks(self):
        for name, module in self._layers.items():
            def before(_module, _args, _kwargs, layer=name):
                stack = getattr(self._local, "layers", [])
                stack.append(layer)
                self._local.layers = stack

            def after(_module, _args, _kwargs, _output, layer=name):
                stack = getattr(self._local, "layers", [])
                if not stack or stack[-1] != layer:
                    raise BoundaryProbeError("Unbalanced STU forward attribution hooks")
                stack.pop()

            self._handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self._handles.append(module.register_forward_hook(
                after, with_kwargs=True, always_call=True))

    def _patch(self, module, name, replacement):
        original = getattr(module, name)
        if getattr(original, "_nan_backward_probe", None) is not None:
            raise BoundaryProbeError(f"A backward boundary probe already wraps {name}")
        replacement._nan_backward_probe = self
        self._patches.append((module, name, original, replacement))
        setattr(module, name, replacement)

    def _install_wrappers(self):
        original = self.compute.triton_hstu_preprocess_and_attention
        signature = inspect.signature(original)

        @functools.wraps(original)
        def forward(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            stack = getattr(self._local, "layers", [])
            if not stack:
                raise BoundaryProbeError("Preprocess forward executed outside a mapped STU layer")
            layer = stack[-1]
            for argument, parameter in (
                ("norm_weight", "_input_norm_weight"),
                ("norm_bias", "_input_norm_bias"),
                ("uvqk_weight", "_uvqk_weight"),
                ("uvqk_bias", "_uvqk_beta"),
            ):
                tensor = bound.arguments.get(argument)
                if isinstance(tensor, torch.Tensor):
                    self._pointers[_pointer(tensor)] = {
                        "layer": layer, "parameter": layer + "." + parameter,
                        "mapping": "forward_argument",
                    }
            return original(*args, **kwargs)

        self._patch(self.compute, "triton_hstu_preprocess_and_attention", forward)
        for operation in self._OPERATIONS:
            original_op = getattr(self.preprocess, operation)
            signature_op = inspect.signature(original_op)

            def make_wrapper(op, function, sig):
                @functools.wraps(function)
                def backward(*args, **kwargs):
                    return self._run(op, function, sig, args, kwargs)
                return backward

            self._patch(self.preprocess, operation,
                        make_wrapper(operation, original_op, signature_op))

    def set_attempt(self, mode: str, repeat: int, step: int) -> None:
        if self.closed or self.failed:
            raise BoundaryProbeError("Cannot reuse a closed or failed boundary probe")
        self.attempt = {"mode": mode, "repeat": repeat, "step": step}
        self.call = 0
        self._pointers.clear()
        for layer, module in self._layers.items():
            for name in ("_input_norm_weight", "_input_norm_bias", "_uvqk_weight", "_uvqk_beta"):
                tensor = getattr(module, name, None)
                if isinstance(tensor, torch.Tensor):
                    self._pointers[_pointer(tensor)] = {
                        "layer": layer, "parameter": layer + "." + name,
                        "mapping": "model_parameter",
                    }
        self._emit({"event": "attempt_start"})

    def _run(self, operation, function, signature, args, kwargs):
        if self.attempt is None:
            raise BoundaryProbeError("Call set_attempt before the replay forward")
        if self.failed:
            raise BoundaryProbeError("Replay continued after the first nonfinite boundary")
        bound = signature.bind(*args, **kwargs)
        weight_argument, output_names = self._OPERATIONS[operation]
        weight = bound.arguments.get(weight_argument)
        owner = self._pointers.get(_pointer(weight)) if isinstance(weight, torch.Tensor) else None
        if owner is None:
            raise BoundaryProbeError(f"Cannot attribute {operation} {weight_argument} pointer to a model layer")
        self.call += 1
        event = {"call": self.call, "operation": operation, **owner}
        if self.target not in ("*", "all") and self.target not in owner["parameter"]:
            self._emit({"event": "skipped", **event, "reason": "target_filter",
                        "inputs": _source_specs(bound.arguments)})
            return function(*args, **kwargs)

        start = time.monotonic()
        pre, input_bytes = _cpu_copy_tree(
            {"args": args, "kwargs": kwargs}, self.chunk_bytes, self.max_bytes)
        saved_bound = signature.bind(*pre["args"], **pre["kwargs"])
        input_specs = _source_specs(bound.arguments)
        input_summary = _summaries(saved_bound.arguments, self.chunk_bytes, input_specs,
                                   self.abs_threshold)
        bad_inputs = [item["name"] for item in input_summary if not item["finite"]]
        extreme_inputs = [item["name"] for item in input_summary if item["extreme_count"]]
        event.update(inputs=input_summary, input_bytes=input_bytes)
        self._emit({"event": "before", **event})
        if bad_inputs or extreme_inputs:
            self._freeze(operation, owner["layer"], "input", pre, None,
                         {**event, "bad_inputs": bad_inputs, "bad_outputs": [],
                          "extreme_inputs": extreme_inputs, "extreme_outputs": [],
                          "operation_executed": False, "seconds": time.monotonic() - start})

        output = function(*args, **kwargs)
        if not isinstance(output, (tuple, list)) or len(output) != len(output_names):
            raise BoundaryProbeError(f"Unexpected {operation} return structure")
        copied_output, output_bytes = _cpu_copy_tree(
            output, self.chunk_bytes, self.max_bytes - input_bytes)
        named_output = dict(zip(output_names, output))
        named_copied = dict(zip(output_names, copied_output))
        output_summary = _summaries(named_copied, self.chunk_bytes, _source_specs(named_output),
                                    self.abs_threshold)
        bad_outputs = [item["name"] for item in output_summary if not item["finite"]]
        extreme_outputs = [item["name"] for item in output_summary if item["extreme_count"]]
        event.update(outputs=output_summary, output_bytes=output_bytes,
                     seconds=time.monotonic() - start)
        self._emit({"event": "after", **event})
        if bad_outputs or extreme_outputs:
            self._freeze(operation, owner["layer"], "output", pre, copied_output,
                         {**event, "bad_inputs": [], "bad_outputs": bad_outputs,
                          "extreme_inputs": [], "extreme_outputs": extreme_outputs,
                          "operation_executed": True})
        if self.save_all:
            self._save(operation, owner["layer"], "finite", pre, copied_output,
                       {**event, "bad_inputs": [], "bad_outputs": [],
                        "extreme_inputs": [], "extreme_outputs": [],
                        "operation_executed": True})
        # All CPU copies are local and fall out of scope here. No rolling cache
        # retains operation tensors, even across calls in the same backward.
        return output

    def _freeze(self, operation, layer, stage, inputs, outputs, event):
        self.failed = True
        target = self._save(operation, layer, stage, inputs, outputs, event)
        raise BoundaryAnomalyError(target, operation, layer, stage)

    def _save(self, operation, layer, stage, inputs, outputs, event):
        frame = self.attempt
        identity = f"{frame['mode']}-r{frame['repeat']}-s{frame['step']}"
        filename = f"boundary-{self.session}-{identity}-{self.call:04d}-{operation}-{stage}.pt"
        target = self.directory / filename
        temporary = target.with_suffix(".pt.tmp")
        payload = {
            "format_version": 1, "session": self.session, "attempt": self.attempt,
            "operation": operation, "layer": layer, "stage": stage,
            "args": inputs["args"], "kwargs": inputs["kwargs"], "outputs": outputs,
            "event": event,
            "note": "Inputs are independent pre-call CPU storage copies; load only as a trusted local artifact.",
        }
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._emit({"event": "dump_complete", **event, "stage": stage, "dump": str(target)})
        os.fsync(self._events.fileno())
        print(f"[nan-boundary] saved {stage} boundary: {layer} / {operation}: {target}", flush=True)
        return target

    def close(self) -> None:
        if self.closed:
            return
        for module, name, original, replacement in reversed(self._patches):
            if getattr(module, name) is replacement:
                setattr(module, name, original)
        for handle in self._handles:
            handle.remove()
        self._patches.clear()
        self._handles.clear()
        self._pointers.clear()
        self._events.close()
        self.closed = True


def install(model: torch.nn.Module, directory: str | Path) -> BackwardBoundaryProbe:
    return BackwardBoundaryProbe(model, directory)
