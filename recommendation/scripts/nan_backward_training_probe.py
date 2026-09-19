"""Opt-in bounded training monitor over the existing backward boundaries.

``pristine`` retains the original probe's pre-call CPU snapshots.
``monitor_on_anomaly`` performs bounded device min/max reductions and copies
complete CPU storages only after a scalar check detects an anomaly. Its output
fault snapshots cannot prove input bytes were unchanged during execution.
Neither mode changes a numerical operation or consumes random numbers. Both
change timing, synchronization, and allocation behavior.
"""

from __future__ import annotations

import math
import os
import time

import torch

from nan_backward_boundaries import (
    BackwardBoundaryProbe,
    BoundaryProbeError,
    _cpu_copy_tree,
    _execution_controls,
    _logical_chunks,
    _pointer,
    _source_specs,
    _summaries,
    _synchronize,
    _tensors,
)


DEFAULT_OPERATIONS = (
    "hstu_output_grad_mm",
    "triton_layer_norm_mul_dropout_bwd",
)
MONITOR_FORMAT = "training_backward_monitor_v1"


def _json_number(value):
    return value if math.isfinite(value) else str(value)


def monitor_summaries(value, *, chunk_bytes, abs_threshold):
    """Scan every real floating element, with one scalar readback per tensor.

    Native-dtype min/max is exact for the supported real input dtypes. Only
    endpoints are promoted to float64; comparing a threshold in half/bfloat
    precision would otherwise round the threshold. NaN/Inf endpoints cause a
    fault even if other elements also contain large finite values. Full counts
    and locations are obtained from the CPU snapshot after a fault.
    """
    if chunk_bytes <= 0 or not math.isfinite(abs_threshold) or abs_threshold <= 0:
        raise ValueError("Monitor chunk bytes and finite threshold must be positive")
    tensors = list(_tensors(value))
    for name, tensor in tensors:
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise BoundaryProbeError(f"Monitor requires strided nonquantized tensor: {name}")
        if tensor.device.type not in ("cpu", "cuda"):
            raise BoundaryProbeError(f"Unsupported monitor device at {name}: {tensor.device}")
        if tensor.is_complex():
            raise BoundaryProbeError(f"Training monitor does not support complex tensor: {name}")
    # Include work on any device stream before establishing the observation.
    # Scalar CPU copies below also wait for the reductions themselves.
    _synchronize(value)
    specs = _source_specs(value)
    results = []
    with torch.no_grad():
        for name, tensor in tensors:
            entry = {
                "name": name, **specs[name], "numel": tensor.numel(),
                "observation": "device_logical_minmax", "abs_threshold": abs_threshold,
                "scanned": tensor.is_floating_point(), "finite": True,
                "extreme": False, "min": None, "max": None,
                "max_abs_finite": None,
            }
            if tensor.is_floating_point() and tensor.numel():
                minimum = maximum = None
                # Autocast is scoped to diagnostics and never covers the real
                # operation. Logical chunks bound possible internal copies for
                # noncontiguous views without flattening a full tensor.
                with torch.autocast(device_type=tensor.device.type, enabled=False):
                    limit = max(1, chunk_bytes // tensor.element_size())
                    for chunk, _ in _logical_chunks(tensor.detach(), limit):
                        low, high = torch.aminmax(chunk)
                        minimum = low if minimum is None else torch.minimum(minimum, low)
                        maximum = high if maximum is None else torch.maximum(maximum, high)
                    low, high = torch.stack((minimum, maximum)).to(torch.float64).cpu().tolist()
                finite = math.isfinite(low) and math.isfinite(high)
                magnitude = max(abs(low), abs(high)) if finite else None
                entry.update(
                    finite=finite, extreme=finite and magnitude > abs_threshold,
                    min=_json_number(low), max=_json_number(high), max_abs_finite=magnitude,
                )
            results.append(entry)
    return results


def _monitor_faults(summaries):
    return ([item["name"] for item in summaries if not item["finite"]],
            [item["name"] for item in summaries if item["extreme"]])


def _snapshot_faults(summaries):
    return ([item["name"] for item in summaries if not item["finite"]],
            [item["name"] for item in summaries if item["extreme_count"]])


def _storage_bytes(value):
    storages = {(str(tensor.device), tensor.untyped_storage()._cdata):
                tensor.untyped_storage().nbytes() for _, tensor in _tensors(value)}
    return sum(storages.values())


class TrainingBackwardBoundaryProbe(BackwardBoundaryProbe):
    """Boundary probe with explicit operation selection and capture policy."""

    def __init__(self, model, directory, *, mode="pristine", selected_operations=None,
                 preprocess_module=None, compute_module=None, output_module=None):
        if mode not in ("pristine", "monitor_on_anomaly"):
            raise ValueError(f"Unsupported training boundary mode: {mode!r}")
        operations = DEFAULT_OPERATIONS if selected_operations is None else selected_operations
        if isinstance(operations, str):
            raise ValueError("selected_operations must be a sequence of operation names")
        operations = tuple(operations)
        if not operations or len(set(operations)) != len(operations):
            raise ValueError("Select at least one operation, without duplicate names")
        unknown = set(operations) - set(self._OPERATIONS)
        if unknown:
            raise ValueError(f"Unknown backward operations: {sorted(unknown)}")
        if mode == "monitor_on_anomaly" and os.environ.get("NAN_BACKWARD_SAVE_ALL") == "1":
            raise ValueError("monitor_on_anomaly requires NAN_BACKWARD_SAVE_ALL=0")
        self.mode = mode
        self.selected_operations = operations
        super().__init__(model, directory, preprocess_module=preprocess_module,
                         compute_module=compute_module, output_module=output_module)

    def _emit(self, event):
        super()._emit({"probe_mode": self.mode,
                       "selected_operations": list(self.selected_operations), **event})

    def _run(self, operation, function, signature, args, kwargs):
        if self.attempt is None:
            raise BoundaryProbeError("Call set_attempt before the training forward")
        if self.closed or self.failed:
            raise BoundaryProbeError("Cannot execute a closed or failed training probe")
        if operation not in self.selected_operations:
            self.call += 1
            self._emit({"event": "skipped", "call": self.call, "operation": operation,
                        "reason": "operation_filter"})
            return function(*args, **kwargs)
        if self.mode == "pristine":
            return super()._run(operation, function, signature, args, kwargs)
        return self._run_monitor(operation, function, signature, args, kwargs)

    def _run_monitor(self, operation, function, signature, args, kwargs):
        bound = signature.bind(*args, **kwargs)
        weight_argument, output_names = self._OPERATIONS[operation]
        weight = bound.arguments.get(weight_argument)
        owner = self._pointers.get(_pointer(weight)) if isinstance(weight, torch.Tensor) else None
        if owner is None:
            raise BoundaryProbeError(
                f"Cannot attribute {operation} {weight_argument} pointer to a model layer")
        self.call += 1
        event = {"call": self.call, "operation": operation, **owner}
        if self.target not in ("*", "all") and self.target not in owner["parameter"]:
            self._emit({"event": "skipped", **event, "reason": "target_filter",
                        "inputs": _source_specs(bound.arguments)})
            return function(*args, **kwargs)

        started = time.monotonic()
        event["execution_controls"] = _execution_controls()
        input_monitor = monitor_summaries(bound.arguments, chunk_bytes=self.chunk_bytes,
                                          abs_threshold=self.abs_threshold)
        bad_inputs, extreme_inputs = _monitor_faults(input_monitor)
        event.update(inputs=input_monitor, input_device_monitor=input_monitor,
                     input_observation={
                         "source": "pre_call_device_scalar_reductions",
                         "scope": "all real floating-point logical input elements",
                         "source_devices_synchronized_before_scan": True,
                         "scan_completed_unix_time": time.time(),
                         "operation_started": False, "snapshot_present": False,
                     })
        self._emit({"event": "before", **event})
        if bad_inputs or extreme_inputs:
            self._capture_monitor_fault(operation, owner["layer"], "input", signature,
                                        args, kwargs, None, event, started,
                                        bad_inputs, extreme_inputs, [], [])

        output = function(*args, **kwargs)
        if not isinstance(output, (tuple, list)) or len(output) != len(output_names):
            raise BoundaryProbeError(f"Unexpected {operation} return structure")
        named_output = dict(zip(output_names, output))
        output_monitor = monitor_summaries(named_output, chunk_bytes=self.chunk_bytes,
                                           abs_threshold=self.abs_threshold)
        bad_outputs, extreme_outputs = _monitor_faults(output_monitor)
        event.update(outputs=output_monitor, output_device_monitor=output_monitor,
                     seconds=time.monotonic() - started)
        self._emit({"event": "after", **event})
        if bad_outputs or extreme_outputs:
            self._capture_monitor_fault(operation, owner["layer"], "output", signature,
                                        args, kwargs, output, event, started,
                                        [], [], bad_outputs, extreme_outputs)
        return output

    def _capture_monitor_fault(self, operation, layer, stage, signature, args, kwargs,
                               output, event, started, bad_inputs, extreme_inputs,
                               bad_outputs, extreme_outputs):
        # A failed copy/save must not make a known anomalous probe reusable.
        self.failed = True
        originals = {"inputs": {"args": args, "kwargs": kwargs}, "outputs": output}
        input_bytes = _storage_bytes(originals["inputs"])
        output_bytes = _storage_bytes(output)
        capture_started = time.time()
        # One union copy preserves aliases across the input/output boundary.
        # The hard limit counts each backing storage once for this snapshot.
        copied, capture_bytes = _cpu_copy_tree(originals, self.chunk_bytes, self.max_bytes)
        capture_completed = time.time()
        source_bound = signature.bind(*args, **kwargs)
        saved_bound = signature.bind(*copied["inputs"]["args"], **copied["inputs"]["kwargs"])
        inputs = _summaries(saved_bound.arguments, self.chunk_bytes,
                            _source_specs(source_bound.arguments), self.abs_threshold)
        names = self._OPERATIONS[operation][1]
        outputs = [] if output is None else _summaries(
            dict(zip(names, copied["outputs"])), self.chunk_bytes,
            _source_specs(dict(zip(names, output))), self.abs_threshold)
        snapshot_bad_inputs, snapshot_extreme_inputs = _snapshot_faults(inputs)
        snapshot_bad_outputs, snapshot_extreme_outputs = _snapshot_faults(outputs)
        snapshot_trigger_bad = snapshot_bad_inputs if stage == "input" else snapshot_bad_outputs
        snapshot_trigger_extreme = (snapshot_extreme_inputs if stage == "input"
                                    else snapshot_extreme_outputs)
        trigger_bad = bad_inputs if stage == "input" else bad_outputs
        trigger_extreme = extreme_inputs if stage == "input" else extreme_outputs
        event = {
            **event, "inputs": inputs, "outputs": outputs if output is not None else None,
            "input_bytes": input_bytes, "output_bytes": output_bytes,
            "capture_storage_bytes": capture_bytes,
            "capture_storage_accounting": "unique union of complete input and output storages",
            "input_snapshot_pristine": stage == "input",
            "input_snapshot_observation": {
                "source": ("pristine_pre_call_cpu_snapshot" if stage == "input"
                           else "post_call_fault_cpu_snapshot"),
                "operation_started": stage == "output",
                "input_mutation_during_operation_excluded": stage == "input",
                "source_devices_synchronized_before_copy": True,
                "capture_started_unix_time": capture_started,
                "capture_completed_unix_time": capture_completed,
                "scan_completed_unix_time": time.time(),
            },
            "bad_inputs": bad_inputs, "extreme_inputs": extreme_inputs,
            "bad_outputs": bad_outputs, "extreme_outputs": extreme_outputs,
            "snapshot_bad_inputs": snapshot_bad_inputs,
            "snapshot_extreme_inputs": snapshot_extreme_inputs,
            "snapshot_bad_outputs": snapshot_bad_outputs,
            "snapshot_extreme_outputs": snapshot_extreme_outputs,
            "device_trigger_persists_in_cpu_snapshot": (
                set(trigger_bad) <= set(snapshot_trigger_bad)
                and set(trigger_extreme) <= set(snapshot_trigger_extreme)),
            "operation_executed": stage == "output",
            "seconds": time.monotonic() - started,
        }
        # Freeze even if the later CPU snapshot no longer shows the anomaly.
        # That discrepancy is evidence, never a reason to rerun or continue.
        self._freeze(operation, layer, stage, copied["inputs"], copied["outputs"], event)

    def _save(self, operation, layer, stage, inputs, outputs, event):
        if self.mode == "pristine":
            return super()._save(operation, layer, stage, inputs, outputs, event)
        frame = self.attempt
        identity = f"{frame['mode']}-r{frame['repeat']}-s{frame['step']}"
        target = self.directory / (
            f"training-monitor-{self.session}-{identity}-{self.call:04d}-{operation}-{stage}.pt")
        temporary = target.with_suffix(".pt.tmp")
        pristine = event["input_snapshot_pristine"]
        payload = {
            "format": MONITOR_FORMAT, "format_version": 1,
            "session": self.session, "attempt": frame,
            "probe_mode": self.mode, "operation": operation, "layer": layer, "stage": stage,
            "input_snapshot_pristine": pristine,
            "args": inputs["args"], "kwargs": inputs["kwargs"], "outputs": outputs,
            "execution_controls": event["execution_controls"], "event": event,
            "note": (
                "Inputs are independent pre-call CPU storage copies; operation did not execute. "
                if pristine else
                "Inputs and outputs are post-call fault-time CPU storage copies. Input bytes "
                "may have changed during execution; pre-call scalar checks are not pristine "
                "input bytes and cannot establish exact original-call replay. "
            ) + "Load only as a trusted local artifact.",
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
        print(f"[training-boundary] saved {stage}: {layer} / {operation}: {target}", flush=True)
        return target


def install(model, directory, *, mode="pristine", selected_operations=None):
    return TrainingBackwardBoundaryProbe(model, directory, mode=mode,
                                          selected_operations=selected_operations)
