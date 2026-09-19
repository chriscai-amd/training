"""Selected output-backward monitor with batched endpoint readback.

Each boundary still synchronizes its source devices, scans every floating
logical input/output element, and resolves all pre-call flags before running
the operation. Native-dtype extrema are promoted to float64 and transferred
in one batch per source device. This changes scheduling of the diagnostic
reductions, without adding delayed attribution or changing capture provenance.

Output-fault input snapshots remain post-call copies, as in the original
monitor. The scoped subclass never replaces globals in the frozen base probe.
"""
from __future__ import annotations

import functools
import math
import types

import torch

from nan_backward_boundaries import BoundaryProbeError, _logical_chunks, _source_specs, _synchronize, _tensors
from nan_backward_training_probe import TrainingBackwardBoundaryProbe, _json_number


OUTPUT_OPERATIONS = ("hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd")
BATCHED_POLICY = {
    "format": "training_boundary_batched_endpoints_v1",
    "scope": "all selected floating logical elements; native-dtype chunk aminmax",
    "source_devices_synchronized_before_scan": True,
    "endpoint_dtype": "torch.float64",
    "readbacks": "one synchronous endpoint batch per source device per boundary scan",
    "pre_call_fault_resolved_before_operation": True,
    "output_fault_inputs_pristine": False,
    "changes": "diagnostic reduction queue/readback scheduling; no production operation change",
}


def _read_endpoint_batch(endpoints):
    """The only device-to-host value transfer in one device's boundary scan."""
    return endpoints.cpu().tolist()


def batched_monitor_summaries(value, *, chunk_bytes, abs_threshold):
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
    _synchronize(value)
    specs = _source_specs(value)
    results = []
    pending = {}
    with torch.no_grad():
        for name, tensor in tensors:
            entry = {
                "name": name, **specs[name], "numel": tensor.numel(),
                "observation": "device_logical_minmax", "abs_threshold": abs_threshold,
                "scanned": tensor.is_floating_point(), "finite": True,
                "extreme": False, "min": None, "max": None,
                "max_abs_finite": None,
            }
            results.append(entry)
            if not tensor.is_floating_point() or not tensor.numel():
                continue
            minimum = maximum = None
            with torch.autocast(device_type=tensor.device.type, enabled=False):
                limit = max(1, chunk_bytes // tensor.element_size())
                for chunk, _ in _logical_chunks(tensor.detach(), limit):
                    low, high = torch.aminmax(chunk)
                    minimum = low if minimum is None else torch.minimum(minimum, low)
                    maximum = high if maximum is None else torch.maximum(maximum, high)
                endpoints = torch.stack((minimum, maximum)).to(torch.float64)
            pending.setdefault(str(tensor.device), []).append((entry, endpoints))

        # No tensor's values are read on the host before all reductions have
        # been queued. Resolve each complete device batch before returning any
        # summary to the existing immediate-trigger control flow.
        for group in pending.values():
            endpoints = torch.stack([values for _, values in group])
            rows = _read_endpoint_batch(endpoints)
            if len(rows) != len(group):
                raise BoundaryProbeError("Incomplete batched endpoint readback")
            for (entry, _), (low, high) in zip(group, rows):
                finite = math.isfinite(low) and math.isfinite(high)
                magnitude = max(abs(low), abs(high)) if finite else None
                entry.update(finite=finite, extreme=finite and magnitude > abs_threshold,
                             min=_json_number(low), max=_json_number(high), max_abs_finite=magnitude)
    return results


def _scoped_monitor_method():
    """Reuse proven capture/control flow without mutating its module globals."""
    original = TrainingBackwardBoundaryProbe._run_monitor
    namespace = {**original.__globals__, "monitor_summaries": batched_monitor_summaries}
    replacement = types.FunctionType(original.__code__, namespace, original.__name__,
                                     original.__defaults__, original.__closure__)
    replacement.__kwdefaults__ = original.__kwdefaults__
    return functools.update_wrapper(replacement, original)


class BatchedTrainingBackwardBoundaryProbe(TrainingBackwardBoundaryProbe):
    """Immediate monitor for the output-gradient GEMM and norm/multiply stage."""

    def __init__(self, model, directory, *, mode="monitor_on_anomaly", selected_operations=None,
                 preprocess_module=None, compute_module=None, output_module=None):
        operations = OUTPUT_OPERATIONS if selected_operations is None else selected_operations
        if mode != "monitor_on_anomaly":
            raise ValueError("Batched endpoint probe requires monitor_on_anomaly")
        if isinstance(operations, str) or not operations or set(operations) - set(OUTPUT_OPERATIONS):
            raise ValueError("Batched endpoint probe supports only output-gradient GEMM and normalization")
        super().__init__(model, directory, mode=mode, selected_operations=operations,
                         preprocess_module=preprocess_module, compute_module=compute_module,
                         output_module=output_module)

    _run_monitor = _scoped_monitor_method()

    def _emit(self, event):
        super()._emit({"boundary_scan_policy": BATCHED_POLICY, **event})

    def _save(self, operation, layer, stage, inputs, outputs, event):
        return super()._save(operation, layer, stage, inputs, outputs,
                             {**event, "boundary_scan_policy": BATCHED_POLICY})
