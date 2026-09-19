"""Deferred scalar-only backward trace with one preclip readback per device.

Reductions are queued on the operation's current autograd stream. Only tiny
endpoint tensors and CPU metadata survive a boundary; record_stream protects
the storage lifetime of diagnostic reads. The completion callback joins all
observed streams and resolves flags before clipping and the explicit optimizer.
This changes scheduling and allocation lifetimes. It captures no replay bytes,
and host enqueue order does not establish total GPU causality across streams.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
import os
import time

import torch

from nan_backward_boundaries import (
    BoundaryAnomalyError, BoundaryProbeError, _execution_controls, _logical_chunks,
    _pointer, _source_specs, _tensors,
)
from nan_backward_training_all_layers import (
    AllLayersBatchedTrainingBackwardBoundaryProbe, DenseTrainingPhaseSentinel,
)
from nan_backward_training_batched import _read_endpoint_batch
from nan_backward_training_extended import _argument_layouts
from nan_backward_training_probe import _json_number
from nan_backward_training_upstream import ALL_OPERATIONS

DEFERRED_POLICY = {
    "format": "training_deferred_scalar_trace_v1",
    "layer_indices": [0, 1, 2], "operations_per_layer": list(ALL_OPERATIONS),
    "phases": ["before", "after"], "expected_operation_endpoints_per_complete_backward": 42,
    "scope": "all floating logical elements; native-dtype chunk minmax; FP64 scalar endpoints",
    "boundary_global_synchronization": False, "boundary_host_value_readback": False,
    "flush": "autograd completion before clipping; wait on recorded scan streams; one endpoint batch readback per device",
    "source_lifetime": "record_stream for every CUDA diagnostic input; no retained activation references",
    "ordering": "host enqueue order and per-stream FIFO; not a proof of cross-stream causality",
    "raw_snapshot_present": False, "pristine_input_replay_claim": False,
    "delayed_stop": "before-call and before-forward flags remain unresolved until backward completion; forward and downstream backward can execute on already-bad inputs",
    "sparse_updates": "fused sparse updates may already occur in backward before deferred fault resolution",
}
DEFERRED_DENSE_POLICY = {
    "format": "deferred_dense_phases_v1",
    "queued_phases": ["before_forward", "after_backward_before_clip"],
    "selection": "ordinary strided trainable model parameters in explicit optimizer param_groups",
    "after_backward_scan": "selected parameters and accumulated gradients",
    "postclip_or_postoptimizer_scans": False,
    "stop": "all queued flags resolved before clipping and explicit optimizer.step",
    "gradient_read_ordering": "relies on PyTorch autograd completion-callback caller/leaf-stream synchronization for accumulated gradients; boundary scalar producer streams are explicitly joined before readback",
    "excluded": "sparse/sharded and in-backward fused optimizers and optimizer moments",
}


class DeferredScalarQueue:
    def __init__(self, *, chunk_bytes, abs_threshold):
        if chunk_bytes <= 0 or not math.isfinite(abs_threshold) or abs_threshold <= 0:
            raise ValueError("Positive chunk size and finite threshold required")
        self.chunk_bytes, self.abs_threshold = chunk_bytes, abs_threshold
        self.scans = []
        self.pending = defaultdict(list)
        self.streams = defaultdict(dict)
        self.events = []
        self.flushed = False

    def enqueue(self, value, metadata):
        if self.flushed:
            raise BoundaryProbeError("Cannot enqueue into a flushed scalar trace")
        tensors = list(_tensors(value))
        for name, tensor in tensors:
            if (tensor.layout != torch.strided or tensor.is_quantized or tensor.is_complex()
                    or tensor.device.type not in ("cpu", "cuda")):
                raise BoundaryProbeError(f"Unsupported deferred diagnostic tensor: {name}")
        specs = _source_specs(value)
        scan = {**copy.deepcopy(metadata), "enqueue_index": len(self.scans),
                "host_enqueue_unix_time": time.time(), "observation": "deferred_device_reduction",
                "values_resolved_on_host": False, "summaries": []}
        self.scans.append(scan)
        with torch.no_grad():
            for name, tensor in tensors:
                device = str(tensor.device)
                stream = torch.cuda.current_stream(tensor.device) if tensor.device.type == "cuda" else None
                stream_id = int(stream.cuda_stream) if stream is not None else None
                entry = {"name": name, **specs[name], "numel": tensor.numel(),
                         "scanned": tensor.is_floating_point(), "stream_id": stream_id,
                         "abs_threshold": self.abs_threshold}
                scan["summaries"].append(entry)
                if not tensor.is_floating_point() or not tensor.numel():
                    continue
                if stream is not None:
                    # Preserve only allocator lifetime, not a Python reference
                    # to the potentially multi-GiB source activation.
                    tensor.record_stream(stream)
                    self.streams[device][stream_id] = stream
                minimum = maximum = None
                with torch.autocast(device_type=tensor.device.type, enabled=False):
                    limit = max(1, self.chunk_bytes // tensor.element_size())
                    for chunk, _ in _logical_chunks(tensor.detach(), limit):
                        low, high = torch.aminmax(chunk)
                        minimum = low if minimum is None else torch.minimum(minimum, low)
                        maximum = high if maximum is None else torch.maximum(maximum, high)
                    endpoints = torch.stack((minimum, maximum)).to(torch.float64)
                self.pending[device].append((entry, endpoints))
        return copy.deepcopy(scan)

    def flush(self):
        if self.flushed:
            raise BoundaryProbeError("Scalar trace can be flushed only once per attempt")
        self.flushed = True
        for device, group in self.pending.items():
            if torch.device(device).type == "cuda":
                collector = torch.cuda.current_stream(torch.device(device))
                for producer in self.streams[device].values():
                    event = torch.cuda.Event()
                    event.record(producer)
                    collector.wait_event(event)
                    self.events.append(event)
                # Every tiny scalar is retained until this readback completes.
                # Sources on other streams are joined explicitly above.
                with torch.cuda.stream(collector):
                    batch = torch.stack([endpoints for _, endpoints in group])
                    rows = _read_endpoint_batch(batch)
            else:
                rows = _read_endpoint_batch(torch.stack([endpoints for _, endpoints in group]))
            if len(rows) != len(group):
                raise BoundaryProbeError("Incomplete deferred endpoint readback")
            for (entry, _), (minimum, maximum) in zip(group, rows):
                finite = math.isfinite(minimum) and math.isfinite(maximum)
                magnitude = max(abs(minimum), abs(maximum)) if finite else None
                entry.update(finite=finite, extreme=finite and magnitude > self.abs_threshold,
                             min=_json_number(minimum), max=_json_number(maximum), max_abs_finite=magnitude)
        for scan in self.scans:
            scan["values_resolved_on_host"] = True
            for entry in scan["summaries"]:
                if "finite" not in entry:
                    entry.update(finite=True, extreme=False, min=None, max=None, max_abs_finite=None)
            scan["flagged_names"] = [entry["name"] for entry in scan["summaries"]
                                     if not entry["finite"] or entry["extreme"]]
        result = copy.deepcopy(self.scans)
        self.pending.clear()
        self.streams.clear()
        self.events.clear()
        return result


class DeferredDenseSentinel(DenseTrainingPhaseSentinel):
    def begin(self):
        self.queued = self.backward_complete = self.optimizer_complete = False
        self.scan("before_forward", include_gradients=False)

    def scan(self, phase, *, include_gradients):
        gradients = {name: parameter.grad for name, parameter in self.parameters.items()
                     if parameter.grad is not None}
        if include_gradients and not gradients:
            raise BoundaryProbeError("Completed backward has no selected dense gradients")
        value = {"parameters": self.parameters}
        if include_gradients:
            value["gradients"] = gradients
        scan = self.probe.scalar_queue.enqueue(value, {
            "kind": "dense_phase", "phase": phase,
            "missing_gradient_names": [name for name in self.parameters if name not in gradients]})
        self.probe._emit({"event": "dense_phase_queued", "dense_phase_policy": DEFERRED_DENSE_POLICY,
                          **scan})

    def _after_backward(self):
        if self.probe.failed:
            return
        self.probe.assert_complete_operation_scans()
        self.scan("after_backward_before_clip", include_gradients=True)
        self.probe.flush_before_clip()
        self.backward_complete = True

    def before_optimizer_step(self):
        if not self.backward_complete or self.optimizer_complete:
            raise BoundaryProbeError("Optimizer step reached without one completed deferred flush")

    def after_optimizer_step(self):
        self.optimizer_complete = True


class DeferredAllLayersTrainingBackwardProbe(AllLayersBatchedTrainingBackwardBoundaryProbe):
    def __init__(self, model, directory, *, optimizer, **options):
        self.scalar_queue = None
        super().__init__(model, directory, optimizer=None, **options)
        try:
            self.sentinel = DeferredDenseSentinel(model, optimizer, self)
        except BaseException:
            self.close()
            raise

    def set_attempt(self, mode, repeat, step):
        if self.scalar_queue is not None and not self.scalar_queue.flushed:
            raise BoundaryProbeError("Prior attempt's deferred scalars were not resolved")
        self.scalar_queue = DeferredScalarQueue(chunk_bytes=self.chunk_bytes, abs_threshold=self.abs_threshold)
        super().set_attempt(mode, repeat, step)

    def _metadata(self, event):
        event = {**event, "boundary_scan_policy": DEFERRED_POLICY,
                 "resolved_layer_operations": self.resolved_layer_operations}
        if event.get("event") == "dense_sentinel_installed":
            event["dense_phase_policy"] = DEFERRED_DENSE_POLICY
        scalars = getattr(getattr(self, "_local", None), "readwrite_scalar_arguments", None)
        if event.get("operation") == "hstu_attention_bwd" and scalars is not None:
            event["scalar_arguments"] = scalars
        return event

    def _queue_endpoint(self, operation, owner, call_id, phase, value, extra=None):
        if self.attempt is None or self.closed or self.failed or self.scalar_queue is None:
            raise BoundaryProbeError("Deferred trace requires a live nonfailed attempt")
        metadata = {"kind": "operation", "call": call_id, "operation": operation,
                    **owner, "phase": phase, **(extra or {})}
        scalars = getattr(self._local, "readwrite_scalar_arguments", None)
        if operation == "hstu_attention_bwd" and scalars is not None:
            metadata["scalar_arguments"] = copy.deepcopy(scalars)
        scan = self.scalar_queue.enqueue(value, metadata)
        self._emit({"event": phase, **scan})

    def _run_monitor(self, operation, function, signature, args, kwargs):
        bound = signature.bind(*args, **kwargs)
        weight_name, output_names = self._OPERATIONS[operation]
        weight = bound.arguments.get(weight_name)
        owner = self._pointers.get(_pointer(weight)) if isinstance(weight, torch.Tensor) else None
        if owner is None:
            raise BoundaryProbeError(f"Cannot attribute deferred {operation} to a model layer")
        self.call += 1
        call_id = self.call
        self._queue_endpoint(operation, owner, call_id, "before", bound.arguments,
                             {"execution_controls": _execution_controls()})
        outputs = function(*args, **kwargs)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(output_names):
            raise BoundaryProbeError(f"Unexpected {operation} return structure")
        self._queue_endpoint(operation, owner, call_id, "after", dict(zip(output_names, outputs)))
        return outputs

    def _batched_readwrite_monitor(self, operation, function, signature, args, kwargs, writes, owner):
        # The inherited outer wrapper records actual bound attention scalars.
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        destinations = tuple(bound.arguments[name] for name in writes)
        if not all(isinstance(value, torch.Tensor) for value in destinations):
            raise BoundaryProbeError("Read/write destinations must be tensors")
        reads = {name: value for name, value in bound.arguments.items() if name not in writes}
        before_pointers = tuple(_pointer(value) for value in destinations)
        self.call += 1
        call_id = self.call
        roles = {"read": list(reads), "write_only": list(writes)}
        self._queue_endpoint(operation, owner, call_id, "before", reads,
                             {"argument_roles": roles, "argument_layouts": _argument_layouts(bound.arguments),
                              "destination_initial_values_scanned": False,
                              "execution_controls": _execution_controls()})
        result = function(*args, **kwargs)
        if before_pointers != tuple(_pointer(value) for value in destinations):
            raise BoundaryProbeError("Read/write operation changed destination allocation/layout")
        if operation == "hstu_attention_bwd" and result is not None:
            raise BoundaryProbeError("Attention backward unexpectedly returned values")
        if operation == "hstu_silu_bwd" and (not isinstance(result, torch.Tensor)
                                            or _pointer(result) != _pointer(destinations[0])):
            raise BoundaryProbeError("SiLU did not return its original destination")
        self._queue_endpoint(operation, owner, call_id, "after",
                             dict(zip(self._OPERATIONS[operation][1], destinations)),
                             {"argument_roles": roles})
        return result

    def flush_before_clip(self):
        scans = self.scalar_queue.flush()
        flagged = [scan for scan in scans if scan["flagged_names"]]
        event = {"event": "deferred_flush", "scan_count": len(scans), "scans": scans,
                 "first_flagged_enqueue_index": flagged[0]["enqueue_index"] if flagged else None,
                 "raw_snapshot_present": False, "host_resolution_unix_time": time.time(),
                 "phase": "after_backward_before_clip", "dense_phase_policy": DEFERRED_DENSE_POLICY}
        self._emit(event)
        if not flagged:
            return
        self.failed = True
        first = flagged[0]
        target = self.directory / f"deferred-scalar-trace-{self.session}-s{self.attempt['step']}.json"
        temporary = target.with_suffix(".json.tmp")
        payload = {"format": "training_deferred_scalar_trace_fault_v1", "attempt": self.attempt,
                   "boundary_scan_policy": DEFERRED_POLICY, "dense_phase_policy": DEFERRED_DENSE_POLICY,
                   "first_flagged_observation": first, **event}
        with temporary.open("x") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        descriptor = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._emit({"event": "deferred_trace_fault_complete", "dump": str(target),
                    "first_flagged_enqueue_index": first["enqueue_index"], "raw_snapshot_present": False})
        os.fsync(self._events.fileno())
        raise BoundaryAnomalyError(target, first.get("operation", "dense_phase"),
                                   first.get("layer", "explicit_optimizer_parameters"), first["phase"])
