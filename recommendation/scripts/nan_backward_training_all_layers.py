"""All three STU layers, seven backward operations, and dense phase sentinels.

The operation monitor uses the frozen batched scanners in private function
namespaces. The optional optimizer sentinel queues an autograd completion
callback from parameter hooks, then checks gradients before backward returns
and before production clipping. Sparse/in-backward optimizers are excluded.
"""
from __future__ import annotations

from collections import Counter
import copy
import os
import time

import torch

from nan_backward_boundaries import (
    BoundaryAnomalyError, BoundaryProbeError, _cpu_copy_tree, _source_specs,
    _summaries,
)
from nan_backward_training_batched import BATCHED_POLICY, batched_monitor_summaries
from nan_backward_training_extended import ExtendedTrainingBackwardBoundaryProbe
from nan_backward_training_probe import TrainingBackwardBoundaryProbe, _monitor_faults
from nan_backward_training_upstream import ALL_OPERATIONS, _scoped_summary_method

ALL_LAYER_POLICY = {
    **BATCHED_POLICY,
    "format": "training_all_layers_batched_endpoints_v1",
    "layer_indices": [0, 1, 2],
    "operations_per_layer": list(ALL_OPERATIONS),
    "phases": ["before", "after"],
    "expected_operation_endpoints_per_complete_backward": 42,
    "write_only_destinations_scanned_before_call": False,
    "attention_scalar_arguments": "bound original public-call signature; no reconstructed defaults beyond signature binding",
}
DENSE_PHASE_POLICY = {
    "format": "training_dense_phase_sentinel_v1",
    "phases": ["before_forward", "after_backward_before_clip", "before_optimizer_step", "after_optimizer_step"],
    "selection": "ordinary strided trainable model parameters also present in explicit optimizer param_groups",
    "before_clip_hook": "one autograd engine completion callback queued from parameter gradient hooks",
    "excluded": "sparse/sharded and in-backward/fused optimizers; optimizer moments are not scanned or captured",
    "fault_capture": "complete selected dense parameter and gradient storages at the observed phase; not a full training replay",
    "optimizer_step_interception": "local delegating proxy; original optimizer and methods are unchanged",
}


class _OptimizerProxy:
    def __init__(self, original, sentinel):
        self._original, self._sentinel = original, sentinel

    def __getattr__(self, name):
        return getattr(self._original, name)

    def step(self, *args, **kwargs):
        if args or kwargs:
            raise BoundaryProbeError("Dense sentinel requires the production no-argument optimizer.step")
        self._sentinel.before_optimizer_step()
        result = self._original.step()
        self._sentinel.after_optimizer_step()
        return result


class DenseTrainingPhaseSentinel:
    def __init__(self, model, optimizer, probe):
        self.probe = probe
        all_optimizer_parameters = {id(parameter): parameter
                                    for group in optimizer.param_groups for parameter in group["params"]}
        self.parameters = {}
        self.excluded = []
        seen = set()
        for name, parameter in model.named_parameters():
            if id(parameter) not in all_optimizer_parameters:
                continue
            seen.add(id(parameter))
            if (hasattr(parameter, "local_shards") or hasattr(parameter, "to_local")
                    or getattr(parameter, "_in_backward_optimizers", None)
                    or not parameter.requires_grad):
                self.excluded.append(name)
                continue
            if (not isinstance(parameter, torch.Tensor) or parameter.layout != torch.strided
                    or not parameter.is_floating_point()):
                raise BoundaryProbeError(f"Unsupported explicit optimizer parameter: {name}")
            self.parameters[name] = parameter
        if seen != set(all_optimizer_parameters):
            raise BoundaryProbeError("Explicit optimizer contains parameters missing from model.named_parameters")
        if not self.parameters:
            raise BoundaryProbeError("Dense phase sentinel found no ordinary trainable optimizer parameters")
        self.optimizer_proxy = _OptimizerProxy(optimizer, self)
        self.handles = []
        self.queued = self.backward_complete = self.optimizer_complete = False
        try:
            for parameter in self.parameters.values():
                self.handles.append(parameter.register_hook(self._queue_after_backward))
            probe._emit({"event": "dense_sentinel_installed", "dense_phase_policy": DENSE_PHASE_POLICY,
                         "dense_parameter_names": list(self.parameters), "excluded_optimizer_parameters": self.excluded})
        except BaseException:
            self.close()
            raise

    def begin(self):
        self.queued = self.backward_complete = self.optimizer_complete = False
        self.scan("before_forward", include_gradients=False)

    def _queue_after_backward(self, gradient):
        if self.probe.attempt is None or self.probe.closed or self.probe.failed:
            raise BoundaryProbeError("Dense gradient hook requires a live nonfailed attempt")
        if self.backward_complete:
            raise BoundaryProbeError("Multiple backwards in one training attempt are unsupported")
        if not self.queued:
            self.queued = True
            torch.autograd.Variable._execution_engine.queue_callback(self._after_backward)
        return gradient

    def _after_backward(self):
        if self.probe.failed:
            return  # Preserve a prior, more local operation-boundary failure.
        self.probe.assert_complete_operation_scans()
        self.scan("after_backward_before_clip", include_gradients=True)
        self.backward_complete = True

    def before_optimizer_step(self):
        if not self.backward_complete or self.optimizer_complete:
            raise BoundaryProbeError("Optimizer step reached without exactly one completed backward")
        self.scan("before_optimizer_step", include_gradients=True)

    def after_optimizer_step(self):
        self.scan("after_optimizer_step", include_gradients=True)
        self.optimizer_complete = True

    def scan(self, phase, *, include_gradients):
        probe = self.probe
        if probe.attempt is None or probe.closed or probe.failed:
            raise BoundaryProbeError("Dense phase scan requires a live nonfailed attempt")
        gradients = {name: parameter.grad for name, parameter in self.parameters.items()
                     if parameter.grad is not None}
        if include_gradients and not gradients:
            raise BoundaryProbeError("Completed backward produced no selected dense gradients")
        value = {"parameters": self.parameters}
        if include_gradients:
            value["gradients"] = gradients
        summaries = batched_monitor_summaries(value, chunk_bytes=probe.chunk_bytes,
                                              abs_threshold=probe.abs_threshold)
        bad, extreme = _monitor_faults(summaries)
        event = {"event": "dense_phase", "phase": phase, "dense_phase_policy": DENSE_PHASE_POLICY,
                 "summaries": summaries, "bad": bad, "extreme": extreme,
                 "missing_gradient_names": [name for name in self.parameters if name not in gradients],
                 "scan_completed_unix_time": time.time(),
                 "explicit_optimizer_step_executed": phase == "after_optimizer_step"}
        probe._emit(event)
        if not bad and not extreme:
            return
        probe.failed = True
        # Include gradients even for a before-forward fault. Copies happen only
        # after a trigger and retain complete backing storages/alias relations.
        captured_value = {"parameters": self.parameters, "gradients": gradients}
        captured, size = _cpu_copy_tree(captured_value, probe.chunk_bytes, probe.max_bytes)
        snapshot_summaries = _summaries(captured, probe.chunk_bytes,
                                       _source_specs(captured_value), probe.abs_threshold)
        snapshot_bad = {item["name"] for item in snapshot_summaries if not item["finite"]}
        snapshot_extreme = {item["name"] for item in snapshot_summaries if item["extreme_count"]}
        target = probe.directory / f"dense-phase-{probe.session}-s{probe.attempt['step']}-{phase}.pt"
        temporary = target.with_suffix(".pt.tmp")
        payload = {"format": "training_dense_phase_capture_v1", "format_version": 1,
                   "attempt": copy.deepcopy(probe.attempt), "phase": phase,
                   "dense_phase_policy": DENSE_PHASE_POLICY, "event": event,
                   "snapshot": captured, "snapshot_summaries": snapshot_summaries,
                   "capture_storage_bytes": size,
                   "trigger_persists_in_cpu_snapshot": set(bad) <= snapshot_bad and set(extreme) <= snapshot_extreme}
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        descriptor = os.open(probe.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        probe._emit({"event": "dense_dump_complete", "phase": phase, "dump": str(target),
                     "trigger_persists_in_cpu_snapshot": payload["trigger_persists_in_cpu_snapshot"]})
        os.fsync(probe._events.fileno())
        raise BoundaryAnomalyError(target, "dense_phase_sentinel", "explicit_optimizer_parameters", phase)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class AllLayersBatchedTrainingBackwardBoundaryProbe(ExtendedTrainingBackwardBoundaryProbe):
    def __init__(self, model, directory, *, mode="monitor_on_anomaly", selected_operations=None,
                 optimizer=None, preprocess_module=None, compute_module=None, output_module=None):
        operations = ALL_OPERATIONS if selected_operations is None else selected_operations
        if mode != "monitor_on_anomaly" or isinstance(operations, str) or tuple(operations) != ALL_OPERATIONS:
            raise ValueError("All-layer probe requires monitor_on_anomaly and all seven ordered operations")
        if os.environ.get("NAN_BACKWARD_TARGET", "") not in ("*", "all"):
            raise ValueError("All-layer probe requires NAN_BACKWARD_TARGET='*'")
        self.resolved_layer_operations = {}
        self.endpoint_counts = Counter()
        self.sentinel = None
        super().__init__(model, directory, mode=mode, selected_operations=operations,
                         preprocess_module=preprocess_module, compute_module=compute_module,
                         output_module=output_module)
        try:
            if optimizer is not None:
                self.sentinel = DenseTrainingPhaseSentinel(model, optimizer, self)
        except BaseException:
            self.close()
            raise

    def _install_hooks(self):
        resolved = {}
        for index in range(3):
            selector = f"_stu_layers.{index}."
            matches = [layer for layer in self._layers if selector in layer + "."]
            if len(matches) != 1 or matches[0] in resolved:
                raise BoundaryProbeError(f"Expected one distinct STU owner for {selector!r}; found {matches}")
            resolved[matches[0]] = list(ALL_OPERATIONS)
        if set(resolved) != set(self._layers):
            raise BoundaryProbeError("All-layer diagnostic requires exactly three STU layers")
        self.resolved_layer_operations = resolved
        super()._install_hooks()

    def set_attempt(self, mode, repeat, step):
        if self.sentinel is not None and self.attempt is not None and not self.sentinel.optimizer_complete:
            raise BoundaryProbeError("New attempt reached before previous optimizer step completed")
        self.endpoint_counts.clear()
        super().set_attempt(mode, repeat, step)
        if self.sentinel is not None:
            self.sentinel.begin()

    def _metadata(self, event):
        event = {**event, "boundary_scan_policy": ALL_LAYER_POLICY,
                 "resolved_layer_operations": self.resolved_layer_operations}
        scalar_arguments = getattr(getattr(self, "_local", None), "readwrite_scalar_arguments", None)
        if event.get("operation") == "hstu_attention_bwd" and scalar_arguments is not None:
            event["scalar_arguments"] = scalar_arguments
        return event

    def _emit(self, event):
        if event.get("event") in ("before", "after"):
            key = (event["layer"], event["operation"], event["event"])
            if key[0] not in self.resolved_layer_operations or key[1] not in ALL_OPERATIONS:
                raise BoundaryProbeError(f"Unexpected all-layer endpoint: {key}")
            self.endpoint_counts[key] += 1
            if self.endpoint_counts[key] != 1:
                raise BoundaryProbeError(f"Repeated all-layer endpoint: {key}")
        super()._emit(self._metadata(event))

    def assert_complete_operation_scans(self):
        expected = {(layer, operation, phase) for layer in self.resolved_layer_operations
                    for operation in ALL_OPERATIONS for phase in ("before", "after")}
        if set(self.endpoint_counts) != expected or any(count != 1 for count in self.endpoint_counts.values()):
            raise BoundaryProbeError(f"Incomplete all-layer backward endpoints: {sorted(expected - set(self.endpoint_counts))}")

    _run_monitor = _scoped_summary_method(TrainingBackwardBoundaryProbe._run_monitor)
    _batched_readwrite_monitor = _scoped_summary_method(ExtendedTrainingBackwardBoundaryProbe._run_readwrite)

    def _run_readwrite(self, operation, function, signature, args, kwargs, writes, owner):
        if getattr(self._local, "readwrite_scalar_arguments", None) is not None:
            raise BoundaryProbeError("Nested read/write boundary is unsupported")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        scalars = {}
        if operation == "hstu_attention_bwd":
            for name in ("N", "alpha", "max_attn_len", "contextual_seq_len", "enable_tma", "num_softmax_heads"):
                value = bound.arguments[name]
                if value is not None and type(value) not in (bool, int, float, str):
                    raise BoundaryProbeError(f"Expected ordinary scalar attention argument: {name}")
                scalars[name] = value
        self._local.readwrite_scalar_arguments = scalars
        try:
            return self._batched_readwrite_monitor(operation, function, signature, args, kwargs, writes, owner)
        finally:
            self._local.readwrite_scalar_arguments = None

    def _save(self, operation, layer, stage, inputs, outputs, event):
        return super()._save(operation, layer, stage, inputs, outputs, self._metadata(event))

    def _finish_readwrite(self, operation, layer, stage, signature, args, kwargs, writes,
                          destinations, event, *remaining):
        return super()._finish_readwrite(operation, layer, stage, signature, args, kwargs, writes,
                                        destinations, self._metadata(event), *remaining)

    def close(self):
        if self.sentinel is not None:
            self.sentinel.close()
        super().close()
