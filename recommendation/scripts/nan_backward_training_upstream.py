"""Batched upstream backward checks and an input-only lower-layer witness.

The upper STU layer scans all seven established operations before/after. The
adjacent lower layer scans only incoming output-GEMM inputs and stops before
that GEMM on a fault. Scalar witnesses retain upper residual dout and upper
preprocessing d_x across the attempt. They do not check elementwise addition
or exclude mutation between observation points.

Both monitor implementations use private function namespaces. Existing probe
modules, methods and their global summary functions remain unchanged.
"""
from __future__ import annotations

import copy
import functools
import os
import time
import types

import torch

from nan_backward_boundaries import BoundaryProbeError, _execution_controls, _pointer
from nan_backward_training_batched import BATCHED_POLICY, batched_monitor_summaries
from nan_backward_training_extended import ExtendedTrainingBackwardBoundaryProbe
from nan_backward_training_probe import TrainingBackwardBoundaryProbe, _monitor_faults

ALL_OPERATIONS = ("hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd",
                  "triton_addmm_bwd", "triton_weighted_layer_norm_bwd",
                  "hstu_attention_bwd", "hstu_silu_bwd", "hstu_output_weight_mm")
LOWER_OPERATION = "hstu_output_grad_mm"
UPSTREAM_POLICY = {
    **BATCHED_POLICY,
    "format": "training_upstream_batched_endpoints_v1",
    "upper_layer_phases": ["before", "after"],
    "lower_incoming_witness_phases": ["before"],
    "write_only_destinations_scanned_before_call": False,
    "residual_witness": "scalar summaries of upper original dout, upper preprocessing d_x, and lower incoming dout; no elementwise residual-add check",
}


def layer_selection(upper_layer=2, lower_layer=1):
    if (type(upper_layer) is not int or type(lower_layer) is not int
            or lower_layer < 0 or upper_layer != lower_layer + 1):
        raise ValueError("Select adjacent nonnegative upper/lower STU layer indices")
    return {f"_stu_layers.{upper_layer}.": list(ALL_OPERATIONS),
            f"_stu_layers.{lower_layer}.": [LOWER_OPERATION]}


def _scoped_summary_method(original):
    namespace = {**original.__globals__, "monitor_summaries": batched_monitor_summaries}
    replacement = types.FunctionType(original.__code__, namespace, original.__name__,
                                     original.__defaults__, original.__closure__)
    replacement.__kwdefaults__ = original.__kwdefaults__
    return functools.update_wrapper(replacement, original)


class UpstreamBatchedTrainingBackwardBoundaryProbe(ExtendedTrainingBackwardBoundaryProbe):
    def __init__(self, model, directory, *, mode="monitor_on_anomaly", selected_operations=None,
                 upper_layer=2, lower_layer=1, preprocess_module=None, compute_module=None,
                 output_module=None):
        if mode != "monitor_on_anomaly":
            raise ValueError("Upstream batched probe requires monitor_on_anomaly")
        operations = ALL_OPERATIONS if selected_operations is None else selected_operations
        if (isinstance(operations, str) or set(operations) != set(ALL_OPERATIONS)
                or len(operations) != len(ALL_OPERATIONS)):
            raise ValueError("Upstream probe requires all seven operation wrappers")
        if os.environ.get("NAN_BACKWARD_TARGET", "") not in ("*", "all"):
            raise ValueError("Upstream per-layer selection requires NAN_BACKWARD_TARGET='*'")
        self.layer_operation_selection = layer_selection(upper_layer, lower_layer)
        self.upper_selector = f"_stu_layers.{upper_layer}."
        self.lower_selector = f"_stu_layers.{lower_layer}."
        self.resolved_layer_operations = {}
        self._residual_summaries = {}
        super().__init__(model, directory, mode=mode, selected_operations=operations,
                         preprocess_module=preprocess_module, compute_module=compute_module,
                         output_module=output_module)

    def _install_hooks(self):
        # Resolve before any hooks or operation wrappers are installed.
        resolved = {}
        for selector, operations in self.layer_operation_selection.items():
            matches = [layer for layer in self._layers if selector in layer + "."]
            if len(matches) != 1 or matches[0] in resolved:
                raise BoundaryProbeError(f"Expected one distinct STU owner for {selector!r}; found {matches}")
            resolved[matches[0]] = list(operations)
            if selector == self.upper_selector:
                self.upper_owner = matches[0]
            else:
                self.lower_owner = matches[0]
        self.resolved_layer_operations = resolved
        super()._install_hooks()

    def set_attempt(self, mode, repeat, step):
        self._residual_summaries = {}
        return super().set_attempt(mode, repeat, step)

    def _metadata(self, event):
        return {**event, "boundary_scan_policy": UPSTREAM_POLICY,
                "layer_operation_selection": self.layer_operation_selection,
                "resolved_layer_operations": self.resolved_layer_operations,
                "residual_flow_witnesses": copy.deepcopy(self._residual_summaries)}

    def _emit(self, event):
        role = None
        field = None
        if event.get("layer") == getattr(self, "upper_owner", None):
            if event.get("event") == "before" and event.get("operation") == LOWER_OPERATION:
                role, field = "upper_original_dout", ("inputs", "dout")
            elif event.get("event") == "after" and event.get("operation") == "triton_weighted_layer_norm_bwd":
                role, field = "upper_preprocessing_d_x", ("outputs", "d_x")
        elif (event.get("layer") == getattr(self, "lower_owner", None)
              and event.get("event") == "before" and event.get("operation") == LOWER_OPERATION):
            role, field = "lower_incoming_dout", ("inputs", "dout")
        if role:
            matches = [item for item in event[field[0]] if item["name"] == field[1]]
            if len(matches) != 1 or role in self._residual_summaries:
                raise BoundaryProbeError(f"Missing or repeated residual scalar witness: {role}")
            self._residual_summaries[role] = {
                "call": event["call"], "layer": event["layer"], "operation": event["operation"],
                "phase": event["event"], "summary": copy.deepcopy(matches[0]),
                "observation": "scalar summary only; no retained tensor or byte-identity guarantee"}
        super()._emit(self._metadata(event))

    def _save(self, operation, layer, stage, inputs, outputs, event):
        return super()._save(operation, layer, stage, inputs, outputs, self._metadata(event))

    def _finish_readwrite(self, operation, layer, stage, signature, args, kwargs, writes,
                          destinations, event, *remaining):
        # Extended read/write capture writes its own payload and bypasses _save.
        return super()._finish_readwrite(operation, layer, stage, signature, args, kwargs, writes,
                                        destinations, self._metadata(event), *remaining)

    def _active(self):
        if self.attempt is None or self.closed or self.failed:
            raise BoundaryProbeError("Upstream monitor requires a live, nonfailed attempt")

    def _selected(self, operation, owner):
        return operation in self.resolved_layer_operations.get(owner["layer"], ())

    def _skip(self, operation, function, args, kwargs, owner):
        self.call += 1
        self._emit({"event": "skipped", "call": self.call, "operation": operation,
                    **owner, "reason": "layer_operation_filter"})
        return function(*args, **kwargs)

    _batched_regular_monitor = _scoped_summary_method(TrainingBackwardBoundaryProbe._run_monitor)
    _batched_readwrite_monitor = _scoped_summary_method(ExtendedTrainingBackwardBoundaryProbe._run_readwrite)

    def _run_monitor(self, operation, function, signature, args, kwargs):
        self._active()
        bound = signature.bind(*args, **kwargs)
        weight_argument = self._OPERATIONS[operation][0]
        weight = bound.arguments.get(weight_argument)
        owner = self._pointers.get(_pointer(weight)) if isinstance(weight, torch.Tensor) else None
        if owner is None:
            raise BoundaryProbeError(f"Cannot attribute {operation} to a model layer")
        if not self._selected(operation, owner):
            return self._skip(operation, function, args, kwargs, owner)
        if owner["layer"] == self.lower_owner:
            return self._run_lower_input_witness(operation, function, signature, args, kwargs, bound, owner)
        return self._batched_regular_monitor(operation, function, signature, args, kwargs)

    def _run_readwrite(self, operation, function, signature, args, kwargs, writes, owner):
        self._active()
        if not self._selected(operation, owner):
            return self._skip(operation, function, args, kwargs, owner)
        return self._batched_readwrite_monitor(operation, function, signature, args, kwargs, writes, owner)

    def _run_lower_input_witness(self, operation, function, signature, args, kwargs, bound, owner):
        if not {"upper_original_dout", "upper_preprocessing_d_x"}.issubset(self._residual_summaries):
            raise BoundaryProbeError("Lower incoming witness reached before both upper residual-path observations")
        self.call += 1
        started = time.monotonic()
        inputs = batched_monitor_summaries(bound.arguments, chunk_bytes=self.chunk_bytes,
                                          abs_threshold=self.abs_threshold)
        bad_inputs, extreme_inputs = _monitor_faults(inputs)
        event = {"call": self.call, "operation": operation, **owner,
                 "execution_controls": _execution_controls(),
                 "inputs": inputs, "input_device_monitor": inputs,
                 "input_only_witness": True, "output_values_monitored": False,
                 "input_observation": {
                     "source": "pre_call_device_scalar_reductions",
                     "scope": "all floating logical incoming GEMM inputs; residual-path scalar witness",
                     "source_devices_synchronized_before_scan": True,
                     "scan_completed_unix_time": time.time(), "operation_started": False,
                     "snapshot_present": False}}
        self._emit({"event": "before", **event})
        if bad_inputs or extreme_inputs:
            self._capture_monitor_fault(operation, owner["layer"], "input", signature,
                                        args, kwargs, None, self._metadata(event), started,
                                        bad_inputs, extreme_inputs, [], [])
        return function(*args, **kwargs)
