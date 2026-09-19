"""Optional output-weight GEMM and attention/SiLU training boundaries.

The production preprocessing backward recomputes Q/K/V, allocates one packed
DU/V/Q/K buffer, writes attention gradients, then writes DU using SiLU backward.
Write-only destination bytes are deliberately excluded from pre-call anomaly
scans. Pristine mode retains their initial storage as opaque replay state;
monitor mode retains full storage only after a fault.

The output-weight gradient check observes the actual second output-backward
GEMM, Y.T @ dout. Saved output weights identify the layer but are not arithmetic
inputs. This optional check verifies the actual saved or recomputed Y and the
original dout, and requires both GEMMs and normalization exactly once.

Existing four-boundary behavior delegates unchanged to the base probe. These
wrappers are only installed when an extended operation is explicitly selected.
"""

from __future__ import annotations

import functools
import inspect
import os
import time

import torch

from nan_backward_boundaries import (
    BoundaryAnomalyError, BoundaryProbeError, _cpu_copy_tree, _execution_controls,
    _OutputTorchProxy, _output_grad_mm_signature, _pointer, _source_specs, _summaries, _tensors,
)
from nan_backward_training_probe import (
    TrainingBackwardBoundaryProbe, _monitor_faults, _snapshot_faults, _storage_bytes,
    monitor_summaries,
)


EXTENDED_OPERATIONS = ("hstu_attention_bwd", "hstu_silu_bwd")
OUTPUT_WEIGHT_OPERATION = "hstu_output_weight_mm"
READWRITE_FORMAT = "training_backward_readwrite_v1"


def _silu_signature(grad_output, self, *, grad_input):
    raise AssertionError("Signature placeholder must not execute")


def _output_weight_signature(y, dout):
    raise AssertionError("Signature placeholder must not execute")


def _readwrite_signature(function):
    # torch.library.CustomOpDef.__call__ and AOTI's _PlainFuncWrapper expose
    # only *args/**kwargs. Bind against the actual registered Python function,
    # while still invoking the original public custom-op object exactly once.
    source = getattr(function, "_init_fn", getattr(function, "_func", function))
    signature = inspect.signature(source)
    if any(parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
           for parameter in signature.parameters.values()):
        raise BoundaryProbeError("Cannot establish explicit attention read/write argument signature")
    return signature


def _argument_layouts(arguments):
    """Record live addresses and shared-storage groups without reading bytes."""
    result = _source_specs(arguments)
    groups = {}
    for name, tensor in _tensors(arguments):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata)
        group = groups.setdefault(key, len(groups))
        result[name].update(storage_group=group, storage_bytes=storage.nbytes(),
                            source_data_ptr=tensor.data_ptr(), source_storage_ptr=storage.data_ptr())
    return result


class _AtenProxy:
    def __init__(self, original, probe):
        self._original, self._probe = original, probe

    def __getattr__(self, name):
        return getattr(self._original, name)

    def silu_backward(self, *args, **kwargs):
        return self._probe._silu_backward(self._original.silu_backward, args, kwargs)


class _OpsProxy:
    def __init__(self, original, probe):
        self._original = original
        self.aten = _AtenProxy(original.aten, probe)

    def __getattr__(self, name):
        return getattr(self._original, name)


class _PreprocessTorchProxy:
    def __init__(self, original, probe):
        self._original = original
        self.ops = _OpsProxy(original.ops, probe)

    def __getattr__(self, name):
        return getattr(self._original, name)


class ExtendedTrainingBackwardBoundaryProbe(TrainingBackwardBoundaryProbe):
    _OPERATIONS = {
        **TrainingBackwardBoundaryProbe._OPERATIONS,
        "hstu_attention_bwd": ("_scoped_uvqk_weight", ("dq", "dk", "dv")),
        "hstu_silu_bwd": ("_scoped_uvqk_weight", ("du",)),
        # A temporary Y-pointer entry carries the owner resolved from the
        # saved output weight. Y is an activation input, not a model weight.
        # Neither the signature nor saved arithmetic inputs include a weight.
        OUTPUT_WEIGHT_OPERATION: ("y", ("d_output_weight",)),
    }

    @staticmethod
    def _weight_mm_metadata(event):
        if event.get("operation") != OUTPUT_WEIGHT_OPERATION:
            return event
        return {**event, "read_input_arguments": ["y", "dout"],
                "parameter_role": "layer_attribution_only; output weight is not a matmul input",
                "attribution_source": "saved_output_weight_from_output_backward",
                "y_source": "actual saved Y or fifth normalization return, verified against intercepted transpose"}

    def _emit(self, event):
        super()._emit(self._weight_mm_metadata(event))

    def _save(self, operation, layer, stage, inputs, outputs, event):
        return super()._save(operation, layer, stage, inputs, outputs,
                             self._weight_mm_metadata(event))

    def _install_output_wrappers(self):
        if OUTPUT_WEIGHT_OPERATION not in self.selected_operations:
            return super()._install_output_wrappers()
        function_class = self.output.HSTUComputeOutputFunction
        original_backward = function_class.backward
        original_norm = self.output.triton_layer_norm_mul_dropout_bwd
        norm_signature = inspect.signature(original_norm)
        self._output_mm_signature = inspect.signature(_output_grad_mm_signature)

        @functools.wraps(original_backward)
        def backward(ctx, dout):
            if getattr(self._local, "output_backward", None) is not None:
                raise BoundaryProbeError("Nested HSTU output backward is unsupported")
            if ctx.group_norm:
                raise BoundaryProbeError("Output-weight GEMM diagnostic currently requires layer norm, not group norm")
            scope = {"dout": dout, "output_weight": ctx.saved_tensors[6],
                     "grad_mm_calls": 0, "norm_calls": 0, "dy": None,
                     "weight_mm_calls": 0, "weight_result": None,
                     "recompute_y": ctx.recompute_y_in_backward,
                     "expected_y": None if ctx.recompute_y_in_backward else ctx.saved_tensors[7]}
            self._local.output_backward = scope
            try:
                result = original_backward(ctx, dout)
                if (scope["grad_mm_calls"], scope["norm_calls"], scope["weight_mm_calls"]) != (1, 1, 1):
                    raise BoundaryProbeError("HSTU output backward did not execute both GEMMs and normalization exactly once")
                if len(result) <= 5 or result[5] is not scope["weight_result"]:
                    raise BoundaryProbeError("HSTU output backward did not return its intercepted output-weight gradient")
                return result
            finally:
                self._local.output_backward = None

        @functools.wraps(original_norm)
        def norm_backward(*args, **kwargs):
            scope = getattr(self._local, "output_backward", None)
            if scope is None:
                return original_norm(*args, **kwargs)
            bound = norm_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if (scope["norm_calls"] or scope["weight_mm_calls"]
                    or bound.arguments["dy"] is not scope["dy"]):
                raise BoundaryProbeError("Output layer norm did not consume the captured first GEMM in order")
            scope["norm_calls"] += 1
            result = self._run("triton_layer_norm_mul_dropout_bwd", original_norm,
                               norm_signature, (), dict(bound.arguments))
            if scope["recompute_y"]:
                if not isinstance(result[4], torch.Tensor):
                    raise BoundaryProbeError("Output normalization did not return the recomputed Y")
                scope["expected_y"] = result[4]
            return result

        self._patch(self.output, "torch", _OutputTorchProxy(self.output.torch, self))
        self._patch(self.output, "triton_layer_norm_mul_dropout_bwd", norm_backward)
        self._patch(function_class, "backward", backward)

    def _output_mm(self, function, args, kwargs):
        scope = getattr(self._local, "output_backward", None)
        if (OUTPUT_WEIGHT_OPERATION not in self.selected_operations or scope is None
                or not scope["grad_mm_calls"]):
            return super()._output_mm(function, args, kwargs)
        if scope.get("weight_mm_calls", 0):
            raise BoundaryProbeError("Unexpected additional GEMM in HSTU output backward")
        weight = scope["output_weight"]
        y = scope["expected_y"]
        if (kwargs or len(args) != 2 or not isinstance(args[0], torch.Tensor)
                or args[1] is not scope["dout"] or args[0].ndim != 2 or args[1].ndim != 2
                or args[0].shape[1] != args[1].shape[0]
                or (args[0].shape[0], args[1].shape[1]) != tuple(weight.shape)
                or scope["norm_calls"] != 1 or not isinstance(y, torch.Tensor)):
            raise BoundaryProbeError("Unexpected output-weight-gradient GEMM operands")
        actual_y = args[0].t()
        if (_pointer(actual_y) != _pointer(y)
                or actual_y.untyped_storage()._cdata != y.untyped_storage()._cdata
                or actual_y.storage_offset() != y.storage_offset()
                or actual_y.is_conj() != y.is_conj() or actual_y.is_neg() != y.is_neg()):
            raise BoundaryProbeError("Output-weight GEMM did not consume the actual saved/recomputed Y transpose")
        owner = self._pointers.get(_pointer(weight))
        if owner is None:
            raise BoundaryProbeError("Cannot attribute saved output weight for its gradient GEMM")
        scope["weight_mm_calls"] = 1
        identity = _pointer(y)
        previous = self._pointers.get(identity)
        self._pointers[identity] = {**owner, "mapping": "saved_output_weight_scope"}

        def execute(y, dout):
            # Keep the original transposed operand object and original call.
            # The named Y view exists only for observation and replay inputs.
            return (function(*args, **kwargs),)

        try:
            output = self._run(OUTPUT_WEIGHT_OPERATION, execute,
                               inspect.signature(_output_weight_signature), (),
                               {"y": y, "dout": args[1]})
            scope["weight_result"] = output[0]
            return output[0]
        finally:
            if previous is None:
                self._pointers.pop(identity, None)
            else:
                self._pointers[identity] = previous

    def _install_wrappers(self):
        super()._install_wrappers()
        if not set(self.selected_operations).intersection(EXTENDED_OPERATIONS):
            return
        function_class = self.preprocess._HSTUPreprocessAndAttentionFunction
        original_backward = function_class.backward
        original_attention = self.preprocess.triton_hstu_attention_bwd
        attention_signature = _readwrite_signature(original_attention)
        self._silu_signature = inspect.signature(_silu_signature)

        @functools.wraps(original_backward)
        def backward(ctx, dsilu_u, dout):
            if getattr(self._local, "preprocess_backward", None) is not None:
                raise BoundaryProbeError("Nested preprocessing backward is unsupported")
            # Actual saved forward argument, not a freshly cast model weight.
            weight = ctx.saved_tensors[5]
            owner = self._pointers.get(_pointer(weight))
            if owner is None:
                raise BoundaryProbeError("Cannot attribute preprocessing saved UVQK weight")
            scope = {"owner": owner, "attention_calls": 0, "silu_calls": 0}
            self._local.preprocess_backward = scope
            try:
                result = original_backward(ctx, dsilu_u, dout)
                if (scope["attention_calls"], scope["silu_calls"]) != (1, 1):
                    raise BoundaryProbeError("Preprocessing backward did not execute expected attention/SiLU stages")
                return result
            finally:
                self._local.preprocess_backward = None

        @functools.wraps(original_attention)
        def attention(*args, **kwargs):
            scope = getattr(self._local, "preprocess_backward", None)
            if scope is None:
                return original_attention(*args, **kwargs)
            if scope["attention_calls"] or scope["silu_calls"]:
                raise BoundaryProbeError("Unexpected attention call order in preprocessing backward")
            scope["attention_calls"] += 1
            return self._run_readwrite("hstu_attention_bwd", original_attention,
                                       attention_signature, args, kwargs, ("dq", "dk", "dv"),
                                       scope["owner"])

        self._patch(self.preprocess, "triton_hstu_attention_bwd", attention)
        self._patch(self.preprocess, "torch", _PreprocessTorchProxy(self.preprocess.torch, self))
        self._patch(function_class, "backward", backward)

    def _silu_backward(self, function, args, kwargs):
        scope = getattr(self._local, "preprocess_backward", None)
        if scope is None:
            return function(*args, **kwargs)
        if scope["attention_calls"] != 1 or scope["silu_calls"]:
            raise BoundaryProbeError("Unexpected SiLU call order in preprocessing backward")
        scope["silu_calls"] += 1
        return self._run_readwrite("hstu_silu_bwd", function, self._silu_signature,
                                   args, kwargs, ("grad_input",), scope["owner"])

    def _run_readwrite(self, operation, function, signature, args, kwargs, writes, owner):
        if self.attempt is None or self.closed or self.failed:
            raise BoundaryProbeError("Read/write probe needs a live, nonfailed training attempt")
        self.call += 1
        event = {"call": self.call, "operation": operation, **owner}
        reason = ("operation_filter" if operation not in self.selected_operations else
                  "target_filter" if self.target not in ("*", "all") and self.target not in owner["parameter"] else None)
        if reason is not None:
            self._emit({"event": "skipped", **event, "reason": reason})
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        destinations = tuple(bound.arguments[name] for name in writes)
        if not all(isinstance(tensor, torch.Tensor) for tensor in destinations):
            raise BoundaryProbeError("Read/write destinations must be tensors")
        read_arguments = {name: value for name, value in bound.arguments.items() if name not in writes}
        names = self._OPERATIONS[operation][1]
        before_destination_pointers = tuple(_pointer(tensor) for tensor in destinations)
        event.update(execution_controls=_execution_controls(),
                     argument_roles={"read": list(read_arguments), "write_only": list(writes)},
                     argument_layouts=_argument_layouts(bound.arguments),
                     destination_initial_values_scanned=False)
        argument_bytes = _storage_bytes(bound.arguments)
        destination_bytes = _storage_bytes(destinations)
        required_bytes = argument_bytes + (destination_bytes if self.mode == "pristine" else 0)
        event["capture_budget"] = {
            "argument_unique_storage_bytes": argument_bytes,
            "destination_unique_storage_bytes": destination_bytes,
            "maximum_snapshot_storage_bytes": required_bytes,
            "configured_limit_bytes": self.max_bytes,
            "accounting": ("pre-call arguments plus separate post-call destination storage" if self.mode == "pristine"
                           else "one fault-time union; destination storage is already in arguments"),
        }
        if required_bytes > self.max_bytes:
            self.failed = True
            self._emit({"event": "capture_budget_exceeded", **event})
            raise BoundaryProbeError(
                f"Read/write capture requires {required_bytes} bytes, exceeds remaining capture limit "
                f"{self.max_bytes}; set NAN_BACKWARD_MAX_CAPTURE_GIB explicitly (64 for full-shape extended probes)")
        started = time.monotonic()
        pre = None
        pre_bytes = 0
        if self.mode == "pristine":
            capture_started = time.time()
            pre, pre_bytes = _cpu_copy_tree({"args": args, "kwargs": kwargs}, self.chunk_bytes, self.max_bytes)
            saved = signature.bind(*pre["args"], **pre["kwargs"])
            saved.apply_defaults()
            saved_reads = {name: value for name, value in saved.arguments.items() if name not in writes}
            inputs = _summaries(saved_reads, self.chunk_bytes, _source_specs(read_arguments), self.abs_threshold)
            bad_inputs, extreme_inputs = _snapshot_faults(inputs)
            observation = {"source": "pristine_pre_call_cpu_snapshot",
                           "capture_started_unix_time": capture_started,
                           "capture_completed_unix_time": time.time(), "snapshot_present": True}
        else:
            inputs = monitor_summaries(read_arguments, chunk_bytes=self.chunk_bytes, abs_threshold=self.abs_threshold)
            bad_inputs, extreme_inputs = _monitor_faults(inputs)
            observation = {"source": "pre_call_device_scalar_reductions", "snapshot_present": False}
        event.update(inputs=inputs, input_observation={**observation,
                     "scope": "all floating logical read-input elements; write-only destinations excluded",
                     "source_devices_synchronized_before_observation": True,
                     "scan_completed_unix_time": time.time(), "operation_started": False})
        if self.mode == "monitor_on_anomaly":
            event["input_device_monitor"] = inputs
        self._emit({"event": "before", **event})
        if bad_inputs or extreme_inputs:
            self._finish_readwrite(operation, owner["layer"], "input", signature, args, kwargs, writes,
                                   destinations, event, started, pre, pre_bytes, None, 0,
                                   bad_inputs, extreme_inputs, [], [])

        result = function(*args, **kwargs)
        if before_destination_pointers != tuple(_pointer(tensor) for tensor in destinations):
            raise BoundaryProbeError("Operation resized or changed a write-only destination")
        if operation == "hstu_attention_bwd" and result is not None:
            raise BoundaryProbeError("Attention backward unexpectedly returned values")
        if operation == "hstu_silu_bwd" and (not isinstance(result, torch.Tensor) or _pointer(result) != _pointer(destinations[0])):
            raise BoundaryProbeError("SiLU backward did not return its original out destination")
        named_output = dict(zip(names, destinations))
        post = None
        post_bytes = 0
        if self.mode == "pristine":
            post, post_bytes = _cpu_copy_tree(destinations, self.chunk_bytes, self.max_bytes - pre_bytes)
            outputs = _summaries(dict(zip(names, post)), self.chunk_bytes,
                                 _source_specs(named_output), self.abs_threshold)
            bad_outputs, extreme_outputs = _snapshot_faults(outputs)
        else:
            outputs = monitor_summaries(named_output, chunk_bytes=self.chunk_bytes, abs_threshold=self.abs_threshold)
            bad_outputs, extreme_outputs = _monitor_faults(outputs)
            event["output_device_monitor"] = outputs
        event.update(outputs=outputs, seconds=time.monotonic() - started)
        self._emit({"event": "after", **event})
        if bad_outputs or extreme_outputs or self.save_all:
            self._finish_readwrite(operation, owner["layer"], "output" if bad_outputs or extreme_outputs else "finite",
                                   signature, args, kwargs, writes, destinations, event, started,
                                   pre, pre_bytes, post, post_bytes, [], [], bad_outputs, extreme_outputs)
        return result

    def _finish_readwrite(self, operation, layer, stage, signature, args, kwargs, writes,
                          destinations, event, started, pre, pre_bytes, post, post_bytes,
                          bad_inputs, extreme_inputs, bad_outputs, extreme_outputs):
        if stage != "finite":
            self.failed = True
        pristine = self.mode == "pristine" or stage == "input"
        original_outputs = None if stage == "input" else destinations
        names = self._OPERATIONS[operation][1]
        if pre is None:
            captured_started = time.time()
            copied, total_bytes = _cpu_copy_tree(
                {"inputs": {"args": args, "kwargs": kwargs}, "outputs": original_outputs},
                self.chunk_bytes, self.max_bytes)
            pre, post = copied["inputs"], copied["outputs"]
            accounting = "unique union of fault-time argument and output backing storages"
        else:
            captured_started = event["input_observation"]["capture_started_unix_time"]
            total_bytes = pre_bytes + post_bytes
            accounting = "pre-call argument snapshot plus separate post-call destination snapshot"
        saved = signature.bind(*pre["args"], **pre["kwargs"])
        saved.apply_defaults()
        source = signature.bind(*args, **kwargs)
        source.apply_defaults()
        saved_reads = {name: value for name, value in saved.arguments.items() if name not in writes}
        source_reads = {name: value for name, value in source.arguments.items() if name not in writes}
        input_summary = _summaries(saved_reads, self.chunk_bytes, _source_specs(source_reads), self.abs_threshold)
        output_summary = [] if post is None else _summaries(
            dict(zip(names, post)), self.chunk_bytes, _source_specs(dict(zip(names, destinations))), self.abs_threshold)
        snapshot_bad_inputs, snapshot_extreme_inputs = _snapshot_faults(input_summary)
        snapshot_bad_outputs, snapshot_extreme_outputs = _snapshot_faults(output_summary)
        observed_bad = snapshot_bad_inputs if stage == "input" else snapshot_bad_outputs
        observed_extreme = snapshot_extreme_inputs if stage == "input" else snapshot_extreme_outputs
        trigger_bad = bad_inputs if stage == "input" else bad_outputs
        trigger_extreme = extreme_inputs if stage == "input" else extreme_outputs
        event = {**event, "inputs": input_summary, "outputs": output_summary if post is not None else None,
                 "input_snapshot_pristine": pristine, "initial_destination_bytes_retained": pristine,
                 "capture_storage_bytes": total_bytes, "capture_storage_accounting": accounting,
                 "snapshot_started_unix_time": captured_started, "snapshot_completed_unix_time": time.time(),
                 "bad_inputs": bad_inputs, "extreme_inputs": extreme_inputs,
                 "bad_outputs": bad_outputs, "extreme_outputs": extreme_outputs,
                 "snapshot_bad_inputs": snapshot_bad_inputs, "snapshot_extreme_inputs": snapshot_extreme_inputs,
                 "snapshot_bad_outputs": snapshot_bad_outputs, "snapshot_extreme_outputs": snapshot_extreme_outputs,
                 "trigger_persists_in_cpu_snapshot": set(trigger_bad) <= set(observed_bad) and set(trigger_extreme) <= set(observed_extreme),
                 "operation_executed": stage != "input", "seconds": time.monotonic() - started}
        frame = self.attempt
        target = self.directory / (f"training-readwrite-{self.session}-{frame['mode']}-r{frame['repeat']}"
                                   f"-s{frame['step']}-{self.call:04d}-{operation}-{stage}.pt")
        temporary = target.with_suffix(".pt.tmp")
        payload = {"format": READWRITE_FORMAT, "format_version": 1,
                   "session": self.session, "attempt": frame, "probe_mode": self.mode,
                   "operation": operation, "layer": layer, "stage": stage,
                   "input_snapshot_pristine": pristine, "initial_destination_bytes_retained": pristine,
                   "args": pre["args"], "kwargs": pre["kwargs"], "outputs": post,
                   "execution_controls": event["execution_controls"], "event": event,
                   "note": ("Pre-call read inputs and opaque initial destination storage retained. " if pristine else
                            "Arguments and outputs captured after the call; inputs may have mutated and initial destination bytes are unavailable. ")
                           + "Write-only destinations are excluded from input anomaly scans. Load only as a trusted local artifact."}
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
        if stage != "finite":
            raise BoundaryAnomalyError(target, operation, layer, stage)
