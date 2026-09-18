"""Opt-in, phase-batched non-finite diagnostics for custom Triton autograd ops.

Set NAN_TRIPWIRE_DIR to enable. Call install_from_env after model imports, then
begin/watch/check/end around a training step. Only check transfers finite flags
to the host. No module hooks or embedding-table sweeps are installed.

Tensor references are retained until check, so this changes allocation lifetimes
and can affect a race. A capture contains storage contents at the phase boundary,
not a clone taken before each operation. The report records version changes and
the finite flags measured before/after the operation. A clean instrumented run
does not exonerate the uninstrumented path.

NAN_TRIPWIRE_RETAIN_PAYLOAD=0 keeps only scalar flags and metadata, releasing
operation inputs/outputs normally. This mode localizes failures but cannot replay
them. It permits checking once after backward without retaining forward buffers.
With this mode, NAN_TRIPWIRE_DEFER_STEPS>1 also preserves pending events across
begin calls. The training loop must check at its chosen existing synchronization
boundary and call end only after that check; end explicitly discards pending data.

NAN_TRIPWIRE_CAPTURE_STEP plus NAN_TRIPWIRE_CAPTURE_FUNCTION requests one real
custom-op snapshot even when finite. CAPTURE_FUNCTION is a regular expression;
NAN_TRIPWIRE_CAPTURE_DIRECTION defaults to backward. This requires payload
retention. The first matching event is saved at check and raises CaptureComplete.

NAN_TRIPWIRE_FINITE_CHUNK_ELEMENTS optionally bounds each isfinite/all input.
Zero (the default) keeps the original whole-tensor check. Positive values split
large tensors into views and combine their flags on device without host reads.

NAN_TRIPWIRE_MAX_ABS optionally detects finite values whose magnitude exceeds a
positive finite threshold. Magnitude failures have their own flags and reason;
the finite flags retain their original meaning. Magnitude checks use bounded
chunks even when finite chunking is disabled. Unset or empty disables the bound.

NAN_TRIPWIRE_POISON_CONTEXTUAL_SPLIT_B=1 prefills only active dense-A=8 / jagged-B,
zero-prefix split B outputs with NaN immediately before the original copy kernel.
This diagnostic tests whether the split overwrites every B element. It changes
memory traffic and allocation contents; it is not a production correction.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODULES = (
    "triton_hstu_preprocess_and_attention",
    "triton_hstu_linear",
    "triton_hstu_attention",
    "triton_layer_norm",
    "triton_position",
    "triton_jagged",
    "triton_jagged_tensors",
)


class NonFiniteError(RuntimeError):
    """A diagnostic phase observed a non-finite tensor and wrote its report."""


class MagnitudeError(RuntimeError):
    """A diagnostic phase observed values beyond its configured magnitude bound."""


class CaptureComplete(RuntimeError):
    """A requested finite diagnostic snapshot was saved; the caller may stop."""

    def __init__(self, report_path: Path, capture_path: Path) -> None:
        super().__init__(
            f"Requested diagnostic capture complete: report={report_path}; capture={capture_path}"
        )
        self.report_path = report_path
        self.capture_path = capture_path


def _tensors(value: Any, path: str = ""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _tensors(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _tensors(child, f"{path}[{index}]")


def _detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detach(child) for child in value)
    return value


def _tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "storage_bytes": tensor.untyped_storage().nbytes(),
        # Attention backward can allocate gradients in inference_mode. These
        # tensors deliberately have no version counter; reading _version raises.
        "version": None if tensor.is_inference() else tensor._version,
    }


def _metadata_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_meta(value)
    if isinstance(value, dict):
        return {key: _metadata_tree(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_metadata_tree(child) for child in value)
    return value


def _finite_chunks(tensor: torch.Tensor, max_elements: int):
    """Yield bounded views, including for transposes and expanded tensors."""
    if tensor.numel() <= max_elements:
        yield tensor
        return
    dimension = max(range(tensor.ndim), key=tensor.size)
    elements_per_slice = tensor.numel() // tensor.size(dimension)
    width = max(1, max_elements // elements_per_slice)
    for start in range(0, tensor.size(dimension), width):
        chunk = tensor.narrow(dimension, start, min(width, tensor.size(dimension) - start))
        # A single slice can still exceed the bound; split another dimension.
        yield from _finite_chunks(chunk, max_elements)


def _all_finite(tensor: torch.Tensor, chunk_elements: int) -> torch.Tensor:
    if chunk_elements == 0 or tensor.numel() <= chunk_elements:
        return torch.isfinite(tensor).all()
    result = None
    for chunk in _finite_chunks(tensor, chunk_elements):
        flag = torch.isfinite(chunk).all()
        result = flag if result is None else torch.logical_and(result, flag)
    return result


def _all_within_bound(
    tensor: torch.Tensor, max_abs: float, chunk_elements: int
) -> torch.Tensor:
    result = None
    for chunk in _finite_chunks(tensor, chunk_elements or 16 * 1024**2):
        # Promote low-precision inputs so the threshold is not rounded to BF16
        # or FP16. No host readback or full-tensor temporary is needed.
        values = chunk.float() if chunk.dtype in (torch.float16, torch.bfloat16) else chunk
        flag = (values.abs() <= max_abs).all()
        result = flag if result is None else torch.logical_and(result, flag)
    return result


def _capture_runtime_state() -> dict[str, Any]:
    """Small process settings which affect dispatch or kernel specialization."""
    common = sys.modules.get("generative_recommenders.common")
    common_state = None
    if common is not None:
        common_state = {
            "STATIC_MAX_SEQ_LENS": list(common.STATIC_MAX_SEQ_LENS),
            "USE_RUNTIME_MAX_SEQ_LEN": common.USE_RUNTIME_MAX_SEQ_LEN,
            "BACKEND_ALLOW_TF32": common.BACKEND_ALLOW_TF32,
        }
    matmul = torch.backends.cuda.matmul
    return {
        "common": common_state,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul": {
            name: getattr(matmul, name)
            for name in (
                "allow_tf32",
                "allow_fp16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction",
                "allow_fp16_accumulation",
            )
            if hasattr(matmul, name)
        },
    }


class NaNTripwire:
    def __init__(
        self,
        directory: str | Path,
        *,
        start_step: int = 0,
        end_step: int | None = None,
        function_pattern: str = ".*",
        capture_limit_bytes: int = 24 * 1024**3,
        retain_payload: bool = True,
        defer_steps: int = 1,
        capture_step: int | None = None,
        capture_function_pattern: str = ".*",
        capture_direction: str = "backward",
        finite_chunk_elements: int = 0,
        max_abs: float | None = None,
        poison_contextual_split_b: bool = False,
    ) -> None:
        if defer_steps < 1:
            raise ValueError("NAN_TRIPWIRE_DEFER_STEPS must be positive")
        if defer_steps > 1 and retain_payload:
            raise ValueError("Deferred checks require NAN_TRIPWIRE_RETAIN_PAYLOAD=0")
        if capture_step is not None:
            if not retain_payload:
                raise ValueError("Requested capture requires NAN_TRIPWIRE_RETAIN_PAYLOAD=1")
            if capture_step < start_step or (end_step is not None and capture_step > end_step):
                raise ValueError("NAN_TRIPWIRE_CAPTURE_STEP must lie within the active step range")
        if capture_direction not in ("forward", "backward"):
            raise ValueError("NAN_TRIPWIRE_CAPTURE_DIRECTION must be forward or backward")
        if not isinstance(finite_chunk_elements, int) or finite_chunk_elements < 0:
            raise ValueError("NAN_TRIPWIRE_FINITE_CHUNK_ELEMENTS must be a nonnegative integer")
        if max_abs is not None:
            if isinstance(max_abs, bool) or not isinstance(max_abs, (int, float)) or not math.isfinite(max_abs) or max_abs <= 0:
                raise ValueError("NAN_TRIPWIRE_MAX_ABS must be a positive finite number")
            max_abs = float(max_abs)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.start_step = start_step
        self.end_step = end_step
        self.function_pattern = re.compile(function_pattern)
        self.capture_limit_bytes = capture_limit_bytes
        self.retain_payload = retain_payload
        self.defer_steps = defer_steps
        self.capture_step = capture_step
        self.capture_function_pattern = re.compile(capture_function_pattern)
        self.capture_direction = capture_direction
        self.finite_chunk_elements = finite_chunk_elements
        self.max_abs = max_abs
        self.poison_contextual_split_b = poison_contextual_split_b
        self.poison_contextual_split_b_calls = 0
        self._split_poison_handle: tuple[Any, str, Any] | None = None
        self.active = False
        self.step = -1
        self.metadata: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.handles: list[tuple[type, str, Any]] = []
        self.sequence = 0

    def begin(self, step: int, **metadata: Any) -> None:
        if self.defer_steps == 1:
            self.events.clear()
        if not self.events:
            self.sequence = 0
        self.step = step
        self.metadata = metadata
        self.active = step >= self.start_step and (
            self.end_step is None or step <= self.end_step
        )

    def end(self) -> None:
        self.events.clear()
        self.active = False

    def _observe(self, values: Any) -> tuple[list[Any], list[Any], list[Any]]:
        flags, bound_flags, metadata = [], [], []
        with torch.no_grad():
            for name, tensor in _tensors(values):
                if tensor.layout != torch.strided:
                    raise ValueError(f"tripwire needs a strided tensor: {name}")
                info = {"name": name, **_tensor_meta(tensor)}
                if (tensor.is_floating_point() or tensor.is_complex()) and tensor.numel():
                    info["flag_index"] = len(flags)
                    flags.append(_all_finite(tensor, self.finite_chunk_elements))
                    if self.max_abs is not None:
                        bound_flags.append(_all_within_bound(tensor, self.max_abs, self.finite_chunk_elements))
                metadata.append(info)
        return flags, bound_flags, metadata

    def _record(
        self,
        name: str,
        direction: str,
        inputs: Any,
        outputs: Any,
        input_observation: tuple[list[Any], list[Any], list[Any]],
        replay: dict[str, Any] | None = None,
    ) -> None:
        output_flags, output_bound_flags, output_metadata = self._observe(outputs)
        input_flags, input_bound_flags, input_metadata = input_observation
        self.events.append(
            {
                "sequence": self.sequence,
                "step": self.step,
                "metadata": dict(self.metadata),
                "name": name,
                "direction": direction,
                "input_flags": input_flags,
                "output_flags": output_flags,
                "input_bound_flags": input_bound_flags,
                "output_bound_flags": output_bound_flags,
                "inputs": input_metadata,
                "outputs": output_metadata,
                "payload": (
                    _detach({"inputs": inputs, "outputs": outputs, "replay": replay})
                    if self.retain_payload else None
                ),
                "replay_metadata": (
                    _metadata_tree(replay) if not self.retain_payload else None
                ),
            }
        )
        self.sequence += 1

    def watch(self, name: str, value: Any) -> None:
        if self.active:
            self._record(name, "observation", {}, value, ([], [], []))

    def patch_function(self, cls: type[torch.autograd.Function]) -> None:
        """Patch one class; exposed separately for CPU tests and narrow probes."""
        name = f"{cls.__module__}.{cls.__name__}"
        if not self.function_pattern.search(name):
            return
        for direction in ("forward", "backward"):
            original = getattr(cls, direction)
            if getattr(original, "_nan_tripwire", False):
                continue
            signature = inspect.signature(original)

            @functools.wraps(original)
            def wrapped(
                ctx,
                *args,
                _fn=original,
                _direction=direction,
                _signature=signature,
                **kwargs,
            ):
                if not self.active:
                    return _fn(ctx, *args, **kwargs)
                bound = _signature.bind(ctx, *args, **kwargs)
                named_args = dict(bound.arguments)
                named_args.pop(next(iter(_signature.parameters)), None)
                context = dict(ctx.__dict__)
                saved = tuple(ctx.saved_tensors) if _direction == "backward" else ()
                inputs = {"arguments": named_args, "saved_tensors": saved}
                input_observation = self._observe(inputs)
                replay = {
                    "module": cls.__module__,
                    "class": cls.__name__,
                    "direction": _direction,
                    "args": args,
                    "kwargs": kwargs,
                    "ctx": context,
                    "saved_tensors": saved,
                    "needs_input_grad": tuple(ctx.needs_input_grad),
                    "cuda_autocast_enabled": torch.is_autocast_enabled(),
                    "cuda_autocast_dtype": torch.get_autocast_dtype("cuda"),
                    "runtime_state": _capture_runtime_state(),
                }
                # get_rng_state does not consume RNG or synchronize a launch.
                if self.retain_payload:
                    replay["torch_rng_state"] = torch.get_rng_state()
                    replay["python_rng_state"] = random.getstate()
                    if torch.cuda.is_initialized():
                        replay["cuda_rng_state"] = torch.cuda.get_rng_state()
                result = _fn(ctx, *args, **kwargs)
                if _direction == "forward":
                    replay["ctx_after_forward"] = dict(ctx.__dict__)
                self._record(
                    name, _direction, inputs, result, input_observation, replay
                )
                return result

            wrapped._nan_tripwire = True
            self.handles.append((cls, direction, original))
            setattr(cls, direction, staticmethod(wrapped))
        logger.info("[nan-tripwire] instrumented %s", name)

    def _install_contextual_split_b_poison(self) -> None:
        if not self.poison_contextual_split_b or self._split_poison_handle is not None:
            return
        module_name = "generative_recommenders.ops.triton.triton_jagged_tensors"
        module = sys.modules.get(module_name)
        name = "_triton_split_2D_jagged_internal"
        original = getattr(module, name, None)
        if original is None:
            raise RuntimeError("Contextual split B poison requires triton_jagged_tensors to be imported before install")
        if getattr(original, "_nan_tripwire_split_poison", False):
            raise RuntimeError("Contextual split B poison is already owned by another tripwire")
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if self.active:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                out_b = values["out_b"]
                if (
                    values["is_dense_a"] is True
                    and values["is_dense_b"] is False
                    and values["max_len_a"] == 8
                    and values["n_prefix_to_B"] == 0
                    and isinstance(out_b, torch.Tensor)
                    and out_b.is_floating_point()
                ):
                    with torch.no_grad():
                        out_b.fill_(float("nan"))
                    self.poison_contextual_split_b_calls += 1
                    if self.poison_contextual_split_b_calls == 1:
                        logger.warning("[nan-tripwire] contextual split B NaN prefill first applied at step %d", self.step)
            return original(*args, **kwargs)

        wrapped._nan_tripwire_split_poison = True
        self._split_poison_handle = (module, name, original)
        setattr(module, name, wrapped)

    def install(self) -> None:
        self._install_contextual_split_b_poison()
        prefix = "generative_recommenders.ops.triton."
        for module_name, module in list(sys.modules.items()):
            if module_name not in {prefix + name for name in _DEFAULT_MODULES}:
                continue
            if module is None:
                continue
            for value in vars(module).values():
                if (
                    isinstance(value, type)
                    and value.__module__ == module_name
                    and issubclass(value, torch.autograd.Function)
                ):
                    self.patch_function(value)
        logger.warning(
            "[nan-tripwire] enabled at step %d..%s; %d functions; retain_payload=%s; defer_steps=%d; finite_chunk_elements=%d; max_abs=%s; poison_contextual_split_b=%s; capture limit %.1f GiB",
            self.start_step,
            self.end_step,
            len(self.handles) // 2,
            self.retain_payload,
            self.defer_steps,
            self.finite_chunk_elements,
            self.max_abs,
            self.poison_contextual_split_b,
            self.capture_limit_bytes / 1024**3,
        )

    def uninstall(self) -> None:
        self.end()
        for cls, direction, original in reversed(self.handles):
            setattr(cls, direction, staticmethod(original))
        self.handles.clear()
        if self._split_poison_handle is not None:
            module, name, original = self._split_poison_handle
            setattr(module, name, original)
            self._split_poison_handle = None

    def check(self, phase: str) -> None:
        """One flag transfer per device; raise before caller's next phase on failure."""
        if not self.events:
            return
        flags_by_device: dict[torch.device, list[torch.Tensor]] = {}
        locations: dict[torch.device, list[tuple[dict[str, Any], str, int]]] = {}
        for event in self.events:
            for direction in ("input", "output"):
                for flag_suffix, result_suffix in (("flags", "finite"), ("bound_flags", "within_bound")):
                    flags = event[f"{direction}_{flag_suffix}"]
                    field = f"{direction}_{result_suffix}"
                    event[field] = [None] * len(flags)
                    for index, flag in enumerate(flags):
                        flags_by_device.setdefault(flag.device, []).append(flag)
                        locations.setdefault(flag.device, []).append((event, field, index))
        for device, flags in flags_by_device.items():
            values = torch.stack(flags).to(device="cpu").tolist()
            for (event, field, index), value in zip(locations[device], values):
                event[field][index] = value
        # Earliest observed bad output, including propagated input contamination.
        # Reports distinguish that from an op which received finite inputs.
        bad = None
        failure_side = None
        for side in ("output", "input"):
            # Output failures take precedence; input-only contamination remains
            # visible when every observed output passes both checks.
            bad = next(
                (event for event in self.events
                 if not all(event[f"{side}_finite"]) or not all(event[f"{side}_within_bound"])),
                None,
            )
            if bad is not None:
                failure_side = side
                break
        if bad is not None:
            nonfinite = not all(bad[f"{failure_side}_finite"])
            reason = "nonfinite" if nonfinite else "magnitude"
            report_path = self._dump(phase, bad, reason=reason, failure_side=failure_side)
            self.end()
            error = NonFiniteError if nonfinite else MagnitudeError
            description = "non-finite" if nonfinite else f"magnitude exceeds NAN_TRIPWIRE_MAX_ABS={self.max_abs}"
            raise error(
                f"step={bad['step']} check_step={self.step} phase={phase}: {description} at {bad['name']} "
                f"({bad['direction']}); report={report_path}"
            )
        if self.capture_step is not None:
            selected = next(
                (
                    event for event in self.events
                    if event["step"] == self.capture_step
                    and event["direction"] == self.capture_direction
                    and self.capture_function_pattern.search(event["name"])
                ),
                None,
            )
            if selected is not None:
                report_path = self._dump(phase, selected, reason="requested_capture")
                self.end()
                report = json.loads(report_path.read_text())
                if "capture_path" not in report:
                    raise RuntimeError(
                        f"Requested diagnostic capture failed: {report.get('capture_error', 'no payload')}; "
                        f"report={report_path}"
                    )
                raise CaptureComplete(report_path, Path(report["capture_path"]))
        self.events.clear()

    def _dump(
        self, phase: str, selected: dict[str, Any], *, reason: str = "nonfinite",
        failure_side: str | None = None,
    ) -> Path:
        rank = os.environ.get("RANK", "0")
        suffix = "_requested_capture" if reason == "requested_capture" else ""
        stem = f"step{selected['step']:06d}_{phase}{suffix}_rank{rank}_pid{os.getpid()}"
        report_path = self.directory / f"{stem}.json"
        capture_path = self.directory / f"{stem}.pt"
        if selected["payload"] is not None:
            for field in ("inputs", "outputs"):
                current_tensors = dict(_tensors(selected["payload"][field]))
                for info in selected[field]:
                    tensor = current_tensors[info["name"]]
                    current_version = None if tensor.is_inference() else tensor._version
                    info["version_at_capture"] = current_version
                    info["version_changed"] = (
                        info["version"] != current_version
                        if info["version"] is not None and current_version is not None
                        else None
                    )
        report = {
            "step": selected["step"],
            "check_step": self.step,
            "phase": phase,
            "reason": reason,
            "failure_side": failure_side,
            "metadata": selected["metadata"],
            "check_metadata": self.metadata,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "environment": {
                name: value
                for name, value in os.environ.items()
                if name.startswith(("TRITON_", "AMDGCN_", "HSTU_", "NAN_TRIPWIRE_"))
                or name in (
                    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                    "AMD_SERIALIZE_KERNEL", "AMD_LOG_LEVEL", "HSA_ENABLE_COREDUMP",
                )
            },
            "selected_sequence": selected["sequence"],
            "first_bad_sequence": selected["sequence"] if reason != "requested_capture" else None,
            "all_inputs_finite": all(selected["input_finite"]),
            "all_outputs_finite": all(selected["output_finite"]),
            "all_inputs_within_bound": all(selected["input_within_bound"]) if self.max_abs is not None else None,
            "all_outputs_within_bound": all(selected["output_within_bound"]) if self.max_abs is not None else None,
            "retain_payload": self.retain_payload,
            "defer_steps": self.defer_steps,
            "finite_chunk_elements": self.finite_chunk_elements,
            "max_abs": self.max_abs,
            "poison_contextual_split_b": self.poison_contextual_split_b,
            "poison_contextual_split_b_calls": self.poison_contextual_split_b_calls,
            "capture_timing": (
                "references retained; storage copied at phase boundary"
                if self.retain_payload else "payload retention disabled; metadata only"
            ),
            "events": [
                {
                    key: value
                    for key, value in event.items()
                    if key not in ("payload", "input_flags", "output_flags", "input_bound_flags", "output_bound_flags")
                }
                for event in self.events
            ],
        }
        # Always persist the compact evidence, even if copying a large tensor fails.
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        log_level = logging.INFO if reason == "requested_capture" else logging.ERROR
        logger.log(
            log_level,
            "[nan-tripwire] step=%d phase=%s reason=%s selected=%s direction=%s finite_inputs=%s report=%s",
            selected["step"], phase, reason, selected["name"], selected["direction"],
            report["all_inputs_finite"], report_path,
        )
        if selected["payload"] is None:
            return report_path
        storages: dict[tuple[str, int], torch.Tensor] = {}
        total_bytes = 0
        for _, tensor in _tensors(selected["payload"]):
            key = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if key not in storages:
                storages[key] = tensor
                total_bytes += tensor.untyped_storage().nbytes()
        report["capture_storage_bytes"] = total_bytes
        if total_bytes > self.capture_limit_bytes:
            report["capture_error"] = (
                f"{total_bytes} bytes exceeds NAN_TRIPWIRE_CAPTURE_GIB; "
                "increase the limit or narrow NAN_TRIPWIRE_FUNCTIONS"
            )
        else:
            try:
                storage_ids = {key: index for index, key in enumerate(storages)}
                cpu_storages = {}
                for key, tensor in storages.items():
                    storage = tensor.untyped_storage()
                    raw = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    raw = raw.set_(storage, 0, (storage.nbytes(),), (1,))
                    cpu_storages[storage_ids[key]] = raw.cpu().clone() if raw.device.type == "cpu" else raw.cpu()

                def encode(value: Any) -> Any:
                    if isinstance(value, torch.Tensor):
                        key = (str(value.device), value.untyped_storage().data_ptr())
                        return {
                            "__tripwire_tensor__": storage_ids[key],
                            **_tensor_meta(value),
                        }
                    if isinstance(value, dict):
                        return {key: encode(child) for key, child in value.items()}
                    if isinstance(value, tuple):
                        return tuple(encode(child) for child in value)
                    if isinstance(value, list):
                        return [encode(child) for child in value]
                    return value

                torch.save(
                    {"format_version": 1, "report": report,
                     "payload": encode(selected["payload"]), "storages": cpu_storages},
                    capture_path,
                )
                report["capture_path"] = str(capture_path)
                logger.log(log_level, "[nan-tripwire] captured %.2f GiB -> %s", total_bytes / 1024**3, capture_path)
            except Exception as exc:
                report["capture_error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("[nan-tripwire] tensor capture failed")
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        return report_path


def install_from_env() -> NaNTripwire | None:
    directory = os.environ.get("NAN_TRIPWIRE_DIR", "")
    if not directory:
        return None
    end_step = os.environ.get("NAN_TRIPWIRE_END_STEP", "")
    capture_step = os.environ.get("NAN_TRIPWIRE_CAPTURE_STEP", "")
    max_abs = os.environ.get("NAN_TRIPWIRE_MAX_ABS", "")
    tripwire = NaNTripwire(
        directory,
        start_step=int(os.environ.get("NAN_TRIPWIRE_START_STEP", "0")),
        end_step=int(end_step) if end_step else None,
        function_pattern=os.environ.get("NAN_TRIPWIRE_FUNCTIONS", ".*"),
        capture_limit_bytes=int(float(os.environ.get("NAN_TRIPWIRE_CAPTURE_GIB", "24")) * 1024**3),
        retain_payload=os.environ.get("NAN_TRIPWIRE_RETAIN_PAYLOAD", "1") != "0",
        defer_steps=int(os.environ.get("NAN_TRIPWIRE_DEFER_STEPS", "1")),
        capture_step=int(capture_step) if capture_step else None,
        capture_function_pattern=os.environ.get("NAN_TRIPWIRE_CAPTURE_FUNCTION", ".*"),
        capture_direction=os.environ.get("NAN_TRIPWIRE_CAPTURE_DIRECTION", "backward"),
        finite_chunk_elements=int(os.environ.get("NAN_TRIPWIRE_FINITE_CHUNK_ELEMENTS", "0")),
        max_abs=float(max_abs) if max_abs else None,
        poison_contextual_split_b=os.environ.get("NAN_TRIPWIRE_POISON_CONTEXTUAL_SPLIT_B", "0") == "1",
    )
    tripwire.install()
    return tripwire
