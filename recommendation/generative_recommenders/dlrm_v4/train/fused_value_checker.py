"""Optional tripwire checker proposal; None means use the existing Torch path.

No Triton import or GPU probing occurs at module import. try_check launches only
when enabled and given a supported CUDA/ROCm tensor. It performs no host tensor
readback or synchronization. Compiler/launch errors propagate; they are not
silently retried through Torch after a potentially asynchronous device failure.
"""

from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import math

import torch


BLOCK = 4096
MAX_I64 = 2**63 - 1
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def layout_metadata(shape, strides, storage_offset, itemsize, storage_bytes):
    """Host metadata only; bounds are in elements relative to the view pointer."""
    shape, strides = tuple(shape), tuple(strides)
    if len(shape) != len(strides) or any(type(n) is not int or n < 0 for n in (*shape, *strides)):
        raise ValueError("shape and strides must be nonnegative concrete integers")
    numel = math.prod(shape)
    blocks = (numel + BLOCK - 1) // BLOCK
    last = 0 if not numel else sum((n - 1) * s for n, s in zip(shape, strides))
    if storage_offset < 0 or max(storage_offset, last, numel, blocks * BLOCK) > MAX_I64:
        raise ValueError("layout exceeds signed 64-bit indexing")
    if blocks > 2**31 - 1:
        raise ValueError("layout exceeds conservative grid-x limit")
    end_bytes = (storage_offset + last + (1 if numel else 0)) * itemsize
    if end_bytes > storage_bytes or end_bytes > MAX_I64:
        raise ValueError("layout addresses bytes beyond storage or signed 64-bit range")
    return dict(shape=shape, strides=strides, numel=numel, blocks=blocks,
                storage_offset=storage_offset, maximum_relative_element_offset=last,
                status_bytes=blocks)


def reduce_status(status, has_bound):
    """Bit 0=nonfinite, bit 1=abs>bound. NaN must also fail within-bound."""
    if not status.numel():
        return torch.ones((), dtype=torch.bool, device=status.device), (
            torch.ones((), dtype=torch.bool, device=status.device) if has_bound else None)
    finite = (status & 1).amax() == 0
    within_bound = (status.amax() == 0) if has_bound else None
    return finite, within_bound


def json_metadata(value):
    if is_dataclass(value) and not isinstance(value, type):
        return json_metadata(asdict(value))
    if hasattr(value, "_asdict"):
        return json_metadata(value._asdict())
    if isinstance(value, dict):
        return {str(key): json_metadata(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metadata(child) for child in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def unsupported_reason(tensor):
    if tensor.layout != torch.strided:
        return "non_strided"
    if tensor.dtype not in SUPPORTED_DTYPES:
        return "unsupported_dtype"
    if tensor.device.type != "cuda":
        return "non_gpu"
    if tensor.is_conj() or tensor.is_neg():
        return "lazy_view_bits"
    return None


class FusedValueChecker:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.calls = 0
        self.empty_calls = 0
        self.fallbacks = Counter()
        self.compiled = {}
        self.last_launch = None
        self._kernel = None
        self._kernel_source_sha256 = None

    def _load_kernel(self):
        if self._kernel is None:
            name = (f"{__package__}." if __package__ else "") + "fused_value_checker_kernel"
            self._kernel = importlib.import_module(name).check_value_blocks
            self._kernel_source_sha256 = hashlib.sha256(self._kernel.src.encode()).hexdigest()
        return self._kernel

    def try_check(self, tensor, max_abs=None):
        if not self.enabled:
            return None
        if max_abs is not None and (isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
                                    or not math.isfinite(max_abs) or max_abs <= 0):
            raise ValueError("max_abs must be a positive finite number or None")
        reason = unsupported_reason(tensor)
        if reason:
            self.fallbacks[reason] += 1
            return None
        if max_abs is not None and tensor.dtype != torch.float64 and max_abs > torch.finfo(torch.float32).max:
            # Torch casts an overflowing FP32 comparison scalar to Inf, making
            # Inf <= threshold true. Keep that existing bound-flag behavior.
            self.fallbacks["threshold_outside_compute_dtype"] += 1
            return None
        try:
            layout = layout_metadata(tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
                                     tensor.element_size(), tensor.untyped_storage().nbytes())
        except ValueError:
            self.fallbacks["unsupported_layout"] += 1
            return None
        self.calls += 1
        if not layout["numel"]:
            self.empty_calls += 1
            return torch.ones((), dtype=torch.bool, device=tensor.device), (
                torch.ones((), dtype=torch.bool, device=tensor.device) if max_abs is not None else None)
        status = torch.empty((layout["blocks"],), dtype=torch.uint8, device=tensor.device)
        kernel = self._load_kernel()
        compiled = kernel[(layout["blocks"],)](
            tensor, status, layout["numel"], layout["shape"], layout["strides"],
            HAS_BOUND=max_abs is not None, MAX_ABS=0.0 if max_abs is None else float(max_abs),
            IS_FP64=tensor.dtype == torch.float64, BLOCK=BLOCK, num_warps=4, num_stages=1)
        # These are ordinary Python attributes on the exact launched kernel.
        # Do not call launch_metadata(), warmup(), device_caches, or synchronize.
        identity = str(compiled.hash)
        if identity not in self.compiled:
            self.compiled[identity] = dict(hash=identity, name=compiled.name,
                                           metadata=json_metadata(compiled.metadata))
        self.last_launch = dict(**layout, dtype=str(tensor.dtype), device=str(tensor.device),
                                compiled_hash=identity, max_abs=max_abs)
        return reduce_status(status, max_abs is not None)

    def metadata(self):
        return dict(enabled=self.enabled, calls=self.calls, empty_calls=self.empty_calls,
                    fallbacks=dict(self.fallbacks), block=BLOCK, num_warps=4, num_stages=1,
                    packed_bits={"0": "nonfinite", "1": "abs > threshold"},
                    kernel_jit_source_sha256=self._kernel_source_sha256,
                    compiled=list(self.compiled.values()), last_launch=json_metadata(self.last_launch),
                    hash_meaning="Triton compiled cache-key identity; not a binary digest")
