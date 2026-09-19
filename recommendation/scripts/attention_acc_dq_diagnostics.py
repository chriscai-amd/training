"""Isolated DQ-accumulator variants for the attention backward raw runner.

These kernels are diagnostics, not replacement production implementations.
Every returned JIT function owns fresh dependency globals and JIT caches. The
production attention module, autotuner, config and zeroing pre-hook are untouched.
The caller must retain the usual pre-hook, arguments, strides and launch options.

The barrier variants preserve the DQ calculation. load_only, dot_only and
store_zero deliberately change that calculation and require all-zero dOut for
an exact-zero oracle. The overwrite variants can erase a fault from an earlier
key block, and every variant can change compiler scheduling and register layout.
A passing arm therefore does not establish a mechanism by itself.
"""

from __future__ import annotations

import hashlib
from types import FunctionType

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction


VARIANTS = (
    "baseline",
    "barrier_before_load",
    "barrier_after_store",
    "barrier_both",
    "load_only",
    "dot_only",
    "store_zero",
)

_DIAGNOSTIC_MODE = tl.constexpr(0)


@triton.jit
def _diagnostic_acc_dq(
    dq_ptrs_trans,
    start_m,
    stride_dqm,
    k,
    dqk_trans,
    alpha,
    mask_m,
    MAX_SEQ_LEN,
    LOCK,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    tl.static_assert(not ATOMIC_ADD, "DQ diagnostics require non-atomic ownership")
    if _DIAGNOSTIC_MODE == 1 or _DIAGNOSTIC_MODE == 3:
        tl.debug_barrier()
    if _DIAGNOSTIC_MODE == 5:
        dq_trans = tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    elif _DIAGNOSTIC_MODE == 6:
        dq_trans = tl.full((k.shape[1], BLOCK_M), 0, tl.float32)
    else:
        dq_trans = tl.load(
            dq_ptrs_trans + start_m * stride_dqm,
            mask=mask_m[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        if _DIAGNOSTIC_MODE != 4:
            dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    dq_trans = dq_trans.to(k.dtype)
    tl.store(
        dq_ptrs_trans + start_m * stride_dqm,
        dq_trans,
        mask=mask_m[None, :],
        eviction_policy="evict_last",
    )
    if _DIAGNOSTIC_MODE == 2 or _DIAGNOSTIC_MODE == 3:
        tl.debug_barrier()


def _clone_jit(original: JITFunction, overrides: dict, variant: str) -> JITFunction:
    """Copy globals instead of modifying globals already captured by Triton."""
    namespace = dict(original.fn.__globals__)
    namespace.update(overrides)
    fn = FunctionType(
        original.fn.__code__, namespace, original.fn.__name__,
        original.fn.__defaults__, original.fn.__closure__,
    )
    fn.__annotations__ = dict(original.fn.__annotations__)
    fn.__kwdefaults__ = original.fn.__kwdefaults__
    # Avoid replacing production entries in Triton's JIT deserialization registry.
    # Keep source and code locations intact: baseline can reuse the exact binary.
    fn.__module__ = f"{__name__}.{variant}"
    fn.__qualname__ = original.fn.__qualname__
    result = JITFunction(
        fn,
        version=original.version,
        do_not_specialize=list(original.do_not_specialize),
        do_not_specialize_on_alignment=list(original.do_not_specialize_on_alignment),
        debug=original.debug,
        noinline=original.noinline,
        repr=original._repr,
        launch_metadata=original.launch_metadata,
    )
    if result.src != original.src:
        raise RuntimeError("diagnostic clone does not preserve the original JIT source")
    return result


def _raw_jit(kernel) -> JITFunction:
    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    return kernel


def make_diagnostic_kernel(attention, variant: str) -> JITFunction:
    """Return a raw kernel with the production signature and separate JIT state.

    ``baseline`` clones the unmodified dependency chain; its cache key should
    equal production's. Other variants change only the accumulator dependency.
    No device is initialized and no kernel is compiled or launched here.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown DQ diagnostic variant: {variant!r}")
    raw = _raw_jit(attention._hstu_attn_bwd)
    one_block = attention._hstu_attn_bwd_one_block
    one_col = attention._hstu_attn_bwd_one_col_block
    production_acc = one_block.fn.__globals__["acc_dq"]
    if variant == "baseline":
        acc = _clone_jit(production_acc, {}, variant)
    else:
        acc = _clone_jit(
            _diagnostic_acc_dq,
            {"_DIAGNOSTIC_MODE": tl.constexpr(VARIANTS.index(variant))},
            variant,
        )
    block = _clone_jit(one_block, {"acc_dq": acc}, variant)
    col = _clone_jit(one_col, {"_hstu_attn_bwd_one_block": block}, variant)
    kernel = _clone_jit(raw, {"_hstu_attn_bwd_one_col_block": col}, variant)
    kernel.diagnostic_metadata = {
        "variant": variant,
        "preserves_dq_calculation": variant in VARIANTS[:4],
        "requires_all_zero_dout": variant in VARIANTS[4:],
        "caller_must_zero_dq": True,
        "requires_sequence_parallel_false": True,
        "production_jit_cache_key": raw.cache_key,
        "diagnostic_jit_cache_key": kernel.cache_key,
        "baseline_cache_key_matches": kernel.cache_key == raw.cache_key,
        "dependencies": {
            name: {
                "jit_cache_key": fn.cache_key,
                "source_file": fn.file_name,
                "source_sha256": hashlib.sha256(fn.src.encode()).hexdigest(),
            }
            for name, fn in (
                ("acc_dq", acc), ("one_block", block),
                ("one_col_block", col), ("kernel", kernel),
            )
        },
    }
    if variant == "baseline" and kernel.cache_key != raw.cache_key:
        raise RuntimeError("baseline clone unexpectedly changes the JIT cache key")
    return kernel
