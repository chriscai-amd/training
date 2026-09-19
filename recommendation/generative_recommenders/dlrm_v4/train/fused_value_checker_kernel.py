"""Imported only for an enabled, supported, nonempty GPU check."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Numel"])
def check_value_blocks(
    Values, Status, Numel, Shape, Strides,
    HAS_BOUND: tl.constexpr, MAX_ABS: tl.constexpr, IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    flat = tl.cast(block, tl.int64) * BLOCK + tl.cast(tl.arange(0, BLOCK), tl.int64)
    valid = flat < tl.cast(Numel, tl.int64)
    quotient = flat
    offset = tl.full((BLOCK,), 0, tl.int64)
    # Tuple leaves are runtime scalars, except Triton's coarse specialization
    # (notably literal 1). tl.cast works for both scalars and constexpr leaves.
    for dim in tl.static_range(len(Shape) - 1, -1, -1):
        size = tl.cast(Shape[dim], tl.int64)
        stride = tl.cast(Strides[dim], tl.int64)
        offset += (quotient % size) * stride
        quotient = quotient // size
    value = tl.load(Values + offset, mask=valid, other=0)
    if IS_FP64:
        absolute = tl.abs(value.to(tl.float64))
        infinity = tl.full((), float("inf"), tl.float64)
        threshold = tl.full((), MAX_ABS, tl.float64)
    else:
        absolute = tl.abs(value.to(tl.float32))
        infinity = tl.full((), float("inf"), tl.float32)
        threshold = tl.full((), MAX_ABS, tl.float32)
    # NaN < Inf is false, so this detects both NaNs and either infinity.
    nonfinite = valid & ~(absolute < infinity)
    nonfinite_bit = tl.max(nonfinite.to(tl.int32), axis=0)
    if HAS_BOUND:
        above_bit = tl.max((valid & (absolute > threshold)).to(tl.int32), axis=0)
    else:
        above_bit = 0
    tl.store(Status + block, nonfinite_bit | (above_bit << 1))
