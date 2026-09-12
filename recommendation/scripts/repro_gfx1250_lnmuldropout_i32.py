#!/usr/bin/env python3
"""Standalone reproducer: `_ln_mul_dropout_fwd_rng` truncates its output row
offset to int32 and writes ~4.3 GB outside the destination tensor.

    AMDGCN_USE_BUFFER_OPS=0 python3 scripts/repro_gfx1250_lnmuldropout_i32.py

torch + triton only. No repo import, no fbgemm, no TorchRec, no dataset. The
kernel below is copied verbatim from
`generative_recommenders/ops/triton/triton_hstu_linear.py:148` (Triton 3.8 /
commit 7ff97e3109); only the `@triton_autotune` decorator is dropped so the
launch config is explicit.

WHAT IS WRONG
-------------
In the CONCAT_U and CONCAT_X branch the three output stores index `Y` with a
raw int32 row id:

    rows = start_row + tl.arange(0, BLOCK_N)          # int32
    ...
    tl.store(Y + rows[:, None] * stride_y + cols[None, :], ...)

`stride_y` is a plain Python int, so Triton specializes it to i32. The product
`rows * stride_y` is therefore an i32 multiply that wraps, and Triton
sign-extends the already-wrapped result when forming the pointer.

With the production DLRM-v4 HSTU shape -- D = num_heads * linear_dim = 4 * 128
= 512, concat_u and concat_x both true, so stride_y = 3 * D = 1536 -- the
product reaches 2**31 at

    row 1_398_102        (1_398_102 * 1536 = 2_147_484_672 > 2**31 - 1)

Every row at or above that index stores to `Y - 4.29 GB` instead of to `Y`.

The same function already knows this is required. Three lines above the stores,
the dropout-mask pointer IS widened:

    row_offsets_i64 = rows.to(tl.int64)               # line 231
    offsets = row_offsets_i64[:, None] * stride_mask + cols[None, :]

and the surrounding kernels in the same file widen too (`X += row.to(tl.int64)
* stride_x` at :336, `row_i64 = row.to(tl.int64)` at :490, `pid.to(tl.int64)`
at :114). Only the eight `Y` stores at :267, :272, :277, :283, :288, :294,
:299, :305 were missed. The loads of `X` and `U` at :183 and :189 have the same
defect with stride D = 512, which wraps at row 4_194_304.

`row_mask = rows < N` does NOT contain the damage. The mask is evaluated on the
*unwrapped* row id, so it is true exactly when the store is in range; the
wrapped negative offset is then written under a passing mask.

WHY IT IS SILENT RATHER THAN A FAULT
------------------------------------
`Y` is a fresh `torch.empty` late in a process whose caching allocator already
holds the ~140 GiB embedding table. `Y - 4.29 GB` therefore lands inside other
live allocations rather than in unmapped memory: no page fault, no traceback,
just other tensors quietly overwritten with bf16 activations. That is the
signature of docs/mi450_a0/mi450_a0.md defect [3] (`train_loss=nan` with no fault, reached
through poisoned embedding tables).

WHY IT LOOKS LIKE A BATCH-SIZE BUG
----------------------------------
N here is the total number of jagged rows in the batch, not the batch size, and
the kernel's autotune key is `["BLOCK_D"]` only -- codegen is byte-identical at
every batch size. Batch size enters solely by scaling N past 1_398_102. That
matches the observed gate on commit 7ff97e3109: BATCH_SIZE=128 clean over 4000
steps (2/2 runs), BATCH_SIZE=1024 nan within ~200 steps (2/2 runs).

EXPECTED OUTPUT
---------------
    N = 1398101  (last in-range row)   -> PASS
    N = 1398102  (first wrapping row)  -> FAIL, rows >= 1398102 never written,
                                          canary buffer corrupted
    --fix                              -> PASS at every N

FIX
---
Widen the row id once and use it for the loads and all stores, exactly as the
mask pointer already does:

    rows_i64 = rows.to(tl.int64)
    tl.store(Y + rows_i64[:, None] * stride_y + cols[None, :], ...)

`--fix` runs that variant so the one-line change can be verified before it is
made.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

import triton
import triton.language as tl

# stride_y = 3 * D with the production HSTU shape (4 heads x 128 linear_dim).
PROD_D = 512
PROD_STRIDE_Y = 3 * PROD_D
INT32_MAX = 2**31 - 1
# First row whose `row * stride_y` exceeds int32. 2147483647 // 1536 = 1398101,
# so 1398101 is the last safe row and 1398102 is the first that wraps.
FIRST_WRAPPING_ROW = INT32_MAX // PROD_STRIDE_Y + 1  # 1398102

POISON = -12345.0  # bf16-representable sentinel pre-written into Y and canary


def _make_kernel(widen: bool):
    """Build the kernel with int64 row offsets on or off.

    Both variants are compiled from one source so the only difference between
    PASS and FAIL is the `.to(tl.int64)`, with no other codegen drift.
    """

    @triton.jit
    def _ln_mul_dropout_fwd_rng(
        X,
        U,
        Y,
        W,
        B,
        Mean,
        Rstd,
        RANDOM_MASK,
        N,
        D,
        eps,
        dropout_ratio,
        stride_x,
        stride_u,
        stride_y,
        stride_mask,
        SILU_U: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        TRAINING: tl.constexpr,
        CONCAT_U: tl.constexpr,
        CONCAT_X: tl.constexpr,
        MUL_U_ACTIVATION_TYPE: tl.constexpr,
        WIDEN: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        start_row = block_id * BLOCK_N

        cols = tl.arange(0, BLOCK_D)
        col_mask = cols < D
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        mask_2d = row_mask[:, None] & col_mask[None, :]

        # The one difference between the shipping kernel and the fix.
        if WIDEN:
            rows_off = rows.to(tl.int64)
        else:
            rows_off = rows

        x_block = tl.load(
            X + rows_off[:, None] * stride_x + cols[None, :],
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)
        u_block = tl.load(
            U + rows_off[:, None] * stride_u + cols[None, :],
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)

        inv_D = 1.0 / D

        mean = tl.sum(x_block, axis=1) * inv_D
        tl.store(Mean + rows, mean, mask=row_mask)
        mean = tl.expand_dims(mean, 1)

        x_mean = x_block - mean
        x_mean = tl.where(mask_2d, x_mean, 0.0)
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) * inv_D
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)
        rstd = tl.expand_dims(rstd, 1)

        y = x_mean * rstd
        w = tl.load(W + cols, mask=col_mask).to(tl.float32)
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)
        y = y * w[None, :] + b[None, :]

        sigmoid_u_block = tl.sigmoid(u_block)
        silu_u_block = u_block * sigmoid_u_block

        if MUL_U_ACTIVATION_TYPE == "silu":
            y = y * silu_u_block
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            y = y * sigmoid_u_block
        else:
            y = y * u_block

        if CONCAT_U and SILU_U:
            u_block = silu_u_block

        if TRAINING:
            # Already int64 in the shipping kernel -- the mask pointer is the
            # one place the widening was not forgotten.
            row_offsets_i64 = rows.to(tl.int64)
            dropout_scale = 1.0 / (1.0 - dropout_ratio)
            offsets = row_offsets_i64[:, None] * stride_mask + cols[None, :]

            if CONCAT_U or CONCAT_X:
                compressed = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=0).to(
                    tl.int32
                )
                y_keep = (compressed & 1) != 0

                if CONCAT_U and CONCAT_X:
                    x_keep = (compressed & 2) != 0
                    u_keep = (compressed & 4) != 0
                    u_block = tl.where(u_keep, u_block * dropout_scale, 0.0)
                    x_block = tl.where(x_keep, x_block * dropout_scale, 0.0)
                elif CONCAT_U:
                    u_keep = (compressed & 2) != 0
                    u_block = tl.where(u_keep, u_block * dropout_scale, 0.0)
                else:
                    x_keep = (compressed & 2) != 0
                    x_block = tl.where(x_keep, x_block * dropout_scale, 0.0)

                y = tl.where(y_keep, y * dropout_scale, 0.0)
            else:
                y_keep = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=True)
                y = tl.where(y_keep, y * dropout_scale, 0.0)

        if CONCAT_U and CONCAT_X:
            tl.store(
                Y + rows_off[:, None] * stride_y + cols[None, :],
                u_block.to(Y.dtype.element_ty),
                mask=mask_2d,
            )
            tl.store(
                Y + rows_off[:, None] * stride_y + (cols + D)[None, :],
                x_block.to(Y.dtype.element_ty),
                mask=mask_2d,
            )
            tl.store(
                Y + rows_off[:, None] * stride_y + (cols + 2 * D)[None, :],
                y.to(Y.dtype.element_ty),
                mask=mask_2d,
            )
        else:
            tl.store(
                Y + rows_off[:, None] * stride_y + cols[None, :],
                y.to(Y.dtype.element_ty),
                mask=mask_2d,
            )

    return _ln_mul_dropout_fwd_rng


def _print_env() -> None:
    print("-- environment", flush=True)
    print(f"   torch          {torch.__version__}", flush=True)
    print(f"   triton         {triton.__version__}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "<n/a>")
        total = props.total_memory / 2**30
        print(f"   device         {props.name} ({arch}) {total:.1f} GiB", flush=True)
    for var in (
        "AMDGCN_USE_BUFFER_OPS",
        "TRITON_ALWAYS_COMPILE",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        print(f"   {var:<14} {os.environ.get(var, '<unset>')}", flush=True)


def _require_buffer_ops_off() -> None:
    """Buffer ops wedge this node (docs/mi450_a0/mi450_a0.md defect [1]) and cost a reboot."""
    val = os.environ.get("AMDGCN_USE_BUFFER_OPS")
    if val != "0":
        raise RuntimeError(
            "AMDGCN_USE_BUFFER_OPS must be set to 0 before running on gfx1250; "
            f"got {val!r}. Re-run as: AMDGCN_USE_BUFFER_OPS=0 python3 {sys.argv[0]}"
        )


def _reference_row(x_row, u_row, w, b, eps, mul_u_activation_type, silu_u):
    """Expected [u | x | y] for one row, in fp32, on the host device.

    Inputs are identical across rows and the dropout mask is all-keep with
    ratio 0, so one row is the expected value for every row. That keeps the
    check O(D) instead of materializing a 4 GB fp32 reference.
    """
    x = x_row.float()
    u = u_row.float()
    mean = x.mean()
    xm = x - mean
    rstd = 1.0 / torch.sqrt(xm.pow(2).mean() + eps)
    y = xm * rstd * w.float() + b.float()
    sig = torch.sigmoid(u)
    if mul_u_activation_type == "silu":
        y = y * (u * sig)
    elif mul_u_activation_type == "sigmoid":
        y = y * sig
    else:
        y = y * u
    u_out = (u * sig) if silu_u else u
    return torch.cat([u_out, x, y])


def run_case(
    n: int,
    d: int,
    block_n: int,
    num_warps: int,
    widen: bool,
    canary_gib: float,
    device: torch.device,
) -> bool:
    """Run one N and report whether the kernel stayed inside Y. True == PASS."""
    dtype = torch.bfloat16
    eps = 1e-6
    mul_u_activation_type = "none"  # stu.py:350, production DLRM-v4
    silu_u = False
    stride_y = 3 * d

    wrap_row = INT32_MAX // stride_y + 1
    n_wrapping = max(0, n - wrap_row)

    # Canary allocated BEFORE Y so the caching allocator places Y just above it
    # and the -4.29 GB wild store lands inside. This is best-effort evidence of
    # where the writes go; the authoritative check is the Y content below.
    canary = None
    if canary_gib > 0:
        n_elem = int(canary_gib * 2**30 / 2)
        canary = torch.full((n_elem,), POISON, dtype=dtype, device=device)

    # One distinct row pattern, broadcast, so the expected output is a single
    # row and the comparison costs nothing.
    x_row = torch.linspace(-2.0, 2.0, d, device=device, dtype=torch.float32)
    u_row = torch.linspace(1.0, -1.0, d, device=device, dtype=torch.float32)
    x = x_row.to(dtype).expand(n, d).contiguous()
    u = u_row.to(dtype).expand(n, d).contiguous()
    w = torch.linspace(0.5, 1.5, d, device=device, dtype=torch.float32).to(dtype)
    b = torch.linspace(-0.1, 0.1, d, device=device, dtype=torch.float32).to(dtype)

    # dropout_ratio 0 with an all-keep mask makes the output exact while
    # keeping TRAINING=True, so the real code path (mask load included) runs.
    random_mask = torch.full((n, d), 0b111, dtype=torch.int8, device=device)

    y = torch.full((n, 3 * d), POISON, dtype=dtype, device=device)
    mean = torch.empty((n,), dtype=torch.float32, device=device)
    rstd = torch.empty((n,), dtype=torch.float32, device=device)

    kernel = _make_kernel(widen)
    grid = (triton.cdiv(n, block_n),)
    kernel[grid](
        x,
        u,
        y,
        w,
        b,
        mean,
        rstd,
        random_mask,
        n,
        d,
        eps,
        0.0,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        random_mask.stride(0),
        SILU_U=silu_u,
        BLOCK_D=d,
        BLOCK_N=block_n,
        TRAINING=True,
        CONCAT_U=True,
        CONCAT_X=True,
        MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
        WIDEN=widen,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()

    expected = _reference_row(
        x_row.to(dtype), u_row.to(dtype), w, b, eps, mul_u_activation_type, silu_u
    ).to(dtype)

    # Rows the kernel failed to write keep POISON. Scan in chunks so the
    # comparison never allocates another copy of Y.
    first_bad = -1
    n_bad = 0
    chunk = 1 << 16
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = y[start:stop]
        bad = (block != expected).any(dim=1)
        cnt = int(bad.sum())
        if cnt:
            n_bad += cnt
            if first_bad < 0:
                first_bad = start + int(bad.nonzero()[0])

    canary_hits = 0
    if canary is not None:
        canary_hits = int((canary != POISON).sum())

    tag = "fixed" if widen else "shipping"
    print(
        f"-- N={n:<9} BLOCK_N={block_n} num_warps={num_warps} kernel={tag}",
        flush=True,
    )
    print(
        f"   int32 wrap row = {wrap_row}  ->  {n_wrapping} of {n} rows expected to wrap",
        flush=True,
    )
    print(f"   rows not written correctly : {n_bad}", flush=True)
    if first_bad >= 0:
        print(f"   first bad row              : {first_bad}", flush=True)
    if canary is not None:
        print(f"   canary elements clobbered  : {canary_hits}", flush=True)

    ok = n_bad == 0 and canary_hits == 0
    if ok:
        print("   PASS", flush=True)
    else:
        print("   !! FAIL", flush=True)
        if first_bad >= 0 and first_bad == wrap_row:
            print(
                f"   !! first bad row is exactly the int32 wrap row {wrap_row} "
                f"(= (2**31-1)//{stride_y} + 1)",
                flush=True,
            )
        if canary_hits:
            print(
                "   !! writes landed in an unrelated allocation ~4.29 GB below Y: "
                "silent corruption of other live tensors, no page fault",
                flush=True,
            )

    del x, u, y, mean, rstd, random_mask, canary
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n",
        type=int,
        nargs="*",
        default=None,
        help="row counts to test (default: bisect around the int32 wrap row)",
    )
    p.add_argument("--d", type=int, default=PROD_D, help="feature dim (default 512)")
    p.add_argument("--block-n", type=int, default=1, help="BLOCK_N (autotuned 1..16)")
    p.add_argument("--num-warps", type=int, default=4, help="num_warps (autotuned 1..4)")
    p.add_argument(
        "--fix",
        action="store_true",
        help="run the int64-widened variant instead of the shipping kernel",
    )
    p.add_argument(
        "--sweep-config",
        action="store_true",
        help="repeat the first N over every autotune (BLOCK_N, num_warps) pair",
    )
    p.add_argument(
        "--canary-gib",
        type=float,
        default=2.0,
        help=(
            "GiB of canary placed below Y (0 disables). Y sits ~3.4 GiB above the "
            "canary's end (x, u and the mask are allocated between them), and the "
            "wild store is at Y-4.29 GiB, so ~1 GiB is enough to catch it."
        ),
    )
    args = p.parse_args()

    _require_buffer_ops_off()
    if not torch.cuda.is_available():
        print("!! no GPU visible", flush=True)
        return 1
    _print_env()

    stride_y = 3 * args.d
    wrap_row = INT32_MAX // stride_y + 1
    print(
        f"-- D={args.d} stride_y=3*D={stride_y} "
        f"int32 wrap at row {wrap_row}",
        flush=True,
    )

    ns = args.n if args.n else [wrap_row - 1, wrap_row + 1024]
    device = torch.device("cuda:0")

    results = []
    if args.sweep_config:
        n = ns[0]
        for block_n in (1, 2, 4, 8, 16):
            for num_warps in (1, 2, 4):
                results.append(
                    (
                        n,
                        block_n,
                        num_warps,
                        run_case(
                            n,
                            args.d,
                            block_n,
                            num_warps,
                            args.fix,
                            args.canary_gib,
                            device,
                        ),
                    )
                )
    else:
        for n in ns:
            results.append(
                (
                    n,
                    args.block_n,
                    args.num_warps,
                    run_case(
                        n,
                        args.d,
                        args.block_n,
                        args.num_warps,
                        args.fix,
                        args.canary_gib,
                        device,
                    ),
                )
            )

    print("-- summary", flush=True)
    for n, block_n, num_warps, ok in results:
        print(
            f"   N={n:<9} BLOCK_N={block_n:<3} num_warps={num_warps}  "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )

    failed = [r for r in results if not r[3]]
    if args.fix:
        # The widened variant must be clean at every N; a failure here means
        # the one-line fix is not sufficient.
        if failed:
            print("!! the int64-widened kernel still fails -- fix is incomplete", flush=True)
            return 1
        print("-- int64-widened kernel is correct at every N tested", flush=True)
        return 0

    if failed:
        print(
            "!! shipping kernel corrupts memory above the int32 wrap row "
            "-- re-run with --fix to confirm the one-line change resolves it",
            flush=True,
        )
        return 1
    print(
        "?? shipping kernel did not corrupt at the N values tested; "
        "raise --n above the wrap row",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"!! {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
