#!/usr/bin/env python3
"""Standalone reproducer: AMD buffer-op pass group hangs gfx1250 on a jagged
split kernel whose inputs straddle the 2 GiB pointer-narrowing cutoff.

Depends only on torch + triton. No other packages, no dataset, no fbgemm.

The kernel is a verbatim copy of `_split_2D_jagged_multirow` from
mlcommons/training (recommendation_v4), reduced to a plain `@triton.jit` entry
point with the autotuner removed and the offending config pinned, so the
launch below is a single deterministic dispatch rather than a `do_bench` sweep.

  compile only (safe, never dispatches):
      python repro_gfx1250_buffer_ops.py --compile-only

  launch (hangs the GPU when buffer ops are enabled):
      AMDGCN_USE_BUFFER_OPS=1 python repro_gfx1250_buffer_ops.py   # hangs
      AMDGCN_USE_BUFFER_OPS=0 python repro_gfx1250_buffer_ops.py   # passes

Observed on: MI450 A0 eng. sample, gfx1250 (warp_size=32), ROCm 7.14,
torch 2.11.0+rocm7.14.0a20260625, triton 3.6.0 / 3.8.0 / main @ 76940ad3.
"""

import argparse
import os
import re

import triton
import triton.language as tl
import torch

# Shapes from the upstream test that hangs
# (jagged_tensors_test.py::test_concat_2D_jagged_large_tensor, backward pass):
# batch 130, max_len_a 32768, max_len_b 10, D 512, fp32, both sides jagged.
# Lengths are fixed here instead of random so the repro is deterministic.
BATCH = 130
MAX_LEN_A = 32768
MAX_LEN_B = 10
LEN_A = 16384  # per-row length of the left side; mean of the random test
LEN_B = 5
D = 512
DTYPE = torch.float32

# The autotune config that is in flight when the GPU wedges.
BLOCK_N = 2
NUM_WARPS = 1
NUM_STAGES = 1


@triton.jit
def split_2D_jagged_multirow(
    JaggedIn,
    OffsetsA,
    OffsetsB,
    MaxLenA,
    MaxLenB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_z = tl.program_id(1)
    block_n = tl.program_id(0)

    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b
    seq_start = seq_start_a + seq_start_b

    start_n = block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_mask = offs_n < seq_len

    in_ptrs = (
        JaggedIn
        + (seq_start + offs_n[:, None]).to(tl.int64) * stride_id
        + offs_d[None, :]
    )

    v = tl.load(in_ptrs, mask=valid_mask[:, None] & (offs_d[None, :] < D), other=0.0)

    to_prefix_b_mask = (offs_n < n_prefix_to_B) & valid_mask
    to_a_mask = (
        (offs_n >= n_prefix_to_B) & (offs_n < seq_len_a + n_prefix_to_B) & valid_mask
    )
    to_suffix_b_mask = (offs_n >= seq_len_a + n_prefix_to_B) & valid_mask

    out_b1_ptrs = (
        OutB
        + (offs_n[:, None] + seq_start_b).to(tl.int64) * stride_bd
        + offs_d[None, :]
    )
    tl.store(out_b1_ptrs, v, mask=to_prefix_b_mask[:, None] & (offs_d[None, :] < D))

    off_a = offs_n - n_prefix_to_B
    out_a_ptrs = (
        OutA + (off_a[:, None] + seq_start_a).to(tl.int64) * stride_ad + offs_d[None, :]
    )
    tl.store(out_a_ptrs, v, mask=to_a_mask[:, None] & (offs_d[None, :] < D))

    off_b = offs_n - seq_len_a
    out_b2_ptrs = (
        OutB + (off_b[:, None] + seq_start_b).to(tl.int64) * stride_bd + offs_d[None, :]
    )
    tl.store(out_b2_ptrs, v, mask=to_suffix_b_mask[:, None] & (offs_d[None, :] < D))


def _offsets(per_row, batch, device):
    lengths = torch.full((batch,), per_row, dtype=torch.int64, device=device)
    offs = torch.zeros((batch + 1,), dtype=torch.int64, device=device)
    offs[1:] = torch.cumsum(lengths, dim=0)
    return offs


def build_inputs(device="cuda"):
    offsets_a = _offsets(LEN_A, BATCH, device)
    offsets_b = _offsets(LEN_B, BATCH, device)
    total_a = int(offsets_a[-1].item())
    total_b = int(offsets_b[-1].item())

    jagged_in = torch.empty(
        (total_a + total_b, D), dtype=DTYPE, device=device
    ).uniform_(-1.0, 1.0)
    out_a = torch.empty((total_a, D), dtype=DTYPE, device=device)
    out_b = torch.empty((total_b, D), dtype=DTYPE, device=device)
    return offsets_a, offsets_b, jagged_in, out_a, out_b


def describe(name, t):
    n = t.untyped_storage().size()
    cutoff = 2**31 - 1
    verdict = "narrowed to 32-bit buffer ops" if n <= cutoff else "stays 64-bit global"
    print(f"  {name:10s} {n / 2**30:7.3f} GiB  ({n / cutoff:.3f}x cutoff)  -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="compile and report the emitted addressing modes; never dispatches",
    )
    args = ap.parse_args()

    print(f"triton {triton.__version__} | torch {torch.__version__}")
    print(f"AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS', '<unset>')}")
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.gcnArchName} warp_size={props.warp_size}")

    offsets_a, offsets_b, jagged_in, out_a, out_b = build_inputs()
    print("\nargument storage vs the 2 GiB narrowing cutoff:")
    for name, t in (
        ("JaggedIn", jagged_in),
        ("OutA", out_a),
        ("OutB", out_b),
        ("OffsetsA", offsets_a),
        ("OffsetsB", offsets_b),
    ):
        describe(name, t)

    max_seq_len = MAX_LEN_A + MAX_LEN_B
    grid = (triton.cdiv(max_seq_len, BLOCK_N), BATCH)
    print(f"\ngrid={grid} BLOCK_N={BLOCK_N} num_warps={NUM_WARPS}")

    kwargs = dict(
        JaggedIn=jagged_in,
        OffsetsA=offsets_a,
        OffsetsB=offsets_b,
        MaxLenA=MAX_LEN_A,
        MaxLenB=MAX_LEN_B,
        OutA=out_a,
        OutB=out_b,
        D=D,
        stride_id=jagged_in.stride(-2),
        stride_ad=out_a.stride(-2),
        stride_bd=out_b.stride(-2),
        n_prefix_to_B=0,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    if args.compile_only:
        compiled = split_2D_jagged_multirow.warmup(grid=grid, **kwargs)
        asm = compiled.asm["amdgcn"]
        counts = {}
        for m in re.finditer(r"^\s*((?:buffer|global)_(?:load|store)\w*)", asm, re.M):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        print("\nemitted memory instructions:")
        for op, n in sorted(counts.items()):
            print(f"  {op:24s} {n}")
        hybrid = any(k.startswith("buffer_") for k in counts) and any(
            k.startswith("global_") for k in counts
        )
        print(f"\nhybrid addressing (both buffer_* and global_* in one kernel): {hybrid}")
        print(f"s_endpgm count: {len(re.findall(r'\bs_endpgm\b', asm))}")
        return

    print("\nlaunching (this is the step that hangs with buffer ops enabled) ...")
    split_2D_jagged_multirow[grid](**kwargs)
    torch.cuda.synchronize()
    print("launch completed without hanging")

    ok = True
    for z in range(0, BATCH, 17):
        src = jagged_in[z * (LEN_A + LEN_B) : z * (LEN_A + LEN_B) + LEN_A]
        got = out_a[z * LEN_A : (z + 1) * LEN_A]
        if not torch.equal(src, got):
            ok = False
            bad = (src != got).any(dim=1).sum().item()
            print(f"  MISMATCH in row-block {z}: {bad} of {LEN_A} rows differ")
    print(f"correctness spot-check: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
