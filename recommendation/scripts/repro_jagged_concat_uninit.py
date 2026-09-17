#!/usr/bin/env python3
"""Standalone reproducer: `concat_2D_jagged_multirow` silently leaves output rows
UNINITIALIZED when a batch element's combined length exceeds the launch grid.

Depends only on torch + triton. No repo import, no dataset, no fbgemm.

WHAT THIS IS
------------
`_concat_2D_jagged_multirow` (generative_recommenders/ops/triton/
triton_jagged_tensors.py:182) is on the ranker path on ROCm. Note the routing:
the model calls the ordinary `concat_2D_jagged` / `split_2D_jagged`, and
`_triton_concat_2D_jagged_internal` (triton_jagged_tensors.py:52-69) and
`_triton_split_2D_jagged_internal` (:120-131) send every HIP launch to the
*multirow* kernels, because the basic one fails to lower in
`TritonAMDGPUCanonicalizePointers`. (The `*_multirow` autograd wrapper classes
at :813 and :921 are dead code -- zero callers outside tests. Do not aim at
those line numbers.) The kernel is launched with

    grid = (triton.cdiv(max_seq_len, meta["BLOCK_N"]), B)

and program `(block_n, off_z)` stores to `Out` rows

    out_seq_start = seq_start_a + seq_start_b + offs_n
    offs_n        = block_n * BLOCK_N + arange(0, BLOCK_N)

under the single mask `valid_mask = offs_n < seq_len`, where
`seq_len = seq_len_a(z) + seq_len_b(z)` comes from the *offsets*.

The row index is bounded by the offsets, but the number of programs is bounded by
`max_seq_len`. Nothing ties the two together. So the rows actually written for
batch element `z` are

    offs_n in [0, min(seq_len(z), ceil(max_seq_len / BLOCK_N) * BLOCK_N))

  COVERAGE CONDITION -- for every z:
      seq_len_a(z) + seq_len_b(z)  <=  ceil(max_seq_len / BLOCK_N) * BLOCK_N

When that is violated the tail rows of `Out` are never stored to. They are not
zero and they are not garbage-but-harmless: `Out` is allocated with
`torch.empty`, so those rows hold whatever the caching allocator last left
there.

Why that is a `nan` candidate rather than a cosmetic issue: the buffer is a
*gradient*. `_Split2DJaggedFunction.backward` (triton_jagged_tensors.py:738)
does

    d_jagged_in = torch.empty((ctx.total_seq_len, ctx.D), ...)   # UNINITIALIZED
    _triton_concat_2D_jagged_internal(..., values_out=d_jagged_in, ...)
    return None, d_jagged_in, None, None, None, None, None, None, None

so any uncovered row flows straight into the optimizer. Four more allocations on
this path have the same shape: `_Concat2DJaggedFunction.forward` values_out
(:583), `_Split2DJaggedFunction.forward` values_a/values_b (:697, :700). And
`_Concat2DJaggedFunction.backward` allocates `d_values_a` with `torch.zeros`
(:621) but `d_values_b` with `torch.empty` (:624) -- an asymmetry with no stated
justification, which reads as someone having already been bitten on one side.

`BLOCK_N` is autotuned over {1, 2, 4, 8} (triton_jagged_tensors.py:168-178), so
the rounding slack is at most 7 rows and varies per process with the config
draw. The condition is therefore effectively `combined_len <= max_seq_len`.

STATUS -- READ THIS BEFORE CITING THE SCRIPT
--------------------------------------------
What this script does establish, deterministically: given offsets that violate
the coverage condition, it shows exactly which rows are left uninitialized and
proves the observed set matches a host-computed prediction row for row.
`--violate 0` is the control -- it asserts *full* coverage when the condition
holds, so a PASS is meaningful rather than vacuous.

What it does NOT establish is that the production DLRM v4 ranker violates the
condition at BATCH_SIZE=128. A static audit of all eight ranker call sites of
`concat_2D_jagged` / `split_2D_jagged` (dlrm_hstu.py:510 and :726,
preprocessors.py:288 and :300, contextual_interleave_preprocessor.py:180 and
:191, action_encoder.py:100, content_encoder.py:87, hstu_transducer.py:216 and
:233) found that every one passes

    max_seq_len = max_len_left + max_len_right

or larger, which bounds `len_a(z) + len_b(z)` structurally. So on the ranker
path the condition appears to HOLD, and the residual exposure is narrow: it
requires `max_len_left` or `max_len_right` to be smaller than the true maximum
of its offsets at runtime -- a stale, config-derived, or untruncated length.
That is data-dependent and cannot be settled from source. Treat this script as
a calibrated detector for that residual case, not as a demonstration that the
case occurs.

One further point against, stated because it cuts the other way: the non-HIP
path launches the basic `_concat_2D_jagged` on grid `(max_seq_len, B)`, which
covers *fewer* rows than the multirow grid's `cdiv(max_seq_len, BLOCK_N) *
BLOCK_N`. A real coverage violation would therefore break on NVIDIA at least as
badly, and this code runs there. That is evidence the condition is not violated
in practice.

Two constraints from defect [3] that the hypothesis does still satisfy:
`HSTU_HAMMER_KERNEL=PYTORCH` was verified to bypass these ops entirely (the
dispatch ladder in ops/jagged_tensors.py:110 and :196, driven model-wide by
`set_hammer_kernel` from dlrm_v4/train/utils.py:411-417), which is consistent
with PYTORCH being finite 3/3; and uncovered rows scale linearly with batch,
consistent with a batch-128-only symptom.

This is a SOFTWARE defect class, not a codegen one. It reproduces on any GPU and
needs no gfx1250-specific behaviour; the kernel below is a verbatim copy. If it
turns out to explain the batch-128 `nan`, the fix is in this repo (allocate with
`torch.zeros`, or size the grid from the offsets rather than `max_seq_len`) and
nothing goes to the LLVM/AMDGPU team.

USAGE
-----
    export AMDGCN_USE_BUFFER_OPS=0          # mandatory on gfx1250, see mi450.md [1]

    # compile only -- never dispatches, cannot fault
    python scripts/repro_jagged_concat_uninit.py --compile-only

    # the control: coverage condition holds, every row must be written
    python scripts/repro_jagged_concat_uninit.py --violate 0

    # the defect: each batch element overruns max_seq_len by 3 rows
    python scripts/repro_jagged_concat_uninit.py --violate 3

    # how the uncovered extent scales with batch size
    python scripts/repro_jagged_concat_uninit.py --violate 3 --scan-batch

    # show that a real torch.empty picks up poisoned memory (impact demo)
    python scripts/repro_jagged_concat_uninit.py --violate 3 --allocator-demo

Exit code 0 = behaved as predicted, 1 = did not.
"""

import argparse
import math
import os
import sys

import torch
import triton
import triton.language as tl

# Defaults chosen to be small and fast; the defect is structural, not shape
# dependent. D=512 matches the ranker's transducer dim.
DEF_BATCH = 8
DEF_MAX_SEQ_LEN = 64
DEF_LEN_A = 40
DEF_LEN_B = 24  # LEN_A + LEN_B == MAX_SEQ_LEN exactly, so --violate 0 is tight
DEF_D = 512
DEF_BLOCK_N = 4

# A quiet-NaN payload that is recognisable in hex, so an unwritten row is
# distinguishable from a legitimately-computed NaN.
SENTINEL_BITS = 0x7FC0BEEF


def _sentinel() -> torch.Tensor:
    return torch.tensor([SENTINEL_BITS], dtype=torch.int32).view(torch.float32)


# ---------------------------------------------------------------------------
# Verbatim copy of _concat_2D_jagged_multirow from
# generative_recommenders/ops/triton/triton_jagged_tensors.py:182, with the
# autotuner removed and BLOCK_N pinned so the launch is a single deterministic
# dispatch rather than a do_bench sweep. Body is unmodified.
# ---------------------------------------------------------------------------
@triton.jit
def concat_2D_jagged_multirow(
    ValuesA,
    ValuesB,
    OffsetsA,
    OffsetsB,
    MaxLenA,
    MaxLenB,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_od,
    n_prefix_from_B,
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

    start_n = block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_mask = offs_n < seq_len

    out_seq_start = seq_start_a + seq_start_b + offs_n
    out_ptrs = Out + out_seq_start[:, None].to(tl.int64) * stride_od + offs_d[None, :]

    from_prefix_b_mask = (offs_n < n_prefix_from_B) & valid_mask
    from_a_mask = (
        (offs_n >= n_prefix_from_B)
        & (offs_n < seq_len_a + n_prefix_from_B)
        & valid_mask
    )
    from_suffix_b_mask = (offs_n >= seq_len_a + n_prefix_from_B) & valid_mask

    in_b1_ptrs = (
        ValuesB
        + (offs_n[:, None] + seq_start_b).to(tl.int64) * stride_bd
        + offs_d[None, :]
    )
    v_b1 = tl.load(
        in_b1_ptrs, mask=from_prefix_b_mask[:, None] & (offs_d[None, :] < D), other=0.0
    )
    tl.store(out_ptrs, v_b1, mask=from_prefix_b_mask[:, None] & (offs_d[None, :] < D))

    off_a = offs_n - n_prefix_from_B
    in_a_ptrs = (
        ValuesA
        + (off_a[:, None] + seq_start_a).to(tl.int64) * stride_ad
        + offs_d[None, :]
    )
    v_a = tl.load(
        in_a_ptrs, mask=from_a_mask[:, None] & (offs_d[None, :] < D), other=0.0
    )
    tl.store(out_ptrs, v_a, mask=from_a_mask[:, None] & (offs_d[None, :] < D))

    off_b = offs_n - seq_len_a
    in_b2_ptrs = (
        ValuesB
        + (off_b[:, None] + seq_start_b).to(tl.int64) * stride_bd
        + offs_d[None, :]
    )
    v_b2 = tl.load(
        in_b2_ptrs, mask=from_suffix_b_mask[:, None] & (offs_d[None, :] < D), other=0.0
    )
    tl.store(out_ptrs, v_b2, mask=from_suffix_b_mask[:, None] & (offs_d[None, :] < D))


# ---------------------------------------------------------------------------
# Host-side model of the kernel's coverage. This is the whole point: we do not
# merely look for NaNs, we PREDICT the exact uncovered row set and then check
# that the observed set matches it.
# ---------------------------------------------------------------------------
def predict_uncovered_rows(len_a, len_b, max_seq_len, block_n):
    """Return the sorted list of Out row indices the kernel will never store to.

    Mirrors the kernel exactly:
      - programs cover offs_n in [0, cdiv(max_seq_len, BLOCK_N) * BLOCK_N)
      - stores are additionally masked by offs_n < seq_len_a + seq_len_b
      - the destination row is seq_start_a + seq_start_b + offs_n
    """
    grid_rows = triton.cdiv(max_seq_len, block_n) * block_n
    uncovered = []
    start_a = 0
    start_b = 0
    for z in range(len(len_a)):
        seq_len = len_a[z] + len_b[z]
        base = start_a + start_b
        for offs_n in range(grid_rows, seq_len):
            uncovered.append(base + offs_n)
        start_a += len_a[z]
        start_b += len_b[z]
    return uncovered


def build_case(batch, max_seq_len, len_a_val, len_b_val, violate, device, dtype):
    """Build a case where every batch element overruns max_seq_len by `violate`."""
    len_a = [len_a_val] * batch
    len_b = [len_b_val + violate] * batch
    total_a, total_b = sum(len_a), sum(len_b)

    offs_a = torch.tensor(
        [0] + list(torch.tensor(len_a).cumsum(0)), dtype=torch.int64, device=device
    )
    offs_b = torch.tensor(
        [0] + list(torch.tensor(len_b).cumsum(0)), dtype=torch.int64, device=device
    )
    return len_a, len_b, total_a, total_b, offs_a, offs_b


def run_case(args, batch, violate, device, verbose=True):
    """One launch. Returns (n_predicted, n_observed, matched)."""
    dtype = torch.float32
    D = args.dim
    block_n = args.block_n

    len_a, len_b, total_a, total_b, offs_a, offs_b = build_case(
        batch, args.max_seq_len, args.len_a, args.len_b, violate, device, dtype
    )
    total = total_a + total_b

    # Distinctive finite payloads so a written row is unmistakably written.
    values_a = torch.arange(1, total_a + 1, device=device, dtype=dtype)[:, None].repeat(
        1, D
    )
    values_b = -torch.arange(1, total_b + 1, device=device, dtype=dtype)[:, None].repeat(
        1, D
    )

    # PRIMARY DETECTOR: pre-fill the output with a sentinel. This does not rely
    # on any allocator behaviour -- any row the kernel does not store to
    # provably retains the sentinel.
    out = torch.empty((total, D), device=device, dtype=dtype)
    out.fill_(_sentinel().item())

    grid = (triton.cdiv(args.max_seq_len, block_n), batch)
    concat_2D_jagged_multirow[grid](
        ValuesA=values_a,
        ValuesB=values_b,
        OffsetsA=offs_a,
        OffsetsB=offs_b,
        MaxLenA=args.len_a,
        MaxLenB=args.len_b + violate,
        Out=out,
        D=D,
        stride_ad=values_a.stride(0),
        stride_bd=values_b.stride(0),
        stride_od=out.stride(0),
        n_prefix_from_B=0,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_N=block_n,
    )
    torch.cuda.synchronize()

    predicted = predict_uncovered_rows(len_a, len_b, args.max_seq_len, block_n)

    # A row is "untouched" iff every one of its D lanes still holds the sentinel
    # bit pattern. Compare as integers so NaN != NaN does not bite.
    out_bits = out.view(torch.int32)
    untouched_mask = (out_bits == SENTINEL_BITS).all(dim=1)
    observed = torch.nonzero(untouched_mask, as_tuple=False).flatten().tolist()

    matched = observed == predicted

    if verbose:
        print(
            f"-- batch={batch} violate={violate} BLOCK_N={block_n} "
            f"grid={grid} rows={total}",
            flush=True,
        )
        print(
            f"   grid covers offs_n < {triton.cdiv(args.max_seq_len, block_n) * block_n}, "
            f"combined len per element = {len_a[0] + len_b[0]}",
            flush=True,
        )
        print(
            f"   predicted uninitialized rows: {len(predicted)}   "
            f"observed: {len(observed)}   match: {matched}",
            flush=True,
        )
        if observed and len(observed) <= 24:
            print(f"   rows: {observed}", flush=True)
        elif observed:
            print(
                f"   rows: {observed[:12]} ... {observed[-4:]}",
                flush=True,
            )

    return len(predicted), len(observed), matched


def allocator_demo(args, device):
    """Show that a real torch.empty inherits poisoned memory.

    The sentinel prefill above proves coverage. This shows the consequence in
    the shape the production code actually has: `torch.empty` with no prefill,
    landing on a recycled block.
    """
    print("\n== allocator reuse demo (production shape: torch.empty, no prefill) ==")
    dtype = torch.float32
    D = args.dim
    batch, violate = args.batch, args.violate

    len_a, len_b, total_a, total_b, offs_a, offs_b = build_case(
        batch, args.max_seq_len, args.len_a, args.len_b, violate, device, dtype
    )
    total = total_a + total_b

    # Poison a block of exactly the right size, then free it. The caching
    # allocator hands the same block back to the next same-size request.
    poison = torch.empty((total, D), device=device, dtype=dtype)
    poison.fill_(float("nan"))
    poison_ptr = poison.data_ptr()
    del poison

    out = torch.empty((total, D), device=device, dtype=dtype)
    reused = out.data_ptr() == poison_ptr
    print(f"-- allocator returned the poisoned block: {reused}", flush=True)
    if not reused:
        print(
            "!! block not reused; the demo is inconclusive (the sentinel test "
            "above is the authoritative one)",
            flush=True,
        )

    values_a = torch.ones((total_a, D), device=device, dtype=dtype)
    values_b = torch.ones((total_b, D), device=device, dtype=dtype)

    grid = (triton.cdiv(args.max_seq_len, args.block_n), batch)
    concat_2D_jagged_multirow[grid](
        ValuesA=values_a,
        ValuesB=values_b,
        OffsetsA=offs_a,
        OffsetsB=offs_b,
        MaxLenA=args.len_a,
        MaxLenB=args.len_b + violate,
        Out=out,
        D=D,
        stride_ad=values_a.stride(0),
        stride_bd=values_b.stride(0),
        stride_od=out.stride(0),
        n_prefix_from_B=0,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_N=args.block_n,
    )
    torch.cuda.synchronize()

    n_nonfinite_rows = int((~torch.isfinite(out)).any(dim=1).sum().item())
    total_sum = out.sum().item()
    print(
        f"-- rows containing non-finite values: {n_nonfinite_rows} of {total}",
        flush=True,
    )
    print(f"-- out.sum() = {total_sum}", flush=True)
    if n_nonfinite_rows:
        print(
            "!! a gradient buffer produced by this op is non-finite purely "
            "because rows outside the launch grid were never written",
            flush=True,
        )
    return n_nonfinite_rows


def compile_only(args):
    """Compile the kernel and report its allocation. Never dispatches."""
    print("== compile-only: no dispatch, cannot fault ==", flush=True)
    src = triton.compiler.ASTSource(
        fn=concat_2D_jagged_multirow,
        signature={
            "ValuesA": "*fp32",
            "ValuesB": "*fp32",
            "OffsetsA": "*i64",
            "OffsetsB": "*i64",
            "MaxLenA": "i32",
            "MaxLenB": "i32",
            "Out": "*fp32",
            "D": "i32",
            "stride_ad": "i32",
            "stride_bd": "i32",
            "stride_od": "i32",
            "n_prefix_from_B": "i32",
            "IS_DENSE_A": "constexpr",
            "IS_DENSE_B": "constexpr",
            "BLOCK_D": "constexpr",
            "BLOCK_N": "constexpr",
        },
        constexprs={
            "IS_DENSE_A": False,
            "IS_DENSE_B": False,
            "BLOCK_D": triton.next_power_of_2(args.dim),
            "BLOCK_N": args.block_n,
        },
    )
    compiled = triton.compile(src)
    asm = compiled.asm["amdgcn"]
    n_store = asm.count("global_store")
    print(f"-- compiled: BLOCK_D={triton.next_power_of_2(args.dim)} "
          f"BLOCK_N={args.block_n}", flush=True)
    print(f"-- num_warps={compiled.metadata.num_warps}, "
          f"global_store count={n_store}", flush=True)
    print(
        "-- coverage is a HOST-side property (the grid), so it is not visible "
        "in the machine code; run without --compile-only to test it",
        flush=True,
    )
    return 0


def _print_env():
    print(f"torch      {torch.__version__}")
    print(f"triton     {triton.__version__}")
    print(
        f"AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS', '<unset>')}"
    )
    if torch.cuda.is_available():
        print(f"device     {torch.cuda.get_device_name(0)}")
    print("", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--max-seq-len", type=int, default=DEF_MAX_SEQ_LEN)
    ap.add_argument("--len-a", type=int, default=DEF_LEN_A)
    ap.add_argument("--len-b", type=int, default=DEF_LEN_B)
    ap.add_argument("--dim", type=int, default=DEF_D)
    ap.add_argument("--block-n", type=int, default=DEF_BLOCK_N, choices=[1, 2, 4, 8])
    ap.add_argument(
        "--violate",
        type=int,
        default=3,
        help="rows by which each element's combined length overruns max_seq_len; "
        "0 is the control and must show full coverage",
    )
    ap.add_argument("--scan-batch", action="store_true",
                    help="show how the uncovered extent scales with batch size")
    ap.add_argument("--allocator-demo", action="store_true",
                    help="also show a real torch.empty inheriting poisoned memory")
    ap.add_argument("--compile-only", action="store_true",
                    help="compile and exit; never dispatches")
    args = ap.parse_args()

    _print_env()

    if args.compile_only:
        return compile_only(args)

    if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0" and torch.version.hip:
        raise SystemExit(
            "AMDGCN_USE_BUFFER_OPS=0 is required on gfx1250 (see docs/mi450.md [1]); "
            "buffer ops hang the node and cost a reboot."
        )
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible")

    device = torch.device("cuda")
    torch.manual_seed(0)

    rc = 0

    print("== control: coverage condition HOLDS (violate=0) ==", flush=True)
    n_pred, n_obs, ok = run_case(args, args.batch, 0, device)
    if n_pred != 0 or n_obs != 0:
        print("!! control failed: rows were uninitialized even though the "
              "coverage condition holds", flush=True)
        rc = 1
    else:
        print("-- PASS: every output row was written\n", flush=True)

    if args.violate > 0:
        print(f"== defect: coverage condition VIOLATED by {args.violate} rows "
              f"per element ==", flush=True)
        n_pred, n_obs, ok = run_case(args, args.batch, args.violate, device)
        if not ok:
            print("!! observed uninitialized set does NOT match the prediction",
                  flush=True)
            rc = 1
        elif n_pred == 0:
            print("!! predicted no uncovered rows; the case is vacuous", flush=True)
            rc = 1
        else:
            print(
                f"-- DEFECT CONFIRMED: {n_obs} rows of a gradient-shaped buffer "
                f"were never written, exactly as predicted\n",
                flush=True,
            )

    if args.scan_batch:
        print("== scaling with batch size ==", flush=True)
        print("   batch   uncovered rows", flush=True)
        for b in [8, 16, 32, 64, 128]:
            n_pred, n_obs, ok = run_case(args, b, args.violate, device, verbose=False)
            flag = "" if ok else "   (MISMATCH)"
            print(f"   {b:5d}   {n_obs:6d}{flag}", flush=True)
            if not ok:
                rc = 1
        print(
            "\n-- uncovered rows scale linearly with batch: 16x more exposure "
            "at B=128 than at B=8\n",
            flush=True,
        )

    if args.allocator_demo:
        allocator_demo(args, device)

    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
