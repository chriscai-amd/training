#!/usr/bin/env python3
"""Standalone HSTU-layer stress probe for the residual gfx1250 `train_loss=nan`.

Handoff reproducer for mi450_b0.md section 12.2. Calls the repo's public op
wrappers only -- **no dataset, no torchrec, no embedding tables, no training
loop**. It does import `fbgemm_gpu`, but only because the PYTORCH *reference*
path needs `jagged_to_padded_dense`; the Triton path under test does not, and
`--no-reference` drops the dependency entirely.

    AMDGCN_USE_BUFFER_OPS=0 PYTORCH_CUDA_ALLOC_CONF= \
        python3 repro_gfx1250_hstu_layer.py

WHAT THIS IS FOR
----------------
On MI450/gfx1250 the batch-1024 e2e goes `train_loss=nan` at step ~1370 and
stays there, with no page fault, no traceback and a clean dmesg. An int32 row
address wrap in `triton_layer_norm.py` was found and fixed first; that fix is
necessary but NOT sufficient -- it moved the onset 300 -> 1370. The remaining
defect is NOT root-caused. See mi450_b0.md section 12.2.

This script drives one full HSTU transformer layer -- the same ops the training
step runs, at the same production shapes -- forward AND backward, over many
iterations of fresh random data, and checks three things each iteration:

  1. non-finite values in any output or gradient          (the e2e symptom)
  2. TRITON vs PYTORCH divergence                          (silent wrong results)
  3. a canary buffer clobbered by a wild store             (stray addressing)

Check 2 is the important one. The e2e only ever shows us a `nan` thousands of
steps after the fact; comparing the Triton path against the PyTorch path at the
same shapes turns "something is wrong somewhere" into "op X diverges at
iteration N", which is what a Triton/LLVM engineer can act on.

WHAT A FAILURE MEANS
--------------------
  nonfinite / divergence, TRITON only   -> defect is in the Triton op stack.
                                           The printed tensor name localises it.
  divergence, PYTORCH too               -> not a Triton bug. Suspect the
                                           platform (see mi450_b0.md sections 2-9:
                                           this host had latched gather
                                           corruption that needed an AC cycle).
  canary_bad > 0                        -> a wild store. Address arithmetic.
  clean over many iters                 -> the defect is NOT in these ops at
                                           these shapes. Look at fbgemm TBE, the
                                           optimizer, or accumulation over steps.
                                           A clean run is a real result: record it.

MEMORY
------
`uvqk` is N x 2048 fp32, so peak is roughly N x 14 KB. At the default
--rows 2_800_000 that is ~40 GiB. Unset PYTORCH_CUDA_ALLOC_CONF -- on this
stack `expandable_segments:True` caps the caching allocator at 19 GiB of 432
(mi450_b0.md section 11) and this script will OOM long before it reaches a
production N.

EXAMPLES
--------
    # default: production shapes, 200 iterations, both kernels compared
    python3 repro_gfx1250_hstu_layer.py

    # sweep N across the int32 wrap row for D=512 (4_194_304)
    python3 repro_gfx1250_hstu_layer.py --rows 4194303,4194305,8388609 --iters 1

    # long soak, Triton only, looking for a nan (fastest per iteration)
    python3 repro_gfx1250_hstu_layer.py --iters 5000 --no-reference

    # exercise the dropout RNG path (disables the reference compare: the two
    # kernels draw different random numbers, so only the nan check is valid)
    python3 repro_gfx1250_hstu_layer.py --dropout 0.1
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("REPO_ROOT", "/workspace/recommendation"))

try:
    import fbgemm_gpu  # noqa: F401  -- registers torch.ops.fbgemm.*
except ImportError:
    # Only the PYTORCH reference path needs these; --no-reference works without.
    pass

from generative_recommenders.common import HammerKernel  # noqa: E402
from generative_recommenders.ops.hstu_compute import (  # noqa: E402
    hstu_compute_output,
    hstu_preprocess_and_attention,
)

# Production config -- dlrm_v4/configs.py get_hstu_configs() + yambda_5b.gin.
D = 512  # hstu_transducer_embedding_dim
NUM_HEADS = 4  # hstu_num_heads
ATTN_DIM = 128  # hstu_attn_qk_dim
HIDDEN_DIM = 128  # hstu_attn_linear_dim
MAX_SEQ_LEN = 4096  # get_hstu_configs.max_seq_len
NORM_EPS = 1e-6
WRAP_ROW = 2**31 // D  # 4_194_304 -- where `rows * stride_x` overflows int32

ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--rows", default="2800000",
                help="comma-separated total jagged rows N to test (default 2800000, "
                     "~the batch-1024 corpus average). N varies per batch in the real "
                     "run, so sweep it.")
ap.add_argument("--batch", type=int, default=1024, help="sequences per batch")
ap.add_argument("--iters", type=int, default=200, help="iterations per N")
ap.add_argument("--dropout", type=float, default=0.0,
                help="dropout ratio; any non-zero value disables the reference "
                     "compare, since the two kernels draw different RNG")
ap.add_argument("--no-reference", action="store_true",
                help="skip the PYTORCH reference (faster; only checks for nan)")
ap.add_argument("--no-backward", action="store_true", help="forward only")
ap.add_argument("--canary-gib", type=float, default=4.0, help="0 to disable")
ap.add_argument("--atol", type=float, default=2e-2)
ap.add_argument("--rtol", type=float, default=2e-2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stop-on-fail", action="store_true", default=True)
a = ap.parse_args()

dev = torch.device("cuda")
torch.manual_seed(a.seed)

props = torch.cuda.get_device_properties(0)
print(f"# {props.gcnArchName}  {props.total_memory / 2**30:.1f} GiB  torch={torch.__version__}")
print(f"# D={D} heads={NUM_HEADS} attn_dim={ATTN_DIM} hidden_dim={HIDDEN_DIM} "
      f"max_seq_len={MAX_SEQ_LEN}")
print(f"# int32 wrap row for stride={D}: {WRAP_ROW:,}")
print(f"# AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS')} "
      f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}")

reference = not a.no_reference and a.dropout == 0.0
if a.dropout != 0.0 and not a.no_reference:
    print("# NOTE: --dropout is non-zero, so the PYTORCH reference compare is OFF "
          "(the kernels draw different RNG). Only the nan check is meaningful.")
print(f"# reference_compare={reference} backward={not a.no_backward}")

# Canary allocated FIRST so it sits below everything the layer allocates. The
# allocator hands out DESCENDING addresses on this stack, so a negative wrapped
# offset lands below `y` -- which is where this buffer is.
canary = None
if a.canary_gib > 0:
    canary = torch.full((int(a.canary_gib * (1 << 30)) // 4,), 1.5,
                        device=dev, dtype=torch.float32)
    print(f"# canary {a.canary_gib} GiB @ {canary.data_ptr():#x}")


def make_lengths(total_rows: int, batch: int) -> torch.Tensor:
    """Jagged sequence lengths summing to exactly `total_rows`, each <= MAX_SEQ_LEN.

    Real batches are ragged, and raggedness is what decides which rows land in
    which tile, so this deliberately does not hand out equal lengths.
    """
    if total_rows > batch * MAX_SEQ_LEN:
        raise SystemExit(
            f"N={total_rows:,} needs > {batch} x {MAX_SEQ_LEN} rows; raise --batch "
            f"(min {-(-total_rows // MAX_SEQ_LEN)})")
    w = torch.rand(batch, device=dev) + 0.1
    lengths = (w / w.sum() * total_rows).long().clamp(1, MAX_SEQ_LEN)
    # Fix up the rounding drift so the total is exact.
    while int(lengths.sum()) != total_rows:
        delta = total_rows - int(lengths.sum())
        room = (MAX_SEQ_LEN - lengths) if delta > 0 else (lengths - 1)
        idx = torch.nonzero(room > 0).flatten()
        if idx.numel() == 0:
            raise SystemExit("cannot fit N into the given --batch")
        take = min(abs(delta), int(room[idx[0]]))
        lengths[idx[0]] += take if delta > 0 else -take
    return lengths


def run_layer(x, w, kernel, seq_offsets, max_seq_len, training):
    """One full HSTU layer: preprocess + attention, then the output projection."""
    attn, u, _, _ = hstu_preprocess_and_attention(
        x=x,
        norm_weight=w["in_w"], norm_bias=w["in_b"], norm_eps=NORM_EPS,
        num_heads=NUM_HEADS, attn_dim=ATTN_DIM, hidden_dim=HIDDEN_DIM,
        uvqk_weight=w["uvqk_w"], uvqk_bias=w["uvqk_b"],
        max_seq_len=max_seq_len, seq_offsets=seq_offsets,
        attn_alpha=1.0 / (ATTN_DIM**0.5), causal=True, num_targets=None,
        max_attn_len=0, contextual_seq_len=0,
        recompute_uvqk_in_backward=False, recompute_normed_x_in_backward=False,
        sort_by_length=False, kernel=kernel,
    )
    y = hstu_compute_output(
        attn=attn, u=u, x=x,
        norm_weight=w["out_w"], norm_bias=w["out_b"], norm_eps=NORM_EPS,
        output_weight=w["out_proj"],
        num_heads=NUM_HEADS, linear_dim=HIDDEN_DIM,
        dropout_ratio=a.dropout, training=training,
        concat_u=False, concat_x=True, mul_u_activation_type="silu",
        group_norm=False, recompute_y_in_backward=False, kernel=kernel,
    )
    return y


def check(name, t, failures):
    if t is None:
        return
    n_nonfinite = int((~torch.isfinite(t)).sum())
    if n_nonfinite:
        failures.append(f"{name}: {n_nonfinite:,} non-finite of {t.numel():,}")


def main() -> int:
    ns = [int(v) for v in a.rows.split(",") if v.strip()]
    overall_bad = 0

    for N in ns:
        print(f"\n=== N={N:,} rows  ({'ABOVE' if N > WRAP_ROW else 'below'} the "
              f"D={D} wrap row {WRAP_ROW:,})  batch={a.batch}  iters={a.iters} ===")

        # Weights are held fixed across iterations; only the data changes, which
        # keeps a divergence attributable to the data/shape rather than to drift.
        g = torch.Generator(device=dev).manual_seed(a.seed)
        w = {
            "in_w": torch.ones(D, device=dev),
            "in_b": torch.zeros(D, device=dev),
            "uvqk_w": torch.randn(D, 2 * NUM_HEADS * (HIDDEN_DIM + ATTN_DIM),
                                  device=dev, generator=g) * 0.02,
            "uvqk_b": torch.zeros(2 * NUM_HEADS * (HIDDEN_DIM + ATTN_DIM), device=dev),
            "out_w": torch.ones(NUM_HEADS * HIDDEN_DIM, device=dev),
            "out_b": torch.zeros(NUM_HEADS * HIDDEN_DIM, device=dev),
            "out_proj": torch.randn(NUM_HEADS * HIDDEN_DIM + D, D,
                                    device=dev, generator=g) * 0.02,
        }
        if not a.no_backward:
            for v in w.values():
                v.requires_grad_(True)

        t0 = time.time()
        for it in range(a.iters):
            failures = []
            lengths = make_lengths(N, a.batch)
            seq_offsets = torch.cat([
                torch.zeros(1, device=dev, dtype=torch.long), lengths.cumsum(0)
            ]).long()
            max_seq_len = int(lengths.max())

            x = torch.randn(N, D, device=dev, generator=g)
            if not a.no_backward:
                x.requires_grad_(True)

            y = run_layer(x, w, HammerKernel.TRITON, seq_offsets, max_seq_len,
                          training=a.dropout > 0.0)
            check("triton y", y, failures)

            if not a.no_backward:
                y.sum().backward()
                check("triton dx", x.grad, failures)
                for k, v in w.items():
                    check(f"triton d{k}", v.grad, failures)

            if reference:
                with torch.no_grad():
                    y_ref = run_layer(x.detach(), {k: v.detach() for k, v in w.items()},
                                      HammerKernel.PYTORCH, seq_offsets, max_seq_len,
                                      training=False)
                check("pytorch y_ref", y_ref, failures)
                if torch.isfinite(y).all() and torch.isfinite(y_ref).all():
                    bad = ~torch.isclose(y.detach(), y_ref, atol=a.atol, rtol=a.rtol)
                    nbad = int(bad.sum())
                    if nbad:
                        idx = torch.nonzero(bad)[0].tolist()
                        failures.append(
                            f"TRITON vs PYTORCH: {nbad:,} of {y.numel():,} differ "
                            f"(max |d| {float((y.detach() - y_ref).abs().max()):.3e}); "
                            f"first at row {idx[0]:,} col {idx[1]}")

            if canary is not None:
                nbad = int((canary != 1.5).sum())
                if nbad:
                    failures.append(f"CANARY CLOBBERED: {nbad:,} of {canary.numel():,} "
                                    f"elements -- a wild store landed in it")

            if failures:
                overall_bad += 1
                print(f"  !! FAIL  N={N:,} iter={it} max_seq_len={max_seq_len} "
                      f"lengths[min/mean/max]={int(lengths.min())}/"
                      f"{float(lengths.float().mean()):.0f}/{int(lengths.max())}")
                for f in failures:
                    print(f"       {f}")
                if a.stop_on_fail:
                    return 1
            elif it % 25 == 0:
                torch.cuda.synchronize()
                print(f"  ok   iter {it:5d}  max_seq_len={max_seq_len:5d}  "
                      f"{(time.time() - t0) / max(it, 1) * 1000:7.1f} ms/iter")

            if not a.no_backward:
                x.grad = None
                for v in w.values():
                    v.grad = None
            del x, y

        torch.cuda.synchronize()
        print(f"  PASS N={N:,}: {a.iters} iterations clean in {time.time() - t0:.1f}s")

    print(f"\n{'FAIL' if overall_bad else 'PASS'}: {overall_bad} failing configuration(s)")
    return 1 if overall_bad else 0


if __name__ == "__main__":
    sys.exit(main())
