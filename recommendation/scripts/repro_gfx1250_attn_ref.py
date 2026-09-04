#!/usr/bin/env python3
"""Standalone gfx1250 probe for defect [3]: HSTU attention against its reference.

**As of 2026-09-04 this script PASSES on every configuration tried, including the
true e2e shape.** It is checked in as a *control*: it is the strongest statement
available that defect [3]'s `nan` does not reduce to the attention kernel in
isolation. Read the pass as evidence, not as a missing feature.

Defect [3] in docs/mi450.md is `train_loss=nan`, which at batch 128 appears
within ~50 steps in most runs while the PyTorch kernels stay finite. This script
takes the attention op standalone and checks it two ways on identical inputs.

Why a *correctness* probe and not a finiteness one: the e2e `nan` is not the
attention kernel emitting `nan` directly. At the first non-finite forward, 80
embedding tables are already non-finite, so the chain is small wrong gradients
-> a poisoned table -> `nan` several steps later. `repro_gfx1250_attn_bwd.py
--check-finite --batch-size 128` passes 300 iterations, so the kernel does not
produce `nan` on its own; what would have to be caught is the wrong value
preceding it.

What this varies over `scripts/attn_check.py`, which also passes: that check is
eight fixed single-shot shapes, while this loops fresh random jagged layouts at
a configurable row count -- what the trainer feeds, and the axis [3] tracks.

Two modes, because the dense reference does not scale to the real shape:

  * default: compare against `HammerKernel.PYTORCH` on the same inputs. Exact
    but limited to short sequences, since the reference pads to dense and this
    stack fails single allocations around 4 GiB even with 400+ GiB free.
  * ``--chunk N``: compare the whole batch against the same kernel run on
    chunks of N rows. Causal jagged attention rows are independent, so row
    count must not change any row's result. No dense reference, so this reaches
    batch 128 at `--max-uih-len 4086`, where it is **bit-exact** (0.0 max diff)
    over 40 iterations. `--chunk` equal to `--batch-size` is the harness
    self-test and must report 0.0.

Needs the repo on PYTHONPATH. No dataset. One process per run::

    # batch-invariance at the true e2e shape -- the strongest check here
    AMDGCN_USE_BUFFER_OPS=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python scripts/repro_gfx1250_attn_ref.py \
        --batch-size 128 --chunk 8 --max-uih-len 4086 --contextual 8 --iters 40

    # dense reference, short sequences only
    AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_attn_ref.py \
      --batch-size 128 --max-uih-len 512 --iters 200
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import triton

import fbgemm_gpu  # noqa: F401  registers the fbgemm ops the reference path needs
from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.ops.hstu_attention import hstu_mha

# torch.testing.assert_close defaults for bfloat16, same as attn_check.py.
ATOL, RTOL = 1e-5, 0.016
DTYPE = torch.bfloat16


def _print_env() -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} | hip {torch.version.hip}", flush=True)
    print(f"triton {triton.__version__} | arch {props.gcnArchName}", flush=True)
    print(
        f"AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS', '<unset>')} "
        f"TRITON_ALLOW_PIPELINING="
        f"{os.environ.get('TRITON_ALLOW_PIPELINING', '<unset>')}",
        flush=True,
    )


def _worst(ref: torch.Tensor, got: torch.Tensor):
    """Return (bad element count, max abs diff, all-finite) on CPU copies."""
    r = ref.detach().float().cpu()
    g = got.detach().float().cpu()
    diff = (r - g).abs()
    bad = int((diff > ATOL + RTOL * r.abs()).sum())
    return bad, float(diff.max()), bool(torch.isfinite(g).all())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-uih-len", type=int, default=512)
    p.add_argument("--min-history", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--attn-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--targets", type=int, default=1)
    p.add_argument("--contextual", type=int, default=0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument(
        "--skip-backward",
        action="store_true",
        help="compare the forward only; the backward is the suspect, so this is "
        "a control rather than the probe",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="batch-invariance mode: instead of the dense PyTorch reference, "
        "run the Triton kernels on the whole batch and again on chunks of this "
        "many rows, and require the two to agree. Rows of a causal jagged "
        "attention are independent, so row count must not change any row's "
        "result. Needs no dense reference, which is what makes the true e2e "
        "sequence length reachable at batch 128",
    )
    args = p.parse_args()

    if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
        raise RuntimeError("set AMDGCN_USE_BUFFER_OPS=0 before running this repro")

    _print_env()
    max_seq_len = args.max_uih_len + args.targets + args.contextual
    if args.chunk:
        print(
            f"-- batch={args.batch_size} max_seq_len={max_seq_len} "
            f"heads={args.heads} -> batch-invariance against chunks of "
            f"{args.chunk} rows, no dense reference",
            flush=True,
        )
    else:
        dense_gib = (
            args.batch_size * args.heads * max_seq_len * max_seq_len * 4 / (1 << 30)
        )
        print(
            f"-- batch={args.batch_size} max_seq_len={max_seq_len} "
            f"heads={args.heads} -> reference scores ~{dense_gib:.1f} GiB",
            flush=True,
        )
        if dense_gib > 2:
            raise SystemExit(
                f"the dense reference would need ~{dense_gib:.0f} GiB of "
                "attention scores, and this stack fails single allocations at "
                "~4 GiB even with hundreds of GiB free. Lower --max-uih-len or "
                "--batch-size, or use --chunk for the reference-free check"
            )

    set_dev_mode(True)
    device = torch.device("cuda")
    rng = torch.Generator().manual_seed(args.seed)
    alpha = 1.0 / (args.attn_dim**0.5)

    def call(kernel, q, k, v, offsets, num_targets):
        return hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=offsets,
            causal=True,
            num_targets=num_targets,
            dropout_pr=0.0,
            max_attn_len=0,
            contextual_seq_len=args.contextual,
            kernel=kernel,
            enable_tma=False,
        )

    worst_seen = 0.0
    for step in range(args.iters):
        uih = torch.randint(
            args.min_history,
            args.max_uih_len,
            (args.batch_size,),
            generator=rng,
            dtype=torch.int64,
        ).to(device)
        num_targets = torch.full_like(uih, args.targets)
        lengths = uih + args.targets + args.contextual
        offsets = torch.zeros((args.batch_size + 1,), dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)
        total = int(offsets[-1].item())

        def mk(dim):
            return (
                torch.empty((total, args.heads, dim), dtype=DTYPE, device=device)
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )

        q, k, v = mk(args.attn_dim), mk(args.attn_dim), mk(args.hidden_dim)
        if args.print_every and step % args.print_every == 0:
            print(f"-- step={step} N={total} worst_so_far={worst_seen:.3e}", flush=True)

        if args.chunk:
            # Same kernel, same inputs, same max_seq_len -- only the row count
            # differs. Every row must come back identical.
            full_out = call(HammerKernel.TRITON, q, k, v, offsets, num_targets)
            dout = torch.randn_like(full_out) * 0.1
            full_grads = torch.autograd.grad(full_out, [q, k, v], dout)
            # Per-tensor lists of the whole-batch slice and the chunked result,
            # in row order, so they concatenate back into comparable tensors.
            whole = {"out": [], "dq": [], "dk": [], "dv": []}
            chunked = {"out": [], "dq": [], "dk": [], "dv": []}
            for lo in range(0, args.batch_size, args.chunk):
                hi = min(lo + args.chunk, args.batch_size)
                t0, t1 = int(offsets[lo].item()), int(offsets[hi].item())
                sub_off = offsets[lo : hi + 1] - offsets[lo]
                sq = q[t0:t1].detach().clone().requires_grad_()
                sk = k[t0:t1].detach().clone().requires_grad_()
                sv = v[t0:t1].detach().clone().requires_grad_()
                sub_out = call(
                    HammerKernel.TRITON, sq, sk, sv, sub_off, num_targets[lo:hi]
                )
                whole["out"].append(full_out[t0:t1])
                chunked["out"].append(sub_out)
                if not args.skip_backward:
                    sub_grads = torch.autograd.grad(
                        sub_out, [sq, sk, sv], dout[t0:t1]
                    )
                    for name, full_g, sub_g in zip(
                        ("dq", "dk", "dv"), full_grads, sub_grads, strict=True
                    ):
                        whole[name].append(full_g[t0:t1])
                        chunked[name].append(sub_g)
            named = [
                (name, torch.cat(whole[name]), torch.cat(chunked[name]))
                for name in ("out", "dq", "dk", "dv")
                if whole[name]
            ]
        else:
            ref_out = call(HammerKernel.PYTORCH, q, k, v, offsets, num_targets)
            got_out = call(HammerKernel.TRITON, q, k, v, offsets, num_targets)
            named = [("out", ref_out, got_out)]

            if not args.skip_backward:
                dout = torch.randn_like(ref_out) * 0.1
                ref_grads = torch.autograd.grad(
                    ref_out, [q, k, v], dout, retain_graph=True
                )
                got_grads = torch.autograd.grad(got_out, [q, k, v], dout)
                named += list(
                    zip(("dq", "dk", "dv"), ref_grads, got_grads, strict=True)
                )

        failures = []
        for name, ref, got in named:
            bad, mx, finite = _worst(ref, got)
            worst_seen = max(worst_seen, mx)
            if bad or not finite:
                failures.append((name, bad, mx, finite, ref.numel()))
        if failures:
            print(
                f"!! MISMATCH step={step} N={total} batch={args.batch_size} "
                f"lengths={lengths.tolist()}",
                flush=True,
            )
            for name, bad, mx, finite, numel in failures:
                print(
                    f"   {name}: {bad} of {numel} outside tolerance, "
                    f"max|diff|={mx:.3e}, all_finite={finite}",
                    flush=True,
                )
            return 1

    print(
        f"-- PASS ({args.iters} iterations, batch {args.batch_size}, "
        f"worst max|diff| {worst_seen:.3e} within atol={ATOL} rtol={RTOL})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
