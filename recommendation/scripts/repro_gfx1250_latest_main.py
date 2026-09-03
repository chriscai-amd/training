#!/usr/bin/env python3
"""Reproduce the gfx1250 first-step memory-access fault without yambda data.

This is defect 6 in docs/mi450.md. Isolated HSTU layers pass; the e2e trainer
dies at ``gstep=0`` right after the first TBE HIP warning. This script walks
the same first-step kernel mix the trainer runs — TBE lookup, jagged concat of
contextual tokens, timestamp/position embeddings, then the HSTU encoder —
with synthetic indices and lengths. No dataset.

    AMDGCN_USE_BUFFER_OPS=0 AMD_SERIALIZE_KERNEL=3 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
      python scripts/repro_gfx1250_latest_main.py

    # Default --phase connected: one autograd graph TBE → jagged → position →
    # HSTU → backward into TBE. Isolated phases still work:
    #   --phase tbe,jagged,position,hstu
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python scripts/repro_gfx1250_latest_main.py
    # Under True, TBE tables are capped at 3x250k so construction fits.

Needs the repo on ``PYTHONPATH`` and ``fbgemm_gpu``. One process per run.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")

import torch
import triton

import fbgemm_gpu  # noqa: F401

from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output,
    hstu_preprocess_and_attention,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.position import add_timestamp_positional_embeddings


# gin-default yambda-5b HSTU ranker (see DlrmHSTUConfig in the e2e log).
BATCH = 8
HEADS = 4
EMBED_DIM = 512
ATTN_DIM = 128
HIDDEN_DIM = 128
MAX_SEQ_LEN = 4096
CONTEXTUAL_SEQ_LEN = 8
MAX_TARGETS = 1
DROPOUT = 0.1
NUM_LAYERS = 3
DTYPE = torch.bfloat16

# EMBEDDING_ROW_SCALE=0.25 tables from the gstep=0 e2e log.
TBE_ROW_COUNTS = (
    2_347_656,
    323_348,
    841_923,
    250_000,
    25_000_000,
    10_000_000,
    6_000_000,
    10_000_000,
    8_000_000,
    500_000,
    10_000_000,
)
def _expandable_segments_on() -> bool:
    return "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


def _require_buffer_ops_off() -> None:
    if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
        raise RuntimeError("set AMDGCN_USE_BUFFER_OPS=0 before running this repro")


def _print_env() -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} | hip {torch.version.hip}", flush=True)
    print(f"triton {triton.__version__} | arch {props.gcnArchName}", flush=True)
    print(
        f"AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS', '<unset>')} "
        f"AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL', '<unset>')} "
        f"TRITON_ALLOW_PIPELINING={os.environ.get('TRITON_ALLOW_PIPELINING', '<unset>')} "
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}",
        flush=True,
    )
    from generative_recommenders import common
    from generative_recommenders.ops.utils import use_separated_rng_ln_mul_dropout

    print(f"_NO_PIPELINING = {common._NO_PIPELINING}", flush=True)
    print(
        f"use_separated_rng_ln_mul_dropout = {use_separated_rng_ln_mul_dropout()}",
        flush=True,
    )


def _offsets(lengths: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros(
        (lengths.numel() + 1,), dtype=torch.int64, device=lengths.device
    )
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return offsets


def _batch_layout(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    uih = torch.tensor(
        [499, 500, 1405, 1405, 1405, 1405, 1405, 1405], device=device
    )
    seq_lengths = uih + MAX_TARGETS
    num_targets = torch.full((BATCH,), MAX_TARGETS, dtype=torch.int64, device=device)
    seq_offsets = _offsets(seq_lengths)
    return seq_lengths, seq_offsets, num_targets, int(seq_lengths.max().item())


def _tbe_row_counts() -> tuple[int, ...]:
    if _expandable_segments_on():
        # Successive ~4 GiB allocs already OOM under True; keep tables tiny so
        # the connected graph can still run with the e2e allocator knob.
        return (250_000, 250_000, 250_000)
    return TBE_ROW_COUNTS


def _make_tbes(device: torch.device, row_counts: tuple[int, ...]):
    from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
        BoundsCheckMode,
        ComputeDevice,
        EmbeddingLocation,
        OptimType,
        PoolingMode,
        SparseType,
        SplitTableBatchedEmbeddingBagsCodegen,
    )

    tbes = []
    for i, rows in enumerate(row_counts):
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (rows, EMBED_DIM, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
            ],
            pooling_mode=PoolingMode.NONE,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=1e-6,
            weights_precision=SparseType.FP32,
            output_dtype=SparseType.FP32,
            device=device,
            bounds_check_mode=BoundsCheckMode.WARNING,
            stochastic_rounding=True,
        )
        tbes.append(tbe)
        print(f"-- tbe: table {i} rows={rows:,} constructed", flush=True)
    return tbes


def phase_tbe(device: torch.device, seq_lengths: torch.Tensor) -> None:
    # One TBE per table. A single SplitTable with all 11 specs tries to allocate
    # ~140 GiB in one HIP block and OOM'd with 432 GiB free; e2e succeeds because
    # TorchRec places each table as its own allocation.
    row_counts = _tbe_row_counts()
    print(
        f"-- tbe: {len(row_counts)} tables, "
        f"{sum(row_counts):,} rows x {EMBED_DIM} fp32 "
        f"(~{sum(row_counts) * EMBED_DIM * 4 / 1e9:.1f} GB weights, per-table alloc)",
        flush=True,
    )
    tbes = _make_tbes(device, row_counts)
    offsets = _offsets(seq_lengths)
    n = int(offsets[-1].item())
    outs = []
    for i, (tbe, rows) in enumerate(zip(tbes, row_counts)):
        indices = torch.randint(0, rows, (n,), device=device)
        print(f"-- tbe: table {i} forward n={n}", flush=True)
        outs.append(tbe(indices, offsets))
    torch.cuda.synchronize()
    print(f"-- tbe: out0={tuple(outs[0].shape)} {outs[0].dtype}", flush=True)
    torch.autograd.backward(
        outs, [torch.randn_like(o) * 0.01 for o in outs]
    )
    torch.cuda.synchronize()
    print("-- tbe: PASS", flush=True)
    del tbes, outs
    torch.cuda.empty_cache()


def phase_jagged(
    device: torch.device, seq_lengths: torch.Tensor, seq_offsets: torch.Tensor
) -> torch.Tensor:
    total = int(seq_offsets[-1].item())
    max_seq = int(seq_lengths.max().item())
    left = (
        torch.empty((BATCH * CONTEXTUAL_SEQ_LEN, EMBED_DIM), device=device, dtype=DTYPE)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    right = (
        torch.empty((total, EMBED_DIM), device=device, dtype=DTYPE)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    print(
        f"-- jagged: concat contextual {CONTEXTUAL_SEQ_LEN} + seq max={max_seq} "
        f"N_right={total} D={EMBED_DIM}",
        flush=True,
    )
    y = concat_2D_jagged(
        max_seq_len=CONTEXTUAL_SEQ_LEN + max_seq,
        values_left=left,
        values_right=right,
        max_len_left=CONTEXTUAL_SEQ_LEN,
        max_len_right=max_seq,
        offsets_left=None,
        offsets_right=seq_offsets,
        kernel=HammerKernel.TRITON,
    )
    torch.cuda.synchronize()
    print(f"-- jagged: y={tuple(y.shape)}", flush=True)
    y.backward(torch.randn_like(y) * 0.1)
    torch.cuda.synchronize()
    print("-- jagged: PASS", flush=True)
    return y.detach()


def phase_position(
    device: torch.device,
    seq_embeddings: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    lengths = seq_lengths + CONTEXTUAL_SEQ_LEN
    offsets = _offsets(lengths)
    max_seq = int(lengths.max().item())
    total = int(offsets[-1].item())
    if seq_embeddings.shape[0] != total:
        seq_embeddings = (
            torch.empty((total, EMBED_DIM), device=device, dtype=DTYPE)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
    else:
        seq_embeddings = seq_embeddings.detach().to(DTYPE).requires_grad_()
    pos_w = (
        torch.empty((MAX_SEQ_LEN, EMBED_DIM), device=device, dtype=torch.float32)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    ts_w = (
        torch.empty((1001, EMBED_DIM), device=device, dtype=torch.float32)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    timestamps = torch.randint(1, 10_000_000, (total,), device=device)
    print(
        f"-- position: N={total} max_seq_len={max_seq} ctx={CONTEXTUAL_SEQ_LEN}",
        flush=True,
    )
    y = add_timestamp_positional_embeddings(
        alpha=EMBED_DIM ** 0.5,
        max_seq_len=max_seq,
        max_contextual_seq_len=CONTEXTUAL_SEQ_LEN,
        position_embeddings_weight=pos_w,
        timestamp_embeddings_weight=ts_w,
        seq_offsets=offsets,
        seq_lengths=lengths,
        seq_embeddings=seq_embeddings,
        timestamps=timestamps,
        num_targets=num_targets,
        interleave_targets=False,
        kernel=HammerKernel.TRITON,
    )
    torch.cuda.synchronize()
    print(f"-- position: y={tuple(y.shape)}", flush=True)
    y.backward(torch.randn_like(y) * 0.1)
    torch.cuda.synchronize()
    print("-- position: PASS", flush=True)
    return y.detach()


def _one_hstu_layer(
    x: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    d = x.shape[1]
    uvqk_out = 2 * HEADS * (HIDDEN_DIM + ATTN_DIM)
    in_w = torch.empty((d,), device=device, dtype=dtype).uniform_(-0.1, 0.1).requires_grad_()
    in_b = torch.empty_like(in_w).uniform_(-0.1, 0.1).requires_grad_()
    uvqk_w = (
        torch.empty((d, uvqk_out), device=device, dtype=dtype)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    uvqk_b = (
        torch.empty((uvqk_out,), device=device, dtype=dtype)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    out_w = (
        torch.empty((3 * HEADS * HIDDEN_DIM, d), device=device, dtype=dtype)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    out_nw = (
        torch.empty((HEADS * HIDDEN_DIM,), device=device, dtype=dtype)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    out_nb = torch.empty_like(out_nw).uniform_(-0.1, 0.1).requires_grad_()
    u, attn, _, _ = hstu_preprocess_and_attention(
        x=x,
        norm_weight=in_w,
        norm_bias=in_b,
        norm_eps=1e-6,
        num_heads=HEADS,
        attn_dim=ATTN_DIM,
        hidden_dim=HIDDEN_DIM,
        uvqk_weight=uvqk_w,
        uvqk_bias=uvqk_b,
        max_seq_len=max_seq_len,
        seq_offsets=seq_offsets,
        attn_alpha=1.0 / (ATTN_DIM ** 0.5),
        causal=True,
        num_targets=num_targets,
        max_attn_len=0,
        contextual_seq_len=CONTEXTUAL_SEQ_LEN,
        recompute_uvqk_in_backward=False,
        recompute_normed_x_in_backward=False,
        sort_by_length=False,
        kernel=HammerKernel.TRITON,
    )
    return hstu_compute_output(
        attn=attn,
        u=u,
        x=x,
        norm_weight=out_nw,
        norm_bias=out_nb,
        norm_eps=1e-6,
        dropout_ratio=DROPOUT,
        output_weight=out_w,
        group_norm=False,
        num_heads=HEADS,
        linear_dim=HIDDEN_DIM,
        concat_u=True,
        concat_x=True,
        mul_u_activation_type="none",
        training=True,
        kernel=HammerKernel.TRITON,
        recompute_y_in_backward=False,
    )


def phase_hstu(
    device: torch.device,
    seq_embeddings: torch.Tensor | None,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> None:
    lengths = seq_lengths + CONTEXTUAL_SEQ_LEN
    offsets = _offsets(lengths)
    total = int(offsets[-1].item())
    max_seq = int(lengths.max().item())
    if seq_embeddings is None or seq_embeddings.shape[0] != total:
        x = (
            torch.empty((total, EMBED_DIM), device=device, dtype=DTYPE)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
    else:
        x = seq_embeddings.detach().to(DTYPE).requires_grad_()
    print(
        f"-- hstu: N={total} max_seq_len={max_seq} layers={NUM_LAYERS} dropout={DROPOUT}",
        flush=True,
    )
    for layer in range(NUM_LAYERS):
        print(f"-- hstu layer {layer} forward", flush=True)
        y = _one_hstu_layer(x, offsets, num_targets, max_seq)
        torch.cuda.synchronize()
        print(f"-- hstu layer {layer} backward y={tuple(y.shape)}", flush=True)
        y.backward(torch.randn_like(y) * 0.1)
        torch.cuda.synchronize()
        print(f"-- hstu layer {layer}: PASS", flush=True)
        x = x.detach().clone().requires_grad_()
    print("-- hstu: PASS", flush=True)


def phase_connected(
    device: torch.device,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> None:
    """One autograd graph: TBE lookup → jagged → position → HSTU → backward into TBE."""
    row_counts = _tbe_row_counts()
    print(
        f"-- connected: {len(row_counts)} tables, "
        f"{sum(row_counts):,} rows, one backward through TBE+HSTU",
        flush=True,
    )
    tbes = _make_tbes(device, row_counts)
    offsets = _offsets(seq_lengths)
    n = int(offsets[-1].item())
    seq_emb = None
    for i, (tbe, rows) in enumerate(zip(tbes, row_counts)):
        indices = torch.randint(0, rows, (n,), device=device)
        print(f"-- connected: table {i} lookup n={n}", flush=True)
        out = tbe(indices, offsets)
        seq_emb = out if seq_emb is None else seq_emb + out
    assert seq_emb is not None
    ctx_lengths = torch.full(
        (BATCH,), CONTEXTUAL_SEQ_LEN, dtype=torch.int64, device=device
    )
    ctx_offsets = _offsets(ctx_lengths)
    ctx_n = int(ctx_offsets[-1].item())
    ctx = tbes[0](
        torch.randint(0, row_counts[0], (ctx_n,), device=device),
        ctx_offsets,
    )
    max_seq = int(seq_lengths.max().item())
    print(
        f"-- connected: concat ctx={tuple(ctx.shape)} seq={tuple(seq_emb.shape)}",
        flush=True,
    )
    jagged = concat_2D_jagged(
        max_seq_len=CONTEXTUAL_SEQ_LEN + max_seq,
        values_left=ctx,
        values_right=seq_emb,
        max_len_left=CONTEXTUAL_SEQ_LEN,
        max_len_right=max_seq,
        offsets_left=None,
        offsets_right=offsets,
        kernel=HammerKernel.TRITON,
    )
    lengths = seq_lengths + CONTEXTUAL_SEQ_LEN
    pos_offsets = _offsets(lengths)
    total = int(pos_offsets[-1].item())
    max_full = int(lengths.max().item())
    pos_w = (
        torch.empty((MAX_SEQ_LEN, EMBED_DIM), device=device, dtype=torch.float32)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    ts_w = (
        torch.empty((1001, EMBED_DIM), device=device, dtype=torch.float32)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    timestamps = torch.randint(1, 10_000_000, (total,), device=device)
    print(f"-- connected: position N={total}", flush=True)
    positioned = add_timestamp_positional_embeddings(
        alpha=EMBED_DIM ** 0.5,
        max_seq_len=max_full,
        max_contextual_seq_len=CONTEXTUAL_SEQ_LEN,
        position_embeddings_weight=pos_w,
        timestamp_embeddings_weight=ts_w,
        seq_offsets=pos_offsets,
        seq_lengths=lengths,
        seq_embeddings=jagged.to(DTYPE),
        timestamps=timestamps,
        num_targets=num_targets,
        interleave_targets=False,
        kernel=HammerKernel.TRITON,
    )
    x = positioned
    for layer in range(NUM_LAYERS):
        print(f"-- connected: hstu layer {layer} forward", flush=True)
        x = _one_hstu_layer(x, pos_offsets, num_targets, max_full)
        torch.cuda.synchronize()
        print(f"-- connected: hstu layer {layer} y={tuple(x.shape)}", flush=True)
    print("-- connected: backward through HSTU+TBE", flush=True)
    x.backward(torch.randn_like(x) * 0.1)
    torch.cuda.synchronize()
    print("-- connected: PASS", flush=True)


MIN_HISTORY = 64
HISTORY_LENGTH = 4086


def _random_batch_layout(
    device: torch.device, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """A yambda-5b-shaped batch: per-row history in [MIN_HISTORY, HISTORY_LENGTH).

    Sampled on the host so drawing a layout costs no device sync -- the point of
    the step loop is that every step gets different jagged extents, the way the
    real dataloader feeds the trainer.
    """
    uih = torch.randint(
        MIN_HISTORY, HISTORY_LENGTH, (BATCH,), generator=rng, dtype=torch.int64
    ).to(device)
    seq_lengths = uih + MAX_TARGETS
    num_targets = torch.full((BATCH,), MAX_TARGETS, dtype=torch.int64, device=device)
    return seq_lengths, _offsets(seq_lengths), num_targets, int(uih.max()) + MAX_TARGETS


def phase_steps(device: torch.device, iters: int) -> None:
    """Many training-shaped steps, each with a different jagged layout.

    ``--phase connected`` reuses one fixed layout and passes; the e2e trainer
    faults after tens to thousands of steps. The variable that phase misses is
    the per-step change in sequence extents, so this loop redraws the layout
    every step and runs the full Triton encoder forward/backward on it.
    """
    rng = torch.Generator().manual_seed(1234)
    print(f"-- steps: {iters} iterations, fresh jagged layout per step", flush=True)
    for step in range(iters):
        seq_lengths, seq_offsets, num_targets, _ = _random_batch_layout(device, rng)
        lengths = seq_lengths + CONTEXTUAL_SEQ_LEN
        offsets = _offsets(lengths)
        total = int(offsets[-1].item())
        max_full = int(lengths.max().item())
        max_seq = int(seq_lengths.max().item())

        ctx = (
            torch.empty(
                (BATCH * CONTEXTUAL_SEQ_LEN, EMBED_DIM), device=device, dtype=DTYPE
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        seq_emb = (
            torch.empty(
                (int(seq_offsets[-1].item()), EMBED_DIM), device=device, dtype=DTYPE
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        jagged = concat_2D_jagged(
            max_seq_len=CONTEXTUAL_SEQ_LEN + max_seq,
            values_left=ctx,
            values_right=seq_emb,
            max_len_left=CONTEXTUAL_SEQ_LEN,
            max_len_right=max_seq,
            offsets_left=None,
            offsets_right=seq_offsets,
            kernel=HammerKernel.TRITON,
        )
        pos_w = (
            torch.empty((MAX_SEQ_LEN, EMBED_DIM), device=device, dtype=torch.float32)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        ts_w = (
            torch.empty((1001, EMBED_DIM), device=device, dtype=torch.float32)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        x = add_timestamp_positional_embeddings(
            alpha=EMBED_DIM**0.5,
            max_seq_len=max_full,
            max_contextual_seq_len=CONTEXTUAL_SEQ_LEN,
            position_embeddings_weight=pos_w,
            timestamp_embeddings_weight=ts_w,
            seq_offsets=offsets,
            seq_lengths=lengths,
            seq_embeddings=jagged.to(DTYPE),
            timestamps=torch.randint(1, 10_000_000, (total,), device=device),
            num_targets=num_targets,
            interleave_targets=False,
            kernel=HammerKernel.TRITON,
        )
        for _ in range(NUM_LAYERS):
            x = _one_hstu_layer(x, offsets, num_targets, max_full)
        x.backward(torch.randn_like(x) * 0.1)
        if step % 25 == 0:
            torch.cuda.synchronize()
            print(
                f"-- steps: {step} lengths={seq_lengths.tolist()} N={total}",
                flush=True,
            )
    torch.cuda.synchronize()
    print(f"-- steps: PASS ({iters} iterations)", flush=True)


def phase_prep(device: torch.device, iters: int) -> None:
    """Minimal reproducer: only ``hstu_preprocess_and_attention``, fresh shapes.

    Per-op bisect of the e2e trainer (scripts/bisect_hooks/sitecustomize.py)
    puts every other HSTU op group on Triton for 1000+ steps without a fault --
    including ``hstu_mha`` on its own -- while this fused LN + uvqk addmm +
    attention path faults on its own within tens of steps. So the loop below is
    the whole reproducer: no TBE, no dataloader, no TorchRec, no dataset.
    """
    rng = torch.Generator().manual_seed(1234)
    print(
        f"-- prep: {iters} iterations of hstu_preprocess_and_attention "
        f"(fwd+bwd), fresh jagged layout per step",
        flush=True,
    )
    for step in range(iters):
        seq_lengths, _, num_targets, _ = _random_batch_layout(device, rng)
        lengths = seq_lengths + CONTEXTUAL_SEQ_LEN
        offsets = _offsets(lengths)
        total = int(offsets[-1].item())
        max_full = int(lengths.max().item())
        x = (
            torch.empty((total, EMBED_DIM), device=device, dtype=DTYPE)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        uvqk_out = 2 * HEADS * (HIDDEN_DIM + ATTN_DIM)
        norm_w = (
            torch.empty((EMBED_DIM,), device=device, dtype=DTYPE)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_b = torch.empty_like(norm_w).uniform_(-0.1, 0.1).requires_grad_()
        uvqk_w = (
            torch.empty((EMBED_DIM, uvqk_out), device=device, dtype=DTYPE)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        uvqk_b = (
            torch.empty((uvqk_out,), device=device, dtype=DTYPE)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        u, attn, _, _ = hstu_preprocess_and_attention(
            x=x,
            norm_weight=norm_w,
            norm_bias=norm_b,
            norm_eps=1e-6,
            num_heads=HEADS,
            attn_dim=ATTN_DIM,
            hidden_dim=HIDDEN_DIM,
            uvqk_weight=uvqk_w,
            uvqk_bias=uvqk_b,
            max_seq_len=max_full,
            seq_offsets=offsets,
            attn_alpha=1.0 / (ATTN_DIM**0.5),
            causal=True,
            num_targets=num_targets,
            max_attn_len=0,
            contextual_seq_len=CONTEXTUAL_SEQ_LEN,
            recompute_uvqk_in_backward=False,
            recompute_normed_x_in_backward=False,
            sort_by_length=False,
            kernel=HammerKernel.TRITON,
        )
        (u.sum() + attn.sum()).backward()
        if step % 25 == 0:
            torch.cuda.synchronize()
            print(
                f"-- prep: {step} lengths={seq_lengths.tolist()} N={total}",
                flush=True,
            )
    torch.cuda.synchronize()
    print(f"-- prep: PASS ({iters} iterations)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        default="connected",
        help="comma-separated: prep, steps, connected, tbe, jagged, position, hstu",
    )
    parser.add_argument("--iters", type=int, default=4000)
    args = parser.parse_args()
    phases = [p.strip() for p in args.phase.split(",") if p.strip()]
    unknown = set(phases) - {
        "prep",
        "connected",
        "tbe",
        "jagged",
        "position",
        "hstu",
        "steps",
    }
    if unknown:
        raise SystemExit(f"unknown --phase values: {unknown}")

    _require_buffer_ops_off()
    set_dev_mode(True)
    _print_env()
    torch.manual_seed(1)
    device = torch.device("cuda")
    seq_lengths, seq_offsets, num_targets, max_seq = _batch_layout(device)
    print(
        f"batch lengths={seq_lengths.tolist()} max_seq={max_seq} "
        f"phases={phases}",
        flush=True,
    )

    if "prep" in phases:
        phase_prep(device, args.iters)
        print("standalone hstu_preprocess_and_attention loop: PASS", flush=True)
        return 0

    if "steps" in phases:
        phase_steps(device, args.iters)
        print("standalone varying-shape step loop: PASS", flush=True)
        return 0

    if "connected" in phases:
        phase_connected(device, seq_lengths, seq_offsets, num_targets)
        print("standalone connected TBE→HSTU graph: PASS", flush=True)
        return 0

    seq_embeddings = None
    if "tbe" in phases:
        phase_tbe(device, seq_lengths)
    if "jagged" in phases:
        seq_embeddings = phase_jagged(device, seq_lengths, seq_offsets)
    if "position" in phases:
        seq_embeddings = phase_position(
            device, seq_embeddings if seq_embeddings is not None else
            torch.empty(0, device=device),
            seq_lengths, seq_offsets, num_targets,
        )
    if "hstu" in phases:
        phase_hstu(device, seq_embeddings, seq_lengths, seq_offsets, num_targets)

    print("standalone first-step mix: PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
