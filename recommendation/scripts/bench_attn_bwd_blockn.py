"""Time the HSTU attention backward at a chosen pinned BLOCK_N.

Quantifies the throughput cost of pinning BLOCK_N=64 on gfx1250 (docs/mi450.md
row 2). One BLOCK_N per process, so a fault in the 128 case cannot take the 64
measurement with it.
"""

import argparse
import os

os.environ.setdefault("AMD_SERIALIZE_KERNEL", "0")

import torch

import fbgemm_gpu  # noqa: E402,F401  registers the fbgemm ops the harness needs
import generative_recommenders.ops.triton.triton_hstu_attention as A  # noqa: E402
from generative_recommenders.common import HammerKernel, set_dev_mode  # noqa: E402
from generative_recommenders.ops.hstu_attention import hstu_mha  # noqa: E402


def pin_block_n(block_n):
    """Replace the pinned backward config's BLOCK_N, keeping everything else."""
    configs = A._hstu_attn_bwd.configs
    assert len(configs) == 1, f"expected one pinned config, got {len(configs)}"
    configs[0].kwargs["BLOCK_N"] = block_n
    return configs[0]


def build(batch_size, heads, max_uih_len, max_targets, attn_dim, hidden_dim,
          dtype, seed=0):
    set_dev_mode(True)
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    lengths = torch.randint(max_uih_len + 1, size=(batch_size,), device=dev)
    num_targets = torch.randint(1, max_targets + 1, size=(batch_size,), device=dev)
    lengths = lengths + num_targets
    max_seq_len = max_uih_len + max_targets

    seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=dev)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(seq_offsets[-1].item())

    def mk(dim):
        return (torch.empty((total, heads, dim), dtype=dtype, device=dev)
                .uniform_(-0.1, 0.1).requires_grad_())

    q, k, v = mk(attn_dim), mk(attn_dim), mk(hidden_dim)
    return dict(
        max_seq_len=max_seq_len, alpha=1.0 / (attn_dim ** 0.5), q=q, k=k, v=v,
        seq_offsets=seq_offsets, causal=True, num_targets=num_targets,
        dropout_pr=0.0, max_attn_len=max_uih_len // 5, contextual_seq_len=0,
        kernel=HammerKernel.TRITON, enable_tma=False,
    ), total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block-n", type=int, required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--max-uih-len", type=int, default=1024)
    p.add_argument("--max-targets", type=int, default=32)
    p.add_argument("--attn-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    args = p.parse_args()

    cfg = pin_block_n(args.block_n)
    print(f"pinned bwd config: {cfg.kwargs} num_warps={cfg.num_warps} "
          f"num_stages={cfg.num_stages}")

    kw, total = build(args.batch_size, args.heads, args.max_uih_len,
                      args.max_targets, args.attn_dim, args.hidden_dim,
                      torch.bfloat16)
    print(f"shape: total_tokens={total} heads={args.heads} "
          f"attn_dim={args.attn_dim} hidden_dim={args.hidden_dim}")

    out = hstu_mha(**kw)
    grad = torch.randn_like(out)

    def one():
        for t in (kw["q"], kw["k"], kw["v"]):
            t.grad = None
        hstu_mha(**kw).backward(grad)

    for _ in range(args.warmup):
        one()
    torch.cuda.synchronize()

    # Guard against measuring a path that never reaches the pinned kernel, and
    # report VGPRs so the tile change is visible in the codegen, not just the
    # config dict.
    compiled = [k for cache in A._hstu_attn_bwd.fn.device_caches.values()
                for k in cache[0].values()]
    assert compiled, "_hstu_attn_bwd never launched -- wrong dispatch path"
    for k in compiled:
        print(f"  compiled: n_regs={k.n_regs} n_spills={k.n_spills}")

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(args.iters):
        one()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / args.iters
    print(f"RESULT block_n={args.block_n} fwd+bwd={ms:.3f} ms/iter "
          f"over {args.iters} iters")


if __name__ == "__main__":
    main()
