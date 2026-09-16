#!/usr/bin/env python3
"""Per-AUTOTUNE_MAX_SEQ_LEN VGPR census for _hstu_attn_bwd -- docs/mi450_b0/mi450_b0.md section 3.

AUTOTUNE_MAX_SEQ_LEN is a constexpr kernel argument, so prev_power_of_2(max_seq_len)
compiles a SEPARATE binary per bucket. The single-stream fault turns on exactly when
that bucket reaches 2048 (see the section 3 bucket table), and
_get_bw_pinned_configs() already drops BLOCK_N 128->64 on gfx1250 with the comment
"BLOCK_N=128 reaches the 1024-VGPR ceiling and intermittently corrupts a store
address. 64 uses 758 VGPRs without spilling."

So: does the pin still hold at bucket >= 2048, or does the larger specialization
climb back toward the ceiling / start spilling? This prints n_regs and n_spills for
each bucket straight out of the Triton compilation cache.

Run with AMDGCN_USE_BUFFER_OPS=0 and PYTORCH_CUDA_ALLOC_CONF= empty.
"""
import os, pathlib, re, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fbgemm_gpu  # noqa
except ImportError:
    pass
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output,
    hstu_preprocess_and_attention,
)

D, NH, AD, HD, EPS = 512, 4, 128, 128, 1e-6
dev = torch.device("cuda")
CACHE = pathlib.Path(os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")))


def mkw(g, n=D):
    w = {
        "in_w": torch.ones(n, device=dev), "in_b": torch.zeros(n, device=dev),
        "uvqk_w": torch.randn(n, 2 * NH * (HD + AD), device=dev, generator=g) * 0.02,
        "uvqk_b": torch.zeros(2 * NH * (HD + AD), device=dev),
        "out_w": torch.ones(NH * HD, device=dev), "out_b": torch.zeros(NH * HD, device=dev),
        "out_proj": torch.randn(NH * HD + n, n, device=dev, generator=g) * 0.02,
    }
    for v in w.values():
        v.requires_grad_(True)
    return w


def step(w, so, msl, x):
    attn, u, _, _ = hstu_preprocess_and_attention(
        x=x, norm_weight=w["in_w"], norm_bias=w["in_b"], norm_eps=EPS, num_heads=NH,
        attn_dim=AD, hidden_dim=HD, uvqk_weight=w["uvqk_w"], uvqk_bias=w["uvqk_b"],
        max_seq_len=msl, seq_offsets=so, attn_alpha=1.0 / (AD ** 0.5), causal=True,
        num_targets=None, max_attn_len=0, contextual_seq_len=0,
        recompute_uvqk_in_backward=False, recompute_normed_x_in_backward=False,
        sort_by_length=False, kernel=HammerKernel.TRITON)
    return hstu_compute_output(
        attn=attn, u=u, x=x, norm_weight=w["out_w"], norm_bias=w["out_b"], norm_eps=EPS,
        output_weight=w["out_proj"], num_heads=NH, linear_dim=HD, dropout_ratio=0.0,
        training=True, concat_u=False, concat_x=True, mul_u_activation_type="silu",
        group_norm=False, recompute_y_in_backward=False, kernel=HammerKernel.TRITON)


def census():
    """Map every cached _hstu_attn_bwd binary to its register census.

    Triton's .json metadata has no register counts; they live in the AMDGCN
    assembly's kernel descriptor (.vgpr_count / .vgpr_spill_count / Occupancy).
    """
    out = {}
    for a in CACHE.rglob("_hstu_attn_bwd.amdgcn"):
        try:
            txt = a.read_text(errors="ignore")
        except OSError:
            continue

        def grab(pat, cast=int):
            m = re.search(pat, txt)
            return cast(m.group(1)) if m else None

        out[a.parent.name] = (
            grab(r"\.vgpr_count:\s*(\d+)"),
            grab(r"\.vgpr_spill_count:\s*(\d+)"),
            grab(r"\.sgpr_spill_count:\s*(\d+)"),
            grab(r"; Occupancy:\s*(\d+)"),
        )
    return out


# One fixed batch of sequences per bucket: batch=1024 seqs all of length `L`, so
# prev_power_of_2(L) is the bucket under test and concurrency is held constant.
print(f"# cache={CACHE}")
print(f"{'bucket':>8} {'rows':>10} {'vgprs':>6} {'vspill':>7} {'sspill':>7} {'occ':>4}  verdict")
seen = dict(census())
for L in (1024, 2048, 4096):
    B = int(os.environ.get("PROBE_BATCH", "512"))
    g = torch.Generator(device=dev).manual_seed(0)
    w = mkw(g)
    lengths = torch.full((B,), L, device=dev, dtype=torch.long)
    so = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), lengths.cumsum(0)]).long()
    x = torch.randn(B * L, D, device=dev, generator=g, requires_grad=True)
    step(w, so, L, x).sum().backward()
    torch.cuda.synchronize()
    cur = census()
    new = {k: v for k, v in cur.items() if k not in seen}
    seen = cur
    if not new:
        print(f"{L:>8} {B*L:>10,} {'-':>6} {'-':>7} {'-':>7} {'-':>4}  (cache hit; rerun cold)")
    for _k, (regs, vsp, ssp, occ) in new.items():
        bad = []
        if regs and regs >= 1024:
            bad.append("AT/OVER 1024-VGPR CEILING")
        elif regs and regs >= 896:
            bad.append("within 12% of the 1024-VGPR ceiling")
        if vsp:
            bad.append(f"VGPR SPILLS {vsp}")
        if ssp:
            bad.append(f"SGPR SPILLS {ssp}")
        print(f"{L:>8} {B*L:>10,} {regs:>6} {vsp:>7} {ssp:>7} {occ:>4}  {' ; '.join(bad) or 'ok'}")
    del w, x, so, lengths
    torch.cuda.empty_cache()
