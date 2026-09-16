#!/usr/bin/env python3
"""Fast reproducers for the gfx1250 wild store -- docs/mi450_b0/mi450_b0.md section 3.

    --rows 1200000 --batch 1024 --streams 1 --seconds 240
        ==> SINGLE-STREAM hard fault in _hstu_attn_bwd, 2 of 5 runs within ~4 min

    --rows 400000 --streams 2   ==>  hard fault in ~25s, 3/3 runs (TWO streams)

Each loop is an independent HSTU fwd+bwd on its own torch.cuda.Stream() in its own
thread, sharing nothing but the GPU. The fault is the same wild-store signature as
the step-1370 nan: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, and in dmesg the
GC_UTCL2 / TCP (0x8) / RW: 0x1 triple.

WHICH ARM TO USE. Prefer the single-stream arm: the e2e that NaNs at step 1370 is
single-stream (use_pipeline=False), so --streams 1 is the only fast arm in the same
concurrency regime. The two-stream arm is ~6x faster but is a DIFFERENT regime and
has not been shown to be the same defect, only the same signature -- and neither
available serializer can decide that, because AMD_SERIALIZE_KERNEL=3 and a
per-thread torch.cuda.synchronize() both leave the two threads free to overlap
across HSA queues (both still fault, measured).

WHAT TURNS THE SINGLE-STREAM FAULT ON. Not row count, not footprint, not total
work. AUTOTUNE_MAX_SEQ_LEN = prev_power_of_2(max_seq_len) is a constexpr kernel
argument, so each bucket compiles a separate _hstu_attn_bwd binary. The fault
appears only when that bucket is >= 2048 AND there are many concurrent sequences:

    rows      batch   msl    bucket   seqs   result
    400000     1024    718      512   1024   clean, 4986 iters / 319s
    900000     1024   1617     1024   1024   clean, 1838 iters / 337s
    1060000    1024   1905     1024   1024   clean,  890 iters / 207s
    1200000    2048   1138     1024   2048   clean,  979 iters / 204s
    1200000    1024   2156     2048   1024   FAULT  (<50 iters; ~650 iters), then
                                             clean 979, clean 972  -> 2 of 5
    1600000    1024   2875     2048   1024   clean,  460 iters (underpowered)
    2400000    2048   2149     2048   2048   clean,  334 iters (underpowered)
    2800000    1024   4096     4096   1024   FAULT (the original plain arm)
    400000      128   4096     4096    128   clean, 1270 iters / 232s  <- bucket is
                                             high but only 128 sequences: high VGPR
                                             alone is not enough, concurrency is
                                             also required
    400000     1024    718      512   1024   clean + 8 GiB ballast, 4039 iters/259s
                                             -> not footprint-sensitive

That boundary lands on a known sore spot. _get_bw_pinned_configs() in
ops/triton/triton_hstu_attention.py already drops BLOCK_N 128->64 for gfx1250:
"BLOCK_N=128 reaches the 1024-VGPR ceiling and intermittently corrupts a store
address. 64 uses 758 VGPRs without spilling." We still fault with that pin in
place, at exactly the bucket where the specialization changes. See
scripts/probe_gfx1250_bwd_vgpr.py for the per-bucket n_regs / n_spills census.

Note the arms marked "underpowered": at roughly 1 fault per few hundred iterations,
a clean run of 334 or 460 iterations is not evidence of anything. Only compare
configs by iterations completed, not by wall clock.

  --rows N        jagged rows (default 400000)
  --batch B       sequences (default 1024). Controls max_seq_len for a given N, so
                  it controls the autotune bucket -- see the table above.
  --streams N     independent loops, each on its own stream (1 = e2e regime)
  --seconds S     give up after S seconds (default 300)
  --ballast GB    allocate GB and never touch it; pure address-space footprint
  --sync          torch.cuda.synchronize() every iteration, per thread

Run with AMDGCN_USE_BUFFER_OPS=0 and PYTORCH_CUDA_ALLOC_CONF= empty, on an IDLE GPU.
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fbgemm_gpu  # noqa: F401
except ImportError:
    pass
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output, hstu_preprocess_and_attention,
)

D, NUM_HEADS, ATTN_DIM, HIDDEN_DIM, MAX_SEQ_LEN, NORM_EPS = 512, 4, 128, 128, 4096, 1e-6

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=400000)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seconds", type=float, default=300.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--streams", type=int, default=1,
                help="run this many INDEPENDENT copies of the loop, each on its own "
                     "torch.cuda.Stream() in its own thread, sharing nothing but the GPU. "
                     "2 is the fast arm: concurrent streams are what made this fault in "
                     "<20s (mi450_b0.md 3.6). 1 keeps the single-stream regime, which is "
                     "the regime the e2e actually runs in.")
ap.add_argument("--ballast", type=float, default=0.0, metavar="GB",
                help="allocate GB of VRAM and never touch it again. Pure dead weight: it "
                     "adds no kernels and no concurrency, only address-space footprint. If "
                     "this alone shortens time-to-fault, the defect is footprint-sensitive "
                     "(an int32 offset wrap, the [b3] class) rather than concurrency-driven.")
ap.add_argument("--sync", action="store_true",
                help="control arm: torch.cuda.synchronize() every iteration")
a = ap.parse_args()

dev = torch.device("cuda")
torch.manual_seed(a.seed)
print(f"# {torch.cuda.get_device_properties(0).gcnArchName} torch={torch.__version__} "
      f"pid={os.getpid()}")
print(f"# N={a.rows:,} batch={a.batch} budget={a.seconds:.0f}s streams={a.streams} ballast={a.ballast}GB sync={a.sync} "
      f"BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS')}", flush=True)

def build(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = {"in_w": torch.ones(D, device=dev), "in_b": torch.zeros(D, device=dev),
         "uvqk_w": torch.randn(D, 2*NUM_HEADS*(HIDDEN_DIM+ATTN_DIM), device=dev, generator=g)*0.02,
         "uvqk_b": torch.zeros(2*NUM_HEADS*(HIDDEN_DIM+ATTN_DIM), device=dev),
         "out_w": torch.ones(NUM_HEADS*HIDDEN_DIM, device=dev),
         "out_b": torch.zeros(NUM_HEADS*HIDDEN_DIM, device=dev),
         "out_proj": torch.randn(NUM_HEADS*HIDDEN_DIM+D, D, device=dev, generator=g)*0.02}
    for v in w.values():
        v.requires_grad_(True)
    wt = torch.rand(a.batch, device=dev, generator=g) + 0.1
    lengths = (wt/wt.sum()*a.rows).long().clamp(1, MAX_SEQ_LEN)
    while int(lengths.sum()) != a.rows:
        d = a.rows - int(lengths.sum())
        room = (MAX_SEQ_LEN - lengths) if d > 0 else (lengths - 1)
        i = torch.nonzero(room > 0).flatten()[0]
        take = min(abs(d), int(room[i]));  lengths[i] += take if d > 0 else -take
    so = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), lengths.cumsum(0)]).long()
    x = torch.randn(a.rows, D, device=dev, generator=g, requires_grad=True)
    return w, so, int(lengths.max()), x

def step(w, so, msl, x):
    attn, u, _, _ = hstu_preprocess_and_attention(
        x=x, norm_weight=w["in_w"], norm_bias=w["in_b"], norm_eps=NORM_EPS,
        num_heads=NUM_HEADS, attn_dim=ATTN_DIM, hidden_dim=HIDDEN_DIM,
        uvqk_weight=w["uvqk_w"], uvqk_bias=w["uvqk_b"], max_seq_len=msl,
        seq_offsets=so, attn_alpha=1.0/(ATTN_DIM**0.5), causal=True,
        num_targets=None, max_attn_len=0, contextual_seq_len=0,
        recompute_uvqk_in_backward=False, recompute_normed_x_in_backward=False,
        sort_by_length=False, kernel=HammerKernel.TRITON)
    return hstu_compute_output(
        attn=attn, u=u, x=x, norm_weight=w["out_w"], norm_bias=w["out_b"],
        norm_eps=NORM_EPS, output_weight=w["out_proj"], num_heads=NUM_HEADS,
        linear_dim=HIDDEN_DIM, dropout_ratio=0.0, training=False, concat_u=False,
        concat_x=True, mul_u_activation_type="silu", group_norm=False,
        recompute_y_in_backward=False, kernel=HammerKernel.TRITON)

_ballast = None
if a.ballast:
    _ballast = torch.empty(int(a.ballast * (1 << 30) // 2), dtype=torch.float16, device=dev)
    print(f"# ballast: {a.ballast:.1f} GiB allocated and never touched", flush=True)

WL = [build(a.seed + 11*i) for i in range(a.streams)]
print(f"# max_seq_len={WL[0][2]}  workloads={len(WL)}", flush=True)

# Warm the Triton JIT in the main thread. Two threads hitting an uncompiled
# kernel at once race inside the JIT cache and one dies with a TypeError, which
# silently turns a two-stream run into a one-stream run.
for _w, _so, _msl, _x in WL:
    step(_w, _so, _msl, _x).sum().backward()
    _x.grad = None
    for _v in _w.values():
        _v.grad = None
torch.cuda.synchronize()
print("# JIT warm", flush=True)

t0 = time.time()
counts = [0]*a.streams

def loop(i, stream):
    w, so, msl, x = WL[i]
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with ctx:
        while time.time() - t0 < a.seconds:
            step(w, so, msl, x).sum().backward()
            x.grad = None
            for v in w.values():
                v.grad = None
            if a.sync:
                torch.cuda.synchronize()
            counts[i] += 1
            if i == 0 and counts[0] % 25 == 0:
                print(f"    iter {counts[0]}  t={time.time()-t0:.1f}s  "
                      f"others={counts[1:]}", flush=True)

import contextlib, threading
threads = [threading.Thread(target=loop, args=(i, torch.cuda.Stream()), daemon=True)
           for i in range(1, a.streams)]
for th in threads:
    th.start()
loop(0, None)
for th in threads:
    th.join(timeout=30)
torch.cuda.synchronize()
el = time.time() - t0
print(f"\nNO FAULT: {sum(counts)} iterations across {a.streams} stream(s) in {el:.0f}s "
      f"({el/max(counts[0],1):.2f}s/iter on stream 0)")
