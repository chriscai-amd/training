#!/usr/bin/env python3
"""Repeat-call determinism check for one HSTU layer on gfx1250.

Feeds the SAME bytes to the same layer R times and hashes each output. A correct
implementation gives one hash. On heliosr-1b112-a30-4 both the TRITON and the
PYTORCH paths give several, which is why this exists.

  --repeats R      calls per path (default 10)
  --rows N         total jagged rows (default 100000)
  --path both|triton|pytorch
"""
import argparse, hashlib, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fbgemm_gpu  # noqa: F401  -- registers torch.ops.fbgemm.*
except ImportError:
    pass
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output, hstu_preprocess_and_attention,
)

D, NUM_HEADS, ATTN_DIM, HIDDEN_DIM, MAX_SEQ_LEN, NORM_EPS = 512, 4, 128, 128, 4096, 1e-6

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=100000)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--path", default="both", choices=["both", "triton", "pytorch"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sync-between", action="store_true",
                help="torch.cuda.synchronize() between forward and backward. Drains the "
                     "forward before the backward is launched, so the two cannot overlap. "
                     "Discriminator for mi450_b0.md 13.1: if the fault survives this, the "
                     "fault does not need forward/backward overlap.")
ap.add_argument("--aggressor", type=int, default=0, metavar="ROWS",
                help="run an INDEPENDENT HSTU fwd+bwd loop on a second CUDA stream in a "
                     "daemon thread, sized ROWS jagged rows (400000 is a good start). The "
                     "suspected mechanism is kernel overlap (mi450_b0.md 3.2 conclusion 3), "
                     "so saturating the GPU with a second stream should make the fault need "
                     "far fewer repeats. NOTE: this changes the regime under test -- a fault "
                     "here is a two-stream fault, which is not the same evidence as a fault "
                     "in the plain single-stream arm.")
ap.add_argument("--backward", action="store_true",
                help="also check that dx and the weight grads repeat bit-exactly; "
                     "the forward-only check does not cover _hstu_attn_bwd")
a = ap.parse_args()

dev = torch.device("cuda")
torch.manual_seed(a.seed)
props = torch.cuda.get_device_properties(0)
print(f"# {props.gcnArchName} torch={torch.__version__} pid={os.getpid()}")
print(f"# N={a.rows:,} batch={a.batch} repeats={a.repeats} "
      f"BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS')}")

_t0 = time.time()

def md5(t):
    return hashlib.md5(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]

g = torch.Generator(device=dev).manual_seed(a.seed)
w = {"in_w": torch.ones(D, device=dev), "in_b": torch.zeros(D, device=dev),
     "uvqk_w": torch.randn(D, 2*NUM_HEADS*(HIDDEN_DIM+ATTN_DIM), device=dev, generator=g)*0.02,
     "uvqk_b": torch.zeros(2*NUM_HEADS*(HIDDEN_DIM+ATTN_DIM), device=dev),
     "out_w": torch.ones(NUM_HEADS*HIDDEN_DIM, device=dev),
     "out_b": torch.zeros(NUM_HEADS*HIDDEN_DIM, device=dev),
     "out_proj": torch.randn(NUM_HEADS*HIDDEN_DIM+D, D, device=dev, generator=g)*0.02}

wt = torch.rand(a.batch, device=dev, generator=g) + 0.1
lengths = (wt/wt.sum()*a.rows).long().clamp(1, MAX_SEQ_LEN)
while int(lengths.sum()) != a.rows:
    d = a.rows - int(lengths.sum())
    room = (MAX_SEQ_LEN - lengths) if d > 0 else (lengths - 1)
    i = torch.nonzero(room > 0).flatten()[0]
    take = min(abs(d), int(room[i]));  lengths[i] += take if d > 0 else -take
seq_offsets = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), lengths.cumsum(0)]).long()
msl = int(lengths.max())
x = torch.randn(a.rows, D, device=dev, generator=g)
print(f"# max_seq_len={msl}  x_md5={md5(x)}")

def run(kernel, grad=False, _x=None, _w=None, _so=None, _msl=None):
    import contextlib
    x, w, seq_offsets, msl = (_x if _x is not None else globals()["x"],
                              _w if _w is not None else globals()["w"],
                              _so if _so is not None else globals()["seq_offsets"],
                              _msl if _msl is not None else globals()["msl"])
    with (contextlib.nullcontext() if grad else torch.no_grad()):
        attn, u, _, _ = hstu_preprocess_and_attention(
            x=x, norm_weight=w["in_w"], norm_bias=w["in_b"], norm_eps=NORM_EPS,
            num_heads=NUM_HEADS, attn_dim=ATTN_DIM, hidden_dim=HIDDEN_DIM,
            uvqk_weight=w["uvqk_w"], uvqk_bias=w["uvqk_b"], max_seq_len=msl,
            seq_offsets=seq_offsets, attn_alpha=1.0/(ATTN_DIM**0.5), causal=True,
            num_targets=None, max_attn_len=0, contextual_seq_len=0,
            recompute_uvqk_in_backward=False, recompute_normed_x_in_backward=False,
            sort_by_length=False, kernel=kernel)
        return hstu_compute_output(
            attn=attn, u=u, x=x, norm_weight=w["out_w"], norm_bias=w["out_b"],
            norm_eps=NORM_EPS, output_weight=w["out_proj"], num_heads=NUM_HEADS,
            linear_dim=HIDDEN_DIM, dropout_ratio=0.0, training=False, concat_u=False,
            concat_x=True, mul_u_activation_type="silu", group_norm=False,
            recompute_y_in_backward=False, kernel=kernel)

# Second-stream aggressor. Independent tensors and its own generator, so it
# shares nothing with the workload under test except the GPU -- any effect it has
# on the victim's results is contention, not aliasing.
_agg_stop = None
if a.aggressor:
    import threading
    ga = torch.Generator(device=dev).manual_seed(a.seed + 7)
    wa = {"in_w": torch.ones(D, device=dev), "in_b": torch.zeros(D, device=dev),
          "uvqk_w": torch.randn(D, 2*NUM_HEADS*(HIDDEN_DIM+ATTN_DIM), device=dev, generator=ga)*0.02,
          "uvqk_b": torch.zeros(2*NUM_HEADS*(HIDDEN_DIM+ATTN_DIM), device=dev),
          "out_w": torch.ones(NUM_HEADS*HIDDEN_DIM, device=dev),
          "out_b": torch.zeros(NUM_HEADS*HIDDEN_DIM, device=dev),
          "out_proj": torch.randn(NUM_HEADS*HIDDEN_DIM+D, D, device=dev, generator=ga)*0.02}
    wta = torch.rand(a.batch, device=dev, generator=ga) + 0.1
    la = (wta/wta.sum()*a.aggressor).long().clamp(1, MAX_SEQ_LEN)
    while int(la.sum()) != a.aggressor:
        d = a.aggressor - int(la.sum())
        room = (MAX_SEQ_LEN - la) if d > 0 else (la - 1)
        i = torch.nonzero(room > 0).flatten()[0]
        take = min(abs(d), int(room[i]));  la[i] += take if d > 0 else -take
    soa = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), la.cumsum(0)]).long()
    msla, xa = int(la.max()), torch.randn(a.aggressor, D, device=dev, generator=ga)
    xa.requires_grad_(True)
    for v in wa.values():
        v.requires_grad_(True)
    _agg_stream, _agg_stop = torch.cuda.Stream(), threading.Event()
    def _aggressor():
        with torch.cuda.stream(_agg_stream):
            while not _agg_stop.is_set():
                ya = run(HammerKernel.TRITON, grad=True, _x=xa, _w=wa, _so=soa, _msl=msla)
                ya.sum().backward()
                xa.grad = None
                for v in wa.values():
                    v.grad = None
    threading.Thread(target=_aggressor, daemon=True).start()
    print(f"# aggressor: {a.aggressor:,} rows on a 2nd stream, warming up", flush=True)
    time.sleep(20)

# The PYTORCH reference densifies attention to batch x heads x seq x seq, so it
# OOMs at large N. Skip it when only the TRITON path is under test.
ref = None
if a.path != "triton":
    ref = run(HammerKernel.PYTORCH)
    print(f"# reference y: mean|y|={float(ref.abs().mean()):.3f} max|y|={float(ref.abs().max()):.3f}")

bad = 0
for name, kern in (("PYTORCH", HammerKernel.PYTORCH), ("TRITON", HammerKernel.TRITON)):
    if a.path != "both" and a.path != name.lower():
        continue
    # Compare on the GPU against the first output and only pay for an md5 when
    # something actually differs -- hashing a multi-GB tensor on the CPU every
    # repeat is far slower than the kernel under test and hides the signal.
    hashes, diffs, first = {}, [], None
    for _ in range(a.repeats):
        y = run(kern)
        if first is None:
            first, fh = y.clone(), md5(y)
            hashes[fh] = 1
        elif torch.equal(y, first):
            hashes[fh] = hashes.get(fh, 0) + 1
        else:
            h = md5(y)
            hashes[h] = hashes.get(h, 0) + 1
            print(f"    ! repeat diverged: {h}  max|d| vs first "
                  f"{float((y - first).abs().max()):.3e}", flush=True)
        diffs.append(float((y - ref).abs().max()) if ref is not None else 0.0)
    hs = ", ".join(f"{h}x{c}" for h, c in hashes.items())
    worst = max(diffs)
    flag = "  <== NONDETERMINISTIC" if len(hashes) > 1 else ""
    print(f"{name:8s} x{a.repeats}: {len(hashes)} distinct | {hs}{flag}")
    print(f"{'':8s}   maxdiff vs reference: {min(diffs):.3e} .. {worst:.3e}")
    if len(hashes) > 1:
        bad += 1
if a.backward:
    # Gradients, not just the output: the forward-only check never enters
    # _hstu_attn_bwd, which is where the same-process two-stream run faults.
    print(f"\n# backward determinism, {a.repeats} repeats")
    for v in w.values():
        v.requires_grad_(True)
    xg = x.detach().clone().requires_grad_(True)
    first, nbad = None, 0
    for r in range(a.repeats):
        xg.grad = None
        for v in w.values():
            v.grad = None
        y = run(HammerKernel.TRITON, grad=True, _x=xg)
        if a.sync_between:
            torch.cuda.synchronize()
        y.sum().backward()
        if xg.grad is None:
            raise RuntimeError("backward produced no input gradient; dx determinism was not checked")
        if a.sync_between:
            torch.cuda.synchronize()
        # Only `dx` is required to repeat bit-exactly. The WEIGHT gradients are
        # accumulated with `tl.atomic_add` (triton_layer_norm.py:1021,
        # triton_hstu_linear.py:1828) over millions of rows, and float atomics
        # commit in nondeterministic order -- so they vary run to run by design.
        # Asserting on them reports a false failure on healthy hardware.
        cur = {"dx": xg.grad}
        info = {f"d{k}": v.grad for k, v in w.items()}
        if first is None:
            first = {k: t.clone() for k, t in cur.items() if t is not None}
            first_info = {k: t.clone() for k, t in info.items() if t is not None}
        else:
            for k, t in first.items():
                if cur[k] is not None and not torch.equal(cur[k], t):
                    nbad += 1
                    print(f"    ! repeat {r} grad {k} diverged: max|d| "
                          f"{float((cur[k] - t).abs().max()):.3e}", flush=True)
            if r == a.repeats - 1:
                drift = max(float((info[k] - t).abs().max())
                            for k, t in first_info.items() if info[k] is not None)
                print(f"    (weight-grad atomic reorder drift, expected: "
                      f"max|d| {drift:.3e} -- not a failure)")
    print(f"BACKWARD: {'FAIL - dx diverged ' + str(nbad) + 'x' if nbad else 'PASS - dx bit-exact every repeat'}")
    bad += nbad

if _agg_stop is not None:
    _agg_stop.set()
_el = time.time() - _t0
print(f"\n# elapsed {_el:.0f}s for {a.repeats} repeats "
      f"({_el/max(a.repeats,1):.2f}s/repeat){' with aggressor' if a.aggressor else ''}")
print("\nRESULT:", "FAIL - repeated identical calls disagree" if bad else "PASS - all repeats identical")
sys.exit(1 if bad else 0)
