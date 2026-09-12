#!/usr/bin/env python3
"""Standalone reproducer: silent data corruption in torch.index_select (gfx1250).

Needs only torch+GPU -- no dataset, no fbgemm, no Triton.
src[i,:] == i, so index_select(src,0,idx)[:,0] must equal idx exactly.
Any mismatch is silent corruption.

See ``docs/mi450_b0/mi450_b0.md``.

Run:  python3 scripts/repro_gfx1250_gather_corruption.py
Expect on a healthy host: every line 0/600 bad.
"""
import torch
d = torch.device('cuda'); torch.manual_seed(11)
N = 2_347_656                      # shrunk yambda item-table rows
p = torch.cuda.get_device_properties(0)
print(f"{p.name} {p.gcnArchName} torch={torch.__version__}")

def mk(D):
    s = torch.empty((N, D), device=d, dtype=torch.float32)
    s.copy_(torch.arange(N, dtype=torch.float32, device=d).unsqueeze(1).expand(N, D))
    return s

def trial(D, iters=600, sync=False, mode="index_select"):
    import torch.nn.functional as F
    src = mk(D); bad = 0
    for _ in range(iters):
        idx = torch.randint(0, N, (16384,), device=d, dtype=torch.long)
        o = src[idx] if mode == "getitem" else (
            F.embedding(idx, src) if mode == "embedding" else torch.index_select(src, 0, idx))
        if sync: torch.cuda.synchronize()
        if not torch.equal(o[:, 0].long(), idx): bad += 1
    del src; torch.cuda.empty_cache(); return bad, iters

# NOTE: allocation history matters -- keep this exact sequence.
for mode in ["index_select", "getitem", "embedding"]:
    b, n = trial(256, mode=mode); print(f"{mode:14s} D=256 plain        : {b}/{n} bad ({b/n*100:.2f}%)", flush=True)
b, n = trial(256, sync=True);  print(f"index_select   D=256 +synchronize : {b}/{n} bad ({b/n*100:.2f}%)", flush=True)
b, n = trial(256);             print(f"index_select   D=256 plain        : {b}/{n} bad ({b/n*100:.2f}%)", flush=True)
for D in [4, 16, 64, 128, 512]:
    b, n = trial(D); print(f"index_select   D={D:<4d} plain        : {b}/{n} bad ({b/n*100:.2f}%)", flush=True)

# --- Independent-implementation arms -------------------------------------
# Same access pattern via a different ATen kernel and a hand-written Triton
# kernel. If these corrupt too, the defect is below any single kernel.
try:
    import triton, triton.language as tl
    @triton.jit
    def _gk(SRC, IDX, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0); row = tl.load(IDX + r)
        off = tl.arange(0, BLOCK); m = off < D
        tl.store(OUT + r * D + off, tl.load(SRC + row * D + off, mask=m, other=0.0), mask=m)
    def trial_alt(D, iters=600, mode="gather"):
        src = mk(D); bad = 0
        for _ in range(iters):
            idx = torch.randint(0, N, (16384,), device=d, dtype=torch.long)
            if mode == "gather":
                o = torch.gather(src, 0, idx.unsqueeze(1).expand(-1, D))
            else:
                o = torch.empty((16384, D), device=d, dtype=torch.float32)
                _gk[(16384,)](src, idx, o, D=D, BLOCK=triton.next_power_of_2(D))
            if not torch.equal(o[:, 0].long(), idx): bad += 1
        del src; torch.cuda.empty_cache(); return bad, iters
    for D in [4, 16]:
        for mode in ["gather", "triton"]:
            b, n = trial_alt(D, mode=mode)
            print(f"{mode:14s} D={D:<4d} plain        : {b}/{n} bad ({b/n*100:.2f}%)", flush=True)
except Exception as e:
    print("alt arms skipped:", e)
