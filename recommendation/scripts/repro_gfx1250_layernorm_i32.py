#!/usr/bin/env python3
"""Direct-call probe of the un-widened int32 row offsets in triton_layer_norm.py.

Calls the repo's public op wrappers only -- no dataset, no fbgemm, no training.

    AMDGCN_USE_BUFFER_OPS=0 python3 repro_layernorm_i32.py

_weighted_layer_norm_fwd (:208,:242) and _weighted_rms_norm_fwd (:817,:839)
index X/Y with a raw int32 `rows`:

    rows = start_row + tl.arange(0, BLOCK_N)      # i32
    tl.load (X + rows[:, None] * stride_x + ...)
    tl.store(Y + rows[:, None] * stride_y + ...)

stride == D == 512 here, so `rows * stride` reaches 2**31 at row 4_194_304.
`row_mask = rows < N` is evaluated on the UNWRAPPED row id, so it passes while
the wrapped negative offset is used -- a0 defect [3]'s exact pattern.
"""
import argparse, os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/recommendation")
from generative_recommenders.ops.layer_norm import layer_norm, rms_norm
from generative_recommenders.common import HammerKernel

D = 512
WRAP = 2**31 // D          # 4_194_304
dev = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("--canary-gib", type=float, default=6.0)
ap.add_argument("--op", default="layer_norm", choices=["layer_norm", "rms_norm"])
ap.add_argument("--ns", default="")
a = ap.parse_args()

print(f"# {torch.cuda.get_device_properties(0).gcnArchName} torch={torch.__version__}")
print(f"# D={D}  wrap row = 2**31/{D} = {WRAP:,}")
print(f"# op={a.op}  AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS')}")

# Canary allocated FIRST so it sits below later allocations; a wrapped store
# lands ~4.29 GB below Y and should land inside it.
ncan = int(a.canary_gib * (1 << 30) // 4)
canary = torch.full((ncan,), 1.5, device=dev, dtype=torch.float32)
print(f"# canary {a.canary_gib} GiB @ {canary.data_ptr():#x}")

w = torch.ones(D, device=dev, dtype=torch.float32)
b = torch.zeros(D, device=dev, dtype=torch.float32)

def run(N):
    torch.cuda.empty_cache()
    free, tot = torch.cuda.mem_get_info()
    x = torch.randn((N, D), device=dev, dtype=torch.float32)
    if a.op == "layer_norm":
        y = layer_norm(x, w, b, eps=1e-5, kernel=HammerKernel.TRITON)
    else:
        y = rms_norm(x, w, eps=1e-5, kernel=HammerKernel.TRITON)
    torch.cuda.synchronize()

    # Chunked comparison: never materialise a full reference tensor.
    CH = 1 << 18
    nbadrow = 0; first = -1; nonfinite = 0
    for s0 in range(0, N, CH):
        s1 = min(s0 + CH, N)
        xc = x[s0:s1]; yc = y[s0:s1]
        if a.op == "layer_norm":
            rc = F.layer_norm(xc, (D,), w, b, 1e-5)
        else:
            xf = xc.float()
            rc = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        badrow = (~torch.isclose(yc, rc, atol=1e-3, rtol=1e-3)).any(dim=1)
        nb = int(badrow.sum())
        if nb and first < 0:
            first = s0 + int(badrow.nonzero()[0, 0])
        nbadrow += nb
        nonfinite += int((~torch.isfinite(yc)).sum())
        del xc, yc, rc, badrow
    ncorrupt = int((canary != 1.5).sum())

    print(f"N={N:<10,} free={free/2**30:6.1f}G ptr(y)={y.data_ptr():#x}  "
          f"bad_rows={nbadrow:<10,} first_bad={first:<12,} "
          f"nonfinite={nonfinite:<10,} canary_bad={ncorrupt:,}"
          f"   {'PASS' if (nbadrow==0 and ncorrupt==0) else 'FAIL'}", flush=True)
    del x, y
    torch.cuda.empty_cache()

ns = [int(v) for v in a.ns.split(",")] if a.ns else \
     [1_000_000, 2_740_000, WRAP - 1, WRAP, WRAP + 1, WRAP + 100_000]
for N in ns:
    run(N)
