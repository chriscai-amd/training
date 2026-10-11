#!/usr/bin/env python3
"""T8 single-GEMM numerics check (GPU; a HUMAN runs this, inside the training image, ONE shape
per process, after the hipblaslt-bench checks pass). Not run by the T8 builder.

Issues exactly the training weight-gradient call  dW = torch.mm(dY.t(), X)  with X [K, m],
dY [K, n] (bf16, row-major), which hipBLASLt sees as NT  m x n x K  (transA=N lda=m, transB=T
ldb=n, batch 1) -- the shape T8 moves from sol 102 to sol 103 (MT256x128x64, 676 VGPRs).

  --mode zero : dY = 0, so every output word must be exactly +0. This is the A0 isolated DW
                oracle (docs/mi450_a0/mi450_a0.md L923: sol 103 failed at call 196 with 512
                wrong words; sol 102 passed 2 x 1000). Checked after EVERY call.
  --mode rand : random X, dY. Every call must be bit-identical to call 1 (SK0, GSU 1: no
                atomics, so the result is deterministic), and call 1 is compared once with an
                fp32 reference (different kernel) by max relative error.

The first call runs under torch.profiler and prints the GPU kernel names, so the run itself
shows whether MT256x128x64 (sol 103) or MT32x16x32 (sol 102) executed.
Exit 0 = pass, 1 = numerical failure, 2 = wrong kernel / setup refused.
"""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True, help="hipBLASLt M = X width (layer input)")
    ap.add_argument("--n", type=int, required=True, help="hipBLASLt N = dY width (layer output)")
    ap.add_argument("--k", type=int, required=True, help="K = T (token count)")
    ap.add_argument("--calls", type=int, default=1000)
    ap.add_argument("--mode", choices=("zero", "rand"), default="zero")
    ap.add_argument("--expect", default="MT256x128x64", help="macro-tile substring the kernel must have")
    ap.add_argument("--rtol", type=float, default=2e-2)
    a = ap.parse_args()

    if os.environ.get("AMDGCN_USE_BUFFER_OPS", "0") != "0":
        print("refusing: AMDGCN_USE_BUFFER_OPS must be 0 on a37-1"); return 2
    for v in ("TENSILE_GRIDBASED_KDTREE", "TENSILE_GRIDBASED_BATCH_EXP", "TENSILE_METRIC",
              "TENSILE_SOLUTION_INDEX", "HIPBLASLT_TUNING_OVERRIDE_FILE"):
        if os.environ.get(v):
            print("refusing: %s is set" % v); return 2
    for v in ("LD_PRELOAD", "LD_LIBRARY_PATH", "HIPBLASLT_TENSILE_LIBPATH"):
        print("%s=%s" % (v, os.environ.get(v)))

    import torch
    try:
        torch.backends.cuda.preferred_blas_library("hipblaslt")
    except Exception as e:  # older/newer spelling; training routes bf16 mm to hipBLASLt anyway
        print("WARN: preferred_blas_library: %s; using %s" % (e, torch.backends.cuda.preferred_blas_library()))
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(1234)
    X = torch.randn(a.k, a.m, device=dev, generator=g, dtype=torch.bfloat16).mul_(0.05)
    if a.mode == "zero":
        dY = torch.zeros(a.k, a.n, device=dev, dtype=torch.bfloat16)
    else:
        dY = torch.randn(a.k, a.n, device=dev, generator=g, dtype=torch.bfloat16).mul_(0.05)
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out1 = torch.mm(dY.t(), X)
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    gemm = [n for n in names if n.startswith("Cijk_")]
    print("kernels in call 1:", gemm or names)
    if not gemm:
        print("WARN: profiler saw no Cijk_ kernel; rely on the hipblaslt-bench --print_kernel_info check")
    elif not any(a.expect in n for n in gemm):
        print("FAIL: expected a %s kernel" % a.expect); return 2

    bad_calls, first_bad = 0, None
    ref_bits = out1.view(torch.int16).clone()
    if a.mode == "rand":
        ref = torch.zeros(a.n, a.m, device=dev, dtype=torch.float32)
        for k0 in range(0, a.k, 1 << 18):          # chunk K: one fp32 copy of a full operand can exceed the allocation cap
            ref += torch.mm(dY[k0:k0 + (1 << 18)].t().float(), X[k0:k0 + (1 << 18)].float())
        err = ((out1.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
        print("call 1 max|out-ref|/max|ref| = %.3e (rtol %.1e)" % (err, a.rtol))
        if not (err <= a.rtol) or not torch.isfinite(out1).all():
            print("FAIL: call 1 vs fp32 reference"); return 1
        del ref
    t0 = time.time()
    for c in range(1, a.calls + 1):
        out = out1 if c == 1 else torch.mm(dY.t(), X)
        if a.mode == "zero":
            nbad = int(torch.count_nonzero(out).item())                     # value != 0 (NaN counts)
            nneg = int(torch.count_nonzero(out.view(torch.int16)).item()) - nbad
            if nneg and c == 1:
                print("info: %d words are -0.0 (not counted as wrong)" % nneg)
        else:
            nbad = int((out.view(torch.int16) != ref_bits).sum().item())
        if nbad:
            bad_calls += 1
            first_bad = first_bad or c
            print("call %d: %d wrong words (max |v| %.3e)" % (c, nbad, out.float().abs().max().item()))
            if bad_calls >= 5:
                break
    dt = (time.time() - t0) / max(1, c)
    print("shape m=%d n=%d k=%d mode=%s calls=%d bad_calls=%d first_bad=%s  %.2f ms/call (incl. check)"
          % (a.m, a.n, a.k, a.mode, c, bad_calls, first_bad, dt * 1e3))
    print("PASS" if bad_calls == 0 else "FAIL")
    return 0 if bad_calls == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
