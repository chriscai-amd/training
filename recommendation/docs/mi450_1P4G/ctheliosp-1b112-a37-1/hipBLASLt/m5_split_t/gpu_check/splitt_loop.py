#!/usr/bin/env python3
"""M5 numerics check for ONE wgrad shape and split factor (GPU; a HUMAN runs this through
run_splitt_check.sh inside the training image, after the gemm_watch.sh single-GEMM stage passed).
Not run by the M5 builder.

Calls the ACTUAL M5 code path, generative_recommenders.ops.triton.triton_addmm.splitt_wgrad_mm
(the function the training hooks call), configured through the same env vars the gin switch sets
(HSTU_SPLITT_WGRAD=all, HSTU_SPLITT_WGRAD_SPLITS=<M>x<N>:<s>) BEFORE the module is imported. The
module import also checks s against the lib dir's m5_splits.json (unpinned s -> ValueError -> exit 2).
Operands mimic the training call  dW = a.t() @ b, a [T, D], b [T, O] bf16 row-major
(hipBLASLt NT M=O, N=D):
  G20 a = normed_x [T,512]  b = duvqk (dY) [T,2048]     (hstu: dY is b)
  G19 a = y        [T,1536] b = dout  (dY) [T,512]
  G25 a = dY [T,512]  b = x [T,256];  G29 a = dY [T,256] b = x [T,1024]
  G31 a = dY [T,256]  b = x [T,512];  G27 a = dY [T,256] b = x [T,24]   (preproc: dY is a)
Inputs are generated directly in bf16 on the GPU (no fp32 copies of a full operand).

  --mode zero : dY = 0, so every output word must be +0 (the A0 isolated DW oracle, now on the
                BATCHED kernels + fp32 reduce + tail). Checked after EVERY call; the strided-batched
                bf16 partials are ALSO checked every call by re-running the same bmm on the same views
                (--check-partials, default on; doubles the batched launches = 2x training's exposure).
  --mode rand : every call bit-identical to call 1 (SK0, GSU1, fixed reduction order); call 1 vs an
                fp32 reference built in K chunks:
                  * elementwise bound |split - ref| <= 2^-8 (sum_i |P_i| + |ref|) + 2^-12 (|A|^T |B|)
                    (one bf16 rounding per partial + one final rounding + fp32 accumulation slack;
                    holds for any data),
                  * rel-rms error vs the batch-1 torch.mm of THIS lib, compared with the model
                    ratio ~ sqrt(1 + q^2), q = ||sqrt(sum_i P_i^2)|| / ||sum_i P_i||
                    (iid: q ~ 1 -> ~1.41x, must be <= --rms-ratio; structured: within 0.5x..2x of model).
                --pattern iid (independent randn) | drift (dY mean drifts -1 -> +1 along T: chunk
                partials cancel, the worst case for bf16 partials) | halves (opposite-sign halves) |
                users (sign per 512-token "user", chunk-independent). Run iid AND drift.
Call 1 runs under torch.profiler and must show EXACTLY one bf16 NT Tensile kernel launch
(Cijk_Ailk_Bjlk_BBS*, hipBLASLt "UserArgs" form) with the expected macro tile (from the lib dir's
m5_splits.json mapped sol: 103 -> MT256x128x64, 102 -> MT32x16x32; --expect overrides), no rocBLAS
kernel, and for a tail of r >= 2 rows exactly one fp32 NT kernel (Cijk_Ailk_Bjlk_S*, MT32x32x16);
r = 1 must issue no GEMM for the tail (elementwise addr_). The preferred BLAS backend is the
process DEFAULT (as in training); it must be hipBLASLt (torch reports Cublaslt) or the check fails.
Also prints ms/call (CUDA events) for the split path and for the batch-1 torch.mm on THIS lib dir
(s103: sol 103 = the T8 baseline; s102 and the 102-mapped shapes of mix: sol 102, NOT T8).
Exit 0 = pass, 1 = numerical failure, 2 = wrong kernel / backend / setup refused.
"""
import argparse
import json
import os
import sys
import time

SHAPES = {  # name: (hipBLASLt M = O = width of b, N = D = width of a, dY operand)
    "G20": (2048, 512, "b"), "G19": (512, 1536, "b"),
    "G25": (256, 512, "a"), "G29": (1024, 256, "a"), "G31": (512, 256, "a"), "G27": (24, 256, "a"),
}
TILE = {102: "MT32x16x32", 103: "MT256x128x64"}


def lib_manifest():
    lp = os.environ.get("HIPBLASLT_TENSILE_LIBPATH", "")
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(lp.rstrip("/")))), "m5_splits.json")
    return json.load(open(p)) if lp and os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=sorted(SHAPES), required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--t", type=int, required=True, help="T (token count)")
    ap.add_argument("--calls", type=int, default=1000)
    ap.add_argument("--mode", choices=("zero", "rand"), default="zero")
    ap.add_argument("--pattern", choices=("iid", "drift", "halves", "users"), default="iid")
    ap.add_argument("--expect", default=None, help="macro-tile substring of the bmm kernel (default: from manifest)")
    ap.add_argument("--check-partials", type=int, default=1)
    ap.add_argument("--rms-ratio", type=float, default=1.6, help="iid limit of split/batch-1 rms error")
    ap.add_argument("--force-blas", action="store_true", help="set preferred_blas_library('hipblaslt') (NOT training)")
    ap.add_argument("--repo", default="/workspace/recommendation")
    a = ap.parse_args()
    M, N, dy_op = SHAPES[a.shape]

    if os.environ.get("AMDGCN_USE_BUFFER_OPS", "0") != "0":
        print("refusing: AMDGCN_USE_BUFFER_OPS must be 0 on a37-1"); return 2
    for v in ("TENSILE_GRIDBASED_KDTREE", "TENSILE_GRIDBASED_BATCH_EXP", "TENSILE_METRIC",
              "TENSILE_SOLUTION_INDEX", "HIPBLASLT_TUNING_OVERRIDE_FILE", "HSTU_TN_WGRAD",
              "HSTU_SPLITT_ALLOW_UNPINNED", "HSTU_SPLITT_AUDIT_EVERY"):
        if os.environ.get(v) not in (None, "", "0"):
            print("refusing: %s is set" % v); return 2
    man = lib_manifest()
    if man is None:
        print("refusing: no m5_splits.json next to HIPBLASLT_TENSILE_LIBPATH (not a v3 lib dir)"); return 2
    ent = man["shapes"]["%dx%d" % (M, N)]
    want_sol = ent["sol"]
    expect = a.expect or TILE[want_sol]
    print("lib variant %s: %s mapped to sol %d; s=%d selects sol %s -> expect %s" % (
        man["variant"], a.shape, want_sol, a.s, ent["sol_by_s"].get(str(a.s)), expect))
    os.environ["HSTU_SPLITT_WGRAD"] = "all"
    os.environ["HSTU_SPLITT_WGRAD_SPLITS"] = "%dx%d:%d" % (M, N, a.s)
    os.environ["HSTU_SPLITT_MIN_ROWS"] = os.environ.get("HSTU_SPLITT_MIN_ROWS", "4096")
    for v in ("LD_PRELOAD", "LD_LIBRARY_PATH", "HIPBLASLT_TENSILE_LIBPATH", "HSTU_SPLITT_WGRAD",
              "HSTU_SPLITT_WGRAD_SPLITS", "HSTU_SPLITT_MIN_ROWS"):
        print("%s=%s" % (v, os.environ.get(v)))
    sys.path.insert(0, a.repo)

    import torch
    try:
        from generative_recommenders.ops.triton import triton_addmm as ta
    except ValueError as e:
        print("refusing: M5 module import: %s" % e); return 2
    print("M5 module:", ta.__file__, "splits:", ta._SPLITT_SPLITS, "expected:", ta._SPLITT_EXPECT)
    if a.force_blas:
        torch.backends.cuda.preferred_blas_library("hipblaslt")
    blas = str(torch.backends.cuda.preferred_blas_library())
    print("preferred BLAS backend (%s): %s" % ("forced" if a.force_blas else "process default = training", blas))
    if "cublaslt" not in blas.lower() and "hipblaslt" not in blas.lower():
        print("FAIL: default BLAS backend is not hipBLASLt; training would not use the v3 catalog"); return 2
    dev = torch.device("cuda", 0)
    T, D, O, s = a.t, N, M, a.s
    g = torch.Generator(device=dev).manual_seed(1234)
    A = torch.randn(T, D, device=dev, generator=g, dtype=torch.bfloat16).mul_(0.05)
    B = torch.randn(T, O, device=dev, generator=g, dtype=torch.bfloat16).mul_(0.05)
    dY = A if dy_op == "a" else B
    xop = B if dy_op == "a" else A
    if a.mode == "zero":
        dY.zero_()
    else:
        xop.add_(0.05)                       # activations with a nonzero mean (as after silu / norm)
        if a.pattern != "iid":
            W = dY.shape[1]
            for k0 in range(0, T, 1 << 18):  # bounded temporaries
                k1 = min(T, k0 + (1 << 18))
                if a.pattern == "drift":
                    base = torch.linspace(-1, 1, T, device=dev)[k0:k1].unsqueeze(1) * 0.05
                elif a.pattern == "halves":
                    base = torch.where(torch.arange(k0, k1, device=dev) < T // 2, -0.05, 0.05).unsqueeze(1)
                else:                        # users: random sign per 512-token block
                    gs = torch.Generator(device=dev).manual_seed(99 + k0)
                    sg = torch.randint(0, 2, ((k1 - k0 + 511) // 512, 1, W), device=dev, generator=gs) * 0.1 - 0.05
                    base = sg.expand(-1, 512, W).reshape(-1, W)[: k1 - k0]
                dY[k0:k1].mul_(0.05).add_(base.to(torch.bfloat16))
    torch.cuda.synchronize()
    s_plan = ta.splitt_wgrad_plan(A, B, ta._SPLITT_SPLITS, ta._SPLITT_MIN_ROWS)
    L = T // s
    r = T - s * L
    print("shape %s M=%d N=%d T=%d s=%d L=%d tail=%d plan=%d strides a/b=%d/%d mode=%s pattern=%s" % (
        a.shape, M, N, T, s, L, r, s_plan, L * D, L * O, a.mode, a.pattern))
    if s_plan != s:
        print("refusing: the M5 helper would not split this case (plan=%d)" % s_plan); return 2

    def partials():
        return torch.bmm(A[: s * L].view(s, L, D).transpose(1, 2), B[: s * L].view(s, L, O))

    from torch.profiler import profile, ProfilerActivity
    # Warm-up outside the profiler: the first hipBLASLt call loads the catalogs, and on
    # a37-1 profiling that first call returned 0 events (ROCTracer duplicate-flow warning).
    ta.splitt_wgrad_mm(A, B)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as prof:
        out1 = ta.splitt_wgrad_mm(A, B)
        torch.cuda.synchronize()
    if out1 is None:
        print("FAIL: splitt_wgrad_mm returned None"); return 2
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    if not any(n.startswith("Cijk_") for n in names):
        # On a37-1 since 2026-10-11 the profiler records no or partial GPU-side events (GEMMs missing).
        # Fall back to hipBLASLt's own call log (HIPBLASLT_LOG_MASK=32, set by run_splitt_check.sh):
        # the warm-up + profiled call = the first two M5 calls, each = 1 bf16 batched NT (+ fp32 tail).
        import glob, shlex
        lines = []
        for f in sorted(glob.glob("/tmp/hbl_*.log")):
            lines += [l for l in open(f) if l.startswith("hipblaslt-bench")]
        print("profiler saw no GEMM events (%d other); using hipBLASLt log (%d calls so far)" % (len(names), len(lines)))
        names = []
        def kv(l):
            t = shlex.split(l); d = {}
            for i, x in enumerate(t):
                if x.startswith("--"): d[x[2:]] = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True
            return d
        sol_expect = {"MT256x128x64": "103", "MT32x16x32": "102"}[expect]
        for l in lines[:4]:
            print("   ", l.strip()[:220])
        for l in lines:
            d = kv(l)
            if d.get("a_type") == "bf16_r" and int(d.get("batch_count", 1)) == s:
                names.append(("Cijk_Ailk_Bjlk_BBS_UserArgs_%s_sol%s" % (expect if d.get("solution_index") == sol_expect else "OTHER", d.get("solution_index"))))
            elif d.get("a_type") == "f32_r":
                names.append("Cijk_Ailk_Bjlk_S_UserArgs_MT32x32x16_sol%s" % d.get("solution_index"))
            else:
                names.append("Cijk_other_" + str(d.get("solution_index")))
        n_calls = 2  # warm-up + profiled call
        per = 1 + (1 if r >= 2 else 0)
        names = names[per:per * n_calls]
    print("kernels in call 1 (%d launches):" % len(names))
    for n in names:
        print("   ", n[:170])
    bf16_nt = [n for n in names if n.startswith("Cijk_Ailk_Bjlk_BBS")]
    f32_nt = [n for n in names if n.startswith("Cijk_Ailk_Bjlk_S")]
    other_gemm = [n for n in names if (n.startswith("Cijk_") and n not in bf16_nt + f32_nt) or "rocblas" in n.lower()]
    errs = []
    if len(bf16_nt) != 1 or expect not in bf16_nt[0] or "UserArgs" not in bf16_nt[0]:
        errs.append("want exactly 1 hipBLASLt bf16 NT launch with %s, got %s" % (expect, bf16_nt))
    want_f32 = 1 if r >= 2 else 0
    if len(f32_nt) != want_f32 or any("MT32x32x16" not in n or "UserArgs" not in n for n in f32_nt):
        errs.append("want %d fp32 NT MT32x32x16 tail launch(es) (r=%d), got %s" % (want_f32, r, f32_nt))
    if other_gemm:
        errs.append("unexpected GEMM / rocBLAS kernels: %s" % other_gemm)
    if errs:
        print("FAIL (kernel check): " + "; ".join(errs)); return 2
    print("kernel check OK: 1 x %s%s" % (expect, ", 1 x fp32 tail MT32x32x16" if r >= 2 else ""))
    print("output", tuple(out1.shape), out1.dtype, out1.stride(), "contiguous", out1.is_contiguous())
    if out1.shape != (D, O) or out1.dtype != torch.bfloat16 or not out1.is_contiguous():
        print("FAIL: output contract"); return 1

    if a.mode == "rand":
        CH = 1 << 18
        ref = torch.zeros(D, O, device=dev, dtype=torch.float32)
        absref = torch.zeros(D, O, device=dev, dtype=torch.float32)
        for k0 in range(0, T, CH):
            ref += torch.mm(A[k0:k0 + CH].t().float(), B[k0:k0 + CH].float())
            absref += torch.mm(A[k0:k0 + CH].t().float().abs(), B[k0:k0 + CH].float().abs())
        base = torch.mm(A.t(), B)        # batch-1 kernel of THIS lib dir (today's call)
        p = partials().float()
        q = (p.square().sum(0).sqrt().norm() / ref.norm().clamp_min(1e-30)).item()
        psum = p.abs().sum(0)
        del p
        model = (1.0 + q * q) ** 0.5
        d1, d0 = out1.float() - ref, base.float() - ref
        e1 = (d1.norm() / ref.norm().clamp_min(1e-30)).item()
        e0 = (d0.norm() / ref.norm().clamp_min(1e-30)).item()
        ratio = e1 / max(e0, 1e-30)
        bound = 2.0 ** -8 * (psum + ref.abs()) + 2.0 ** -12 * absref
        nviol = int((d1.abs() > bound).sum().item())
        print("call 1 vs fp32 ref: split rms-rel %.3e, batch-1 mm (this lib) rms-rel %.3e, ratio %.2f; "
              "chunk cancellation q %.2f -> model ratio %.2f; elementwise-bound violations %d" % (
                  e1, e0, ratio, q, model, nviol))
        bad = not torch.isfinite(out1).all() or nviol > 0
        if a.pattern == "iid":
            bad |= ratio > a.rms_ratio
        else:
            bad |= ratio > 2.0 * model   # one-sided: real kernels came in far below the model (G20 s=16 drift: 41x vs 387x)
        if bad:
            print("FAIL: call 1 vs fp32 reference"); return 1
        del ref, base, absref, psum, bound

    # timing (no checks): split path vs batch-1 mm on THIS lib dir
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ev[0].record()
    for _ in range(20):
        ta.splitt_wgrad_mm(A, B)
    ev[1].record()
    ev[2].record()
    for _ in range(3):
        torch.mm(A.t(), B)
    ev[3].record()
    torch.cuda.synchronize()
    b1sol = ent["sol_by_s"]["1"]
    print("timing: split-T %.3f ms/call (20 calls), batch-1 mm on this lib (sol %s%s) %.3f ms/call (3 calls)" % (
        ev[0].elapsed_time(ev[1]) / 20, b1sol, " = T8 baseline" if b1sol == 103 else ", NOT the T8 baseline",
        ev[2].elapsed_time(ev[3]) / 3))

    ref_bits = out1.view(torch.int16).clone()
    bad_calls, first_bad, c = 0, None, 0
    t0 = time.time()
    for c in range(1, a.calls + 1):
        out = out1 if c == 1 else ta.splitt_wgrad_mm(A, B)
        if a.mode == "zero":
            nbad = int(torch.count_nonzero(out).item())                       # NaN counts as nonzero
            nneg = int(torch.count_nonzero(out.view(torch.int16)).item()) - nbad
            if a.check_partials:
                pp = partials()
                pbad = int(torch.count_nonzero(pp).item())
                if pbad:
                    print("call %d: %d nonzero words in the bf16 partials" % (c, pbad))
                    nbad += pbad
            if nneg and c == 1:
                print("info: %d output words are -0.0 (not counted as wrong)" % nneg)
        else:
            nbad = int((out.view(torch.int16) != ref_bits).sum().item())
        if nbad:
            bad_calls += 1
            first_bad = first_bad or c
            print("call %d: %d wrong words (max |v| %.3e)" % (c, nbad, out.float().abs().max().item()))
            if bad_calls >= 5:
                break
    dt = (time.time() - t0) / max(1, c)
    print("shape %s m=%d n=%d T=%d s=%d mode=%s pattern=%s calls=%d bad_calls=%d first_bad=%s  %.2f ms/call (incl. check)"
          % (a.shape, M, N, T, s, a.mode, a.pattern, c, bad_calls, first_bad, dt * 1e3))
    print("PASS" if bad_calls == 0 else "FAIL")
    return 0 if bad_calls == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
