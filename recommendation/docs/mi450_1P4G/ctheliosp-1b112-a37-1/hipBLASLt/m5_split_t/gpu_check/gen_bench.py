#!/usr/bin/env python3
"""Generate the M5 single-GEMM hipblaslt-bench YAMLs (CPU only; NOT run here).

One file per T, because gemm_watch.sh's T=<n> substitution only rewrites the literal 2097152 and
cannot fix K = T//s or the batch strides:
  m5_bench_T2097152.yaml  m5_bench_T2533005.yaml   (+ .index.txt: line -> shape, s, expected sol)
Run them WITHOUT setting T:
  gemm_watch.sh /home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v3/s103 <dev> <out> m5_bench_T2097152.yaml
Lines per shape: s=1 baseline (batch 1, K=T); every candidate s as the exact strided-batched problem
torch.bmm(a3.transpose(1,2), b3) issues (hipBLASLt NT, M=O, N=D, K=L=T//s, lda=M, ldb=N, ldc=ldd=M,
stride_a=M*L, stride_b=N*L, stride_c=stride_d=M*N, batch_count=s, bf16 in/out, f32 compute); and,
when T % s >= 2, the fp32 tail GEMM (NT, K=T%s, beta=1, f32 in/out; SS catalog). T % s == 1 is an
elementwise fp32 addr_ in the M5 helper (no GEMM; a [D,1] operand's NN/NT layout is ambiguous), so it
is listed in the index only. v3/mix = G20/G19 on sol 103, the others on 102: the s103 and s102 runs
together cover it.
Cases the M5 helper refuses (L*max(M,N) >= 2^31) are listed in the index and not emitted.
"""
import os
import sys

sys.path.insert(0, "/home/chcai/m5_work")
from make_v3 import SHAPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TS = (2_097_152, 2_533_005)
LINE = ("- {{ function: matmul, M: {m}, N: {n}, K: {k}, lda: {m}, ldb: {n}, ldc: {m}, ldd: {m}, "
        "stride_a: {sa}, stride_b: {sb}, stride_c: {sc}, stride_d: {sc}, alpha: 1.000000, beta: {beta}, "
        "transA: N, transB: T, batch_count: {b}, scaleA: 0, scaleB: 0, scaleAlpha_vector: false, gradient: false, "
        "use_e: false, bias_vector: false, bias_source: d, a_type: {t}, b_type: {t}, c_type: {t}, d_type: {t}, "
        "scale_type: f32_r, bias_type: f32_r, aux_type: {t}, compute_type: c_f32_r, activation_type: none, "
        "flush: false, any_stride: true, rotating: 0, cold_iters: 0, iters: 0 }}")


def gen(T):
    out, idx = [], []
    for nm, m, n, _, ss in SHAPES:
        for s in (1,) + tuple(ss):
            L = T // s
            r = T - s * L
            if s > 1 and L * max(m, n) >= 1 << 31:
                idx.append("--  %s s=%d skipped: batch stride %d >= 2^31 (M5 helper falls back to torch.mm)"
                           % (nm, s, L * max(m, n)))
                continue
            out.append(LINE.format(m=m, n=n, k=L, sa=m * L if s > 1 else 0, sb=n * L if s > 1 else 0,
                                   sc=m * n if s > 1 else 0, b=s, t="bf16_r", beta="0.000000"))
            idx.append("%-3d %s s=%-3d bf16 NT M=%d N=%d K=%d batch=%d  expect: v3/s103 MT256x128x64 (sol 103), "
                       "v3/s102 MT32x16x32 (sol 102), v3/mix sol %d" % (len(out), nm, s, m, n, L, s,
                                                                       103 if nm in ("G20", "G19") else 102))
            if r == 1:
                idx.append("--  %s s=%-3d tail r=1: elementwise fp32 addr_ (no GEMM), not benched" % (nm, s))
            elif r:
                out.append(LINE.format(m=m, n=n, k=r, sa=0, sb=0, sc=0, b=1, t="f32_r", beta="1.000000"))
                idx.append("%-3d %s s=%-3d fp32 tail M=%d N=%d K=%d beta=1  expect: SS NT catalog (sol 2821 MT32x32x16)"
                           % (len(out), nm, s, m, n, r))
    # gemm_watch.sh T=<n> rewrites every literal 2097152 -- also inside strides (e.g. G27 s=256
    # stride_b = 256*8192 = 2097152) -- so these files must be run with T unset.
    hazard = [i + 1 for i, ln in enumerate(out) if "2097152" in ln and "K: 2097152," not in ln]
    return out, idx, hazard


def main():
    for T in TS:
        out, idx, hazard = gen(T)
        base = os.path.join(HERE, "m5_bench_T%d" % T)
        with open(base + ".yaml", "w") as f:
            f.write("\n".join(out) + "\n")
        with open(base + ".index.txt", "w") as f:
            f.write("# %s.yaml: T=%d, %d lines (gemm_watch.sh line numbers). Do NOT set T= (lines with a "
                    "literal 2097152 in a stride: %s)\n" % (os.path.basename(base), T, len(out), hazard or "none"))
            f.write("\n".join(idx) + "\n")
        print("%s.yaml: %d lines" % (base, len(out)))


if __name__ == "__main__":
    main()
