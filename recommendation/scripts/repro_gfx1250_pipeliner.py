"""Reproducer: Triton 3.8.0 miscompiles a software-pipelined loop on gfx1250.

Standalone -- pure Triton + torch, no repo or fbgemm imports. Exact on Triton
3.6.0 and 3.7.1, wrong on 3.8.0+git4cff872c, same torch/ROCm in all three.

This is the mechanism behind the HSTU attention failure on 3.8.0 (docs/mi450.md
defect 4). `_hstu_attn_fwd` runs its inner loop at num_stages=2 and builds K as
a transposed view

    tl.make_block_ptr(base=K, shape=(BLOCK_D_Q, seq_len), strides=(1, stride_kn),
                      block_shape=(BLOCK_D_Q, BLOCK_N), order=(0, 1))

which it feeds to tl.dot. In the e2e run the defect shows up as wrong addresses
("Memory access fault ... on address (nil)"); here it lands on mapped memory and
shows up as garbage values instead.

Three checks, in order of increasing specificity:

  A. load only -- the transposed tile is loaded and stored with no tl.dot.
     Exact on every version, so addressing is not the problem.

  B. num_stages x K operand source -- the core result. num_stages=1 is exact for
     both a transposed block pointer and plain pointer arithmetic; num_stages=2
     and 3 are wrong, but *only* for the block-pointer operand. So the defect
     needs software pipelining, and the transposed block pointer is the trigger
     the pipeliner mishandles.

  C. Q x K operand source at num_stages=2 -- only the transposed K operand
     matters; a row-major order=(1,0) operand (Q) is fine either way.

Two things this rules out. tl.advance is present in passing and failing variants
alike, so it is not implicated. And the block pointer is not sufficient on its
own -- at num_stages=1 it is exact -- which is why the fix is to clamp
num_stages (common.py clamp_num_stages) rather than to rewrite the pointer.

Caveat on scope: in the real attention kernel the pipelined loop faults even
with Q/K/V all on plain pointer arithmetic, so pipelining is broken more broadly
than this minimal case shows. This script captures the smallest trigger, not the
full blast radius.

Usage:
    python scripts/repro_gfx1250_blockptr_dot.py
Reports whether the defect is present, so it is meaningful on a healthy version
too. Exits non-zero only on a result that invalidates the analysis: a wrong
load-only tile, a wrong num_stages=1 result, or wrong plain pointer arithmetic
(which would mean neither workaround is usable).
"""

import sys

import triton
import triton.language as tl
import torch

BLOCK_M, BLOCK_N, D = 32, 32, 64
SEQ_LEN = 200  # deliberately not a multiple of BLOCK_N, so boundary_check is live
LOW = 64  # multiple of BLOCK_N, matching attention's masked-block boundary


@triton.jit
def _load_only(K, Out, seq_len, stride_kn, low, TRANSPOSED: tl.constexpr,
               BLOCK_N: tl.constexpr, D: tl.constexpr):
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    if TRANSPOSED:
        kp = tl.make_block_ptr(K, (D, seq_len), (1, stride_kn), (0, low),
                               (D, BLOCK_N), (0, 1))
        k = tl.load(kp, boundary_check=(1,), padding_option="zero")
    else:
        k = tl.load(K + offs_d[:, None] + (low + offs_n)[None, :] * stride_kn,
                    mask=(low + offs_n)[None, :] < seq_len, other=0.0)
    tl.store(Out + offs_d[:, None] * BLOCK_N + offs_n[None, :], k)


@triton.jit
def _dot(Q, K, Out, seq_len, stride_kn, low,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
         Q_BP: tl.constexpr, K_BP: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    if Q_BP:
        qp = tl.make_block_ptr(Q, (BLOCK_M, D), (D, 1), (0, 0),
                               (BLOCK_M, D), (1, 0))
        q = tl.load(qp, boundary_check=(0,), padding_option="zero")
    else:
        q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])

    kp = tl.make_block_ptr(K, (D, seq_len), (1, stride_kn), (0, low),
                           (D, BLOCK_N), (0, 1))
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start in tl.range(low, seq_len, BLOCK_N):
        if K_BP:
            k = tl.load(kp, boundary_check=(1,), padding_option="zero")
        else:
            offs_n = start + tl.arange(0, BLOCK_N)
            k = tl.load(K + offs_d[:, None] + offs_n[None, :] * stride_kn,
                        mask=offs_n[None, :] < seq_len, other=0.0)
        acc += tl.sum(tl.dot(q, k), axis=1)
        kp = tl.advance(kp, (0, BLOCK_N))
    tl.store(Out + offs_m, acc)


def _run_dot(q, k, ref, q_bp, k_bp, stages):
    out = torch.zeros(BLOCK_M, device="cuda", dtype=torch.float32)
    _dot[(1,)](q, k, out, SEQ_LEN, k.stride(0), LOW,
               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D, Q_BP=q_bp, K_BP=k_bp,
               num_stages=stages, num_warps=4)
    torch.cuda.synchronize()
    return ((out - ref).abs() / ref.abs().clamp(min=1.0)).max().item()


def main() -> int:
    print(f"triton {triton.__version__} | torch {torch.__version__}")
    print(f"arch {torch.cuda.get_device_properties(0).gcnArchName}")
    torch.manual_seed(0)
    q = torch.randn(BLOCK_M, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(SEQ_LEN, D, device="cuda", dtype=torch.bfloat16)
    ref = (q.float() @ k.float()[LOW:].T).sum(dim=1)
    failures = []

    print("\nA. load only, no tl.dot")
    ref_tile = k[LOW:LOW + BLOCK_N].T.contiguous()
    for transposed in (True, False):
        out = torch.zeros(D, BLOCK_N, device="cuda", dtype=torch.bfloat16)
        _load_only[(1,)](k, out, SEQ_LEN, k.stride(0), LOW,
                         TRANSPOSED=transposed, BLOCK_N=BLOCK_N, D=D)
        torch.cuda.synchronize()
        bad = (out != ref_tile).sum().item()
        source = "blockptr order=(0,1)" if transposed else "plain arithmetic"
        print(f"   K via {source:<21}: mismatched {bad:>5} / {ref_tile.numel()}"
              f"  {'OK' if bad == 0 else 'WRONG'}")
        if bad:
            failures.append(f"load-only via {source} is wrong (expected exact)")

    print("\nB. num_stages x K operand source (Q always plain)")
    defect = False
    for stages in (1, 2, 3):
        for k_bp in (True, False):
            err = _run_dot(q, k, ref, q_bp=False, k_bp=k_bp, stages=stages)
            wrong = err >= 1e-2
            src = "blockptr" if k_bp else "plain"
            print(f"   num_stages={stages}  K={src:<8}: max rel err {err:.3e}"
                  f"  {'WRONG' if wrong else 'OK'}")
            if wrong and (stages == 1 or not k_bp):
                failures.append(
                    f"num_stages={stages} K={src} is wrong; expected exact "
                    "(this invalidates the num_stages=1 workaround)"
                )
            defect |= wrong and k_bp and stages >= 2

    print("\nC. Q x K operand source at num_stages=2")
    for q_bp in (True, False):
        for k_bp in (True, False):
            err = _run_dot(q, k, ref, q_bp=q_bp, k_bp=k_bp, stages=2)
            print(f"   Q={'blockptr' if q_bp else 'plain':<8} "
                  f"K={'blockptr' if k_bp else 'plain':<8}: max rel err "
                  f"{err:.3e}  {'WRONG' if err >= 1e-2 else 'OK'}")

    print()
    if defect:
        print("DEFECT PRESENT: pipelined (num_stages >= 2) tl.dot on a "
              "transposed block-pointer operand is wrong; num_stages=1 and "
              "plain pointer arithmetic are exact.")
    else:
        print("DEFECT ABSENT: every variant agrees on this version.")
    for f in failures:
        print("UNEXPECTED:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
