#!/usr/bin/env python3
"""gfx1250 A0: does the extended-VGPR hazard corrupt *values*, not just addresses?

Depends only on torch + triton. No repo imports, no fbgemm, no TorchRec, no
dataset -- the same bar as ``repro_gfx1250_buffer_ops.py``, so this can be handed
to the LLVM/AMDGPU backend team on its own.

WHY THIS EXISTS
---------------
raikonenfnu/training#3 root-caused defect [6] (``_hstu_attn_bwd`` faulting on a
real address) to LLVM machine code: an address-critical, work-item-derived lane
offset is kept live in extended VGPR ``v257`` across the WMMA body under mutable
``S_SET_VGPR_MSB`` state, and on this A0 **lane 0 of it comes back holding an
accumulator-like FP32 bit pattern**. A later integer multiply amplifies that one
word and the first final-DV ``global_store_b128`` faults.

That story has an untested corollary. The store is only the *observer*. If the
value stranded in an extended VGPR is arithmetic rather than address-critical,
the same corruption produces **silently wrong numbers instead of a page fault**
-- no fault, no kernel name, no module boundary to hook. That is precisely the
profile of defect [3] (``train_loss=nan`` at ``BATCH_SIZE=128``, 3 of 4 runs,
``DEBUG_NAN_HOOKS`` unable to name a producer).

This script tests the corollary directly. It builds a kernel with the same
hazardous *allocation shape* as the failing one -- a cheap work-item-derived
witness computed before a long WMMA chain and consumed after it -- then stores
the witness **as data** and compares every lane against an analytic expectation.

WHAT IT FOUND (2026-09-09, gfx1250 A0, Triton 7ff97e3109)
---------------------------------------------------------
1. The allocation reproduces standalone: 602 VGPRs, **zero spills**, with
   ``v_and_b32 v82 /*v594*/, 31, v0`` defined before the first ``v_wmma`` and
   read back after the last across 66 static ``s_set_vgpr_msb`` mode changes.
   ``docs/mi450_a0/mi450_a0.md`` previously recorded no standalone reproducer at all.
2. **The corollary was NOT confirmed.** Under clean launch conditions that
   kernel is bit-exact over 200 iterations x 6 repeats. An extended-VGPR carrier
   alone does not produce wrong numbers.
3. A *different* defect does show up at ``--occupancy 16 --streams 4``: a page
   fault ~4 GiB outside every data buffer, which needs both high register
   pressure and stream concurrency. It is **not** the extended-VGPR mechanism --
   ``-mattr=-1024-addressable-vgprs`` makes it worse (5/6 vs 3/6), and it
   survives with zero extended registers. See ``docs/gfx1250_extvgpr_handoff.md``.

So detection here is numeric, but note that at high occupancy with concurrent
streams this reproducer *can* fault (finding 3), unlike what its first version
claimed.

    # 1. Find a config that actually produces the hazard (never dispatches):
    AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_extvgpr_value.py --sweep

    # 2. Inspect one config's codegen (never dispatches):
    AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_extvgpr_value.py --compile-only

    # 3. Run it:
    AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_extvgpr_value.py --iters 2000

NOT MINIMIZED, ON PURPOSE
-------------------------
raikonenfnu/training#3 records that reducing to two isolated WMMAs deletes the
symptom: "the full kernel's register pressure and instruction stream are required
to observe the intermittent corruption." So pressure here is a *parameter*, and
``--compile-only`` refuses to report a meaningful result unless the emitted
AMDGCN actually contains the hazard. A PASS from a kernel with no
``s_set_vgpr_msb`` proves nothing, and this script says so rather than
reporting a green tick.

CONTROLS
--------
  --remat      recompute the witness after the last dot (issue #3's fix 1)
  --no-wmma    identical pressure, no dots -- isolates WMMA as the ingredient
  --occupancy  fill the 256 WGPs; every existing [3] probe runs one small grid,
               while the failing e2e run at batch 128 has ~16x the workgroups
               resident. Shape has been tested to exhaustion; occupancy has not.
  --dump-llir  emit the .ll for an -mattr=-1024-addressable-vgprs rebuild
               (issue #3's fix 2, the whole-function control)

One process per run.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

# Pressure defaults chosen to land near the 1024-VGPR ceiling the doc records for
# the faulting BLOCK_N=128 attention-backward allocation (944/986/1024/1024 VGPRs,
# 0/0/29/33 spilled). --sweep exists because the exact knob values that reach the
# hazardous allocation depend on the Triton and LLVM revision.
# These are not guesses: --sweep measured 60 configurations on gfx1250 with
# Triton 7ff97e3109, and this is the one whose AMDGCN actually contains the
# mechanism. It emits `v_and_b32 v82 /*v594*/, 31, v0` -- a lane-id mask in an
# EXTENDED VGPR -- before the first `v_wmma`, and reads it back afterwards to
# form an address (`v_lshlrev_b32 v2, 2, v82 /*v594*/`) across 66 static
# `s_set_vgpr_msb` mode changes, with **zero spills**. That is the same
# instruction family, the same register class and the same mutable-msb exposure
# as the `v257` lane offset in raikonenfnu/training#3.
#
# Bigger is not better here. BLOCK_N=256/NACC=4 reaches the doc's 1024-VGPR
# ceiling but spills 255 registers, and a value spilled to scratch is NOT live
# in an extended VGPR across the region -- that configuration cannot exhibit the
# bug at all, and would report a meaningless PASS.
DEF_BLOCK_M = 128
DEF_BLOCK_N = 128
DEF_BLOCK_K = 32
DEF_NACC = 4
DEF_NDOT = 64
DEF_WARPS = 4
DTYPE = torch.bfloat16

# Number of independent long-lived work-item-derived values held across the WMMA
# chain. One is not enough: with a single witness the register allocator has no
# reason to place it above v255, and measurement confirmed it does not -- the
# store came out `global_store_b32 v128, v0`, both low, so the binary could not
# exhibit the mechanism at all. Issue #3's kernel has enough simultaneously-live
# work-item values that some are forced into extended registers; NWIT reproduces
# that pressure directly instead of hoping the allocator obliges.
DEF_NWIT = 64

# The witness is affine in the work-item index, like the lane offset in the
# failing kernel (v_mbcnt_lo -> v_and -> v_lshrrev). Odd multiplier and offset so
# a corrupted lane cannot coincidentally match a neighbour, and a large per-row
# stride so a corrupted value cannot coincidentally match another row's.
WIT_MUL = 3
WIT_ADD = 1
WIT_STRIDE = 1000003


@triton.jit
def _extvgpr_value_kernel(
    A,
    B,
    WitOut,
    AccOut,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NACC: tl.constexpr,
    NDOT: tl.constexpr,
    NWIT: tl.constexpr,
    USE_WMMA: tl.constexpr,
    REMAT: tl.constexpr,
    WIT_MUL: tl.constexpr,
    WIT_ADD: tl.constexpr,
    WIT_STRIDE: tl.constexpr,
):
    """Long-lived work-item value across a WMMA chain, read back as data.

    The shape that matters is the *live range*, not the arithmetic: `wit` is
    defined before the first `tl.dot` and consumed after the last one, with
    enough simultaneously-live accumulators in between to push the allocation
    into extended VGPRs. That is the allocation raikonenfnu/training#3 found in
    `_hstu_attn_bwd`, with the store address swapped for a plain data write.
    """
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = tl.arange(0, NWIT)

    # --- witnesses: defined BEFORE the WMMA region, NWIT of them, all live ---
    base = (pid.to(tl.int32) * BLOCK_M + offs_m) * WIT_MUL + WIT_ADD
    wit = base[None, :] + offs_w[:, None] * WIT_STRIDE

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --- the WMMA region: NDOT iterations x NACC live accumulators ---
    for i in range(NDOT):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if USE_WMMA:
            acc0 += tl.dot(a, b)
            if NACC > 1:
                acc1 += tl.dot(a, b * 1.0001)
            if NACC > 2:
                acc2 += tl.dot(a * 1.0002, b)
            if NACC > 3:
                acc3 += tl.dot(a * 1.0003, b * 1.0003)
        else:
            # Same live state and same loop trip count, no matrix ops. The
            # control for "is WMMA necessary, or is it just pressure?".
            s = tl.sum(a.to(tl.float32), axis=1)[:, None] + tl.sum(
                b.to(tl.float32), axis=0
            )[None, :]
            acc0 += s
            if NACC > 1:
                acc1 += s * 1.0001
            if NACC > 2:
                acc2 += s * 1.0002
            if NACC > 3:
                acc3 += s * 1.0003
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- witness consumed AFTER the WMMA region ---
    if REMAT:
        # Issue #3's fix 1, expressed at Triton level: recompute the lane
        # expression here instead of carrying it across the dots. NOTE: LLVM is
        # free to CSE this back into the original definition, which would make
        # the control vacuous -- check with --compile-only rather than trusting
        # the flag. The paper-proof version of this fix is a machine-level
        # rematerialisation, not a source-level one.
        rebase = (pid.to(tl.int32) * BLOCK_M + tl.arange(0, BLOCK_M)) * WIT_MUL + WIT_ADD
        wit = rebase[None, :] + tl.arange(0, NWIT)[:, None] * WIT_STRIDE

    tl.store(
        WitOut + (pid * NWIT + offs_w)[:, None] * BLOCK_M + offs_m[None, :],
        wit,
    )

    # Reduce the accumulators so none of the pressure is dead-code-eliminated.
    tot = acc0
    if NACC > 1:
        tot += acc1
    if NACC > 2:
        tot += acc2
    if NACC > 3:
        tot += acc3
    tl.store(AccOut + pid * BLOCK_M + offs_m, tl.sum(tot, axis=1))


# --------------------------------------------------------------------------
# AMDGCN analysis
# --------------------------------------------------------------------------
# The LLVM AMDGPU assembly printer annotates extended VGPR references with the
# logical register in a comment, because the encoding only carries the low 8
# bits -- e.g. `v_lshrrev_b32_e32 v1 /* v257 */, 1, v16` in issue #3. That
# annotation is what makes the hazard visible in text.
_EXT_REF = re.compile(r"/\*\s*v(\d+)(?::(\d+))?\s*\*/")
_WMMA = re.compile(r"^\s*(v_wmma\w*)", re.M)
_SETMSB = re.compile(r"^\s*s_set_vgpr_msb\b", re.M | re.I)

# Instructions with no VGPR destination: every VGPR operand is a *read*. Without
# this list a store's address operand would be misread as a definition.
_NO_VDST = re.compile(
    r"^(global|scratch|buffer|flat|ds)_(store|write|atomic)"
    r"|^s_|^v_cmp|^exp\b|^v_nop|^ds_write"
)


def _operands(line: str) -> List[Tuple[str, List[Tuple[int, int, bool]]]]:
    """Split one asm line into (mnemonic, [(reg_lo, reg_hi, is_write)]).

    `is_write` is true only for the destination operand, so a value can be
    tracked as a real def-use chain rather than as "some line mentions v770".
    That distinction is the whole point: LLVM recycles register numbers, and two
    references to v770 either side of a WMMA chain are usually two unrelated
    values, not one long live range.
    """
    body = line.split(";", 1)[0].strip()
    if not body or body.endswith(":") or body.startswith("."):
        return []
    out: List[Tuple[str, List[Tuple[int, int, bool]]]] = []
    # `v_dual_mov_b32 v0, v2 :: v_dual_mov_b32 v1, v3` is two instructions, each
    # with its own destination.
    for part in body.split("::"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(None, 1)
        mnemonic = bits[0]
        ops = bits[1].split(",") if len(bits) > 1 else []
        has_vdst = not _NO_VDST.search(mnemonic)
        regs: List[Tuple[int, int, bool]] = []
        for idx, op in enumerate(ops):
            for m in _EXT_REF.finditer(op):
                lo = int(m.group(1))
                hi = int(m.group(2)) if m.group(2) else lo
                regs.append((lo, hi, has_vdst and idx == 0))
        out.append((mnemonic, regs))
    return out


def analyze_amdgcn(asm: str) -> Dict[str, object]:
    """Locate extended-VGPR values whose live range spans the WMMA region.

    Returns raw evidence, not just a verdict: the LLVM team needs the line
    numbers and the surrounding instructions, and a heuristic that silently
    said "hazard: no" would be worse than useless here.

    Liveness here is a straight-line approximation: a value counts as spanning
    only if it is *written* before the first `v_wmma`, *read* after the last one,
    and never rewritten in between. That is sound for the flat body this kernel
    compiles to, and it is deliberately conservative -- it will miss values that
    live across a branch. It will not invent one, which is what matters.
    """
    lines = asm.splitlines()
    wmma_idx = [i for i, ln in enumerate(lines) if re.match(r"^\s*v_wmma", ln)]
    msb_idx = [i for i, ln in enumerate(lines) if re.match(r"^\s*s_set_vgpr_msb", ln, re.I)]

    # (line, reg, is_write) for every extended-register operand, in order.
    refs: List[Tuple[int, int, bool]] = []
    ext_regs = set()
    for i, ln in enumerate(lines):
        for _mn, regs in _operands(ln):
            for lo, hi, is_write in regs:
                for reg in range(lo, hi + 1):
                    if reg >= 256:
                        refs.append((i, reg, is_write))
                        ext_regs.add(reg)

    by_reg: Dict[int, List[Tuple[int, bool]]] = {}
    for i, reg, w in refs:
        by_reg.setdefault(reg, []).append((i, w))

    spanning: List[Dict[str, object]] = []
    if wmma_idx:
        first_wmma, last_wmma = wmma_idx[0], wmma_idx[-1]
        for reg, hist in sorted(by_reg.items()):
            writes_before = [i for i, w in hist if w and i < first_wmma]
            reads_after = [i for i, w in hist if not w and i > last_wmma]
            if not (writes_before and reads_after):
                continue
            d, u = writes_before[-1], reads_after[0]
            # A rewrite between the two means these are different values.
            if any(w and d < i < u for i, w in hist):
                continue
            # Mode changes that occur while this value is live -- the
            # "hundreds of mode transitions while v257 remains live" in #3.
            spanning.append({
                "reg": reg,
                "def": d,
                "use": u,
                "msb_between": sum(1 for i in msb_idx if d < i < u),
                "kind": _classify(lines, d, u),
            })

    return {
        "n_wmma": len(wmma_idx),
        "n_set_vgpr_msb": len(msb_idx),
        "ext_regs": sorted(ext_regs),
        "spanning": spanning,
        "lines": lines,
        "first_wmma": wmma_idx[0] if wmma_idx else None,
        "last_wmma": wmma_idx[-1] if wmma_idx else None,
    }


# Accumulators are *supposed* to live across a WMMA chain -- they are the dot's
# own operands, and finding one there says nothing. Issue #3's hazard is
# specifically a cheap value LLVM could have rematerialised but chose to carry.
# Conflating the two would let this script report the mechanism when all it has
# found is a matmul, so the two cases are separated and reported separately.
_ACC_DEF = re.compile(r"v_(dual_)?mov_b32|v_accvgpr|v_mfma|v_wmma|v_pk_")
_ACC_USE = re.compile(r"v_pk_add_f32|v_add_f32|v_wmma|v_mfma|v_cvt_")
# Issue #3's victim: `v_mbcnt_lo_u32_b32` / `v_and_b32` / `v_lshrrev_b32`, i.e.
# a lane index massaged into an offset.
_WI_DEF = re.compile(r"mbcnt|lshrrev|v_and_b32|v_bfe|workitem|v_lshl")


def _classify(lines: List[str], def_i: int, use_i: int) -> str:
    d, u = lines[def_i], lines[use_i]
    if _WI_DEF.search(d):
        return "work-item-derived"
    if _ACC_DEF.search(d) and _ACC_USE.search(u):
        return "accumulator-like"
    return "other"


def hazard_present(info: Dict[str, object]) -> bool:
    """True only if this binary can actually exhibit the #3 mechanism.

    Requires extended-VGPR mode state (`s_set_vgpr_msb`) and at least one value
    carried across the whole WMMA region. Accumulator-like carriers still count:
    if the A0 fails to preserve extended VGPRs across mode changes, a corrupted
    accumulator is exactly the silently-wrong-number outcome defect [3] shows,
    and is if anything a closer analogue than a corrupted address.
    """
    return bool(info["n_set_vgpr_msb"]) and bool(info["spanning"])


# --------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------
def _kernel_kwargs(args, use_wmma: bool) -> dict:
    return dict(
        BLOCK_M=args.block_m,
        BLOCK_N=args.block_n,
        BLOCK_K=args.block_k,
        NACC=args.nacc,
        NDOT=args.ndot,
        NWIT=args.nwit,
        USE_WMMA=use_wmma,
        REMAT=args.remat,
        WIT_MUL=WIT_MUL,
        WIT_ADD=WIT_ADD,
        WIT_STRIDE=WIT_STRIDE,
        num_warps=args.warps,
        num_stages=args.stages,
    )


def compile_only(args, use_wmma: bool):
    """Compile without dispatching. Cannot fault, cannot wedge the node."""
    device = torch.device("cuda")
    a = torch.empty((args.block_m, args.block_k * (args.ndot + 1)), device=device, dtype=DTYPE)
    b = torch.empty((args.block_k * (args.ndot + 1), args.block_n), device=device, dtype=DTYPE)
    wit = torch.empty((args.nwit * args.block_m,), device=device, dtype=torch.int32)
    acc = torch.empty((args.block_m,), device=device, dtype=torch.float32)

    compiled = _extvgpr_value_kernel.warmup(
        a,
        b,
        wit,
        acc,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        grid=(1,),
        **_kernel_kwargs(args, use_wmma),
    )
    return compiled


def _regs(compiled) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """VGPRs, spilled VGPRs, scratch bytes -- read out of the AMDGCN metadata.

    `warmup()` does not populate `CompiledKernel.n_regs`/`n_spills` (those are
    filled in when the module is actually loaded, which is the path
    `bench_attn_bwd_blockn.py` uses via `fn.device_caches`). Parsing the
    assembly's own `.vgpr_count` / `; ScratchSize:` metadata keeps this
    reportable without dispatching anything.
    """
    asm = compiled.asm["amdgcn"]

    def find(pat: str) -> Optional[int]:
        m = re.search(pat, asm, re.M)
        return int(m.group(1)) if m else None

    n_regs = find(r"^\s*\.vgpr_count:\s*(\d+)") or find(r"^;\s*NumVgprs:\s*(\d+)")
    n_spills = find(r"^\s*\.vgpr_spill_count:\s*(\d+)")
    scratch = find(r"^;\s*ScratchSize:\s*(\d+)")
    return n_regs, n_spills, scratch


def report_codegen(args, compiled, label: str) -> Dict[str, object]:
    asm = compiled.asm["amdgcn"]
    info = analyze_amdgcn(asm)
    n_regs, n_spills, scratch = _regs(compiled)

    print(f"\n-- codegen [{label}] "
          f"BLOCK_M={args.block_m} BLOCK_N={args.block_n} BLOCK_K={args.block_k} "
          f"NACC={args.nacc} NDOT={args.ndot} warps={args.warps} stages={args.stages}",
          flush=True)
    print(f"   n_regs={n_regs} n_spills={n_spills} scratch={scratch}", flush=True)
    print(f"   v_wmma*={info['n_wmma']}  s_set_vgpr_msb={info['n_set_vgpr_msb']}  "
          f"extended regs referenced={len(info['ext_regs'])}", flush=True)

    spanning = info["spanning"]
    if spanning:
        by_kind: Dict[str, int] = {}
        for s in spanning:
            by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
        print(f"   extended VGPRs live ACROSS the whole WMMA region: {len(spanning)} "
              f"({', '.join(f'{v} {k}' for k, v in sorted(by_kind.items()))})",
              flush=True)
        # Report per kind, not overall. "work-item-derived" is the kind that
        # matches issue #3's victim and it is rare; a flat head-of-list would be
        # all accumulators and would hide it.
        for kind in ("work-item-derived", "other", "accumulator-like"):
            group = [s for s in spanning if s["kind"] == kind]
            if not group:
                continue
            print(f"     [{kind}] {len(group)}:", flush=True)
            for s in group[: args.max_report]:
                print(f"       v{s['reg']}: def line {s['def']} -> "
                      f"use line {s['use']}, {s['msb_between']} s_set_vgpr_msb "
                      f"mode changes in between", flush=True)
                if args.show_asm:
                    lines = info["lines"]
                    for i in (s["def"], s["use"]):
                        lo, hi = max(0, i - 2), min(len(lines), i + 3)
                        print(f"         ...line {i}...", flush=True)
                        for j in range(lo, hi):
                            mark = ">>" if j == i else "  "
                            print(f"         {mark} {lines[j].rstrip()}", flush=True)
            if len(group) > args.max_report:
                print(f"       ... {len(group) - args.max_report} more "
                      f"(raise --max-report to show)", flush=True)
    return info


def dump_llir(compiled, path: str) -> None:
    ir = compiled.asm.get("llir") or compiled.asm.get("llvm-ir")
    if ir is None:
        print(f"!! no LLVM IR in compiled.asm (keys: {sorted(compiled.asm)})", flush=True)
        return
    with open(path, "w") as fh:
        fh.write(ir)
    print(f"-- wrote LLVM IR to {path}", flush=True)
    print("   rebuild the whole-function control (issue #3 fix 2) with:", flush=True)
    print(f"     llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 "
          f"-mattr=-1024-addressable-vgprs -O3 -filetype=asm -o out.s {path}", flush=True)


# --------------------------------------------------------------------------
# Runtime check
# --------------------------------------------------------------------------
def run(args, use_wmma: bool, info: Dict[str, object]) -> int:
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    n_blocks = args.occupancy * props.multi_processor_count

    k_total = args.block_k * (args.ndot + 1)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    a = (torch.rand((args.block_m, k_total), generator=gen) * 0.1).to(device=device, dtype=DTYPE)
    b = (torch.rand((k_total, args.block_n), generator=gen) * 0.1).to(device=device, dtype=DTYPE)

    # One output buffer PER STREAM. Concurrent launches writing the same buffer
    # would store identical values, so the aliasing is arguably benign -- but
    # "arguably benign" is not good enough for an artifact whose whole job is to
    # attribute a corrupted value. With separate buffers a reported mismatch
    # cannot be this harness racing with itself.
    n_buf = max(1, args.streams)
    wits = [torch.empty((n_blocks * args.nwit * args.block_m,), device=device,
                        dtype=torch.int32) for _ in range(n_buf)]
    accs = [torch.empty((n_blocks * args.block_m,), device=device,
                        dtype=torch.float32) for _ in range(n_buf)]

    # Analytic expectation, computed on the host. Every element is predictable:
    # element (pid, w, m) is (pid*BLOCK_M + m)*WIT_MUL + WIT_ADD + w*WIT_STRIDE.
    m = torch.arange(args.block_m, device=device, dtype=torch.int32)
    p = torch.arange(n_blocks, device=device, dtype=torch.int32)
    w = torch.arange(args.nwit, device=device, dtype=torch.int32)
    expect = (
        ((p[:, None, None] * args.block_m + m[None, None, :]) * WIT_MUL + WIT_ADD)
        + w[None, :, None] * WIT_STRIDE
    ).reshape(-1)

    streams = [torch.cuda.Stream() for _ in range(args.streams)] if args.streams > 1 else [None]
    kwargs = _kernel_kwargs(args, use_wmma)

    print(f"\n-- launching: grid={n_blocks} blocks "
          f"({args.occupancy} x {props.multi_processor_count} WGPs), "
          f"streams={args.streams}, iters={args.iters}", flush=True)

    # Print every buffer's address range BEFORE dispatching. A page fault kills
    # the process, so this is the only chance to record them -- and comparing the
    # faulting address in dmesg against these ranges is what distinguishes a
    # corrupted store address (inside/near a buffer) from a scratch-backing
    # problem (nowhere near any of them).
    for nm, t in ([("a", a), ("b", b)]
                  + [(f"wit[{j}]", t) for j, t in enumerate(wits)]
                  + [(f"acc[{j}]", t) for j, t in enumerate(accs)]):
        lo = t.data_ptr()
        hi = lo + t.numel() * t.element_size()
        print(f"   {nm:9s} 0x{lo:012x} .. 0x{hi:012x} "
              f"({t.numel() * t.element_size() / 2**20:.1f} MiB)", flush=True)

    def launch() -> None:
        for j, st in enumerate(streams):
            ctx = torch.cuda.stream(st) if st is not None else _null()
            with ctx:
                _extvgpr_value_kernel[(n_blocks,)](
                    a, b, wits[j], accs[j],
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    **kwargs,
                )
        torch.cuda.synchronize()

    # Reference from the first launch. The kernel is deterministic and the
    # inputs never change, so every later launch must be BIT-IDENTICAL. This is
    # the same methodology repro_gfx1250_attn_ref.py uses ("bit-exact over 40
    # iterations"), and it needs no tolerance and no host reference -- any
    # difference at all is corruption.
    for t in wits:
        t.fill_(-1)
    launch()
    ref_wit = wits[0].clone()
    ref_acc = accs[0].clone()

    n_wit_wrong = int((ref_wit != expect).sum())
    if n_wit_wrong:
        print(f"!! witness wrong on the very first launch: {n_wit_wrong} of "
              f"{ref_wit.numel()} lanes. Either the kernel is buggy or the "
              f"corruption is deterministic -- check --no-wmma before "
              f"blaming the hardware.", flush=True)
    if not torch.isfinite(ref_acc).all():
        print(f"!! accumulator non-finite on the very first launch", flush=True)

    bad_steps = 0
    for step in range(args.iters):
        if step % args.print_every == 0:
            print(f"-- step={step} blocks={n_blocks} streams={args.streams}", flush=True)
        for t in wits:
            t.fill_(-1)
        launch()

        hits: List[str] = []
        for j, (wit, acc) in enumerate(zip(wits, accs)):
            tag = f"stream {j}" if len(wits) > 1 else "witness"

            # (1) the cheap work-item value -- the direct analogue of #3's lane
            #     offset, checked against a closed-form host expectation
            mism = wit != expect
            if int(mism.sum()):
                n = int(mism.sum())
                first = int(torch.nonzero(mism)[0])
                got, exp = int(wit[first]), int(expect[first])
                m_i = first % args.block_m
                w_i = (first // args.block_m) % args.nwit
                pid_i = first // (args.block_m * args.nwit)
                hits.append(f"{tag} witness: {n} of {wit.numel()} elements wrong "
                            f"({100.0 * n / wit.numel():.4f}%); first at "
                            f"block={pid_i} witness={w_i} row={m_i} "
                            f"expected {exp} (0x{exp & 0xffffffff:08x}) got {got} "
                            f"(0x{got & 0xffffffff:08x}) = {_as_f32(got)!r} as fp32 "
                            f"[#3's signature is an accumulator-like FP32 pattern]; "
                            f"lane-in-wave {m_i % props.warp_size}")

            # (2) run-to-run bit-exactness of the WMMA result -- the analogue of
            #     [3], where the victim is arithmetic rather than an address
            acc_diff = acc != ref_acc
            if int(acc_diff.sum()):
                n = int(acc_diff.sum())
                first = int(torch.nonzero(acc_diff)[0])
                hits.append(f"{tag} accumulator: {n} of {acc.numel()} elements "
                            f"differ from launch 0 on identical inputs; first at "
                            f"{first}: {float(ref_acc[first])!r} -> "
                            f"{float(acc[first])!r}")

            # (3) outright non-finite
            if not torch.isfinite(acc).all():
                hits.append(f"{tag} accumulator: "
                            f"{int((~torch.isfinite(acc)).sum())} of "
                            f"{acc.numel()} non-finite")

        if hits:
            bad_steps += 1
            print(f"!! CORRUPTION step={step} blocks={n_blocks} "
                  f"streams={args.streams}", flush=True)
            for h in hits:
                print(f"   {h}", flush=True)
            if not args.keep_going:
                return 1

    torch.cuda.synchronize()
    verdict = "PASS" if bad_steps == 0 else f"FAIL ({bad_steps} of {args.iters} steps)"
    print(f"\n-- {verdict}: {args.iters} iterations, {n_blocks} blocks/launch, "
          f"{args.streams} stream(s)", flush=True)
    if bad_steps == 0 and not hazard_present(info):
        if not info["n_wmma"]:
            # --no-wmma is a control for "is WMMA necessary?", but it is a weak
            # one: with no v_wmma there is no region to span, so this analyzer is
            # silent BY CONSTRUCTION rather than by evidence. Saying "hazard
            # absent" here would be claiming a measurement that was never made.
            print("!! note: this binary contains no v_wmma at all, so the "
                  "spanning-value analysis is not applicable -- it reports "
                  "nothing here because there is no WMMA region to span, not "
                  "because the hazard was checked and found absent.", flush=True)
        else:
            print("!! ...but this PASS is VACUOUS: the emitted AMDGCN contains no "
                  "extended-VGPR value live across the WMMA region, so the hazard "
                  "was never present to begin with. Re-run --sweep and pick a "
                  "config whose codegen shows the mechanism.", flush=True)
        return 2
    return 0 if bad_steps == 0 else 1


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _as_f32(i: int) -> float:
    import struct
    return struct.unpack("<f", struct.pack("<i", i))[0]


# --------------------------------------------------------------------------
def sweep(args) -> int:
    """Compile-only search for a configuration that produces the hazard."""
    print("-- sweeping for a configuration whose codegen shows an extended VGPR "
          "live across the WMMA region (compile only, never dispatches)", flush=True)
    grid = []
    for block_m in (64, 128):
        for block_n in (128, 256):
            for nacc in (2, 4):
                for nwit in (16, 64, 128):
                    for warps in (4, 8):
                        grid.append((block_m, block_n, nacc, nwit, warps))

    found = []
    for block_m, block_n, nacc, nwit, warps in grid:
        (args.block_m, args.block_n, args.nacc, args.nwit, args.warps) = (
            block_m, block_n, nacc, nwit, warps)
        tag = f"M={block_m} N={block_n} NACC={nacc} NWIT={nwit} warps={warps}"
        try:
            compiled = compile_only(args, use_wmma=True)
        except Exception as exc:  # noqa: BLE001
            print(f"   {tag}: compile failed ({type(exc).__name__}: {exc})", flush=True)
            continue
        info = analyze_amdgcn(compiled.asm["amdgcn"])
        n_regs, n_spills, _ = _regs(compiled)
        haz = hazard_present(info)
        kinds: Dict[str, int] = {}
        for s in info["spanning"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        n_wi = kinds.get("work-item-derived", 0)
        print(f"   {tag}: n_regs={n_regs} spills={n_spills} "
              f"wmma={info['n_wmma']} set_msb={info['n_set_vgpr_msb']} "
              f"spanning={len(info['spanning'])} "
              f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items())) or '-'}) -> "
              f"{'HAZARD PRESENT' if haz else 'no hazard'}", flush=True)
        if haz:
            # Rank by work-item-derived carriers first: those are the direct
            # analogue of #3's victim, and spills second, since the real kernel
            # spills 29-33 and a 257-spill config is a long way from it.
            found.append((tag, n_regs, n_spills, len(info["spanning"]), n_wi))

    print("", flush=True)
    if not found:
        print("!! no configuration in the sweep produced an extended-VGPR value "
              "live across the WMMA region.", flush=True)
        print("   That is a real result, not a harness failure: it means this "
              "Triton/LLVM revision does not build the #3 allocation from a "
              "synthetic kernel, and the fallback is to replay the real "
              "_hstu_attn_bwd LLVM IR at BLOCK_N=128 with the faulting store "
              "replaced by a data write.", flush=True)
        return 1
    found.sort(key=lambda r: (-r[4], r[2]))
    print(f"-- {len(found)} configuration(s) show the hazard, best first "
          f"(most work-item-derived carriers, then fewest spills):", flush=True)
    for tag, n_regs, n_spills, n_span, n_wi in found:
        print(f"     {tag}  (n_regs={n_regs} spills={n_spills}, {n_span} spanning, "
              f"{n_wi} work-item-derived)", flush=True)
    return 0


def _print_env() -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} | hip {torch.version.hip}", flush=True)
    print(f"triton {triton.__version__} | arch {props.gcnArchName} "
          f"warp_size={props.warp_size} wgps={props.multi_processor_count}", flush=True)
    print(f"AMDGCN_USE_BUFFER_OPS={os.environ.get('AMDGCN_USE_BUFFER_OPS', '<unset>')} "
          f"AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL', '<unset>')}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true",
                    help="report the emitted allocation and exit; never dispatches")
    ap.add_argument("--sweep", action="store_true",
                    help="compile-only search for a config that produces the hazard")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--block-m", type=int, default=DEF_BLOCK_M)
    ap.add_argument("--block-n", type=int, default=DEF_BLOCK_N)
    ap.add_argument("--block-k", type=int, default=DEF_BLOCK_K)
    ap.add_argument("--nacc", type=int, default=DEF_NACC, choices=(1, 2, 3, 4),
                    help="simultaneously-live accumulators; the pressure knob")
    ap.add_argument("--ndot", type=int, default=DEF_NDOT,
                    help="loop trip count; sets how many WMMAs and mode changes "
                         "the witness has to survive")
    ap.add_argument("--nwit", type=int, default=DEF_NWIT,
                    help="independent work-item-derived values held live across "
                         "the WMMA chain; must be a power of two. Too few and the "
                         "allocator keeps them all below v255, where the hazard "
                         "cannot occur -- check with --compile-only")
    ap.add_argument("--warps", type=int, default=DEF_WARPS)
    ap.add_argument("--stages", type=int, default=1,
                    help="1 matches the [4] clamp the working stack runs with")
    ap.add_argument("--occupancy", type=int, default=1,
                    help="blocks per WGP. The axis no existing [3] probe varies: "
                         "the failing e2e run at batch 128 has ~16x the "
                         "workgroups resident that a single-op probe does")
    ap.add_argument("--streams", type=int, default=1,
                    help="concurrent dispatches, to overlap WMMA co-execution windows")
    ap.add_argument("--remat", action="store_true",
                    help="recompute the witness after the last dot (issue #3 fix 1); "
                         "verify it survived CSE with --compile-only")
    ap.add_argument("--no-wmma", action="store_true",
                    help="same pressure and trip count, no tl.dot; isolates WMMA")
    ap.add_argument("--show-asm", action="store_true",
                    help="print the AMDGCN around each spanning def/use")
    ap.add_argument("--max-report", type=int, default=8, metavar="N",
                    help="show at most N spanning extended VGPRs per kind "
                         "(a 1024-VGPR kernel has hundreds)")
    ap.add_argument("--dump-llir", metavar="PATH",
                    help="write the LLVM IR for the -mattr=-1024-addressable-vgprs rebuild")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop at the first corrupted step; measure a rate")
    ap.add_argument("--allow-vacuous", action="store_true",
                    help="run even when the codegen shows no hazard. Needed for "
                         "the fix controls, where a hazard-free binary is the "
                         "POINT: e.g. TRITON_AMD_TARGET_FEATURES="
                         "-1024-addressable-vgprs (issue #3 fix 2), which drops "
                         "s_set_vgpr_msb to 0. Without this the script refuses "
                         "to run and the control cannot be measured.")
    args = ap.parse_args()

    if args.nwit < 1 or (args.nwit & (args.nwit - 1)):
        raise SystemExit(f"--nwit must be a power of two (tl.arange requires it), "
                         f"got {args.nwit}")

    if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
        raise RuntimeError("set AMDGCN_USE_BUFFER_OPS=0 before running this repro "
                           "(buffer ops are the separate [1] hang)")

    _print_env()

    if args.sweep:
        return sweep(args)

    use_wmma = not args.no_wmma
    compiled = compile_only(args, use_wmma=use_wmma)
    label = "wmma" if use_wmma else "no-wmma"
    if args.remat:
        label += "+remat"
    info = report_codegen(args, compiled, label)

    if args.dump_llir:
        dump_llir(compiled, args.dump_llir)

    if use_wmma and not hazard_present(info):
        print("\n!! this configuration does NOT contain the hazard: no "
              "extended-VGPR value is live across the WMMA region.", flush=True)
        if not args.allow_vacuous:
            print("   Running it would produce a meaningless PASS. "
                  "Use --sweep to find a configuration that does, or "
                  "--allow-vacuous if a hazard-free binary is the point "
                  "(the fix controls).", flush=True)
            return 1
        print("   --allow-vacuous given: running anyway as a CONTROL. A PASS "
              "here is evidence only in contrast with a hazard-present FAIL "
              "under identical launch parameters.", flush=True)

    if args.compile_only:
        return 0

    return run(args, use_wmma, info)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
