#!/usr/bin/env python3
"""Catch the gfx1250 wild store in the act and RECOVER ITS ADDRESS.

WHY THIS EXISTS
---------------
mi450_b0.md section 3 establishes what the defect IS -- a wild store, needing the
backward pass and forward/backward overlap -- but not WHERE it comes from. The
two existing reproducers cannot answer that:

  repro_gfx1250_nondeterminism.py   detects that a fault/divergence happened
  repro_gfx1250_hstu_layer.py       detects that a canary was clobbered, and
                                    reports a COUNT of bad words

A count is not attribution. The one defect on this host that DID get root-caused,
[b3] in mi450_b0.md section 5, was cracked by a single number: the faulting
address was 8.004 GiB below `Y`, against `2**31 x 4 B = 8 GiB`, which named the
bug class outright. That number came from a hardware fault record.

The e2e never produces one. Per mi450_b0.md's central idea, a wild store in the
~150 GB training process lands in a LIVE ALLOCATION and poisons it silently,
where the same store in a small probe hits unmapped memory and faults loudly. So
the loud case gives an address we can use but is the wrong regime, and the silent
case is the right regime but gives us nothing.

This script makes the silent case yield the address anyway. It deliberately
reproduces the e2e's large-footprint condition with a canary ARENA of known
bytes bracketing the workload, then, when a store lands in it, reports:

  * the absolute virtual address that was written
  * the VALUE written -- often the strongest clue of all, because a float
    gradient, an int index and a pointer look nothing alike
  * the signed delta to every workload tensor, in bytes AND in elements
  * an automatic int32-wrap test: is |delta| within a hair of 2**31 elements
    for any plausible stride? That is exactly [b3]'s and a0 [3]'s signature

WHAT EACH OUTCOME MEANS
-----------------------
  TRAPPED, delta ~= 2**31 * itemsize    a third int32 wrap. mi450_b0.md section
                                        2 reading (1), and section 3.3 rank 3 --
                                        the 24 unaudited `<var>[:, None] *
                                        stride_*` sites become the search space.
  TRAPPED, delta small / unstructured   not a wrap. An indexing or codegen bug;
                                        the written VALUE localises the source.
  TRAPPED, delta varies run to run      the wild-store/allocator-accident class,
                                        mi450_b0.md section 2 reading (2).
  FAULT instead of a trap               arena too small to cover the store's
                                        target -- raise --arena-gib.
  Clean over many iters                 underpowered unless iters >> 300; see
                                        the rate note in mi450_b0.md section 3.

CONTROL, AND WHY IT MATTERS
---------------------------
`--sync-between` is mi450_b0.md section 3.1 arm 8, the one intervention proven to
suppress this defect (1500 repeats clean). Run it both ways. Traps under the
plain arm and none under `--sync-between` is what tells you the trap is catching
THIS defect and not some unrelated bug in the harness. A trap that survives
`--sync-between` is a DIFFERENT finding and must be reported as one.

MANDATORY ENVIRONMENT
---------------------
  AMDGCN_USE_BUFFER_OPS=0     `1` wedges the whole node and costs a reboot.
  PYTORCH_CUDA_ALLOC_CONF=    empty. `expandable_segments:True` caps the caching
                              allocator at 19 GiB of 432, so the arena -- the
                              entire point of this script -- cannot be built.
  An otherwise IDLE GPU.      mi450_b0.md section 3.6: concurrent GPU work
                              corrupts unrelated work on this host at up to 35 %
                              of calls. A trap taken next to another job is
                              worthless.

EXAMPLES
--------
  # primary arm, ~0.75 s/iter, expect a trap or fault within a few hundred iters
  python scripts/repro_gfx1250_wildstore_trap.py --rows 2800000 --iters 600

  # the control: same thing with fwd/bwd serialised. Expect NO trap.
  python scripts/repro_gfx1250_wildstore_trap.py --rows 2800000 --iters 600 --sync-between

  # cheap smoke test of the trap machinery itself (seconds, no defect needed)
  python scripts/repro_gfx1250_wildstore_trap.py --rows 200000 --iters 2 --self-test
"""
import argparse, os, sys, time
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

# Sentinel is checked as int32 so the comparison is exact and NaN-free. The value
# is arbitrary but must be recognisable by eye in a hex dump and must not be a
# plausible float or small integer, or a real store of a benign value could be
# mistaken for an untouched word.
SENTINEL = 0x5AFEC0DE


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_800_000,
                    help="total jagged rows. 2.8M is the production-shape arm "
                         "(max_seq_len 4096) that faults at ~1 per 300 iters")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=600,
                    help="fwd+bwd iterations. Under ~300 a clean result means "
                         "nothing -- see mi450_b0.md section 3")
    ap.add_argument("--arena-gib", type=float, default=0.0,
                    help="canary arena size. 0 (default) = auto: fill most of "
                         "what is free after the workload is built")
    ap.add_argument("--scan-every", type=int, default=1,
                    help="scan the arena every N iters. The scan is a full "
                         "memory sweep, so raising this trades detection "
                         "latency for throughput")
    ap.add_argument("--sync-between", action="store_true",
                    help="mi450_b0.md section 3.1 arm 8: one synchronize() "
                         "between fwd and bwd. THE CONTROL. Expect no trap")
    ap.add_argument("--layers", type=int, default=1,
                    help="stack N HSTU blocks. PRODUCTION DEPTH IS 3 "
                         "(HSTU_NUM_LAYERS=3). Depth is not cosmetic here: "
                         "mi450_b0.md section 3.2 conclusion 3 makes fwd/bwd "
                         "OVERLAP a precondition, and one layer generates less "
                         "concurrent kernel activity than three, so --layers 1 "
                         "may under-trigger")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--self-test", action="store_true",
                    help="deliberately clobber one arena word before the loop, "
                         "to prove the trap and its arithmetic work. Exits "
                         "non-zero on a real trap only")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop at the first trap; collect more samples. "
                         "Several deltas from one run is much stronger evidence "
                         "than one, since it shows whether the delta is stable")
    return ap.parse_args()


class Trap:
    """A canary arena that brackets the workload, plus the address arithmetic."""

    def __init__(self, dev):
        self.dev = dev
        self.blocks = []        # canary tensors
        self.tracked = {}       # name -> (base_va, nbytes, itemsize)
        self.hits = 0

    def add_block(self, gib):
        """Allocate one canary block. Returns False when the GPU says no more."""
        n = int(gib * (1 << 30)) // 4
        if n <= 0:
            return False
        try:
            t = torch.full((n,), SENTINEL, device=self.dev, dtype=torch.int32)
        except torch.cuda.OutOfMemoryError:
            return False
        self.blocks.append(t)
        return True

    def track(self, name, t):
        """Register a workload tensor so deltas can be measured against it."""
        if t is not None and t.is_cuda:
            self.tracked[name] = (t.data_ptr(), t.numel() * t.element_size(),
                                  t.element_size())

    def arena_bytes(self):
        return sum(b.numel() * 4 for b in self.blocks)

    def scan(self):
        """Return a list of (block_index, flat_index) for clobbered words.

        The common path is a full-arena reduction with no host transfer and no
        allocation: only when something is actually wrong do we pay to locate
        it. Scanning word-by-word on the host would be slower than the kernel
        under test and would hide the very signal we are after.
        """
        out = []
        for bi, b in enumerate(self.blocks):
            if bool(torch.any(b != SENTINEL)):
                idx = torch.nonzero(b != SENTINEL).flatten()
                out.extend((bi, int(i)) for i in idx[:64].tolist())
        return out

    def report(self, hits, iteration):
        """Turn raw hits into the address arithmetic that names the bug class."""
        print(f"\n{'=' * 78}\nWILD STORE TRAPPED at iteration {iteration} "
              f"-- {len(hits)} clobbered word(s) shown\n{'=' * 78}", flush=True)
        for bi, i in hits[:16]:
            blk = self.blocks[bi]
            va = blk.data_ptr() + i * 4
            raw = int(blk[i].item()) & 0xFFFFFFFF
            as_f32 = torch.tensor([raw], dtype=torch.int32).view(torch.float32).item()
            print(f"\n  address   {va:#018x}   (arena block {bi}, word {i:,})")
            print(f"  value     {raw:#010x}   as f32 {as_f32:.6e}   as i32 "
                  f"{raw - (1 << 32) if raw >> 31 else raw}")
            print(f"  {'tensor':<12} {'delta (bytes)':>18} {'delta (elements)':>20}"
                  f"   int32-wrap?")
            for name, (base, nbytes, isz) in sorted(self.tracked.items()):
                d = va - base
                # "inside" is not a wild store at all -- it means the arena and a
                # live tensor overlap, which would be an allocator bug in this
                # script rather than a finding. Say so explicitly.
                if 0 <= d < nbytes:
                    print(f"  {name:<12} {d:>18,} {'INSIDE THIS TENSOR':>20}"
                          f"   <== harness bug, not a defect")
                    continue
                elems = d / isz
                print(f"  {name:<12} {d:>18,} {elems:>20,.1f}   "
                      f"{self._wrap_verdict(d, isz)}")
        print(f"\n{'=' * 78}", flush=True)

    @staticmethod
    def _wrap_verdict(delta, itemsize):
        """Is this delta an int32 overflow of some pointer arithmetic?

        There is more than one magnitude to look for, and testing only one is
        how you miss the bug. Both known cases on this host are int32 wraps, yet
        they displace by DIFFERENT amounts:

          [b3] triton_hstu_linear.py   "overflows at row 1_398_102 and the store
                                       lands ~4.29 GB below Y"  -> 2**32 BYTES
          2**31-element form           product wraps sign at 2**31 elements and
                                       is then scaled by itemsize -> 2**31*isz

        Which one you get depends on whether the wrapped product is consumed as
        an element index or as a byte offset, and on whether it wrapped at the
        sign bit or all the way around. So test the whole family and report
        which member matched -- the member itself narrows the search.
        """
        # `2**31 bytes` (2 GiB) is deliberately NOT in this list. It is not a
        # real wrap displacement -- int32 overflow moves you 2**32 BYTES (the
        # [b3] form) or 2**31/2**32 ELEMENTS scaled by itemsize -- and 2 GiB is
        # such a commonplace distance inside a multi-GiB arena that including it
        # made the self-test's single planted clobber read as a wrap against
        # nearly every tracked tensor at once. A candidate set must stay sparse
        # relative to the match band or the verdict means nothing.
        cands = [
            (1 << 32,              "2**32 B (b3 form, ~4.29 GB)"),
            ((1 << 31) * itemsize, f"2**31 elems x {itemsize}B"),
            ((1 << 32) * itemsize, f"2**32 elems x {itemsize}B"),
        ]
        best = min(cands, key=lambda c: abs(abs(delta) - c[0]))
        err = abs(abs(delta) - best[0])
        # 16 MiB band: the wrap lands 2**31/2**32 from the base of the INDEXED
        # ROW, not of the tensor, so the row's own offset appears as slack. A
        # band tight enough to exclude that would have missed [b3] itself.
        if err <= (1 << 24):
            return f"YES {best[1]} -- |d|={abs(delta) / 2**30:.3f} GiB"
        if err <= (1 << 28):
            return f"near {best[1]} ({abs(delta) / 2**30:.3f} GiB)"
        return "no"


def build_jagged(rows, batch, dev, g):
    """Jagged lengths summing exactly to `rows`, as the other reproducers do."""
    wt = torch.rand(batch, device=dev, generator=g) + 0.1
    lengths = (wt / wt.sum() * rows).long().clamp(1, MAX_SEQ_LEN)
    while int(lengths.sum()) != rows:
        d = rows - int(lengths.sum())
        room = (MAX_SEQ_LEN - lengths) if d > 0 else (lengths - 1)
        i = torch.nonzero(room > 0).flatten()[0]
        take = min(abs(d), int(room[i]))
        lengths[i] += take if d > 0 else -take
    offsets = torch.cat([torch.zeros(1, device=dev, dtype=torch.long),
                         lengths.cumsum(0)]).long()
    return offsets, int(lengths.max())


def main():
    a = parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    props = torch.cuda.get_device_properties(0)

    if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
        print("!! AMDGCN_USE_BUFFER_OPS is not 0 -- this wedges the node. Refusing.",
              file=sys.stderr)
        return 2
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        print(f"!! PYTORCH_CUDA_ALLOC_CONF="
              f"{os.environ['PYTORCH_CUDA_ALLOC_CONF']!r} caps the allocator at "
              f"19 GiB on this stack; the arena cannot be built. Unset it.",
              file=sys.stderr)
        return 2

    print(f"# {props.gcnArchName} torch={torch.__version__} pid={os.getpid()}")
    print(f"# rows={a.rows:,} batch={a.batch} iters={a.iters} "
          f"sync_between={a.sync_between} seed={a.seed}")

    trap = Trap(dev)
    g = torch.Generator(device=dev).manual_seed(a.seed)

    # Half the arena BEFORE the workload and half after, so the canaries bracket
    # it in the address space. [b3]'s bad address was 8 GiB BELOW its tensor, so
    # an arena built only above the workload would have missed it entirely.
    pre_gib = (a.arena_gib / 2) if a.arena_gib else 32.0
    while pre_gib > 0.5 and not trap.add_block(min(pre_gib, 16.0)):
        pre_gib /= 2
    built_pre = trap.arena_bytes()
    if a.arena_gib:
        remaining = a.arena_gib - built_pre / 2**30
        while remaining > 0.5 and trap.add_block(min(remaining, 16.0)):
            remaining -= 16.0

    def make_layer():
        return {"in_w": torch.ones(D, device=dev), "in_b": torch.zeros(D, device=dev),
                "uvqk_w": torch.randn(D, 2 * NUM_HEADS * (HIDDEN_DIM + ATTN_DIM),
                                      device=dev, generator=g) * 0.02,
                "uvqk_b": torch.zeros(2 * NUM_HEADS * (HIDDEN_DIM + ATTN_DIM),
                                      device=dev),
                "out_w": torch.ones(NUM_HEADS * HIDDEN_DIM, device=dev),
                "out_b": torch.zeros(NUM_HEADS * HIDDEN_DIM, device=dev),
                "out_proj": torch.randn(NUM_HEADS * HIDDEN_DIM + D, D,
                                        device=dev, generator=g) * 0.02}

    layers = [make_layer() for _ in range(a.layers)]
    w = {f"L{i}.{k}": v for i, lw in enumerate(layers) for k, v in lw.items()}
    seq_offsets, msl = build_jagged(a.rows, a.batch, dev, g)
    x = torch.randn(a.rows, D, device=dev, generator=g).requires_grad_(True)
    for v in w.values():
        v.requires_grad_(True)

    trap.track("x", x)
    trap.track("seq_offsets", seq_offsets)
    for k, v in w.items():
        trap.track(k, v)

    # Post-workload half of the arena: take whatever is left, since the bigger
    # the mapped footprint the more faithfully this mimics the ~150 GB e2e in
    # which the store lands silently instead of faulting.
    if not a.arena_gib:
        free, _ = torch.cuda.mem_get_info()
        want = free / 2**30 - 24.0        # headroom for uvqk and the backward
        while want > 0.5 and trap.add_block(min(want, 16.0)):
            want -= 16.0

    print(f"# max_seq_len={msl}  arena={trap.arena_bytes() / 2**30:.1f} GiB in "
          f"{len(trap.blocks)} blocks  tracked={len(trap.tracked)} tensors")
    free, total = torch.cuda.mem_get_info()
    print(f"# device memory: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
    if trap.arena_bytes() < (8 << 30):
        print("# WARNING: arena < 8 GiB. [b3]'s store landed 8 GiB from its "
              "tensor, so a wrap of that shape may fall outside the arena and "
              "fault instead of being trapped.")

    if a.self_test:
        # Prove the machinery end to end: clobber a word ourselves, at a chosen
        # offset, and confirm the scan finds it and the arithmetic prints.
        blk = trap.blocks[-1]
        victim = blk.numel() // 2
        blk[victim] = 0x0BADF00D
        hits = trap.scan()
        assert hits, "SELF-TEST FAILED: scan did not see a known clobber"
        trap.report(hits, iteration=-1)
        blk[victim] = SENTINEL
        assert not trap.scan(), "SELF-TEST FAILED: arena did not reset"
        print("SELF-TEST PASS: trap detects, locates and resets.\n")

    def step():
        h = x
        for lw in layers:
            h = one_layer(h, lw)
        return h

    def one_layer(inp, w):
        attn, u, _, _ = hstu_preprocess_and_attention(
            x=inp, norm_weight=w["in_w"], norm_bias=w["in_b"], norm_eps=NORM_EPS,
            num_heads=NUM_HEADS, attn_dim=ATTN_DIM, hidden_dim=HIDDEN_DIM,
            uvqk_weight=w["uvqk_w"], uvqk_bias=w["uvqk_b"], max_seq_len=msl,
            seq_offsets=seq_offsets, attn_alpha=1.0 / (ATTN_DIM ** 0.5),
            causal=True, num_targets=None, max_attn_len=0, contextual_seq_len=0,
            recompute_uvqk_in_backward=False, recompute_normed_x_in_backward=False,
            sort_by_length=False, kernel=HammerKernel.TRITON)
        return hstu_compute_output(
            attn=attn, u=u, x=inp, norm_weight=w["out_w"], norm_bias=w["out_b"],
            norm_eps=NORM_EPS, output_weight=w["out_proj"], num_heads=NUM_HEADS,
            linear_dim=HIDDEN_DIM, dropout_ratio=0.0, training=False,
            concat_u=False, concat_x=True, mul_u_activation_type="silu",
            group_norm=False, recompute_y_in_backward=False,
            kernel=HammerKernel.TRITON)

    t0, trapped = time.time(), 0
    for it in range(a.iters):
        x.grad = None
        for v in w.values():
            v.grad = None
        y = step()
        if a.sync_between:
            torch.cuda.synchronize()
        y.sum().backward()
        if a.sync_between:
            torch.cuda.synchronize()

        # Gradients are fresh allocations each iteration, so re-register them:
        # a store landing near a gradient is just as diagnostic as one near a
        # forward tensor, and their addresses move.
        trap.track("dx", x.grad)
        for k, v in w.items():
            trap.track("d" + k, v.grad)

        if (it + 1) % a.scan_every == 0:
            hits = trap.scan()
            if hits:
                trapped += 1
                trap.report(hits, it)
                if not a.keep_going:
                    break
                for bi in {b for b, _ in hits}:
                    trap.blocks[bi].fill_(SENTINEL)  # re-arm for the next trap
        if (it + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  iter {it + 1}/{a.iters}  {el:.0f}s  "
                  f"{el / (it + 1):.2f}s/iter  traps={trapped}", flush=True)

    el = time.time() - t0
    print(f"\n# elapsed {el:.0f}s for {a.iters} iters ({el / max(a.iters, 1):.2f}s/iter)")
    if trapped:
        print(f"RESULT: TRAPPED {trapped}x -- a wild store landed in the arena. "
              f"The deltas above are the attribution; read them against "
              f"mi450_b0.md section 3.3.")
        return 1
    print(f"RESULT: no trap in {a.iters} iters."
          + ("" if a.iters >= 300 else
             "  UNDERPOWERED: the rate is ~1 per 300 iters, so this proves "
             "nothing. Raise --iters."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
