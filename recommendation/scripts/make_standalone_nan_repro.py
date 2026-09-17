#!/usr/bin/env python3
"""Emit a self-contained gfx1250 nan/wild-store reproducer for hand-off.

The point of this bundle is that someone who has never seen this repo can run
it. The e2e that actually goes nan needs the yambda-5b dataset, TorchRec, FBGEMM,
the MLPerf harness and a 140 GiB embedding table; none of that is required to
drive the HSTU Triton kernels, which is where the defect is believed to live.

So this walks the import closure of the HSTU forward/backward path and copies
exactly those files, plus a driver and a README, into one directory whose only
third-party dependencies are torch and triton.

Two deliberate choices:

  * Files are vendored BYTE-FOR-BYTE, not rewritten. A receiving team's first
    question is always "is this really the code that failed?", and a plain
    `diff` against the upstream tree has to answer yes. Rewriting imports to
    flatten the package would make the bundle prettier and unverifiable.
  * The PyTorch reference path (`ops/pytorch/pt_*.py`) IS included, because
    `ops/layer_norm.py` imports it at module scope and the bundle will not
    import without it. It does not drag in an FBGEMM dependency: its
    `torch.ops.fbgemm.*` calls resolve lazily at call time, and the Triton path
    under test never calls them. So FBGEMM is needed only by someone who
    deliberately runs the fp32 reference oracle.

Usage:  python scripts/make_standalone_nan_repro.py [--out DIR]
Then:   cd DIR && python repro.py --help
"""
import argparse, os, re, shutil, sys

SEEDS = ["generative_recommenders/ops/hstu_compute.py"]
# Nothing is excluded. Dropping the fp32 reference path looks tempting -- it is
# the only caller of torch.ops.fbgemm -- but ops/layer_norm.py imports it at
# module scope, so the bundle fails to import without it. Since those calls
# resolve lazily and the Triton path never reaches them, keeping the files costs
# nothing at runtime.
EXCLUDE = re.compile(r"(?!)")  # matches nothing


def closure(repo):
    seen, queue = set(), list(SEEDS)
    while queue:
        rel = queue.pop()
        if rel in seen or EXCLUDE.search(rel):
            continue
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            continue
        seen.add(rel)
        for m in re.finditer(r"from\s+(generative_recommenders[\w.]*)\s+import",
                             open(path).read()):
            cand = m.group(1).replace(".", "/") + ".py"
            if not os.path.exists(os.path.join(repo, cand)):
                cand = m.group(1).replace(".", "/") + "/__init__.py"
            if cand not in seen:
                queue.append(cand)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="standalone_gfx1250_nan")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.abspath(a.out)

    files = closure(repo)
    if os.path.isdir(out):
        shutil.rmtree(out)
    total = 0
    for rel in files:
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(repo, rel), dst)
        total += os.path.getsize(dst)

    # Package __init__ files, so the vendored tree imports without the repo.
    pkgs = set()
    for rel in files:
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            pkgs.add("/".join(parts[:i]))
    for p in sorted(pkgs):
        init = os.path.join(out, p, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()

    # The driver and the trap travel with it.
    for src in ("scripts/repro_gfx1250_wildstore_trap.py",):
        shutil.copy2(os.path.join(repo, src), os.path.join(out, "repro.py"))

    open(os.path.join(out, "README.md"), "w").write(README)
    print(f"wrote {len(files)} vendored files + driver to {out}")
    print(f"  vendored payload: {total / 1024:.0f} KiB")
    for rel in files:
        print(f"    {rel}")
    print("\nverify with:")
    print(f"  cd {out} && python repro.py --rows 200000 --batch 256 "
          f"--iters 2 --arena-gib 8 --self-test")
    return 0


README = """# gfx1250 / MI450 silent `nan` + wild-store reproducer

Self-contained. Requires **torch + triton and a gfx1250 GPU** — no dataset, no
TorchRec, no FBGEMM, no MLPerf harness.

## The bug

MLPerf HSTU ranker training at batch 1024 produces `train_loss=nan` partway
through a 3000-step run, and never recovers:

| board | first `nan` | last finite loss | fault? | traceback? | dmesg? | throughput across the transition |
|---|---|---|---|---|---|---|
| MI450 **B0** | step **1370** | — | none | none | clean | flat |
| MI450 **A0** | step **1340** | 0.13629 @ 1330 | none | none | clean | flat (308.8 -> 312.3 -> 320.9 sps) |

Two different boards, 30 steps apart, on the same ROCm 7.14 / Triton
`7ff97e3109` / torch `2.11.0+rocm7.14.0a20260625` stack. **It is not a board
defect.**

Training settings were audited and do not explain it:

* learning rate at the failure is **~1.3e-8** (step 1340 of a 24000-step
  warmup toward a 1e-6 peak) — far too small to diverge
* the loss is **flat** for all 133 logged steps before the failure
  (0.13859 -> 0.13629) and then goes to `nan` within one 10-step logging
  interval. A hard discontinuity, not a divergence
* the loss is `binary_cross_entropy_with_logits` (no `log(0)`), and its
  normaliser is `mt_weights.sum(-1).clamp(min=1.0)` (no `0/0`)
* the local fp16 a2a quantiser patch was verified bit-identical to upstream
  across `inf`/`nan`/overflow inputs

## Why it is believed to be a wild store

Prior work on this stack found and fixed a defect of exactly this class: an
int32 address wrap in `triton_layer_norm.py`, where `rows * stride` overflowed
past 2**31 and went negative while the bounds check `rows < N` was computed on
the unwrapped row id — so the bound and the address came from different values.
The store landed ~4.29 GB (2**32 bytes) from the tensor base. Fix: widen once
with `rows_i64 = rows.to(tl.int64)`.

The decisive asymmetry: **a wild store in a small probe hits unmapped memory
and faults loudly; the same store in the ~150 GB training process lands in a
live allocation and poisons it silently.** That is why the e2e shows `nan` with
no fault record, and why a naive small reproducer shows nothing.

Known trigger conditions from prior bisection:

1. the **backward** pass must be present
2. forward and backward must **overlap** — inserting one
   `torch.cuda.synchronize()` between them suppressed it over 1500 repeats
3. it is rare: roughly **1 event per 300 iterations** at production shape

## Running it

```bash
# REQUIRED. AMDGCN_USE_BUFFER_OPS=1 wedges the node and costs a reboot.
export AMDGCN_USE_BUFFER_OPS=0
# REQUIRED. expandable_segments:True caps the allocator at 19 GiB of 432,
# which makes the canary arena — the whole mechanism — impossible to build.
unset PYTORCH_CUDA_ALLOC_CONF

# prove the harness works (seconds)
python repro.py --rows 200000 --batch 256 --iters 2 --arena-gib 8 --self-test

# primary arm, production depth
python repro.py --rows 2800000 --batch 1024 --layers 3 --iters 600 --arena-gib 120

# the control: same thing with fwd/bwd serialised. Expect NO trap.
python repro.py --rows 2800000 --batch 1024 --layers 3 --iters 600 --arena-gib 120 --sync-between
```

The GPU must be **otherwise idle**: concurrent GPU work on this host has been
measured corrupting unrelated calls at up to 35 % of calls, which makes any
result taken alongside another job meaningless.

## What it reports

The reproducer fills memory around the workload with a known sentinel, so a
wild store lands in a canary instead of in unmapped memory. On a hit it prints
the absolute address, the value written, and the signed delta to every tracked
tensor in both bytes and elements, with an automatic int32-wrap verdict.

* **trap with delta ~= 2**32 bytes or 2**31 elements x itemsize** — another
  int32 address wrap; the delta names the tensor and the stride
* **trap with a small or unstructured delta** — an indexing/codegen bug; the
  written value identifies what stored it
* **delta varies run to run** — a genuine wild store rather than fixed
  arithmetic
* **a fault instead of a trap** — raise `--arena-gib`
* **clean** — only meaningful at `--iters` well above 300

## Status / open

A 600-iteration run at 2.8M rows, single layer, produced no trap and no fault.
A run at production depth (`--layers 3`) is the next arm, because overlap is a
precondition and one layer generates less concurrent kernel activity than three.

A static audit of int32 wrap candidates in the vendored kernels found the
unbounded index variables (`seq_start` in the attention and jagged kernels)
already explicitly widened with `.to(tl.int64)`; the remaining unwidened
stride multiplies use block-local indices bounded by `BLOCK_N` or
`max_seq_len <= 4096`, which cannot reach 2**31. So a third wrap of the [b3]
shape is **not** well supported, and the overlap-dependent wild-store class is
the stronger hypothesis.
"""

if __name__ == "__main__":
    sys.exit(main())
