# Batch-128 `nan` ([3]): the uninitialized-output hypothesis — REFUTED

> **Status: this hypothesis is dead.** A systematic audit of every reachable
> call site (17 agents, adversarial) found the coverage condition violated by
> **0 rows everywhere**, and — decisively — *the coverage condition contains no
> batch term at all*, so it could not produce the B=128 vs B=8 asymmetry even
> if it were violated. Details in "Why it is refuted" below.
>
> `scripts/repro_jagged_concat_uninit.py` remains a correct demonstration of
> the defect *class* and a regression test for the invariant, but it is no
> longer a lead for [3]. `scripts/nan_poison_hook.py` retains independent value
> as a general uninitialized-read detector.
>
> The live instrument for [3] is now `scripts/nan_tripwire.py`.

## What the hypothesis was (kept for the record)

Two artifacts, both validated on the MI450 A0 in `triton-7ff97e-test`
(torch `2.11.0+rocm7.14.0a20260625`, Triton `3.8.0` @ `7ff97e3109`).

Neither is an LLVM handoff. If this hypothesis is what [3] is, the fix is in this
repo. That is a change of target from `docs/gfx1250_extvgpr_handoff.md`, which
covers a genuine codegen finding that does **not** explain the `nan`.

## The hypothesis

`concat_2D_jagged` / `split_2D_jagged` allocate their outputs with
`torch.empty` and fill them with Triton kernels whose launch grid is sized from
`max_seq_len`, while their store mask is `offs_n < seq_len_a + seq_len_b` taken
from the *offsets*. Nothing ties the two together, so the kernel writes every
output row only if

    for all z:   len_a(z) + len_b(z)  <=  ceil(max_seq_len / BLOCK_N) * BLOCK_N

`BLOCK_N` is autotuned over `{1,2,4,8}` (`triton_jagged_tensors.py:168-178`), so
the rounding slack is at most 7 rows and the condition is effectively
`len_a + len_b <= max_seq_len`. Violate it and the tail rows of a **gradient**
are whatever the caching allocator last left there — finite on a lucky run,
NaN/Inf bits on an unlucky one. That profile matches [3] exactly: no producer to
name, nondeterministic across runs, and worse with more rows.

Routing note, because it is easy to aim at the wrong line numbers: the model
calls the ordinary `concat_2D_jagged`/`split_2D_jagged`, and
`_triton_concat_2D_jagged_internal` (`triton_jagged_tensors.py:52-69`) sends
every HIP launch to the `*_multirow` kernels — the basic kernel fails to lower in
`TritonAMDGPUCanonicalizePointers`. The `*_multirow` **autograd wrapper classes**
at `:813`/`:921` are dead code. The live allocations are `:583`, `:621`/`:624`
(a `zeros`/`empty` asymmetry with no stated justification), `:697`, `:700`,
and `:738` (`d_jagged_in`, the gradient).

## Where the evidence stands

**Consistent with [3]:** `HSTU_HAMMER_KERNEL=PYTORCH` was verified to bypass
these ops entirely (`ops/jagged_tensors.py:110` and `:196`, driven model-wide by
`set_hammer_kernel` from `dlrm_v4/train/utils.py:411-417`), matching PYTORCH
finite 3/3. Uncovered rows scale linearly with batch.

**Against it, and this is the stronger half.** A static audit of all eight ranker
call sites — `dlrm_hstu.py:510`, `:726`; `preprocessors.py:288`, `:300`;
`contextual_interleave_preprocessor.py:180`, `:191`; `action_encoder.py:100`;
`content_encoder.py:87`; `hstu_transducer.py:216`, `:233` — found every one
passes `max_seq_len = max_len_left + max_len_right` or larger, which bounds the
condition structurally. And the non-HIP path launches the basic kernel on grid
`(max_seq_len, B)`, which covers *fewer* rows than the multirow grid; a real
violation would break NVIDIA at least as badly, and this code runs there.

So the residual exposure is narrow and data-dependent: it needs `max_len_left`
or `max_len_right` to be below the true runtime maximum of its offsets. Source
cannot settle that. Hence artifact 2.

## Artifact 1 — `scripts/repro_jagged_concat_uninit.py`

`torch` + `triton` only. No repo import, no dataset. Verbatim copy of the kernel;
predicts the uncovered row set on the host and asserts the observed set matches
it row for row, rather than merely looking for a NaN.

```
$ AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_jagged_concat_uninit.py --violate 3 --scan-batch

== control: coverage condition HOLDS (violate=0) ==
   predicted uninitialized rows: 0   observed: 0   match: True
-- PASS: every output row was written

== defect: coverage condition VIOLATED by 3 rows per element ==
   grid covers offs_n < 64, combined len per element = 67
   predicted uninitialized rows: 24   observed: 24   match: True
-- DEFECT CONFIRMED

   batch   uncovered rows
       8       24    ...    128      384
```

`--allocator-demo` drops the sentinel and reproduces the production shape — a
bare `torch.empty` landing on a recycled block — giving `out.sum() = nan`.
`--compile-only` never dispatches.

This proves the mechanism. It does not prove the mechanism fires.

## Artifact 2 — `scripts/nan_poison_hook.py`

The instrument that decides it, in one training run. Fills every uninitialized
CUDA float allocation with `0x7FC0BEEF`, so *reading uninitialized memory becomes
an immediate NaN every time* instead of 3-in-4. Also asserts the coverage
precondition at each live call site (rebinding the `from ... import` bindings in
place, so install order does not matter) and can name the first non-finite
gradient.

Validated end-to-end through the real repo op:

```
max_seq_len=67 (coverage OK):        out rows=134   nonfinite rows=0
max_seq_len=64 (coverage VIOLATED):  out rows=134   nonfinite rows=6
```

Zero false positives on the control; exactly the predicted 6 rows on the
violation.

**The experiment, cheapest first — and it is decisive in both directions:**

```bash
export AMDGCN_USE_BUFFER_OPS=0
NAN_POISON=1 NAN_POISON_COVERAGE=1 BATCH_SIZE=8 <train cmd>
```

- `BATCH_SIZE=8` goes `nan` under poisoning (it is clean today) → an uninit read
  exists and luck has been masking it. Hypothesis confirmed; the coverage line or
  the gradient check names the op.
- `BATCH_SIZE=8` stays finite for a few hundred steps → nothing on this path
  reads uninitialized memory, and this entire hypothesis class dies, artifact 1
  included.

Given the call-site audit, expect the second. Killing the class for one cheap run
is worth it — it is the last hypothesis standing that explains [3]'s
nondeterminism without invoking a miscompile.

**Limitation to state when reporting the result:** the poison is a Python-level
patch of `torch.empty`, so it covers repo-level allocations but not allocations
made inside ATen/C++. A clean run bounds the hypothesis; it does not prove the
process reads no uninitialized memory anywhere.

## Why it is refuted

Four independent findings, any one of which is sufficient:

1. **No call site violates it.** All 7 reachable sites: violation = 0 rows.
2. **The invariant is structural, not accidental.** `max_seq_len` is always
   produced by `fx_infer_max_len()` (`common.py:483-493`, `int(lengths.max())`)
   applied to the *same length tensors the offsets are cumsum'd from*. It
   cannot drift below its own offsets.
3. **The specific bug shape hunted does not occur.** Item `max_seq_len` against
   contextual+item offsets would violate it, but `hstu_transducer.forward`
   (`:277-296`) rebinds `max_seq_len` / `seq_lengths` / `total_uih_len`
   together from `_preprocess`, and `preprocessors.py:314/315/319-322`
   increments all three in lockstep. The stale `max_uih_len`/`max_targets`
   forwarded at `hstu_transducer.py:309-310` are ignored, because both offsets
   are non-None (`triton_jagged_tensors.py:333-346`).
4. **It is batch-independent.** This is the one that ends it. The condition
   `len_a(z) + len_b(z) <= ceil(max_seq_len/BLOCK_N)*BLOCK_N` is per-element.
   `B` appears nowhere in it. A defect that fires 3-in-4 at B=128 and 0-in-3 at
   B=8 cannot be this.

`BLOCK_N` rounding slack can only ever *add* coverage, and `n_prefix_from_B = 0`
at every reachable site, so neither offers an escape hatch.

## What this leaves for [3]

If artifact 2 comes back clean, the surviving hypotheses are the ones already on
record, minus two now refuted this cycle: the extended-VGPR *value* victim
(bit-exact over 200 iterations × 6 repeats) and the `AUTOTUNE_B` config lottery
(the config pool is identical at both batch sizes, so a noisy draw cannot produce
a B=128-only failure). The cheapest untaken measurement remains Track C — re-run
batch 128 with `METRIC_LOG_FREQ=1` to establish the true onset step, since
"nan at step 50" is currently an artifact of the log cadence.
