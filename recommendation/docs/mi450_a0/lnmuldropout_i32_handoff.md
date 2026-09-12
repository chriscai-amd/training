# `_ln_mul_dropout_fwd_rng` truncates its output row offset to int32

**Status:** root-caused, reproduced deterministically on hardware, one-line fix
verified standalone **and confirmed end-to-end** (600 clean steps at batch 1024,
2026-09-10).

**Owner:** this repo (`generative_recommenders/ops/triton/`), *not* the
LLVM/AMDGPU backend team. The search for defect [3] started out aimed at LLVM
because the symptom looked like a miscompile. It is not one — the generated code
faithfully implements what the kernel source asks for, and the kernel source
asks for a 32-bit multiply.

**Reproducer:** [`scripts/repro_gfx1250_lnmuldropout_i32.py`](../../scripts/repro_gfx1250_lnmuldropout_i32.py)
— torch + triton only, no repo import, no fbgemm, no dataset, deterministic,
cannot fault.

---

## What is confirmed

In `generative_recommenders/ops/triton/triton_hstu_linear.py`, kernel
`_ln_mul_dropout_fwd_rng` (defined at :148) indexes its output with a raw int32
row id:

```python
178:    start_row = block_id * BLOCK_N
179:    rows = start_row + tl.arange(0, BLOCK_N)      # int32
180:    row_mask = rows < N
...
267:    tl.store(Y + rows[:, None] * stride_y + cols[None, :], ...)           # int32 multiply
272:    tl.store(Y + rows[:, None] * stride_y + (cols + D)[None, :], ...)     # int32 multiply
277:    tl.store(Y + rows[:, None] * stride_y + (cols + 2 * D)[None, :], ...) # int32 multiply
```

`stride_y` arrives as a plain Python int, so Triton specializes it to `i32`.
`rows * stride_y` is an `i32` multiply that wraps, and Triton sign-extends the
*already-wrapped* product when it forms the pointer.

The same function already knows the widening is required. Three lines above the
stores, the dropout-mask pointer is widened:

```python
231:    row_offsets_i64 = rows.to(tl.int64)
234:    offsets = row_offsets_i64[:, None] * stride_mask + cols[None, :]
```

and so is every neighbouring kernel in the same file — `pid.to(tl.int64)` at
:114, `X/U/Y += row.to(tl.int64) * stride_*` at :336-338, `row_i64` at :490,
`tile_num_i64` at :518. Only the eight `Y` stores at :267, :272, :277, :283,
:288, :294, :299, :305 were missed, along with the `X` and `U` loads at :183 and
:189.

### `row_mask` does not contain the damage

`row_mask = rows < N` is evaluated on the *unwrapped* row id, so it is true
exactly when the row is legitimately in range. The wrapped negative offset is
then written under a passing mask. The bounds check and the address are computed
from different values.

### The wrap point at the production shape

DLRM-v4 HSTU output: `D = num_heads * linear_dim = 4 * 128 = 512`
(`configs.py:137-138`), with `concat_u=concat_x=True` (`stu.py:349-350`), so
`y` is `(N, 3 * D)` and `stride_y = 1536`.

| quantity | value |
|---|---|
| `stride_y` | 1536 |
| last in-range row | 1,398,101 |
| **first wrapping row** | **1,398,102** (`(2**31 - 1) // 1536 + 1`) |
| byte offset of a wrapped store | **Y − 4.29 GB** |
| `stride_x = D` wrap row (loads at :183/:189) | 4,194,304 |

## Input

`N` is the **total number of jagged rows in the batch**, not the batch size:
`N, D = x.shape` at `triton_hstu_linear.py:988`, where `x` is the packed jagged
attention output.

The kernel's autotune key is `key=["BLOCK_D"]` (:146) — **codegen is identical at
every batch size.** Batch size enters only by scaling `N` past 1,398,102. That is
why this presents as a batch-size-gated defect while being a pure row-count bug.

At the production config, the per-sample row count is set by
`HISTORY_STRATEGY`, which the run script leaves at its default `interleaved`
(`yambda_5b.gin:495`). The gin file records a **measured** effective length for
each strategy, so `N` no longer has to be guessed from `HISTORY_LENGTH`:

| strategy | measured events/sample | N @ b=128 | N @ b=1024 | rows wrapping @ 1024 | crossover batch |
|---|---|---|---|---|---|
| `interleaved` (**default, in use**) | ~2663 | 342 K | **2.74 M** | 1.34 M (48.9%) | 523 |
| `last_n` | ~4085 | 524 K | 4.19 M | 2.79 M (66.7%) | 342 |

`HISTORY_LENGTH=4086` is a *per-pool* cap under `interleaved` (nominal
`3 x 1362`), not the achieved length — the like pool fills only ~105 of its 1362
slots and that budget is not reallocated, so the sequence comes up ~35% short of
the nominal 4095. The earlier ~4.19 M figure assumed the nominal fill.

**The verdict is the same under either strategy**, which is what makes it safe to
rely on: batch 128 is below the wrap row with ~4x margin, batch 1024 is above it
by ~2x, and the crossover sits between them either way. This matches the observed
split exactly (128 clean over 4000 steps 2/2; 1024 `nan` within ~200 steps 2/2).

**Caveat, reduced but not eliminated:** these are dataset statistics recorded in
the gin file, not `N` read out of the running kernel. The instrumented run that
would have printed `N` directly wedged the node (see "Not yet done"). The
inference now rests on a measured corpus statistic rather than on a config
ceiling, and is insensitive to the one free parameter, but it is still an
inference.

## Current output

```
$ AMDGCN_USE_BUFFER_OPS=0 python3 scripts/repro_gfx1250_lnmuldropout_i32.py --n 1398101 --canary-gib 0
-- D=512 stride_y=3*D=1536 int32 wrap at row 1398102
-- N=1398101   BLOCK_N=1 num_warps=4 kernel=shipping
   rows not written correctly : 0
   PASS

$ AMDGCN_USE_BUFFER_OPS=0 python3 scripts/repro_gfx1250_lnmuldropout_i32.py --n 1399126 --canary-gib 2
-- N=1399126   BLOCK_N=1 num_warps=4 kernel=shipping
   int32 wrap row = 1398102  ->  1024 of 1399126 rows expected to wrap
   rows not written correctly : 1024
   first bad row              : 1398102
   canary elements clobbered  : 1572864
   !! FAIL
   !! first bad row is exactly the int32 wrap row 1398102 (= (2**31-1)//1536 + 1)
```

Every number is exact, not approximate:

- first bad row `1398102` = the predicted wrap row, to the row;
- bad rows `1024` = exactly the rows above the threshold;
- canary elements clobbered `1572864` = `1024 rows x 1536 elements` — **every
  single wrapped store landed inside an unrelated live tensor**, and none of them
  faulted.

## Desired output

`PASS` at every `N`. Verified with `--fix`:

```
$ AMDGCN_USE_BUFFER_OPS=0 python3 scripts/repro_gfx1250_lnmuldropout_i32.py --n 1399126 --canary-gib 2 --fix
-- N=1399126   BLOCK_N=1 num_warps=4 kernel=fixed
   rows not written correctly : 0
   canary elements clobbered  : 0
   PASS
```

## Fix recommendation

Widen once, next to the widening that is already there, and use it for the loads
and all stores:

```python
rows = start_row + tl.arange(0, BLOCK_N)
row_mask = rows < N
rows_i64 = rows.to(tl.int64)          # add
...
x_block = tl.load(X + rows_i64[:, None] * stride_x + cols[None, :], ...)
u_block = tl.load(U + rows_i64[:, None] * stride_u + cols[None, :], ...)
...
tl.store(Y + rows_i64[:, None] * stride_y + cols[None, :], ...)
```

`Mean + rows` and `Rstd + rows` are fine as-is (element strides of 1; they cannot
reach 2**31 before `N` does).

The backward kernels that consume the same `Y` layout were audited for the
identical pattern and are clear — see "Not yet done" item 3.

## Why it is silent instead of a page fault

`y` is a fresh `torch.empty` allocated late in a process whose caching allocator
already holds the ~140 GiB embedding table. `Y − 4.29 GB` therefore lands inside
other live allocations rather than in unmapped memory. No fault, no traceback —
just other tensors quietly overwritten with bf16 activations, which is the
recorded signature of defect [3] (`train_loss=nan` reached through poisoned
embedding tables, no fault). The canary in the reproducer reproduces exactly this:
1,572,864 wild writes, zero faults.

It also explains the nondeterministic onset step (91 vs ~185 across two runs with
an identical `SEED=1` data stream): *which* tensor sits 4.29 GB below `Y` is an
allocator-layout accident that differs per process, so how fast the corruption
reaches the loss differs per process, while the corruption itself is
deterministic.

## Why this was never seen on other hardware

`_ln_mul_dropout_fwd_rng` is not the default path. `hstu_compute.py` reaches it
only when `use_separated_rng_ln_mul_dropout()` is true
(`generative_recommenders/ops/utils.py:100-165`):

```python
return (is_sm100_plus() or is_amd_mi350() or _fused_rng_ln_mul_dropout_is_broken())
```

`_fused_rng_ln_mul_dropout_is_broken()` returns true for **gfx1250 on Triton >=
3.8** — a correctness workaround for defect [4c]. Everything else (MI300, A100,
H100) takes the legacy `_ln_mul_dropout_fwd`, whose row offsets **are** widened
(`Y += row.to(tl.int64) * stride_y`, :338). So the bug is arch-correlated without
being an arch bug: gfx1250 + Triton 3.8 is simply the only configuration routed
onto the un-widened kernel.

## Relationship to the earlier batch-128 `nan`

This is **not** the same defect as the batch-128 `nan` seen on Triton 3.6.0
(by step 50) and 3.7.1 (step ~2200). On those versions
`_fused_rng_ln_mul_dropout_is_broken()` is false, so the separated-RNG kernel is
not on the path at all, and at batch 128 `N` never approaches the wrap row
regardless. Consistent with that, batch 128 is now clean over 4000 steps (2/2)
on `7ff97e3109` while batch 1024 fails: the older batch-128 defect was fixed, and
this int32 defect arrived with the separated-RNG path. Two defects, sequentially
exposed — which is why the "onset moved from step 50 to step 2200 to never" trail
never converged on a single cause.

## Validation matrix

| case | expected | result |
|---|---|---|
| `--n 4096` (small) | PASS, bit-exact vs reference | PASS |
| `--n 1398101` (last in-range row) | PASS | PASS |
| `--n 1399126` (1024 rows over) | FAIL at row 1398102 | FAIL, exact |
| `--n 1399126 --fix` | PASS | PASS |
| **e2e batch 1024, 600 steps, fix applied** | **no `nan`** | **PASS — 0/60 non-finite, exit 0** |
| `--sweep-config` (BLOCK_N 1..16 x num_warps 1..4) | FAIL at every config | **not yet run** |

## Not yet done

1. **`N` measured directly in the e2e run at batch 1024.** The instrumented run
   (`PROBE_LNMD_N=1`) wedged the node at step 0 with a GPU hang
   (`svm_migrate_copy_done` blocked on a dma_fence that never signals) before it
   printed. The instrument was reverted; it needs a re-run after a reboot. This
   is the one number in this document that is inferred rather than measured.
2. **`--sweep-config`** across all 15 autotune configs, to confirm the defect is
   config-independent as the source reading implies.
3. ~~**Backward-kernel audit** for the same un-widened pattern.~~ **DONE — clear.**
   `_ln_mul_dropout_bwd_dwdb` (:922) and the Helion variant (:2711) do use
   `rows[:, None] * D`, but their `N` argument is
   `tile_num = max(1, min(sms * 64, N // 4))` (~16 K), four orders of magnitude
   below the wrap row. The `dx_du` backward kernels already widen everywhere
   (:523, :714-719, :2578-2583). **The forward was the only defect.**
4. ~~**E2E confirmation**: batch 1024 with the fix applied, run past step 200.~~
   **DONE (2026-09-10).** 600/600 steps, 60 logged losses, **zero non-finite**,
   exit 0, no fault or hang — `~/mi450_logs/b1024_fixed.log`. Loss flat in
   0.13744-0.14013 (last 0.13826 @ 600), i.e. **3.2x past the later prior onset
   (~190) and 6.6x past the earlier (91)**; the unfixed run on the same config
   went `nan` at 190 after a last finite 0.13981 @ 180.

   Two readings to guard against: `run_stop status="aborted"` is the MLPerf
   convergence verdict, not a crash — every run in `mi450_a0.md` reports it,
   including the clean batch-128 ones; and eval AUC 0.4994 reflects 614,400
   samples of a 5B corpus at `dense_lr=2.5e-08` inside a 24,000-step warmup.

   **Still n=1** against a nondeterministic onset, so this is strong evidence,
   not proof. A second independent 600-step run is the cheapest way to
   strengthen it and is the top remaining item.

## Aside: a single 12 GiB allocation fails with 419 GiB free

While sizing the reproducer past the `stride_x` threshold (N = 4,194,304, needing
~36 GiB), the allocator refused a 12 GiB block:

```
CUDA out of memory. Tried to allocate 12.00 GiB. GPU 0 has a total capacity of
432.00 GiB of which 419.56 GiB is free.
```

with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set. Unrelated to the
above and not investigated, but it caps how large a single tensor this stack can
allocate and is worth a separate look.
