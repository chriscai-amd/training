> **Historical snapshot: September 1–11, 2026.** The complete original text
> from revision `8377ae9` is preserved below. The [main A0 document](mi450_a0.md)
> is authoritative for the current NaN investigation. Historical claims about
> convergence status, unchanged benchmark math, the absence of matrix-core
> kernels, and there being only one source fix have been superseded. The gates
> and re-test instructions below are historical records, not current operating
> instructions.
>
> Original body SHA256: `aecfbe1c6638c037ff77f3f049f458b5b30e596b44a26327110a69aa36944ed4`.

<!-- BEGIN EXACT ORIGINAL BODY FROM 8377ae9 -->

# Bring-up on MI450 A0 (gfx1250, single GPU)

The yambda-5b HSTU ranker on an **MI450 A0 engineering sample**, one GPU. As of
**2026-09-10 it trains end to end on Triton kernels at the production batch size,
1024**: 600/600 steps, 60 logged losses, zero non-finite, `EXIT=0` — and
**reproduced twice, independently, across an `amdgpu` driver reload**.

That took workarounds for six Triton/LLVM defects ([1], [2], [4], [4c], [5], [6]),
three fbgemm patches for wave32, a table shrink to fit one GPU, and **one genuine
source fix** — [3], an int32 row-offset overflow in this repo's own kernel
([row 15 of §4](#4-changes-applied-to-master)). **None of it is in the
benchmark's math.**

**This is functional bring-up at one configuration, not convergence.** The MI350
reference converges to target across 60 runs; 600 steps at batch 1024 is ~1/3.7 of
the way to that reference's *first* eval checkpoint. Read
[1.2](#12-against-the-mi350-reference) before quoting the result.

The working stack is upstream Triton
[`7ff97e3109`](https://github.com/triton-lang/triton/commit/7ff97e310935b4a79794878dbc911f9af25d38d9).
Everything about 3.6.0, AMD 3.7.1 and AMD 3.8.0 is **reference only** and lives in
[9. Historical](#9-historical-earlier-triton-versions). Branch `chcai/mi450_a0`;
companion to [`training_recipe.md`](../training_recipe.md) and
[`perf_opt.md`](../perf_opt.md).

## Contents

1. [Status](#1-status) — what works today, and what it is worth
   - [1.1 Where this stands](#11-where-this-stands) ·
     [1.2 Against the MI350 reference](#12-against-the-mi350-reference) ·
     [1.3 What it does not establish](#13-what-it-does-not-establish)
2. [The working stack](#2-the-working-stack) — pins, configuration, commands
   - [2.1 Pins](#21-pins) · [2.2 Run configuration](#22-run-configuration) ·
     [2.3 Build and run](#23-build-and-run) ·
     [2.4 Operating notes](#24-operating-notes)
3. [Results at batch 1024](#3-results-at-batch-1024) — the production result
   - [3.1 The confirming runs](#31-the-confirming-runs) ·
     [3.2 Secondary configurations](#32-secondary-configurations) ·
     [3.3 What breaks it](#33-what-breaks-it)
4. [Changes applied to master](#4-changes-applied-to-master) — **one table**:
   every delta from `b8b3adc`, each with its full motivation, downside and
   follow-up in the row itself
5. [Open items](#5-open-items) — **what to run next**, ranked, with what each
   clean result would let us delete
   - [5.1 Re-test queue](#51-re-test-queue) ·
     [5.2 Defect detail](#52-defect-detail) ·
     [5.3 Cost of the workarounds still in place](#53-cost-of-the-workarounds-still-in-place)
6. [Defect and port reference for the current stack](#6-defect-and-port-reference-for-the-current-stack)
   - [6.1 Defect record](#61-defect-record) ·
     [6.2 Root cause of the attention-backward fault ([6] / [4d])](#62-root-cause-of-the-attention-backward-fault-6--4d) ·
     [6.3 Triton language API port ([5])](#63-triton-language-api-port-5)
7. [Reproducers](#7-reproducers)
8. [The node and environment](#8-the-node-and-environment)
   - [8.1 Hardware and recovery](#81-hardware-and-recovery) ·
     [8.2 Image, fbgemm and dataset](#82-image-fbgemm-and-dataset) ·
     [8.3 Performance](#83-performance)
9. [Historical: earlier Triton versions](#9-historical-earlier-triton-versions) —
   reference only
   - [9.1 Images tested](#91-images-tested) ·
     [9.2 Defects by version](#92-defects-by-version) ·
     [9.3 Row 7 follow-up](#93-row-7-follow-up-coexec-scheduler-root-cause) ·
     [9.4 The batch-128 nan on 3.6.0 / 3.7.1](#94-the-batch-128-nan-on-360--371) ·
     [9.5 The batch-8 pipelining nan ([4], pre-fix)](#95-the-batch-8-pipelining-nan-4-pre-fix-stack)
10. [Retracted](#10-retracted)
11. [Change inventory](#11-change-inventory)

## 1. Status

### 1.1 Where this stands

Every configuration measured on the working stack
([2](#2-the-working-stack)). **Clean** means zero non-finite loss and `EXIT=0`.

| Configuration | Steps | Samples | Result | Runs | Log |
|---|---|---|---|---|---|
| **Batch 1024 — production** | 600 | 614,400 | **clean**, `train_loss` 0.13826 | **2 / 2** | `b1024_fixed.log`, `b1024_fixed_rep2.log` |
| Batch 128 | 4000 | 512,000 | clean, `train_loss` 0.15350 / 0.15343 | 2 / 2 | `nan128/long{1,2}.log` |
| Batch 1024, **pipelining unclamped** | 600 | 614,400 | clean, `train_loss` 0.13833, **2.8 % slower** | 1 | `retest_0910/pipe_b1024_r1.log` |
| Batch 8 — **retired configuration** | 4000 | 32,000 | clean, `train_loss` 0.10000, AUC 0.5233 | 1 | `results/smoke_7ff97e_pr7/` |
| Batch 1024, **before** the [3] fix | ≤ 200 | — | `nan` at step 91 and ~190 | **0 / 2** | `prod_b1024_smoke.log`, `b1024_rep2.log` |

**Batch 1024 is the working point**, and it is the batch size the production gin
is written for. Until 2026-09-10 it was the *failing* point — the last row is the
same configuration as the first, without
[the i32 fix](#4-changes-applied-to-master).

**The caveat that has not moved.** This doc's own bar for calling a stack clean is
**≥ 3000 steps** ([9.2](#92-defects-by-version), row 3: AMD 3.7.1 looked fixed at
250 steps and went `nan` at 2200). Batch 1024 has 600. The int32 defect *is*
fixed — proven by a deterministic standalone reproducer, not by the absence of a
symptom — but "no `nan` defect remains at batch 1024 beyond 600 steps" is **not**
established. A ≥ 3000-step run at batch 1024 is the top open item
([5.1](#51-re-test-queue)).

### 1.2 Against the MI350 reference

**These are two different categories, not two scores on one scale.** MI350 is
**convergence-validated**: [`rcp_logs/`](../../rcp_logs) holds 60 MLPerf logs,
`submission_platform: MI350`, closed division, at global batch 8192 / 16384 /
32768 with 20 seeds each — **60 of 60 reach the 0.75 AUC target**. MI450 A0 is at
**single-configuration functional bring-up**: the configuration completes and its
loss matches the PyTorch-kernel reference. It makes **no quality claim**.

| | MI350 (RCP) | MI450 A0 today |
|---|---|---|
| Global batch | 8192 / 16384 / 32768 | **1024** — 8× below the smallest RCP point |
| Samples | 69M to reach 0.75 at gbs 8192 (112M at 32768) | **614,400** |
| Eval AUC | 0.7636–0.7875, 60/60 runs ≥ 0.75 | **0.4994** — chance |
| Seeds | 20 per batch size | 1 (2 runs, same seed) |

For scale: **this run is ~1/3.7 of the way to MI350's *first* eval checkpoint**,
which itself reads `0.5010`, and ~1/112 of samples-to-target at gbs 8192. It is
not a short convergence run; it is shorter than the reference's first datapoint.

**Batch size is no longer the blocker — run length is.** The previous edition of
this document recorded the gap as blocked in *both* dimensions, because batch 128
`nan`'d in 3 of 4 runs. On this stack batch 128 is clean 2/2 over 4000 steps and
batch 1024 is clean 2/2 over 600, so the remaining distance is machine time and a
≥ 3000-step confirmation, not a defect that has to be root-caused first. That is
the single biggest change since 2026-09-04.

Two things that cut opposite ways, since readers assume the wrong one:

- **Stronger than "it didn't crash."** The full Triton op set compiles and
  dispatches, and at batch 8 `train_loss` is `0.10000` against the
  PyTorch-kernel reference's `0.1015` — numerics agree with the reference
  implementation at the working point.
- **Weaker than "functional."** That point needs workarounds for six Triton/LLVM
  defects, three fbgemm wave32 patches and a 4× table shrink. None touch the
  benchmark's math, but three **narrow the configuration space instead of fixing
  anything**: pipelining clamped to `num_stages=1`, `BLOCK_N` pinned to 64,
  buffer ops off.

**This is A0 engineering silicon** and at least [6] is an A0 hazard, so some of
the gap may not exist on B0. That split is unmeasured.

Throughput is not comparable yet either, for an unrelated reason: there is no
matrix-core GEMM path for gfx1250, so HFU reads ~1 % (see
[8.3](#83-performance)).

Ladder, if you need one for tracking: compiles → op-level correctness →
**single-config e2e completes at production batch (MI450 today)** →
benchmark-valid config, multi-GPU → **convergence to target (MI350)** →
competitive throughput.

### 1.3 What it does not establish

| Claim a reader might infer | Why it is not established |
|---|---|
| It converges | **No.** 600 steps at batch 1024 is 614,400 samples of a 5B-event corpus, with `dense_lr=2.5e-08` still inside a 24,000-step warmup — there is no opportunity to converge. Eval AUC `0.4994` is chance and is **not a symptom of anything**; it says the eval path runs and returns a finite metric. `run_stop status="aborted"` is the MLPerf convergence verdict, reported by *every* run in this document, clean or not — not a failure signal |
| The stack is clean | **Not by this doc's own bar**, which is ≥ 3000 steps ([1.1](#11-where-this-stands)). Batch 1024 has 600. [3]'s mechanism is fixed — proven by a deterministic reproducer, not by absence of symptom |
| The model is the benchmark's | **No.** `EMBEDDING_ROW_SCALE=0.25` (140.01 GiB against a 560.04 GiB fp32 reference), ids modulo-wrapped into the smaller vocabulary. Not an MLPerf-valid configuration, and the vocabulary change means accuracy is not comparable to the RCP reference **even in principle** |
| It scales | **Untested.** One GPU; nothing exercises sharding, and row 2's gate touches device-0 initialisation at import time — exactly where a multi-rank run would notice |
| The hang class is fixed | **Avoided, not fixed.** [1b] shows hangs that survive the knob |
| The workarounds are free | **One of three is measured, and it is not the big one.** Row 3 has an e2e number and it favours the workaround — the clamp is **2.8 % faster** at batch 1024 ([5.3](#53-cost-of-the-workarounds-still-in-place)). Row 2 is kernel-level only (1.54× slower); no e2e number, because `BLOCK_N=128` faults. **Row 1 — buffer ops off for every kernel — has no number at all and cannot get one on this host**, since the A/B needs the dispatch that hangs. It is the largest unpriced item in the document ([5.1](#51-re-test-queue), rank 1) |

## 2. The working stack

### 2.1 Pins

| Component | Pin | Note |
|---|---|---|
| GPU | 1× MI450 A0 engineering sample, `gfx1250`, device rev `0x00` | wave32, 256 WGPs, ~432 GiB HBM |
| Triton | upstream `triton-lang/triton` @ [`7ff97e3109`](https://github.com/triton-lang/triton/commit/7ff97e310935b4a79794878dbc911f9af25d38d9) (reports `3.8.0`), bundled LLVM `b010a18d` | built from source; **not** the `:triton-main` tag, which is `58895270` and will not compile ([5]) |
| torch | `2.11.0+rocm7.14.0a20260625` | gfx1250 code objects present |
| ROCm | 7.14 nightly wheels, at `site-packages/_rocm_sdk_devel` | driver amdgpu DKMS 7.1.1 |
| fbgemm | `10b7757` + three patches (rows 6–8 below) | built from source in the image |
| Image | `recommendation-amdprimus0815:gfx1250` with Triton rebuilt on top | [`Dockerfile.amdprimus0815`](../../Dockerfile.amdprimus0815) |
| Host | Linux `6.14.0-37-generic`, Docker 29.6.1 | `sudo modprobe amdgpu` after every boot |

The same fix set also passes on AMD `3.8.0+git4cff872c`
([raikonenfnu/training#2](https://github.com/raikonenfnu/training/issues/2)), so
it is not specific to upstream `main`.

### 2.2 Run configuration

**The production configuration**, driven by
[`scripts/run_prod_convergence_1gpu.sh`](../../scripts/run_prod_convergence_1gpu.sh),
which is the harness that produced both batch-1024 results in
[3.1](#31-the-confirming-runs). It runs the production `yambda_5b.gin`
**unmodified except for `EMBEDDING_ROW_SCALE`**; everything else is left unset so
the gin supplies its own defaults.

| Knob | Value | Why it is load-bearing |
|---|---|---|
| `BATCH_SIZE` | **1024** | Production batch size. `WORLD_SIZE` is derived, so on one GPU this is also the *global* batch |
| `EMBEDDING_ROW_SCALE` | 0.25 | 73,262,927 rows = 140.01 GiB. Unscaled is 560.04 GiB = 1.30× the device. **The only intended model deviation** |
| `AMDGCN_USE_BUFFER_OPS` | **0** | The only correctness flag you must pass — `1` wedges the node ([1]) and costs a reboot |
| `HSTU_HAMMER_KERNEL` | `TRITON` | Gin default. `PYTORCH` is the reference arm, and much slower |
| `HISTORY_LENGTH` / `MIN_HISTORY` / `MAX_SEQ_LEN` | 4086 / 4086 / 4096 | Gin production defaults — the true e2e sequence shape |
| `HSTU_NUM_LAYERS` | 3 | Production depth. The 5 at `configs.py:140` and 12 at `dlrm_hstu.py:83` are dead constructor defaults |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | |
| `SMOKE` / `SMOKE_BATCHES` | 1 / 600 | Bounded validation, **not** a convergence attempt |
| `SEED` | 1 | Gin default; both runs share it, which is why they agree to the digit |

```bash
SMOKE=1 SMOKE_BATCHES=600 scripts/run_prod_convergence_1gpu.sh <log>   # the validation run
scripts/run_prod_convergence_1gpu.sh <log>                             # full convergence, 40-90 h
```

The script runs inside the **existing** container (default `triton-7ff97e-test`),
which carries the `7ff97e3109` build [2.1](#21-pins) pins — a fresh `docker run`
of the base image would get a different Triton.

Two knobs are deliberately left unset, and they are **not** the same kind of
decision any more:

| Unset knob | What staying unset holds in place | What setting it costs today |
|---|---|---|
| `TRITON_FULL_AUTOTUNE` | The `BLOCK_N=64` pin for [6] | **A defect.** `BLOCK_N=128` becomes reachable and the attention backward faults — reproducible every run ([3.3](#33-what-breaks-it)) |
| `TRITON_ALLOW_PIPELINING` | [4]'s `num_stages=1` clamp | **Throughput, not correctness.** Measured 2026-09-11 at batch 1024: unclamped is **2.8 % slower** and 600 steps were clean. [4] did not reproduce ([5.1](#51-re-test-queue), rank 7) |

So only the first is a correctness workaround that applies itself. The clamp is
now kept on the narrower ground that lifting it is a measured regression for no
observed benefit.

At global batch 1024 the gin's tuned defaults are off-recipe in one respect worth
knowing: the RCP recipe scales peak LR linearly with global batch (1e-6 at 8192),
so the stock `1e-6` is arguably 8× high here. It is left at the production
default deliberately; override `DENSE_LR` / `SPARSE_LR` if the loss misbehaves
after warmup.

### 2.3 Build and run

Triton is `pip install -e`'d **into a container** and produces no image, so the
shape of this is: start a long-lived container, build Triton in it, train in it.
That container is what [2.2](#22-run-configuration)'s harness then execs into.
Note [`training_recipe.md`](../training_recipe.md) is the **MI350, 8-GPU** recipe
(batch 32 × world 8) and does **not** apply here.

**The production run is [2.2](#22-run-configuration).** The batch-8 recipe below
is the *original bring-up* configuration and is **retired as a test target** —
kept for two reasons only: it is the e2e reproducer for [6] cited in
[6.1](#61-defect-record), and it is where the PyTorch-kernel numeric comparison
was made ([3.2](#32-secondary-configurations)).
Every value is confirmed against
[`results/smoke_7ff97e_pr7/train.log`](../../results/smoke_7ff97e_pr7/train.log),
not from the command we believe we typed.

| Knob | Value | Why it is load-bearing |
|---|---|---|
| `BATCH_SIZE` | **8** | The bring-up point, superseded by 1024 ([3.1](#31-the-confirming-runs)) |
| `WORLD_SIZE` / `GPUS_PER_NODE` | 1 / 1 | Single GPU; nothing exercises sharding |
| `HSTU_HAMMER_KERNEL` | `TRITON` | `PYTORCH` is the reference arm, and much slower |
| `AMDGCN_USE_BUFFER_OPS` | **0** | The only correctness flag you must pass — `1` hangs the node ([1]) |
| `EMBEDDING_ROW_SCALE` | 0.25 | ~140 GB of tables, to fit one GPU. Not MLPerf-valid |
| `HBM_CAP_GB` | 400 | Planner topology cap, of ~432 GiB |
| `MAX_SEQ_LEN` / `HISTORY_LENGTH` / `MIN_HISTORY` | 4096 / 4086 / 64 | True e2e sequence shape; the jagged extents [6] needs |
| `HSTU_NUM_LAYERS` | 3 | |
| `NUM_TRAIN_BATCHES` | 4000 | = 32,000 samples. **Not** the runner default (5) |
| `NUM_EVAL_BATCHES` | 20 | What produced AUC `0.5233`. **Not** the runner default (2) |
| `START_TS` / `NUM_TRAIN_TS` | 150 / 1 | One streaming window |
| `AUC_THRESHOLD` | 1.0 | Never fires, so `run_stop` reads `aborted` by design |

The same two unset variables as [2.2](#22-run-configuration) apply here, and for
the same reason. Build script:
[`scripts/build_triton_from_source.sh`](../../scripts/build_triton_from_source.sh).

```bash
# 1. Long-lived container from the :gfx1250 image, repo and yambda cache mounted.
docker run -d --name triton-7ff97e --entrypoint sleep \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --security-opt seccomp=unconfined --shm-size=64g \
  -v "$PWD:/workspace/recommendation" \
  -v "${DLRM_DATA_PATH:-$HOME/dlrm_data}:/data/mlperf_dlrm_v4" \
  -e PYTHONPATH=/workspace/recommendation -e DLRM_DATA_PATH=/data/mlperf_dlrm_v4 \
  recommendation-amdprimus0815:gfx1250 infinity

# 2. Build Triton main in it (~5 min on this node).
docker exec -w /workspace/recommendation triton-7ff97e \
  env TRITON_COMMIT=7ff97e310935b4a79794878dbc911f9af25d38d9 \
  ./scripts/build_triton_from_source.sh

# 3. Train: 4000 batches, one GPU, shrunk tables. Mostly the
#    run_smoke_shrink_1gpu.sh defaults -- NUM_TRAIN_BATCHES and NUM_EVAL_BATCHES
#    are not (they default to 5 and 2) -- passed explicitly because the trainer
#    is invoked directly rather than through the runner's own `docker run`.
docker exec -w /workspace/recommendation \
  -e AMDGCN_USE_BUFFER_OPS=0 -e HSTU_HAMMER_KERNEL=TRITON \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HSA_ENABLE_COREDUMP=0 -e HSA_COREDUMP_PATTERN=/tmp/gpucore.%p.gpu \
  -e EMBEDDING_ROW_SCALE=0.25 -e HBM_CAP_GB=400 \
  -e BATCH_SIZE=8 -e NUM_WORKERS=2 -e PREFETCH_FACTOR=2 \
  -e HISTORY_LENGTH=4086 -e MIN_HISTORY=64 -e MAX_SEQ_LEN=4096 -e HSTU_NUM_LAYERS=3 \
  -e START_TS=150 -e NUM_TRAIN_TS=1 \
  -e NUM_TRAIN_BATCHES=4000 -e NUM_EVAL_BATCHES=20 \
  -e EVAL_EVERY_N_WINDOWS=1 -e EVAL_EVERY_DATA_PCT=0 -e AUC_THRESHOLD=1.0 \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e WORLD_SIZE=1 \
  -e MASTER_ADDR=localhost -e MASTER_PORT=29501 -e NCCL_SOCKET_IFNAME=lo \
  -e PROGRESS_EVERY=100 \
  triton-7ff97e python -m generative_recommenders.dlrm_v4.train.train_ranker \
    --dataset yambda-5b --mode streaming-train-eval
```

`docker commit triton-7ff97e <tag>` then

```bash
IMG=<tag> NUM_TRAIN_BATCHES=4000 NUM_EVAL_BATCHES=20 ./scripts/run_smoke_shrink_1gpu.sh
```

is the same run through the runner — where the rest of those defaults come from
— at the cost of a ~47 GB layer. **Both overrides are needed:** the runner
defaults to 5 train and 2 eval batches, so plain `NUM_TRAIN_BATCHES=4000` trains
correctly but evaluates on 2 batches and will not reproduce AUC `0.5233`.

### 2.4 Operating notes

- **`AMDGCN_USE_BUFFER_OPS=0` is the only correctness flag you must pass**, and
  both runners default it. Every other workaround is gated in code on arch and
  Triton version, so it applies itself. `AMD_SERIALIZE_KERNEL=3` is optional and
  costs throughput — leave it off outside debugging.
- **Start from an idle GPU.** A faulted rank and its orphaned dataloader workers
  keep holding HBM, so the next run starts under artificial memory pressure.
  `scripts/bisect_hooks/gpu_cleanup.sh` clears them; check `rocm-smi
  --showmemuse` reads 0 % first.
- **`sudo modprobe amdgpu` after every boot** — the module is blacklisted on the
  kernel cmdline; `/dev/kfd` appears ~15 s later. A container created while
  `/dev/kfd` was absent will not see the GPU: recreate or restart it after the
  modprobe. See [8.1](#81-hardware-and-recovery).
- **A 600-step batch-1024 run takes ~34 min** — ~1.9 min of startup (140 GiB
  table allocation, autotune) then **3.40 s/step** (measured `step_ms` 3397 and
  3403 across the two runs; 3492 unclamped).

## 3. Results at batch 1024

### 3.1 The confirming runs

Two independent 600-step runs on the configuration in
[2.2](#22-run-configuration), with the [3] fix applied. Recorded 2026-09-10.

| Measure | Run 1 (`b1024_fixed`) | Run 2 (`b1024_fixed_rep2`) | Unfixed baseline |
|---|---|---|---|
| Started | 03:00:08 UTC | 22:41:42 UTC | 01:20 / 01:35 UTC |
| Exit | **0** | **0** | 0 (completed with `nan`) |
| Steps | **600 / 600** | **600 / 600** | `nan` at 91 / ~190 |
| Logged losses | 60 | 60 | — |
| Non-finite | **0** | **0** | present |
| Final `train_loss` | **0.13826** | **0.13826** | `nan` |
| Loss range | 0.13744 – 0.14013 | 0.13743 – 0.14014 | 0.1377 – 0.1398 then `nan` |
| Faults / hangs | 0 | 0 | 0 — silent corruption |
| Wall clock | 34m 19s | 34m 14s | ~10 min to onset |
| `window_auc` | 0.4994353 | 0.4994041 | — |
| `lifetime_auc` | 0.4994354 | 0.4994046 | — |

**Why run 2 is an independent trial and not a rerun.** Between the two, the
`amdgpu` module was unloaded and reloaded and the container was recreated against
the new `/dev/kfd`. No warm GPU state, allocator layout or autotune cache carries
across — which matters because [3]'s onset was *nondeterministic* and its silent
mechanism depended on a per-process allocator accident (which live tensor sits
4.29 GB below `Y`).

**Why the two runs agree to the digit.** `SEED=1` and a deterministic data
stream. That agreement is the point rather than a red flag: the unfixed runs had
the *same* seed and still diverged into `nan` at different steps (91 and ~190),
because the corruption was allocator-dependent. Determinism now extends to step
600 instead of breaking partway.

The AUCs differ in the 5th decimal (0.4994353 vs 0.4994041) while the losses
match to all five — ordinary non-determinism in the eval reduction, and at
614,400 samples both figures are chance either way.

### 3.2 Secondary configurations

**Batch 128 — clean, 2 / 2 over 4000 steps** (2026-09-10,
`~/mi450_logs/nan128/long{1,2}.log`). Final `train_loss=0.15350` and `0.15343`,
zero non-finite, both satisfying the ≥ 3000-step rule. Both walked through AMD
3.7.1's old onset window finite (step 2150 → 0.13860, **2200 → 0.13925**, 2250 →
0.13541). Against the historical 3-in-4 failure rate at this batch, two clean
runs is `0.25² ≈ 6 %` — evidence, not proof; see
[9.4](#94-the-batch-128-nan-on-360--371) for what that defect was and why it is
considered separate.

**Batch 8 — the original bring-up point, now a retired configuration**
(2026-09-04, `results/smoke_7ff97e_pr7/`). It is **not re-tested and no live
claim rests on it**; it is retained for two artefacts that exist nowhere else:
the PyTorch-kernel numeric comparison below, and the e2e reproducer for [6].

| Measure | Result | Reference |
|---|---|---|
| Batches | **4000 / 4000**, `run_stop` | PyTorch kernels also complete 4000 |
| Wall clock | 5.2 min, train + eval | — |
| Final `train_loss` | **0.10000** | `0.1015` with `HSTU_HAMMER_KERNEL=PYTORCH` |
| `nan` | none in 80 logged steps | 3.6.0 goes `nan` by step 50 ([3]) |
| Eval AUC | `0.5233` | — |

`run_stop` says `status: "aborted"` only because `AUC_THRESHOLD=1.0` never fires.
The fix author reports the same run passing serialised and unserialised, on
`7ff97e3109` and AMD `3.8.0+git4cff872c`
([#2](https://github.com/raikonenfnu/training/issues/2)). The standalone
attention-backward reproducer is also clean: 4000/4000 on all four layouts, each
of which faulted by iteration ~125–150 before the fix.

### 3.3 What breaks it

**Two knobs, both reproducible every run.** Each is one change away from the
recipe in [2.2](#22-run-configuration), and both are debug knobs you would
otherwise leave alone.

Nothing else in this document belongs here. **Batch size does not:**
`BATCH_SIZE=128` was the failing configuration on 3.6.0 / 3.7.1 and batch 1024
failed 2/2 before the [3] fix; both are clean on this stack
([9.4](#94-the-batch-128-nan-on-360--371)).
**`TRITON_ALLOW_PIPELINING=1` no longer does either:** it was removed from this
table on 2026-09-11 after 600 clean steps at batch 1024, and its symptom,
candidate-kernel list and triage trail moved to
[9.5](#95-the-batch-8-pipelining-nan-4-pre-fix-stack). The clamp that makes it a
non-issue is [row 3 of §4](#4-changes-applied-to-master).

| Change | Defect | Deterministic? | Symptom | Suspicious component | Standalone reproducer for triage |
|---|---|---|---|---|---|
| `AMDGCN_USE_BUFFER_OPS=1` | [1] | **Yes** — the launch never returns, every run | **Node hang, reboot required** | **The AMD backend buffer-op pass group**: `canonicalize_pointers` → `convert_to_buffer_ops` → `optimize_buffer_op_ptr`, driven by `HIPBackend.get_tensor_specialization` setting `tt.pointer_range=32` for args ≤ 2 GiB. That leaves **hybrid 32/64-bit addressing in one kernel body**, with masked lanes carrying sentinel offset `0x80000000` against `num_records=0xffffff` | **Yes — the one to hand over.** `python scripts/repro_gfx1250_buffer_ops.py`: torch+triton only, no repo imports, no fbgemm, no dataset. `--compile-only` never dispatches, so triage can inspect the hybrid addressing **without risking a node**, and it still reproduces on `7ff97e3109` |
| `TRITON_FULL_AUTOTUNE=1`, or reverting the pin to `BLOCK_N=128` | [6] | **That it faults, yes; where, no** — standalone faults by iteration ~125–150 every run, but the e2e batch index moved across 0, 13, 17, 55, 114, 527 | `Memory access fault … Page not present`; recoverable | **LLVM AMDGPU register allocation — already root-caused.** An address-critical work-item value is kept live in extended VGPR `v257` across the WMMA region, under mutable `S_SET_VGPR_MSB` mode state, and lane 0 of it comes back corrupted. **Not** spilling and not the coexec scheduler alone. Fix belongs in a subtarget-gated erratum; `-mattr=-1024-addressable-vgprs` is the blunt control | **Yes, at two levels.** `python scripts/repro_gfx1250_attn_bwd.py --part bwd --layout contiguous` faults in ~15 s (needs the repo, no dataset), with `--part fwd`, `--layout guarded` and `--contextual 0 --no-targets` as controls. For the compiler team, a checked **MIR reproducer** in [#3](https://github.com/raikonenfnu/training/issues/3) — the level the fix has to land at |

## 4. Changes applied to master

**One table, and it is the only place a change is described.** Against
`mlcommons/training` @ `b8b3adc`, the commit this branched from: every change
that is a workaround or a fix, with its full rationale in the Motivation column
rather than in a parallel prose section. Scripts, docs and traces are excluded
and listed under [11. Change inventory](#11-change-inventory).

Rows 1-5 are what the e2e run depends on, 6-8 are what makes the image build on
wave32, the rest are harness and diagnostics. **Row 15 is the only entry that is
a true fix rather than a configuration dodge** — the owner is this repo, not the
Triton or LLVM teams, and it is what moved the working point to the production
batch size. Rows 5 and 15 are **Triton-language API changes** for current
upstream rather than miscompile dodges; the mapping and the one pitfall the port
introduces are in [6.3](#63-triton-language-api-port-5). Bracketed ids are
stable — cite those, not row numbers.

| # | Change (where) | Motivation | Perf impact | Potential downside | Potential follow-up |
|---|---|---| --- |---|---|
| 1 | **`AMDGCN_USE_BUFFER_OPS=0`** — env, defaulted by [`scripts/run_smoke_shrink_1gpu.sh`](../../scripts/run_smoke_shrink_1gpu.sh) | [1]: the buffer-op pass group wedges the **whole node** on kernels whose pointer args straddle the 2 GiB narrowing cutoff, so one body mixes 32-bit `buffer_*` and 64-bit `global_*` addressing. [2]: the same group silently corrupts 20 / 7640 elements in the position kernel | **Unmeasured, and may be unmeasurable here.** An A/B needs buffer ops *on*, which hangs the node ([1]) | Disables `canonicalize_pointers` → `convert_to_buffer_ops` → `optimize_buffer_op_ptr` for every kernel; the perf cost is unmeasured. Does **not** cover [1b], where hangs still occur with the knob off | Triton fix for hybrid 32/64-bit narrowing on gfx1250, and CI above the 2 GiB cutoff. Owed locally: a matched [1b] control, knob on vs off, one test per process |
| 2 | **`BLOCK_N=64`** for the pinned attention-backward config — `_get_bw_pinned_configs()` in `ops/triton/triton_hstu_attention.py`, gated on gfx1250 + Triton ≥ 3.8 ([PR #7](https://github.com/chriscai-amd/training/pull/7)) | [6] / [4d]: `_hstu_attn_bwd` stores to a wild address mid-run. Root cause is an address-critical value held in an **extended VGPR across the WMMA region** ([#2](https://github.com/raikonenfnu/training/issues/2), [#3](https://github.com/raikonenfnu/training/issues/3)); the N=128 tile is what exposes it | **Measured: −1.54× on the kernel.** 11.22 ms at `BLOCK_N=64` vs 7.30 ms at 128, ±0.05 ms, at 654 vs 987 VGPRs with no spills. e2e A/B impossible — 128 faults. The largest confirmed regression in this table | Halves the N tile; perf cost unmeasured. The hazard itself is untouched, so another shape that produces the same allocation can still hit it, and `TRITON_FULL_AUTOTUNE=1` restores the N=128 config. The gate calls `torch.cuda.get_device_properties(0)` at **import time**, initialising a HIP context on device 0 before a spawned rank selects its own | Land the LLVM-side fix ([#3](https://github.com/raikonenfnu/training/issues/3): rematerialise address-critical work-item values after WMMA regions, gated as a gfx1250-A0 erratum), then drop the pin. Run the cold-boot **native B0** matrix to decide whether this is A0 silicon or a backend requirement. Measure the N=64 vs N=128 throughput delta |
| 3 | **`clamp_num_stages`** forcing `num_stages=1` — `common.py`, gated on HIP + Triton ≥ 3.8, overridable with `TRITON_ALLOW_PIPELINING`. It sits in the OSS `triton_autotune` wrapper, so it reaches the 30 kernels that route through it across eight `ops/triton` files | [4]: 3.8.0 miscompiles software-pipelined loops. With `num_stages ≥ 2` attention faults at `gstep=0` or returns 3.3e+28 garbage; at `1` it is bit-comparable to 3.6.0. Not attention-specific — the jagged and layer-norm kernels autotune to `num_stages=3` and fault too | **Measured 2026-09-11: none — slightly _positive_.** 3397/3403 ms clamped against 3492 ms unclamped at batch 1024, so the clamp is **2.8 % faster**. The earlier "likely material" cost is withdrawn | Loses pipelining across all of those kernels. **Cost measured 2026-09-11 at batch 1024 and it is negative: the clamp is 2.8 % _faster_** (3397/3403 ms clamped against 3492 ms unclamped), so the earlier "likely material" cost is withdrawn. Inert on 3.6.0 / 3.7.1, which keep their original staging. **Coverage is not total:** `_weighted_rms_norm_fwd` (`triton_layer_norm.py:785`) uses a raw `@triton.autotune` and is never clamped — it has not faulted, but it is not protected either | **Re-checked 2026-09-11 at batch 1024: [4] did not reproduce** — 600 steps unclamped, `EXIT=0`, finite ([5.1](#51-re-test-queue), rank 7). The `nan`-at-step-100 evidence behind the 2026-09-04 re-check was **batch 8, now retired** ([9.5](#95-the-batch-8-pipelining-nan-4-pre-fix-stack)). The gate stays on the narrower ground that removing it is a measured regression for no observed benefit, not that [4] is proven live. Owed: a plain-pointer rewrite of [`repro_gfx1250_pipeliner.py`](../../scripts/repro_gfx1250_pipeliner.py) — it cannot compile on current main ([5]) — aimed at a jagged or layer-norm kernel |
| 4 | **Separated-RNG dropout path forced on** — `_fused_rng_ln_mul_dropout_is_broken()` in `ops/utils.py`, folded into `use_separated_rng_ln_mul_dropout()`, gated on gfx1250 + Triton ≥ 3.8 | [4c]: `_ln_mul_dropout_bwd_dx_du` dies with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` whenever dropout is active. Proximate cause is the gfx1250 coexec machine schedule under LLVM expert mode ([#1](https://github.com/raikonenfnu/training/issues/1)) | **Unmeasured.** Splits one fused kernel into separate RNG and LN/dropout work, so extra launches and traffic are plausible but unquantified. The removal candidate with the clearest theoretical upside | Low risk — it is an upstream-supported path that sm100–103 and gfx950 already take — but whether gfx1250 wants it on **performance** grounds was never measured, and settling that on a miscompiling version would be meaningless | [#1](https://github.com/raikonenfnu/training/issues/1)'s retest reports latest main passing the original two- and four-warp launches, so re-check on `7ff97e3109` and narrow the gate to AMD 3.8.0. Then decide the separated path on perf alone |
| 5 | **Pointer port off `tl.make_block_ptr` / `tl.advance`** — 26 + 8 call sites across `ops/triton/triton_hstu_attention.py`, `triton_layer_norm.py`, `triton_hstu_linear.py`. See [API changes](#63-triton-language-api-port-5) | [5]: upstream removed block pointers, so `tl.make_block_ptr` raises `NotImplementedError` and the kernels do not compile at all on any `main` tried — including `76940ad3`, the revision sglang pins for gfx1250 | **Unmeasured, expected neutral** — the same addressing expressed through the descriptor API. A port, not an algorithm change | Hand-written index arithmetic and masks replace `boundary_check` / `padding_option`. Feeding the remapped contextual/multi-target coordinates into pointer arithmetic is now a **silent layout bug** rather than a compile error. Perf against block pointers unmeasured | Revisit if upstream's `TensorDescriptor` gains an AMD lowering; `_should_enable_tma()` is `False` on HIP today because that path lowers to Hopper TMA. `scripts/attn_check.py` is the numeric guard if these kernels are touched again |
| 6 | **fbgemm host `kWarpSize` 64 → 32** — `include/fbgemm_gpu/utils/cuda_prelude.cuh`, patched in the Dockerfile | wave32: the host constant is a "stop-gap" 64 for ROCm while device compilation correctly uses 32, so the TBE backward launched `dim3(64, 32)` = 2048 threads and died on the 1024-per-block limit | **Unmeasured, structural.** Required for wave32 correctness; there is no wave64 configuration on this part to compare against | The wheel is then **single-arch**: wrong for wave64 CDNA parts. With it, TBE is bit-exact against CPU | Upstream fbgemm should derive the host constant from the device warp size instead of hardcoding per-vendor |
| 7 | **fbgemm codegen `items_per_warp` 256 → 128** — `codegen/genscript/jinja_environment.py`, patched in the Dockerfile | wave32: codegen emits `kThreadGroupSize` instantiations from `items_per_warp // 4` and hardcodes 256 for ROCm, which then disagrees with the patched call sites — undefined symbol at load | **Unmeasured, structural** — same reasoning as row 6 | Same single-arch caveat as row 6 | Same as row 6 |
| 8 | **`Fbgemm.cmake` fp16/bf16 convert sources added** — patched in the Dockerfile | At that fbgemm commit the file omits `FbgemmFloat16Convert*.cc` / `FbgemmBfloat16Convert*.cc` and relies on torch's bundled FBGEMM; amdprimus torch 2.11 exports no `fbgemm::Float16ToFloat_avx2`, so `fbgemm.so` fails to load | **None** — build-time source list only | Diverges from that commit's build and will conflict on an fbgemm bump | Upstream HEAD already ships this source list — drop the patch when bumping fbgemm |
| 9 | **`EMBEDDING_ROW_SCALE`** (runner default `0.25`) plus id modulo-wrap — `dlrm_v4/configs.py`, `dlrm_v4/datasets/yambda.py` | One ~432 GiB GPU against ~559 GiB of fp32 reference embedding tables | **Not a perf knob; it changes the workload.** Tables shrink to 140 GiB to fit one GPU, so throughput here is not comparable to a full-table run | **Changes the model**, not just its footprint: smaller vocabulary with ids wrapped into it. Not a valid MLPerf configuration, and accuracy is not comparable to the reference | Multi-GPU sharding to run unscaled; or a scale sweep to bound how much of the accuracy gap is the shrink |
| 10 | **Peak FLOPS keyed on `gcnArchName`** — `dlrm_v4/utils.py` | These parts report the generic name `AMD Radeon Graphics`, so `get_gpu_peak_flops` matched nothing and silently fell back to **MI350X's 2300 TFLOPS** — every earlier HFU number in this repo's logs is against the wrong chip | **None at runtime — it corrects reported numbers.** MFU/HFU only; earlier logs were scored against MI350X's 2300 TFLOPS | The denominator is 90 % of *silicon* peak while this stack's BLAS has no matrix-core kernel (~75 TFLOP/s measured), so HFU reads ~1 %. Deliberate: normalising against measured throughput would conflate model efficiency with BLAS immaturity | Revisit the denominator once matrix-core GEMMs land for gfx1250 |
| 11 | **`HSA_ENABLE_COREDUMP=0` + `HSA_COREDUMP_PATTERN=/tmp/gpucore.%p.gpu`** — runner defaults | A memory-access fault writes a coredump the size of allocated VRAM (9–36 GB observed), which otherwise lands in the working tree | **None** on the normal path; avoids writing a 9–36 GB dump on fault | `HSA_ENABLE_COREDUMP=0` is **not honoured** on this runtime, so the dump is relocated rather than suppressed — the disk write still happens | Runtime bug: honour the suppression knob |
| 12 | **Diagnostics** — `DEBUG_NAN_HOOKS` in `dlrm_v4/train/train_ranker.py`; `scripts/bisect_hooks/` (`sitecustomize.py` per-op Triton/PyTorch forcing, `gpu_cleanup.sh`) | Attributing [3]'s `nan` to the kernel that *manufactured* it, and attributing an e2e fault to a single op group — which is how [6] was placed on `hstu_preprocess_and_attention` | **None when off.** `DEBUG_NAN_HOOKS` and the bisect hooks add per-op synchronization when enabled | `DEBUG_NAN_HOOKS=2`'s pre-forward param sweep has itself wedged the GPU ([1b]), so it is opt-in, and hooks are unusable on 3.8.0 (they fault at `gstep=0`) | — |
| 13 | **`mul_u_activation_type` passed to the fused linear forward** — `ops/triton/triton_hstu_linear.py` | Inherited code bug, not a GPU bug: the forward dropped the parameter and applied no activation while its own backward applied one. One line takes that parameter grid from 32 mismatches to 0 | **Negligible** — one activation in the forward that the backward already applied | None | Send upstream. Dormant in the benchmark, which only ever passes `"none"` |
| 14 | **`&` / `\|` instead of Python `and` / `or` on masks** — `ops/triton/triton_position.py`. The other Triton-language change besides row 5; see [API changes](#63-triton-language-api-port-5) | Not compile-blocking today — the kernels already built — but Triton **warns and will error** on the Python operators applied to non-scalar tensors inside `@triton.jit`, so this is the second API accommodation for current upstream | **None** — the same predicate in a Triton-legal spelling | None. For boolean masks the two are equivalent: 3.6.0's `visit_BoolOp` already lowered `and`/`or` to `logical_and`/`logical_or` | None — and note the "this left loads unmasked" reading of it was **retracted**; see [Retracted](#10-retracted) |
| 15 | **`rows` widened to `int64` in `_ln_mul_dropout_fwd_rng`** — `rows_i64 = rows.to(tl.int64)` defined once at `ops/triton/triton_hstu_linear.py:186`, used by the loads (:189, :194), the mask-pointer reuse (:236) and **all eight `Y` stores** (:272-:310); no raw `rows[:, None]` survives in the kernel | **[3] `train_loss=nan` at batch 1024 — the one genuine source bug on this branch, and what moved the working point from batch 8 to 1024.**<br>• **Mechanism:** the eight `Y` stores and the `X`/`U` loads indexed with a raw int32 `rows`, while the dropout-mask pointer three lines above *is* widened and every neighbouring kernel in the file widens too. At the production shape (`D = 4 heads x 128 = 512`, `concat_u=concat_x=True`) `stride_y = 3*D = 1536`, so `rows * stride_y` overflows int32 at **row 1,398,102** and the store lands at **`Y - 4.29 GB`**. `row_mask = rows < N` does not save it — the mask is computed on the *unwrapped* row id, so the wild store proceeds under a passing mask. **A bounds check and an address computed from different values** is the pattern worth recognising elsewhere in these kernels.<br>• **Why it is silent:** `Y` is a fresh `torch.empty` in a process already holding ~140 GiB of embedding table, so `Y - 4.29 GB` lands in *other live allocations*, not unmapped memory — no fault, just poisoned tensors, which is [3]'s exact recorded signature. It also explains the wandering onset (91 vs ~190 under an identical `SEED=1` stream): which tensor sits below `Y` is a per-process allocator accident.<br>• **Why it looks batch-gated but is not:** the autotune key is `key=["BLOCK_D"]`, so **codegen is identical at every batch size**. `N` is total jagged rows, so batch size enters only by scaling `N` past the wrap row. Under the in-use `interleaved` strategy (~2663 events/sample, a *measured* figure from `yambda_5b.gin:463-478`, not the nominal 4086) N is 342 K at batch 128 and **2.74 M at 1024** — 48.9 % of rows wrapping, crossover at batch 523. Under `last_n` (~4085) it is 524 K / 4.19 M, crossover 342. **The verdict is the same either way**, so it does not rest on the one free parameter.<br>• **Verified on hardware, exactly:** `N=1398101` PASS; `N=1399126` FAIL with first bad row **1398102** — the predicted wrap row to the row — 1024 bad rows and **1,572,864 = 1024 x 1536 canary elements clobbered in an unrelated live tensor, zero faults**. `--fix` clean at every N.<br>• **Why only this kernel:** reached only when `use_separated_rng_ln_mul_dropout()` is true (`ops/utils.py:100-165`) — on AMD that means **gfx1250 + Triton >= 3.8**, itself the [4c] workaround. MI300/A100/H100 take the legacy `_ln_mul_dropout_fwd`, whose offsets are already widened. Arch-correlated without being an arch bug.<br>• **Not the batch-128 `nan` of 3.6.0/3.7.1** — that kernel is not on the path on those versions and batch 128 never approaches the wrap row ([9.4](#94-the-batch-128-nan-on-360--371)).<br>• **Still an inference in one respect:** `N` is from corpus statistics in the gin file, not read out of the live kernel — the instrumented run wedged the node before printing.<br>• Handoff: [`lnmuldropout_i32_handoff.md`](lnmuldropout_i32_handoff.md). Reproducer: `AMDGCN_USE_BUFFER_OPS=0 python3 scripts/repro_gfx1250_lnmuldropout_i32.py --n 1399126 --canary-gib 2` (torch+triton only, deterministic, cannot fault). Results: [3.1](#31-the-confirming-runs). | **Unmeasured, expected small.** One `int64` conversion hoisted to a single definition and reused by every load and store. No pre-fix batch-1024 baseline exists — it `nan`'d | **None.** A strict widening, matching what the dropout-mask pointer three lines below and every neighbouring kernel already did. `Mean + rows` / `Rstd + rows` are deliberately untouched (element stride 1, cannot reach 2**31 before `N` does).<br>**Backward kernels audited — no defect found:** `_ln_mul_dropout_bwd_dwdb` (:922) and the Helion variant (:2711) do use `rows[:, None] * D`, but their `N` is `tile_num = max(1, min(sms * 64, N // 4))` (~16 K), four orders of magnitude below the wrap row; the `dx_du` backwards already widen throughout (:523, :714-719, :2578-2583). **The forward was the only defect.** | Upstream the same widening to `mlcommons/training`. The legacy `_ln_mul_dropout_fwd` is already correct, so only the separated-RNG path (row 4) needs it. **Owner is this repo, not the LLVM/AMDGPU team** — the generated code faithfully implements what the source asks for, and the source asks for a 32-bit multiply. |

## 5. Open items

**Vocabulary, because this document used to blur it.** A defect has a *state* and
a *gate*, and they are independent — a defect whose test passes can still have its
gate applied, which is most of the table below. These words are used in exactly
these senses from here on.

| Word | Means | Does **not** mean |
|---|---|---|
| **Fixed** | Root cause removed in source and proven by a deterministic reproducer that fails before and passes after | "We stopped seeing it" |
| **Gated** | The defect is intact; a workaround narrows the configuration so it cannot fire. The gate is still in the tree | Fixed. Remove the gate and it comes back |
| **Passing** | The defect's own test or reproducer passes on the current stack — **while its gate is still applied**. A statement about one test, not about the defect | Absent, or safe to remove the gate |
| **Not reproducing** | Previously observed; not observed on the current stack; mechanism unexplained and no working reproducer. The weakest state in this document | Fixed, and not Passing either — nothing is being tested |
| **Live** | Reproducible on the current stack, every run | — |
| **Cleared** | Evidence meeting this document's own bar — **≥ 3000 steps** ([1.1](#11-where-this-stands)) — that the defect is absent | A clean run. **Almost nothing here is cleared**, and that is the honest state |
| **Not tested on this stack** | No result exists on the current pin, in either direction. Said of defects whose test needs a configuration we will not run — in practice, buffer ops on | Passing. Nothing has been observed |
| **Retired** | Said of *evidence*, never of a defect: the configuration the claim rested on is no longer tested (batch 8) | The defect went away |

Current state, one line each:

| Defect — what it actually is | State | Last tested | Gate |
|---|---|---|---|
| **[3] int32 row-offset overflow.** `rows * stride_y` in `_ln_mul_dropout_fwd_rng` is computed in 32 bits, so at `stride_y = 3D = 1536` it wraps at row **1,398,102** and the store lands **4.29 GB below `Y`** — inside another live allocation, so no fault, just poisoned tensors. The bounds mask is computed on the *unwrapped* row id and does not save it → [row 15 of §4](#4-changes-applied-to-master), [handoff](lnmuldropout_i32_handoff.md) | **Fixed** | **2026-09-10** — standalone reproducer (`N=1398101` PASS / `N=1399126` FAIL at the predicted row) plus 2 × 600 steps clean | row 15 — the fix itself |
| **[5] Block pointers removed upstream.** `tl.make_block_ptr` / `tl.advance` were deleted in favour of the tensor-descriptor API, which lowers to Hopper TMA and is not available on HIP. 34 call sites would not compile → [6.3](#63-triton-language-api-port-5) | **Fixed** | **2026-09-11** — every run this session compiles the ported kernels | row 5 — the pointer port |
| **[1] Buffer-op node hang.** The buffer-op pass group wedges the **whole node** on kernels whose pointer args straddle the 2 GiB narrowing cutoff, so one kernel body mixes 32-bit `buffer_*` and 64-bit `global_*` addressing. Recovery is a reboot plus `modprobe amdgpu` → [9.2](#92-defects-by-version) row 1, [8.1](#81-hardware-and-recovery) | **Live**, re-confirmed by dispatch | **2026-09-11 — dispatched, and it hung.** First dispatch-level check on this stack; all prior buffer-op evidence was Triton 3.6.0. Compile-only the same day still shows hybrid addressing (16 `buffer_store_b128` + 8 `global_load_b128` + 8 `global_store_b128`; the knob-off control is 8 + 24, all `global_`, and differs in `s_endpgm` count 1 vs 2). Cost: a wedged node and a reboot | row 1, applied |
| **[6] Extended-VGPR attention-backward fault.** `_hstu_attn_bwd` faults on a real address mid-run, at a nondeterministic batch index (0–527 observed). An address-critical lane offset held in extended VGPR `v257` across the WMMA body comes back corrupted on this A0; the first final-`DV` `global_store_b128` faults on the amplified address. **Not spilling** → [6.2](#62-root-cause-of-the-attention-backward-fault-6--4d) | **Live** at `BLOCK_N=128` | **2026-09-04** — not re-tested since; row 2 has been in place throughout, and testing it means unpinning `BLOCK_N` | row 2, applied |
| **[7] High-VGPR kernel + concurrent streams.** A high-occupancy, extended-register kernel faults outside its own buffers when run against concurrent streams — `--occupancy 16 --streams 4`, 3 of 6 runs. Found while building the [3] reproducer, and the controls rule out [6] → [9.3](#93-row-7-follow-up-coexec-scheduler-root-cause), [5.2](#52-defect-detail) | **Live**, 3 of 6 | **2026-09-10** — 3 of 6, found on this stack | none — newly found |
| **[4c] Fused-RNG backward miscompile.** 3.8.0 miscompiles the fused layer-norm × mul × dropout **backward**, but only with dropout actually on — which is why an earlier shape sweep exonerated the kernel, having used `dropout_ratio=0.0` → [9.2](#92-defects-by-version) row 7 | **Passing** | **2026-09-11** — `backward: PASS`, zero mismatches on `dx`/`du`/`dweight`/`dbias`/`y` and all three masks | row 4, **still applied** |
| **[2] Buffer-op silent wrong results.** The *same* pass group as [1], different symptom: 20 / 7640 elements corrupted in the position kernel. No fault, no hang — wrong numbers → [9.2](#92-defects-by-version) row 2 | **Not tested on this stack** | **Never, on this stack.** The one passing result is **2026-09-01 on Triton 3.6.0** (`maskfix_bufops1.log`), not the current pin — an earlier `2026-09-04` date in this document was wrong. Checking it needs buffer ops **on**, i.e. another wedge | covered by row 1 |
| **[4] Pipelining `nan`.** 3.8.0 was seen to miscompile software-pipelined loops: at `num_stages ≥ 2`, kernels faulted at `gstep=0` or returned 3.3e+28 garbage; at 1 they were bit-comparable to 3.6.0. All of that evidence was batch 8 → [9.5](#95-the-batch-8-pipelining-nan-4-pre-fix-stack) | **Not reproducing** | **2026-09-11** — 600 steps unclamped at batch 1024, `EXIT=0`, finite, `num_stages` verified 3. n=1 | row 3, applied — but see [5.3](#53-cost-of-the-workarounds-still-in-place): kept for throughput now, not for [4] |
| **[1b] Hangs that survive the knob.** Same `GPU Hang` signature as [1] but with `AMDGCN_USE_BUFFER_OPS=0` already set — so row 1 avoids the two *known* hangs, not the class. Seen on an fp16 position test and on a plain read of the embedding tables → [9.2](#92-defects-by-version) row 4 | **Live**, not gated by row 1 | **2026-09-01** — not re-tested. Its matched control needs buffer ops on | none |

### 5.1 Re-test queue

**Two questions, one table.** What is worth running again now that [3] is fixed,
and — since MI450 is supposed to *beat* MI350, not merely complete — **which of
these follow-ups actually give throughput back?** The table is ordered by the
second question: **rank 1 is the largest recovery available, rank 9 the
smallest.** Every defect id is listed, including the Passing ones — a defect whose test
passes is exactly a candidate for *deleting a gate*, which is why they are not
filed separately.

**Set the scale first, or this table will mislead you.** Every workaround here is
second-order against the gap in [8.3](#83-performance): the matrix cores are
idle, measured bf16 GEMM is only **1.46× fp32**, and HFU reads **1.1–1.9 %**.
Reverting *every* row below still leaves the model running on the vector path.
**MI450-versus-MI350 throughput is a BLAS-maturity question, not a
workaround-revert question** — this table is worth a few percent e2e and up to
1.54× on one kernel, against a ~50× denominator that none of it touches.

**Rank is not do-order.** The two largest recoveries are both *blocked* — rank 1
on a reboot that should not be spent, rank 2 on an upstream LLVM fix. The first
thing to actually run is **rank 3** ([4c], already Passing, minutes), then the
validation items at ranks 8 and 9, which buy no throughput but are what this
document's claims rest on.

Read the **Evidence** column second. Everything except the validation rows rests
on measurements taken **2026-09-04, before the [3] fix**. [4]'s rests on batch 8,
which is **no longer a configuration this project tests** — so that evidence is
retired rather than refreshed, and [4] has to be re-established at batch 1024 or
not at all. Batch 1024 is the only configuration any live claim may rest on.

| Perf rank | Item | Evidence it rests on | Re-test | A clean result buys | Perf impact if reverted | Cost |
|---|---|---|---|---| --- |---|
| **1** | **[1] / [1b] buffer-op hang** | **2026-09-11: dispatched on this stack and it hung** — `GPU Hang`, `MES(6,0)`/`MES(7,0) failed to respond to msg=REMOVE_QUEUE`, node wedged and **still wedged** — a reboot plus `sudo modprobe amdgpu` is owed before any further GPU work. Compile-only still shows hybrid addressing. First dispatch-level evidence on the `7ff97e3109` pin | **Two different actions, and only one is safe.** (a) *Confirm the defect*: `--compile-only` re-check of the codegen precondition — safe, and until 2026-09-11 all that had ever been done. (b) *Revert the workaround*: run with `AMDGCN_USE_BUFFER_OPS=1` — **this is where the throughput is.** (b) was run on 2026-09-11 and **wedged the node**, so the defect is Live on this pin and the knob is confirmed load-bearing — but the *throughput* half is still unmeasured, because the run hung before producing a number. Do not run (b) again on this host | **Depends which action.** (a) buys little: it confirms the precondition is still visibly present in the emitted code, which is already assumed. (b) buys **the perf number in the next column, which nobody has** — and, if clean, the right to delete row 1 outright. The knob is **not** free; that it looks free is an artefact of never having measured it | **Potentially the largest single win, and the least knowable.** Buffer ops are a whole pass group — `canonicalize_pointers` → `convert_to_buffer_ops` → `optimize_buffer_op_ptr` — turned off for **every kernel in the model**, not one tile in one backward. Nothing else in this table has that scope. **Unmeasured, and on this host unmeasurable:** the A/B needs the dispatch that hangs ([row 1 of §4](#4-changes-applied-to-master) says the same) | **a reboot** — incurred 2026-09-11 for the confirmation and **not yet paid**; the node is wedged as of this writing. Paying it again buys nothing new until an LLVM-side fix lands, which is why the largest possible win is still the one not to chase |
| **2** | **[6] extended-VGPR fault** | Root-caused, MIR reproducer upstream | Nothing local until an LLVM fix lands; then unpin `BLOCK_N` and run `repro_gfx1250_attn_bwd.py` | Deleting row 2 and its measured **1.54× kernel slowdown** ([5.3](#53-cost-of-the-workarounds-still-in-place)) | **+1.54× on the attention-backward kernel** — unpinning `BLOCK_N` to 128 recovers 11.22 ms → 7.30 ms. The largest quantified win available | ~15 s once there is something to test |
| **3** | **[4c] fused RNG** — **Passing**, gate still applied | `backward: PASS`, **re-confirmed 2026-09-11** (zero mismatches on `dx`/`du`/`dweight`/`dbias`/`y` and all three masks) | Re-run the fused-RNG check, then drop row 4's gate behind it | Deleting row 4. **See the interaction note below — this one also touches [3]** | **The only unblocked removal with a plausible gain** — ranks 1 and 2 are both larger and both blocked. Dropping row 4 restores the fused RNG kernel — fewer launches and less traffic — but the size is unmeasured, so this rank is an upper bound on belief, not a number | minutes |
| **4** | **Only the autotuner's chosen config has run clean** | Both 600-step runs, whatever config was selected | `--sweep-config` over all 15 autotune configs | Coverage of the other fourteen. `key=["BLOCK_D"]` makes codegen identical across batch sizes, so a clean run at one config is **not** evidence for the rest | **None directly**, but it is the one item that could *find* a faster config: 14 of 15 have never run | ~15 × kernel time |
| **5** | **[7] high-VGPR + concurrent streams** | 3 of 6 faults, 2026-09-10, **found on this stack** | Not a re-test — it is new and needs *narrowing*, not confirming | — | **Unknown.** Today it constrains concurrent-stream use rather than throughput directly | — |
| **6** | **[2] silent wrong results** — **not tested on this stack** | `position_test::…_triton` passed with buffer ops **on** — but on **2026-09-01, Triton 3.6.0** (`maskfix_bufops1.log`), *not* the current pin. The `2026-09-04` previously recorded here was wrong | The same test plus the rest of the op suite under `AMDGCN_USE_BUFFER_OPS=1`, **`--compile-only` or on a machine you can reboot** | Evidence the *class* is gone, not one kernel. Does **not** clear [1]: that is a hang, not wrong numbers | **None on its own.** It changes evidence, not configuration; row 1's knob stays either way | minutes, but see [1]'s risk |
| **7** | **[4] pipelining** — **premise refuted, see below** | **2026-09-11: clean at batch 1024**, 600 steps, `EXIT=0`, and **2.8 % slower** unclamped. Prior `nan` evidence was batch 8 only and is retired | 2 more × 600 steps at batch 1024 to make it n=3 | **Nothing on throughput** — that is now measured and negative. Only the right to delete a workaround that is not known to be doing anything | **Negative, and measured.** Deleting row 3 costs **2.8 %** at batch 1024. There is no throughput case for this work | ~1.1 h for 2 runs |
| **8** | **Run length at batch 1024** (not a defect) | 2 × 600 steps, 2026-09-10, current stack | `SMOKE_BATCHES=3000` at batch 1024 | The doc's own ≥ 3000-step bar, met at the production batch size for the first time. Today the fix rests on the standalone reproducer, not on run length | **None** — validation only; it changes what may be claimed, not what runs | ~2.7 h |
| **9** | **`N` has never been observed** (not a defect) | Derived from `HISTORY_STRATEGY` event counts in `yambda_5b.gin:463-478`, not measured | Print `N` from inside `_ln_mul_dropout_fwd_rng` at batch 1024 | Turning the wrap-row arithmetic — 2.74 M against a wrap at 1,398,102 — from a calculation into an observation. **The whole [3] story rests on this number**, including the batch-523 crossover and the claim that batch 8 is 65× below the wrap | **None** — instrumentation | minutes |

**The [4] re-test result, 2026-09-11 — it argues for keeping row 3, not removing it.**
One 600-step run at batch 1024 with `TRITON_ALLOW_PIPELINING=1` (clamp lifted;
verified `num_stages` 3 rather than 1) finished `EXIT=0` with finite loss
throughout — no `nan`, no fault, past both step-100 and step-300 onsets that the
retired batch-8 evidence recorded.

| Batch-1024, 600 steps | `step_ms` | `local_sps` | `tflops_real/gpu` | Step 600 `train_loss` | window AUC |
|---|---|---|---|---|---|
| **Pipelining ON** (clamp lifted) | **3491.99** | 293.2 | 50.0 | 0.13833 | 0.49938 |
| Clamped, run 1 | 3397.41 | 301.4 | 51.4 | 0.13826 | 0.49944 |
| Clamped, run 2 | 3403.39 | 300.9 | 51.3 | 0.13826 | 0.49940 |

Two conclusions, and the second is the one that matters:

1. **[4] did not reproduce at batch 1024.** n=1, so this does not clear it — but
   combined with the retirement of the batch-8 evidence, [4] now has **no live
   evidence at any configuration this project tests**.
2. **Lifting the clamp is 2.8 % *slower*, not faster** (3492 ms against 3397 and
   3403 ms, the two clamped runs agreeing to 0.2 %). This **refutes the premise
   that motivated the re-test.** Row 3 was described as the largest unpriced perf
   cost in the table; at the production batch size it costs nothing, and removing
   it is a small measured regression. The reason to keep row 3 is no longer that
   it is protective — that is unproven at 1024 — but that **deleting it buys
   nothing measurable while giving up a safety margin.** That is why [4] sits at
   **rank 7 of 9**: it is the only row whose follow-up work has a *negative*
   expected return.

**A blocker found while re-testing: the standalone reproducers no longer run.**
`scripts/repro_gfx1250_pipeliner.py` fails at compile —
`tl.make_block_ptr` has been **removed from this Triton** "in favor of the tensor
descriptor API". The repo's kernels were ported under [5] ([6.3](#63-triton-language-api-port-5));
`scripts/` was not. So [4]'s mechanism-level evidence is unreachable, and every
other `scripts/repro_gfx1250_*.py` using block pointers is presumed equally
stale. **Porting them is a prerequisite for any further [4] or [6] triage.**

**The [4c] / [3] interaction, worth deciding deliberately.** Row 4's gate is what
puts gfx1250 on `_ln_mul_dropout_fwd_rng` — the kernel [3] lived in. Every other
arch takes the legacy `_ln_mul_dropout_fwd`, whose row offsets were **already
widened**, which is why [3] was arch-correlated without being an arch bug. So
dropping row 4 would move this arch off the kernel that had the overflow. That
does **not** make row 15 redundant — the separated path has to be correct for
anyone who needs it, and the widening is what gets upstreamed — but rows 4 and 16
should be evaluated in one go rather than separately.

### 5.2 Defect detail

Evidence behind the table above. Dates and configurations as recorded; where an
entry predates the [3] fix, the table says so.

- **[7] NEW: high-VGPR Triton kernel + concurrent streams faults outside its own
  buffers.** Found while building the [3] reproducer, and **it is not [6]** — the
  controls rule that out. `repro_gfx1250_extvgpr_value.py --occupancy 16
  --streams 4` faults 3 of 6 runs. Rates over 6 repeats × 200 iterations:

  | Configuration | VGPRs | ext regs | PASS | FAULT | wrong numbers |
  |---|---|---|---|---|---|
  | baseline, occ 16 × 4 streams | 602 | 283 | 3 | **3** | 0 |
  | `-mattr=-1024-addressable-vgprs` | 256 | **0** | 1 | **5** | 0 |
  | low pressure (`--nacc 2 --block-n 64`) | 234 | **0** | 6 | 0 | 0 |
  | occ 16 × **1** stream | 602 | 283 | 6 | 0 | 0 |
  | occ **4** × 4 streams | 602 | 283 | 6 | 0 | 0 |
  | plain `torch.mm`, 4 and 8 streams | low | 0 | 6 | 0 | 0 |

  **Why it is not the extended-VGPR hazard:** [6]'s own fix
  (`-mattr=-1024-addressable-vgprs`, which removes every extended register)
  makes it *worse*, 5/6 vs 3/6 — a fix cannot raise the failure rate of the
  mechanism it fixes. And the faulting address, `0x774fa86fc000`, is ~4 GiB past
  the end of every data buffer (`0x774e86a00000`–`0x774ea7200000`); a corrupted
  lane offset would land in or near a buffer. Both high register pressure *and*
  stream concurrency are necessary and neither alone suffices. Points at
  scratch/private-memory backing under concurrent dispatch — **a runtime
  concern, not codegen.** Details in
  [`gfx1250_extvgpr_handoff.md`](gfx1250_extvgpr_handoff.md).
- **[4] pipelining: does not reproduce at batch 1024 (2026-09-11).** One
  600-step run with the clamp lifted — `num_stages` verified 3, not 1 — finished
  `EXIT=0`, finite throughout, and **2.8 % slower** than clamped
  ([5.3](#53-cost-of-the-workarounds-still-in-place)). n=1, so [4] is **not
  cleared**; but its only positive evidence was at batch 8, which is retired, so
  **[4] currently has no live evidence at any tested configuration.** Row 3 stays
  — not because it is proven protective, but because removing it is a measured
  regression for no observed benefit. Pre-fix batch-8 narrative archived to
  [9.5](#95-the-batch-8-pipelining-nan-4-pre-fix-stack).
- **[4]'s mechanism-level reproducer still cannot run** — confirmed again
  2026-09-11. `scripts/repro_gfx1250_pipeliner.py` dies at compile:
  `tl.make_block_ptr` is **removed from this Triton**. Repo kernels were ported
  under [5]; `scripts/` was not. Already owed in
  [row 3's follow-up](#4-changes-applied-to-master); restated here because it is
  what makes [4] un-triageable — with the e2e `nan` gone at batch 1024 and the
  reproducer dead, there is currently **no way to observe [4] at all**.
- **[1] buffer-op hang: Live, confirmed by dispatch 2026-09-11.** Every prior
  buffer-op result in this document was Triton 3.6.0; this is the first on the
  `7ff97e3109` pin. `--compile-only` still emits hybrid addressing — `JaggedIn`
  (4.064 GiB) and `OutA` (4.062 GiB) stay 64-bit while the three small args
  narrow, giving 16 `buffer_store_b128` beside 8 `global_load_b128` + 8
  `global_store_b128`; the knob-off control is 8 + 24, all `global_`. The two
  also differ in `s_endpgm` count (1 on, 2 off), so the pass group changes
  control flow, not only addressing. **The dispatch then hung the node** —
  `HW Exception … reason :GPU Hang`, repeated `MES failed to respond to
  msg=REMOVE_QUEUE`, `rocm-smi` reading `N/A` for clock, power and temperature —
  and cost a reboot. Knob stays load-bearing.
  Logs: `~/mi450_logs/bufop_0911/`.
  **[1b]'s matched control is still owed** and still cannot be run without a
  second wedge.
- **[6] upstream.** The LLVM hazard, plus native B0 validation. Until it lands,
  row 2 is load-bearing and `TRITON_FULL_AUTOTUNE=1` is unsafe on gfx1250.

**Cleared on this stack, and not yet acted on.** Both are priority 3 and 4 above.

- **[2] silent wrong results: not tested on this stack, and this document
  previously said otherwise.** `position_test`'s `…_triton` does pass with
  `AMDGCN_USE_BUFFER_OPS=1`, where it had reported 20/7640 wrong elements at max
  abs diff 1.38 — but that run is `maskfix_bufops1.log`, **2026-09-01 on Triton
  3.6.0**, not the current pin. No buffer-ops-on result exists on `7ff97e3109`
  for this or any other kernel. Two reasons that matters: it is one kernel, so
  the class was never covered anyway; and [1] was just shown Live on this pin by
  dispatch, so the pass group is demonstrably not benign here.
- **[4c] fused RNG passes on the current stack** (`backward: PASS`,
  re-confirmed 2026-09-11: zero mismatches on `dx`, `du`, `dweight`, `dbias`,
  `y` and all three masks). Row 4's gate is the best removal candidate, pending
  the [4c]/[3] interaction note above.

### 5.3 Cost of the workarounds still in place

What a successful re-test above would actually recover.

- **Row 2 is 1.54× slower**, now measured: `bench_attn_bwd_blockn.py` at a
  training-scale shape (8×8 heads, 13,809 tokens, `D=128`) gives **11.22 ms at
  `BLOCK_N=64` against 7.30 ms at 128**, ±0.05 ms across repeats, at 654 vs 987
  VGPRs with **no spills in either** — more evidence spilling is not what [6] is
  about. Kernel-level only; the e2e A/B is unavailable because 128 faults.
- **Row 3 costs nothing at batch 1024 — measured 2026-09-11, and the sign is the
  opposite of what was assumed.** Lifting the clamp gives **3491.99 ms/step
  against 3397.41 and 3403.39** clamped: the clamp is **2.8 % faster**, not
  slower. Both clamped runs agree to 0.2 %, so the gap is well outside run noise.
  The doc previously carried this as "the largest unmeasured perf cost" covering
  27 kernels; that characterisation was wrong and is withdrawn. Table and
  caveats in [5.1](#51-re-test-queue).

## 6. Defect and port reference for the current stack

### 6.1 Defect record

One defect had to be worked around to get the e2e run above, and its full
evidence trail is worth keeping: it is the same kernel as [4d] on AMD 3.8.0, and
the isolation work is what turned a mid-run fault with no kernel name into a
15-second standalone reproducer. Same columns as the [historical
table](#92-defects-by-version).

| # | Tested | Triton version(s) tested | Problem surfaced | Mitigation | E2e reproducer (needs dataset) | Standalone reproducer (no dataset) |
|---|---|---|---|---|---|---|
| 10 | 2026-09-04 | • `main` @ [`7ff97e3109`](https://github.com/triton-lang/triton/commit/7ff97e310935b4a79794878dbc911f9af25d38d9) (`triton-lang/triton`, reports `3.8.0`), LLVM `b010a18d`, built with `TRITON_COMMIT=7ff97e310935b4a79794878dbc911f9af25d38d9 ./scripts/build_triton_from_source.sh` on the `:gfx1250` image<br>• **env:** `AMDGCN_USE_BUFFER_OPS=0`, `TRITON_ALLOW_PIPELINING` unset ([4] clamp active), [4c] separated-RNG path active — the same defaults as the 3.8.0 rows | **[6]** **`_hstu_attn_bwd` faults on a real address, mid-run.** This is what is left once [5] no longer blocks compile — and it is [4d]'s kernel, still faulting on current upstream.<br>• **Symptom:** `Memory access fault by GPU node-2 … Reason: Page not present or supervisor privilege`, no `run_stop`.<br>• **Not a first-step fault.** It was first read as `gstep=0` because the run that found it happened to die there. Same config, six runs: batch **0, 13, 17, 55, 114, 527**. The batch index is not reproducible; that it faults is. **Any run shorter than a few hundred batches can pass by luck** — the earlier `AMD_LOG_LEVEL=3` run that looked like a Heisenbug had simply stopped at 5 batches.<br>• **It is a Triton kernel:** `HSTU_HAMMER_KERNEL=PYTORCH` completes all 4000 batches, `train_loss=0.1015`, reaching `run_stop` and eval.<br>• **Narrowed to one op, then to its backward.** Per-op bisect (`scripts/bisect_hooks/sitecustomize.py`) puts one op group on Triton and the rest on PyTorch: `ln` clean 4000, and `jagged`, `pos`, `uqvk`, `out`, `mm`, `mha` each clean 1000+ — while **`prep` (`hstu_preprocess_and_attention`) faults alone, at batch 55**. Standalone, `triton_hstu_attention_fwd` is clean for 2000 iterations and `triton_hstu_attention_bwd` faults within ~150. **`mha` looking clean does not rule out the attention kernel:** the fused `prep` autograd function calls `triton_hstu_attention_fwd`/`_bwd` directly, so the hook's `hstu_mha` override never reaches it.<br>• **Not memory pressure:** it faults with `EMBEDDING_ROW_SCALE=0.02` (~11 GB of tables, batch 17) as readily as at `0.25` (~140 GB). Peak HBM was ~140 GB of 432 GiB, never near full.<br>• **Not the allocator knob:** `expandable_segments:False` faults too (batch 114).<br>• **Not a small overrun:** over-allocating `dq`/`dk`/`dv` with a 128 KiB sentinel tail (`--layout guarded`) shows the guard untouched at the fault, and fault addresses land far outside the heap range (e.g. `0xf7a9bbb7d000`). This reads as a wild pointer, not an off-by-N past the gradient buffers.<br>• **Not the packed-uvqk stride:** the fused call site passes q/k/v as strided views of one `uvqk`, but `--layout strided` and `--layout contiguous` both fault.<br>• **Kernel unnamed by the runtime:** `AMD_SERIALIZE_KERNEL=3` never printed `kernel:`; the attribution above is from the bisect, not from the fault message.<br>• **Root-caused to LLVM machine code.** An address-critical lane offset is held in extended VGPR `v257` across the WMMA body; on this A0 lane 0 of it comes back corrupted, and the first final-`DV` `global_store_b128` faults on the amplified address. High register pressure is what exposes it — the N=128 tile reaches the gfx1250 1024-VGPR ceiling — but **spilling is not the cause**: a `waves_per_eu=2` variant carrying 5× the scratch passes 4000 iterations. See [Root cause of the attention-backward fault](#62-root-cause-of-the-attention-backward-fault-6--4d), [#2](https://github.com/raikonenfnu/training/issues/2) and [#3](https://github.com/raikonenfnu/training/issues/3). | **`BLOCK_N=64` for the pinned attention-backward config on gfx1250 + Triton ≥ 3.8** ([PR #7](https://github.com/chriscai-amd/training/pull/7)); `BLOCK_N=128` is kept on every other arch and on older Triton. With it, e2e completes 4000 batches in 5.2 min, `train_loss=0.10000` (PyTorch-kernel reference `0.1015`), eval AUC `0.5233`, `run_stop` reached, no `nan` — though at batch 8 that is only 32,000 samples, so it said nothing about [3] at the time; [3] has since been root-caused and fixed independently (row 15 of [4](#4-changes-applied-to-master)).<br>• **A workaround, not a fix.** The LLVM-level extended-VGPR hazard is untouched, so another shape that produces the same allocation can hit it again — including under `TRITON_FULL_AUTOTUNE=1`, which restores the full config list and can pick a 128 tile.<br>• Before this: `HSTU_HAMMER_KERNEL=PYTORCH` was the only thing that completed a run (4000 batches, `train_loss=0.1015`), at a large speed cost, and it is what proved the defect is in a Triton kernel. | • Do not use `IMG=:triton-main` (that tag is `58895270` / [5]). Build `7ff97e3109` into the `:gfx1250` image, then:<br>• **Run:** `AMDGCN_USE_BUFFER_OPS=0 AMD_SERIALIZE_KERNEL=3 NUM_TRAIN_BATCHES=4000 NUM_EVAL_BATCHES=2 EMBEDDING_ROW_SCALE=0.25 BATCH_SIZE=8 PROGRESS_EVERY=1 timeout 900 ./scripts/run_smoke_shrink_1gpu.sh`<br>• **Expect, with the `BLOCK_N=64` pin in place:** all 4000 batches, `run_stop`, `train_loss≈0.10`, AUC ≈ `0.523`, ~5 min.<br>• **To see the fault again**, force the old tile — edit `_get_bw_pinned_configs()` back to `BLOCK_N=128`, or set `TRITON_FULL_AUTOTUNE=1` — and expect `Page not present` somewhere in the first few hundred batches, **not** necessarily at `gstep=0`. Allow ~10 minutes; a 5-batch run proves nothing.<br>• **Start from an idle GPU.** A faulted rank stays wedged and its dataloader workers are orphaned, and both keep holding HBM, so a following run starts under artificial memory pressure and its batch index means nothing. `scripts/bisect_hooks/gpu_cleanup.sh` clears them; check `rocm-smi --showmemuse` reads 0% first.<br>• **To re-narrow:** prepend `scripts/bisect_hooks` to `PYTHONPATH` and set `HSTU_HAMMER_KERNEL=PYTORCH BISECT_TRITON_OPS=prep` (one group on Triton) or `HSTU_HAMMER_KERNEL=TRITON BISECT_PYTORCH_OPS=prep` (all but that group). `EMBEDDING_ROW_SCALE=0.02` makes each round ~2 min. | • **Script** (repo only, no dataset, no TBE, no TorchRec, no dataloader — faulted in ~15 s at `BLOCK_N=128`, passes 4000 iterations at 64):<br>`AMDGCN_USE_BUFFER_OPS=0 AMD_SERIALIZE_KERNEL=3 python scripts/repro_gfx1250_attn_bwd.py --part bwd --layout contiguous`<br>• **Verified against the workaround:** with `BLOCK_N=64` this and all three faulting controls (`--contextual 0 --no-targets`, `--layout strided`, `--layout guarded`) run 4000/4000 clean; reverting the pinned config to 128 puts the fault back at step ~125 in the same session.<br>• Loops `triton_hstu_attention_bwd` over a **fresh random jagged layout each iteration** (batch 8, per-row length in [64, 4086], bf16, `H=4`, `D=128`) and faults by iteration ~150. The variation is the point: the pre-existing fixed-shape op tests pass, and so does `repro_gfx1250_latest_main.py --phase connected`.<br>• **Controls that make it a bug report:** `--part fwd` clean for 2000 iterations; `--contextual 0 --no-targets` still faults (fastest, by iteration ~148, so neither the contextual-prefix nor multiple-targets masking path is needed); `--layout guarded` shows no out-of-bounds write into a sentinel tail; `--lengths ...` replays one fixed layout and still faults, so no single magic shape is required.<br>• **Wider stack, same commit:** `python scripts/repro_gfx1250_latest_main.py --phase steps --iters 4000` faults at step ~275 (jagged concat → position → 3 HSTU layers, varying shapes). `--phase connected` (fixed shapes, e2e-sized TBE) **passes** — which is why the fixed-shape reproducer missed this for so long.<br>• **Passing controls:** `python scripts/repro_gfx1250_fused_rng.py` and `python scripts/attn_check.py`. |

### 6.2 Root cause of the attention-backward fault ([6] / [4d])

The application fix is row 2, `BLOCK_N=64`. The defect under it was root-caused
by the fix author to LLVM machine code:
[raikonenfnu/training#2](https://github.com/raikonenfnu/training/issues/2)
(application) and [#3](https://github.com/raikonenfnu/training/issues/3) (LLVM,
with a checked MIR reproducer). It matters here because it changes what the
workaround means.

gfx1250 encodes only the low eight bits of a VGPR index, so `v1` and `v257` share
an encoded number and the mutable `S_SET_VGPR_MSB` state supplies the high bits.
LLVM keeps a work-item-derived lane offset live in **extended VGPR `v257` across
the whole WMMA body**, then consumes it in a global-address calculation. On this
A0, lane 0 of that value returns an FP32 accumulator-like bit pattern (lanes 1–31
correct), a later integer multiply amplifies it, and the **first final-`DV`
`global_store_b128`** faults. ROCgdb placed the fault there, same displacement
`0x6d0c6e1a` in all four waves. The store is the *observer*, not the producer:
TTIR, TTGIR and LLVM IR all carry the correct address expression; AMDGCN is the
first representation that differs.

Two changes remove it without touching the address formula — rematerialising the
lane offset after the final WMMA, and compiling the same LLVM IR with
`-mattr=-1024-addressable-vgprs`. Both stop the address depending on the old
`v257`.

**This corrects an earlier reading here:** spilling is *not* the cause. The two
tiles do sit in different regimes —

| Tile | LDS | VGPR | Spilled VGPRs | Scratch |
|---|---|---|---|---|
| `BLOCK_N=128` | 32768 | 944, 986, **1024, 1024** | 0, 0, 29, 33 | 0, 0, 84, 92 |
| `BLOCK_N=64` | 16384 | 649, 654, 758 | 0 | 0 |

— but a `waves_per_eu=2` variant with **2372 bytes** of scratch, 5× the spill
traffic, passes 4000 iterations. High pressure *exposes* the victim; it is not
the defect. The coexec schedule likewise increases exposure without being the
whole story: with coexec disabled the full N=128 form still faulted at step 0 in
one run, from the same `v257`.

Validation matrix, from [#3](https://github.com/raikonenfnu/training/issues/3):

| Variant | Result |
|---|---|
| Original N=128, extended VGPRs, seed 4321 | fault at step 0 |
| Identical LLVM IR, `-mattr=-1024-addressable-vgprs` | PASS, 4000 iterations |
| Late lane-offset rematerialisation, clean LLIR replay | PASS, 4000 iterations, twice |
| **N=64 application workaround (row 2)** | **PASS, reduced and full, 4000 iterations** |
| `s_nop 7` after every WMMA | **retracted** — passed once, identical repeat faulted at step 0 |
| NOPs around every `S_SET_VGPR_MSB`, `V_NOP`, `s_wait_alu` drains | fault at step 0 |

Newer LLVM does not resolve it. `85c730242e` adds the gfx1250 pre-RA WMMA
co-execution-window model and Triton `7ff97e3109` pins LLVM `b010a18d`, which
predates it — but rebuilding against post-fix `ce352942` still faults at step 0.
Relevant, insufficient.

The recommended fix is a subtarget-gated backend erratum: stop register
allocation carrying cheap, address-critical work-item values in extended VGPRs
across WMMA regions, rematerialising them after instead.
`-mattr=-1024-addressable-vgprs` is the conservative fallback, at the cost of
heavy spilling. **Native cold-boot B0 validation is owed**, to separate an A0
silicon erratum from a backend requirement for later steppings;
[#2](https://github.com/raikonenfnu/training/issues/2) has the handoff matrix.

### 6.3 Triton language API port ([5])

The **Triton language API** edits that make the HSTU kernels compile on current
upstream `main` (defect [5]). Not workarounds for [1]/[4]/[4c]/[6] — those are
runtime gates, and the kernels already compiled without them.

#### What upstream removed

On every `main` tried, including sglang's `76940ad3` pin, `tl.make_block_ptr`
raises:

```
NotImplementedError: Block pointers have been removed in favor of the tensor
descriptor API
```

`tl.advance` went with them. Upstream's replacement,
`triton.tools.tensor_descriptor.TensorDescriptor`, lowers to NVIDIA Hopper TMA
(`cp.async.bulk.tensor.*`). That path is already in the HSTU attention host
wrapper and is **not** what we turned on: `_should_enable_tma()` returns `False`
on HIP, with an extra guard because gfx950 reports capability `(9, 5)` and would
otherwise pass the Hopper check.

#### What we replaced them with

Masked plain-pointer `tl.load` / `tl.store`. The 2-D sites map one-for-one:

| Before (Triton ≤ 3.8 AMD SDK) | After (this branch; compiles on `main`) |
|---|---|
| `tl.make_block_ptr(base=X, shape=(N, D), strides=(stride_x, 1), offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))` | `rows = start_row + tl.arange(0, BLOCK_N)` and `cols = tl.arange(0, BLOCK_D)` |
| `tl.load(ptr, boundary_check=(0, 1), padding_option="zero")` | `tl.load(X + rows[:, None] * stride_x + cols[None, :], mask=(rows < N)[:, None] & (cols < D)[None, :], other=0.0)` |
| `tl.store(ptr, y, boundary_check=(0, 1))` | `tl.store(Y + rows[:, None] * stride_y + cols[None, :], y, mask=…)` |
| `ptr = tl.advance(ptr, (0, BLOCK_N))` (attention K/V tiles) | rebuild the pointer from the physical sequence offset each tile: `K + offset_kh + (seq_start + load_offs_n)[:, None] * stride_kn + …` — there is no longer a cursor to advance |

`boundary_check` + `padding_option="zero"` is the mask; `other=0.0` is the
padding. Nothing else in the kernels needed a language-level rewrite.

#### Files

26 `tl.make_block_ptr` call sites and 8 `tl.advance` call sites, all gone.
`grep make_block_ptr generative_recommenders` is empty on this branch.

- `ops/triton/triton_hstu_attention.py` — fused/cached HSTU attention forward
  and backward (`_hstu_attn_fwd`, `_hstu_attn_bwd`, and the one-block helpers).
  **Do not feed the remapped `pos_offs_*` into pointer arithmetic.** Those
  coordinates are for the causal / contextual / multi-target *mask* only; V is
  indexed at the physical sequence offset. Mixing them is a silent layout bug,
  not a compile error.
- `ops/triton/triton_layer_norm.py` — `_layer_norm_fwd`,
  `_weighted_layer_norm_fwd`, `_weighted_layer_norm_bwd_dx`,
  `_weighted_rms_norm_fwd`, `_weighted_rms_norm_bwd`.
- `ops/triton/triton_hstu_linear.py` — the fused layer-norm × mul × dropout
  kernels that used 2-D block pointers for X/Y/U.

Jagged concat/split, position embeddings, and addmm were already on plain
pointers (or on the unused Hopper descriptor path) and did not need a port.

#### Compatibility

The same sources still compile and run on AMD 3.6.0 / 3.7.1 / 3.8.0 — masked
`tl.load`/`tl.store` was never removed — so this is a **widening**, not a fork.

`scripts/attn_check.py` is the numeric check that the port matches the old
kernels. All eight deterministic BF16 cases pass forward and backward at the
original `atol=1e-5`, `rtol=0.016`, on both `7ff97e3109` and AMD
`3.8.0+git4cff872c`:

```bash
env -u TRITON_ALLOW_PIPELINING \
  AMDGCN_USE_BUFFER_OPS=0 AMD_SERIALIZE_KERNEL=3 \
  PYTHONPATH=/path/to/triton/python:. \
  python scripts/attn_check.py
```

Those cases are **fixed-shape, single-shot**, which is why they passed throughout
defect [6] and why placing it took so long — what the trainer adds is a different
jagged layout every step. Run it if these kernels are touched, but do not read it
as ruling out the backward.

One **not compile-blocking** tweak: `ops/triton/triton_position.py` uses `&` /
`|` instead of Python `and` / `or` on boolean masks, which Triton now warns on
(and will error on) inside `@triton.jit`.

## 7. Reproducers

All run in one of the four repo images, which differ only in the Triton wheel —
pick one with `IMG=`.

| Reproducer | Needs | State on the `7ff97e3109` pin |
|---|---|---|
| [`repro_gfx1250_lnmuldropout_i32.py`](../../scripts/repro_gfx1250_lnmuldropout_i32.py) | torch + Triton | **Works.** Deterministic, cannot fault — the one to hand over for [3] |
| [`repro_gfx1250_buffer_ops.py`](../../scripts/repro_gfx1250_buffer_ops.py) | torch + Triton | Works, **and hangs the node** ([1]). Run last, if at all |
| [`repro_gfx1250_extvgpr_value.py`](../../scripts/repro_gfx1250_extvgpr_value.py) | torch + Triton | Works; `--compile-only` needs no dispatch |
| [`repro_gfx1250_pipeliner.py`](../../scripts/repro_gfx1250_pipeliner.py) | torch + Triton | **Does not compile** — `tl.make_block_ptr` in 6 places, removed by [5]. A plain-pointer rewrite is owed ([5.1](#51-re-test-queue), rank 7) |
| `repro_gfx1250_fused_rng.py`, `repro_gfx1250_attn_bwd.py`, `repro_gfx1250_latest_main.py` | repo on `PYTHONPATH` | Work |

```bash
IMG=recommendation-amdprimus0815:triton-3.8.0-amd  # or :gfx1250, :triton-3.7.1-amd, :triton-main
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --security-opt seccomp=unconfined --shm-size=64g \
  -v "$PWD:/workspace/recommendation" -w /workspace/recommendation \
  -e PYTHONPATH=/workspace/recommendation \
  -e AMDGCN_USE_BUFFER_OPS=0 \
  -e HSA_COREDUMP_PATTERN=/tmp/gpucore.%p.gpu \
  "$IMG" <the command from either reproducer column>
```

- **A hang costs a reboot** plus `sudo modprobe amdgpu`, since `amdgpu` is
  blacklisted on the kernel cmdline. A *fault* is recoverable — the process dies
  and the GPU comes back. So leave hang reproducers for last and run **one test
  per process under a hard timeout**: `pytest` orders `unittest` methods
  alphabetically, so a sweep that wedges blames the wrong test.
- **`AMD_SERIALIZE_KERNEL=3` names the faulting kernel**, by making the async
  fault synchronous. Without it you get an address, and a queue wedged behind the
  fault looks exactly like a hang. `AMD_LOG_LEVEL=3` filtered to `ShaderName`
  gives the last dispatch before it.
- **A fault writes a coredump the size of allocated VRAM** (9–36 GB observed).
  `HSA_ENABLE_COREDUMP=0` does *not* suppress it, so keep
  `HSA_COREDUMP_PATTERN` pointed out of the working tree.
- **To time a tile,
  [`scripts/bench_attn_bwd_blockn.py --block-n 64|128`](../../scripts/bench_attn_bwd_blockn.py)**,
  one tile per process so a fault at 128 cannot take the 64 measurement with it.
  It asserts `_hstu_attn_bwd` launched and prints its VGPR count, because the
  easy failure is silently timing the PyTorch path: `hstu_mha` needs
  `kernel=HammerKernel.TRITON`, and the Autotuner's `.cache` holds best-configs,
  not compiled kernels — those are in `fn.device_caches`.
- **Unit tests** need `import fbgemm_gpu` before collection and
  `pip install hypothesis`. In `hstu_attention_test.py`, `test_attn` is a helper
  `pytest` mis-collects, so name the class test explicitly.
- **E2e reproducers** need the yambda-5b cache under `DLRM_DATA_PATH`. The
  batch-1024 production path is
  [`scripts/run_prod_convergence_1gpu.sh`](../../scripts/run_prod_convergence_1gpu.sh)
  ([2.2](#22-run-configuration)); the retired batch-8 path goes
  through [`scripts/run_smoke_shrink_1gpu.sh`](../../scripts/run_smoke_shrink_1gpu.sh),
  which defaults `AMDGCN_USE_BUFFER_OPS=0` and forwards `IMG`,
  `NUM_TRAIN_BATCHES`, `PROGRESS_EVERY`, `AMD_SERIALIZE_KERNEL`,
  `TRITON_ALLOW_PIPELINING` and `DEBUG_NAN_HOOKS`. Check `ps -eo pid,cmd | grep
  train_ranker` first: an interrupted run can outlive its shell, and two
  trainings sharing the GPU invalidates the result.

## 8. The node and environment

### 8.1 Hardware and recovery

1× MI450 A0 engineering sample, `gfx1250`, PCI `1002:75c1`, SKU `M4500001`, rev
`0x00` — 256 WGPs, ~432 GiB HBM, **`warp_size=32`** (RDNA, not CDNA wave64).
Software pins are in [2.1](#21-pins) and are not repeated here; note only that
the image's *stock* Triton is 3.6.0, which is why every recipe rebuilds it.

Two facts drive every change here: **one GPU** (the model must fit in one HBM)
and **wave32** (fbgemm's CDNA assumptions are wrong).

`amdgpu` is blacklisted by the kernel cmdline
(`modprobe.blacklist=amdgpu,device_dax,dax_hmem`), so run `sudo modprobe amdgpu`
after every boot — `/dev/kfd` appears ~15 s later.

**Recovery from a hang is a reboot.** The process sticks in uninterruptible `D`
state, `docker kill` returns *"did not receive an exit event"*, MODE2 reset fails
with `error code -62`, `modprobe -r amdgpu` deadlocks. The wedge is whole-device
and survives container teardown: `rocm-smi` still answers while holding VRAM with
no processes alive, and a bare 1024² bf16 matmul in a fresh container never
returns. Untried: `flr`/`bus` reset and the `umr` wave debugger.

### 8.2 Image, fbgemm and dataset

**Image** — [`Dockerfile.amdprimus0815`](../../Dockerfile.amdprimus0815). Stock
`rocm/primus:v26.3` torch has no gfx1250 code objects, so *any* kernel fails with
`device kernel image is invalid`. `amdprimus/amdprimus:0815` has gfx1250 torch
but no fbgemm/torchrec, so those are built on top (~8 min). ROCm lives in the
wheel at `site-packages/_rocm_sdk_devel`, not `/opt/rocm`, so
`ROCM_HOME`/`HIP_ROOT_DIR` must point there or configure dies with
`BUILD_ROCM_VERSION is not set`. Also needs `libtbb-dev`, `libnuma-dev`,
`libssl-dev`, `ninja-build`.

**Three fbgemm patches** — `Fbgemm.cmake` omits the fp16/bf16 convert sources;
host/device **warp-size disagreement** (host assumed wave64); codegen also
hardcodes wave64. With these, TBE is **bit-exact** against CPU.

**Embedding shrink** — `EMBEDDING_ROW_SCALE=0.25`: 560.04 GiB of fp32 reference
tables against one ~432 GiB GPU, cut to 140.01 GiB
([2.2](#22-run-configuration)).

**Dataset** — use the **MLCommons prepared download** (README Option A): 227.41
GB, 19 files, md5/sha256 verified. Do **not** self-preprocess; the cache build
needs ~190 GB RAM and OOM-killed this 251 GB box.

### 8.3 Performance

**Measured on AMD Triton 3.7.1 at batch 128, not on the working stack** — these
numbers predate `7ff97e3109` and the [3] fix and have not been re-taken at batch
1024. They are kept because the *denominator* analysis below is what matters and
is stack-independent. Config: gin-default depth 3, `MAX_SEQ_LEN=4096`,
`EMBEDDING_ROW_SCALE=0.25`, `BATCH_SIZE=128`, buffer ops off — measured in the
finite region, while the loss was still converging. Step 50 is warmup (1920 ms, carrying compile and autotune).
Steady state is **~322 ms/step, ~395 samples/s, `tflops_real` 35–61, HFU
1.1–1.9%**. HFU tracks `fill` (29–49%) rather than being a fixed number, because
`fill` is the fraction of the dense-equivalent work the jagged batch actually
contains.

**Read `hfu`, not `mfu`.** `num_flops_per_sample` is a *static dense* estimate
(315.7 GFLOP/sample at 3 layers) while jagged attention performs only the `fill`
fraction of it, so `mfu` (3.7–4.2%) describes work that is never executed.
`tflops_real` is the honest count and `hfu` its utilization.

HFU is against **3150 TFLOP/s = 90% of MI450's 3500 TFLOPS dense BF16 matrix
peak**. Two notes on that denominator. `get_gpu_peak_flops` matched nothing and
silently fell back to **MI350X's 2300 TFLOPS**, because these parts report the
generic name `AMD Radeon Graphics` — every earlier perf line in this repo's logs
is normalized against the wrong chip. It is now keyed on `gcnArchName`. And the
silicon peak is deliberate, not the ~75 TFLOP/s this node reaches today:
measured bf16 GEMM is only **1.46× fp32** (67.8 vs 46.5 TF at N=8192), so the
**matrix cores are idle** and everything runs on the vector path at ~1.1 GHz
against a 2.4 GHz max. Normalizing against measured throughput would conflate
model efficiency with BLAS immaturity and shift meaning as kernels improve.

## 9. Historical: earlier Triton versions

**Reference only.** [The working stack](#2-the-working-stack) is `7ff97e3109`
with the fix set in [4](#4-changes-applied-to-master). Nothing in this section is
a current result: 3.6.0, AMD 3.7.1 and AMD 3.8.0 each fail earlier than
`7ff97e3109` and none is a recommended path. Kept for when you are handed one of
these images, bisecting a regression, or writing up a defect for the Triton team.

All are in the Triton path, none in the benchmark's math. **Ordered by Triton
version, oldest first**, so it answers "what breaks on the version I am about to
run". Only the Triton wheel was ever swapped (`--no-deps`), holding torch, ROCm
and driver constant, so Triton is the only variable.

A defect spanning versions is listed once. The `#` column is row position and
renumbers on re-sort, so **cite the bracketed id, not the row number** — row 6 is
defect [4]. Those ids are referenced from code comments in `common.py`,
`ops/utils.py` and `dlrm_v4/train/utils.py`. **Tested** is the UTC day the row's
result was recorded.

Reproducers split by cost: **e2e** needs the yambda-5b cache and a training run,
**standalone** is everything runnable without a dataset. Both cells are always
filled, so a "none" is stated rather than inferred.

### 9.1 Images tested

All four are the same
[`Dockerfile.amdprimus0815`](../../Dockerfile.amdprimus0815) build — same torch,
ROCm, fbgemm and torchrec — with **only the Triton wheel swapped** (`--no-deps`).
Select with `IMG=`; the runner defaults to the 3.6.0 image. For [the working
stack](#2-the-working-stack), rebuild Triton on top of `:gfx1250` rather than
picking one of these.

- **`recommendation-amdprimus0815:gfx1250`** — Triton `3.6.0+rocm7.14.0` (AMD SDK,
  as shipped in the base image). Runner default. `nan` by step 50 ([3]).
- **`recommendation-amdprimus0815:triton-3.7.1-amd`** — Triton
  `3.7.1+git0263a6a6.rocm7.14.0`. Best of the older set: trains to step 2150,
  then [3].
- **`recommendation-amdprimus0815:triton-3.8.0-amd`** — Triton
  `3.8.0+git4cff872c.rocm7.14.0`. Needed [4] and [4c] to complete even a
  20-batch run, then [4d] stops it a few hundred batches in. With the current fix
  set it also passes 4000 batches
  ([#2](https://github.com/raikonenfnu/training/issues/2)).
- **`recommendation-amdprimus0815:triton-main`** — Triton built from `main` @
  `58895270`. Will not compile the kernels *without* the pointer port ([5]).
  (`docker commit` of a `sleep`-kept container inherits that `ENTRYPOINT`, so this
  tag resets it back to `bash`.)

Base images, for reference: `amdprimus/amdprimus:0815` is what the repo image is
built `FROM`, and `amdprimus/amdprimus:gfx1250-20260831` was evaluated and
rejected — its `libtriton.so` is byte-identical to `:0815`, so a newer image is
not a newer Triton.

### 9.2 Defects by version

| # | Tested | Triton version(s) tested | Problem surfaced | Mitigation | E2e reproducer (needs dataset) | Standalone reproducer (no dataset) |
|---|---|---|---|---|---|---|
| 1 | 2026-08-17 | **every version tried**<br>• `3.6.0+rocm7.14.0` (AMD SDK, in `amdprimus:0815`) — launch-confirmed<br>• `3.8.0` (PyPI `v3.8.0`) — launch-confirmed<br>• `3.8.0` @ upstream `76940ad3` — launch-confirmed, and a **regression**: it also hangs `position_test::…_large_tensor`, which passes on 3.6.0<br>• `3.7.1+git0263a6a6` and `3.8.0+git4cff872c` (both AMD `repo.amd.com/rocm/whl-multi-arch/`) — **assumed** affected: byte-identical hybrid codegen, launch not run<br>• `amdprimus:gfx1250-20260831` has a byte-identical `libtriton.so` to `:0815`, so a newer image is not a newer Triton<br>• **env to hit it:** `AMDGCN_USE_BUFFER_OPS=1` (the backend default) | **[1]** **GPU hang, node unrecoverable.**<br>• **Symptom:** `HW Exception by GPU node-2 … reason :GPU Hang`, 100 % GFX / 0 % HBM, and recovery is a reboot.<br>• **Trigger:** a kernel whose pointer args straddle the **2 GiB narrowing cutoff**, so one body carries both 32-bit `buffer_*` and 64-bit `global_*` addressing. `HIPBackend.get_tensor_specialization` sets `tt.pointer_range=32` when `use_buffer_ops and arg.untyped_storage().size() <= 2**31 - 1`; in the failing kernel `JaggedIn` (4.064 GiB) and `OutA` (4.062 GiB) stay 64-bit while `OutB` and the offsets narrow. Confirmed hybrid in asm: 16 `buffer_store_b128` + 8 `global_load_b128` + 8 `global_store_b128`.<br>• **Not an infinite loop:** every config compiles, one `s_endpgm` each, and Tarjan SCC over the CFG finds no cycle.<br>• **Masking is by address, not predicate:** masked lanes get sentinel offset `0x80000000`, expected to be dropped for exceeding `num_records=0xffffff`.<br>• **Crossing 2 GiB is necessary but not sufficient:** a *passing* dense bf16 case is also above the cutoff but takes different `IS_DENSE_*` branches.<br>• **The real workload hits it:** at batch 128 `concat_2D_jagged` crosses 12× (largest arg 2135 MiB, 1.04×) and hangs; at batch 32 the largest is 552 MiB (~3.7× under), so small runs never see it. | **`AMDGCN_USE_BUFFER_OPS=0`**. Disables `canonicalize_pointers` → `convert_to_buffer_ops` → `optimize_buffer_op_ptr` together, so the A/B implicates the **group**, not one transform. `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1` is the other knob on the group; untested | • **Not needed** — the standalone script and the unit tests hit it in seconds, at far less cost than a wedged node.<br>• For the record it does appear in training: at `BATCH_SIZE=128` with `AMDGCN_USE_BUFFER_OPS=1`, `concat_2D_jagged` crosses the cutoff 12× and hangs. At `BATCH_SIZE=32` the largest arg is ~3.7× under the cutoff, so small runs never see it. | • **Script** (torch+triton only, no repo imports):<br>• `AMDGCN_USE_BUFFER_OPS=1 python scripts/repro_gfx1250_buffer_ops.py` → the launch never returns and the node wedges with `HW Exception by GPU node-2 … reason :GPU Hang`.<br>• `AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_buffer_ops.py` → prints `launch completed without hanging` and `correctness spot-check: PASS`. That pair is the whole result.<br>• `--compile-only` never dispatches, so it is safe on a machine you cannot reboot; it prints each argument against the 2 GiB cutoff and the emitted `buffer_*`/`global_*` counts. The kernel is verbatim, the autotuner is removed and the config is pinned to `BLOCK_N=2, num_warps=1`, so there is exactly one dispatch.<br>• **Unit test, same hang**, inside `do_bench` autotuning of the **backward**: `python -m pytest generative_recommenders/ops/tests/jagged_tensors_test.py::JaggedTensorsTest::test_concat_2D_jagged_large_tensor` → `split_2D_jagged_multirow`.<br>• **Unit test, on `76940ad3` only:** `python -m pytest generative_recommenders/ops/tests/position_test.py::PositionEmbeddingsTest::test_add_timestamp_positional_embeddings_triton_large_tensor` → `_add_embeddings_bwd_kernel` @ `BLOCK=128, num_warps=8, num_stages=4`. |
| 2 | 2026-08-17 | **every version tried** — same set as [1]<br>• **env to hit it:** `AMDGCN_USE_BUFFER_OPS=1` (the backend default) | **[2]** **Silent wrong results, kernel completes.**<br>• **Symptom:** 20 / 7640 elements wrong, max abs diff 1.38 against tol 1e-5, with no fault, no hang and a normal exit.<br>• **Reading:** a few badly-wrong lanes is the signature of a mis-lowered mask or bounds check, consistent with the address-sentinel masking in [1]. | same knob | • **None.** Wrong-but-finite values in this kernel have no separately visible e2e signature; the unit test is the whole story. | • **Unit test:** `python -m pytest generative_recommenders/ops/tests/position_test.py::PositionEmbeddingsTest::test_add_timestamp_positional_embeddings_triton`<br>• **Expect:** an `assert_close` failure reporting `Mismatched elements: 20 / 7640` and greatest absolute difference 1.38 against a 1e-5 tolerance.<br>• **Cost:** seconds, the kernel completes, and the GPU is **not** wedged — so this is the risk-free way to see the buffer-op defect. |
| 3 | 2026-09-01 | • `3.6.0` — `nan` by **step 50**<br>• `3.7.1+git0263a6a6` (AMD) — finite and tracking PyTorch to step 2150, `nan` from **step 2200**: a ~44× delay, **not a fix**<br>• `3.8.0+git4cff872c` (AMD) — not assessable, dies first ([4d])<br>• `main` @ `58895270` — not assessable, will not compile ([5])<br>• `main` @ `7ff97e3109` — with [6] worked around (`BLOCK_N=64`), finite through 4000 steps at batch 8, `train_loss=0.10000`, no `nan`. That batch-8 run did **not** clear it: 3.7.1's onset was step 2200 at batch **128** (281,600 samples), and 4000 × 8 is 32,000 samples, ~9× short.<br>• **2026-09-10 — the owed batch-128 comparison, `main` @ `7ff97e3109`:** two independent 4000-step batch-128 runs (`~/mi450_logs/nan128/long{1,2}.log`), **both reached `run_stop` with zero non-finite loss** — final `train_loss=0.15350` and `0.15343`. Both satisfy the ≥3000-step rule, and both walked through 3.7.1's onset window finite (step 2150 → 0.13860, **2200 → 0.13925**, 2250 → 0.13541, 2300 → 0.14369). Env matched the doc's stack: driver `amdgpu 7.1.1`, `AMDGCN_USE_BUFFER_OPS=0`, `HSTU_HAMMER_KERNEL=TRITON`, `EMBEDDING_ROW_SCALE=0.25`.<br>&nbsp;&nbsp;**What this does and does not establish.** Against the doc's own 3-in-4 base rate, two clean runs is `0.25² ≈ 6%` — evidence, not proof, and n=2 cannot distinguish "fixed" from "rate dropped". [3] is **not observed** on this pin; it is not demonstrated absent. The onset also migrated once before (step 50 → 2200, ~44×) without being fixed, so a further migration past 4000 steps is the live alternative and would need a longer or higher-batch run to exclude.<br>&nbsp;&nbsp;Note the eval AUC on these runs is ~0.483, which is **not** a symptom: 4000 × 128 = 512k samples of a 5B-event corpus, one of 299 windows, with LR still inside a 24,000-step warmup — no opportunity to converge.<br>• **2026-09-10 — batch 1024 DOES reproduce on this same pin: `nan` at step ~190 (~10 min).** 18 finite points steps 10–180 (flat 0.1377–0.1398), `nan` at 190/200, `run_stop status="aborted"`, no fault or traceback. So on `7ff97e3109` the trigger has **moved up in batch size** rather than disappeared, and batch 1024 is now the cheapest known [3] reproducer. Consistent with `AUTOTUNE_B = prev_power_of_2(B)` giving 128 and 1024 separate autotune entries. n=1 — repeat before trusting the onset step. Log: `~/mi450_logs/prod_b1024_smoke.log`.<br>&nbsp;&nbsp;**Superseded:** this is the pre-fix state. Root-caused and fixed on 2026-09-10 — see row 15 of [4](#4-changes-applied-to-master); batch 1024 is now clean 2/2 over 600 steps ([3.1](#31-the-confirming-runs)).<br>• **env:** `AMDGCN_USE_BUFFER_OPS=0`, `HSTU_HAMMER_KERNEL=TRITON` (both runner defaults) | **[3]** **`train_loss=nan`, and it stays.**<br>• **Symptom:** the loss goes `nan` mid-run and never recovers.<br>• **How it reaches the loss:** via **poisoned embedding tables** — at the first non-finite forward output, 80 params are already non-finite, all embedding tables. So a Triton *backward* emits non-finite gradients, the TBE update writes them in, and every later forward inherits them.<br>• **Not divergence:** `lr≈4e-9` cannot diverge, but one NaN gradient poisons a weight permanently, since `NaN × 4e-9` is still `NaN`. Init is clean (it is shared with the passing PyTorch run).<br>• **A nondeterministic corruption, not a fixed miscompile:** it merely *moves* from step 50 to step 2200 across compiler versions, which points at an under-masked store or a race — codegen changes how often it is hit, not whether. | None. Triton 3.7.1 buys ~12 min. `HSTU_HAMMER_KERNEL=PYTORCH` avoids it entirely — finite and stable, 0.13847→0.14034 over 250 steps — but is slower, and is what proves the defect is in a Triton kernel rather than the model, data or optimizer | • **This is the only way to see it.**<br>• **Run:** `IMG=recommendation-amdprimus0815:gfx1250 NUM_TRAIN_BATCHES=3000 PROGRESS_EVERY=1 ./scripts/run_smoke_shrink_1gpu.sh`, then watch `train_loss` in the log.<br>• **Expect:** `train_loss=nan` by **step 50** on 3.6.0 (~4 min); not until **step 2200** on 3.7.1 (~12 min), so a 250-step run is not enough to clear a version — 3.7.1 looked fixed at 250, and only the 30-min run exposed it. **Any future claim that a version is clean needs ≥3000 steps.**<br>• **To find the kernel that produced it:** add `DEBUG_NAN_HOOKS=1`. It names the first module emitting a non-finite value, and flags the kernel that *manufactured* the bad gradient (finite `grad_output` in, non-finite `grad_input` out, logged `PRODUCER`) as against those merely propagating it — meaningful only on the **first** poisoned backward. `=2` also sweeps every param pre-forward; it is opt-in because that sweep has itself wedged the GPU ([1b]). | • **None — no unit test reproduces it.** That is the core difficulty: reaching it takes thousands of training steps, so every hypothesis costs minutes rather than seconds. |
| 4 | 2026-09-01 | • `3.6.0`<br>• **env:** `AMDGCN_USE_BUFFER_OPS=0` — i.e. it hangs *despite* the [1] mitigation | **[1b]** **Hangs that survive the knob.**<br>• **Symptom:** the same `GPU Hang` signature as [1], but with `AMDGCN_USE_BUFFER_OPS=0`. So the mitigation avoids the two *known* hangs, not the class.<br>• **Seen on:** an fp16 position test, and a plain read of the embedding tables. | none known | • **None dedicated.** The embedding-table read that wedged was `DEBUG_NAN_HOOKS=2`'s pre-forward param sweep inside an e2e run, which is why that mode is opt-in ([3]). | • **None — not isolated.**<br>• What is owed is a matched control: the fp16 position test and the embedding-table read, each run with `AMDGCN_USE_BUFFER_OPS=1` and `=0`, one process per run, so the knob's effect on *this* class is measured rather than assumed. |
| 5 | 2026-09-01 | • `3.7.1+git0263a6a6` (AMD)<br>• **env:** `AMDGCN_USE_BUFFER_OPS=0` | **[4b]** **`3.7.1` attention is marginally out of tolerance — on the version we actually train on.**<br>• **Symptom:** the attention unit test fails at clean `HEAD` with no local edits, so it is pre-existing: greatest absolute difference **1.38e-05 against a 1e-05 tolerance** (12 / 63104 elements) on one hypothesis case, and 2.40e-05 (1108 / 259584, 0.4 %) on another. The reported `inf` relative difference is only a zero reference.<br>• **Precision-scale, not corruption:** contrast 3.8.0's 3.3e+28 garbage and 3.6.0's clean pass. 3.7.1 shifts instruction selection just enough to drift a few elements past a tight tolerance.<br>• **Open question:** benign fp drift, or the leading edge of [3]? It is the only known numeric discrepancy on the version we train on, which makes it the natural next thread to pull for the `nan`. | none; magnitudes are at tolerance scale, and the e2e run tracks PyTorch to step 2150 | • **Nothing specific to run.** The 3.7.1 e2e run tracks PyTorch to step 2150 before [3] lands, so this drift has no separately visible e2e signature — which is exactly why it is still an open question whether it is benign. | • **Harness:** `python scripts/attn_check.py` on the `:triton-3.7.1-amd` image — 10 s, deterministic, and it reports forward and backward separately, so the drift is visible instead of hidden behind an `assert_close` abort.<br>• **Expect:** backward `dk` ~9e-05 and `dv` ~3e-05 on the largest case, against ~1e-06 for the same case on 3.6.0 — roughly 100× looser, and reproducible run to run.<br>• **Stock unit test, same defect:** `python -m pytest generative_recommenders/ops/tests/hstu_attention_test.py::HSTUAttentionTest::test_attn_triton` fails with greatest absolute difference 1.38e-05 against a 1e-05 tolerance, in ~94 s. |
| 6 | 2026-09-01 | • `3.8.0+git4cff872c` (AMD)<br>• **env:** `AMDGCN_USE_BUFFER_OPS=0`, and `TRITON_ALLOW_PIPELINING=1` now that `clamp_num_stages` is in — without that override the defect is suppressed | **[4]** **`3.8.0` computes wrong addresses in HSTU attention.**<br>• **Symptom:** `Memory access fault by GPU node-2 … on address 0x2000. Reason: Page not present or supervisor privilege` at `gstep=0`, before step 1. The address **varies run to run** (`0x2000`, `0x3000`) and is a near-null base plus a block offset, not a fixed bad pointer. **Recoverable** — unlike the hangs, the GPU came back clean with no reboot, which also makes it a far better bug report than a wedge.<br>• **Attribution:** `AMD_SERIALIZE_KERNEL=3` makes the async fault synchronous — `_weighted_layer_norm_fwd` autotuning *completes* (3.36 s, best `BLOCK_N=4, num_warps=2`), then the **attention** kernel is traced, then it faults on its first launch.<br>• **Same defect on mapped memory:** the unit test shows it as 485 / 1280 elements wrong (37.9 %) with greatest absolute difference **3.3e+28** — the magnitude of garbage, not of a precision error. Either way the kernel is reading from wrong addresses.<br>• **Root cause: 3.8.0 miscompiles the *software-pipelined* attention loop.** With `num_stages >= 2` it faults or returns garbage; with `num_stages == 1` it is bit-comparable to 3.6.0.<br>• **Autotuning is not involved:** the forward is pinned on HIP to one config (`BLOCK_M=128, BLOCK_N=32, matrix_instr_nonkdim=16, kpack=2, num_stages=2, num_warps=8`), so exactly one config faults. Sweeping it, every failing combination carries `num_stages=2` (`m=32/64/128`, `warps=4/8` all fail with it), every `num_stages=1` combination passes, and dropping the AMD-specific `matrix_instr_nonkdim`/`kpack` knobs changes nothing — they are **not** implicated.<br>• **Block pointers are not the cause**, which retracts the earlier "transposed block pointer" root cause. With `num_stages=1` and the original block pointers left in place, all 8 shapes pass at 3.6.0-identical accuracy (6e-08 forward, 1e-06 gradients). Crossing `num_stages` against the K operand source shows `num_stages=1` exact for *both* block-pointer and plain-pointer K, while `2` and `3` are wrong **only** for the transposed block pointer, with 3.6.0 clean in all six cells — the earlier probe simply pipelined by default. The transposed operand is therefore the *minimal trigger*, useful for a bug report, but the loop breaks under pipelining regardless of pointer style: with Q/K/V all on plain arithmetic and `num_stages=2`, the kernel still faults.<br>• **Ruled out:** `tl.advance` (ruled out earlier, and it stays ruled out), and plain addressing — the load in isolation is exact on every version. | **Applied globally: `clamp_num_stages` in `common.py` forces `num_stages=1`** on every autotuned kernel when on HIP with Triton ≥ 3.8. It sits inside the OSS `triton_autotune` wrapper, which the 30 autotuned kernels across eight `ops/triton` files route through, so one edit covers them — except `_weighted_rms_norm_fwd`, which uses a raw `@triton.autotune` and is not clamped; configs are mutated in place so per-config extras survive — some backward configs carry a required `pre_hook` that rebuilding would drop. `TRITON_ALLOW_PIPELINING=1` keeps the original staging (to re-check a newer build), `=0` forces the clamp on a version that does not get it by default. **Inert on 3.6.0 and 3.7.1**, which still report `num_stages=2` and pass. **This rules attention out completely but does not make 3.8.0 usable.** Scoping it to attention alone moved the e2e fault from `gstep=0` deep into the *backward* pass (past `_weighted_layer_norm_bwd_dx`, faulting after `concat_2D_jagged_multirow` with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`) — evidence the defect is general, since those kernels autotune to `num_stages=3`. Clamping globally then removes the fault at that point but the run appears to **hang** instead: 100 % GFX, **0 % HBM**, 855 W, no step ever logged. **That "hang" is the same fault, not a hang** — re-running under `AMD_SERIALIZE_KERNEL=3` turns it back into `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, so without serialization the queue simply wedges behind the async fault. It is **not** pipelining: the kernels around it now autotune to `num_stages: 1`, confirming the clamp is applied. Attributed by filtering `AMD_LOG_LEVEL=3` to `ShaderName`: the last dispatch before the violation is **`_ln_mul_dropout_bwd_dx_du`** — a **separate defect, now root-caused and worked around in [4c]**. With both workarounds in place **3.8.0 completes a 20-batch smoke run**, the first time it has finished one. Unlike [1] the fault was **killable and the GPU recovered without a reboot** (verified by re-running the attention harness clean afterwards). Perf cost of losing pipelining everywhere is unmeasured and likely material, so keep using 3.7.1 until it is | • **The `gstep=0` fault — the symptom at the head of this row (~5 min):**<br>• **Run:** `IMG=recommendation-amdprimus0815:triton-3.8.0-amd TRITON_ALLOW_PIPELINING=1 NUM_TRAIN_BATCHES=20 PROGRESS_EVERY=1 AMD_SERIALIZE_KERNEL=3 ./scripts/run_smoke_shrink_1gpu.sh`<br>• **Expect:** `Memory access fault by GPU node-2 … on address 0x2000. Reason: Page not present or supervisor privilege` before step 1.<br>• `TRITON_ALLOW_PIPELINING=1` is required — `clamp_num_stages` otherwise suppresses the defect. `AMD_SERIALIZE_KERNEL=3` is what pins it to the attention kernel's first launch; without it you get a bare address and a queue wedged behind the async fault. | • **The same miscompile as wrong values, harness (10 s):** `python scripts/attn_check.py --fw-config m=128,n=32,stages=2,warps=8,nonkdim=16,kpack=2` → 485 / 1280 elements wrong (37.9 %), greatest absolute difference **3.3e+28**. The override replaces the pinned config in-process (it patches `Autotuner.configs`, no repo edit and no env var, so it bypasses the clamp). **Control:** the same command at `stages=1` passes at 3.6.0-identical accuracy — 6e-08 forward, 1e-06 gradients. `TRITON_ALLOW_PIPELINING=1 python scripts/attn_check.py` is equivalent and uses the real pinned config. `--case` picks one shape, `--skip-backward` narrows further. This lands on *mapped* memory, so it reports garbage rather than faulting. The stock `hstu_attention_test.py::HSTUAttentionTest::test_attn_triton` shows the same thing, but takes 143 s and its hypothesis shapes move run to run.<br>• **The root cause, minimal — this is the one to hand over (seconds):** `python scripts/repro_gfx1250_pipeliner.py`, torch+triton only, no repo imports, and unaffected by the clamp for that reason. It crosses `num_stages` against the K-operand source, so it carries its own controls and is meaningful on a healthy version. → on 3.8.0, `DEFECT PRESENT` with `num_stages=1` exact and `2`/`3` wrong for the transposed block pointer; on 3.6.0 and 3.7.1, `DEFECT ABSENT` with all six cells clean. It reports **wrong values, not a fault** — same miscompile, benign landing site, which is what makes it safe to run anywhere. |
| 7 | 2026-09-01 | • `3.8.0+git4cff872c` (AMD)<br>• **env:** `AMDGCN_USE_BUFFER_OPS=0`, `TRITON_ALLOW_PIPELINING` unset so the [4] `num_stages=1` clamp is active (its default on HIP + Triton ≥ 3.8) | **[4c]** **`3.8.0` miscompiles the fused-RNG layer-norm-mul-dropout backward, but only with dropout on.** This is what is left of 3.8.0's e2e failure once [4]'s clamp is in.<br>• **Symptom:** `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address`, with the runtime naming `kernel: _ln_mul_dropout_bwd_dx_du` outright.<br>• **Dropout is the whole discriminator:** the identical launch with `dropout_ratio=0.0` passes, and 3.6.0 passes either way. The forward is fine; only the backward faults.<br>• **Not the [4] pipelining defect:** this kernel is a plain `@triton.jit` launch that passes no `num_stages`, so the autotuner clamp never sees it, and forcing `num_stages=1`, `2` or `3` faults identically. `FAST_DROPOUT=True` (`rand3x` instead of three `tl.rand`) does **not** avoid it either.<br>• **Addressing is being clobbered:** the dropout code only calls `tl.rand` and `tl.where`, which touch no memory — consistent with register allocation at `BLOCK_D=512`. A stripped kernel with the same grid-strided loop and the same three `tl.rand` calls is **clean**, so it needs the full kernel's pressure and does not reduce further.<br>• **Found by dumping the real launch arguments rather than guessing shapes** — a shape sweep had already *exonerated* this kernel, because it used `dropout_ratio=0.0`. | **Force the separated-RNG path on gfx1250 for Triton ≥ 3.8** (`_fused_rng_ln_mul_dropout_is_broken` in `ops/utils.py`, folded into `use_separated_rng_ln_mul_dropout`). That precomputes the dropout mask and dispatches `_ln_mul_dropout_bwd_dx_du_rng` instead, avoiding the faulting kernel entirely. It is an **upstream-supported path**, already taken by sm100-103 and MI350 (gfx950); gfx1250 simply did not match the gate. Gated on Triton ≥ 3.8 so **3.6.0 and 3.7.1 are untouched** — whether gfx1250 wants this path on its own merits is a *performance* question, not to be settled on a miscompiling version. **With this plus [4]'s clamp, 3.8.0 finished a 20-batch smoke run with no fault, no hang and no `nan`** (NE 1.4817, and it reached `run_stop` and eval) — the first time it has completed a run. **But 20 batches does not generalise — a longer run still faults, a few hundred batches in; that is [4d].** Separately, `DEBUG_NAN_HOOKS=1` is **not usable on 3.8.0** to chase it: with hooks installed the run faults at `gstep=0` in the first backward, consistent with [1b]'s "plain read of the embedding tables" | • **How it surfaced, and the check that it is fixed:**<br>• With [4]'s clamp applied but this row's workaround absent, 3.8.0 e2e faults here: `AMD_LOG_LEVEL=3` filtered to `ShaderName` names `_ln_mul_dropout_bwd_dx_du` as the last dispatch before the violation.<br>• With both in place, a 20-batch smoke run completes — no fault, no hang and no `nan` (NE 1.4817, reaching `run_stop` and eval), the first time 3.8.0 has finished a run. **20 batches does not generalise:** a longer run still faults, which is [4d]. | • **Run (~8 s; needs the repo):** `triton_norm_mul_dropout` forward+backward at the recorded launch — `N=18187, D=512` bf16, `CONCAT_U=CONCAT_X=COMPUTE_Y=True`, `SILU_U=False`, `TRAINING=True`, `dropout_ratio=0.1`, `num_warps=2`, grid `(4546,)` = `N//4`.<br>• **Expect:** `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on **call #1** of the backward.<br>• **Controls:** the identical launch at `dropout_ratio=0.0` passes, and 3.6.0 passes at either ratio. Those two comparisons are the diagnosis.<br>• **To re-find an argument set like this:** wrap the kernel object in a recorder that dumps shapes, strides, dtypes and constexprs and `fsync`s on every call, overwriting. The fault is asynchronous and kills the process without unwinding, so the last file written is the faulting launch.<br>• **Script** (needs the repo): `AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_fused_rng.py` — launches that recorded config directly, bypassing the separated-RNG workaround. On `3.8.0+git4cff872c` expect the aperture violation (or silent corruption at four warps). On `7ff97e3109` expect `backward: PASS`. |
| 8 | 2026-09-01 | • `3.8.0+git4cff872c` (AMD)<br>• **env:** `AMDGCN_USE_BUFFER_OPS=0`, `TRITON_ALLOW_PIPELINING` unset ([4] clamp active), and the [4c] separated-RNG path active — both are the defaults on gfx1250 + Triton ≥ 3.8<br>• `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (runner default) or `False`: faults either way | **[4d]** **`3.8.0` faults *nondeterministically* in the HSTU attention backward, a few hundred batches in.** This is what is left of 3.8.0 once [4] and [4c] are worked around, and it is the **most interesting result on the board**: `_hstu_attn_bwd` is the *same kernel* as [4b]'s 3.7.1 `dk`/`dv` drift and the natural suspect for [3]'s `nan`, so a hard fault on 3.8.0 and bad gradients on 3.7.1 may be one defect rather than three — and this one is cheap to hit, which makes it a far better handle than [3]'s 2200-step wait.<br>• **Symptom:** `Memory access fault by GPU node-2 … Page not present or supervisor privilege`, attributed under `AMD_SERIALIZE_KERNEL=3` to `kernel: _hstu_attn_bwd`.<br>• **The batch index moves between runs** — 216 unserialised, 351 serialised — and a 250-batch run completed all 250 cleanly, consistent with the fault simply lying beyond 250 in that run. All runs read the **identical** data window (`[12960040, 13046440)`, 13,940,552 anchors, `global_batch_size=8`, verified in the logs), so batch count is the only variable and the moving index is **not** a data-dependent trigger at a fixed batch.<br>• **[4]'s clamp does not cover it:** the backward *is* autotuned, so it already runs at `num_stages=1`, and the forward is clean.<br>• **The config is not a variable:** the backward is *pinned* on HIP to one config (`BLOCK_M=32, BLOCK_N=128, matrix_instr_nonkdim=16, waves_per_eu=0, SEQUENCE_PARALLEL=False, UNROLL=1, num_stages=1, num_warps=4`), so `SEQUENCE_PARALLEL` is off and the `LOCK` spin-lock path is **not** in play. Every config carries the required `pre_hook` that zeroes `DQ` (and the `LOCK`, which is otherwise `torch.empty`), so an uninitialised accumulator is ruled out.<br>• **The faulting batch is not an outlier:** recording all 408 launches of a run that died at batch 135 shows the faulting batch is *small* — 9429 rows against a 17672 mean (7th percentile), `MAX_SEQ_LEN=1405` against a typical ~2800, lengths `[499, 500, 1405×6]`, `num_targets` all 1, and the same `Q` stride as every other launch. The three launches of that batch (one per HSTU layer) are identical, so the shape is unambiguous even unserialised.<br>• **Shape and layout alone do not reproduce it:** 200 iterations of exactly that batch pass on 3.8.0, both with contiguous `q/k/v` and with the true fused-`uvqk` views (row stride **2048**, not the contiguous 512, verified against the recording). So the trigger needs something the isolated kernel does not have — allocator/memory state, concurrent activity, or corruption from an earlier kernel — which is also why the moving batch index is better explained by run-to-run *data order* (multiple dataloader workers) than by the shape itself.<br>• **`expandable_segments` is not the cause:** `Page not present` is an *unmapped*-page signature and the runner defaults to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which maps and unmaps physical pages under a reserved virtual range — a good fit for a pressure-dependent fault invisible to isolated replays. But rerunning with `expandable_segments:False` still faults, at batch 111.<br>• **It is not being fed corrupt `seq_offsets`:** the recorded values at the faulting launch are monotonic and consistent with the row count, which rules out the most natural "victim of an upstream jagged kernel" story. Serialisation also attributes the fault to this kernel directly, rather than to a later async victim. | none yet | • **This is the only way to see it.**<br>• **Run:** `IMG=recommendation-amdprimus0815:triton-3.8.0-amd NUM_TRAIN_BATCHES=400 PROGRESS_EVERY=1 AMD_SERIALIZE_KERNEL=3 ./scripts/run_smoke_shrink_1gpu.sh`<br>• **Expect:** `Memory access fault by GPU node-2 … Page not present or supervisor privilege` with `kernel: _hstu_attn_bwd` somewhere in batches ~100–400. ~4 min serialised — serialising is *faster* per batch here and reached batch 351 — against ~7–10 min unserialised.<br>• **`PROGRESS_EVERY=1` is not optional:** the `[lr-warmup]` line fires only once, so a run that dies mid-window looks identical to one that never started, which is what made this look like a `gstep=0` fault at first. | • **None yet.**<br>• `attn_check.py` passes all 8 shapes, and 200 replays of the exact faulting launch pass — both with contiguous `q/k/v` and with the true fused-`uvqk` strides — so shape and layout alone are not enough. It needs either the e2e context or far more launches. |
| 9 | 2026-09-01 | • `main` @ `58895270`<br>• `76940ad3` — sglang's gfx1250 / ROCm 7.14 pin<br>• both built with [`scripts/build_triton_from_source.sh`](../../scripts/build_triton_from_source.sh) (`pip install -e .`, pass `TRITON_COMMIT`; ~5 min on this node, layered on the `:gfx1250` image so Triton stays the only variable)<br>• **env:** none applies — it fails before any kernel runs | **[5]** **Upstream Triton will not compile this codebase — including the revision sglang pins for gfx1250.**<br>• **Symptom:** `Block pointers have been removed in favor of the tensor descriptor API`, a `NotImplementedError` raised from `tl.make_block_ptr`. The HSTU kernels call it in 26 places, plus 8 `tl.advance`, across `triton_hstu_attention.py`, `triton_layer_norm.py` and `triton_hstu_linear.py`. An upstream API removal, not a gfx1250 bug.<br>• **Neither sglang Triton is a way out.** Its `gfx1250-rocm1000` stage sets `BUILD_TRITON=0` and takes `ROCM_TRITON_VERSION=3.8.0+git4cff872c` from the ROCm SDK — byte-for-byte the `triton-3.8.0-amd` image here, which already compiles and runs, so compilation was never that image's problem ([4]/[4c]/[4d] are). Its ROCm 7.14 flavor instead overrides AMD's Triton with upstream `TRITON_COMMIT_DEFAULT=76940ad3`, recorded there as the "known-good fallback if the SDK's Triton turns out to miscompile a gfx1250 kernel" — which is exactly our situation — but **76940ad3 is already past the removal too**, so the kernels do not compile on it either.<br>• **When reading sglang's bring-up as evidence:** sglang is **inference-only** and never runs an HSTU attention backward, a fused layer-norm/dropout backward, or the jagged training kernels, while every defect in [4]/[4c]/[4d] is in a *backward* pass. | **On this branch the kernels are ported off `tl.make_block_ptr`** to masked plain-pointer `tl.load`/`tl.store` (attention, layer-norm, linear) — not to `TensorDescriptor`. That unblocks compile on `7ff97e3109`; what remains is [6]. **Do not flip on the existing descriptor path**: `_should_enable_tma()` returns `False` on HIP by construction — it lowers to PTX `cp.async.bulk.tensor.*` (Hopper TMA hardware) and on AMD "either fails to compile or produces a kernel with mismatched reduction-dim shapes", with a further guard because gfx950 reports capability `(9, 5)` and would otherwise pass the Hopper check. Upstream still inherits 3.8.0's defect lineage (it also regressed [1]: `76940ad3` hangs `position_test`'s large-tensor case, which passes on 3.6.0). Without the port, stay on 3.7.1. | • **Cannot get that far.** The failure is at model construction, before a batch is ever loaded, so no dataset and no GPU are involved. | • **No GPU needed — it fails at compile.**<br>• **Run:** `TRITON_COMMIT=76940ad3 ./scripts/build_triton_from_source.sh`, then construct the model (importing the HSTU kernels is enough).<br>• **Expect:** `NotImplementedError: Block pointers have been removed in favor of the tensor descriptor API`, raised from the first `tl.make_block_ptr` at model construction, before any kernel runs.<br>• **Note:** `scripts/repro_gfx1250_pipeliner.py` does not compile there either, so it cannot answer whether [4] survives upstream — that would need a plain-pointer rewrite of the probe. |

Questions for the Triton team: (1) Is a hybrid kernel — some pointers narrowed to
32-bit buffer ops, others left 64-bit, in one body — intended and tested on
gfx1250, or does the pass group assume all-or-nothing narrowing? (2) Is
`0x80000000`-offset-plus-`num_records` masking sound on RDNA wave32 given
`num_records = 0xffffff`? (3) Do
[#11313](https://github.com/triton-lang/triton/pull/11313),
[#11246](https://github.com/triton-lang/triton/pull/11246),
[#11379](https://github.com/triton-lang/triton/pull/11379),
[#10804](https://github.com/triton-lang/triton/pull/10804) bear on this? We
tested `main` @ `76940ad3` and still reproduce. (4) Is gfx1250 covered by CI for
the buffer-op path above 2 GiB? (5) Separately, for defect 4: which `3.8.0` change
altered the dot-operand layout for a block-pointer load with `order=(0, 1)`, and
is that path covered by CI on gfx1250? It is exact on `3.6.0` and `3.7.1`, the
load is exact in isolation on `3.8.0`, and only the transposed operand of
`tl.dot` regresses.

### 9.3 Row 7 follow-up: coexec scheduler root cause

The standalone [`repro_gfx1250_fused_rng.py`](../../scripts/repro_gfx1250_fused_rng.py)
reproduces row 7 directly at the recorded `N=18187`, `D=512`, bf16 launch.  It
also checks the gradients, recomputed output, and all three dropout masks, which
exposed silent corruption at four warps in addition to the two-warp fault.

The proximate compiler root cause is the gfx1250 **coexec machine schedule under
LLVM expert-scheduling mode**, enabled by ROCm/triton commit
[`89db5e470c`](https://github.com/ROCm/triton/commit/89db5e470c330ea6f51ea3763a5dda58083f780c).
That commit is an ancestor of `4cff872c`, but is absent from the passing AMD
3.7.1 commit `0263a6a6`.  Holding the source, LLVM, input, two-warp launch, and
all other options fixed gives this matrix:

| coexec attribute | expert mode | result | backward VGPRs |
|---|---|---|---:|
| on | on | memory-aperture fault | 254 |
| off | on | correct | 135 |
| on | off | correct | 254 |

The last two rows separate the scheduler defect from register pressure alone.
With both controls enabled, two warps fault and four warps silently corrupt
results; eight warps is correct because Triton does not request coexec above
four warps.  Disabling coexec fixes the two- and four-warp kernels, including
`FAST_DROPOUT=True` and `COMPUTE_Y=False`.

For an unmodified `3.8.0+git4cff872c`, the immediately available compiler
workaround is:

```bash
AMDGCN_USE_BUFFER_OPS=0 TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
  python scripts/repro_gfx1250_fused_rng.py
```

A narrower compiler workaround is to backport ROCm/triton
[`#11272`](https://github.com/ROCm/triton/pull/11272), then run with
`TRITON_HIP_USE_COEXEC_SCHEDULER=0`.  The backport adds a cache-invalidating
knob so toggling the scheduler cannot reuse an incompatible cached binary.  The
existing separated-RNG application workaround remains preferable when only
this operation needs protection; `num_warps=8` is another verified but
performance-untuned kernel workaround.  The recommended image already contains
the proposed ROCm 7.14 / torch 2.11 gfx1250 nightly stack, so reinstalling those
wheels does not address this compiler-scheduling defect.

**Latest-main retest (2026-09-03): fixed.** Upstream `triton-lang/triton` main
at [`7ff97e3109`](https://github.com/triton-lang/triton/commit/7ff97e310935b4a79794878dbc911f9af25d38d9),
built from source with bundled LLVM `b010a18d`, passes the original two-warp
launch and the four-warp and `FAST_DROPOUT=True` corruption cases with default
settings.  The generated LLIR still carries
`"amdgpu-sched-strategy"="coexec"`, expert scheduling is still enabled, and the
two-warp kernel reports 260 VGPRs.  Therefore the row-7 defect does **not**
persist in that current compiler stack; it was fixed by intervening
Triton/LLVM codegen changes rather than by disabling coexec.  The precise
fixing commit has not been bisected.

### 9.4 The batch-128 nan on 3.6.0 / 3.7.1

**Not live.** Batch 128 is clean 2/2 over 4000 steps on the working stack
([3.2](#32-secondary-configurations)); this section is the record of what it did
on 3.6.0 and AMD 3.7.1, kept because the narrowing work is reusable and because
the defect was never explained — it was overtaken, not closed.

**Not the defect fixed in [row 15 of §4](#4-changes-applied-to-master), and
not reproducible on the current stack.** On 3.6.0 and 3.7.1 the separated-RNG
kernel is not on the code path at all, and batch 128 never approaches the int32
wrap row. Batch 128 is now clean 2/2 over 4000 steps
([3.2](#32-secondary-configurations)), so this is retained as an **unexplained
historical defect** rather than a live one — two sequentially exposed problems,
which is why the "step 50 → step 2200 → never" trail never converged.

- **[3] historical — `train_loss=nan` at batch 128 on 3.6.0/3.7.1, still
  unexplained.** Retained because it is a *different* defect from the one above.
  The owed
  batch-128 run (`results/nan_probe_b128/`) goes `nan` at the first logged point,
  **step 50** — not step 2200 as on AMD 3.7.1 — so the wait is now ~50 steps.
  **Caveat: "step 50" is the log cadence, not the onset.** `train_loss` is only
  logged every `METRIC_LOG_FREQ` steps (default 50, `utils.py:1305`), so all this
  establishes is "non-finite by step 50"; the true onset could be step 1. Re-run
  with `METRIC_LOG_FREQ=1` before treating the onset as a property of the defect.
  But it is **nondeterministic: 3 of 4 batch-128 runs**, with the fourth finite
  for 80 steps, so a clean run clears nothing.
  - **Status on `main` @ `7ff97e3109` (2026-09-10): the trigger MOVED UP IN BATCH
    SIZE. Batch 128 is clean; batch 1024 goes `nan` in ~10 minutes.**
    - **Batch 128 — clean.** Two 4000-step runs both finished with zero
      non-finite loss (`0.25² ≈ 6%` against the 3-in-4 rate above). Batch 128 is
      no longer a reliable trigger on this pin.
    - **Batch 1024 — `nan` at step ~190.** 18 consecutive finite points at
      `METRIC_LOG_FREQ=10` (steps 10–180, `train_loss` flat 0.1377–0.1398), then
      `nan` at **190** and 200, `run_stop status="aborted"`. No traceback, no page
      fault, no HW exception — silent numeric corruption, [3]'s exact profile.
      Onset is bounded to **steps 181–190** by the log cadence.
    - **Why this matters: it is the cheapest [3] reproducer yet.** ~190 steps ×
      3.3 s ≈ **10 min**, against 3.7.1's ~12 min at batch 128 and a stack where
      batch 128 no longer reproduces at all. Config: `EMBEDDING_ROW_SCALE=0.25`,
      production `yambda_5b.gin` otherwise, `AMDGCN_USE_BUFFER_OPS=0`,
      `HSTU_HAMMER_KERNEL=TRITON`. Log: `~/mi450_logs/prod_b1024_smoke.log`.
    - **Consistent with the `AUTOTUNE_B` hypothesis below.** `AUTOTUNE_B =
      prev_power_of_2(B)` makes 128 and 1024 *different* autotune cache entries
      selecting different `BLOCK`/`num_stages`/`num_warps`, so a codegen-sensitive
      corruption that vacated the 128 entry can still occupy the 1024 one. This is
      the first e2e evidence tying [3]'s batch dependence to that key rather than
      to occupancy or step count alone.
    - **`DEBUG_NAN_HOOKS=1` DOES NOT WORK on this stack — it cannot be used to
      name the producer.** It dies at **step 1** with `RuntimeError: Output 2 of
      BackwardHookFunctionBackward is a view and its base or another view of its
      base has been modified inplace`, i.e. PyTorch's backward hooks are
      incompatible with an in-place view op on the HSTU path. This is independent
      of batch size and of [1b] (the `=2` param sweep that wedged the GPU); `=1`
      simply crashes. Any triage plan below that depends on it needs another
      instrument. Log: `~/mi450_logs/b1024_nan_hooks.log`.
    - **Still n=1 at batch 1024**, and [3] is historically nondeterministic —
      repeat before treating the onset step as a property of the defect.
    - Everything below was measured on 3.6.0/3.7.1 at batch 128 and is retained
      because the defect is unexplained; the batch it reproduces at has changed.
  - **It is a Triton defect.** With `HSTU_HAMMER_KERNEL=PYTORCH` at the same
    batch and step, 3 of 3 runs are finite.
  - **The one provable batch-128 codegen delta:** `_add_embeddings_bwd_kernel`
    (`triton_position.py:190`) is the *only* kernel on the ranker path whose
    autotune key contains the batch size — `key=[…,"AUTOTUNE_B",…]` with
    `AUTOTUNE_B = prev_power_of_2(ctx.B)`. B=8 and B=128 are separate cache
    entries and independently select `BLOCK`/`num_stages`/`num_warps`. Every
    other kernel keys on model dims only and gets identical codegen at both
    batches, so attention backward is byte-identical at 8 and 128. Testable
    compile-only, in seconds. **Not yet run.**

#### The symptom, as recorded when it was live

| Field | On 3.6.0 / AMD 3.7.1 |
|---|---|
| Deterministic? | **No** — 3 of 4 runs |
| Symptom | `train_loss` `nan` at the first logged point (step 50), never recovers. The 4th run stayed finite for 80 steps. PyTorch kernels at the same batch and step: finite, 3 of 3 |
| Suspicious component | **A Triton HSTU op, and not attention.** TBE and the optimizer run in *both* arms and the PyTorch arm is finite, so neither is the sole producer. Attention in isolation is bit-exact at the true e2e shape. The module hooks only ever report `downstream`, so the producer is most likely inside a **fused autograd function that is not a module boundary** — `hstu_preprocess_and_attention` fits, and it is also [6]'s kernel. Chain: wrong gradient → TBE writes it into an embedding table → `nan` a few steps later |
| Standalone reproducer | **None**, and none was ever built. `DEBUG_NAN_HOOKS=1` does *not* name a producer: 80 embedding tables are already poisoned by the first non-finite forward. The attraction of this configuration for triage was speed — ~50 steps to onset instead of 2200 — so a hypothesis cost minutes |

The hypothesis it was eventually overtaken by is in
[row 15 of §4](#4-changes-applied-to-master): the int32 row-offset truncation
root-caused at batch 1024. **That is a different defect.** At batch 128 the
separated-RNG kernel is not on the code path on these versions and the wrap row
is never approached, so nothing in [4.2] explains the observations above.

#### What triage established, and failed to establish

These were recorded against the batch-128 `nan` while it was live. They are
listed here rather than under [7] — where the previous edition of this document
had them nested — because every one of them is a batch-128 [3] observation, not
an extended-VGPR one.

- **`DEBUG_NAN_HOOKS=1` cannot name a producer.** At the first non-finite
  forward (step 47) **80 params are already non-finite, all embedding
  tables**, and every backward hook reports `downstream` — the bad gradient
  was made outside the hooked modules, so the TBE backward or the optimizer
  is where to look next, not an `nn.Module`.
- **The per-op bisect does not attribute it at n=1.** One pass over all eight
  groups put `nan` on `mha` alone, but two repeats of that same
  configuration came back finite. With a 3-in-4 base rate, single runs cannot
  separate the groups; an attribution needs ~5 repeats per group.
- **Three standalone probes fail to reproduce it**, which is the useful
  negative result. Attention in isolation is clean at batch 128:
  `repro_gfx1250_attn_bwd.py --check-finite --batch-size 128` passes 300
  iterations; `repro_gfx1250_attn_ref.py` matches the PyTorch reference over
  200 random layouts at batch 128 (short sequences, the most the dense
  reference allows); and its `--chunk 8` batch-invariance mode is **bit-exact
  over 40 iterations at the true e2e shape** (batch 128, 4086-long histories).
  So row count alone does not change the kernel's output, and whatever [3] is,
  it needs something the isolated op does not have — mirroring [4d], where
  replaying the exact faulting launch also passed.

#### An intermittent batch-128 fault, seen once

At step 0 with buffer ops off: `Memory access fault … Page not present` on
address `0xa4699000`. No recurrence in three later batch-128 runs, and none
under `AMD_SERIALIZE_KERNEL=3` + `HIP_LAUNCH_BLOCKING=1` — an async signature.
The process wedged behind it but `docker kill` freed it and the GPU returned to
0 % without a reboot. **Never reached the bar for a defect id**; it needs a
repeat count, and batch 128 has since run 8000 clean steps without showing it
again.

### 9.5 The batch-8 pipelining `nan` ([4], pre-fix stack)

**Archived 2026-09-11.** This was the sole positive evidence for [4] and the
stated reason row 3's clamp was load-bearing. It is retained for provenance
only — batch 8 is no longer a tested configuration, and the claim it supported
(that lifting the clamp produces `nan`) **did not reproduce** in a 600-step run
at batch 1024. See [5.1](#51-re-test-queue) rank 7 for current status.

As recorded when it was live:

- **[4] pipelining is not fixed upstream — it changed symptom.** With
  `TRITON_ALLOW_PIPELINING=1` the forward returns to `num_stages=2` (verified on
  the decorated kernel) and `attn_check.py` passes all eight cases, where AMD
  3.8.0 returned `3.3e+28`. But e2e goes `nan` in 2 of 3 runs within 300 steps at
  batch 8 — step 100, step 300, then none — against clamped controls finite in
  2 of 2 plus the clean 4000-step run.
- **It is NOT the same defect as [3].** An earlier edition argued the two were
  probably one corruption. [3]'s root cause settles it: the int32 wrap needs
  `N > 1,398,102` jagged rows, and batch 8 produces `N ≈ 21,000` — 65× below the
  wrap row — so the overflow cannot fire in the configuration where [4] `nan`s.
  **This refutation still stands** and is why [3]'s fix does not explain [4].

**Moved out of "What breaks it" on 2026-09-11.** The row below described the
knob while it was a live hazard. It is kept because its *suspicious component*
analysis is the only written account of which kernels could host [4], and
because the caution under it is the standing reason a single clean run is not
allowed to clear this defect.

| Change | Defect | Deterministic? | Symptom | Suspicious component | Standalone reproducer for triage |
|---|---|---|---|---|---|
| `TRITON_ALLOW_PIPELINING=1` | [4] | **No — and currently not reproducible at all.** 2 of 3 at batch 8 (retired); **0 of 1 at batch 1024** | **Not observed on this stack.** The `nan`-within-300-steps evidence (step 100, step 300, none) was batch 8 and is archived to [9.5](#95-the-batch-8-pipelining-nan-4-pre-fix-stack). Unclamped at batch 1024: 600 steps clean, and **2.8 % slower** than clamped. No fault, no garbage in either era | **The HIP software pipeliner at `num_stages ≥ 2`.** Attention is ruled out at 2 (`attn_check.py` passes), so the witness is among the **other 26** kernels that pipeline unclamped: jagged concat/split and `jagged_dense_bmm` (3–5), layer-norm forward and `dwdb` backwards (3), the fused LN×mul×dropout kernels (3), position (2–4), swiglu (2–4), addmm (2). All are grid-strided loops over jagged extents | **None on this stack**, and this is the worst gap. [`repro_gfx1250_pipeliner.py`](../../scripts/repro_gfx1250_pipeliner.py) is torch+triton only but calls `tl.make_block_ptr` in 6 places, so on `7ff97e3109` it dies with [5]'s removal error before testing anything. Owed: a plain-pointer rewrite aimed at a non-attention kernel, and it must survive a 2-in-3 hit rate to prove anything |

**Why one clean run clears nothing.** [4] was never deterministic — it appeared
to raise the *rate* of a corruption rather than switch it on, which is exactly
[3]'s original signature (onset moving from step 50 to step 2200 across compiler
versions). That is why [3.1](#31-the-confirming-runs) reports two runs rather
than one, and it applies with full force to the n=1 batch-1024 result that
retired this row. [5.1](#51-re-test-queue) rank 7 asks for two more runs.

**Listing [4]'s candidate kernels.** The row above names them; this prints them,
with or without `TRITON_ALLOW_PIPELINING=1`, so a probe author can work from the
live set rather than the prose:

```bash
python - <<'EOF'
import importlib, pkgutil, generative_recommenders.ops.triton as T
for m in pkgutil.iter_modules(T.__path__):
    mod = importlib.import_module(f"generative_recommenders.ops.triton.{m.name}")
    for name in dir(mod):
        cfgs = getattr(getattr(mod, name), "configs", None)
        if isinstance(cfgs, list) and cfgs and hasattr(cfgs[0], "num_stages"):
            stages = sorted({c.num_stages for c in cfgs})
            if max(stages) > 1:
                print(f"{m.name}.{name}: num_stages={stages}")
EOF
```

That prints 28 kernels unclamped and exactly one clamped — the clamp covers 27,
and `_weighted_rms_norm_fwd` uses a raw `@triton.autotune`, so it stays at
`num_stages=3` either way.

## 10. Retracted

**`and`/`or` on tensors is not a bug** and the `triton_position.py` rewrite was
cosmetic. Triton warns, but `code_generator.py:visit_BoolOp` (3.6.0) lowers them
to `logical_and`/`logical_or`, which for boolean masks is exactly `&`/`|`.
Nothing goes unmasked; the position test passing afterwards was nondeterminism.

**Register spilling is not what causes defect [6]**, and this document said it
was. The `BLOCK_N=128` tile does reach the 1024-VGPR ceiling and spill while
`BLOCK_N=64` does not, but the causal chain runs through an extended VGPR held
across the WMMA region — a `waves_per_eu=2` variant with 5× the scratch passes
4000 iterations. See [Root cause of the attention-backward
fault](#62-root-cause-of-the-attention-backward-fault-6--4d).

**`s_nop 7` after every WMMA is not a fix** for it either (retracted upstream):
it passed 4000 iterations once and an identical repeat faulted at step 0. The
durable controls are late lane-offset rematerialisation and disabling
1024-addressable VGPRs.

**`hstu_compute::test_compute_output` was an inherited code bug, not a GPU bug**:
the forward dropped `mul_u_activation_type`, applying no activation while its own
backward applied one. A one-line fix takes that parameter grid from 32 mismatches
to 0. Our file is byte-identical to `mlcommons/training@master`, and it is dormant
in the benchmark, which only passes `"none"`.

## 11. Change inventory

- `Dockerfile.amdprimus0815` — gfx1250 image: gfx1250 torch plus fbgemm built from
  source with the three wave32 patches.
- `scripts/run_smoke_shrink_1gpu.sh` — defaults `AMDGCN_USE_BUFFER_OPS=0` and
  `HSA_ENABLE_COREDUMP=0`; forwards `DEBUG_NAN_HOOKS`; `IMG` selects the image.
- `scripts/repro_gfx1250_buffer_ops.py` — standalone hang reproducer, torch+triton
  only.
- `scripts/repro_gfx1250_pipeliner.py` — standalone reproducer for defect [4]'s
  transposed block-pointer `tl.dot` miscompile, torch+triton only. **Does not run
  on the current pin:** it calls `tl.make_block_ptr` in 6 places, removed by [5],
  so it dies at compile before testing anything. A plain-pointer rewrite is owed
  ([5.1](#51-re-test-queue), rank 7).
- `scripts/repro_gfx1250_fused_rng.py` — standalone reproducer for defect 4c's
  fused-RNG backward (needs the repo); passes on `7ff97e3109`.
- `scripts/repro_gfx1250_attn_bwd.py` — standalone reproducer for defect 6:
  `triton_hstu_attention_bwd` over a fresh jagged layout per iteration, faulted in
  ~15 s at `BLOCK_N=128` (needs the repo; no dataset). `--part fwd`,
  `--layout guarded`, `--contextual 0 --no-targets` are the controls that place
  the blame. Passes 4000 iterations with the `BLOCK_N=64` pin; revert the pin to
  reproduce.
- `scripts/repro_gfx1250_latest_main.py` — wider standalone HSTU stack for defect
  6 (needs the repo; no dataset). `--phase steps` faults; `--phase connected`
  (fixed shapes, e2e-sized TBE) passes.
- `scripts/bisect_hooks/` — `sitecustomize.py` forces individual HSTU op groups
  onto Triton or PyTorch inside the trainer (`BISECT_TRITON_OPS`,
  `BISECT_PYTORCH_OPS`) to attribute an e2e fault to one op; `gpu_cleanup.sh`
  clears wedged ranks and orphaned dataloader workers between runs.
- `scripts/build_triton_from_source.sh` — builds Triton at any `TRITON_COMMIT`,
  same recipe as sglang's Dockerfile.
- `dlrm_v4/utils.py` — peak-FLOPS table keyed on `gcnArchName` for gfx1250, so HFU
  stops using MI350X's number.
- `dlrm_v4/train/train_ranker.py` — `DEBUG_NAN_HOOKS` non-finite forward/backward
  hooks (diagnostic for defect 3).
- `ops/triton/triton_hstu_attention.py`, `ops/triton/triton_layer_norm.py`,
  `ops/triton/triton_hstu_linear.py` — pointer port for defect [5]: every
  `tl.make_block_ptr` / `tl.advance` replaced with masked plain-pointer
  `tl.load`/`tl.store` so the kernels compile on current Triton `main`. See
  **API changes**. Linear also contains the earlier `mul_u_activation_type`
  forward fix (inherited code bug, not a Triton API change).
- `ops/triton/triton_hstu_attention.py` — `_get_bw_pinned_configs()` also pins
  `BLOCK_N=64` on gfx1250 + Triton ≥ 3.8 to keep the attention backward off the
  1024-VGPR ceiling (defect [6] workaround, from
  [PR #7](https://github.com/chriscai-amd/training/pull/7)); 128 elsewhere.
- `common.py` — `clamp_num_stages` forces `num_stages=1` on every kernel routed
  through the OSS `triton_autotune` wrapper on HIP + Triton ≥ 3.8 (defect 4);
  inert on 3.6.0/3.7.1, forced either way with `TRITON_ALLOW_PIPELINING`.
  `_weighted_rms_norm_fwd` uses a raw `@triton.autotune` and is not covered.
- `scripts/attn_check.py` — deterministic attention check (fixed shapes, CPU-side
  comparison, forward and backward reported separately, `--fw-config` to override
  the pinned autotune config). 10 s; this is what the defect 4 bisection used.
- `scripts/bench_attn_bwd_blockn.py` — times attention forward+backward at a
  chosen pinned `BLOCK_N`, to quantify row 2's cost. One tile per process; it
  asserts the Triton kernel actually launched and prints its VGPR count.
- `scripts/repro_gfx1250_attn_ref.py` — **control for defect [3]**, and it
  passes: attention against its PyTorch reference over random jagged layouts,
  plus a `--chunk` batch-invariance mode that needs no dense reference and so
  reaches batch 128 at the true 4086-long e2e shape, where it is bit-exact.
- `scripts/repro_gfx1250_attn_bwd.py` — also gained `--batch-size` and
  `--check-finite` for the same purpose; batch 128 passes 300 iterations.
- `scripts/run_smoke_shrink_1gpu.sh` — now also forwards `BISECT_TRITON_OPS` and
  `BISECT_PYTORCH_OPS`, so a per-op bisect can be driven through the runner.
- `ops/triton/triton_position.py` — cosmetic: `&` instead of `and`, silencing
  Triton's deprecation warning. Note `triton_hstu_attention.py` still uses
  `and`/`or` on mask tensors at lines 1940–1942 and warns on current main.
- `scripts/run_prod_convergence_1gpu.sh` — the batch-1024 production harness
  behind both results in [3.1](#31-the-confirming-runs); runs `yambda_5b.gin`
  unmodified except `EMBEDDING_ROW_SCALE`, `SMOKE=1 SMOKE_BATCHES=N` for the
  bounded mode.
- `scripts/repro_gfx1250_lnmuldropout_i32.py` — standalone deterministic
  reproducer for defect [3], torch+triton only, cannot fault. `--fix` applies the
  widening; `--canary-gib` sizes the victim allocation.
- `scripts/repro_gfx1250_extvgpr_value.py` — standalone extended-VGPR codegen
  reproducer, and the `--occupancy/--streams` matrix that found [7].
  `--compile-only` needs no GPU dispatch.
- `ops/triton/triton_hstu_linear.py` — also carries the defect [3] fix:
  `rows_i64` widening in `_ln_mul_dropout_fwd_rng`
  ([row 15 of §4](#4-changes-applied-to-master)).
- `docs/lnmuldropout_i32_handoff.md`, `docs/gfx1250_extvgpr_handoff.md` — the two
  handoff write-ups those reproducers accompany.
