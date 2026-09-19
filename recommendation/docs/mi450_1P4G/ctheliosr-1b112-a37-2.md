# MI450 1P4G, host `ctheliosp-1b112-a37-2` (gfx1250, full model)

Updated **2026-09-19** after the second 3,000-step run and its post-run checks.
The requested filename is `ctheliosr-1b112-a37-2.md`; the hostname captured in
every experiment is **`ctheliosp-1b112-a37-2.mnb.dcgpu`**. This document retains
that distinction so the artifacts can be matched to the correct machine.
All experiment and capture times below are **UTC**. Firmware build dates are
quoted as reported; their source does not specify a timezone.

**NaN loss did not reproduce in either of two fresh-process runs.** Each ran
the full Yambda-5b HSTU model on four GPUs for **3,000 optimizer steps**, using
**local batch 1024, global batch 4096, seed 1, `START_TS=0` and all full
embedding tables**. Actual FBGEMM allocations verify **11 tables, 293,051,712
logical rows, width 512 and 600,169,906,176 FP32 weight bytes**. Neither run
reported a GPU fault/reset; all available total GPU ECC counters stayed zero.
All four GPUs passed post-run checks and returned to idle.

This is evidence of repeatability at this configuration. **It does not resolve
the A0/B0 NaN, validate every intermediate tensor, or establish convergence.**
Both launches use the same seed and boot, each stops at step 3000, and all
observed updates are still inside the 24,000-step LR warmup. Evaluation was
disabled. The local-2048/global-8192 AUC estimate in [§7](#7-auc-075-projection-at-local-batch-2048)
is an extrapolation; that batch size was not run.

Companion investigations:
[MI450 A0](../mi450_a0/mi450_a0.md),
[MI450 B0](../mi450_b0/mi450_b0.md), and
[A0/B0 reconciliation](../mi450_a0_b0_reconciliation_20260918.md).

## Start here

| Question | Recorded answer |
|---|---|
| Does the full model fit? | **Yes at local batch 1024 on four GPUs.** All embedding rows were retained; peak sampled VRAM was about 391–406 GiB per 432 GiB GPU. |
| Did NaN loss reproduce? | **No: 3,000/3,000 finite losses in each of two runs.** There are no missing or duplicate MLLOG loss records. |
| Was the model shrunk? | **No.** `EMBEDDING_ROW_SCALE=1.0`, full 512-wide tables, three HSTU layers and sequence limit 4096. Column sharding retains every vocabulary row. |
| Which attention arm ran? | **Uncapped baseline**, `HSTU_BWD_MAX_VGPR=0`, exact upstream Triton `7ff97e310935b4a79794878dbc911f9af25d38d9`; no cap256 training comparison. |
| Was training resumed? | **No.** Empty checkpoint path and fresh trainer processes in each launch. |
| Is this a 6,000-step training result? | **No.** It is two 3,000-step runs from the same configured start, totaling 6,000 step executions. |
| Was holdout AUC 0.75 measured? | **No.** Holdout evaluation was disabled. The conditional estimate at local 2048/global 8192 is about **11 hours**, plus startup, if memory fit, throughput and convergence assumptions hold. |

## Contents

1. [Host and software stack](#1-host-and-software-stack)
2. [Dataset and full model](#2-dataset-and-full-model)
3. [Exact training settings](#3-exact-training-settings)
4. [Experiment record](#4-experiment-record)
5. [Results and comparison](#5-results-and-comparison)
6. [Reproduction and analysis commands](#6-reproduction-and-analysis-commands)
7. [AUC 0.75 projection at local batch 2048](#7-auc-075-projection-at-local-batch-2048)
8. [Evidence inventory and limits](#8-evidence-inventory-and-limits)

## 1. Host and software stack

### 1.1 Host, CPU, GPUs and firmware

The run-time host inventory was captured at **03:58:41 UTC**. A supplemental
read-only inventory at **08:32:37 UTC**, on the same boot, records OS, CPU
topology, Docker versions and container configuration omitted from the first
snapshot. These are `host_inventory.json` and `documentation_inventory.json`
under the setup directory listed in [§8](#8-evidence-inventory-and-limits).

| Item | Recorded value |
|---|---|
| Actual hostname | `ctheliosp-1b112-a37-2.mnb.dcgpu` |
| Host OS | CentOS Stream 9; supplemental documentation-time snapshot |
| CPU | `AMD Eng Sample: 100-000001046-04`, family `0x1a`, model `0x51`, stepping 0 |
| CPU topology | One socket, 128 physical cores, two threads/core; 256 present logical CPUs, **255 online (0–254), CPU255 offline** |
| NUMA topology | 38 nodes reported by the supplemental `lscpu` snapshot |
| Host-visible RAM / swap capacity | 539,757,096,960 bytes RAM (502.688 GiB); 8,589,930,496 bytes swap |
| GPUs | Four MI450-class devices, target `gfx1250`, PCI vendor/device `1002:75c1`, revision `0x00`, 256 reported compute units each |
| HBM | **463,856,467,968 bytes = 432 GiB per GPU**; 1,728 GiB aggregate |
| IFWI, all four GPUs | `AMD MI450X_GENERIC`, part `113-M4500001-640C`, version `00198930`, built `2026/08/19 23:17` |
| AMDGPU driver | `7.1.0.31300009` |
| Kernel | `6.16.1-0_fbk2_brcmrdma5_35_g5ba27bd1d6b9` |
| Loaded driver controls | `gpu_recovery=0`, `halt_if_hws_hang=0` |
| Boot ID for both runs | `3cb36f24-0832-402b-af25-ff917fa8350f` |
| Recovery actions during this experiment | No reboot, AC cycle, driver reload or GPU reset performed |

Saved boot messages independently report one CPU package and CPU255's
invalid-APIC-ID bring-up failure. The nominal 256-thread topology must not be
quoted as 256 online CPUs. GPU PCI endpoints in the boot log are
`0001:04:00.0`, `0002:04:00.0`, `0003:04:00.0` and `0004:04:00.0`; these
records alone do not establish their mapping to training rank ordinals.

The GPU market-name string is `AMD Radeon Graphics`, while IFWI identifies
`AMD MI450X_GENERIC`. Board product/model/FRU and PCIe width/speed fields are
`N/A`. Revision `0x00` does not establish an A0/B0 stepping distinction.
The captured kernel command line includes `pci=realloc=off`, `iommu=pt`,
`mitigations=off` and an `amdgpu,device_dax,dax_hmem` module blacklist; the
driver was already loaded. These settings were recorded, not changed here.

### 1.2 Runtime and source pins

| Component | Exact recorded version or source |
|---|---|
| Checkout | `/home/chcai/training`, branch `chcai/mi450` |
| Training commit | `82e3d1c87bbb409c2679977f3b1677e04bbc21a1` |
| Python | `3.12.3`, `/opt/venv/bin/python` |
| Torch | `2.11.0+rocm7.14.0a20260625` |
| Torch source | `f55dda6ca78b73027840c1bc4014fe703d4f5473` |
| HIP | `7.14.60850` |
| Triton | Distribution `3.8.0+git7ff97e31`; imported version string `3.8.0` |
| Triton source | **`7ff97e310935b4a79794878dbc911f9af25d38d9`** |
| Triton import path | `/opt/triton-custom/python/triton/__init__.py` |
| TorchRec | `1.7.0a0+bf55480`, source `bf554808b185cc7c694525e34ae2b910311842cf` |
| FBGEMM | `10b775730212923f65f7b78f79b6a01d80cf3c29` plus Dockerfile patches; installed `fbgemm-gpu-nightly-rocm==2026.9.19` |
| RCCL | Runtime reports `2.30.4-Unknown`; used through PyTorch's `nccl` backend |
| Data/config/logging packages | `polars-u64-idx==1.33.1`, `gin-config==0.5.0`, `mlperf-logging==4.1.51` |
| Docker client / server | `29.5.2` / `29.5.2`; supplemental documentation-time snapshot |
| containerd / runc | `2.2.4` / `1.3.5` |

No trainer/model source was changed for these experiments. Both launches
captured the same **150 source SHA-256 entries** and matched the checked-out
files. The existing A0/B0 kernel fixes and workarounds in that commit remain
part of the tested stack; the full-table result does not isolate their effects.

### 1.3 Images and container

| Item | Value |
|---|---|
| Base image | `amdprimus/amdprimus:gfx1250-20260910` |
| Base digest | `sha256:6e656de79e6c8d7f3b3db3690606c3e6cb0536ec871f0521735746015297c9a1` |
| Build recipe | [`Dockerfile.amdprimus0815`](../../Dockerfile.amdprimus0815), with the September 10 base and exact Triton commit overridden explicitly |
| Prepared image | `recommendation-mi450:0910-7ff-full` |
| Prepared manifest-list digest / recorded image identity | `sha256:87b89eb5bd7d9abb6e1fdab91456e0b83eae9b5520299d3f282c7e750bbcab0e` |
| Build platform manifest | `sha256:05db327aa12133b407c1e6ebeb082198f28b4da8e7be14b76836d4754f8c73f0` |
| Container | `mi450-fullmodel`, created `2026-09-19T04:16:09.524887932Z` |
| Container ID | `ed0eae3e81bce83d66ca6010cd898fb94b24c1f224152d5bdc270e6e437f10e7` |
| Execution | Host network and host IPC, nonprivileged, `/dev/kfd` and `/dev/dri`, supplementary `video` group, `seccomp=unconfined`, `label=disable` |

| Host path, mounted read/write | Container path |
|---|---|
| `/home/chcai/training` | `/workspace` |
| `/home/chcai/data/mlperf_dlrm_v4` | `/data/mlperf_dlrm_v4` |
| `/home/chcai/runs` | `/runs` |

Host IPC uses host shared memory. Docker's `ShmSize` metadata is 64 MiB;
it is not evidence of a private 64 GiB shared-memory allocation.

The Dockerfile retains native ROCm Torch and builds FBGEMM for `gfx1250`.
Its patches add the missing FP16/BF16 conversion sources, align the host warp
constant to wave32, and align code generation to that warp size. Triton is
built from the exact commit after dependency installation. Torch's package
metadata requests Triton 3.6, so the override emits a dependency warning;
the build import check, prepared-stack checks and both training runs succeeded
with the documented custom version.

**The image's default allocator setting is not the setting used in training.**
The launcher explicitly clears both allocator variables, overriding the
Dockerfile's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It also sets
the required library directory:

```text
HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
```

## 2. Dataset and full model

### 2.1 Verified Yambda-5b cache

| Item | Value |
|---|---|
| Dataset / mode | `yambda-5b` / `streaming-train-eval` |
| Host cache root | `/home/chcai/data/mlperf_dlrm_v4` |
| Prepared cache | `processed_5b/hstu_cache_L4086` |
| Downloaded bytes | 227,409,136,794 |
| Integrity checks | All **20 official MD5** and **16 repository SHA-256** checks passed |
| Event store | 4,763,272,693 events, 1,000,000 users |
| Floating source scan | All 4,763,272,693 floating source values scanned; **zero NaN/Inf** |
| Training anchors | 2,290,835,423 across windows 0–298 |
| Fixed holdout | Window 299, one window, 2,058,204 anchors |

Integrity and finite-source checks validate the prepared inputs. They do not
validate every intermediate tensor produced by training. The full catalog
sizes were retained; no quarter-vocabulary remapping was selected.

### 2.2 Architecture and precision

The four `DlrmHSTUConfig` startup records agree with the production settings in
[`yambda_5b.gin`](../../generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin).

| Model setting | Value |
|---|---|
| HSTU layers / heads | **3 / 4** |
| Table / transducer width | **512 / 512** |
| Per-head attention Q/K and linear/value dimensions | 128 / 128 |
| Preprocessor hidden width | 256 |
| History / minimum history | **`HISTORY_LENGTH=4086`, `MIN_HISTORY=4086`**, `HISTORY_STRATEGY=interleaved` |
| Maximum sequence length / candidates | **`MAX_SEQ_LEN=4096` / 1** |
| Dropout | Input 0.2; linear/output 0.1; rank dropout decorrelation disabled |
| Attention | Causal and target-aware; no sliding attention-length restriction; sorting by length enabled |
| Recomputation | Normalized input, UVQK and output recomputed |
| Group normalization | Disabled |
| Context | Eight contextual features, maximum length one each, minimum UIH lengths zero |
| Task | Binary `listen_plus`, task weight 1, causal multitask weight 0.2 |
| Action features | Weights `[1, 2, 4]`, embedding dimension 8, initialization standard deviation 0.1 |
| Timestamp postprocessor | Layer norm enabled, epsilon `1e-5`; hour-of-day and day-of-week features |
| Position / time buckets | 8192 / 2048 |

**Precision is mixed, not uniformly BF16.** Actual FBGEMM TBE records report
**FP32 embedding weights and FP32 outputs**. Gin enables
`make_model.bf16_training=True`; the HSTU user/transducer path converts inputs
to BF16 and runs under BF16 autocast, then converts candidate-user embeddings
back to the incoming embedding dtype. Dense parameters are constructed with
the FP32 default, while STU projections cast weights to the activation dtype
for compute. The item MLP and prediction head sit outside this HSTU autocast
scope. Sparse communication precision is separately set to FP16 in both
directions ([§3.2](#32-optimizer-and-communication)).

### 2.3 Full embedding tables and actual sharding

`EMBEDDING_ROW_SCALE=1.0`, `EMB_PLACEMENT=hbm`, with both
`EMB_PLACEMENT_OVERRIDES` and `EMB_SHARDING_OVERRIDES` empty.
Every logical table has **512 FP32 columns**. The following layout was verified
from **eight native FBGEMM TBE initialization records in each run**:

| Table | Full logical rows | Actual placement |
|---|---:|---|
| `item_id` | 9,390,624 | Whole table on rank 0 |
| `artist_id` | 1,293,395 | 128 columns on each of ranks 0–3 |
| `album_id` | 3,367,692 | Whole table on rank 3 |
| `uid` | 1,000,001 | 128 columns on each of ranks 0–3 |
| `user_x_artist` | 100,000,000 | Whole table on rank 0 |
| `user_x_album` | 40,000,000 | Whole table on rank 1 |
| `user_x_hour` | 24,000,000 | Whole table on rank 2 |
| `item_x_hour` | 40,000,000 | Whole table on rank 2 |
| `artist_x_hour` | 32,000,000 | Whole table on rank 1 |
| `user_x_is_organic` | 2,000,000 | 128 columns on each of ranks 0–3 |
| `user_x_artist_x_hour` | 40,000,000 | Whole table on rank 3 |
| **Total** | **293,051,712** | **Eight whole tables and three column-sharded tables** |

The sum of actual allocations is **600,169,906,176 weight bytes =
558.951782 GiB**. Column sharding partitions the 512 columns and retains every
row. The inferred row-wise Adagrad state for the actual local shards is
**1,223,727,600 bytes**, including duplicated row state for column-sharded
tables. Weights plus that state total **560.091467 GiB**; this excludes dense
model state, activations, communication buffers and allocator cache.

| Rank | Actual FP32 embedding weight bytes |
|---|---:|
| 0 | 226,230,216,704 |
| 1 | 149,654,218,752 |
| 2 | 133,270,218,752 |
| 3 | 91,015,251,968 |

`HBM_CAP_GB=260` is the per-rank planner setting. It is not a cap on total
runtime VRAM usage; measured peaks include the rest of training memory.

## 3. Exact training settings

These tables describe **both completed runs**. Their saved explicit settings
differ only in `RUN_NAME`: `baseline_noeval` versus `repeat01`.
The complete machine-readable settings are each run's
`effective_training_env.json`; inherited environment is not exhaustively
represented by that file. The initialization-only attempt's one setting
difference is recorded in [§4](#4-experiment-record).

### 3.1 Distributed execution, batching and input pipeline

| Setting | Value |
|---|---|
| `NNODES`, `GPUS_PER_NODE`, `WORLD_SIZE`, `NODE_RANK` | `1`, `4`, `4`, `0` |
| Visible GPUs | `HIP_VISIBLE_DEVICES=0,1,2,3` |
| **Local / global batch** | **`BATCH_SIZE=1024` per rank / 4096 aggregate** |
| Gradient accumulation | 1 |
| Rendezvous | `MASTER_ADDR=127.0.0.1`; `MASTER_PORT` empty in explicit settings, resolved by the trainer |
| RCCL socket/debug | `NCCL_SOCKET_IFNAME=lo`, `NCCL_DEBUG=WARN` |
| Python | `PYTHONUNBUFFERED=1`, `PYTHONPATH=/workspace/recommendation` |
| Dataset path inside container | `DLRM_DATA_PATH=/data/mlperf_dlrm_v4` |
| Start / configured training windows | `START_TS=0`, `NUM_TRAIN_TS=299`; windows 0–298, 86,400 seconds each |
| Per-window batch limits | `NUM_TRAIN_BATCHES=0`, `NUM_EVAL_BATCHES=0` (uncapped) |
| Global optimizer bound | **`DIE_AT_STEP=3000`** |
| Model seed / dropout | `SEED=1`, `DECORRELATE_DROPOUT=0`, `INIT_CHECKSUM=0` |
| Data shuffle | `STREAMING_SHUFFLE_FRACTION=0.0`, `STREAMING_SHUFFLE_SEED=0`; no within-window sorting |
| Split | `TRAIN_SPLIT_PERCENTAGE=1.0`, `SPLIT_SALT=0` |
| Loader | `NUM_WORKERS=4` per rank, `PREFETCH_FACTOR=8`, `PERSISTENT_LOADER=1`, `DOUBLE_BUFFER=1` |
| Incomplete batches | Dropped (`drop_last=True`) |

With global batch 4096, windows **0–5 contain no full batch**. The 3,000-step
bound is reached at **window 46, batch 190**. The sample counter increases by
4096 per optimizer step, ending at **12,288,000** in each run. Dividing by the
old one-GPU batch of 1024 would miscount progress by four.

### 3.2 Optimizer and communication

| Setting | Value |
|---|---|
| Dense optimizer | Adam, betas `(0.95, 0.999)`, epsilon `1e-8`, weight decay 0 |
| Sparse optimizer | Fused `EXACT_ROWWISE_ADAGRAD`, epsilon `1e-8`, weight decay 0 |
| Base learning rates | `DENSE_LR=1e-6`, `SPARSE_LR=1e-6` |
| Warmup | `LR_WARMUP_STEPS=24000`, `LR_WARMUP_START_LR=0`, linear global-step schedule |
| Dense gradient clipping | `GRAD_CLIP_NORM=1.0`, after backward and before dense update |
| Native TBE options | FP32 weights/output; stochastic rounding enabled; native sparse `gradient_clipping=False` |
| Sparse communication | **`SPARSE_A2A_FWD=fp16`, `SPARSE_A2A_BWD=fp16`** |
| Communication codec | `QCOMM_LOWMEM_CODEC=1`, cast-then-clamp conversion |
| Lookup deduplication | `EC_INDEX_DEDUP=1` |

Dense gradient clipping does not clip the fused sparse update. Warmup is
propagated to the fused TBE kernel in the same step; startup readbacks check
the dense, fused and kernel LR values.

**All 3,000 observed steps are inside warmup.** The first logged LR is zero;
the last is **`1.2495833333333334e-7`**, approximately 12.5% of the base LR.
The schedule uses the preceding zero-based step, so the last value is
`1e-6 × 2999 / 24000`. Stability at the post-warmup LR was not tested.

### 3.3 Kernel and allocator controls

| Setting | Value |
|---|---|
| HSTU implementation | `HSTU_HAMMER_KERNEL=TRITON`, `HSTU_NUM_LAYERS=3` |
| Attention backward cap | **`HSTU_BWD_MAX_VGPR=0`**, uncapped baseline |
| Triton controls | `TRITON_FULL_AUTOTUNE=0`, `TRITON_ALLOW_PIPELINING=0`, `AMDGCN_USE_BUFFER_OPS=0` |
| Allocator variables | **Both `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` empty** |
| GPU coredumps | `HSA_ENABLE_COREDUMP=0` |
| hipBLASLt library | Explicit gfx1250 `HIPBLASLT_TENSILE_LIBPATH` from [§1.3](#13-images-and-container) |
| Serialization/debug inheritance | Launcher removes launch-blocking, kernel-serialization, `NAN_*` and `BISECT_*` overrides before applying the baseline; removed-variable list was empty in these launches |

The verified backward configuration is `BLOCK_M=32`, **`BLOCK_N=64`**,
matrix instruction dimension 16, `waves_per_eu=0`, four warps, one stage,
`SEQUENCE_PARALLEL=False`, `UNROLL=1`, and no `maxnreg` cap. Its required
pre-hook zeroes `DQ`. Forward source pins are `BLOCK_M=128`, `BLOCK_N=32`,
matrix instruction dimension 16, `waves_per_eu=0`, `kpack=2`, eight warps,
two stages, with TLX disabled. See the pinned
[attention implementation](../../generative_recommenders/ops/triton/triton_hstu_attention.py)
and `prepared_stack.json`.

### 3.4 Evaluation, checkpoints, logging and stop conditions

| Setting | Value |
|---|---|
| **Evaluation cadence** | **`EVAL_EVERY_N_WINDOWS=0`, `EVAL_EVERY_DATA_PCT=0`** |
| Holdout configuration | `EVAL_HOLDOUT_TS=299`, `EVAL_HOLDOUT_NUM_WINDOWS=1` |
| Dormant eval settings | `SKIP_EVAL_EPOCH_PCT=0.025`, `AUC_THRESHOLD=0.75`; no holdout evaluation AUC was measured |
| Checkpoint input | **`CKPT_PATH` empty**, fresh initialization |
| Checkpoint output | `CKPT_STEP_FREQ=0`, `IN_WINDOW_CKPT_FREQ=0`, `CKPT_TIME_INTERVAL_S=0` |
| Loss logging | `METRIC_LOG_FREQ=1`, `MLPERF_TRAIN_LOSS_LOG_FREQ=1`, `MLPERF_LOGGING=1`, `PROGRESS_EVERY=10` |
| NaN probes | `NAN_MODULE_PROBE=0`, `NAN_TRIPWIRE=0`, `DEBUG_NAN_HOOKS=0` |
| Other diagnostics | `OUTPUT_TRACE=0`, `EVAL_INTERIM_LOG=0`, `DIAG_UNIQUE_EMB=0`, empty `TENSORBOARD_LOG_PATH` |
| Nonfinite stop | Supervisor stops after losses at **two distinct optimizer steps** are nonfinite |
| Manual stop | A `stop.request` file in the run directory signals only the token-verified owned trainer process group |
| Planned completion | At `DIE_AT_STEP=3000`, worker exit 42; multiprocessing parent exit 1; controller verifies the bound and returns 0 |

The training logger still emits `metric/window_auc` and
`metric/lifetime_auc`. These training-stream metrics are distinct from the
fixed-holdout `eval_accuracy` used for the RCP convergence target; they do not
establish that target was reached.

Console and MLLOG each report every loss. The supervisor therefore records
6,000 reports for one 3,000-step run; the independent audit counts **only the
3,000 MLLOG loss events**. A console/MLLOG pair at one step cannot satisfy
the two-distinct-step nonfinite stop rule.

The final `ProcessExitedException` is expected from the explicit worker exit
42. A finite verdict requires the bound marker, complete loss coverage,
controller result and process cleanup; parent exit 1 alone is not the verdict.

## 4. Experiment record

### 4.1 Preparation and initialization-only attempt

| Experiment | UTC time | Result and scope |
|---|---|---|
| Dataset preparation | Before training; details in dataset logs | Full cache downloaded and all checksums/finite-source checks passed ([§2.1](#21-verified-yambda-5b-cache)). |
| Four-GPU Torch/RCCL preflight | **03:59:37.723**, elapsed 4.102 s | All four GPUs passed small indexing, matmul and collective checks. |
| CPU-only supervisor validation | Approximately **04:01:59–04:02:19** | Healthy bound, synthetic nonfinite losses, manual stop and orphan cleanup paths passed; unrelated process remained untouched. Synthetic NaNs here are not a GPU reproduction. |
| Prepared-stack verification | Recorded **04:16:55.309** | Exact Triton commit, full row counts and backward pin/pre-hook verified; all four GPUs passed finite 4096-square matmul and exact 1,048,576-element Triton addition. |
| Initial `baseline` launch | **04:17:07.976–04:18:50.860** | Full model and tables initialized; **zero optimizer steps**. Intentionally stopped during evaluation-related startup scans. |

The initial `baseline` used `EVAL_EVERY_DATA_PCT=0.001`. The startup sample-count
path scans all 299 windows; evaluation cadence was then changed to zero for
the NaN-only experiment. The only explicit differences between this attempt
and `baseline_noeval` are `RUN_NAME` and that cadence. The initial attempt
has manual stop reason, trainer exit `-15`, controller/launcher exit **20**,
and no surviving owned processes. It is evidence that full allocation worked,
**not a third training trial, a NaN failure or a 3,000-step pass**.

### 4.2 First full-model training run

Run directory: `/home/chcai/runs/mi450_4gpu_baseline_noeval/`.

- Host launch **04:19:00.467**; completion **06:07:26.083**: **1h 48m 25.616s**.
- First logged step **04:20:33.065**; final logged step **06:07:22.015**.
- Exactly **3,000 consecutive finite MLLOG losses**, 12,288,000 samples.
- Final loss **0.13064220547676086**; planned window-46/batch-190 stop.
- Controller exit 0, no owned survivors, no recorded GPU fault/reset.
- Postflight at **06:07:53.907**, elapsed **4.068 s**: all four GPUs passed.

### 4.3 Fresh-process repeat

Run directory: `/home/chcai/runs/mi450_4gpu_repeat01/`.

- Host launch **06:12:35.179**; completion **08:00:56.280**: **1h 48m 21.101s**.
- First logged step **06:13:41.675**; final logged step **08:00:52.444**.
- Exactly **3,000 consecutive finite MLLOG losses**, 12,288,000 samples.
- Final loss **0.13066786527633667**; same window-46/batch-190 stop.
- Same seed, boot, container, source hashes, model layout and settings; fresh
  trainer processes and no checkpoint. Only `RUN_NAME` differs.
- Controller exit 0, no owned survivors, no recorded GPU fault/reset.
- Postflight at **08:01:26.052**, elapsed **3.775 s**: all four GPUs passed.

Each Torch/RCCL preflight/postflight checks 1,000 `index_select` and 1,000
`gather` calls per dtype per GPU using FP32/BF16 and varied index patterns,
1024-square FP32/BF16 matmuls against CPU references, and ten all-reduce plus
ten all-to-all operations per dtype using int64/FP32. These are small checks;
they do not exercise production-size communication, the full HSTU backward
or fused embedding updates.

## 5. Results and comparison

### 5.1 Losses, coverage and repeatability

| Measurement | First run | Repeat |
|---|---:|---:|
| Consecutive optimizer steps | 1–3000 | 1–3000 |
| Missing / duplicate MLLOG steps | 0 / 0 | 0 / 0 |
| Samples processed | 12,288,000 | 12,288,000 |
| NaN or infinite logged losses | **0** | **0** |
| First loss | 0.1411261260509491 | 0.1411261260509491 |
| Final loss | 0.13064220547676086 | 0.13066786527633667 |
| Minimum loss, both at step 2951 | 0.11432307213544846 | 0.11430364102125168 |
| Maximum loss, both at step 2703 | 0.18032076954841614 | 0.18053200840950012 |
| Final logged LR | 1.2495833333333334e-7 | 1.2495833333333334e-7 |
| Full table weight bytes | 600,169,906,176 | 600,169,906,176 |
| Controller result | `finite_through_step_bound` | `finite_through_step_bound` |

All 150 captured source hashes, runtime pins, container identity, launcher,
generated controller and actual table placements match. Of 3,000 paired
full-precision losses, **14 are exactly equal and 2,986 differ**. The first
difference is at step 10. Maximum absolute difference is
**0.0005288273096084595**; mean absolute difference is
**0.000017682102819283802**. Small numerical differences exist despite matching
seed and configuration; bitwise-identical training is not established.

The two launches total **6,000 step executions and 24,576,000 sample
processings**, repeating the same configured starting range. They are not one
6,000-step trajectory or 24,576,000 unique samples. Sample counters and
configuration align; batch tensor/anchor-identity hashes were not captured.
The observed failure count is **0/2 launches**, without a statistical failure
probability inferred from correlated steps or same-seed runs.

### 5.2 HBM use and hardware observations

| GPU | First peak VRAM (MiB) | Repeat peak VRAM (MiB) | Repeat peak (GiB) | Capacity (GiB) |
|---|---:|---:|---:|---:|
| 0 | 408,877 | 408,878 | 399.29 | 432 |
| 1 | 416,119 | 416,120 | 406.37 | 432 |
| 2 | 400,501 | 400,502 | 391.12 | 432 |
| 3 | 408,211 | 408,211 | 398.64 | 432 |

These are sampled driver-used VRAM values, approximately every ten seconds;
instantaneous peaks can be missed. AMD-SMI labels its raw units `MB` (total
442,368 per GPU); the table uses their binary-capacity interpretation. Active
tensors and allocator-reserved cache were not separately recorded. Worst
observed headroom is about **25.6 GiB**, so local batch 2048 fit cannot be
assumed from the local-1024 result.

| Observation | First run | Repeat |
|---|---:|---:|
| GPU telemetry snapshots | 613 | 613 |
| Failed telemetry calls / missing devices | 0 / 0 | 0 / 0 |
| Recorded GPU fault/reset or MES timeout signatures | 0 | 0 |
| Total correctable / uncorrectable / deferred GPU ECC | All zero on all four devices | All zero on all four devices |
| Corrected CPU MC57 records | 40 | 38 |
| Post-run GPU processes | None | None |

Cache/block ECC, PCIe error counters and XGMI errors report **`N/A`**, not
zero. Only the three available total GPU ECC counters support the zero-count
claim. Both runs returned to about 166 MiB of driver-used VRAM per GPU;
AMD-SMI's final integer display is 165 `MB`.

The corrected CPU records split evenly between CPU106 and CPU218: 20 each
in the first run and 19 each in the repeat. Status `0xdc604000003e0119`,
IPIDs ending `0406`/`0407` and syndrome `0x02010018` match records already
present in `dmesg.before.log` from **September 16**. They are retained as
corrected CPU/system observations, with no demonstrated training attribution.
They are not counted as GPU faults.

### 5.3 Measured throughput

The completed first run sustains about **1,880–1,940 aggregate samples/s**
after early startup, depending on the measured window:

| First-run step interval | Mean step time | Aggregate samples/s |
|---|---:|---:|
| 501–1000 | 2.114 s | 1,938 |
| 1001–2000 | 2.159 s | 1,897 |
| 2001–3000 | 2.177 s | 1,882 |

The broad steps-100–3000 estimate used for the AUC projection is
**1,910.591879 samples/s**. Corresponding early steps 101–300 differ by under
0.5% between runs (about 2,012 versus 2,003 samples/s). These are measurements
at **local 1024/global 4096**, with evaluation disabled and per-step logging.
They do not measure local-2048 throughput or later-window cost.

Two log metadata fields require correction when reading raw output: MLLOG's
`submission_platform=MI355X` is a stale template label, and the planner's
printed batch 512 is its estimation default. Runtime global batch is 4096
with local batch 1024; actual hardware is documented in [§1](#1-host-and-software-stack).

### 5.4 Relation to A0 and B0

A0 captured nonfinite dense gradients with a finite forward loss. B0 has a
positive attention-backward corruption reproducer suppressed by cap256, yet
capped training also fails. Later B0 captures already receive huge gradients;
the downstream GEMM, layer-norm overflow and exact jagged split are not
established producers. The shared NaN root cause remains unresolved.

The earlier single-GPU runs used quarter-size tables. Moving to four GPUs
and full tables changes vocabulary, sharding, initialization, global batch,
window eligibility and embedding communication. Hardware, firmware, host
kernel and execution history also differ. These experiments do not show
that full tables fix NaN or that shrinking tables caused it. No same-host
quarter/full-table A/B or capped/uncapped A/B was performed.

## 6. Reproduction and analysis commands

The commands below use the **already prepared host-local container, dataset
and saved launch scripts**. They are recorded for future reproduction; this
documentation update does not launch another experiment.

For a source-matched repeat, first ensure the checkout mounted at `/workspace`
is at the recorded commit `82e3d1c87bbb409c2679977f3b1677e04bbc21a1`.
The branch can advance after these experiments. Because the checkout is bind
mounted, retaining the prepared image alone does not freeze the trainer source.

### 6.1 Build inputs

The checked-out source is the branch/commit in [§1.2](#12-runtime-and-source-pins).
The image was built with these inputs:

```bash
cd /home/chcai/training/recommendation
docker build -f Dockerfile.amdprimus0815 \
  --build-arg BASE_IMAGE=amdprimus/amdprimus:gfx1250-20260910 \
  --build-arg TRITON_COMMIT=7ff97e310935b4a79794878dbc911f9af25d38d9 \
  -t recommendation-mi450:0910-7ff-full .
```

The saved image digest identifies what actually ran. A future rebuild is not
guaranteed byte-identical because some package dependencies in the Dockerfile
are not fully locked. Preserve the recorded image and provenance when making
a controlled comparison.

### 6.2 Launch, stop and summarize

```bash
# Use a new name; the launcher refuses to overwrite or resume an existing run.
CONTAINER=mi450-fullmodel \
HSTU_BWD_MAX_VGPR=0 DIE_AT_STEP=3000 METRIC_LOG_FREQ=1 \
HIP_VISIBLE_DEVICES=0,1,2,3 DLRM_DATA_PATH=/data/mlperf_dlrm_v4 \
python3 /home/chcai/runs/mi450_4gpu_setup/run_monitored.py repeat02
```

This calls `launch_full_model.sh`, uses four GPUs and the full tables, clears
the image's allocator default, and creates
`/home/chcai/runs/mi450_4gpu_repeat02/`. It saves the effective settings,
source hashes, provenance, training logs, live status and final outcome.
Host GPU/kernel logs use the `repeat02` prefix in the setup directory.
Use this launcher for matching runs: a default one-GPU launcher can shrink
the vocabulary or restore different allocator settings.

Manual stop, if needed:

```bash
touch /home/chcai/runs/mi450_4gpu_repeat02/stop.request
```

The supervisor verifies ownership before signaling the trainer process group.
It does not terminate the whole container or unrelated jobs. A separately
named `HSTU_BWD_MAX_VGPR=256` launch would be a new experimental arm; neither
completed run used it. `BATCH_SIZE` is explicitly fixed to 1024 by this
launcher, so merely exporting 2048 outside it does not create the proposed
larger-batch experiment.

Refresh the completed repeat's CPU-only summaries:

```bash
python3 /home/chcai/runs/mi450_4gpu_setup/summarize_run.py \
  --run-dir /home/chcai/runs/mi450_4gpu_repeat01 \
  --telemetry-prefix repeat01
python3 /home/chcai/runs/mi450_4gpu_repeat01/audit_fullmodel.py
python3 /home/chcai/runs/mi450_4gpu_setup/compare_full_model_runs.py
```

These inspect saved artifacts. The three GPU health-check reports are already
retained separately; their script is `health_preflight.py` in the setup
directory. A successful small health check is not a substitute for the
training and full-allocation audits.

## 7. AUC 0.75 projection at local batch 2048

**This section contains an estimate, not an executed experiment.** The MI350
reference used **1024 per GPU across eight GPUs**, hence **global batch 8192**.
The proposed **2048 per GPU across four GPUs** has the same global batch.
The reference did not use global batch 1024.

The first `eval_accuracy >= 0.75` event in each of the 20
[`gbs_8192` RCP logs](../../rcp_logs/gbs_8192/) gives:

| Recorded MI350 result | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Training samples at first crossing | 61,931,520 | 69,271,552 | 80,281,600 |
| Optimizer steps at global 8192 | 7,560 | 8,456 | 9,800 |
| Wall time from `run_start` | 2h 22m 53s | 2h 39m 37s | 2h 59m 52s |

All 20 crossed the target. These are first observed crossings on an evaluation
grid, not interpolated exact crossing times. The frozen logs continue after
convergence and intentionally have no `run_stop`; the first qualifying eval
is the analysis endpoint. Their wall times include every periodic evaluation
and work before the first training block.

Assuming this host preserves the measured **1,910.591879 samples/s** when
local batch doubles, an 8192-sample step would take **4.287677 s**. Training
time is reference samples-to-target divided by this throughput.

For evaluation, retain the reference intervals from eval number 25 onward:
`SKIP_EVAL_EPOCH_PCT=0.025`, `EVAL_EVERY_DATA_PCT=0.001`. At global batch 8192,
the cadence is 280 steps / 2,293,760 training samples, and the first retained
eval is at 57,344,000 samples. Target crossings require 3–11 retained full
holdout evaluations. The per-seed calculation is:

```text
T_train = samples_at_first_AUC_crossing / 1910.591879
R_eval  = reference_retained_eval_seconds / reference_training_block_seconds
T_total = T_train * (1 + R_eval)
```

| Conditional projection on this host | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Training | 9.0041 h | 10.0713 h | 11.6720 h |
| Retained evaluation | 0.4282 h | 0.8606 h | 1.5212 h |
| **Total** | **9.4324 h** | **10.9318 h** | **13.1932 h** |

**About 11 hours plus startup is the planning estimate.** Running every
reference evaluation instead projects to about 14.3 hours on average. New
initialization, JIT and evaluation-related startup scans are not included.

The estimate assumes memory fit at local 2048, roughly unchanged aggregate
sample throughput, comparable convergence at the same global batch, and
comparable evaluation/training cost. None was measured at local 2048 here.
Doubling batch does not imply doubling throughput. Local 1024 already peaked
at 406.4 of 432 GiB on one GPU; without active/reserved memory separation,
these measurements establish neither local-2048 fit nor an inevitable OOM.
The RCP target windows (69–74) are beyond the measured window 46, so later
sequence work may also change throughput. The range is an empirical reference
seed range under these assumptions, not a prediction interval or guarantee.

## 8. Evidence inventory and limits

The paths below are **local to the experiment host**. Raw logs, dataset and
container images are not embedded in this Markdown file.

| Artifact root | Purpose |
|---|---|
| `/home/chcai/runs/mi450_4gpu_setup/` | Build, dataset checks, host inventories, launch/analysis scripts, kernel/GPU telemetry, health checks and RCP analysis |
| `/home/chcai/runs/mi450_4gpu_baseline/` | Initialization-only attempt and full-allocation proof |
| `/home/chcai/runs/mi450_4gpu_baseline_noeval/` | First completed 3,000-step run |
| `/home/chcai/runs/mi450_4gpu_repeat01/` | Completed fresh-process repeat |

| File, relative to the indicated root | Evidence |
|---|---|
| Setup: `host_inventory.json`, `dmesg.before.log`, `documentation_inventory.json` | Run-time host/firmware/boot record and supplemental documentation-time CPU/OS/Docker snapshot |
| Setup: `docker_build.log`, `prepared_stack.json`, `prepared_stack.log` | Build details, exact runtime pins, full row counts and prepared GPU checks |
| Setup: `dataset_md5.log`, `dataset_verify.log`, `dataset_audit.json`, `dataset_window_steps.json` | Download integrity, finite-source scan, shapes/catalog sizes and global-step/window mapping |
| Setup: `launcher_cpu_validation.json` | Synthetic supervisor lifecycle checks, including ownership and cleanup |
| Setup: `health_preflight.json`, `health_postrun.json`, `health_repeat01_postrun.json` | Four-GPU small-check results before training and after each completed run |
| Setup: `{baseline,baseline_noeval,repeat01}.host_result.json` | Host launch/end times and launcher exit codes |
| Setup: `{baseline,baseline_noeval,repeat01}.{kernel.log,gpu.jsonl}` | Captured kernel stream and sampled GPU telemetry for each attempt |
| Setup: `repeat01.postrun_processes.json`, `repeat01.postrun_metrics.json` | Idle GPU state after repeat postflight |
| Initial attempt: `actual_embedding_allocation.json`, `outcome.json` | Full allocation and intentional stop before training |
| Each completed run: `effective_training_env.json`, `command.json`, `container.json`, `provenance.json`, `source_sha256.json` | Exact settings, invocation, container/software identity and captured source hashes |
| Each completed run: `training.log`, `driver.log`, `status.json`, `outcome.json` | Raw progress, bound marker, supervisor decision and cleanup |
| Each completed run: `summary.json`, `loss.csv` | Independently parsed consecutive losses, samples, memory/ECC, fault and termination checks |
| Each completed run: `audit_fullmodel.json`, `audit_fullmodel.md` | Separate actual-allocation, model, source, log and termination audit |
| Setup: `baseline_noeval_vs_repeat01.{json,csv,md}` | Full-precision paired losses, source/runtime/config comparison and coverage limits |
| Setup: `rcp_convergence_analysis.{json,csv}`, `rcp_host_projection.{json,md}` | Per-seed RCP crossings and conditional AUC-time calculation |

All final summary validation fields passed, and a separate audit agreed with
the consecutive loss count, full table allocations and intended termination.
Finite logged loss does not prove that every gradient or the final update is
finite; A0 previously captured bad gradients with a finite forward loss.
No full-state multi-rank replay, post-warmup run, full convergence run,
local-2048 memory test or controlled A0/B0 root-cause comparison was performed.
