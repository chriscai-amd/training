# MI450 1P4G, host `ctheliosp-1b112-a37-2` (gfx1250, full model)

Updated **2026-09-20** after verified holdout convergence and finalization of
the post-AC-cycle run at **08:07:53 UTC**.
The requested filename is `ctheliosr-1b112-a37-2.md`; the hostname captured in
every experiment is **`ctheliosp-1b112-a37-2.mnb.dcgpu`**. This document retains
that distinction so the artifacts can be matched to the correct machine.
All experiment and capture times below are **UTC**. Firmware build dates are
quoted as reported; their source does not specify a timezone.

**The full-model run reached holdout AUC 0.7514703274 at step 14,000 on
2026-09-20 at 08:07:17 UTC.** MLPerf recorded `run_stop: success`; the trainer
and controller exited 0, with no owned processes remaining. All **14,000
unique logged training losses were finite**, covering **57,344,000 training
samples**. Launch-to-target time was **9h 9m 37.875s**.

This fresh run used **local batch 1024, global batch 4096, seed 1, dense and
sparse base LR 5e-7, and 24,000 warmup steps**. It retained **11 full embedding
tables, 293,051,712 logical rows, width 512 and 600,169,906,176 FP32 weight
bytes**. The three holdout readings were **0.7236417 → 0.7416119 → 0.7514703**.
See [§5.5](#55-completed-4k-holdout-convergence-run) for measured timings,
losses, memory and hardware observations.

The earlier two 3,000-step runs used base LR 1e-6 and disabled holdout
evaluation. Five subsequent local-2048/global-8192 attempts ended in OOM,
stall, an intentional initialization stop or driver failure. After the user
reported an AC cycle, the successful 4K run started from scratch on a new
boot; it did not resume any earlier trajectory. The historical local-2048
projection in [§7](#7-auc-075-projection-at-local-batch-2048) never became a
measured convergence result at that batch size.

**The A0/B0 NaN root cause remains unresolved.** This run establishes finite
logged loss and holdout convergence for its recorded configuration; it does
not certify every intermediate tensor or post-warmup behavior. Step 14,000
is still inside the 24,000-step warmup. Baseline settings and comparisons
below are explicitly separated from the latest run.

Companion investigations:
[MI450 A0](../mi450_a0/mi450_a0.md),
[MI450 B0](../mi450_b0/mi450_b0.md), and
[current A0 controls and evidence](../mi450_a0/mi450_a0.md#stack-and-controls).

## Start here

| Question | Recorded answer |
|---|---|
| Does the full model fit? | **Yes at local batch 1024 on four GPUs.** The converged run retained all embedding rows and peaked at approximately **427.81 GiB on one 432 GiB GPU**; see the telemetry-unit convention in §5.5. |
| Did NaN loss reproduce? | **No in the completed 4K runs:** two earlier 3,000-step checks and the latest **14,000-step** convergence run had finite logged losses throughout. |
| Was the model shrunk? | **No.** `EMBEDDING_ROW_SCALE=1.0`, full 512-wide tables, three HSTU layers and sequence limit 4096. Column sharding retains every vocabulary row. |
| Which attention arm ran? | **Uncapped baseline**, `HSTU_BWD_MAX_VGPR=0`, exact upstream Triton `7ff97e310935b4a79794878dbc911f9af25d38d9`; no cap256 training comparison. |
| Was training resumed? | **No.** Empty checkpoint path and fresh trainer processes in each launch. |
| What is the longest completed trajectory? | **14,000 steps in the latest fresh run.** The two earlier 3,000-step checks are separate trajectories. |
| Was holdout AUC 0.75 measured? | **Yes: 0.7514703274 at 08:07:17 UTC on September 20**, with MLPerf success and clean termination. |
| What happened at local batch 2048? | Five attempts on September 19; none reached holdout evaluation. See [§4.4](#44-local-2048global-8192-attempts). |
| What was the host state at completion? | No surviving owned trainer tasks; all four GPUs returned to sampled driver-used VRAM of **165 tool-reported MB**. Available total GPU ECC counters stayed zero; corrected CPU reports are recorded separately. |

## Contents

1. [Host and software stack](#1-host-and-software-stack)
2. [Dataset and full model](#2-dataset-and-full-model)
3. [Exact training settings](#3-exact-training-settings)
4. [Experiment record](#4-experiment-record)
5. [Results and comparison](#5-results-and-comparison), including [latest convergence statistics](#55-completed-4k-holdout-convergence-run)
6. [Reproduction and analysis commands](#6-reproduction-and-analysis-commands)
7. [AUC 0.75 projection at local batch 2048](#7-auc-075-projection-at-local-batch-2048)
8. [Evidence inventory and limits](#8-evidence-inventory-and-limits)

## 1. Host and software stack

### 1.1 Host, CPU, GPUs and firmware

The baseline host inventory was captured on **September 19 at 03:58:41 UTC**.
A supplemental read-only inventory at **08:32:37 UTC**, on the same boot, records OS, CPU
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
| Boot ID for the two morning baseline runs | `3cb36f24-0832-402b-af25-ff917fa8350f` |
| Recovery during the two morning baseline runs | No reboot, AC cycle, driver reload or GPU reset |
| Boot ID for the converged post-AC run | `6ef5adda-fa87-4225-9fa2-99f733c74604` |
| Post-AC driver recovery | AMDGPU loaded at **22:56:54 UTC September 19** with `noretry=0 gpu_recovery=0 ip_block_mask=0xcff`; four GPUs enumerated in SPX/NPS1 |

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
driver was already loaded for the morning baselines. After the later AC cycle,
AMDGPU had to be loaded explicitly; see [§4.5](#45-post-ac-recovery-and-fresh-4k-launch).
No persistent boot configuration was changed by that recovery.

### 1.2 Runtime and source pins

| Component | Exact recorded version or source |
|---|---|
| Checkout | `/home/chcai/training`, branch `chcai/mi450` |
| Training commit, morning baselines | `82e3d1c87bbb409c2679977f3b1677e04bbc21a1` |
| Training commit, converged post-AC run | `191adbccd350a427d44192f9c3005f64681f938f` |
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

The two morning baseline launches captured the same **150 source SHA-256
entries** and matched the checked-out
files. The existing A0/B0 kernel fixes and workarounds in that commit remain
part of the tested stack; the full-table result does not isolate their effects.

The converged run records the later commit above and **276 source hashes**,
identical to the final 8K attempt's manifest. Of the 150 paths shared with the
morning baseline manifest, **143 hashes match and seven differ**; the source
trees must not be described as identical. Changes include opt-in tripwire
instrumentation and a weighted-layer-norm backward block-size override. The
latest run explicitly sets `NAN_TRIPWIRE=0` and `WEIGHTED_LN_BWD_BLOCK_N=0`.
Its captured Torch, Triton, TorchRec and FBGEMM package versions match the
table, and the prepared image/container identity is retained.

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

**The three completed GBS4096 runs override the image's default allocator.**
Their launchers explicitly clear both allocator variables, overriding the
Dockerfile's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. They also set
the required library directory as shown below. The 8K allocator experiments
used the distinct settings recorded in §4.4.

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

The two morning baselines and the converged GBS4096 run use
`EMBEDDING_ROW_SCALE=1.0`, `EMB_PLACEMENT=hbm`, with both
`EMB_PLACEMENT_OVERRIDES` and `EMB_SHARDING_OVERRIDES` empty.
Every logical table has **512 FP32 columns**. The following layout was verified
from **eight native FBGEMM TBE initialization records in each of those runs**:

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

Sections 3.1–3.4 describe **the two morning 3,000-step baseline runs**. Their
saved explicit settings differ only in `RUN_NAME`: `baseline_noeval` versus
`repeat01`.
The complete machine-readable settings are each run's
`effective_training_env.json`; inherited environment is not exhaustively
represented by that file. The initialization-only attempt's one setting
difference is recorded in [§4](#4-experiment-record). The converged run's
configuration changes are listed in [§3.5](#35-converged-run-configuration).

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

### 3.5 Converged run configuration

Run: `mi450_4gpu_4k_auc075_post_ac_20260919T225739Z`. The saved
`effective_training_env.json`, `configuration_verification.json` and native
`allocator_runtime.json` establish these settings:

| Setting | Converged run |
|---|---|
| Local / global batch; accumulation | **1024 / 4096; 1** |
| Dense / sparse base LR | **5e-7 / 5e-7** |
| Warmup | **24,000 optimizer steps from zero**, unchanged step count |
| Training limit | `DIE_AT_STEP=-1`; no optimizer-step or train/eval batch cap |
| Evaluation | `EVAL_EVERY_DATA_PCT=0.001`, `EVAL_EVERY_N_WINDOWS=0` |
| Initial evaluation skip | `SKIP_EVAL_EPOCH_PCT=0.022361844716419856`; runtime skip threshold 12,507 |
| Actual holdout schedule | First at **step 12,880**, then every **560 steps / 2,293,760 training samples** |
| Holdout | Window **299**, one full window, **2,058,204 configured anchors**, no evaluation batch cap |
| Stop target | Finite holdout `eval_accuracy >= 0.75`, followed by MLPerf success and clean exit |
| Initialization / checkpoints | Fresh seed 1, `START_TS=0`, empty `CKPT_PATH`; checkpoint output disabled |
| Model / sharding | All 11 full tables, three HSTU layers, history 4086, sequence limit 4096; original automatic sharding, with the same concrete table placements as `repeat01` |
| Allocator | Both allocator aliases empty; native max split -1, GC threshold 0, memory fraction 1.0, expandable segments false |
| Kernel controls | Uncapped `HSTU_BWD_MAX_VGPR=0`; `WEIGHTED_LN_BWD_BLOCK_N=0`; NaN probes/tripwire disabled |
| MLPerf platform label | `MI450`, correcting the morning baseline's stale `MI355X` label |
| Monitor | Two-distinct-step nonfinite-loss stop; explicit OOM/GPU-fault detection; **1,200-second training-stall guard**, excluding startup and evaluation |

Dense Adam, fused row-wise Adagrad, clipping and FP16 sparse communication
retain the settings in §3.2. The last logged LR was
**2.916458333333333e-7 = 5e-7 × 13,999 / 24,000**, about **58.33%** of the
base rate. Convergence occurred before warmup finished. The sample-based
initial skip controls when evaluation begins; it does not predict when AUC
will cross the target.

## 4. Experiment record

Sections 4.1–4.4 refer to **September 19 UTC**. The fresh run in §4.5 continued
into September 20.

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

### 4.4 Local-2048/global-8192 attempts

Five fresh attempts followed the morning checks. Each retained the full model
and used physical local batch 2048 on four GPUs. **None reached a holdout
evaluation or established convergence.** The run directory names below are
relative to `/home/chcai/runs/`.

| Run directory | Recorded outcome |
|---|---|
| `mi450_4gpu_8k_auc075_20260919T201418Z` | **19 finite steps**, then a **17.82 GiB** HSTU-backward allocation failure and logged RCCL memory fault with original sharding. |
| `mi450_4gpu_8k_auc075_alloc1024_20260919T203058Z` | Same **19-step** failure. The intended max-split setting was shadowed by empty `PYTORCH_CUDA_ALLOC_CONF`; this did not test the intended allocator protection. |
| `mi450_4gpu_8k_auc075_cw_20260919T204609Z` | Full column-wise sharding; **80 finite steps**, then a sustained stall after **21:01:51.335**. Rank 1 was in ROCr scratch reclamation through `hipMalloc`/`index_select`; no logged NaN, OOM or GPU fault. Owned cleanup completed **21:08:28.167**. |
| `mi450_4gpu_8k_auc075_cw_gc70_20260919T211127Z` | Intentional initialization-only stop at **21:18:17.836**, with **zero optimizer steps**. GC threshold 0.7 parsed, but native memory fraction 1.0 disabled the GC call. |
| `mi450_4gpu_8k_auc075_cw_gc70_f99_20260919T212335Z` | **Five finite steps**, then MES/driver failure. Both allocator aliases correctly set max split 1024 MiB, GC 0.7 and memory fraction 0.99. MES `INVALIDATE_TLBS` errors began **21:34:48**, before the final loss at **21:34:53.164**. |

The last attempt retained non-zombie D/R tasks after SIGKILL, despite its
terminal controller label. Its saved thread audit establishes incomplete
cleanup on that boot. The subsequent AC cycle and new-boot preflight are the
recovery evidence; the old `stopped` label is not. The further prepared
`mi450_4gpu_8k_cw_maxsplit_setup` was not launched. Detailed failure, allocator
and cleanup records are linked from the host-local
[8K handoff](/home/chcai/runs/mi450_4gpu_8k_handoff.md).

### 4.5 Post-AC recovery and fresh 4K launch

After the user reported an AC cycle, the boot ID was
`6ef5adda-fa87-4225-9fa2-99f733c74604`. All four GPUs were present on PCI, but
AMDGPU and `/dev/kfd` were absent because the existing boot configuration
blacklists automatic driver loading. The recorded restore command was
`sudo -n modprobe amdgpu noretry=0 gpu_recovery=0 ip_block_mask=0xcff`; it
completed at **22:56:54.826 UTC September 19**, without a persistent boot
configuration change. The existing container, image and mounts were retained.

All four ranks passed a bounded FP32/BF16 indexing/matmul and RCCL
all-reduce/all-to-all preflight in **3.959 seconds**. The host wrapper began
at **22:57:39.448 UTC** and recorded the detached training launch at
**22:57:39.634 UTC**. The prepared 4K run had not launched before the AC cycle,
and no checkpoint existed. This was fresh initialization with the settings
in §3.5, original automatic sharding and default allocator behavior.

The run directory is
`/home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z/`.
The saved [recovery record](/home/chcai/runs/mi450_4gpu_4k_auc_setup/POST_AC_RECOVERY.md)
and `configuration_verification.json` establish recovery and setup;
`outcome.json` and `finalization.json` establish the subsequent successful
completion. The 1,200-second stall guard never fired, and no manual stop or
automatic relaunch was recorded.

## 5. Results and comparison

Sections 5.1–5.4 retain the **two morning 3,000-step baseline results**.
The completed September 19–20 convergence run is documented in §5.5.

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

### 5.5 Completed 4K holdout convergence run

**Verified success:** finite holdout AUC **0.7514703273773193** at step
**14,000**, followed by MLPerf `run_stop: success`, trainer/controller/host
launcher exit **0**, and an empty owned-process and whole-group survivor
census. Finalization completed at **2026-09-20 08:07:53.377 UTC**.

The run processed **57,344,000 training samples**, corresponding to logged
epoch fraction **0.0250319161**. Its **14,000 MLPerf loss events** are
consecutive, with no missing steps, duplicate steps, malformed complete
MLPerf events or nonfinite losses. The controller's **28,000 reports** count
both console and MLPerf output; they are not 28,000 optimizer steps.

| Loss statistic | Value |
|---|---:|
| First loss, step 1 | 0.14112599194049835 |
| Last loss, step 14,000 | **0.10529641807079315** |
| Mean over all 14,000 steps | 0.1295759480819106 |
| Minimum, step 13,820 | 0.08038505166769028 |
| Maximum, step 3,756 | 0.1869322508573532 |
| Nonfinite logged losses / evaluations | **0 / 0** |

All three scheduled holdout evaluations completed. Sample counts below are
**training samples accumulated before evaluation**, not holdout sizes. The
fixed holdout configuration contains 2,058,204 anchors with no batch cap;
the final evaluation log records 502 batches.

| Step | Training samples | Eval start, September 20 UTC | Result time UTC | Eval duration | Holdout AUC |
|---:|---:|---|---|---:|---:|
| 12,880 | 52,756,480 | 07:03:42.498 | 07:10:52.135 | 429.637 s | 0.7236416935920715 |
| 13,440 | 55,050,240 | 07:31:54.190 | 07:39:03.338 | 429.148 s | 0.741611897945404 |
| **14,000** | **57,344,000** | **08:00:07.599** | **08:07:17.323** | **429.724 s** | **0.7514703273773193** |

Timing uses embedded MLPerf event timestamps and the host wrapper's recorded
launch/completion times. The first training block began at **23:07:48.161
UTC September 19**; the first loss followed at **23:08:42.209**. The final
training loss was recorded at **08:00:07.588 UTC September 20**.

| Timing scope | Measured duration | Aggregate training samples/s |
|---|---:|---:|
| Host launch → first training block | 10m 8.713s | — |
| Sum of three training blocks | **8h 38m 0.652s** | **1,845.01** |
| Sum of three full evaluations | **21m 28.509s** | — |
| MLPerf `run_start` → successful `run_stop` | **9h 9m 27.096s** | **1,739.43** |
| Host launch → qualifying AUC | **9h 9m 37.875s** | — |
| Host launch → host wrapper completion | **9h 9m 50.633s** | **1,738.19** |

Training-block throughput excludes startup and the logged evaluation
intervals, but includes loading, synchronization and other waits inside those
blocks. It is not pure GPU compute throughput. The last two 560-step training
blocks took **1,262.055 s** and **1,264.260 s**, about **2.26 s/step**; adding
the roughly **7m 10s** holdout produced results about **28m 13s** apart.

The 05:00 UTC projection had a 10:00 UTC working ETA before any holdout AUC
was known. After two measured evaluations, the saved 07:45 projection moved
the planning window to **08:07–08:35 UTC**, with the next evaluation as the
best estimate. The actual crossing at **08:07:17 UTC** is now the result;
those projections remain historical assumptions rather than current status.

There were **3,104 saved telemetry samples per GPU**, from September 19
**22:57:39.449** through September 20 **08:07:29.412 UTC**, with no telemetry
call failures. Peak sampled memory was higher than in the morning baselines:

| GPU | Peak used VRAM, tool-reported MB | Approximate peak GiB | Peak timestamp UTC | Final used VRAM, tool-reported MB |
|---|---:|---:|---|---:|
| 0 | **438,077** | **427.81** | September 19 23:14:09.044 | 165 |
| 1 | 415,863 | 406.12 | September 19 23:10:04.497 | 165 |
| 2 | 435,563 | 425.35 | September 19 23:14:09.044 | 165 |
| 3 | 408,217 | 398.65 | September 20 07:11:16.003 | 165 |

AMD-SMI labels these values `MB`, including **442,368 MB** total per GPU.
The approximate GiB column uses the same binary-capacity interpretation as
§5.2: divide the reported count by 1024, consistent with the independently
recorded **432 GiB** capacity. The smallest sampled headroom was about
**4.19 GiB** on GPU 0. Sampling can miss instantaneous peaks and does not
separate active tensors from allocator cache.

All available **total correctable, uncorrectable and deferred GPU ECC
counters remained zero on all four GPUs**. Per-block/cache and other
unavailable counters remain `N/A`. A review of the saved training and kernel
logs found **no GPU fault, OOM or reset signature**. The kernel log separately
contains **260 corrected CPU error reports**: CPU 218 **101**, CPU 106 **99**,
and CPU 73 **60**, each reporting “Corrected error, no action required.”
Suppressed MCE callback messages and `ifoe` devlink-port warnings are also
retained. These observations do not establish training causation and are not
GPU ECC events.

At recorded completion, `termination_incomplete=false`, both survivor lists
were empty, and sampled GPU memory had returned to its idle level. These are
the saved completion observations; no additional post-run GPU health probe
was performed for this documentation update. The run establishes convergence
at the recorded settings, while the lack of intermediate-tensor validation,
post-warmup training and a controlled A0/B0 root-cause comparison remains.

Primary evidence: [final result](/home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z/RESULT.md),
[summary](/home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z/summary.json),
[MLPerf events](/home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z/mlperf.log),
[per-step losses](/home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z/loss.csv),
and [finalization](/home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z/finalization.json).

## 6. Reproduction and analysis commands

The commands below use the **already prepared host-local container, dataset
and saved launch scripts**. They are recorded for future reproduction; this
documentation update does not launch another experiment.

For a source-matched repeat, the morning baselines used commit
`82e3d1c87bbb409c2679977f3b1677e04bbc21a1`; the converged run used
`191adbccd350a427d44192f9c3005f64681f938f`. Match the intended run's captured
source manifest and configuration in the checkout mounted at `/workspace`.
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

### 6.2 Baseline launch, stop and summarize

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
morning baseline used it. `BATCH_SIZE` is explicitly fixed to 1024 by this
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

### 6.3 Convergence-run artifacts and reproduction setup

The separate
[/home/chcai/runs/mi450_4gpu_4k_auc_setup/README.md](/home/chcai/runs/mi450_4gpu_4k_auc_setup/README.md)
records the 4K convergence launcher and watcher commands, including the
1,200-second training-stall timeout. Its initial “run remains active” text is
a launch-time observation; the final artifacts in §5.5 supersede it.
Use this setup's LR/evaluation configuration for a matching convergence run;
the baseline launcher above intentionally stops after 3,000 steps.

The completed run's saved logs can be summarized without GPU work:

```bash
python3 /home/chcai/runs/mi450_4gpu_4k_auc_setup/summarize_4k.py \
  --run-dir /home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z \
  --observations-dir /home/chcai/runs/mi450_4gpu_4k_auc_setup/observations/auc075_post_ac_20260919T225739Z
```

## 7. AUC 0.75 projection at local batch 2048

**Historical projection prepared before the 8K attempts.** Those subsequent
attempts are recorded in §4.4; none produced a measured time to AUC 0.75.
The completed run in §5.5 used global batch 4096 and has its own measured
result. The MI350 reference used **1024 per GPU across eight GPUs**, hence
**global batch 8192**.
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

**About 11 hours plus startup was the planning estimate.** Running every
reference evaluation instead projects to about 14.3 hours on average. New
initialization, JIT and evaluation-related startup scans are not included.

The historical estimate assumed memory fit at local 2048, roughly unchanged
aggregate sample throughput, comparable convergence at the same global batch, and
comparable evaluation/training cost. The original-sharding attempts later
failed allocations, and the full-column-wise attempts did not establish
sustained training or reach evaluation. These assumptions therefore remain
unvalidated for local 2048. Doubling batch does not imply doubling throughput.
The projection used only the morning measurements through window 46, before
the RCP target windows 69–74. Its range is an empirical reference-seed range
under those assumptions, not a prediction interval or a current ETA.

## 8. Evidence inventory and limits

The paths below are **local to the experiment host**. Raw logs, dataset and
container images are not embedded in this Markdown file.

| Artifact root | Purpose |
|---|---|
| `/home/chcai/runs/mi450_4gpu_setup/` | Build, dataset checks, host inventories, launch/analysis scripts, kernel/GPU telemetry, health checks and RCP analysis |
| `/home/chcai/runs/mi450_4gpu_baseline/` | Initialization-only attempt and full-allocation proof |
| `/home/chcai/runs/mi450_4gpu_baseline_noeval/` | First completed 3,000-step run |
| `/home/chcai/runs/mi450_4gpu_repeat01/` | Completed fresh-process repeat |
| `/home/chcai/runs/mi450_4gpu_8k_handoff.md` | Index of the five 8K attempts, failure evidence and incomplete cleanup before the AC cycle |
| `/home/chcai/runs/mi450_4gpu_4k_auc_setup/` | Post-AC recovery, driver and preflight records, convergence launch/monitor scripts |
| `/home/chcai/runs/mi450_4gpu_4k_auc075_post_ac_20260919T225739Z/` | **Completed 14,000-step run, holdout AUC 0.7514703274 and clean termination** |
| `/home/chcai/runs/mi450_4gpu_4k_auc_setup/observations/auc075_post_ac_20260919T225739Z/` | Latest run's host timing, kernel log and GPU telemetry |

| File, relative to the indicated root | Evidence |
|---|---|
| Setup: `host_inventory.json`, `dmesg.before.log`, `documentation_inventory.json` | Run-time host/firmware/boot record and supplemental documentation-time CPU/OS/Docker snapshot |
| Setup: `docker_build.log`, `prepared_stack.json`, `prepared_stack.log` | Build details, exact runtime pins, full row counts and prepared GPU checks |
| Setup: `dataset_md5.log`, `dataset_verify.log`, `dataset_audit.json`, `dataset_window_steps.json` | Download integrity, finite-source scan, shapes/catalog sizes and global-step/window mapping |
| Setup: `launcher_cpu_validation.json` | Synthetic supervisor lifecycle checks, including ownership and cleanup |
| Setup: `health_preflight.json`, `health_postrun.json`, `health_repeat01_postrun.json` | Four-GPU small-check results before training and after each morning baseline |
| Setup: `{baseline,baseline_noeval,repeat01}.host_result.json` | Host launch/end times and launcher exit codes |
| Setup: `{baseline,baseline_noeval,repeat01}.{kernel.log,gpu.jsonl}` | Captured kernel stream and sampled GPU telemetry for each attempt |
| Setup: `repeat01.postrun_processes.json`, `repeat01.postrun_metrics.json` | Idle GPU state after repeat postflight |
| Initial attempt: `actual_embedding_allocation.json`, `outcome.json` | Full allocation and intentional stop before training |
| Each completed run: `effective_training_env.json`, `command.json`, `container.json`, `provenance.json`, `source_sha256.json` | Exact settings, invocation, container/software identity and captured source hashes |
| Each completed run: `training.log`, `driver.log`, `status.json`, `outcome.json` | Raw progress, completion markers, supervisor decision and cleanup |
| Each completed run: `summary.json`, `loss.csv` | Independently parsed consecutive losses, samples, memory/ECC, fault and termination checks |
| Each morning baseline: `audit_fullmodel.json`, `audit_fullmodel.md` | Separate actual-allocation, model, source, log and termination audit |
| Setup: `baseline_noeval_vs_repeat01.{json,csv,md}` | Full-precision paired losses, source/runtime/config comparison and coverage limits |
| Setup: `rcp_convergence_analysis.{json,csv}`, `rcp_host_projection.{json,md}` | Per-seed RCP crossings and conditional AUC-time calculation |
| 4K AUC setup: `POST_AC_RECOVERY.md`, `post_ac_host_inventory.json`, `post_ac_20260919T225636Z.driver_load.json` | New boot, restored AMDGPU and retained container identity |
| 4K AUC setup: `health_post_ac_20260919T225706Z.json`, `active_launch.json` | Passed four-GPU preflight and recorded detached launch |
| Latest run: `configuration_verification.json`, `allocator_runtime.json`, `source_sha256.json`, `provenance.json` | Runtime settings, full allocation/sharding, native allocator and later source/software provenance |
| Latest run: `mlperf.log`, `eval_accuracy.json`, `mlperf_run_stop.json` | Three holdout results, exact interval timing and successful MLPerf termination |
| Latest run: `summary.json`, `loss.csv`, `RESULT.md`, `outcome.json`, `finalization.json` | Verified finite coverage, final statistics, clean exits and completed finalization |
| Latest run: `eta_projection_20260920T0500Z.json`, `eta_projection_20260920T074522Z.json` | Historical ETA assumptions before evaluation and after two measured holdout results |
| Latest observations: `host_result.json`, `kernel.log`, `gpu.jsonl` | Host launch/completion, corrected CPU reports and sampled GPU memory/ECC |

The latest terminal artifacts establish finite loss coverage, qualifying
holdout AUC, MLPerf success and clean trainer/controller termination. The
runtime configuration audit establishes full table allocations and actual
sharding. The saved kernel and telemetry review supports the stated observed
fault/ECC findings; controller success alone is not a hardware-health audit.

Finite logged loss does not prove that every gradient or the final update is
finite; A0 previously captured bad gradients with a finite forward loss. The
latest run establishes convergence **during warmup** at global batch 4096.
No full-state multi-rank replay, post-warmup run, successful global-8192
convergence run or controlled A0/B0 root-cause comparison was performed.
