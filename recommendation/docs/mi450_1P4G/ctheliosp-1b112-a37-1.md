# MI450 1P4G, host `ctheliosp-1b112-a37-1` (gfx1250, full model)

Updated **2026-09-19 after the 08:09 UTC completion**. The full yambda-5b HSTU
ranker completed **3,000 consecutive optimizer steps with finite logged loss**
on all four GPUs. The run allocated **all 11 FP32 embedding tables**, with
**293,051,712 rows × 512 dimensions = 558.95 GiB of embedding weights**.
`EMBEDDING_ROW_SCALE=1.0`; the model and vocabulary were not shrunk.

**No training-loss NaN was reproduced in this investigation on this host.**
The completed run started at timestamp zero, seed 1, with batch **1,024 per
rank / 4,096 global**. Both console and MLPerf loss streams contain exactly
steps 1–3,000. Final loss is **0.13069**. The earlier full-model run separately
recorded **1,530 finite steps** before an unexplained host interruption; the
completed run was a cold restart, not a continuation of those steps.

Training required **host-staged Socket communication** after direct GPU
communication hung and produced persistent MES errors. The Socket workaround
passed two fresh four-GPU preflights and then the full-model run. The fabric
registration mismatch remains unresolved. This result does not establish an
A0/B0 NaN fix, convergence, correctness of every intermediate gradient, or
working direct GPU fabric transport.

All experiment and capture times below are **UTC**. Firmware build dates are
quoted as reported; their source does not specify a timezone.

Both full-model training runs used repository commit
**`082e3849dd9588ee5fbfd61eeae8c0754311d05b`** on `chcai/mi450`.
The branch was subsequently pulled to **`9444bd4452b2c3343049dcf05a1954e4a37efcf7`**
for this documentation update and the companion 1P4G report. The newer source
and diagnostics have not been validated by these runs.

Companion records: [the other 1P4G host](ctheliosr-1b112-a37-2.md),
[A0](../mi450_a0/mi450_a0.md),
[B0](../mi450_b0/mi450_b0.md), and
[current A0 controls and evidence](../mi450_a0/mi450_a0.md#stack-and-controls).
Unless stated otherwise, artifact paths below are **host-local**, relative to
`/home/chcai/mi450_fullmodel_20260919/`; the large logs, images, wheels and
dataset are not embedded in this document.

## Start here

| Question | Recorded answer |
|---|---|
| Does the full model fit? | **Yes at local batch 1024 on four GPUs.** Actual allocations retain all rows and all 512 columns; embedding weights total **558.95 GiB**. |
| Did NaN loss reproduce? | **No.** One completed run has **3,000/3,000 finite console and MLPerf losses**; the prior interrupted run records 1,530 finite steps. |
| Was the model shrunk? | **No.** `EMBEDDING_ROW_SCALE=1.0`, full tables, three HSTU layers, sequence limit 4096. |
| Which attention arm ran? | **Uncapped**, `HSTU_BWD_MAX_VGPR=0`, exact Triton `7ff97e310935b4a79794878dbc911f9af25d38d9`. |
| Was training resumed? | **No.** Checkpointing was disabled; the completed run was a cold restart from timestamp zero and seed 1. |
| Is this 4,530 continuous steps? | **No.** The interrupted and completed runs are separate trajectories. The first run's raw log also ends before its monitor count. |
| What unblocked four-GPU training? | **Host-staged NET/Socket**, with P2P/SHM and direct fabric paths disabled; two complete preflight passes. |
| Is the fabric fixed? | **No.** Four PPOD commits still fail, with AFM/driver registration disagreement. Socket success does not isolate that failure's cause. |
| Was convergence or post-warmup stability measured? | **No.** Evaluation was disabled, and all 3,000 steps are inside the 24,000-step warmup. |

## Contents

1. [Host and software stack](#1-host-and-software-stack)
2. [Dataset and full model](#2-dataset-and-full-model)
3. [Exact training settings](#3-exact-training-settings)
4. [Experiment record](#4-experiment-record)
5. [Results and comparison](#5-results-and-comparison)
6. [Reproduction and analysis commands](#6-reproduction-and-analysis-commands)
7. [Fabric registration and recovery limits](#7-fabric-registration-and-recovery-limits)
8. [Evidence inventory and limits](#8-evidence-inventory-and-limits)

## 1. Host and software stack

### 1.1 Host, CPU, GPUs and firmware

| Item | Recorded value |
|---|---|
| Hostname | **`ctheliosp-1b112-a37-1.mnb.dcgpu`** |
| Host OS | CentOS Stream 9; supplemental documentation-time snapshot |
| CPU topology | **1 socket**, 128 cores, 256 present logical CPUs, **255 online (0–254), CPU255 offline**; `AMD Eng Sample: 100-000001046-06` |
| GPUs | **4 × MI450**, `gfx1250`, PCI `1002:75c1`, device revision `0x00`, 256 reported compute units per GPU, wave32 |
| PCI addresses | `0001:04:00.0`, `0002:04:00.0`, `0003:04:00.0`, `0004:04:00.0` |
| HBM | **432 GiB per GPU**, 1,728 GiB aggregate; reported as 442,368 MiB per device |
| IFWI | **`113-M4500001-640C`**, build **2026/08/19 23:29**, version `00198932`, name `AMD MI450X_GENERIC_SEC` |
| Kernel | `6.16.1-0_fbk2_brcmrdma5_35_g5ba27bd1d6b9` |
| amdgpu | `7.1.0.31300009`; installed source package `amdgpu-7.1.0-2404163.el9` |
| Completed-run boot ID | `a76a97be-2fed-4e62-8f55-b5c1bd2a4aea` |

The CPU/OS/Docker inventory was captured read-only at **08:56:04 UTC**, after
training, on the same boot, in `documentation_inventory_20260919.json`. The
nominal 256-thread topology is not 256 online CPUs. GPU/driver/IFWI identity
is preserved in `hardware.txt` and the run manifests. These PCI,
revision and architecture identifiers do **not** identify A0 versus B0 silicon.
No stepping-based explanation is claimed without board/tray records.

### 1.2 Runtime, image and source pins

| Component | Version / source |
|---|---|
| Base image | `amdprimus/amdprimus:gfx1250-20260910` |
| Base image digest | `sha256:6e656de79e6c8d7f3b3db3690606c3e6cb0536ec871f0521735746015297c9a1` |
| Final local image | **`recommendation-gfx1250-20260910:triton-7ff97e`** |
| Final image ID | `sha256:1ca53982ee9c71beab4131e01d2ba0c8dfc2329933961ee79726b93b47450e3e` |
| PyTorch | `2.11.0+rocm7.14.0a20260625`, Git `f55dda6ca78b73027840c1bc4014fe703d4f5473` |
| HIP / RCCL | `7.14.60850` / reported `(2, 30, 4)` |
| Triton | Runtime `3.8.0`; distribution `3.8.0+git7ff97e31`; exact clean source **`7ff97e310935b4a79794878dbc911f9af25d38d9`** |
| FBGEMM | Source `10b775730212923f65f7b78f79b6a01d80cf3c29`, the three gfx1250 patches from the repository recipe; installed distribution `2026.9.19` |
| TorchRec | `1.7.0a0+bf55480`, recipe tag `v2026.06.01.00` |
| Polars | `polars-u64-idx==1.33.1` |
| Docker client / server | **29.7.2 / 29.7.2**, supplemental documentation-time snapshot |
| Tested training source | `082e3849dd9588ee5fbfd61eeae8c0754311d05b` |

The native build follows
[`Dockerfile.amdprimus0815`](../../Dockerfile.amdprimus0815), using the September
10 base and preserving the built FBGEMM wheel/source/patch. Triton was built
from the exact pin and installed as a wheel with `--no-deps`; its tracked source
is also preserved in `/opt/triton-custom`. Native Torch/ROCm fingerprints match
the untouched base. CPU imports, FBGEMM cumsum and offline gfx1250 compilation
passed before GPU preflight.

Both FBGEMM and Triton wheels were exported to the host. Earlier uncommitted
build snapshots did not survive the first power cycle; the completed rebuild
and saved image supersede those temporary artifacts. See `IMAGE_BUILD_RESULT.md`,
`image-build-summary.json`, `fbgemm-artifacts/` and `triton-build/artifacts/`.

### 1.3 Container execution

Both full runs used the preserved image and a new named container. The saved
launcher and supplemental container inspection record **bridge networking**,
**host IPC**, nonprivileged execution, `/dev/kfd` and `/dev/dri`, supplementary
`video` group, and `seccomp=unconfined`. Inspection also records `label=disable`.
The four ranks communicate over loopback inside their container.

| Host path | Container path |
|---|---|
| `/home/chcai/training/recommendation` | `/workspace/recommendation` |
| `/home/chcai/dlrm_data` | `/data/mlperf_dlrm_v4` |
| Per-run evidence directory | `/evidence` |
| Per-run `results/` | `/workspace/recommendation/results` |

The command requests `--shm-size=64g`, but `--ipc=host` means host shared memory
is used; that option does not establish a private 64 GiB shared-memory
allocation. Trainer source is bind-mounted, so the retained image alone does
not freeze the source after a repository pull.

The image's allocator default is overridden: the launcher clears both
`PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF`, and explicitly selects:

```text
HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
```

## 2. Dataset and full model

### 2.1 Verified Yambda-5b cache

This experiment used the full model and full embedding vocabulary on four
GPUs. The A0/B0 single-GPU row-scale reduction was not carried over. The
dataset at `/home/chcai/dlrm_data` passed downloader MD5 checks and
the repository's `verify_dataset.sh` SHA256 verification. All 14 required cache
arrays had complete NPY payloads before training.

### 2.2 Architecture and precision

| Model setting | Value |
|---|---|
| HSTU layers / heads | **3 / 4** |
| Table / transducer width | **512 / 512** |
| Per-head attention Q/K and linear/value dimensions | **128 / 128** |
| Preprocessor hidden width | 256 |
| History / minimum history / sequence limit | **4086 / 4086 / 4096** |
| Dropout | Input **0.2**, linear **0.1** |
| Maximum candidates | 1 |
| Task | Binary `listen_plus`, task weight 1, causal multitask weight 0.2 |

Precision is mixed. Native TBE constructor records verify **FP32 embedding
weights and FP32 outputs**. The saved gin enables `make_model.bf16_training=True`
for HSTU training; this does not mean every model parameter or operation is
BF16. The manifest preserves the complete model configuration.

### 2.3 Full embedding tables and actual sharding

| Table | Full rows | Observed sharding |
|---|---:|---|
| `item_id` | 9,390,624 | Table-wise |
| `artist_id` | 1,293,395 | Four 128-column shards |
| `album_id` | 3,367,692 | Table-wise |
| `uid` | 1,000,001 | Four 128-column shards |
| `user_x_artist` | 100,000,000 | Table-wise |
| `user_x_album` | 40,000,000 | Table-wise |
| `user_x_hour` | 24,000,000 | Table-wise |
| `item_x_hour` | 40,000,000 | Table-wise |
| `artist_x_hour` | 32,000,000 | Table-wise |
| `user_x_is_organic` | 2,000,000 | Four 128-column shards |
| `user_x_artist_x_hour` | 40,000,000 | Table-wise |
| **Total** | **293,051,712** | **Full width 512 for every table** |

Eight actual TBE constructor records contain 20 shard records. Independent
reconstruction verifies complete rectangular coverage without gaps or overlaps,
FP32 precision, and DEVICE placement. Their total is exactly
**600,169,906,176 bytes = 558.9518 GiB of weights**. This is an allocation proof,
not just an environment-variable or planner claim. Optimizer state and
activations require additional memory.

| Rank | Embedding weights, GiB | Sampled used VRAM near the end, GiB |
|---|---:|---:|
| 0 | 253.8173 | 422.89 |
| 1 | 124.1176 | 302.65 |
| 2 | 96.2524 | 408.58 |
| 3 | 84.7646 | 396.95 |

Used-VRAM values are samples, not continuous peak measurements. Initial windows
0–5 have too few anchors for a global batch of 4,096; the first optimizer step
occurs in window 6. This is expected batching behavior and does not reduce the
model or vocabulary.

## 3. Exact training settings

### 3.1 Model, batching, optimizer and observation

| Setting | Effective value |
|---|---|
| World size / batch | **4 ranks**, **1,024 per rank**, **4,096 global** |
| Embeddings | **11 tables**, width **512**, **FP32**, `EMBEDDING_ROW_SCALE=1.0`, all HBM |
| Dense HSTU | **3 layers**, width **512**, **4 heads**, QK and linear dimensions **128**, BF16 HSTU training enabled |
| History / sequence | `HISTORY_LENGTH=4086`, `MIN_HISTORY=4086`, `MAX_SEQ_LEN=4096`, interleaved history |
| Dataset start / seed | `START_TS=0`, `NUM_TRAIN_TS=299`, seed **1**, cold start |
| Step limit | **3,000** optimizer steps; no per-window batch cap |
| Learning rate | Dense/sparse **1e-6**; 24,000-step warmup from zero; clipping norm **1** |
| Dropout | Input **0.2**, linear **0.1** |
| Attention | Triton, **uncapped** `HSTU_BWD_MAX_VGPR=0`, `HSTU_BWD_BLOCK_N=64` |
| Compiler / allocator | Buffer operations off, pipeline clamp, full autotune off, both PyTorch allocator variables empty |
| Placement planner | Fused HBM embeddings, `HBM_CAP_GB=400` per GPU |
| Observation | Console and MLPerf training loss every step; host observer stops on first reported nonfinite loss |
| Disabled | Evaluation, checkpointing, profiling, heavy NaN hooks, full-state capture/replay |

The saved gin selects dense Adam (betas 0.95/0.999, epsilon 1e-8, no weight
decay). Native TBE records select `EXACT_ROWWISE_ADAGRAD`, stochastic rounding
enabled and sparse `gradient_clipping=False`; the dense clipping setting is
not a claim that the fused sparse update is clipped.

**All 3,000 observed steps are inside the 24,000-step warmup.** Stability at
the full base learning rate was not tested. No holdout AUC was measured.

### 3.2 Socket transport and runtime controls

The principal communication/runtime overrides are:

```bash
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_NET=Socket
NCCL_PROTO=Simple
NCCL_ALGO=Ring
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=lo
GLOO_SOCKET_IFNAME=lo
RCCL_DDA_ENABLE=0
NCCL_MNNVL_ENABLE=0
NCCL_NVLS_ENABLE=0
RCCL_MSCCL_ENABLE=0
RCCL_MSCCLPP_ENABLE=0
RCCL_MSCCLPP_FORCE_ENABLE=0
GPU_MAX_HW_QUEUES=2
HSA_ENABLE_SDMA=1
HSA_NO_SCRATCH_RECLAIM=1
HIP_FORCE_DEV_KERNARG=1
HSA_ENABLE_IPC_MODE_LEGACY=0
```

The complete environment is saved as each run's `collective.env` and
`environment.txt`. The source file remains named `collective-socket-proposed.env`
and retains historical pre-test comments; the two saved preflight results
establish its later validation. Do not infer an untested status from that name.

Each passing preflight and the completed training run has **88 actual
`NET/Socket` channel records**. The training counts are 64 `NET/Socket/0` and
24 `NET/Socket/0/Shared`; `/Shared` denotes shared NET buffers, not SHM transport.
Checking only the banner `Using network Socket` is insufficient: an earlier
failed attempt printed that banner while channels still used P2P/IPC.

This workaround changes communication paths and timing. It keeps all four
GPUs and all embedding rows, but does not repair the underlying fabric
configuration or isolate one causal environment variable.

## 4. Experiment record

### 4.1 Driver initialization and failed communication attempts

Plain `modprobe amdgpu gpu_recovery=0` failed VCN firmware initialization with
`LOAD_IP_FW` status **0x11** and ring-test timeout **-110**. Searching the host
found the existing `/home/yanyuqin/load_driver.sh`, whose command succeeded:

```bash
sudo modprobe amdgpu noretry=0 gpu_recovery=0 ip_block_mask=0xcff
```

The mask disables **VCN/JPEG** and retains compute/UALink. It is a host-specific
driver-load workaround, not a numerical fix or a demonstrated recovery for an
already-stuck MES scheduler. No driver reload was performed during training.

| Attempt | Observed result | Interpretation |
|---|---|---|
| Initial base-image preflight | All four GPUs pass FP32/BF16 matmul, Triton add and 1,000 trials each of D4/D16 `index_select`/`gather`; RCCL initialization stalls. Stopping it is followed by GPU0 MES queue/TLB errors. | Individual kernels pass; distributed communication does not. The small Triton-add arm used stock Triton 3.6. |
| Collective overrides on the same failed boot | No valid collective pass. | Inconclusive for those overrides because the GPU state was already unhealthy. |
| Targeted GPU0 AMD-SMI reset | Enters mode-2 recovery, then kernel soft-lockup/RCU-stall symptoms and uninterruptible tasks. | Reset worsened the observed state; no successful local MES recovery was demonstrated. |
| Fresh preflight after the user's power cycle, `preflight_recovery_20260919T034707Z` | Final pinned image; individual checks pass on all four GPUs and RCCL initializes, but the first 1,024-element all-reduce times out after 90 seconds. All ranks report sequence 1 enqueued and -1 completed. GPU3 develops MES failures at about **03:47:16 UTC**. Container exits **137**. | Fresh reproduction of a communication/MES failure; larger collectives and all-to-all are not reached. No training NaN was observed. |
| After the next reboot | Same driver-load command succeeds; Socket preflight passes. | First validated four-GPU communication configuration for this investigation. |

The persistent errors include `REMOVE_QUEUE`, `ADD_QUEUE`,
`INVALIDATE_TLBS`, `MES might be in unrecoverable state`, and repeated ring-full
messages. The fresh GPU3 failure starts with queue-removal errors; the ordering
from another host's TLB-first failure should not be substituted here.

The recovery review found no locally documented, demonstrated procedure that
clears this state in-band. Reset interfaces existed, but their presence did not
establish a working procedure. The user performed the requested host recoveries;
this investigation did not change shared-fabric services or repeat the harmful
in-band reset after the later failure. This does not prove that every possible
reset method or warm reboot must fail.

### 4.2 Successful four-GPU preflights

Two fresh preflights passed with the final image and the same saved environment:

| Preflight directory | Boot | Result |
|---|---|---|
| `preflight_collective-socket-proposed_20260919T045707Z` | `2f0619db-6143-4474-bcb9-0b8a6b545be0` | **16/16 check groups on each of four ranks**, clean group destruction, Docker/tee **0/0**, no new MES fault. |
| `preflight_collective-socket-proposed_20260919T061801Z` | `a76a97be-2fed-4e62-8f55-b5c1bd2a4aea` | Same complete pass before the cold restart; Docker/tee **0/0**. |

The checks cover exact FP32/BF16 matmul, pinned-Triton addition, 1,000 trials
each of D4/D16 gather/index-select, two NCCL groups, all-reduce, broadcast,
all-gather, reduce-scatter, FP16 unequal/zero/reversed all-to-all splits, and
larger FP32/FP16/BF16 operations, including **16 MiB per rank FP16/BF16
all-to-all**. The numerical comparisons use independent CPU expectations.

### 4.3 Full-model run ledger

| Run | Start, UTC | End / observed progress | Verdict |
|---|---|---|---|
| `full4_start0_20260919T045742Z` | **04:57:42** | Monitor records **1,530 finite steps**; raw loss lines survive through **1,525**; a new boot begins around **06:12**. | **Interrupted**, no recorded NaN, 3,000-step target not reached, interruption cause unknown. |
| `full4_start0_20260919T062014Z` | **06:20:14** | Planned stop at about **08:09:28**, **3,000 finite steps**, final loss **0.13069**. | **Completed bounded negative NaN result**, independently verified. |

### 4.4 First full run: interrupted

The first run used the full tables and successful same-boot Socket preflight.
The saved observer records consecutive finite steps **1–1,530** and
`first_nonfinite=null`. Raw training/launcher logs survive through **1,525**
(loss **0.13790**); Docker's raw log survives through **1,522**. Short NUL tails
and an unclean/corrupted previous journal show interrupted persistence.

No retained GPU/MES fault, OOM, panic, orderly host shutdown, or reliable BMC
restart cause explains the interruption. Docker's exit **255** and finish time
around **06:12:13** were assigned during daemon restoration on the new boot;
they do not identify the original failure time or cause. Absence of a persisted
fault does not rule out an unpersisted one.

Checkpointing was disabled, so no model/optimizer/RNG checkpoint existed to
resume. The later run starts again from zero. Its core model fields and all
eight TBE constructor argument records match the first run. Printed losses
need not match exactly; bitwise determinism was not established.

Original interrupted-run files were preserved. The detailed report is
`full4_start0_20260919T045742Z/INTERRUPTED_RUN_REPORT.md`, with hashed evidence
under `interruption_evidence_20260919T061705Z/` in that directory.

### 4.5 Cold restart and evidence persistence

The completed run kept the same model, image and collective settings. Only
the host observer and outcome handling were strengthened: sync `train.log`
before atomically persisting monitor state every ten losses/every five seconds,
on nonfinite loss, and at completion. Persistence errors stay visible and
prevent a finite-pass classification; write failures cannot skip the
nonfinite stop action. No diagnostic GPU kernels or model changes were added.

## 5. Results and comparison

### 5.1 Completed run: 3,000 finite losses

| Independent final check | Result |
|---|---|
| Console loss stream | **3,000** records, exactly steps **1–3,000**, ordered, no gaps/duplicates, all finite |
| MLPerf loss stream | Same **3,000** steps, all finite, consistent with console rounding |
| Printed loss range | **0.11418–0.18071**; first **0.14109**, final **0.13069** |
| Completion marker | `die_at_step=3000 hit at train_ts=46 batch=190 global_step=3000` |
| Global sample count | **12,288,000** |
| Full-table allocation | Exact **600,169,906,176 FP32 weight bytes**, complete coverage on four ranks |
| Evidence persistence | Final sync covers all **363,226,507** training-log bytes; zero NULs, zero persistence errors |
| Observer sync overhead | 1,508 sync records; median **1.07 ms**, maximum **4.95 ms** host commit duration |
| Process outcome | Deliberate worker **42**, multiprocessing wrapper/Docker **1**, tee **0**, watcher **0**; launcher successful |
| Kernel evidence | No new GPU/MES fault, OOM, panic or lockup matched; **144 IFoE devlink-type notices** plus ordinary container-network messages retained |
| Independent verdict | `verified_3000_finite_logged_losses`, **zero issues** |

The exit-1 traceback follows the explicit step-limit `sys.exit(42)`. It is not
a NaN failure, and the exit code alone is not the evidence for completion.
Both loss streams, the stop marker, observer state and final statuses agree.

The live kernel collector covered **06:38:25–08:09:29 UTC**, supplementing the
initial and final snapshots. It stopped normally after `launcher.finished`.
Training/monitor sessions ended, no running training containers remained, and a later
sysfs sample showed approximately **0.162 GiB used VRAM per GPU**. This final
release of memory is a process-cleanup observation, not an additional GPU
health test.

### 5.2 Relation to A0 and B0

The recent A0/B0 NaN baselines and this host share the principal Torch, HIP,
exact Triton source, FBGEMM/TorchRec recipe, and HSTU compiler workarounds.
Matching those pins does not make the independently built images or execution
trajectories identical.

| Comparison | A0/B0 record | This host |
|---|---|---|
| Training failure | A0 uninstrumented NaN at **775**; recorder catches **38 nonfinite gradients at 101** after finite forward. B0 start-zero uninstrumented NaN at sampled **60/70/80**. | **3,000 finite logged losses** in the completed run; no gradient scan. |
| Model distribution | Single-GPU shrunken tables; maintained batch **1,024 global**. | Four ranks, full tables, **4,096 global**; different initialization, data distribution, allocation and communication. |
| Host software | Documented kernel **6.14**, amdgpu **7.1.1**. | Custom kernel **6.16.1**, amdgpu **7.1.0.31300009**. |
| IFWI | B0 **650D**; A0 value unrecorded. | **640C**; no ASIC-stepping inference. |
| Attention cap | A0 baseline uncapped; B0 cap **256** repairs a captured attention component but does not fix training. | **Uncapped**, `HSTU_BWD_MAX_VGPR=0`; success cannot be credited to cap256. |
| Allocator | Historical expandable-segment arms and later empty-allocator arms. | Both allocator variables empty, matching the later shared baseline. |
| Observation | A0 full-state/gradient recorder; B0 operation/magnitude captures, plus uninstrumented loss runs. | Uninstrumented training with per-step reduced-loss observation and host evidence capture. |

The latest pulled B0 notes add a standalone weighted-input-LN zero-DY
reproducer: `BLOCK_N=8` fails at iteration **115**, `BLOCK_N=1` passes
**1,000/1,000**, and restoring 8 fails at **212**. However, BN1 plus attention
cap256 still fails training at sampled **60/70**. The step-51 output-gradient
GEMM already receives huge finite `dout`; its producer remains unconfirmed.
B0's later **07:11 UTC** D16 gather/index-select health failure is another
observation without a saved failing-tensor proof or established link to
training. See the [B0 continuation](../mi450_b0/nan_triage_resume_20260919.md).
These later diagnostics and `WEIGHTED_LN_BWD_BLOCK_N` were not part of this
host's tested `082e384` source.

The supported conclusion is **no nonfinite logged training loss within this
3,000-step, start-zero, uncapped full-model run under the recorded configuration**.
It does not identify the root cause of earlier failures, rule out later failure,
validate every activation/gradient/updated embedding row, or establish
benchmark convergence. No evaluation was run. A0's full-state replay path is
single-rank and was not enabled on this four-rank run.

The remaining separate questions are the fabric provisioning mismatch/direct
communication failure, the cause of the interrupted first run, and whether
the A0/B0 component defects reproduce on this host under controlled inputs.
None was resolved merely by obtaining finite full-model losses.

## 6. Reproduction and analysis commands

### 6.1 Recheck the completed evidence without GPU work

```bash
python3 /home/chcai/mi450_fullmodel_20260919/verify_full_run.py \
  /home/chcai/mi450_fullmodel_20260919/full4_start0_20260919T062014Z
```

The verifier reads saved files and prints JSON. Exit **0** means verified finite
logged losses, **1** means incomplete/inconsistent evidence, and **2** means
nonfinite logged loss found. The saved result is that run's
`final-verification.json`. It checks model/shard coverage, both loss streams,
completion, transport, final synced-byte coverage and kernel evidence.

### 6.2 Recorded preflight and launch procedure

These are host-local helpers used for the investigation, not generic scripts
added to the repository. The driver command in §4.1 is used only when loading
an unloaded driver after recovery. A new GPU workload requires a healthy host;
the earlier passing boot is not proof of readiness on a later one.

```bash
bash /home/chcai/mi450_fullmodel_20260919/retry_preflight_after_recovery.sh \
  /home/chcai/mi450_fullmodel_20260919/collective-socket-proposed.env

# Use the newly completed, inspected passing preflight directory.
PREFLIGHT_DIR='/home/chcai/mi450_fullmodel_20260919/<successful-preflight-directory>' \
  bash /home/chcai/mi450_fullmodel_20260919/run_full_4gpu.sh
```

The launch guard requires the same boot, image and unchanged saved collective
environment as that preflight, all four passing ranks, and successful Docker/
tee statuses. It checks the dataset cache and full table dimensions, records
source hashes and effective configuration, then stops at first reported
nonfinite loss or 3,000 steps. Actual channel logs must confirm NET/Socket.

The launcher reads source from `/home/chcai/training/recommendation`. **That
checkout was pulled after the recorded run.** Running the command against it
now tests the new checkout, not an exact repetition of `082e384`. To repeat
the recorded source, use an isolated checkout at the recorded commit and a
separate copy of the launcher pointing `ROOT` to that checkout; preserve the
original launcher/evidence and record the chosen source in the new run.

## 7. Fabric registration and recovery limits

The disagreement was captured on multiple boots, including a before/after
provisioning sequence on boot `2f0619db-6143-4474-bcb9-0b8a6b545be0`, before
GPU preflight:

| Time, September 19 UTC | Captured state |
|---|---|
| **04:56:11**, after driver load | Driver and AFM agree on accelerator IDs **3,2,1,0**, phase **ACTIVE**. |
| **04:56:30–31**, node-agent provisioning | Agent tries IDs **7,6,5,4**; all four PPOD commits fail with `Invalid argument` and `UAL_SET_PPOD_CONFIG` status **0xFFFF0007**. |
| **04:56:44**, after failed provisioning | AFM reports IDs **7,6,5,4**, phase **PROVIDER**; driver retains IDs **3,2,1,0**, state **ACTIVE**. |

The active node-agent uses slot 1/controller `10.5.219.30`; a slot-0 backup is
historical configuration, not authority for changing the live service. The
controller object also reports alternating host and SONiC/Ryzen identities;
their meaning is unverified. No slot, controller, fabric service or shared
configuration was changed by this investigation.

This establishes a provisioning failure independently of a training workload.
It does **not** establish that the mismatch causes the collective hang, identify
which disabled path makes Socket work, or connect the mismatch to the A0/B0 NaN.
See `FABRIC_CONFIGURATION_FINDINGS.md` and `FABRIC_POSTREBOOT_2f0619db.md`.

The persistent-MES recovery observations and unsuccessful in-band reset are
recorded in [§4.1](#41-driver-initialization-and-failed-communication-attempts).
A successful Socket run establishes a usable workload configuration; it does
not demonstrate that the fabric mismatch is repaired or that an already-wedged
scheduler can recover without a host reboot.

## 8. Evidence inventory and limits

All paths in this table are relative to the host-local evidence root
`/home/chcai/mi450_fullmodel_20260919/`.

| Artifact | Purpose |
|---|---|
| `RESULTS.md`, `README.md`, `provenance.json` | Final outcome, investigation history, exact image/source/settings |
| `result-evidence-sha256.json` | SHA256 manifest of final run evidence and reports |
| `documentation_inventory_20260919.json` | Supplemental CPU/OS/Docker and stopped-container inventory, captured after training |
| `hardware.txt`, `host-load-driver.reference.sh` | GPU/IFWI/driver identity and successful load-script reference |
| `IMAGE_BUILD_RESULT.md`, `image-build-summary.json` | Native image build, preserved wheels, exact source pins and verification |
| `native-torch-fingerprint-comparison.txt` | Base/final native Torch and ROCm identity check |
| `dataset-verification.log`, `dataset-verification.ok` | Completed dataset integrity verification |
| `preflight-base.log`, `preflight-collective-env.log` | Initial failed communication attempts |
| `preflight_recovery_20260919T034707Z/`, `preflight-after-powercycle.launcher.log` | Fresh all-reduce timeout after the power cycle |
| `dmesg.reset-stall.txt`, `dmesg.fresh-preflight-stall.txt`, `MES_RECOVERY_REVIEW.md` | Failed in-band reset, persistent MES evidence and recovery review |
| `FABRIC_CONFIGURATION_FINDINGS.md`, `FABRIC_POSTREBOOT_2f0619db.md` | Registration mismatch, failed PPOD commits, ordered provisioning snapshots |
| `preflight_collective-socket-proposed_20260919T045707Z/` | First passing four-GPU Socket preflight |
| `preflight_collective-socket-proposed_20260919T061801Z/` | Passing preflight for the completed run's boot |
| `full4_start0_20260919T045742Z/` | Interrupted run, original logs and interruption evidence |
| `full4_start0_20260919T062014Z/` | Completed 3,000-step run and final verification |
| `COMPARISON_WITH_A0_B0.md` | Detailed comparison with the A0/B0 evidence available during execution |
| `run_full_4gpu.sh`, `verify_full_run.py`, `collect_kernel_log.py` | Recorded launch, independent saved-evidence verifier and durable kernel collector |

Within the completed run, start with `outcome.json`, `final-verification.json`,
`full-model.json`, `actual-embedding-shards.json`, `environment.txt`,
`loss-monitor.json`, and `train.log`. `kernel-live.log`, its metadata, and the
before/after `dmesg` snapshots retain the kernel evidence. `repo-commit.txt`
and `source-sha256.json` identify the tested source independently of later pulls.

The final verification and an independent review agree on full allocation,
consecutive finite losses and intended termination. No post-run GPU diagnostic
workload, continuous ECC-counter survey, multi-rank full-state replay,
post-warmup run or convergence run was performed. Those results must not be
imported from the companion host report.
