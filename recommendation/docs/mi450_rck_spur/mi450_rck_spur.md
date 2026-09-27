# MI450 A0 on the RCK spur cluster (gfx1250, four GPUs, full model)

Updated **2026-09-27, 02:35 UTC** (all times UTC), while the corrected-driver
run is still in progress.

This cluster (hosts `ctheliosr-rck-g02-j19-*` and `ctheliosp-rck-g02-h21-*`,
scheduled by ROCm spur) was new to this investigation on September 24. The
work asked three questions: does the A0 NaN still reproduce on this cluster's
BKC, can the unshrunk four-GPU model train NaN-free for several hours, and
which component produces the NaN. Companion records:
[MI450 A0](../mi450_a0/mi450_a0.md) (one GPU, shrunk model),
[A0 CWSR defect and fix](../mi450_a0/cwsr_bank_corruption_fix.md),
[MI450 B0](../mi450_b0/mi450_b0.md), and the four-GPU convergence reference
[1P4G a37-2](../mi450_1P4G/ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md), whose configuration
every run here reuses.

## Current status: corrected driver clean past 2,000 steps with the trigger active

**On the corrected CWSR handler, the default configuration has run 2,128 finite
steps and is still going.** The run started at 01:32:50 on September 27 on
`j19-9`, with AutoNUMA on and no memory binding. On the stock driver, the same
configuration went NaN at **step 111** (`j19-1`) and **step 123** (`j19-11`).
The trigger is fully present: AutoNUMA averaged **2.2 million page-table
updates per minute** over the first 50 training minutes (2.5 million in the
stock `j19-1` run), and KFD queue evictions, each of which saves the running
waves through the handler, added 18.5 seconds of evicted time across the four
ranks. The run continues toward AUC 0.75 as the several-hour end-to-end run
([§5.2](#52-corrected-driver-run-on-j19-9)).

**This is the strongest evidence so far that the handler is the culprit, but it
is one run and still in progress.** Hardware, firmware, software stack,
configuration and trigger are the same as in the stock runs; only the handler
and its two size bytes changed ([§7](#7-corrected-cwsr-driver)). Both stock
default runs failed by step 123. A stock run with memory binding once reached
1,912 steps, so run length alone is not conclusive.

**Memory binding is not a mitigation.** A `numactl --membind=0,1` run on the
stock driver on `j19-9` went NaN at **step 354**, with zero AutoNUMA page-table
updates in the minutes before. KFD kept evicting the trainer's queues
throughout that run (6.9–10.8 seconds of evicted time per rank by the failure).
AutoNUMA is the largest source of evictions here but not the only one, so
binding removes one trigger, not the event that runs the faulty handler
([§6.4](#64-what-numactl---membind01-does-and-doesnt-do)).

**The stalls are a separate firmware problem, and they also block reboots.** A
bound tripwire run on `j19-9` wedged at 20:57:47 on September 26: GPU `0004`'s
firmware scheduler (MES `0x7b`) stopped responding to `INVALIDATE_TLBS` and
then `REMOVE_QUEUE`. With `halt_if_hws_hang=1`, each waiting driver thread
halts forever by design, so queue restores and other KFD work piled up, the
ranks could not exit and one GPU read 100% busy for four hours. Clearing the
parameter at runtime released four of the five stuck processes; about two
minutes later the node went dark with a firmware-recorded hardware error and
reset itself ([§8.2](#82-mes-non-response-and-halt_if_hws_hang)).

**The first non-finite tensor is the one the A0 investigation identified.** On
`j19-11` the NaN tripwire found every input finite and exactly one non-finite
dense gradient out of 69, on all four ranks:
`_hstu_transducer._input_preprocessor._additional_embedding_mlp.2.weight`
(`[512, 256]`). A0's attempt-405 raw-DW capture (single GPU, shrunk model) has
its 640 NaNs in the gradient of the same parameter.

**The installed driver carries the known-faulty CWSR trap handler.** Every node
checked embeds the 5,656-byte gfx1250 handler with SHA-256 `0f718b5e…` in
amdgpu 7.1.1, and the DKMS source's `L_NOT_WAVE_START` still has the unpatched
cross-bank lane-0 copy. The corrected handler (`68c31ab2…`) is not in this BKC.

| Question | Recorded answer |
|---|---|
| Does the NaN reproduce on this cluster? | **Yes, on the stock driver**: default runs at steps 111 and 123, and a `numactl`-bound run at step 354. |
| Was the model shrunk? | **No.** `EMBEDDING_ROW_SCALE=1.0` on four GPUs with the a37-2 settings. |
| Which component? | **The gfx1250 CWSR trap handler in amdgpu (KFD).** The corrected-driver default run is clean past 2,128 steps with the trigger active; it is still running ([§6](#6-nan-triage-component-trigger-and-mechanism)). |
| What triggers it? | KFD queue evictions, which save running waves through the handler. AutoNUMA is the largest source here; evictions continue without it. |
| Is there a mitigation short of the fix? | **No.** `numactl --membind=0,1` still went NaN at step 354. |
| Several-hour end-to-end run? | **In progress** on the corrected driver: 2,128 steps (about 50 minutes of training) at 02:35. |
| Why do nodes stall and fail to reboot? | MES `0x7b` stops responding and `halt_if_hws_hang=1` freezes the driver, observed directly on `j19-9`. |

### Node status at 02:35 UTC, September 27

| Node | GPUs | State | Needs |
|---|---|---|---|
| `ctheliosr-rck-g02-j19-9` (`10.190.178.96`) | 4 × `1002:75c1` | Running the corrected-driver validation (job 11120) on the one-boot corrected load from 01:27 | Remove the one-shot files after the test ([§7.4](#74-removal)) |
| `ctheliosr-rck-g02-j19-1` (`10.190.178.47`) | 4 × `1002:75c1` | Hung in shutdown since 20:30 on September 25; absent from `sinfo` | BMC power cycle. It should then boot the corrected driver once ([§7.3](#73-activation-on-j19-1)) |
| `ctheliosr-rck-g02-j19-11` (`10.190.178.179`) | 4 × `1002:75c1` | Hung in shutdown since 18:50 on September 25; absent from `sinfo` | BMC power cycle; `/nfs_spur` export |
| `ctheliosr-rck-g02-j19-7` (`10.190.178.15`) | 4 × `1002:75c1` | `drain` | TheRock CI ran on its GPUs outside the scheduler ([§8.4](#84-ci-workloads-outside-the-scheduler)) |
| `ctheliosp-rck-g02-h21-15` (`10.190.178.90`) | 4 × `1002:75c1` | `down`; sshd up, node agent dead since 11:22:34 on September 25 | spurd restart; `/nfs_spur` export |
| `ctheliosr-rck-g02-j19-16` | not inspected | `idle`, but a job did not start within 90 s | — |
| `ctheliosp-rck-g02-h21-7` | — | `down`; reserved for other users | — |
| `ctheliosp-rck-g02-h21-13`, `ctheliosp-rck-g02-h21-14` | `h21-14`: `1002:75c7` | Other users' jobs | — |
| `ctheliosp-rck-g02-h21-12`, `ctheliosp-rck-g02-h21-16` | `h21-16`: `1002:75c7` | `idle`; no `/shared`, no `unsquashfs` | Not A0-equivalent |

### Needed from admins

1. Power-cycle `j19-1` and `j19-11` through the BMC. Both hung after a systemd
   reboot request: the kernel answers ping, but sshd and spurd have stopped.
2. Add `10.190.178.179` (`j19-11`) and `10.190.178.90` (`h21-15`) to the
   `/nfs_spur` export on `10.210.2.150` ([§8.3](#83-nfs-export-gap)).
3. Restart spurd on `h21-15`.
4. Reconsider `halt_if_hws_hang=1` on shared nodes. It turns every MES
   non-response into a permanent wedge that also blocks reboots
   ([§8.2](#82-mes-non-response-and-halt_if_hws_hang)).
5. Pass the MES `0x7b` non-response on `j19-9` GPU `0004` to the firmware team
   ([evidence/j19-9/dmesg_wedge_excerpt.txt](evidence/j19-9/dmesg_wedge_excerpt.txt)).
6. On `j19-9`, the boot blacklists amdgpu and spurd refuses to start without
   `/dev/kfd`, so after any reboot the node stays out of the scheduler until
   amdgpu is loaded ([§8.6](#86-spurd-needs-devkfd)).
7. Check the flood of corrected PCIe errors on `j19-1` ([§8.5](#85-pcie-aer-flood-on-j19-1)).

### Next steps

1. Let the corrected-driver run continue toward AUC 0.75, checking that
   AutoNUMA activity and queue evictions stay present.
2. If it goes NaN, localize the first non-finite tensor on the corrected
   driver; that would mean the handler is not the whole story.
3. After the run, remove the one-shot files on `j19-9` ([§7.4](#74-removal));
   its `ARMED` flag is already consumed, so later boots behave as before.

## Contents

1. [Cluster, scheduler and access](#1-cluster-scheduler-and-access)
2. [Hardware and BKC](#2-hardware-and-bkc)
3. [Container image and software stack](#3-container-image-and-software-stack)
4. [Dataset and model configuration](#4-dataset-and-model-configuration)
5. [Experiment record](#5-experiment-record)
6. [NaN triage: component, trigger and mechanism](#6-nan-triage-component-trigger-and-mechanism)
7. [Corrected CWSR driver](#7-corrected-cwsr-driver)
8. [Cluster reliability](#8-cluster-reliability)
9. [Reproduction](#9-reproduction)
10. [Evidence and limits](#10-evidence-and-limits)

## 1. Cluster, scheduler and access

| Item | Recorded value |
|---|---|
| Scheduler | ROCm spur `0.11.0` (`11e7fa6b`) with a Slurm-compatible CLI (`sbatch`, `srun`, `squeue`, `sinfo`, `scancel`) |
| Controllers | `10.190.154.20`–`.22`, port 6817; the node agent `spurd` listens on port 6818 |
| Login node | `10.190.154.39` |
| Partitions | `default` (CPU nodes `rck-spur-dev01-cpu[01-09]`) and `ci-cd` (GPU nodes) |
| Shared storage | `10.210.2.150:/nfs_spur` on `/shared` (8.0 TB). Exported to the login and CPU nodes and to `j19-1`, not to `j19-11` or `h21-15` |
| Containers | squashfs images (`--container-image`), named containers unpacked with `unsquashfs` (`--container-name`), bind mounts (`--container-mounts`). Processes run as the submitting UID in a user namespace |
| Host access | SSH to compute nodes is gated by AMD Conductor. Host-context (non-container) jobs on the `j19` nodes have passwordless `sudo`. From a job, `sudo systemctl` fails with "Failed to connect to bus", but `sudo busctl` calls to systemd and logind on the system bus work |
| Kernel log | `dmesg` needs `sudo`; the journal keeps only the current boot |

Container jobs inherit the login shell's environment, for example
`CC=/usr/bin/gcc-14`. Every build and run therefore goes through
`cleanenv.sh`, which runs with only the image's recorded environment.
Nodes without `/shared` were staged from an HTTP server on the login node at
about 770 MB/s. `h21-15` has no `squashfs-tools`, so its image ran under an
unprivileged user-namespace chroot (`userns_runner.sh`) instead of a spur
container.

## 2. Hardware and BKC

### 2.1 `j19` A0 hosts

Recorded on `j19-1` at 02:49:16 on September 25 (boot `c4dfb7f9-9584-485c-a400-42a7cfcd9081`).
`j19-11` and `j19-7` match it in GPU ID, VBIOS, MES and driver srcversion. The
full inventory is [evidence/j19-1/host_inventory_20260925T024916Z.txt](evidence/j19-1/host_inventory_20260925T024916Z.txt).

| Item | Recorded value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | `6.14.0-37-generic` (`linux-image-6.14.0-37-generic 6.14.0-37.37~24.04.1`), built with gcc 13.3.0 |
| CPU | `AMD Eng Sample: 100-000001046-04`; one socket, 128 cores, two threads per core, 256 CPUs online (`j19-7`: 64 online) |
| NUMA | 38 nodes. All CPUs and system memory are on nodes 0 and 1 (257,657 MB and 257,806 MB); nodes 2–37 have no CPUs |
| RAM | 503 GiB |
| GPUs | 4 × `1002:75c1`, revision `0x00`, subsystem `0x75c1`, target `gfx1250`, market name `AMD Radeon Graphics`; PCI `0001:01:00.0` to `0004:01:00.0` |
| VBIOS / IFWI | `113-M4500001-0660` on all four GPUs; product name, number and serial are empty in sysfs |
| GPU firmware | MES `0x7b`, MES KIQ `0x7b`, MEC `0x96a`, RLC `0x1f`, SDMA and SDMA2 `0x00893547`, SMC `0x027d0a03`, TA RAS `0x1b440004`, VCN `0x0910d008` |
| HBM | Capacity was not read on this cluster. Peak sampled driver-used VRAM was 424.6 GiB on `0001:01:00.0`; the 1P4G hosts report 432 GiB per GPU |
| Other DKMS modules | `ionic` 26.08.12.001, `pds` 1.130.6.a.3, `tawk-ipc` 1.130.6.a.3, `ifoe-drv` 0.2.8.1000 |

### 2.2 Driver, firmware packages and boot configuration

| Item | Recorded value |
|---|---|
| Driver package | `amdgpu-dkms 1:7.1.1.31300009-2398876.24.04`; source `/usr/src/amdgpu-7.1.1-2398876.24.04` |
| Loaded module | Version `7.1.1.31300009`, srcversion `ECD716E95BE41CD5A13F14E`, `/lib/modules/6.14.0-37-generic/updates/dkms/amdgpu.ko.zst`. Signed with a per-host DKMS key; Secure Boot disabled and `sig_enforce=N` |
| gfx1250 CWSR handler | 5,656 bytes, SHA-256 `0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290`, **known faulty**. `cwsr_trap_handler_gfx12.asm` (SHA-256 `5e0e339c…`) has the unpatched `L_NOT_WAVE_START` sequence |
| Firmware package | `amdgpu-dkms-firmware 1:31.30.0.9.31300009-2398876.24.04`. `/lib/firmware/updates/amdgpu/gc_12_1_0_*` dated September 2: MES 636,620 B, MES1 617,536 B, uni-MES 333,504 B (SHA-256 `d93a2d40…`), MEC 305,120 B, RLC 253,344 B |
| udev rules | `amdgpu-insecure-instinct-udev-rules 31.30.0.9-2389086.24.04` |
| Module options | `/etc/modprobe.d/amdgpu-j19-6-shape.conf`: `mes_log_enable=1 halt_if_hws_hang=1 gpu_recovery=0`; kernel command line `amdgpu.noretry=1` |
| Other loaded parameters | `cwsr_enable=1`, `uni_mes=1`, `sched_policy=0`, `queue_preemption_timeout_ms=9000`, `hws_max_conc_proc=-1`, `ip_block_mask=0xffffffff`, `use_xgmi_p2p=1`, `pcie_p2p=Y` |
| Kernel command line | `panic=0 nowatchdog msr.allow_writes=on nokaslr amdgpu.noretry=1 pci=realloc=off modprobe.blacklist=device_dax,dax_hmem iommu=pt console=tty0 console=ttyS1,115200n8 preempt=full rcu_nocbs=all` |
| NUMA balancing | `kernel.numa_balancing=1` on `j19-1` and `j19-11`; **0 on `j19-7`** |
| Transparent huge pages | `madvise` |
| CPU microcode | `amd64-microcode 3.20251202.1ubuntu0.24.04.1` |
| Boot loader | UEFI GRUB, `GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.14.0-37-generic"`, `GRUB_TIMEOUT=0`; `/boot` on ext4 |

### 2.3 Host ROCm BKC trees

The container brings its own ROCm 7.14 user space. No run in this document
uses the host trees below.

| Path | Recorded version |
|---|---|
| `/opt/rocm` | 10.1.0 (link target not recorded) |
| `/opt/rocm-10.1.0a20260811`, `/opt/rocm-10.1.0a20260811+bkc.20260818`, `/opt/rocm-10.1.0a20260811-bkc.20260820`, `/opt/rocm-10.1.0a20260812` | 10.1.0 |
| `/opt/rocm-A0-0715` | 7.15.0 |
| `/opt/rocm-pr891-34205166812-166c2660147f` and matching `/opt/rocm-tests-*` trees | 10.1.0 |
| Other `/opt` tooling | `agt-4.1.158.0E`, `amdtools`, `arc-1.6.0-rc2`, `crd-v18-artifacts` |

`amd-smi` isn't on the host PATH. The host's `libhsa-runtime64-1 5.7.1` is
Ubuntu's package and is unused.

### 2.4 `h21-15`: a different BKC

| Item | Recorded value |
|---|---|
| OS and kernel | CentOS Stream 9 (glibc 2.34), kernel 6.16.1, similar to the 1P4G a37 hosts |
| Driver | `amdgpu-7.1.1-2389086.el9`, srcversion `390DDE7DEA1EBD99204E6D5`, **same faulty 5,656-byte handler** |
| GPUs and firmware | 4 × `1002:75c1`; IFWI `650B`, MES `0x7a` |
| Host | 256 CPUs, 515 GB RAM; `kernel.numa_balancing=0`; no `/shared` and no `squashfs-tools` |

PCI ID `0x75c1` and revision `0x00` don't establish an A0 or B0 stepping (see
the 1P4G record). IFWI 650B and MES `0x7a` resemble the B0 host rather than
the `j19` nodes.

### 2.5 Differences from the A0 investigation host

| Item | `j19` nodes | A0 investigation host |
|---|---|---|
| VBIOS | `113-M4500001-0660` | `113-M4500001-630A` |
| MES / MEC / SMC | `0x7b` / `0x96a` / `0x027d0a03` | `0x7b` / `0x956` / `0x027d0701` |
| uni-MES firmware file | SHA-256 `d93a2d40…` | SHA-256 `d93a2d40…` (identical) |
| amdgpu build | 7.1.1 `2398876`, srcversion `ECD716E9…` | 7.1.1 `2397345` builds, srcversions `654C1DDE…` and `EADFD8EC…` |
| gfx1250 CWSR handler | `0f718b5e…` (faulty) | `0f718b5e…` (faulty) until the corrected candidate was loaded |
| Kernel | `6.14.0-37-generic` | `6.14.0-37-generic` |

The 1P4G a37-2 host, which trained 14,000 finite steps to AUC 0.75, ran
`halt_if_hws_hang=0`; the `j19` nodes run `halt_if_hws_hang=1`.

## 3. Container image and software stack

### 3.1 Image

| Item | Recorded value |
|---|---|
| Image | `docker.io/amdprimus/amdprimus:gfx1250-20260910` (private; needs the `amdprimus` Docker Hub credentials, which aren't recorded here) |
| Image ID (config digest) | `sha256:56ae1515d36be72de60cf1f58da2811b3d06f7eda5d175538fecc67f270cfbc2`. The registry manifest digest was not recorded |
| Created | 2026-09-09T17:02:19Z |
| Labels | `image.title=PyTorch + ROCm runtime image (NPI)`, `npi.rocm.version=7.14.0a20260625`, `npi.pytorch.version=2.11.0+rocm7.14.0a20260625`, `npi.pytorch.git_ref=f55dda6ca78b73027840c1bc4014fe703d4f5473`, `primus.base.image=amdprimus/amdprimus:0815`, `primus.rccl.ref=rccl/gfx1250_merge`, `primus.turbo.ref=2f00979` |
| spur squashfs | `/shared/chcai/spur-images/docker.io+amdprimus+amdprimus+gfx1250-20260910.sqsh`, 9,129,103,360 bytes |
| Environment | `ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel`, `PYTORCH_ROCM_ARCH=gfx1250`, `VIRTUAL_ENV=/opt/venv`; `LD_LIBRARY_PATH` starts with `/opt/rccl-gfx1250/lib` |

### 3.2 Stack built in the container

`build_rec_stack.sh` reproduces the `Dockerfile.amdprimus0815` recipe on top of
the image. It took about 16 minutes on `j19-1`. Containers aren't root, so
build dependencies come from `apt-get download` plus `dpkg-deb -x`.

| Component | Exact version or source |
|---|---|
| Python | 3.12, `/opt/venv` |
| PyTorch | `2.11.0+rocm7.14.0a20260625`, HIP `7.14.60850` |
| ROCm SDK wheels | `rocm`, `rocm-sdk-core`, `rocm-sdk-devel`, `rocm-sdk-libraries-gfx1250`, all `7.14.0a20260625` |
| RCCL in use | `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/librccl.so.1` (preflight log) |
| hipBLASLt | Needs `HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`; without it hipBLASLt returns `HIPBLASLT_STATUS_INVALID_VALUE` |
| Triton | 3.8.0, source build of `7ff97e310935b4a79794878dbc911f9af25d38d9` in `/opt/triton-custom` with LLVM `b010a18d` (`llvm-b010a18d-ubuntu-x64-1`). One inert hook reads `TRITON_AMD_TARGET_FEATURES` |
| FBGEMM | `fbgemm_gpu_nightly-rocm 2026.9.25`, built from `10b775730212923f65f7b78f79b6a01d80cf3c29` with gfx1250 patches to `fbgemm_gpu/cmake/Fbgemm.cmake`, `codegen/genscript/jinja_environment.py` and `include/fbgemm_gpu/utils/cuda_prelude.cuh`; wheel SHA-256 `d1cbf8a3…` |
| TorchRec | `1.7.0a0+bf55480`, tag `v2026.06.01.00` (`bf554808b185cc7c694525e34ae2b910311842cf`) |
| MLPerf logging | `3d17f6fb6047955159b69d9c2415317f163fbe43` |
| Other packages | numpy 2.4.1, pandas 3.0.5, pyarrow 25.0.1, polars-u64-idx 1.33.1, gin-config 0.5.0, torchmetrics 1.0.3, tensorboard 2.21.0 |
| Training source | `chriscai-amd/training`, branch `chcai/mi450`, commit `402f7d0a83252bb4d8c32c01b599b1010bb9f109`, clean tree in every run |

## 4. Dataset and model configuration

The dataset is MLCommons' prepared DLRMv4 Yambda-5b
(`processed_5b/hstu_cache_L4086`), downloaded from
`https://training.mlcommons-storage.org/dlrmv4_dataset` with its MD5 and URI
metadata. All 227,409,136,794 bytes were verified against the published MD5
and SHA-256 lists; one corrupted transfer (`anchor_ts_L4086.npy`) was
downloaded again. The MLPerf log reports 2,290,835,423 training samples.

`run_full_model.sh` applies the a37-2 converged configuration from the
[1P4G record](../mi450_1P4G/ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md):

| Group | Settings |
|---|---|
| Topology | One node, four GPUs (`NNODES=1 GPUS_PER_NODE=4`); NCCL and Gloo on `lo` |
| Data | `START_TS=0 NUM_TRAIN_TS=299`, `HISTORY_LENGTH=4086 MIN_HISTORY=4086 MAX_SEQ_LEN=4096`, `HISTORY_STRATEGY=interleaved`, `NUM_WORKERS=4 PREFETCH_FACTOR=8` |
| Batch and seed | `BATCH_SIZE=1024` per rank (global 4096), `SEED=1` |
| Optimizer | `DENSE_LR=SPARSE_LR=5e-7`, `LR_WARMUP_STEPS=24000`, `GRAD_CLIP_NORM=1.0` |
| Embeddings | `EMBEDDING_ROW_SCALE=1.0`, `EMB_PLACEMENT=hbm`, `HBM_CAP_GB=260`, `SPARSE_A2A_FWD=SPARSE_A2A_BWD=fp16`, `QCOMM_LOWMEM_CODEC=1`, `EC_INDEX_DEDUP=1` |
| Kernels | `HSTU_HAMMER_KERNEL=TRITON`, `HSTU_NUM_LAYERS=3`, `HSTU_BWD_MAX_VGPR=0` (uncapped), `WEIGHTED_LN_BWD_BLOCK_N=0`, `TRITON_FULL_AUTOTUNE=0`, `TRITON_ALLOW_PIPELINING=0`, `AMDGCN_USE_BUFFER_OPS=0` |
| Evaluation and stop | `EVAL_EVERY_DATA_PCT=0.001`, `SKIP_EVAL_EPOCH_PCT=0.022361844716419856`, `EVAL_HOLDOUT_TS=299`, `AUC_THRESHOLD=0.75` |
| Disabled | Checkpoints, NaN probes and hooks, output tracing |

Optional knobs: `NUMA_BIND` (wraps the trainer in `numactl --membind`),
`TRIPWIRE_DIR` (NaN tripwire with backward-stage checks, no payload retained),
`PROTECT_KERNELS`, `TRITON_AMD_TARGET_FEATURES`, `TRIGGER_CLEAR_REFS_S` and
`DIE_AT_STEP`. `monitor_run.py` records every step's loss, stops the run after
non-finite loss on two distinct steps, flags a stall after 1,200 s without a
step, and samples GPU busy/VRAM and NUMA `vmstat` counters every 30 s.

## 5. Experiment record

Startup took about 11 minutes; training then ran at about 1.4 s per step.

### 5.1 Stock-driver runs

| Run | Node, boot | Configuration | Result |
|---|---|---|---|
| `full4_a37cfg_20260925T035054Z` | `j19-1`, `c4dfb7f9` | Default (AutoNUMA on) | Steps 1–110 finite; `nan` from **step 111** at 04:04:44 on September 25; stopped by the monitor |
| `full4_a37cfg_numabind_20260925T040549Z` | `j19-1`, `c4dfb7f9` | `numactl --membind=0,1` | Finite through **step 1,912** at 05:02:55; `j19-1` lost at about 05:03 |
| Arm A, 1,500-step bound | `h21-15` | Default (AutoNUMA off on this host) | Node agent lost at 11:22:34, about 19 minutes after launch; the run directory is on the node's local disk and wasn't recovered |
| `reproA_default_tripwire_…_20260925T150501Z` | `j19-11`, `b4ecf5cc` | Default plus tripwire, 600-step bound | Finite through step 122; **step 123** non-finite gradient localized ([§6.1](#61-localization)) |
| `reproB_numabind_…_20260925T152003Z` | `j19-11`, `b4ecf5cc` | `numactl --membind=0,1`, 1,500-step bound | **Stalled after step 6** (15:31:10). One GPU stayed 100 % busy with VRAM held; the ranks couldn't be killed |
| reproC | `j19-11` | Triton without the extended VGPR banks (`TRITON_AMD_TARGET_FEATURES=-1024-addressable-vgprs`) | Not run; the GPU was wedged |
| `long_numabind_ctheliosr-rck-g02-j19-9_20260926T202336Z` | `j19-9`, `964bbe1c` | `numactl --membind=0,1`, no step bound | Steps 1–353 finite; `nan` from **step 354** at 20:43:11 on September 26; stopped by the monitor; node healthy afterwards |
| `tripwire_numabind_ctheliosr-rck-g02-j19-9_20260926T204726Z` | `j19-9`, `964bbe1c` | `numactl --membind=0,1` plus tripwire, 2,000-step bound | **Wedged in the first iteration** at 20:57:48: MES on GPU `0004` stopped responding ([§8.2](#82-mes-non-response-and-halt_if_hws_hang)) |

| Stock-driver telemetry | `j19-1` default (NaN at 111) | `j19-1` bind (finite to 1,912) | `j19-9` bind (NaN at 354) |
|---|---:|---:|---:|
| `numa_pte_updates` per minute | 2,543,480 | 0 | 0 in the 3.5 minutes before the NaN |
| `numa_hint_faults` per minute | 1,059,941 | 15 | 0–12 |
| `numa_pages_migrated` per minute | 158,360 | 2 | 0–6 |
| KFD evicted time per rank at the end | not recorded | not recorded | 6.9–10.8 s |
| Peak sampled VRAM, GPUs `0001`/`0002`/`0003`/`0004` (GiB) | 424.6 / 406.4 / 391.1 / 351.7 | 399.5 / 406.4 / 391.1 / 398.7 | — |

The KFD eviction counter (`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`)
was added to the monitor after the `j19-1` runs. In the `j19-9` bound run it
rose steadily through training on all four ranks, with AutoNUMA idle.

### 5.2 Corrected-driver run on `j19-9`

`corrected_default_ctheliosr-rck-g02-j19-9_20260927T013250Z`: the default
a37-2 configuration (no binding, no tripwire, no step bound) on the corrected
driver, srcversion `D0FD14A94CA9A71E8CE9BB4`, loaded as the first amdgpu load
after a hardware reset ([§7.5](#75-activation-on-j19-9)). `kernel.numa_balancing=1`.

| Milestone | Time (September 27) | Status |
|---|---|---|
| Launch | 01:32:50 | Health check passed on the default transport |
| Step 165 | 01:47:45 | Finite; past the stock failures at 111 and 123 |
| Step 507 | 01:55:45 | Finite; past the stock bound failure at 354 |
| Step 1,007 | 02:07:45 | Finite |
| Step 2,026 | 02:32:46 | Finite; past the 1,912-step bound run on `j19-1` |
| Step 2,128 | 02:35:19 | Finite, no stall, no evaluation yet; run continuing |

| Trigger during the first 50 training minutes (01:44–02:34) | Value |
|---|---:|
| `numa_pte_updates` per minute | 2,232,011 |
| `numa_hint_faults` per minute | 1,368,619 |
| `numa_pages_migrated` per minute | 196,153 |
| KFD evicted time added, all ranks | 18.5 s (9.3–14.7 s per rank since launch) |

## 6. NaN triage: component, trigger and mechanism

### 6.1 Localization

The tripwire in `reproA` fired at step 123 in the backward pass, at the
dense-gradient check. All inputs were finite; the output side failed. Exactly
one of 69 dense parameters had a non-finite gradient, on all four ranks:
`_hstu_transducer._input_preprocessor._additional_embedding_mlp.2.weight`,
shape `[512, 256]`. That is the weight gradient (DW) of a Linear layer in the
HSTU input preprocessor. A0's attempt-405 raw-DW capture has its 640 NaNs in
the same parameter's gradient
([A0 audit](../mi450_a0/evidence/current_20260919/training405_full_payload.json)).
[Tripwire summary](evidence/j19-11/reproA/tripwire_summary.json).

### 6.2 Component

The evidence points to the gfx1250 CWSR (compute wave save/restore) trap
handler embedded in amdgpu's KFD, the component the A0 record corrected:

- **The same faulty bytes are installed.** Every node checked embeds the
  5,656-byte `0f718b5e…` handler, and the DKMS source is unpatched.
- **The same tensor fails.** The first non-finite gradient matches A0's
  attempt 405 ([§6.1](#61-localization)).
- **The event that runs the handler is frequent.** KFD queue evictions save
  the running waves through the handler, and they occur throughout training,
  from AutoNUMA and from other sources ([§6.3](#63-trigger)).
- **Changing only the handler has removed the NaN so far.** The
  corrected-driver default run is finite past 2,128 steps with the trigger
  active, where the stock driver failed by step 123 in the same configuration
  ([§5.2](#52-corrected-driver-run-on-j19-9)).
- **A0's controlled tests establish the mechanism.** On the A0 host the
  corrected driver restores the complete attempt-405 DW baseline in all four
  unchanged ELF arms
  ([A0 fix record](../mi450_a0/cwsr_bank_corruption_fix.md)).

The defect, as the A0 record describes it: at `L_NOT_WAVE_START` the handler
saves `exec_lo`, forces lane 0 active, reads `v1` into `ttmp15`, writes zero to
`v1` for a prefetch workaround, and writes `ttmp15` back. These instructions
run before the handler normalizes the MODE register's VGPR bank selectors. When
the interrupted wave's SRC0 and DST banks differ, lane 0 is copied between two
different physical registers, for example `v513[0] → v257[0]`. Kernels using
the 1,024-addressable-VGPR banks, such as the DW GEMM here, can then resume
with a corrupted accumulator, matrix operand or load address.

### 6.3 Trigger

Every eviction stack captured on the A0 host followed this path:

```text
Linux automatic NUMA balancing: task_numa_work
  → change_prot_numa / change_protection
  → MMU-notifier invalidation
  → svm_range_cpu_invalidate_pagetables
  → kgd2kfd_quiesce_mm
  → KFD queue eviction → CWSR save of resident waves → trap handler entry
```

On `j19-1` the default run made 2.54 million NUMA page-table updates per
minute ([§5.1](#51-stock-driver-runs)), so the scanner was invalidating the
trainer's address space continuously during training.

AutoNUMA is not the only source of evictions. In the `j19-9` bound run, with
zero AutoNUMA page-table updates, each rank still accumulated 6.9–10.8 seconds
of evicted time. The other sources were not traced; candidates include SVM
restore activity, buffer moves under memory pressure and `fork`. The kernel log
shows `svm_range_restore_work` busy just before the wedge
([§8.2](#82-mes-non-response-and-halt_if_hws_hang)).

### 6.4 What `numactl --membind=0,1` does and doesn't do

`numactl --membind` installs an `MPOL_BIND` task policy without
`MPOL_F_NUMA_BALANCING`. The kernel's NUMA-balancing scanner skips VMAs whose
policy doesn't allow migrate-on-fault, so it never write-protects the
trainer's pages, and no MMU-notifier invalidation reaches KFD from this path.
The measured page-table update rate fell from 2.54 million to 0 per minute.
Nodes 0 and 1 hold all of the host's system memory, so the binding doesn't
restrict placement. A0's policy counterfactual agrees: plain binding gave 0
SVM eviction pairs and 0 register copies in its probes, while `--balancing`
with the same node mask restored them.

In full training that is not enough. The `j19-9` bound run went NaN at step
354 while queue evictions continued from other sources, and the earlier
1,912-step bound run on `j19-1` was one sample. Binding lowers exposure but
does not make training safe; only the handler fix removes the defect.

### 6.5 What is not established

- The corrected-driver evidence is one run, still in progress.
- The first non-finite tensor under binding was not localized: the bound
  tripwire run wedged in its first iteration.
- The non-AutoNUMA eviction sources were not traced.
- The victim-side control did not run. It would force the failing DW GEMM onto
  hipBLASLt solution 102 (252 VGPRs, no extended banks), as A0 did; the Triton
  flag in reproC would not have covered this hipBLASLt kernel.
- `h21-15` (AutoNUMA off, different BKC) produced no training steps.

## 7. Corrected CWSR driver

### 7.1 Handler

`make_candidate_handler.py` rebuilds the audited image from the installed
module. It extracts `cwsr_trap_gfx12_1_0_hex`, checks the faulty SHA-256, and
checks that the bytes equal the DKMS header array. It then replaces bytes
60–108 with the patch's 64-byte sequence and moves the byte-4 branch to
`L_RESTORE` from offset 1009 to 1013 (target 4044 → 4060). It writes nothing
unless the result hashes to the A0 team's audited candidate. It did: 5,672
bytes, SHA-256 `68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80`.

Installed bytes 60–108, disassembled with ROCm's `llvm-mc`:

```text
s_mov_b32 ttmp14, exec_lo
s_mov_b32 exec_lo, 1
v_readlane_b32 ttmp15, v1, 0
s_mov_b64 ttmp[2:3], 0
v_mov_b32_e32 v1, 0
global_prefetch_b8 v1, ttmp[2:3] scope:SCOPE_SE
v_writelane_b32 v1, ttmp15, 0
s_mov_b32 exec_lo, ttmp14
```

Corrected bytes 60–124:

```text
s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_MODE, 12, 6)
s_mov_b32 ttmp3, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 6), ttmp3
s_mov_b32 ttmp14, exec_lo
s_mov_b32 exec_lo, 1
v_readlane_b32 ttmp15, v1, 0
v_sub_nc_u32_e64 v1, 63, ttmp2
global_prefetch_b8 v1, ttmp[2:3] offset:-63 scope:SCOPE_SE
v_writelane_b32 v1, ttmp15, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 6), ttmp2
s_mov_b32 exec_lo, ttmp14
```

The added instructions save the six DST/SRC0/SRC1 bank bits of MODE in
`ttmp2`, clear them around the lane-0 copy, and restore them afterwards. The
prefetch address stays zero (`k + (63 − k) − 63`).

### 7.2 Module build and audit

`build_corrected_module.sh` and `audit_modules.sh` built and checked the
module on `j19-1`. The full audit log is
[evidence/j19-1/corrected_module_audit.txt](evidence/j19-1/corrected_module_audit.txt).

| Check | Result |
|---|---|
| Build | Copy of the DKMS tree, `make KERNELVER=6.14.0-37-generic` under a clean environment (the kernel's gcc 13.3.0), then `strip -g` as DKMS does; 23 s on 256 cores |
| Installed module vs unchanged rebuild | Same srcversion `ECD716E95BE41CD5A13F14E`; 0 of 34,004 functions changed; only the 20 build-ID bytes differ in the whole file, excluding the signature |
| Unchanged rebuild vs candidate | Handler 5,656 B `0f718b5e…` → 5,672 B `68c31ab2…`. `.text` differs by 2 bytes, both in `kgd2kfd_device_init` at offsets 3442 and 3448 (the handler-size immediates). `.rodata` grows from 1,947,600 to 1,947,632 bytes (528,176 shifted bytes). `.modinfo` differs by 21 bytes (srcversion); build ID by 20 bytes |
| Candidate identity | srcversion `D0FD14A94CA9A71E8CE9BB4`; stripped module 40,295,376 bytes, SHA-256 `d224829e40dd759028f6ec36573ffedbdb79ee3aae215e014106517b8f23ea18` |
| Agreement with the A0 audit | Same two `.text` offsets and same 528,176-byte `.rodata` difference as [A0's module audit](../mi450_a0/evidence/current_20260919/CWSR_module_byte_audit.json) |

The same build on `j19-9` produced byte-identical candidate and baseline
modules (SHA-256 `d224829e…` and `75fc93ee…`) and the same audit result
([evidence/j19-9/corrected_module_audit.txt](evidence/j19-9/corrected_module_audit.txt)).

### 7.3 Activation on `j19-1`

**Why not a live reload.** On the A0 host, both live `rmmod`/`insmod`
attempts ended in host restarts. The second loaded a rebuilt baseline
identical to the installed module except for its build ID, and left fatal
processor-context-corrupt BERT records. The path the A0 team validated is a
first load on a boot where amdgpu is absent
([reload evidence](../mi450_a0/evidence/current_20260919/CWSR_reload_recovery_20260922_resume.json)).

**One-shot boot.** [tools/cwsr_fix/oneshot/](tools/cwsr_fix/oneshot/) installs
three pieces:

- `/boot/grub/custom.cfg` adds a menu entry `chcai-cwsrfix-oneshot` with the
  default entry's kernel, initrd and command line, plus
  `modprobe.blacklist=device_dax,dax_hmem,amdgpu` and `chcai_cwsrfix=1`.
  `grub-reboot chcai-cwsrfix-oneshot` sets `next_entry`, which GRUB clears as
  it boots the entry once.
- `chcai-cwsrfix-load.service` runs only when the command line has
  `chcai_cwsrfix=1`, and before spurd. It clears `next_entry` again, checks
  the candidate's SHA-256, loads the stock module's 17 dependencies with
  `modprobe`, and runs `insmod /var/lib/chcai-cwsrfix/candidate_amdgpu.ko
  mes_log_enable=1 halt_if_hws_hang=1 gpu_recovery=0 noretry=1`, the same
  parameters `modprobe` passes the stock module.
- `/var/lib/chcai-cwsrfix/` holds the candidate, the loader, a README and
  per-boot receipts (`receipts/<boot_id>/load.log` and dmesg).

Any other boot uses the unchanged default entry and the stock driver.

**State.** The entry was armed at 20:29 and verified: `next_entry` set and the
installed candidate's hash matches ([install log](evidence/j19-1/oneshot_install.txt)).
The reboot was requested at 20:30:01 from boot
`b6eaa6f0-1689-4705-9dfe-fb1abe7c0139` with no KFD processes, and the node
hung in shutdown. Ping never dropped and sshd never returned, so GRUB most
likely never ran and `next_entry` should still be set. In that case the next
power-on boots the corrected driver as a first load after a full power cycle,
the condition the A0 team validated.

After the node returns, check:

1. `/proc/cmdline` contains `chcai_cwsrfix=1`.
2. `/sys/module/amdgpu/srcversion` is `D0FD14A94CA9A71E8CE9BB4`.
3. `/var/lib/chcai-cwsrfix/receipts/<boot_id>/load.log` shows `insmod rc=0`.

### 7.4 Removal

On `j19-1` (GRUB one-shot):

```bash
sudo grub-editenv /boot/grub/grubenv unset next_entry
sudo rm -f /boot/grub/custom.cfg /etc/systemd/system/chcai-cwsrfix-load.service \
     /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
sudo rm -rf /var/lib/chcai-cwsrfix
```

On `j19-9` (armed loader; its `ARMED` flag is already consumed):

```bash
sudo rm -f /etc/systemd/system/chcai-cwsrfix-load.service \
     /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
sudo rm -rf /var/lib/chcai-cwsrfix
```

After removal, later boots behave exactly as before the test.

### 7.5 Activation on `j19-9`

`j19-9`'s boot keeps amdgpu out: its kernel command line has
`modprobe.blacklist=amdgpu` and `/etc/modprobe.d` has `blacklist amdgpu`. On
its previous boot, amdgpu was loaded about six hours after boot, by hand or by
CI. So no GRUB entry is needed; every boot there is already a boot without
amdgpu. [tools/cwsr_fix/oneshot_armed/](tools/cwsr_fix/oneshot_armed/)
installs the same loader and unit as on `j19-1`, gated on
`/var/lib/chcai-cwsrfix/ARMED` instead of a command-line token. The loader
deletes `ARMED` first, so it runs on one boot only.

The files were armed at 01:11 on September 27, while the node was wedged
([install log](evidence/j19-9/armed_install.txt)). After the reset described
in [§8.2](#82-mes-non-response-and-halt_if_hws_hang), the new boot
(`7f22d2fb-18e1-4895-9b5a-a2c054f872b6`) ran the loader. `insmod` started at
01:26:50 and returned 0 at 01:27:16 with the stock parameters, and all four
GPUs bound to the corrected module
([load log](evidence/j19-9/corrected_boot/load_log.txt),
[post-boot check](evidence/j19-9/corrected_boot/post_boot_inspect.txt)). The
`VM_PAGE_FAULT` lines in that boot's log are entries of the
`MEM_RESERVED_INFO` table that every amdgpu load prints, with the same
addresses on the stock load of September 24, not faults.

## 8. Cluster reliability

### 8.1 Node losses

| Time | Node | Activity | Symptom | Recovery |
|---|---|---|---|---|
| About 05:03, September 25 | `j19-1` | `numactl`-bound training, step 1,912 | Node agent lost | Rebooted at about 14:16 (boot `b6eaa6f0`) |
| About 08:29, September 25 | `j19-7` | CPU-only dataset download and verification | Node agent lost | Back by about 20:20 (boot `c7c172b6`); drained by 00:05 on September 26 |
| 11:22:34, September 25 | `h21-15` | Arm A, about 19 minutes after launch | Node agent lost; sshd still up | Still down |
| 18:50, September 25 | `j19-11` | systemd reboot after the reproB GPU wedge | Hung in shutdown | Still hung |
| 20:30, September 25 | `j19-1` | systemd reboot into the one-shot entry, GPUs idle | Hung in shutdown | Still hung |
| 20:57, September 26 | `j19-9` | Bound tripwire run, first iteration | MES non-response; driver frozen by `halt_if_hws_hang` | Frozen until 01:12 on September 27 |
| About 01:14, September 27 | `j19-9` | Two minutes after clearing `halt_if_hws_hang` | Hardware error; node dark, no ping | Reset itself; back at 01:27 on the corrected driver |

I requested the two `j19` reboots, and the `j19-9` reset followed my clearing
of `halt_if_hws_hang`.

### 8.2 MES non-response and `halt_if_hws_hang`

Every `j19` node loads amdgpu with `halt_if_hws_hang=1 gpu_recovery=0`, a
firmware-debug setup; `j19-9`'s boot log records "Setting dangerous option
halt_if_hws_hang - tainting kernel". In this driver:

- `mes_v12_1.c:251` (gfx12.1 MES) runs `while (halt_if_hws_hang) schedule();`
  after `MES(%d, %d) failed to respond to msg=%d`.
- `kfd_device_queue_manager.c:2286` does the same after a queue-preemption
  fence timeout. The source comment says this halts the driver thread "in
  order not to mess up CP states before doing scandumps for FW debugging".

**Observed on `j19-9`** ([kernel log excerpt](evidence/j19-9/dmesg_wedge_excerpt.txt)):

| Time (September 26) | Kernel log |
|---|---|
| 20:57:30 | `workqueue: svm_range_restore_work [amdgpu] hogged CPU for >10000us 35 times` |
| 20:57:47 | `amdgpu 0004:01:00.0: MES(0, 0) failed to respond to msg=INVALIDATE_TLBS`, four times |
| 21:00:55 | `amdgpu 0004:01:00.0: MES(0, 0) failed to respond to msg=REMOVE_QUEUE` |
| 21:04 onward | Hung-task reports: `svm_range_restore_work` → `kgd2kfd_resume_mm` → `restore_process_queues_cpsch` waiting for a lock; `kfd_sdma_activity_worker` and a `kfd_procfs_show` reader blocked behind it |

The trainer's log stopped at 20:57:48. Its KFD eviction counters jumped by
about 14 seconds and then stopped changing after 21:01, which fits queues that
were evicted and never restored. GPU `0004` read 100 % busy for four hours,
and the cancelled job left five KFD processes that would not exit
([telemetry](evidence/j19-9/tripwire_numabind_wedge/telemetry.jsonl)).

**Clearing the parameter at runtime** (`echo 0 >
/sys/module/amdgpu/parameters/halt_if_hws_hang`, mode 0644) released the
halted threads: within 20 seconds four of the five processes exited and three
GPUs' VRAM was freed. About two minutes later the node went dark. The next
boot's BERT reports one hardware error record from the previous boot, and
the node was back about 13 minutes after it went dark
([observation](evidence/j19-9/halt_clear_observation.txt)). Clearing the
parameter therefore unfreezes a wedged node but does not make it safe to keep
running; a reset is still needed.

**Hung reboots.** The two hung `j19` reboots show the same signature: the
kernel answers ping and resets TCP connections, but sshd, rpcbind and spurd
have stopped and the machine never reached firmware. `j19-11` was wedged the
same way as `j19-9` when it was rebooted, so a teardown message that MES never
answers, followed by the halt loop, is the likely cause there. `j19-1` had idle
GPUs, and its retained kernel log shows no MES failures; a serial-console
capture would settle it. The 1P4G a37-2 host, which trained 14,000 steps
without a wedge, ran `halt_if_hws_hang=0`.

### 8.3 NFS export gap

`showmount -e 10.210.2.150` lists the `/nfs_spur` clients: the login and CPU
nodes (`10.190.154.20`–`.39`), `j19-1` (`10.190.178.47`) and a few other
`10.190.178.x` hosts. It doesn't list `j19-11` (`10.190.178.179`) or `h21-15`
(`10.190.178.90`). `j19-11`'s journal shows the same failure on every boot
since at least September 11:

```text
mount.nfs4: mounting 10.210.2.150:/nfs_spur failed, reason given by server: No such file or directory
```

Restarting `shared.mount` doesn't help; the export needs the two addresses.

### 8.4 CI workloads outside the scheduler

At about 21:00 on September 25, `j19-7` was `idle` in `sinfo` but had five KFD
processes owned by root from TheRock CI: a defunct `rocsolver-test`, a Tensile
`pytest`, `test_device_reduce` and two Python processes. A spur reservation
therefore doesn't guarantee exclusive GPUs. Check `/sys/class/kfd/kfd/proc`
before and during runs.

### 8.5 PCIe AER flood on `j19-1`

The kernel log saved before the 20:30 reboot covers uptime 10,414–22,425 s and
contains 2,043 corrected `RxErr` events (Physical Layer, Receiver ID) from
`pcieport 0000:01:00.0`, about one every 6 s. 1,055 hardware-error records
name vendor `0x1dd8` (Pensando) device `0x0008`, and 3 name `0x1022:0x1108`.
This isn't a GPU link (the GPUs are in PCI domains 0001–0004), and nothing
connects it to the NaN, but it rotates the amdgpu boot messages out of the
kernel ring buffer.

### 8.6 spurd needs `/dev/kfd`

`spurd.service` on the `j19` nodes has `ExecStartPre=/usr/bin/test -c /dev/kfd`
and `ExecStartPre=/usr/bin/mountpoint -q /shared`. On a node whose boot keeps
amdgpu out, as on `j19-9`, the scheduler agent therefore refuses to start
until someone loads amdgpu, and the node stays out of the scheduler after any
reboot. The armed loader on `j19-9` loaded the driver before spurd, which is
why that node rejoined on its own. `j19-9` also runs a GitHub Actions CI runner
service, so CI can use its GPUs outside the scheduler, as on `j19-7`.

### 8.7 Staging defects found on `j19-9`

- `cp -a` from `/shared` (NFS) to local disk fails with "preserving
  permissions … Operation not supported": GNU `cp` preserves ACLs with the
  mode, and local ext4 rejects the NFSv4 ACLs. It aborted the stack build and
  made `stage_data_local.sh` exit before verifying. Both scripts now copy with
  `cp -dR --preserve=timestamps`.
- The dataset copy on `/shared` still had the corrupt
  `processed_5b/hstu_cache_L4086/anchor_ts_L4086.npy` from September 25 (MD5
  `24afb7b4…` instead of `4a1f9a06…`); earlier nodes had repaired their local
  copies only. It was downloaded again from MLCommons, the local dataset
  verified (227,409,136,794 bytes), and the `/shared` copy was replaced.

## 9. Reproduction

The scripts live in `/shared/chcai/mi450_a0_newcluster`; copies of the
driver tools are in [tools/cwsr_fix/](tools/cwsr_fix/).

**Training container.** A container job holds the node, builds the stack once
into a named container, stages the dataset to local NVMe if needed, and then
runs request scripts from a requests directory in name order:

```bash
sbatch -p ci-cd -w <node> --exclusive --gpus-per-node 4 -c 256 -t 3-00:00:00 \
  --container-image /shared/chcai/spur-images/docker.io+amdprimus+amdprimus+gfx1250-20260910.sqsh \
  --container-name <name> \
  --container-mounts /shared/chcai:/shared/chcai --container-mounts /var/tmp/chcai:/scratch \
  --container-mounts /var/tmp/chcai/dlrm_data:/data/mlperf_dlrm_v4 \
  --container-mounts /shared/chcai/training:/workspace \
  --container-workdir /workspace/recommendation \
  /shared/chcai/mi450_a0_newcluster/container_job.sh
```

`container_job.sh` reads its request directory from `REQ`
(default `/shared/chcai/mi450_a0_newcluster/requests`), so each node gets its
own queue, for example `REQ=/shared/chcai/mi450_a0_newcluster/requests_j19-9
sbatch …`. The per-host preflight writes `preflight_status_<host>` and
`transport_<host>.env`, so concurrent nodes don't overwrite each other.

Request scripts first run `preflight_4gpu.py` (matmul, gather and RCCL
collectives on four GPUs), then launch a run:

```bash
# default reproduction
cleanenv.sh env RUN_NAME=<name> bash run_full_model.sh
# trigger removed
cleanenv.sh env RUN_NAME=<name> NUMA_BIND=0,1 bash run_full_model.sh
# first non-finite tensor
cleanenv.sh env RUN_NAME=<name> DIE_AT_STEP=600 TRIPWIRE_DIR=<dir> bash run_full_model.sh
```

**Corrected driver.** On a `j19` host, in a host-context job:

```bash
W=/var/tmp/chcai/cwsrmod bash tools/cwsr_fix/build_corrected_module.sh   # handler + module build
W=/var/tmp/chcai/cwsrmod bash tools/cwsr_fix/audit_modules.sh            # baseline rebuild + audit
bash tools/cwsr_fix/oneshot/install_oneshot.sh                           # j19-1 style: GRUB one-shot
bash tools/cwsr_fix/oneshot_armed/install_armed.sh                       # j19-9 style: boot keeps amdgpu out
```

The build and audit need no root access. The install scripts use `sudo` and
do not reboot. Soft reboots hung on these nodes while the driver was frozen
([§8.2](#82-mes-non-response-and-halt_if_hws_hang)), so reboot through the BMC
where possible.

**Watchers.** `watch/j19_1_watch.sh` and `watch/j19_9_watch.sh` wait for a node
to rejoin the scheduler, claim it, check which amdgpu loaded, and launch the
corrected-driver validation only if the corrected module is loaded with no
other GPU users and AutoNUMA on. spur's `srun` output uses CRLF line endings,
so parsers of it must strip `\r`; the first `j19-9` watcher failed its check
on a trailing carriage return, and that launch was done by hand.

## 10. Evidence and limits

Files in this folder:

| Path | Content |
|---|---|
| [evidence/j19-1/host_inventory_20260925T024916Z.txt](evidence/j19-1/host_inventory_20260925T024916Z.txt) | Full `j19-1` host, driver, firmware and package inventory |
| [evidence/j19-1/cwsr_handlers.txt](evidence/j19-1/cwsr_handlers.txt), [evidence/j19-11/cwsr_handlers.txt](evidence/j19-11/cwsr_handlers.txt), [evidence/h21-15/cwsr_handlers.txt](evidence/h21-15/cwsr_handlers.txt) | Handler identity of each installed module |
| [evidence/j19-1/control_default/](evidence/j19-1/control_default/) | Default run: platform, command, loss, telemetry, non-finite stop record |
| [evidence/j19-1/numabind/](evidence/j19-1/numabind/) | `numactl` run: platform, command, loss, telemetry, status |
| [evidence/j19-11/reproA/](evidence/j19-11/reproA/), [evidence/j19-11/reproB/](evidence/j19-11/reproB/) | Tripwire and `numactl` runs: platform, loss, status, tripwire summary |
| [evidence/j19-1/corrected_module_audit.txt](evidence/j19-1/corrected_module_audit.txt) | Candidate and baseline module audit |
| [evidence/j19-1/oneshot_install.txt](evidence/j19-1/oneshot_install.txt), [evidence/j19-1/oneshot_custom.cfg](evidence/j19-1/oneshot_custom.cfg), [evidence/j19-1/reboot_request.txt](evidence/j19-1/reboot_request.txt), [evidence/j19-11/reboot_request.txt](evidence/j19-11/reboot_request.txt) | One-shot installation and the two reboot requests |
| [evidence/j19-9/host_first_look.txt](evidence/j19-9/host_first_look.txt) | `j19-9` GPUs, driver, firmware, handler identity and mounts |
| [evidence/j19-9/long_numabind/](evidence/j19-9/long_numabind/) | Stock-driver `numactl` run that went NaN at step 354: platform, command, loss, telemetry with KFD eviction counters, non-finite stop record |
| [evidence/j19-9/tripwire_numabind_wedge/](evidence/j19-9/tripwire_numabind_wedge/), [evidence/j19-9/dmesg_wedge_excerpt.txt](evidence/j19-9/dmesg_wedge_excerpt.txt) | The wedged tripwire run (telemetry through 21:31) and the kernel log of the MES non-response |
| [evidence/j19-9/halt_clear_observation.txt](evidence/j19-9/halt_clear_observation.txt) | Clearing `halt_if_hws_hang` at runtime, the reset and the return |
| [evidence/j19-9/corrected_module_audit.txt](evidence/j19-9/corrected_module_audit.txt), [evidence/j19-9/armed_install.txt](evidence/j19-9/armed_install.txt), [evidence/j19-9/corrected_boot/](evidence/j19-9/corrected_boot/) | `j19-9` module build and audit, arming, and the corrected-driver load receipt |
| [evidence/j19-9/corrected_default/](evidence/j19-9/corrected_default/) | Corrected-driver default run: platform, command, loss, telemetry and status as of 02:35 on September 27 (run in progress) |
| [tools/cwsr_fix/](tools/cwsr_fix/) | Handler extraction, candidate generation, module build, audit, GRUB one-shot and armed-loader scripts |

Full run directories, training logs and MLPerf logs are under
`/shared/chcai/mi450_a0_newcluster/runs/`. The `j19-11` and `h21-15` run
directories are on those nodes' local disks. The corrected module and its
build tree are on `j19-1` and `j19-9` under `/var/tmp/chcai/cwsrmod` and
`/var/lib/chcai-cwsrfix`.

Limits:

- The corrected-driver result is one run, still in progress. It is compared
  with two stock default runs (NaN at 111 and 123) and two stock bound runs
  (NaN at 354; finite to 1,912 until the node was lost).
- None of the stock runs, and not yet the corrected run, ran for several
  hours.
- The non-AutoNUMA eviction sources were not traced, and the first non-finite
  tensor under binding was not localized.
- The wedge mechanism is observed directly on `j19-9`. The hung reboots of
  `j19-1` and `j19-11` have no console capture.
- Node losses 1–3 have no recovered kernel logs; their causes are unknown.
