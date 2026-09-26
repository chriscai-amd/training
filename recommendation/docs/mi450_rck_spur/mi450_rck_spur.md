# MI450 A0 on the RCK spur cluster (gfx1250, four GPUs, full model)

Updated **2026-09-26** (all times UTC). Node states were last checked at
**00:05 UTC on September 26**.

This cluster (hosts `ctheliosr-rck-g02-j19-*` and `ctheliosp-rck-g02-h21-*`,
scheduled by ROCm spur) was new to this investigation on September 24. The
work asked three questions: does the A0 NaN still reproduce on this cluster's
BKC, can the unshrunk four-GPU model train NaN-free for several hours, and
which component produces the NaN. Companion records:
[MI450 A0](../mi450_a0/mi450_a0.md) (one GPU, shrunk model),
[A0 CWSR defect and fix](../mi450_a0/cwsr_bank_corruption_fix.md),
[MI450 B0](../mi450_b0/mi450_b0.md), and the four-GPU convergence reference
[1P4G a37-2](../mi450_1P4G/ctheliosr-1b112-a37-2.md), whose configuration
every run here reuses.

## Current status: NaN reproduces; corrected driver built, blocked on hung nodes

**The NaN reproduces with the full four-GPU model.** With the a37-2 converged
configuration (all embedding rows, local/global batch 1024/4096, seed 1), the
first non-finite loss appeared at **step 111 on `j19-1`** and at **step 123 on
`j19-11`**. Both nodes run the stock BKC with AutoNUMA on.

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

**AutoNUMA is the trigger here, and binding memory removes it.** The default
run made **2.54 million NUMA page-table updates per minute**. Under
`numactl --membind=0,1` that fell to **zero**, and the same configuration
stayed finite through **step 1,912** (about 46 minutes of training) until
`j19-1` was lost. That is one run: the same arm on `j19-11` stalled at step 6
without a numerical verdict.

**The decisive corrected-driver test has not run yet.** The corrected amdgpu
was built on `j19-1` from its own DKMS source. It differs from a faithful
rebuild of the installed module only in the handler and two handler-size
bytes, exactly as in the A0 team's audit. It is installed as a one-shot boot
entry. The reboot into it **hung in shutdown**, as the earlier reboot of
`j19-11` did, so both A0 nodes need an out-of-band power cycle.

**No several-hour NaN-free run exists yet.** The longest finite run is the
1,912-step `numactl` run. Node losses, not NaNs, ended every longer attempt:
five losses on four nodes in about 16 hours, two of them hung reboots
([§8](#8-cluster-reliability)).

| Question | Recorded answer |
|---|---|
| Does the NaN reproduce on this cluster? | **Yes, in 2 of 2 default runs**: step 111 (`j19-1`) and step 123 (`j19-11`). |
| Was the model shrunk? | **No.** `EMBEDDING_ROW_SCALE=1.0` on four GPUs with the a37-2 settings. |
| Which component? | **The gfx1250 CWSR trap handler in amdgpu (KFD)**, by the chain in [§6](#6-nan-triage-component-trigger-and-mechanism). The direct corrected-driver run is still pending. |
| What triggers it here? | Linux automatic NUMA balancing (`kernel.numa_balancing=1`), which evicts KFD queues and forces CWSR saves. |
| Is there a working mitigation? | `numactl --membind=0,1` removed the trigger and ran 1,912 finite steps (n=1). It isn't a fix: the handler stays faulty. |
| Several-hour end-to-end run? | **Not yet.** The longest finite run is 1,912 steps (46 minutes). |
| Is the corrected driver ready? | **Built, audited and armed on `j19-1`**, but not yet loaded. |
| What blocks progress? | `j19-1` and `j19-11` are hung in shutdown and need a BMC power cycle. |

### Node status at 00:05 UTC, September 26

| Node | GPUs | State | Needs |
|---|---|---|---|
| `ctheliosr-rck-g02-j19-1` (`10.190.178.47`) | 4 × `1002:75c1` | Hung in shutdown since the 20:30 reboot; absent from `sinfo` | BMC power cycle. It should then boot the corrected driver once ([§7.3](#73-activation-on-j19-1)). |
| `ctheliosr-rck-g02-j19-11` (`10.190.178.179`) | 4 × `1002:75c1` | Hung in shutdown since the 18:50 reboot; absent from `sinfo` | BMC power cycle; `/nfs_spur` export |
| `ctheliosr-rck-g02-j19-7` (`10.190.178.15`) | 4 × `1002:75c1` | `drain` | TheRock CI ran on its GPUs outside the scheduler ([§8.4](#84-ci-workloads-outside-the-scheduler)) |
| `ctheliosp-rck-g02-h21-15` (`10.190.178.90`) | 4 × `1002:75c1` | `down`; sshd up, node agent dead since 11:22:34 | spurd restart; `/nfs_spur` export |
| `ctheliosr-rck-g02-j19-16` | not inspected | `idle`, but a job did not start within 90 s | — |
| `ctheliosr-rck-g02-j19-9`, `ctheliosp-rck-g02-h21-7` | — | `resv` and `down`; reserved for other users | — |
| `ctheliosp-rck-g02-h21-13`, `ctheliosp-rck-g02-h21-14` | `h21-14`: `1002:75c7` | Other users' jobs | — |
| `ctheliosp-rck-g02-h21-12`, `ctheliosp-rck-g02-h21-16` | `h21-16`: `1002:75c7` | `idle`; no `/shared`, no `unsquashfs` | Not A0-equivalent |

### Needed from admins

1. Power-cycle `j19-1` and `j19-11` through the BMC. Both hung after a systemd
   reboot request: the kernel answers ping, but sshd and spurd have stopped. If
   possible, capture `j19-1`'s serial console (`ttyS1`, 115200 baud) first; it
   would show what blocks the shutdown ([§8.2](#82-hung-soft-reboots-and-halt_if_hws_hang)).
2. Add `10.190.178.179` (`j19-11`) and `10.190.178.90` (`h21-15`) to the
   `/nfs_spur` export on `10.210.2.150` ([§8.3](#83-nfs-export-gap)).
3. Restart spurd on `h21-15`.
4. Decide whether `halt_if_hws_hang=1` should stay enabled on shared nodes
   ([§8.2](#82-hung-soft-reboots-and-halt_if_hws_hang)).
5. Check the flood of corrected PCIe errors on `j19-1` ([§8.5](#85-pcie-aer-flood-on-j19-1)).

### Next steps

1. When `j19-1` returns, confirm which driver loaded ([§7.3](#73-activation-on-j19-1)).
   On the corrected driver, run the default AutoNUMA-on configuration past
   step 123. No NaN would confirm the handler as the culprit; keep that run
   going toward AUC 0.75 for the several-hour end-to-end result.
2. If `j19-1` comes back on the stock driver, re-arm the one-shot entry and
   ask for another power cycle rather than a soft reboot.
3. After the test, remove the one-shot files ([§7.4](#74-removal)) and
   power-cycle to return to the stock driver.

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
[1P4G record](../mi450_1P4G/ctheliosr-1b112-a37-2.md):

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

| Run (September 25) | Node, boot | Configuration | Result |
|---|---|---|---|
| `full4_a37cfg_20260925T035054Z` | `j19-1`, `c4dfb7f9` | Default (AutoNUMA on) | Steps 1–110 finite; `nan` from **step 111** at 04:04:44; stopped by the monitor |
| `full4_a37cfg_numabind_20260925T040549Z` | `j19-1`, `c4dfb7f9` | `numactl --membind=0,1` | Finite through **step 1,912** at 05:02:55; `j19-1` lost at about 05:03 |
| Arm A, 1,500-step bound | `h21-15` | Default (AutoNUMA off on this host) | Node agent lost at 11:22:34, about 19 minutes after launch; the run directory is on the node's local disk and wasn't recovered |
| `reproA_default_tripwire_…_20260925T150501Z` | `j19-11`, `b4ecf5cc` | Default plus tripwire, 600-step bound | Finite through step 122; **step 123** non-finite gradient localized ([§6.1](#61-localization)) |
| `reproB_numabind_…_20260925T152003Z` | `j19-11`, `b4ecf5cc` | `numactl --membind=0,1`, 1,500-step bound | **Stalled after step 6** (15:31:10). One GPU stayed 100 % busy with VRAM held; the ranks couldn't be killed |
| reproC | `j19-11` | Triton without the extended VGPR banks (`TRITON_AMD_TARGET_FEATURES=-1024-addressable-vgprs`) | Not run; the GPU was wedged |

| `j19-1` telemetry | Default (NaN at step 111) | `numactl` bind (finite to step 1,912) |
|---|---:|---:|
| `numa_pte_updates` per minute | 2,543,480 | 0 |
| `numa_hint_faults` per minute | 1,059,941 | 15 |
| `numa_pages_migrated` per minute | 158,360 | 2 |
| Peak sampled VRAM, GPUs `0001`/`0002`/`0003`/`0004` (GiB) | 424.6 / 406.4 / 391.1 / 351.7 | 399.5 / 406.4 / 391.1 / 398.7 |

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
- **The same trigger is active, and removing it removed the NaN** in the one
  completed test ([§6.3](#63-trigger), [§6.4](#64-why-numactl---membind01-suppresses-it)).
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
minute ([§5](#5-experiment-record)), so the scanner was invalidating the
trainer's address space continuously during training.

### 6.4 Why `numactl --membind=0,1` suppresses it

`numactl --membind` installs an `MPOL_BIND` task policy without
`MPOL_F_NUMA_BALANCING`. The kernel's NUMA-balancing scanner skips VMAs whose
policy doesn't allow migrate-on-fault, so it never write-protects the
trainer's pages, and no MMU-notifier invalidation reaches KFD from this path.
The measured page-table update rate fell from 2.54 million to 0 per minute.
Nodes 0 and 1 hold all of the host's system memory, so the binding doesn't
restrict placement. A0's policy counterfactual agrees: plain binding gave 0
SVM eviction pairs and 0 register copies, while `--balancing` with the same
node mask restored them.

This removes one trigger, not the defect. Queue evictions from other sources,
such as memory pressure, `fork` or explicit preemption, still run the faulty
handler.

### 6.5 What is not established

- The corrected-driver run on this cluster hasn't happened
  ([§7.3](#73-activation-on-j19-1)).
- Suppression rests on one 1,912-step run. The `j19-11` `numactl` arm stalled
  at step 6 for an unknown reason.
- The victim-side control (reproC, Triton without the extended VGPR banks)
  didn't run.
- The trigger rate was measured on `j19-1` only; KFD eviction counts for these
  runs weren't retained.
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

```bash
sudo grub-editenv /boot/grub/grubenv unset next_entry
sudo rm -f /boot/grub/custom.cfg /etc/systemd/system/chcai-cwsrfix-load.service \
     /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
sudo rm -rf /var/lib/chcai-cwsrfix
```

After removal, the next boot loads the stock driver.

## 8. Cluster reliability

### 8.1 Node losses

| Time (September 25) | Node | Activity | Symptom | Recovery |
|---|---|---|---|---|
| About 05:03 | `j19-1` | `numactl`-bound training, step 1,912 | Node agent lost | Rebooted at about 14:16 (boot `b6eaa6f0`) |
| About 08:29 | `j19-7` | CPU-only dataset download and verification | Node agent lost | Back by about 20:20 (boot `c7c172b6`); drained by 00:05 on September 26 |
| 11:22:34 | `h21-15` | Arm A, about 19 minutes after launch | Node agent lost; sshd still up | Still down |
| 18:50 | `j19-11` | systemd reboot after the reproB GPU wedge | Hung in shutdown | Still hung |
| 20:30 | `j19-1` | systemd reboot into the one-shot entry, GPUs idle | Hung in shutdown | Still hung |

The last two were reboots I requested.

### 8.2 Hung soft reboots and `halt_if_hws_hang`

Both hung nodes show the same signature. The kernel answers ping and resets
TCP connections, but sshd, rpcbind and spurd have stopped, and ping never
dropped, so the machine never reached firmware. `j19-11` had unkillable ranks
and one GPU at 100 % busy when rebooted. `j19-1` had no KFD processes.

**Hypothesis, not confirmed.** Every `j19` node loads amdgpu with
`halt_if_hws_hang=1 gpu_recovery=0`, a firmware-debug setup. In this driver:

- `mes_v12_1.c:251` (gfx12.1 MES) runs `while (halt_if_hws_hang) schedule();`
  after `MES(%d, %d) failed to respond to msg=%d`.
- `kfd_device_queue_manager.c:2286` does the same after a queue-preemption
  fence timeout. The source comment says this halts the driver thread "in
  order not to mess up CP states before doing scandumps for FW debugging".

Driver teardown at reboot sends MES messages. The A0 record notes MES
`REMOVE_QUEUE` failures in older logs and a remove-queue hang fix that a
testing MES build lacks. One unanswered message would stop the reboot forever,
and would also explain why the reproB ranks on `j19-11` never exited. Against this, `j19-1`'s kernel log for the 3.3
hours before its reboot has no amdgpu or MES messages. The journal isn't
persistent, so a serial-console capture of a hung shutdown is needed to
settle it.

The parameter is writable at runtime
(`/sys/module/amdgpu/parameters/halt_if_hws_hang`, mode 0644). Clearing it
before a reboot is an untested option for admins.

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
bash tools/cwsr_fix/oneshot/install_oneshot.sh                           # arm the one-shot boot
```

The build and audit need no root access. `install_oneshot.sh` uses `sudo`
and does not reboot; reboot through the BMC, because soft reboots hung on
these nodes ([§8.2](#82-hung-soft-reboots-and-halt_if_hws_hang)).

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
| [tools/cwsr_fix/](tools/cwsr_fix/) | Handler extraction, candidate generation, module build, audit and one-shot scripts |

Full run directories, training logs and MLPerf logs are under
`/shared/chcai/mi450_a0_newcluster/runs/`. The `j19-11` and `h21-15` run
directories are on those nodes' local disks. The corrected module and its
build tree are on `j19-1` under `/var/tmp/chcai/cwsrmod` and
`/var/lib/chcai-cwsrfix`.

Limits:

- Every finding here comes from four runs on two A0 nodes that produced
  training steps. None of them ran for several hours.
- The component attribution rests on matching bytes, tensor and trigger plus
  A0's controlled tests, not on a corrected-driver run on this cluster.
- The hung-reboot explanation is a source-based hypothesis without a console
  capture.
- Node losses 1–3 have no recovered kernel logs; their causes are unknown.
