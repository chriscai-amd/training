# MI450 1P4G, host `ctheliosp-1b112-a37-1` (gfx1250, full model)

Updated **2026-09-28**.

> **Latest status (2026-09-28): NaN / fault repro**
>
> - **No NaN loss so far.** Every logged loss on this host is finite:
>   3,000 steps (2026-09-19), 200 steps (2026-09-26) and every 2026-09-27 run
>   up to its last logged step.
> - **The A0 CWSR defect reproduces here as GPU memory faults, not as NaN.**
>   - With the attention backward at `HSTU_BWD_BLOCK_N=128` (R1), **3 of 3**
>     runs launched 15–16 h after boot died on a GPU memory fault. The two
>     that named the kernel named `_hstu_attn_bwd`, shortly after step 100.
>   - The same config passed 600 steps **twice** 6–7 h after an earlier boot.
>   - Losses at steps 50 and 100 are bit-identical in the passing and failing
>     runs, so the fault is intermittent and timing-dependent. The computed
>     values are unchanged.
>   - The only difference found is the host's memory state
>     ([§8](#8-2026-09-27-nan--fault-repro-r1-faults-and-cwsr-exposure)).
> - **This host needs the CWSR handler fix, as A0 does.**
>   - It loads the known-faulty handler `0f718b5e…`.
>   - The A0/RCK trigger is absent: AutoNUMA is off, and XNACK is on
>     (`noretry=0`).
>   - Kernel memory compaction still forces KFD queue evictions: in one traced
>     run, **1,569 of 1,580** evictions came from compaction.
>   - Faults coincided with compaction bursts. Every eviction saves and
>     restores the running waves through the faulty handler.
> - **Current guidance:**
>   - Until the corrected handler `68c31ab2…` is installed, run R1 only on a
>     freshly booted host, or keep the default `BLOCK_N=64`. `BLOCK_N=64`
>     passed 3,200 steps, but not in the high-compaction state.
>   - Stay at local 1,024 ([§7](#7-2026-09-26-host-halt-at-local-2304--gbs-9216)).
>   - Keep `AMDGCN_USE_BUFFER_OPS=0`.
> - **Driver-team report.** The handler defect and its fix are proven at
>   component level on A0, and RCK adds an end-to-end A/B. Filing can start
>   now. The gaps are a portable minimal reproducer, a source-level fix in the
>   driver team's build flow and the final RCK result. The owner is the
>   amdgpu KFD team. The six to-dos for this host are listed in
>   [§8.5](#85-readiness-for-a-driver-team-report).
> - **The corrected-driver comparison on this host is planned, not run.**
>   Loading the rebuilt driver needs someone who can power-cycle the host
>   ([§8.6](#86-corrected-driver-comparison-on-this-host-plan-not-run)).

This document is split into two parts:

- **[Part A, Performance](#part-a-performance)**: the 2026-09-26 200-step
  full-model perf run, local batch **1,024** / global **4,096**. It measured
  **2.004 s/step = 2,044 samples/s** over steps 50–200, and a Chrome trace of
  steps 52–56 was captured and stitched across the four ranks. GEMMs take
  **62%** of GPU time. Two hipBLASLt kernels with a 32×16×32 macro tile
  account for about 55% of it.
- **[Part B, Platform, functional and stability record](#part-b-platform-functional-and-stability-record)**:
  - the host stack, now reprovisioned (2026-09-24);
  - the full-model fit and sharding;
  - the 2026-09-19 3,000-step finite-loss run and its Socket-transport
    workaround;
  - the 2026-09-26 **host halt (kernel oops)** in an attempt at local 2,304 /
    GBS 9,216, which needed a manual power cycle;
  - the 2026-09-27 **NaN / fault repro**: `BLOCK_N=128` GPU faults and the
    compaction-driven queue evictions behind them;
  - fabric registration findings;
  - reproduction steps and the evidence inventory.

Full-model facts shared by all runs: all **11 FP32 embedding tables** are
allocated, with **293,051,712 rows × 512 dimensions = 558.95 GiB of embedding
weights**. `EMBEDDING_ROW_SCALE=1.0`; the model and vocabulary are not shrunk.
Four-GPU training uses **host-staged Socket communication**
([§5.2](#52-socket-transport-and-runtime-controls)).

All experiment and capture times below are **UTC**. Firmware build dates are
quoted as reported; their source does not specify a timezone.

Source commits:

- The 2026-09-19 runs used repository commit
  **`082e3849dd9588ee5fbfd61eeae8c0754311d05b`** on `chcai/mi450`.
- The 2026-09-26 runs used **`d07045dfe3e39d5d744af5c748febd00742d243a`**.
- The 2026-09-27 runs used **`b29310e3…`**, and the last one used
  **`f6f5fd6f…`**. Outside `docs/` their code matches `d07045d`.

Companion records: [the other 1P4G host](../ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md),
[A0](../../mi450_a0/mi450_a0.md),
[B0](../../mi450_b0/mi450_b0.md),
[current A0 controls and evidence](../../mi450_a0/mi450_a0.md#stack-and-controls), and
[RCK spur / CWSR handler](../../mi450_rck_spur/mi450_rck_spur.md).

Host-local artifact roots are listed in [§11](#11-evidence-inventory-and-limits).
The 2026-09-19 root no longer exists on the host. The traces of the perf run
are stored next to this document in
[`traces/trace_step52.json.gz`](traces/trace_step52.json.gz).

## Start here

| Question | Recorded answer |
|---|---|
| What throughput does the full model reach? | **2.004 s/step, 2,044 global samples/s** (511 per GPU) over steps 50–200 at local 1,024 / GBS 4,096. The trainer logs MFU **5.0–5.2%** and HFU **1.6–2.0%** by its own definitions ([§1.2](#12-throughput)). |
| Where does the step time go? | GEMM **62.4%**, HSTU attention **17.3%** (backward ≈ 6× forward), memcpy/memset **10.8%**, RCCL **4.0%**, of rank-0 GPU kernel time ([§1.5](#15-trace-breakdown)). |
| Is there a trace? | **Yes.** A stitched 4-rank CPU+GPU trace of steps 52–56 (4.9 MB gz) ([§1.4](#14-trace-capture)). |
| Largest optimization lead? | HSTU linear-layer GEMMs run on a **32×16×32 macro tile**. The two top kernels take about 1.04 s/step ([§1.6](#16-optimization-leads)). |
| Can the batch be raised to fill HBM? | **Not safely on this host.** Local 2,304 / GBS 9,216 halted the host with a kernel oops ([§7](#7-2026-09-26-host-halt-at-local-2304--gbs-9216)). Stay at local 1,024. |
| Does the full model fit? | **Yes at local batch 1,024 on four GPUs.** All rows and all 512 columns are allocated; embedding weights total **558.95 GiB**. |
| Did NaN loss reproduce? | **No NaN.** On 2026-09-19 one run finished with **3,000/3,000 finite console and MLPerf losses**. An earlier interrupted run recorded 1,530 finite steps. The 2026-09-26 perf run's 200 losses are finite, and so is every 2026-09-27 loss up to each run's last logged step ([§8](#8-2026-09-27-nan--fault-repro-r1-faults-and-cwsr-exposure)). |
| Does the A0 CWSR defect affect this host? | **Yes, as GPU memory faults.** At `BLOCK_N=128`, 3 of 3 runs launched 15–16 h after boot faulted, and the two with a named kernel named `_hstu_attn_bwd`. The same config passed 600 steps twice at 6–7 h. The faulty handler is loaded, and memory compaction evicts the GPU queues even with AutoNUMA off and XNACK on ([§8](#8-2026-09-27-nan--fault-repro-r1-faults-and-cwsr-exposure)). |
| Is `BLOCK_N=128` (R1) safe here? | **Not until the handler is fixed.** Its losses match `BLOCK_N=64`, and the perf gain holds, but whether it survives depends on the host's memory state. |
| Is there enough evidence for the driver team? | **Yes for the handler defect and fix, at component level** (A0 replays plus the RCK A/B). This host adds supporting evidence only. Gaps are listed in [§8.5](#85-readiness-for-a-driver-team-report); the corrected-driver comparison here is planned, not run ([§8.6](#86-corrected-driver-comparison-on-this-host-plan-not-run)). |
| Was the model shrunk? | **No.** `EMBEDDING_ROW_SCALE=1.0`, full tables, three HSTU layers, sequence limit 4096. |
| Which attention arm ran? | **Uncapped**, `HSTU_BWD_MAX_VGPR=0`, exact Triton `7ff97e310935b4a79794878dbc911f9af25d38d9`. |
| What unblocked four-GPU training? | **Host-staged NET/Socket**, with P2P/SHM and direct fabric paths disabled. |
| Is the fabric fixed? | **No** as of 2026-09-19 ([§10](#10-fabric-registration-and-recovery-limits)). Not rechecked after the reprovisioning. |
| Was convergence measured? | **No** on this host. Evaluation was disabled or not reached, and every step recorded here is inside the 24,000-step warmup. |

## Contents

**Part A, Performance**

1. [200-step full-model perf run, 2026-09-26](#1-200-step-full-model-perf-run-2026-09-26)

**Part B, Platform, functional and stability record**

2. [Host and software stack](#2-host-and-software-stack)
3. [Dataset and full model](#3-dataset-and-full-model)
4. [2026-09-19 3,000-step run: training settings](#4-2026-09-19-3000-step-run-training-settings)
5. [2026-09-19 experiment record](#5-2026-09-19-experiment-record)
6. [2026-09-19 results and comparison](#6-2026-09-19-results-and-comparison)
7. [2026-09-26 host halt at local 2,304 / GBS 9,216](#7-2026-09-26-host-halt-at-local-2304--gbs-9216)
8. [2026-09-27 NaN / fault repro: R1 faults and CWSR exposure](#8-2026-09-27-nan--fault-repro-r1-faults-and-cwsr-exposure)
9. [Reproduction and analysis commands](#9-reproduction-and-analysis-commands)
10. [Fabric registration and recovery limits](#10-fabric-registration-and-recovery-limits)
11. [Evidence inventory and limits](#11-evidence-inventory-and-limits)

---

# Part A, Performance

## 1. 200-step full-model perf run, 2026-09-26

### 1.1 Run identity and configuration

| Item | Value |
|---|---|
| Run | **`perf_gbs4096_s200_20260926T214509Z`** |
| Host stack | Reprovisioned 2026-09-24 stack (kernel `fbk5_npi_brcmrdma9`, el10 amdgpu DKMS), boot `78b316ae-dfc3-4fb7-8959-389789ae0439` ([§2.1](#21-host-cpu-gpus-and-firmware)) |
| Source | `d07045dfe3e39d5d744af5c748febd00742d243a`. `repo-status.txt` lists only untracked CWSR-extractor outputs, no source edits. |
| Image | `recommendation-gfx1250-20260910:triton-7ff97e`, ID `sha256:fc5040f7a9e430aef05bf40a67fcd3375e4a6997d617f992484d4459816da0fe`. Rebuilt on 2026-09-26 from the same recipe, so its ID differs from the 2026-09-19 image. It reports Torch `2.11.0+rocm7.14.0a20260625`, HIP `7.14.60850`, Triton `3.8.0`, TorchRec `1.7.0a0+bf55480`. |
| Model | Full yambda-5b model, MLPerf `yambda_5b.gin` defaults, `EMBEDDING_ROW_SCALE=1.0` |
| Batch | **1,024 per rank × 4 = 4,096 global**. The trainer has no gradient accumulation. |
| Data | `START_TS=0`, seed 1, `--mode streaming-train-eval`. No eval ran within 200 steps. |
| Stop | `DIE_AT_STEP=200` → `die_at_step=200 hit at train_ts=26 batch=21 global_step=200 → sys.exit(42)`. Docker exit 1 is expected. |
| Profiler | `OUTPUT_TRACE=1`, 5-step window starting at step 52, all ranks |
| Communication | Socket-only env ([§5.2](#52-socket-transport-and-runtime-controls)), plus `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET` |
| Runtime env | `HSTU_HAMMER_KERNEL=TRITON`, `HSTU_BWD_MAX_VGPR=0`, `AMDGCN_USE_BUFFER_OPS=0`, both PyTorch allocator variables empty, pinned `HIPBLASLT_TENSILE_LIBPATH` |
| Planner | `hbm_cap_gb=260` (gin default). The 2026-09-19 run used `HBM_CAP_GB=400`, so its shard placement may differ. |
| Kernel health | No new GPU or kernel fault in `dmesg.live.txt`, before or after. The preflight on this boot passed (`preflight_postreboot/`). |

### 1.2 Throughput

Step boundaries are taken from the trainer's `Step N perf` lines. Throughput
is global samples per second.

| Window | s/step | Samples/s (global) | Per GPU | Logged `tflops_algo/gpu` / MFU | Logged `tflops_real/gpu` / HFU |
|---|---:|---:|---:|---|---|
| Steps 50–100 | 1.983 | 2,065 | 516 | — | — |
| Steps 100–150 | 1.984 | 2,065 | 516 | 163.0 / 5.2% | 50.3 / 1.6% |
| Steps 150–200 | 2.045 | 2,003 | 501 | 158.1 / 5.0% | 62.4 / 2.0% |
| **Steps 50–200** | **2.004** | **2,044** | **511** | | |

The 50–200 figure uses wall-clock times: step 50 at 21:57:56.825 and step 200
at 22:02:57.400, i.e. 150 steps in 300.6 s. The trainer's step-50 line
(15.1 s/step) averages over startup and is not a steady-state number.

Losses at steps 50/100/150/200 are **0.13883 / 0.13918 / 0.13856 / 0.13895**,
all finite.

MFU, HFU and `fill` (31–39%) use the trainer's own definitions. Low HFU
together with a GEMM-dominated profile points to small, inefficient GEMM tiles
([§1.5](#15-trace-breakdown)). It does not point to idle GPUs.

### 1.3 Startup and memory

| Phase | Time | Duration |
|---|---|---|
| Launch → `init_start` | 21:45:09 → 21:45:16.4 | 7 s |
| `run_start` → `block_start` | 21:45:21.3 → 21:55:04.8 | **9.7 min**: streaming dataset scan (`train_samples=2,290,835,423`) |
| `block_start` → step 50 | → 21:57:56.8 | 2.9 min: first steps, including compile/autotune warmup |

Peak sampled used VRAM, from the 5 s sysfs sampler in `vram.log`:

| GPU | Peak MiB | Peak GiB | Headroom to 432 GiB |
|---|---:|---:|---:|
| 0 | 366,034 | **357.5** | 74.5 |
| 1 | 347,718 | 339.6 | 92.4 |
| 2 | 330,690 | 322.9 | 109.1 |
| 3 | 301,478 | 294.4 | 137.6 |

The crashed GBS-9,216 attempt sat at **425.0 / 415.5 / 368.8 / 411.3 GiB**,
with GPU0 at 98.4% of HBM ([§7](#7-2026-09-26-host-halt-at-local-2304--gbs-9216)).

### 1.4 Trace capture

| File | Content |
|---|---|
| [`traces/trace_step52.json.gz`](traces/trace_step52.json.gz) | Run `perf_gbs4096_s200_20260926T214509Z`, steps 52–56: **4 ranks stitched, CPU+GPU, 252,272 events, 4.9 MB**. Open it in Perfetto or `chrome://tracing`. |

The trainer writes raw per-rank PyTorch profiler output to
`results/<RUN_NAME>/trace_step52_rank{0..3}.json`, 22–23 MB each. The
stitched file was produced from those files with the repository script, and
the raw files were then deleted rather than tracked:

```bash
cd recommendation
python3 scripts/stitch_traces.py results/<RUN_NAME> --step 52 --gzip \
  --out docs/mi450_1P4G/ctheliosp-1b112-a37-1/traces/trace_step52.json.gz
```

The script drops the `Spans` track, orders CPU tracks before GPU tracks, and
remaps pids and flow ids per rank.

The profiled steps 52–56 took **1.833 / 1.741 / 1.979 / 1.969 / 1.776 s**
(9.298 s window, mean 1.86 s/step). That is about 7% faster than the
steps 50–200 wall-clock average. The window is short, and step time varies
with the jagged token count.

### 1.5 Trace breakdown

Rank 0 GPU kernel time is summed across streams: 9.524 s over the 9.298 s
window, so streams overlap and the GPU is busy almost all the time.

| Category | Share | Notes |
|---|---:|---|
| GEMM (hipBLASLt) | **62.4%** | HSTU linear layers, forward and weight gradient |
| HSTU attention (Triton) | **17.3%** | `_hstu_attn_bwd` 1.416 s vs forward 0.233 s: backward ≈ **6×** forward |
| memcpy / memset | 10.8% | `Memcpy HtoD` 1.003 s ≈ 0.2 s/step of input host-to-device copy |
| Other | 4.3% | Elementwise, reductions, and so on |
| RCCL | 4.0% | 0.383 s on rank 0, 0.281 s on rank 3; Socket transport |
| LayerNorm | 0.7% | |
| TBE (embedding) | 0.4% | |
| Optimizer | ~0% | |

Top kernels:

| Kernel | Time over 5 steps | Maps to |
|---|---:|---|
| `Cijk_Ailk_Bjlk_BBS_BH_Bias…MT32x16x32_MI16x16x1` | **2.645 s** | `aten::addmm` with bias: [M, 512] × [512, 2048] forward |
| `Cijk_Ailk_Bljk…MT32x16x32` | **2.572 s** | `aten::mm` weight gradient: [512, M] × [M, 2048] |
| `…MT256x128x64…` GEMM | 0.567 s | Other linear shapes |

M is the jagged token count, about 2.0–2.26 M rows per step, and varies from
step to step.

The two top kernels account for about **55% of GPU time, ≈ 1.04 s/step**.
They run a 32×16×32 macro tile on GEMMs with millions of rows. This looks
like a fallback or untuned solution for these (variable-M) shapes on gfx1250,
not a hardware limit.

Communication is small (4%), so the Socket-only workaround is not the main
cost at this scale.

### 1.6 Optimization leads

1. **GEMM solution selection or tuning** for the HSTU linear shapes
   ([512|2048] × M with M ≈ 2 M, BF16, with and without bias). Options:
   - offline hipBLASLt tuning for bucketed M;
   - forcing a large-tile solution;
   - padding M to a tuned bucket.

   This is the biggest single lever, since the two kernels take about 1 s of
   a 2 s step. Compare the GEMM-tuning item in [perf_opt.md](../../perf_opt.md).
2. **HSTU attention backward** (≈ 6× forward): block sizes and VGPR cap. The
   uncapped arm is used here; B0 cap-256 evidence in the B0 doc covers
   correctness, not speed.
3. **Input H2D** (≈ 0.2 s/step): overlap it with compute on a side stream, or
   pin and prefetch the next batch.
4. **Startup**: the 9.7 min dataset scan before the first step matters for
   short runs and time-to-train.

None of these was tried in this run.

### 1.7 Comparison with earlier full-model runs and limits

| Run | Host / stack | Batch | Throughput | Caveats |
|---|---|---|---|---|
| **This run, 2026-09-26** | a37-1, fbk5 / el10 amdgpu | 1,024 / 4,096 | **2.004 s/step, 2,044 samples/s** | 150 measured steps, Socket-only env, trace on for 5 steps |
| a37-2 3,000-step run | a37-2, fbk2 / el9 amdgpu | 1,024 / 4,096 | ≈ 2.14 s/step, first to last logged step 04:20:33 → 06:07:22 | Different host and only `NCCL_SOCKET_IFNAME=lo`, so indicative only |
| a37-1 3,000-step run, 2026-09-19 | a37-1, fbk2 / el9 amdgpu | 1,024 / 4,096 | Not extracted; the evidence root was lost in the home-directory reset | `HBM_CAP_GB=400` sharding, observer overhead |

Limits:

- One 200-step run.
- Every step is inside the LR warmup, and there is no eval. Throughput with
  eval and checkpointing is not measured.
- The MFU/HFU values are the trainer's estimates.
- The breakdown is from rank 0 only, except for the RCCL figures.
- Profiler overhead within the window is not quantified.

---

# Part B, Platform, functional and stability record

## 2. Host and software stack

### 2.1 Host, CPU, GPUs and firmware

The host was **reprovisioned on 2026-09-24**: new kernel, OS major version and
amdgpu build. The 2026-09-19 results were measured on the earlier stack.

| Item | 2026-09-19 (3,000-step run) | 2026-09-26 (perf run, halt) |
|---|---|---|
| Hostname | **`ctheliosp-1b112-a37-1.mnb.dcgpu`** | same |
| Host OS | CentOS Stream 9 | **CentOS Stream 10** |
| Host RAM | not recorded | 250 GiB |
| CPU topology | **1 socket**, 128 cores, 256 present logical CPUs, **255 online (0–254), CPU255 offline**; `AMD Eng Sample: 100-000001046-06` | not rechecked |
| GPUs | **4 × MI450**, `gfx1250`, PCI `1002:75c1`, device revision `0x00`, 256 reported compute units per GPU, wave32 | same |
| PCI addresses | `0001:04:00.0`, `0002:04:00.0`, `0003:04:00.0`, `0004:04:00.0` | same |
| HBM | **432 GiB per GPU**, 1,728 GiB aggregate; reported as 442,368 MiB per device | same |
| IFWI | **`113-M4500001-640C`**, build **2026/08/19 23:29**, version `00198932`, name `AMD MI450X_GENERIC_SEC` | not rechecked |
| Kernel | `6.16.1-0_fbk2_brcmrdma5_35_g5ba27bd1d6b9` | **`6.16.1-0_fbk5_npi_brcmrdma9_85_g5d3047c748bd`**; cmdline adds `csdlock_debug=1` and has no `mitigations=off` |
| amdgpu | `7.1.0.31300009`; package `amdgpu-7.1.0-2404163.el9` | `7.1.0.31300009` **el10 DKMS** (`7.1.0.31300009-2404163.el10`), srcversion `892370C5AB08A22771F1DC5`, MES `0x7e` |
| gfx1250 CWSR trap handler | not recorded | **Known-faulty `0f718b5e…` (5,656 B)**, extracted from the loaded module ([§7.2](#72-cwsr-handler-finding)) |
| `numa_balancing` | not recorded | 0 |
| Crash policy | not recorded | **`panic_on_oops=1`, `kernel.panic=0`, kdump.service failed (mkdumprd)**. Any oops halts the host with no vmcore and no auto-reboot. |
| Boot ID | `a76a97be-2fed-4e62-8f55-b5c1bd2a4aea` (completed run) | `93ec7c59…` (halted run), `78b316ae-dfc3-4fb7-8959-389789ae0439` (perf run) |

The 2026-09-19 CPU/OS/Docker inventory was captured read-only at **08:56:04
UTC**, after training, on the same boot. The nominal 256-thread topology does
not mean 256 CPUs are online.

The PCI, revision and architecture identifiers do **not** distinguish A0 from
B0 silicon. No stepping-based explanation is claimed without board or tray
records.

Driver load after every boot (both stacks):
`sudo modprobe amdgpu noretry=0 gpu_recovery=0 ip_block_mask=0xcff`, then
`sudo systemctl start docker` ([§5.1](#51-driver-initialization-and-failed-communication-attempts)).

### 2.2 Runtime, image and source pins

| Component | Version / source |
|---|---|
| Base image | `amdprimus/amdprimus:gfx1250-20260910` |
| Base image digest | `sha256:6e656de79e6c8d7f3b3db3690606c3e6cb0536ec871f0521735746015297c9a1` |
| Final local image | **`recommendation-gfx1250-20260910:triton-7ff97e`** |
| Final image ID | 2026-09-19: `sha256:1ca53982ee9c71beab4131e01d2ba0c8dfc2329933961ee79726b93b47450e3e`; 2026-09-26 rebuild: `sha256:fc5040f7a9e430aef05bf40a67fcd3375e4a6997d617f992484d4459816da0fe` |
| PyTorch | `2.11.0+rocm7.14.0a20260625`, Git `f55dda6ca78b73027840c1bc4014fe703d4f5473` |
| HIP / RCCL | `7.14.60850` / reported `(2, 30, 4)` |
| Triton | Runtime `3.8.0`; distribution `3.8.0+git7ff97e31`; exact clean source **`7ff97e310935b4a79794878dbc911f9af25d38d9`** |
| FBGEMM | Source `10b775730212923f65f7b78f79b6a01d80cf3c29`, the three gfx1250 patches from the repository recipe; installed distribution `2026.9.19` |
| TorchRec | `1.7.0a0+bf55480`, recipe tag `v2026.06.01.00` |
| Polars | `polars-u64-idx==1.33.1` |
| Docker client / server | **29.7.2 / 29.7.2**, supplemental 2026-09-19 snapshot |
| Tested training source | 2026-09-19: `082e3849dd9588ee5fbfd61eeae8c0754311d05b`; 2026-09-26: `d07045dfe3e39d5d744af5c748febd00742d243a` |

The native build follows
[`Dockerfile.amdprimus0815`](../../../Dockerfile.amdprimus0815). It uses the
September 10 base and keeps the built FBGEMM wheel, source and patches. Triton
was built from the exact pin and installed as a wheel with `--no-deps`.

On 2026-09-19:

- Native Torch/ROCm fingerprints matched the untouched base.
- CPU imports, FBGEMM cumsum and offline gfx1250 compilation passed before the
  GPU preflight.

The 2026-09-26 rebuild reports the same Torch, HIP, Triton and TorchRec
versions.

### 2.3 Container execution

All full runs used the preserved image and a new named container. Settings:

- **bridge networking** and **host IPC**;
- nonprivileged execution with `/dev/kfd` and `/dev/dri`;
- supplementary `video` group;
- `seccomp=unconfined` and `label=disable`.

The four ranks communicate over loopback inside their container.

| Host path | Container path |
|---|---|
| `/home/chcai/training/recommendation` | `/workspace/recommendation` |
| `/home/chcai/dlrm_data` | `/data/mlperf_dlrm_v4` |
| Per-run evidence directory | `/evidence` |
| Per-run `results/` | `/workspace/recommendation/results` |

The command requests `--shm-size=64g`, but `--ipc=host` means host shared memory
is used, so that option does not give a private 64 GiB shared-memory
allocation.

Trainer source is bind-mounted, so the retained image alone does not freeze
the source after a repository pull.

The image's allocator default is overridden: the launcher clears both
`PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF`, and explicitly selects:

```text
HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
```

## 3. Dataset and full model

### 3.1 Verified Yambda-5b cache

These experiments use the full model and full embedding vocabulary on four
GPUs. The A0/B0 single-GPU row-scale reduction was not carried over.

The dataset at `/home/chcai/dlrm_data`:

- passed the downloader MD5 checks;
- passed the repository's `verify_dataset.sh` SHA256 verification;
- had complete NPY payloads for all 14 required cache arrays before training.

### 3.2 Architecture and precision

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

Precision is mixed:

- Native TBE constructor records verify **FP32 embedding weights and FP32
  outputs**.
- The saved gin enables `make_model.bf16_training=True` for HSTU training.
  This does not mean every model parameter or operation is BF16.

The manifest preserves the complete model configuration.

### 3.3 Full embedding tables and actual sharding (2026-09-19, `HBM_CAP_GB=400`)

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

Eight actual TBE constructor records contain 20 shard records. An independent
reconstruction verifies:

- complete rectangular coverage without gaps or overlaps;
- FP32 precision;
- DEVICE placement.

Their total is exactly **600,169,906,176 bytes = 558.9518 GiB of weights**.
This is proof from actual allocations, not just an environment variable or a
planner claim. Optimizer state and activations need additional memory.

| Rank | Embedding weights, GiB | Sampled used VRAM near the end, GiB |
|---|---:|---:|
| 0 | 253.8173 | 422.89 |
| 1 | 124.1176 | 302.65 |
| 2 | 96.2524 | 408.58 |
| 3 | 84.7646 | 396.95 |

Used-VRAM values are samples, not continuous peak measurements.

The 2026-09-26 perf run used the gin default `hbm_cap_gb=260`. Its
per-rank shard table was not re-extracted, and its peak VRAM is in
[§1.3](#13-startup-and-memory).

Initial windows 0–5 have too few anchors for a global batch of 4,096, so the
first optimizer step occurs in window 6. This is expected batching behavior and
does not reduce the model or vocabulary.

## 4. 2026-09-19 3,000-step run: training settings

### 4.1 Model, batching, optimizer and observation

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

The saved gin selects dense Adam: betas 0.95/0.999, epsilon 1e-8, no weight
decay.

Native TBE records select `EXACT_ROWWISE_ADAGRAD`, with stochastic rounding
enabled and sparse `gradient_clipping=False`. The dense clipping setting
therefore does not mean the fused sparse update is clipped.

**All 3,000 observed steps are inside the 24,000-step warmup.** Stability at
the full base learning rate was not tested, and no holdout AUC was measured.

### 4.2 Socket transport and runtime controls

See [§5.2](#52-socket-transport-and-runtime-controls) for the environment. The
2026-09-26 runs used the same variable set, saved as `collective-socket.env`.

## 5. 2026-09-19 experiment record

### 5.1 Driver initialization and failed communication attempts

Plain `modprobe amdgpu gpu_recovery=0` failed VCN firmware initialization with
`LOAD_IP_FW` status **0x11** and ring-test timeout **-110**. A search of the
host found the existing `/home/yanyuqin/load_driver.sh`, whose command
succeeded:

```bash
sudo modprobe amdgpu noretry=0 gpu_recovery=0 ip_block_mask=0xcff
```

The mask disables **VCN/JPEG** and keeps compute and UALink. It is a
host-specific driver-load workaround. It is not a numerical fix, and it has
not been shown to recover an MES scheduler that is already stuck. No driver
reload was performed during training.

| Attempt | Observed result | Interpretation |
|---|---|---|
| Initial base-image preflight | All four GPUs pass FP32/BF16 matmul, Triton add and 1,000 trials each of D4/D16 `index_select`/`gather`; RCCL initialization stalls. Stopping it is followed by GPU0 MES queue/TLB errors. | Individual kernels pass; distributed communication does not. The small Triton-add arm used stock Triton 3.6. |
| Collective overrides on the same failed boot | No valid collective pass. | Inconclusive for those overrides because the GPU state was already unhealthy. |
| Targeted GPU0 AMD-SMI reset | Enters mode-2 recovery, then kernel soft-lockup/RCU-stall symptoms and uninterruptible tasks. | Reset worsened the observed state; no successful local MES recovery was demonstrated. |
| Fresh preflight after the user's power cycle, `preflight_recovery_20260919T034707Z` | Final pinned image; individual checks pass on all four GPUs and RCCL initializes, but the first 1,024-element all-reduce times out after 90 seconds. All ranks report sequence 1 enqueued and -1 completed. GPU3 develops MES failures at about **03:47:16 UTC**. Container exits **137**. | Fresh reproduction of a communication/MES failure; larger collectives and all-to-all are not reached. No training NaN was observed. |
| After the next reboot | Same driver-load command succeeds; Socket preflight passes. | First validated four-GPU communication configuration for this investigation. |

The persistent errors include `REMOVE_QUEUE`, `ADD_QUEUE`, `INVALIDATE_TLBS`,
`MES might be in unrecoverable state`, and repeated ring-full messages. The
fresh GPU3 failure starts with queue-removal errors. Do not substitute the
TLB-first ordering seen on another host.

The recovery review found no documented procedure, demonstrated locally, that
clears this state in-band. Reset interfaces existed, but that alone did not
make them a working procedure.

The user performed the requested host recoveries. This investigation did not
change shared-fabric services, and it did not repeat the harmful in-band reset
after the later failure. This does not prove that every possible reset method
or warm reboot must fail.

### 5.2 Socket transport and runtime controls

The principal communication and runtime overrides are:

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
`environment.txt`.

On 2026-09-19 the source file was named `collective-socket-proposed.env` and
kept its historical pre-test comments. The two saved preflight results
establish its later validation, so do not infer from the name that it is
untested.

Each passing preflight and the completed training run has **88 actual
`NET/Socket` channel records**: 64 `NET/Socket/0` and 24
`NET/Socket/0/Shared` in training. `/Shared` denotes shared NET buffers, not
SHM transport.

Checking only the banner `Using network Socket` is not enough. An earlier
failed attempt printed that banner while its channels still used P2P/IPC.

This workaround changes communication paths and timing. It keeps all four GPUs
and all embedding rows. It does not repair the underlying fabric
configuration, and it does not isolate which environment variable is
responsible.

### 5.3 Successful four-GPU preflights

Two fresh preflights passed with the final image and the same saved environment:

| Preflight directory | Boot | Result |
|---|---|---|
| `preflight_collective-socket-proposed_20260919T045707Z` | `2f0619db-6143-4474-bcb9-0b8a6b545be0` | **16/16 check groups on each of four ranks**, clean group destruction, Docker/tee **0/0**, no new MES fault. |
| `preflight_collective-socket-proposed_20260919T061801Z` | `a76a97be-2fed-4e62-8f55-b5c1bd2a4aea` | Same complete pass before the cold restart; Docker/tee **0/0**. |

The checks cover:

- exact FP32/BF16 matmul;
- pinned-Triton addition;
- 1,000 trials each of D4/D16 gather and index-select;
- two NCCL groups;
- all-reduce, broadcast, all-gather and reduce-scatter;
- FP16 all-to-all with unequal, zero and reversed splits;
- larger FP32/FP16/BF16 operations, including **16 MiB per rank FP16/BF16
  all-to-all**.

The numerical comparisons use independent CPU expectations.

### 5.4 Full-model run ledger

| Run | Start, UTC | End / observed progress | Verdict |
|---|---|---|---|
| `full4_start0_20260919T045742Z` | **04:57:42** | Monitor records **1,530 finite steps**; raw loss lines survive through **1,525**; a new boot begins around **06:12**. | **Interrupted**, no recorded NaN, 3,000-step target not reached, interruption cause unknown. |
| `full4_start0_20260919T062014Z` | **06:20:14** | Planned stop at about **08:09:28**, **3,000 finite steps**, final loss **0.13069**. | **Completed bounded negative NaN result**, independently verified. |

### 5.5 First full run: interrupted

The first run used the full tables and a successful Socket preflight on the
same boot.

- The saved observer records consecutive finite steps **1–1,530** and
  `first_nonfinite=null`.
- Raw training and launcher logs survive through step **1,525** (loss
  **0.13790**). Docker's raw log survives through step **1,522**.
- Short NUL tails and an unclean, corrupted previous journal show that
  persistence was interrupted.

Nothing retained explains the interruption: no GPU/MES fault, OOM, panic,
orderly host shutdown, or reliable BMC restart cause. Docker's exit **255**
and its finish time of about **06:12:13** were assigned when the daemon
restored state on the new boot. They do not identify the original failure
time or cause. The absence of a persisted fault does not rule out one that
was never persisted.

The 2026-09-26 halt ([§7](#7-2026-09-26-host-halt-at-local-2304--gbs-9216)) shows
that this host policy turns any oops into a silent halt, and the journal of
that halt also ends without a stack trace. Whether the 2026-09-19 policy was
the same is not recorded.

Checkpointing was disabled, so no model, optimizer or RNG checkpoint existed
to resume from. The later run started again from zero. Its core model fields
and all eight TBE constructor argument records match the first run. Printed
losses need not match exactly, since bitwise determinism was not established.

### 5.6 Cold restart and evidence persistence

The completed run kept the same model, image and collective settings. Only the
host observer and outcome handling were strengthened:

- `train.log` is synced before monitor state is persisted atomically: every
  ten losses or five seconds, on a nonfinite loss, and at completion.
- Persistence errors stay visible and prevent a finite-pass classification.
- Write failures cannot skip the nonfinite stop action.

No diagnostic GPU kernels or model changes were added.

## 6. 2026-09-19 results and comparison

### 6.1 Completed run: 3,000 finite losses

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
Both loss streams, the stop marker, the observer state and the final statuses
agree.

The live kernel collector covered **06:38:25–08:09:29 UTC**, supplementing the
initial and final snapshots, and stopped normally after `launcher.finished`.
Afterwards:

- training and monitor sessions had ended;
- no training containers were still running;
- a later sysfs sample showed about **0.162 GiB used VRAM per GPU**.

This release of memory is a process-cleanup observation, not an additional GPU
health test.

### 6.2 Relation to A0 and B0

The recent A0/B0 NaN baselines and this host share the principal Torch, HIP,
exact Triton source, FBGEMM/TorchRec recipe, and HSTU compiler workarounds.
Matching those pins does not make the independently built images or the
execution trajectories identical.

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
reproducer:

- `BLOCK_N=8` fails at iteration **115**.
- `BLOCK_N=1` passes **1,000/1,000**.
- Restoring 8 fails at **212**.

However, BN1 plus attention cap256 still fails training at sampled steps
**60/70**. The step-51 output-gradient GEMM already receives a huge finite
`dout`, and its producer is unconfirmed.

B0's later **07:11 UTC** D16 gather/index-select health failure is another
observation. There is no saved failing-tensor proof and no established link to
training. See the [B0 continuation](../../mi450_b0/nan_triage_resume_20260919.md).

These later diagnostics and `WEIGHTED_LN_BWD_BLOCK_N` were not part of this
host's tested `082e384` source.

The supported conclusion is **no nonfinite logged training loss within this
3,000-step, start-zero, uncapped full-model run under the recorded
configuration**. It does not:

- identify the root cause of earlier failures;
- rule out a later failure;
- validate every activation, gradient or updated embedding row;
- establish benchmark convergence (no evaluation was run).

A0's full-state replay path is single-rank and was not enabled on this
four-rank run.

Three separate questions remain open:

- the fabric provisioning mismatch and the direct communication failure;
- the cause of the interrupted first run;
- whether the A0/B0 component defects reproduce on this host under controlled
  inputs.

Obtaining finite full-model losses resolved none of them.

## 7. 2026-09-26 host halt at local 2,304 / GBS 9,216

### 7.1 What happened

The run `perf_gbs9216_s200_20260926T204346Z` (boot `93ec7c59…`) used the same
recipe as [§1](#1-200-step-full-model-perf-run-2026-09-26) except for batch
**2,304 per rank / 9,216 global**.

| Time, UTC | Event |
|---|---|
| 20:43:46 | Launch |
| ≈ 20:52 / 20:56:32 | Training loop starts / first all-to-all |
| 20:57 → 21:06:04 | VRAM flat at **425.0 / 415.5 / 368.8 / 411.3 GiB** (GPU0 at 98.4% of HBM); last sampler line at 21:06:04 |
| **21:06:13.93** | `amdgpu 0003:04:00.0` (rank 2, busId 304000): `[gfxhub0] no-retry page fault`, ring 88, vmid 2, trainer pid, at GPU VA **`0x0101010101010000`**. The address is non-canonical and made of 0x01 bytes, i.e. a garbage pointer. |
| same ms | `BUG: unable to handle page fault for address: ffd404f45702ce88` (supervisor read, not-present): a kernel oops in the amdgpu fault-handling path. The journal ends here, with no stack trace persisted. |
| → 21:31 | Host halted (`panic_on_oops=1`, `kernel.panic=0`, kdump failing). The user power-cycled it and the new boot started at about 21:31. |

The run left no training-step output in its log before the halt, and there is
no trace.

This session performed no privileged GPU-state operations: no driver reload,
reset, `amd-smi` call or fabric change. Other users' no-retry faults on the
same boot (09-25 23:15, 09-26 02:48/02:50) were at valid user VAs and **did
not** oops the kernel.

### 7.2 CWSR handler finding

The gfx1250 CWSR trap handler embedded in the loaded amdgpu module was
extracted with
[`mi450_rck_spur/tools/cwsr_fix/extract_cwsr.py`](../../mi450_rck_spur/tools/cwsr_fix/).

- It hashes to the **known-faulty `0f718b5e…` (5,656 B)**; the corrected
  handler is `68c31ab2…`.
- The DKMS assembly contains the unpatched `L_NOT_WAVE_START` `v_readlane` /
  `v_writelane v1` sequence.

According to the [RCK spur doc](../../mi450_rck_spur/mi450_rck_spur.md), this
handler can copy lane-0 VGPRs across banks during save/restore and corrupt a
load address. `numa_balancing=0` here, so the AutoNUMA trigger identified in
that doc is not active. Queue evictions still happen through kernel memory
compaction. On 2026-09-27 they coincided with GPU faults at `BLOCK_N=128`
([§8](#8-2026-09-27-nan--fault-repro-r1-faults-and-cwsr-exposure)).

### 7.3 Hypothesis and comparison

**Hypothesis, not proven:**

1. Near-full HBM triggered TTM/KFD evictions.
2. The evictions forced CWSR save/restore through the faulty handler, which
   corrupted an address register. This is consistent with the `0x0101…`
   address.
3. The driver's fault path then oopsed on the non-canonical address.

It is not established whether the new kernel and el10 driver build are what
turn a GPU fault into a host oops.

| Item | Reference successes (a37-2 converged; a37-1 3,000 steps, 2026-09-19/20) | Halted run |
|---|---|---|
| Local / global batch | 1,024 / 4,096 | **2,304 / 9,216** |
| Peak sampled VRAM | 391–406 GiB (a37-2), ≤ 423 GiB (a37-1) | **425.0 / 415.5 / 368.8 / 411.3 GiB** |
| Kernel / amdgpu | fbk2_brcmrdma5 / el9 package | **fbk5_npi_brcmrdma9 / el10 DKMS** |
| CWSR handler | not recorded | **faulty `0f718b5e…`** |
| Oops policy | not recorded | halt, no vmcore |

a37-2 had already failed local 2,048 in 5/5 attempts: OOM plus an RCCL memory
fault, a scratch-reclaim stall, or MES `INVALIDATE_TLBS` errors (see the
[a37-2 doc](../ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md)). Local 2,304 is
beyond that envelope.

After the power cycle, local 1,024 ran 200 steps on the same stack with no
kernel fault ([§1](#1-200-step-full-model-perf-run-2026-09-26)).

**Operating guidance for this host:**

- Stay at local 1,024 until the CWSR handler is fixed.
- Until then, run `BLOCK_N=128` only on a freshly booted host
  ([§8.4](#84-guidance)).
- Keep a live `dmesg -w` capture running, because the journal can lose the
  final lines.
- Ask the host owner to fix kdump, or set `kernel.panic>0`, so an oops reboots
  the host and leaves evidence.

## 8. 2026-09-27 NaN / fault repro: R1 faults and CWSR exposure

The question was why this host had finished runs without NaN when the A0
record ([CWSR bank-corruption fix](../../mi450_a0/cwsr_bank_corruption_fix.md))
says the driver handler fix is needed for a NaN-free end-to-end run. The
answer is that this host is **not** immune. It has lower exposure: the
defect is present, and it did cause failures once the host's memory state
changed.

### 8.1 Run ledger

Common setup for every row:

- 4 GPUs, local batch 1,024 / GBS 4,096, full model;
- uncapped attention (`HSTU_BWD_MAX_VGPR=0`), `AMDGCN_USE_BUFFER_OPS=0`
  unless noted, both allocator variables empty;
- image `recommendation-gfx1250-20260910:triton-7ff97e`, the same code.

Uptime is measured from boot `78b316ae…` at about 21:31 on 09-26, and from boot
`de975fa0…` at 06:20:50 on 09-27. Run folders are under
`/home/chcai/mi450_perf_20260926/`.

| Run folder | Boot, uptime at launch | Differences | Result |
|---|---|---|---|
| `perf_gbs4096_s200_20260926T214509Z` | `78b316ae`, 0.2 h | `BLOCK_N=64` (default), `OUTPUT_TRACE=1` | 200 steps, finite, 0.13895 at step 200 |
| `bn128_gbs4096_s600_20260927T034620Z` | `78b316ae`, 6.3 h | **R1** `BLOCK_N=128`, `OUTPUT_TRACE=1` | **600 steps**, finite, 0.13791 at step 600 |
| `rv_pipe_gbs4096_s600_20260927T043037Z` | `78b316ae`, 7.0 h | R1 + `TRITON_ALLOW_PIPELINING=1` | **600 steps**, finite, 0.13791 |
| `rv_bufops_gbs4096_s600_20260927T050103Z` | `78b316ae`, 7.5 h | R1 + buffer ops on | MES hang on GPUs 0/1 before any step was logged; reboot (new boot `de975fa0`) |
| `rv_alloc_gbs4096_s600_20260927T064634Z` | `de975fa0`, 0.4 h | R1 + `expandable_segments:True` | OOM at start-up (64.85 GiB allocation); no training |
| `gemmcap_gbs4096_s12_20260927T070447Z` | `de975fa0`, 0.7 h | R1 + hipBLASLt log capture | 12 steps, finite |
| `evict_gbs4096_s300_20260927T211855Z` | `de975fa0`, **15.0 h** | R1, `OUTPUT_TRACE=0`, eviction sampler | **Fault** at 21:31:56 on rank 0, before the step-50 log line: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. ROCr did not name the kernel; the dispatch was 4,096 × 128-thread workgroups with 32 KiB LDS. |
| `evtrace_gbs4096_s300_20260927T213717Z` | `de975fa0`, **15.3 h** | as above + kprobe eviction trace | Step 100 finite (0.13918). **Fault** at about 21:52:15–17: GPU indices 2, 3 and 1, all within 0.2 s, each "Page not present" in `_hstu_attn_bwd` at a wild address. |
| `hoststate_gbs4096_s600_20260927T222021Z` | `de975fa0`, **16.0 h** | **exact `bn128` config** (`OUTPUT_TRACE=1`) + eviction/compaction sampler | Step 100 finite (0.13918). **Fault** at 22:35:37: GPU index 1, "Page not present" in `_hstu_attn_bwd` at `0x7e0c34443000`. |

**No NaN appeared.** All three R1 runs that failed had logged exactly the
same losses as the passing R1 runs up to that point: **0.13883 at step 50** and
**0.13918 at step 100**. The kernel log shows no oops in any of them, and the
host recovered without a reboot.

The `hoststate` run rules out `OUTPUT_TRACE=0` as the cause. With the image,
environment and code identical, the passing R1 config faulted at 16 h
uptime.

### 8.2 What evicts the GPU queues on this host

A0 and RCK found the following trigger:

1. AutoNUMA page-table scans with XNACK off.
2. `svm_range_evict` then quiesces the KFD queues.
3. The CWSR save/restore runs through the faulty handler.

This host lacks both parts of that trigger:

- `numa_balancing=0`: `numa_pte_updates` stayed at **0** throughout.
- amdgpu `noretry=0`, so `kfd_process_xnack_mode` returns true for GC 12.1.0.
  The process reports **`xnack_enabled=1`**, queried with
  `AMDKFD_IOC_SET_XNACK_MODE(-1)` in `xnack_probe/probe.py`.
- With XNACK on, `svm_range_evict` (`kfd_svm.c`) **unmaps** SVM ranges
  instead of quiescing the queues. `GPU_ALWAYS_MAPPED` ranges are the
  exception.

Two paths still quiesce the queues:

- userptr invalidation (`amdgpu_amdkfd_evict_userptr`), which does not
  check XNACK;
- always-mapped SVM ranges.

Both fire when kernel memory compaction migrates the pages behind those
buffers.

Kprobes on `kgd2kfd_quiesce_mm` and `kfd_process_evict_queues`, with stack
traces, recorded the following for the `evtrace` run
(`kfd_evict_trace.txt`, 27,060 events, 0 lost):

| Measure | Value |
|---|---|
| Queue quiesces | **1,580**: userptr 943, SVM 637, TTM 0 |
| Initiator | **memory compaction 1,569** (direct `compact_zone` 1,138, `kcompactd` 431); `svm_migrate_vram_to_ram` 7; `exit_mmap` 4 |
| Before the fault | 490 quiesces in about 5.6 min after the first eviction |
| KFD `evicted_ms` per rank at the fault | about 800–1,050 ms (`evict`: 600–690 ms within about 3 min) |
| Timing | The largest burst (76 quiesces in one second) coincides with the faults within clock uncertainty. Which came first is not resolved: the GPU core dump could itself drive compaction. |

The `hoststate` sampler lined up compaction stalls, evictions and the fault
(`evicted_ms` shown for GPU `30548` on two ranks):

| UTC | `compact_stall` | `evicted_ms`, rank 1 / rank 2 |
|---|---|---|
| 22:29:49 (warm-up) | 158,500 | 0 / 0 |
| 22:30:04 | 164,350 | 95 / 164 |
| 22:34:51 | 166,418 | 418 / 467 |
| **22:35:36** | **168,144** | 589 / 662; fault logged at 22:35:37 |

Host memory at about 22:00 on 09-27:

- 250 GiB RAM, 96 GiB of it page cache;
- 5 of 7 GiB swap in use (`pswpout` 174,223);
- transparent huge pages set to `madvise`.

Compaction stalls climb in bursts. This memory state develops with uptime;
eviction counters were not sampled during the earlier passing runs.

### 8.3 Interpretation

- **Consistent with the A0 CWSR mechanism.** A queue eviction saves and
  restores the running `_hstu_attn_bwd` waves through the faulty handler. At
  `BLOCK_N=128` those waves hold 986 VGPRs, including extended ones. At
  `L_NOT_WAVE_START` the handler copies lane 0 across VGPR banks, corrupting
  an address register, and the kernel then faults at a wild address.
- **Not proven.** Three things are missing:
  - no control run with the corrected handler;
  - no eviction measurements for the passing runs;
  - an unexplained difference in fault attribution. For the `hoststate` and
    `evtrace` faults, the kernel log names `SDMA0` as the UTCL2 client of the
    latched fault, and one `evtrace` fault also lists `TCP`. ROCr names
    `_hstu_attn_bwd` in both runs.
- **Why no NaN so far is not explained.** A0's NaN came from hipBLASLt
  solution 103 (MT256x128x64, 676 VGPRs) in the 256 × 512 × T preprocessor
  wgrad. That kernel runs in every config on this host
  ([GEMM map](hipBLASLt/dlrmv4_gemm_component_map.md)).
  - One hypothesis: compaction evicts in bursts, while AutoNUMA evicts at a
    steady rate.
  - This is untested.
- **The finite-loss history here reflects lower exposure, not immunity.**
  - With AutoNUMA off and XNACK on, the main A0/RCK trigger is gone.
  - The long clean runs were at `BLOCK_N=64`: 3,000 steps on 09-19 on the
    old stack, and 200 steps on 09-26.
  - `BLOCK_N=64` has not been run in the high-compaction state.

### 8.4 Guidance

- **Install the corrected CWSR handler (`68c31ab2…`) before any long or
  convergence run.** The A0 conclusion applies to this host.
- Until then, choose one of the following for R1 (`BLOCK_N=128`) runs:
  - run them only on a freshly booted host;
  - reduce compaction first. Ask the host owner, for example, about
    `vm.compaction_proactiveness=0` and THP defrag off, or run
    `echo 1 > /proc/sys/vm/compact_memory` before launch. This lowers
    exposure; it does not fix the defect.
  - fall back to `BLOCK_N=64`.
- Sample `evicted_ms` in every run (`evict_probe.sh` pattern), so the
  exposure is recorded with the result.
- The R1 performance result in
  [performance_analysis.md](performance_analysis.md) stands. Its
  numerics match `BLOCK_N=64`, but its stability depends on the handler fix.

### 8.5 Readiness for a driver-team report

Assessed on 2026-09-28 against the A0 and RCK records.

**Is the handler defect proven? Yes, at component level.** The evidence is in
[cwsr_bank_corruption_fix.md](../../mi450_a0/cwsr_bank_corruption_fix.md):

- **Mechanism.** At `L_NOT_WAVE_START`, the handler runs
  `v_readlane`/`v_writelane v1` before it resets the MODE VGPR-bank bits. With
  unequal banks, lane 0 is copied between physical registers: v257→v1,
  v257→v513 or v513→v257.
- **Instruction-level matrix.** It covers all 256 MODE bank bytes × 8 EXEC
  masks, 6,144 calls in total. The original prologue produces exactly the
  1,536 predicted copies. The corrected prologue and a no-prologue control
  preserve every tag.
- **Byte-exact replays.** Controlled traps reproduce four historical failures
  byte for byte, and the equal-bank and NOP controls stay clean:
  - the step-405 DW gradient: all 1,030 differences, including 640 NaNs;
  - attention FAIL131;
  - the LayerNorm NaN rows;
  - projection DX.
- **Trigger.** The traced chain is AutoNUMA →
  `svm_range_cpu_invalidate_pagetables` → `kgd2kfd_quiesce_mm`. A
  process-policy counterfactual switches the copies off and on.

**Is the fix proven? Yes at handler level; end to end is thinner.**

- **Handler level.**
  - The corrected handler `68c31ab2…` saves and clears the DST/SRC0/SRC1 bank
    bits with `S_SETREG_B32` before touching v1, then restores them.
  - It passes every replay above, the 400 long-plateau probes and 1,000
    natural DW calls with eviction exposure.
- **End to end.**
  - RCK has the cleanest A/B. The corrected driver ran past 2,000 steps with
    AutoNUMA active, while the stock driver went NaN at steps 111, 123 and 354
    ([RCK](../../mi450_rck_spur/mi450_rck_spur.md#current-status-corrected-driver-clean-past-2000-steps-with-the-trigger-active)).
    That is one run, and its final outcome is not yet recorded.
  - A0's corrected-driver 1,000-step pass had other driver-source, boot and
    prewarm changes at the same time.

**This host's role: supporting evidence, not proof.**

- It adds a third eviction source: memory compaction, with XNACK on
  ([§8.2](#82-what-evicts-the-gpu-queues-on-this-host)).
- Its failures are page faults, not NaN, and the kernel log's `SDMA0`
  attribution is unexplained ([§8.3](#83-interpretation)).
- There is no corrected-handler control run here yet.

**Gaps to close alongside the report:**

1. **A portable minimal reproducer.**
   - A single kernel tags v1/v257/v513 with unequal banks and spins while a
     queue eviction is forced, then checks the tags.
   - Expected result: tags copied on `0f718b5e…`, preserved on `68c31ab2…`.
   - The A0 matrix used SGPR stand-ins and no privileged trap entry. The
     plateau probes rely on AutoNUMA with a low hit rate. The retained A0
     packages are hash-pinned to session paths and must be re-derived before
     reuse.
2. **A source-level fix in the driver team's build flow.**
   - `68c31ab2…` was built from disassembly-derived LLVM assembly.
   - Nobody has tested whether the SP3 assembler accepts
     [the patch](../../mi450_a0/cwsr_gfx1250_bank_fix.patch).
   - An audit of all handler paths is still needed. It should cover every
     VGPR access made while the inherited bank bits are live, and whether
     `s_setreg` of MODE needs a delay before the next VGPR access.
3. **The final RCK corrected-run result**, ideally with a second
   corrected-driver run.
4. **A corrected-driver comparison on this host** ([§8.6](#86-corrected-driver-comparison-on-this-host-plan-not-run)).

Suggested filing: a confirmed handler defect plus a validated fix candidate.
Send the mechanism, the instruction-level matrix, the byte-exact replay A/B,
the trigger stacks and the RCK A/B, and list the gaps above.

**Owner:** the amdgpu KFD (compute kernel driver) team. The fix changes
`amd/amdkfd/cwsr_trap_handler_gfx12.asm` and the `cwsr_trap_gfx12_1_0_hex`
array regenerated from it. Two other teams should review it:

- the gfx1250 shader (SQ) architecture team, for the MODE bank-bit semantics
  and any `s_setreg` hazard;
- the ROCm debugger team, which co-maintains the trap handler.

**To-dos on this host before filing (none started as of 2026-09-28):**

1. **Build the corrected module.**
   - Use the 7.1.0 DKMS source with the `68c31ab2…` handler.
   - Adapt the RCK tool paths.
   - Audit against a baseline rebuild: only the handler array should differ.
2. **Load it safely.**
   - Do a first load after a power cycle with
     `noretry=0 gpu_recovery=0 ip_block_mask=0xcff`, not a live reload.
   - Have someone ready to power-cycle the host by hand.
3. **Run the comparison.**
   - Run `BLOCK_N=128`, local 1,024, 600 steps, several times on the
     corrected driver at 15 h or more of uptime.
   - Run one stock control at a similar host state.
   - Log `compact_stall` and `evicted_ms` in every run.
4. **Build a minimal reproducer.**
   - Use a single kernel with unequal VGPR banks and tagged v1/v257/v513, and
     force a queue eviction while it runs.
   - Tags should be copied on the stock handler and preserved on the
     corrected one.
5. **Explain the `SDMA0` attribution**, or state it as an open question in
   the ticket.
6. **Package the ticket.** Include:
   - the driver identity: version and srcversion, handler hash, and the
     faulty-instruction offsets (68 and 96);
   - the run ledger and eviction traces;
   - the corrected-vs-stock results;
   - a source patch against the 7.1.0 tree.

### 8.6 Corrected-driver comparison on this host (plan, not run)

**Status: not run.** No driver has been rebuilt, loaded or unloaded on this
host for this comparison. It needs someone who can power-cycle a37-1 by hand
if it goes wrong.

**What is already in place (checked 2026-09-28):**

- The loaded module is `/lib/modules/6.16.1-0_fbk5_npi_brcmrdma9_85_g5d3047c748bd/extra/amdgpu.ko`:
  - version `7.1.0.31300009`, srcversion `892370C5AB08A22771F1DC5`;
  - the loaded srcversion matches `modinfo`.
- Its `cwsr_trap_gfx12_1_0_hex` is the faulty 5,656-byte `0f718b5e…` image,
  with `v_readlane_b32 ttmp15, v1, 0` at byte 68 and
  `v_writelane_b32 v1, ttmp15, 0` at byte 96.
- The DKMS source `/usr/src/amdgpu-7.1.0-2404163.el10` is installed, and its
  `cwsr_trap_handler_gfx12.asm` still has the unpatched `L_NOT_WAVE_START`
  sequence. Kernel build headers are present.

**Build.**

- The RCK tools in
  [`mi450_rck_spur/tools/cwsr_fix/`](../../mi450_rck_spur/tools/cwsr_fix/)
  build the corrected module by swapping in the `68c31ab2…` handler.
- They refuse to write anything unless the input handler is `0f718b5e…` and
  the output hashes to `68c31ab2…`.
- They are written for 7.1.1. Two paths must change for this host: the source
  tree becomes `/usr/src/amdgpu-7.1.0-2404163.el10`, and the installed module
  becomes `extra/amdgpu.ko` instead of `updates/dkms/amdgpu.ko.zst`.
- Audit the result against a baseline rebuild. The only differences should be
  the handler array and the layout shifts that follow from it.

**How to load it:**

- **First load after a power cycle (preferred).**
  - This host already loads amdgpu by hand after boot
    ([§7](#7-2026-09-26-host-halt-at-local-2304--gbs-9216)).
  - Load the corrected `.ko` at that point with the same parameters:
    `noretry=0 gpu_recovery=0 ip_block_mask=0xcff`.
  - This avoids unloading a live driver. A0's one live reload ended in an
    unexplained reboot with BERT errors.
- **Live reload (riskier).**
  - It keeps the current high-compaction host state.
  - If it oopses, the host stays down until someone power-cycles it.

**Test design.**

- **The catch.** The stock driver faulted only at 15–16 h uptime (3 of 3 runs)
  and passed at 6–7 h. So after a fresh boot, a clean corrected run counts only
  once the host has reached comparable compaction.
- **Record exposure in every run** with the `hoststate_probe.sh` sampler:
  `compact_stall` and KFD `evicted_ms`.
- **Corrected runs.** Use the same `BLOCK_N=128`, local-1,024, 600-step
  config, several times at high uptime.
- **Stock control.** Run one stock run at a similar state to confirm the
  trigger is still present.
- **One clean run is weak evidence.** The stock driver also passed at lower
  exposure.

**What it would show.**

- If the corrected driver runs clean where the stock driver faults, the
  page faults on this host are tied to the handler, on a second host with a
  different trigger.
- It would not explain the `SDMA0` attribution.

## 9. Reproduction and analysis commands

### 9.1 2026-09-26 perf run

The host-local kit is `/home/chcai/mi450_perf_20260926/`. After the driver-load
command in [§2.1](#21-host-cpu-gpus-and-firmware):

```bash
cd /home/chcai/mi450_perf_20260926
# 4-GPU preflight: matmul + collectives incl. 16 MiB all-to-all, CPU-checked
docker run --rm --network=bridge --ipc=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined --security-opt label=disable \
  --env-file collective-socket.env -v "$PWD":/kit \
  recommendation-gfx1250-20260910:triton-7ff97e python /kit/preflight.py

BATCH_SIZE=1024 STEPS=200 nohup ./run_perf_4gpu.sh > launcher.log 2>&1 &
```

`run_perf_4gpu.sh` records the following:

- repo commit and status, image ID and boot ID;
- `dmesg` before and after;
- a live `dmesg -w`;
- a 5 s VRAM sampler.

It then runs the trainer with `OUTPUT_TRACE=1` and `DIE_AT_STEP`. Traces land
in `recommendation/results/<RUN_NAME>/`. Stitch them with the command in
[§1.4](#14-trace-capture).

The launcher's default `BATCH_SIZE` is 2,304, the value that halted the host.
**Always pass `BATCH_SIZE=1024`.**

### 9.2 Recheck the 2026-09-19 evidence without GPU work

```bash
python3 /home/chcai/mi450_fullmodel_20260919/verify_full_run.py \
  /home/chcai/mi450_fullmodel_20260919/full4_start0_20260919T062014Z
```

The verifier reads saved files and prints JSON:

- exit **0**: finite logged losses verified;
- exit **1**: evidence incomplete or inconsistent;
- exit **2**: a nonfinite logged loss was found.

It checks model and shard coverage, both loss streams, completion, transport,
final synced-byte coverage and kernel evidence. The saved result is that run's
`final-verification.json`.

**The 2026-09-19 root is no longer on the host**
([§11](#11-evidence-inventory-and-limits)), so this command and §9.3 are
historical records.

### 9.3 Recorded 2026-09-19 preflight and launch procedure

```bash
bash /home/chcai/mi450_fullmodel_20260919/retry_preflight_after_recovery.sh \
  /home/chcai/mi450_fullmodel_20260919/collective-socket-proposed.env

# Use the newly completed, inspected passing preflight directory.
PREFLIGHT_DIR='/home/chcai/mi450_fullmodel_20260919/<successful-preflight-directory>' \
  bash /home/chcai/mi450_fullmodel_20260919/run_full_4gpu.sh
```

The launch guard requires:

- the same boot, image and unchanged saved collective environment as that
  preflight;
- all four ranks passing;
- successful Docker and tee statuses.

It checks the dataset cache and full table dimensions, and records source
hashes and the effective configuration. It then stops at the first reported
nonfinite loss or at 3,000 steps. The actual channel logs must confirm
NET/Socket.

To repeat the recorded source exactly, use an isolated checkout at `082e384`
and a launcher whose `ROOT` points to it.

## 10. Fabric registration and recovery limits

The disagreement was captured on multiple boots. One capture is a
before/after provisioning sequence on boot
`2f0619db-6143-4474-bcb9-0b8a6b545be0`, taken before the GPU preflight:

| Time, September 19 UTC | Captured state |
|---|---|
| **04:56:11**, after driver load | Driver and AFM agree on accelerator IDs **3,2,1,0**, phase **ACTIVE**. |
| **04:56:30–31**, node-agent provisioning | Agent tries IDs **7,6,5,4**; all four PPOD commits fail with `Invalid argument` and `UAL_SET_PPOD_CONFIG` status **0xFFFF0007**. |
| **04:56:44**, after failed provisioning | AFM reports IDs **7,6,5,4**, phase **PROVIDER**; driver retains IDs **3,2,1,0**, state **ACTIVE**. |

The active node-agent uses slot 1 / controller `10.5.219.30`. A slot-0 backup
is historical configuration, not authority for changing the live service.

The controller object also reports alternating host and SONiC/Ryzen
identities, whose meaning is unverified. This investigation changed no slot,
controller, fabric service or shared configuration.

This establishes a provisioning failure independently of any training
workload. It does **not** establish that:

- the mismatch causes the collective hang;
- a particular disabled path is what makes Socket work;
- the mismatch is connected to the A0/B0 NaN.

The fabric state was not rechecked after the 2026-09-24 reprovisioning. See
`FABRIC_CONFIGURATION_FINDINGS.md` and `FABRIC_POSTREBOOT_2f0619db.md`, which
are 2026-09-19 artifacts.

The persistent-MES recovery observations and the unsuccessful in-band reset
are recorded in [§5.1](#51-driver-initialization-and-failed-communication-attempts).
A successful Socket run establishes a usable workload configuration. It does
not show that the fabric mismatch is repaired, or that an already-wedged
scheduler can recover without a host reboot.

## 11. Evidence inventory and limits

### 11.1 2026-09-26 (host-local root `/home/chcai/mi450_perf_20260926/`)

| Artifact | Purpose |
|---|---|
| `run_perf_4gpu.sh`, `collective-socket.env`, `preflight.py` | Launcher, Socket env, 4-GPU preflight |
| `dataset-verification.log` | `verify_dataset.sh` pass on the freshly downloaded dataset, 2026-09-26 |
| `preflight/` | Passing preflight on boot `93ec7c59…`, 20:41, before the halted run |
| `preflight_postreboot/` | Passing preflight on boot `78b316ae…` |
| `perf_gbs4096_s200_20260926T214509Z/` | Perf run: `train.log`, `mlperf.log`, `vram.log`, `environment.txt`, `collective.env`, `dmesg.{before,live,after}.txt`, `repo-commit.txt`, `repo-status.txt`, `image-id.txt`, `boot-id.txt`, `exit-status.txt` |
| `perf_gbs9216_s200_20260926T204346Z/` | Halted run: logs, `vram.log`, `RETRO.md`, `kernel.prevboot.tail.txt`, `kernel.prevboot.gpu-faults.txt`, `cwsr_extract/` (handler extraction output) |
| `run_perf_4gpu_bn128.sh`, `run_revert_e2e.sh` | R1 and revert-check launchers (2026-09-27) |
| `bn128_…T034620Z/`, `rv_pipe_…T043037Z/`, `rv_bufops_…T050103Z/`, `rv_alloc_…T064634Z/`, `gemmcap_…T070447Z/` | 2026-09-27 R1 and revert-check runs ([§8.1](#81-run-ledger)); same file set as the perf run |
| `xnack_probe/probe.py` | KFD XNACK-mode query (`AMDKFD_IOC_SET_XNACK_MODE(-1)`) |
| `evict_probe.sh` → `evict_gbs4096_s300_20260927T211855Z/` | R1 run with 5 s KFD `evicted_ms` sampler (`kfd_evicted.log`) |
| `evict_trace.sh` → `evtrace_gbs4096_s300_20260927T213717Z/` | Same, plus kprobe/stacktrace eviction trace (`kfd_evict_trace.txt`, 19.8 MB) |
| `hoststate_probe.sh` → `hoststate_gbs4096_s600_20260927T222021Z/` | Exact `bn128` config with eviction + compaction/swap sampler, `vmstat.{before,after}.txt`, `meminfo.before.txt` |
| In repo: [`traces/trace_step52.json.gz`](traces/trace_step52.json.gz) | Stitched 4-rank trace of steps 52–56 ([§1.4](#14-trace-capture)) |

### 11.2 2026-09-19 (former root `/home/chcai/mi450_fullmodel_20260919/`)

**This directory no longer exists on the host.** The home directory was reset
before 2026-09-26. The table records what the original report cited. The
facts in Part B §2–§6 and §10 are taken from that report and could not be
re-verified against the files.

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

### 11.3 Limits

- No post-run GPU diagnostic workload was run.
- No continuous ECC-counter survey was done.
- No multi-rank full-state replay, post-warmup run or convergence run was
  performed on this host.
- The 2026-09-26 halt hypothesis ([§7.3](#73-hypothesis-and-comparison)) was
  not tested with a fixed CWSR handler.
- The 2026-09-27 fault mechanism
  ([§8.3](#83-interpretation)) was not tested with a fixed CWSR handler.
  Eviction counters were not sampled during the passing R1 runs.
  `BLOCK_N=64` was not run in the high-compaction state.

Do not import any of those results from the companion host report.
