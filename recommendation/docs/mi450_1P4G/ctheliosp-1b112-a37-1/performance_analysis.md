# DLRM-v4 on MI450 (1 node × 4 GPUs): measured vs projected performance

Updated **2026-09-27**. Host `ctheliosp-1b112-a37-1`. Measured run
`perf_gbs4096_s200_20260926T214509Z` (see [host record §1](ctheliosp-1b112-a37-1.md#1-200-step-full-model-perf-run-2026-09-26)).

This document answers three questions:

1. How fast does the MLPerf DLRM-v4 benchmark (Yambda-5B, HSTU ranker) train on MI450 today?
2. How fast *should* it train, according to the Primus performance projection tool?
3. Where does the difference come from, and who needs to act on it?

The difference has two separate causes, and they are kept in separate sections:

- **[§4](#4-gap-part-1-mi450-software-stack-kernels-libraries-platform): the MI450 software stack is slow.** Kernels or libraries run far below what the hardware can do. Libraries or platform teams fix these.
- **[§5](#5-gap-part-2-missing-pieces-in-the-projection-tool): the projection tool is incomplete or wrong.** The tool leaves something out or uses a wrong hardware number. The tool owners fix these.

## 1. Summary

| | Step time | Samples/s (global, 4 GPUs) | Samples/s per GPU |
|---|---:|---:|---:|
| **Measured**, steps 50–200 (wall clock) | **2,004 ms** | **2,044** | 511 |
| **Measured**, profiled steps 52–56 (basis for the breakdown below) | 1,860 ms | 2,202 | 551 |
| **Projected** by the Primus tool for the same workload on MI450 | **210 ms** | **19,538** | 4,885 |
| **Gap** (measured ÷ projected) | **8.9×** (profiled) / 9.6× (logged) | | |

Where the 1,650 ms/step gap comes from:

| Gap source | ms/step | Share of gap | Cause type | Owner |
|---|---:|---:|---|---|
| hipBLASLt GEMMs pick a slow kernel for the model's operand layout | 1,130 | **68%** | MI450 library | hipBLASLt team |
| Triton HSTU attention, mostly backward | 269 | **16%** | MI450 kernel | Triton AMD team + this repo |
| Host-to-device input copy, blocking and unexpectedly slow | 201 | **12%** | Input pipeline / ROCm runtime | this repo, then HIP runtime |
| RCCL collectives, not hidden by compute (Socket fallback) | 24 | 1.5% | MI450 platform (driver/fabric) | RCCL / driver / fabric teams |
| hipBLASLt residual on the already-fast kernels | 18 | 1.1% | MI450 library (partly tool optimism) | hipBLASLt team |
| Everything else (LayerNorm/dropout, embedding, optimizer, idle) | 10 | 0.6% | – | – |

- **The gap is almost entirely a software gap on MI450.** A microbenchmark on the same GPU runs the *same* GEMM shapes at **1.4–2.1 PFLOP/s** when the operand layout is changed, versus **15–78 TFLOP/s** in the layout the model uses ([§4.2](#42-evidence-hipblaslt-speed-on-mi450-depends-on-operand-layout-1290)). The hardware is not the limit.
- **The tool also has gaps** ([§5](#5-gap-part-2-missing-pieces-in-the-projection-tool)). They matter less for the headline number, but they make its projection optimistic in a few places. Examples: a memory bandwidth higher than measured, GEMMs assumed to run near peak, no host-to-device copy model, no dense all-reduce.

## 2. What was compared

### 2.1 Hardware and software of the measured run

| Item | Value |
|---|---|
| GPU | **4 × AMD MI450** (`gfx1250`), 256 compute units per GPU, wave32, **432 GiB HBM per GPU**. A0/B0 stepping not identified from PCI IDs ([host record §2.1](ctheliosp-1b112-a37-1.md#21-host-cpu-gpus-and-firmware)). |
| GPU clock | 2,356–2,371 MHz read at idle (max 2,400). GEMMs reach 2.1 PF in a microbenchmark, which rules out a low clock lock. |
| Node | 1 node, 1 CPU socket (128 cores), 250 GiB host RAM |
| GPU-to-GPU communication | **Host-staged Socket over loopback** (P2P/SHM/fabric paths disabled). This works around driver/fabric failures ([host record §5](ctheliosp-1b112-a37-1.md#5-2026-09-19-experiment-record)). |
| Software | Image `recommendation-gfx1250-20260910:triton-7ff97e`: PyTorch `2.11.0+rocm7.14.0a20260625`, HIP `7.14.60850`, **Triton `3.8.0`**, TorchRec `1.7.0a0`. hipBLASLt is the gfx1250 library shipped in ROCm 7.14 (`_rocm_sdk_libraries_gfx1250`). |
| Source | this repository at `d07045df` |
| Runtime settings that affect performance | `HSTU_HAMMER_KERNEL=TRITON`, `AMDGCN_USE_BUFFER_OPS=0`, `HSTU_BWD_MAX_VGPR=0`, `TRITON_FULL_AUTOTUNE=0` (see [§4.3](#43-triton-changes-made-to-run-on-mi450-compared-with-the-mi350x-code)) |

### 2.2 Projection tool used

| Item | Value |
|---|---|
| Tool | **Primus performance projection**, DLRM-v4 workload (`framework=torchrec_dlrm`, entry point `primus.core.projection.examples.project_dlrm`) |
| Code | Public [`AMD-AGI/Primus`](https://github.com/AMD-AGI/Primus) commit **`292f805f`** (PR #1059, 2026-09-01, "DLRM-v4 workload + first-principles MI350X calibration"). Public `main` (`a48595c8`) has no later projection changes. |
| GEMM and hardware model | **gemmologist** GEMM simulator with the **`mi450x`** hardware profile. It comes from the AMD-internal `Primus-projection-internal` repo, branch `araina/dev/bump-dlrm-projection-accuracy` (`bd8f044`), whose `primus` submodule is the public commit above. Internal `master` pins an older Primus with no `mi450x` target and should not be used. |
| Hardware numbers the tool assumes for `mi450x` | bf16 peak **3,470 TFLOP/s**, HBM **13,824 GB/s** |
| Attention model | `--attn-model flop` (default): causal attention FLOPs at a fixed fraction of peak. Default fraction 0.246 and backward/forward ratio 2.03 are fitted to an **MI350X** trace. |
| Everything else | Tool defaults (preprocessor GEMMs, UVQK recompute in backward, output-projection K = 1,536, embedding scatter efficiency, elementwise passes, glue bytes/token). These are also fitted to the MI350X trace. |

Command (CPU only, no GPU needed):

```bash
# PYTHONPATH: internal repo root (for primus_internal) and its primus submodule
PRIMUS_GEMM_BACKEND=gemmologist \
PRIMUS_GEMM_BACKEND_PLUGINS=primus_internal.gemmologist_backend \
PRIMUS_HW_PROFILE_DIR=<Primus-projection-internal>/hardware_configs \
python -m primus.core.projection.examples.project_dlrm \
  --gpu-arch mi450x --nnodes 1 --gpus-per-node 4 \
  --global-batch-size 4096 --micro-batch-size 1024 \
  --max-seq-len 4096 --fill-factor 0.5082 --fill-factor-std 0.24 \
  --pooling-factors 2074 2074 2074 1 1 1 1 1 1 1 1 --embedding-gb 558.95
```

## 3. Workload configuration

The same configuration was used for the measured run and the projection. The projection flag for each setting is in the last column.

| Setting | Value | Where it comes from | Projection flag |
|---|---|---|---|
| GPUs | **4** (1 node × 4) | run | `--nnodes 1 --gpus-per-node 4` |
| Local batch (per GPU) | **1,024** samples | run | `--micro-batch-size 1024` |
| Global batch | **4,096** (no gradient accumulation) | run | `--global-batch-size 4096` |
| Model | Yambda-5B HSTU ranker, MLPerf `yambda_5b.gin` defaults | gin | – |
| HSTU layers | **3** | gin | default 3 |
| Hidden size / heads / q-k dim / v dim | **512 / 4 / 128 / 128** | gin | defaults |
| Max sequence length | **4,096** (4,086 history + 8 contextual + 1 candidate) | gin | `--max-seq-len 4096` (tool default 3,650 is wrong for us) |
| Mean valid tokens per sample | **2,082** (jagged; from GEMM row counts in the trace, 2.01–2.26 M tokens per GPU per step) | trace | – |
| Fill rate, token-weighted | **0.508** = 2,082 / 4,096 | trace | `--fill-factor 0.5082` |
| Fill rate, spread | std **≈ 0.24**, derived below | derived | `--fill-factor-std 0.24` |
| Trainer's logged `fill` | 38.4% (range 31–41%). This is **FLOP-weighted**, not token-weighted, so it is not the same number as the 0.508 above. | train.log | – |
| Embedding tables | **11 tables, 293.05 M rows × 512, FP32 = 558.95 GiB**; sparse optimizer row-wise Adagrad | gin, trace | `--embedding-gb 558.95` |
| Lookups per sample | item / artist / album: one per history token + candidate (**2,074** each); uid + 7 cross features: 1 each | model | `--pooling-factors 2074 2074 2074 1 ×8` |
| Embedding sharding | 8 tables table-wise, 3 tables column-wise (4 × 128 columns); IDs deduplicated before lookup | trace | tool supports row-wise only (see [§5](#5-gap-part-2-missing-pieces-in-the-projection-tool)) |
| Precision | bf16 dense compute, FP32 embedding tables, FP16 embedding all-to-all | gin | defaults |
| Dense optimizer | Adam | gin | not modelled |
| Kernels | HSTU attention, LayerNorm, dropout: **Triton**. GEMMs: **hipBLASLt**. Embedding: **FBGEMM TBE**. Communication: **RCCL**. No aiter or Composable Kernel kernels run. | trace | – |

**Fill spread derivation.** Attention cost grows with the *square* of sequence length, so the tool needs the average of L², not just the average of L. The trainer's logged fill is (jagged FLOPs ÷ padded FLOPs). Per layer, linear FLOPs are 32.2 G × fill and attention FLOPs are 60.1 G × E[L²]/4096². Setting 0.384 × 92.3 G = 0.508 × 32.2 G + 60.1 G × x gives x ≈ 0.316. Then std = √(0.316 − 0.508²) ≈ 0.24.

## 4. Gap part 1: MI450 software stack (kernels, libraries, platform)

These are real costs on MI450 today. The tool is right to assume they can be much faster; the fix is on the MI450 software side.

### 4.1 Gap table

Measured is rank 0, mean of profiled steps 52–56. The four GPUs agree within 1.5%, and the GPU is busy 99.6% of the step, so kernel time adds up to step time.

| # | Component | Library | Measured ms/step | Projected ms/step | Gap ms | Share of gap | Evidence | Suggested follow-up | Owner |
|---|---|---|---:|---:|---:|---:|---|---|---|
| A1 | GEMMs on the **32×16×32 fallback kernel**: HSTU forward projections (`addmm`) and all weight gradients | **hipBLASLt** | 1,043.4 | 26.0 | **1,017** | **61.7%** | Runs at 56–76 TFLOP/s. The same shapes run at 935–1,979 TFLOP/s with a different operand layout on the same GPU ([§4.2](#42-evidence-hipblaslt-speed-on-mi450-depends-on-operand-layout-1290)). | **File with the hipBLASLt team**: gfx1250 has no fast kernels for the NN (`Ailk_Bljk`) and weight-gradient (`Ailk_Bjlk`) layouts at bf16 and large M; attach the §4.2 repro. Short-term workaround in this repo: store HSTU weights in the fast layout (`F.linear`) and transpose operands before the weight gradient. | hipBLASLt; this repo (workaround) |
| A2 | Weight gradient with N = 256 on a **large tile with no split-K**: only 4 workgroups for 256 compute units | **hipBLASLt** | 113.5 | 1.0 | **112** | **6.8%** | 15 TFLOP/s; 1,357 TFLOP/s in the fast layout | Same hipBLASLt issue. Also ask for split-K / stream-K solutions for tall-K weight gradients (K ≈ 2 M tokens). | hipBLASLt |
| A3 | GEMMs already on good kernels (`Alik_Bljk`, 128×256 / 256×240 tiles) | hipBLASLt | 31.4 | 13.4 | 18 | 1.1% | 440–1,490 TFLOP/s measured vs the tool's ~2,500 | Low priority. Part of this is tool optimism ([§5](#5-gap-part-2-missing-pieces-in-the-projection-tool), B4). | hipBLASLt |
| A4 | **HSTU attention backward** | **Triton** | 283.1 | 41.0 | **242** | **14.7%** | Backward takes **6.1×** the forward time; a healthy flash-attention backward is ~2–2.5×. Runs MI350X-tuned configs plus MI450 correctness workarounds ([§4.3](#43-triton-changes-made-to-run-on-mi450-compared-with-the-mi350x-code)). | 1) Re-autotune on gfx1250 (`TRITON_FULL_AUTOTUNE=1`, this repo). 2) **File with the Triton AMD team**: the VGPR-ceiling spill/fault that forced `BLOCK_N` from 128 to 64. 3) Re-test with buffer ops on once the buffer-op bug is fixed. | Triton AMD team; this repo |
| A5 | HSTU attention forward | Triton | 46.7 | 20.3 | 26 | 1.6% | About 0.11 of peak, vs 0.25 on the MI350X reference | Same as A4 (autotune; buffer ops). | Triton AMD team; this repo |
| A6 | **Host-to-device input copy**: 2 × ~108 MB per step | PyTorch input path / HIP runtime | 200.5 | 0 | **201** | **12.2%** | Blocking `hipMemcpyWithStream` on the compute stream at **~1 GB/s**. The same 100 MB pageable copy on an idle GPU runs at **73 GB/s**. The DataLoader does not use `pin_memory`, and `sample.to(device)` is blocking. | 1) This repo: `pin_memory=True`, `non_blocking=True`, prefetch the next batch on a side stream. 2) If still slow, **report the 70× pageable-copy slowdown to the HIP runtime team** (check CPU contention from DataLoader workers, NUMA, IOMMU). | this repo; HIP runtime |
| A7 | RCCL collectives not overlapped with compute. Kernel time is 77 ms, of which 25 ms is exposed. | **RCCL** / driver | 25.1 | 0.9 | 24 | 1.5% | Socket over loopback at ~2.3 GB/s. Direct GPU paths hang or fault ([host record §5](ctheliosp-1b112-a37-1.md#5-2026-09-19-experiment-record)). | Driver / fabric / RCCL: restore P2P or xGMI transport. This matters much more at larger scale. | RCCL; amdgpu driver; fabric |
| A8 | LayerNorm, dropout, SiLU, elementwise, jagged ops | Triton + PyTorch | 84.5 | 78.9 | 6 | 0.3% | Close to projection | None needed now | – |
| A9 | Embedding lookup + deduplicated gradient scatter | FBGEMM + PyTorch | 24.9 | 27.4 | −3 | −0.2% | Close (but see B10: this match is partly coincidental) | None | – |
| A10 | Dense Adam + GPU idle | PyTorch | 6.8 | 0 | 7 | 0.4% | | None | – |
| | **Total** | | **1,859.7** | **209.6** | **1,650** | 100% | | | |

Projected GEMM times in A1–A3 come from pricing every GEMM in the trace at its exact shape with the tool's GEMM model. This totals 40.4 ms, matching the tool's own 41.2 ms GEMM total. A4/A5 split the tool's 61.3 ms attention using its 2.03 backward/forward ratio.

By library:

| Library | Gap ms/step | Share of gap |
|---|---:|---:|
| **hipBLASLt** | ~1,148 | **~70%** |
| **Triton** | ~274 | **~17%** |
| PyTorch input path / HIP runtime | 201 | 12% |
| RCCL | 24 | 1.5% |
| FBGEMM | ≈ 0 | 0 |
| aiter / Composable Kernel | 0 | 0 (not used by this workload) |

### 4.2 Evidence: hipBLASLt speed on MI450 depends on operand layout (12–90×)

Microbenchmark on GPU 0 of this host, same image, bf16, 2026-09-27. "Model's layout" is exactly how PyTorch calls the GEMM in training. The fast layout does the same math with the weight (B) operand stored transposed.

| GEMM (M × N × K) | Role in model | Model's layout TFLOP/s | Fast layout TFLOP/s | Speed-up |
|---|---|---:|---:|---:|
| 8,192 × 8,192 × 8,192 (square) | reference | **78** | **2,108** | 27× |
| 2,097,152 × 2,048 × 512 | UVQK projection forward | 75 | 935 | 12× |
| 2,097,152 × 512 × 1,536 | output projection forward | 77 | 1,124 | 15× |
| 512 × 2,048 × 1,048,576 tokens | UVQK weight gradient | 71 | 1,979 | 28× |
| 1,536 × 512 × 1,048,576 tokens | output projection weight gradient | 56 | 1,821 | 33× |
| 256 × 512 × 1,048,576 tokens | preprocessor weight gradient | 15 | 1,357 | 90× |

- **Forward.** `torch.addmm(b, X, W)` with W stored as [K, N] is slow. `F.linear(X, W_t, b)` with W_t stored as [N, K] is fast.
- **Weight gradient.** `dY.t() @ X` (either order) is slow. With both operands first materialized transposed (`dYt @ Xt.t()`), it is fast. Even counting the two transposes (naive copies), the weight gradient is 2.3–3× faster than today.
- **MI450 GEMM hardware is fine.** hipBLASLt on gfx1250 reaches 2.1 PFLOP/s in the fast layout, 61% of the 3,470 TFLOP/s peak.
- **Kernel names in the trace.** Slow forward: `Cijk_Ailk_Bljk_…MT32x16x32_MI16x16x1`. Slow weight gradient: `Cijk_Ailk_Bjlk_…MT32x16x32` and `…MT256x128x64`. Fast: `Cijk_Alik_Bljk_…MT256x240x64` / `MT128x256x128`.

**Estimated effect.** The model runs 93.9 TFLOP of GEMMs per GPU per step. At ~1.8 PFLOP/s that is ~52 ms instead of today's 1,188 ms.

### 4.3 Triton changes made to run on MI450, compared with the MI350X code

The HSTU kernels came from the MI350X (gfx950) version of this benchmark. The changes below were needed for the model to run correctly on MI450. They were made for correctness, not speed, and each one may cost performance.

| Change | Why it was needed | Active in the measured run? | Possible performance cost | Follow-up |
|---|---|---|---|---|
| Attention **backward `BLOCK_N` 128 → 64** on gfx1250 with Triton ≥ 3.8 (commit `3e78d97`) | The 128 tile reaches the 1,024-VGPR limit, spills, and faults with a bad store address | **Yes** (Triton 3.8.0) | Smaller tiles mean more loads per FLOP; a likely contributor to the 6× backward/forward ratio | Triton AMD team: fix spill/fault at `BLOCK_N=128`; then retune |
| **`AMDGCN_USE_BUFFER_OPS=0`**, runner default (commit `3f87a93`) | The AMD buffer-op pass hangs or silently corrupts data when a kernel's pointers straddle the 2 GiB cutoff. Standalone repro: `scripts/repro_gfx1250_buffer_ops.py`. | **Yes** | Disables buffer loads/stores in **all** Triton kernels | Triton AMD team: fix the buffer-op pass; then measure with buffer ops on |
| Attention ported off `tl.make_block_ptr` (commit `a7c30bb`) | Newer Triton removed block pointers | Yes | Unknown; not measured | Measure after re-autotune |
| Configs are the **MI350X autotune winners**, pinned (`TRITON_FULL_AUTOTUNE=0`, `_autotune_pinning.py`) | Avoids long autotune; carried over from MI350X | Yes | Tiles were never tuned for gfx1250 (wave32, different register file) | This repo: full autotune on gfx1250, then pin the MI450 winners |
| Optional attention-backward VGPR cap (`HSTU_BWD_MAX_VGPR=256`) | B0 correctness experiment | **No** (0 = off) | Forces spills if turned on | Keep off for performance runs |
| Optional weighted-LayerNorm backward pin (`WEIGHTED_LN_BWD_BLOCK_N`) | B0 NaN investigation | **No** (0 = default) | – | – |
| int64 row offsets in LayerNorm / dropout kernels (commits `1a3dcd6`, `2d68965`) | int32 overflow produced NaN on gfx1250 | Yes | Negligible | None |

### 4.4 Reference: the MI350X trace the tool was calibrated on

The Primus source (`project_dlrm.py`) records a measured MI350X step for a close configuration: 8 GPUs, local batch 1,024, max seq 3,650, fill 0.6, **2.23 M tokens per GPU** (ours: 2.08 M). It is a "flydsl" run, so some kernels may differ from our Triton build. Treat this as a rough reference only.

| Component | MI350X reference ms/step | MI450 measured ms/step | MI450 ÷ MI350X |
|---|---:|---:|---:|
| Dense GEMMs | 132 | 1,188 | 9.0× slower |
| Attention (fwd + bwd) | 131 | 330 | 2.5× slower |
| LayerNorm / dropout / elementwise / glue | 158 | 85 | faster |
| Embedding | 52 | 25 | faster (ID dedup) |
| Exposed collectives | 0.9 | 25 | 29× |
| **Step** | **472** | **1,860** | **3.9× slower** |

MI450 has more compute and memory bandwidth than MI350X, so each MI450 row should be *faster*. The GEMM and attention rows are where MI450 software currently falls short.

### 4.5 What fixing the MI450 stack would give (estimate)

| Fix (cumulative) | Est. step ms | Est. samples/s | vs today |
|---|---:|---:|---:|
| Today (profiled) | 1,860 | 2,202 | 1.0× |
| hipBLASLt fast kernels for all layouts (A1–A2, ~1.8 PF) | ~724 | ~5,660 | 2.6× |
| + pinned, asynchronous input copy (A6) | ~524 | ~7,820 | 3.5× |
| + attention at the projected efficiency (A4–A5) | ~255 | ~16,000 | 7.3× |
| Projection | 210 | 19,538 | 8.9× |

These estimates subtract the saved kernel time from the step. This is reasonable here because the GPU is busy 99.6% of the time and work is mostly serialized, but it is still an estimate.

## 5. Gap part 2: missing pieces in the projection tool

These are cases where the tool itself is incomplete or uses a wrong input. None is the main reason MI450 is slow today. They make the projection optimistic, or blind to a real cost, and they will matter more once the MI450 stack is fixed.

### 5.1 Wrong or unverified hardware numbers

| # | Item | Tool assumes | Measured / actual on this host | Effect on projection | Suggested fix | Owner |
|---|---|---|---|---|---|---|
| B1 | **HBM bandwidth** (`mi450x` profile) | **13,824 GB/s** peak; memory-bound terms use 60% of it (~8.3 TB/s) | Device-to-device copy (8 GiB, read + write): **7,306 GB/s** achieved, i.e. 53% of the tool's peak | Memory-bound terms (elementwise 37 ms, glue 42 ms, embedding 27 ms projected) are optimistic by ~15% or more | Confirm the MI450 HBM spec; calibrate the profile against a measured copy/stream bandwidth | projection tool |
| B2 | **GEMM peak reachability** | Tall-K weight gradients priced at up to **99.6%** of 3,470 TFLOP/s | Best hipBLASLt result on MI450: 1.4–2.1 PF (**40–61%** of peak) | Even with A1/A2 fixed, projected GEMM time (41 ms) is ~20–30% optimistic vs ~52 ms | Cap per-shape efficiency with measured per-arch data (e.g. `hipblaslt-bench` sweep) | projection tool |
| B3 | Architecture names | Only `mi450x`, and only on an unmerged internal branch. `gfx1250` and `mi450` are rejected; internal `master` has no MI450 target at all. | GPUs report `gfx1250` | Users on `master` get an error, or pick the MI455X profile (different HBM) | Merge the branch; accept `gfx1250` / `mi450` as aliases | projection tool |
| B4 | bf16 peak | 3,470 TFLOP/s | This repo's trainer uses 3,500 TFLOP/s for MFU | ~1%, cosmetic | Align the two | projection tool / this repo |

### 5.2 Missing or simplified models

| # | Missing piece | What the tool does now | What actually happens in this run | Effect on projection | Suggested fix | Owner |
|---|---|---|---|---|---|---|
| B5 | **GEMM library behaviour** | Always picks the best tile; treats all operand layouts as equal ("transposes are free") | hipBLASLt speed differs 12–90× by layout on MI450 ([§4.2](#42-evidence-hipblaslt-speed-on-mi450-depends-on-operand-layout-1290)) | Cannot predict or flag the 1,130 ms hipBLASLt gap | Add a per-arch, per-layout efficiency table from `hipblaslt-bench`, or a mode that replays the library's actual solution choice | projection tool |
| B6 | GEMM backward shapes | Backward = 2 × forward time | Backward has distinct data-gradient and weight-gradient shapes. The weight gradient (K = 2 M tokens) is where the library fails. | Hides layout- and shape-specific problems | Price data-gradient and weight-gradient GEMMs explicitly | projection tool |
| B7 | **Attention model** | Causal FLOPs × fixed efficiency **0.246**, backward/forward **2.03**, both fitted on MI350X. The kernel-level `fav3_hstu` model needs the `origami` backend, which is not available. | Triton HSTU kernel on gfx1250: efficiency ≈ **0.11**, backward/forward ≈ **6.1** | Projection assumes MI350X attention efficiency on MI450 | Per-arch calibration, or a model of the actual Triton kernel; ship `origami` or make `fav3_hstu` work with gemmologist | projection tool |
| B8 | Sequence-length distribution | One mean fill + one std | Jagged lengths per sample. We derived std = 0.24 indirectly ([§3](#3-workload-configuration)). | Attention cost depends on E[L²]; an error here shifts attention time | Accept Σ L and Σ L² (or a histogram) per batch as input | projection tool |
| B9 | **Host-to-device input copy** | Not modelled; only accepts a measured value (`--h2d-ms`) | 200 ms/step (A6) | Blind to 12% of today's step | Model the input pipeline: bytes per step, pinned vs pageable, overlapped or blocking | projection tool |
| B10 | Embedding sharding and ID dedup | Row-wise sharding only; no ID deduplication; every lookup costs a full row | 8 table-wise + 3 column-wise tables; IDs deduplicated, so only ~50 k unique rows per step are exchanged, not 6.4 M | The projected 27 ms is close to measured by coincidence; it will not track changes | Support table-wise / column-wise plans and a dedup ratio | projection tool |
| B11 | **Collectives** | Embedding all-to-all only, with xGMI bandwidth. Exposed fraction 0 and 0.86 ms "sync" copied from MI350X. | Also a dense-gradient all-reduce (15 M FP32 params, 32 ms kernel time); Socket transport over loopback; 25 ms exposed | Collectives under-projected by ~24 ms here; far more at multi-node scale | Add the dense all-reduce, a transport choice (Socket / xGMI / RoCE) and an overlap model | projection tool |
| B12 | Dense optimizer, idle, launch overhead | Not modelled | ~7 ms/step | Small | Add an optimizer term | projection tool |
| B13 | Defaults tied to one MI350X trace | Defaults: 8 GPUs, global batch 8,192, max seq 3,650, fill 0.6, and MI350X-fitted efficiencies. `--trace-roles` prints hard-coded MI350X numbers as "measured". | Our run needs every workload flag in [§2.2](#22-projection-tool-used) overridden | Easy to project the wrong workload by accident | Separate workload defaults from per-arch calibration; add a trace-import mode that fills the measured column from a real trace | projection tool |

## 6. Limits of this analysis

- **Single run.** One 200-step run, all within LR warm-up. The breakdown is rank 0 over 5 profiled steps (1,860 ms/step), which is 7% faster than the 150-step wall-clock average (2,004 ms/step) because the jagged token count varies per step.
- **Microbenchmark conditions.** Microbenchmarks ran on an idle GPU 0 in a plain container. The weight-gradient rows use M = 1,048,576 tokens instead of ~2.1 M, because the plain container could not allocate more than ~20 GiB. TFLOP/s rates at this size are independent of M.
- **H2D root cause is open.** Why the in-training pageable copy runs at 1 GB/s instead of 73 GB/s is not known yet.
- **Tool calibration.** Several tool defaults are fitted to the MI350X trace, so the 210 ms projection is a calibrated estimate, not a pure first-principles number.
- **Estimates.** The §4.5 ladder is an estimate, not a measurement.

## 7. Files

| File | Content |
|---|---|
| [`traces/trace_step52.json.gz`](traces/trace_step52.json.gz) | Stitched 4-GPU trace, steps 52–56, used for every measured number above |
| [`ctheliosp-1b112-a37-1.md`](ctheliosp-1b112-a37-1.md) | Host, software stack and run record |
