# DLRM-v4 on MI450 (1 node × 4 GPUs): measured vs projected performance

Updated **2026-09-28**. Host `ctheliosp-1b112-a37-1`. Measured run
`perf_gbs4096_s200_20260926T214509Z` (see [host record §1](ctheliosp-1b112-a37-1.md#1-200-step-full-model-perf-run-2026-09-26)).
Current configuration, after the attention-backward revert R1 ([§4.3.1](#431-reverted-on-this-host)):
run `bn128_gbs4096_s600_20260927T034620Z`.

> **Stability caveat for R1 (2026-09-28):** R1's performance numbers stand.
> However, **R1 is not stable on this host until the CWSR trap handler is
> fixed.**
>
> - The same R1 config passed 600 steps twice at 6–7 h after boot.
> - It then **faulted in 3 of 3 runs launched 15–16 h after boot**. The two
>   runs where the runtime named the kernel named `_hstu_attn_bwd`, shortly
>   after step 100.
> - Losses up to the fault were identical to the passing runs.
> - The faults coincided with GPU queue evictions caused by kernel memory
>   compaction, which save and restore the waves through the known-faulty
>   handler.
>
> Details are in [host record §8](ctheliosp-1b112-a37-1.md#8-2026-09-27-nan--fault-repro-r1-faults-and-cwsr-exposure).

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
| **Measured**, profiled steps 52–56 | **1,735 ms** | **2,361** | 590 |
| **Projected** by the Primus tool for the same workload on MI450 | **210 ms** | **19,538** | 4,885 |
| **Gap** (measured ÷ projected) | **8.3×** | | |

Where the 1,525 ms/step gap comes from:

| Gap source | ms/step | Share of gap | Cause type | Owner |
|---|---:|---:|---|---|
| hipBLASLt GEMMs pick a slow kernel for the model's operand layout | 1,130 | **74%** | MI450 library | hipBLASLt team |
| Host-to-device input copy, blocking and unexpectedly slow | 201 | **13%** | Input pipeline / ROCm runtime | this repo, then HIP runtime |
| Triton HSTU attention (backward 115, forward 26) | 141 | **9%** | MI450 kernel | Triton AMD team + this repo |
| RCCL collectives, not hidden by compute (Socket fallback) | 24 | 1.6% | MI450 platform (driver/fabric) | RCCL / driver / fabric teams |
| hipBLASLt residual on the already-fast kernels | 18 | 1.2% | MI450 library (partly tool optimism) | hipBLASLt team |
| Everything else (LayerNorm/dropout, embedding, optimizer, idle) | 10 | 0.7% | – | – |

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
| Runtime settings that affect performance | `HSTU_HAMMER_KERNEL=TRITON`, `AMDGCN_USE_BUFFER_OPS=0`, `HSTU_BWD_MAX_VGPR=0`, `TRITON_FULL_AUTOTUNE=0`, `PYTORCH_CUDA_ALLOC_CONF` cleared, Triton `num_stages` clamped to 1 by default (see [§4.3](#43-triton-changes-made-to-run-on-mi450-compared-with-the-mi350x-code)). Current configuration adds `HSTU_BWD_BLOCK_N=128` (R1). |

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

Measured is rank 0, mean of profiled steps 52–56 of the run before R1 (`BLOCK_N` 64, 1,860 ms). The four GPUs agree within 1.5%, and the GPU is busy 99.6% of the step, so kernel time adds up to step time. After R1 only A4 changes; its current values are given in the row, and the current total is 1,735 ms (gap 1,525 ms).

| # | Component | Library | Measured ms/step | Projected ms/step | Gap ms | Share of gap | Evidence | Suggested follow-up | Owner |
|---|---|---|---:|---:|---:|---:|---|---|---|
| A1 | GEMMs on the **32×16×32 fallback kernel**: HSTU forward projections (`addmm`) and all weight gradients | **hipBLASLt** | 1,043.4 | 26.0 | **1,017** | **61.7%** | Runs at 56–76 TFLOP/s. The same shapes run at 935–1,979 TFLOP/s with a different operand layout on the same GPU ([§4.2](#42-evidence-hipblaslt-speed-on-mi450-depends-on-operand-layout-1290)). | **File with the hipBLASLt team**: gfx1250 has no fast kernels for the NN (`Ailk_Bljk`) and weight-gradient (`Ailk_Bjlk`) layouts at bf16 and large M; attach the §4.2 repro and the captured GEMM list in [`hipBLASLt/`](hipBLASLt/README.md). Short-term workaround in this repo: store HSTU weights in the fast layout (`F.linear`) and transpose operands before the weight gradient. | hipBLASLt; this repo (workaround) |
| A2 | Weight gradient with N = 256 on a **large tile with no split-K**: only 4 workgroups for 256 compute units | **hipBLASLt** | 113.5 | 1.0 | **112** | **6.8%** | 15 TFLOP/s; 1,357 TFLOP/s in the fast layout | Same hipBLASLt issue. Also ask for split-K / stream-K solutions for tall-K weight gradients (K ≈ 2 M tokens). | hipBLASLt |
| A3 | GEMMs already on good kernels (`Alik_Bljk`, 128×256 / 256×240 tiles) | hipBLASLt | 31.4 | 13.4 | 18 | 1.1% | 440–1,490 TFLOP/s measured vs the tool's ~2,500 | Low priority. Part of this is tool optimism ([§5](#5-gap-part-2-missing-pieces-in-the-projection-tool), B4). | hipBLASLt |
| A4 | **HSTU attention backward** | **Triton** | 283.1; **155.8 after R1** | 41.0 | **242; 115 after R1** | **14.7%; 7.5% of the current gap** | Backward took **6.1×** the forward time, **3.3×** after R1; a healthy flash-attention backward is ~2–2.5×. Runs MI350X-tuned configs plus MI450 correctness workarounds ([§4.3](#43-triton-changes-made-to-run-on-mi450-compared-with-the-mi350x-code)). | 1) Re-autotune on gfx1250 (`TRITON_FULL_AUTOTUNE=1`, this repo). 2) Restore `BLOCK_N` 128: reverted, 283 → 156 ms/step. It passed 600 steps on this host, but later runs faulted, so it is not stable until the CWSR handler is fixed ([§4.3.1](#431-reverted-on-this-host)); **file with the LLVM AMDGPU team** the extended-VGPR bug that forced it to 64. 3) Re-test with buffer ops on only after the buffer-op bug is fixed: turning them on wedged this host on 2026-09-27 (T3). | LLVM AMDGPU team; Triton AMD team; this repo |
| A5 | HSTU attention forward | Triton | 46.7 | 20.3 | 26 | 1.6% | About 0.11 of peak, vs 0.25 on the MI350X reference | Retune on gfx1250 (T1). Keep `num_stages` 1: the pinned MI350X `num_stages=2` is 49% slower on this host (T2). Buffer ops stay off (T3). | Triton AMD team; this repo |
| A6 | **Host-to-device input copy**: 2 × ~108 MB per step | PyTorch input path / HIP runtime | 200.5 | 0 | **201** | **12.2%** | Blocking `hipMemcpyWithStream` on the compute stream at **~1 GB/s**. The same 100 MB pageable copy on an idle GPU runs at **73 GB/s**. The DataLoader does not use `pin_memory`, and `sample.to(device)` is blocking. | 1) This repo: `pin_memory=True`, `non_blocking=True`, prefetch the next batch on a side stream. 2) If still slow, **report the 70× pageable-copy slowdown to the HIP runtime team** (check CPU contention from DataLoader workers, NUMA, IOMMU). | this repo; HIP runtime |
| A7 | RCCL collectives not overlapped with compute. Kernel time is 77 ms, of which 25 ms is exposed. | **RCCL** / driver | 25.1 | 0.9 | 24 | 1.5% | Socket over loopback at ~2.3 GB/s. Direct GPU paths hang or fault ([host record §5](ctheliosp-1b112-a37-1.md#5-2026-09-19-experiment-record)). | Driver / fabric / RCCL: restore P2P or xGMI transport. This matters much more at larger scale. | RCCL; amdgpu driver; fabric |
| A8 | LayerNorm, dropout, SiLU, elementwise, jagged ops | Triton + PyTorch | 84.5 | 78.9 | 6 | 0.3% | Close to projection | None needed now | – |
| A9 | Embedding lookup + deduplicated gradient scatter | FBGEMM + PyTorch | 24.9 | 27.4 | −3 | −0.2% | Close (but see B10: this match is partly coincidental) | None | – |
| A10 | Dense Adam + GPU idle | PyTorch | 6.8 | 0 | 7 | 0.4% | | None | – |
| | **Total** | | **1,859.7; 1,735 after R1** | **209.6** | **1,650; 1,525 after R1** | 100% | | | |

Projected GEMM times in A1–A3 come from pricing every GEMM in the trace at its exact shape with the tool's GEMM model. This totals 40.4 ms, matching the tool's own 41.2 ms GEMM total. A4/A5 split the tool's 61.3 ms attention using its 2.03 backward/forward ratio.

By library:

| Library | Gap ms/step | Share of gap |
|---|---:|---:|
| **hipBLASLt** | ~1,148 | **~70%** |
| **Triton** | ~274; ~147 after R1 | **~17%**; ~10% of the current gap |
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

The HSTU kernels came from the MI350X (gfx950) version of this benchmark. The changes below were needed for the model to run correctly on MI450. They were made for correctness, not speed, and each one may cost performance. The table also lists two run-time settings that differ from the MI350X image: buffer ops (T3) and the PyTorch allocator (T8). Rows are ordered by their estimated performance impact on the measured run, largest first; the least certain estimate (T9) is last. Cost ratings are relative to the 1,860 ms profiled step: **High** ≥ 5% (≥ ~90 ms/step), **Medium** 1–5%, **Low** < 1%, **None** no cost. Each estimate states its basis and a confidence (high = measured, medium = derived from measurements, low = reasoned bound). Changes that have since been reverted on this host are listed separately in [§4.3.1](#431-reverted-on-this-host).

On 2026-09-27 each change was re-checked against the MI350X reference (commit `ad52cb1`). The goal was to match the reference wherever there is no correctness or fault reason not to. End-to-end checks used the same setup as R1: 4 GPUs, local batch 1,024, 600 steps, `BLOCK_N` 128. Three other differences were found to already behave like the reference, so they are not listed: the separated RNG + dropout path (MI350X already takes it), `EMBEDDING_ROW_SCALE` (default 1.0, no effect) and the peak-FLOPS table (reporting only).

| # | Change | Why it was needed | Active in the measured run? | Estimated performance cost (rating, ms/step, confidence) | Revisited on this host (2026-09-27): can it be reverted? | Follow-up |
|---|---|---|---|---|---|---|
| T1 | Configs are the **MI350X autotune winners**, pinned (`TRITON_FULL_AUTOTUNE=0`, `_autotune_pinning.py`) | Avoids long autotune; carried over from MI350X | Yes | **Medium: est. 25–50 ms/step (1.3–2.7%).** Backward: at `BLOCK_N` 64 the 8-warp tile was 9% faster per call than the pin, about 25 ms/step of 283. How much a retune adds on top of R1 is not measured. Forward: at most the 26 ms/step gap to the projection (A5). *Confidence: medium for the backward (measured per call); low for the forward (upper bound only).* | **Partly retuned (backward only).** 48 backward candidates compiled without dispatch. Timed: 6 of them whose VGPR use is no deeper than production, plus 2 `SEQUENCE_PARALLEL` and 2 VGPR-capped variants. At `BLOCK_N` 64, only **8 warps** beat the pin (100.3 vs 110.0 ms per call); the reverted `BLOCK_N` 128 tile with 4 warps is faster still (R1). `matrix_instr_nonkdim` 16 and 32 produce identical code on gfx1250. `SEQUENCE_PARALLEL` is slower (130–147 ms). A full autotune was not run: it dispatches every candidate, including high-VGPR tiles that have not been checked for the extended-VGPR fault (R1). The forward (22.5 ms per call) was not retuned. | This repo: retune the backward around the 128 tile with the compile-first method; retune the forward the same way. |
| T2 | **`num_stages` forced to 1** on every autotuned Triton kernel, on HIP with Triton ≥ 3.8 (`clamp_num_stages` in `common.py`, commit `d690810`; `TRITON_ALLOW_PIPELINING=1` turns it off) | On the MI450 A0 bring-up, Triton 3.8's software pipeliner (`num_stages` > 1) produced faults or wrong results in the attention, jagged and LayerNorm kernels. `num_stages=1` matched Triton 3.6. | Yes. In the trace, only the attention forward (pinned `num_stages=2`) changes time. | **None; the clamp is a gain of 22.7 ms/step (1.2%).** With pipelining on, the attention forward was 22.7 ms/step slower, and no other kernel changed. *Confidence: high (measured end to end).* | **Reverting is safe here but slower, so keep the clamp.** In the 600-step run with pipelining on there was no GPU fault and no NaN, and the loss matched R1 at steps 100–600 (0.13918 → 0.13791). Attention forward went from **46.6 to 69.3 ms/step (+49%)**. Every other kernel was unchanged. The profiled step went from 1,735 to 1,757 ms (+1.3%). Compiled without dispatch, the stages-2 forward uses async global→LDS copies (`global_load_async_to_lds_b128`) and has 245 vs 234 VGPRs and 34 vs 32 KiB LDS. It has the same occupancy (4), no spills and an almost identical loop body, so the slowdown does not show in the code. The pinned `num_stages=2` is an MI350X autotune result. Run folder: `rv_pipe_gbs4096_s600_20260927T043037Z`. | Keep the clamp. Include `num_stages` in the gfx1250 retune (T1). Triton AMD team: profile the async-copy pipelined forward (LDS bank conflicts, copy throughput) before relying on pipelining on gfx1250. |
| T3 | **`AMDGCN_USE_BUFFER_OPS=0`**, runner default (commit `3f87a93`). The MI350X image leaves buffer ops on (Triton default). | The AMD buffer-op pass hangs or silently corrupts data when a kernel's pointers straddle the 2 GiB cutoff. Standalone repro: `scripts/repro_gfx1250_buffer_ops.py`. | **Yes** | **Low: est. 0–10 ms/step (≤ 0.5%).** Cannot be measured, because turning buffer ops on wedges the node. At the benchmark shape the attention kernels get no buffer ops even when enabled. Any gain would come from the LayerNorm, dropout, jagged and embedding kernels, which take ≤ 85 ms/step in total (A8, including PyTorch ops). Assumes buffer ops save at most ~10% of that (less address arithmetic in memory-bound kernels). *Confidence: low.* | **No: reverting it wedged the node.** The run had buffer ops on, as on MI350X, plus a guard that checked every compiled kernel. The known hang pattern is a "hybrid" kernel that mixes buffer and global memory instructions. The guard rebuilt each hybrid kernel with buffer ops off, so no hybrid kernel ever ran. Of 1,163 compiles, 253 (10 kernels) were hybrid: both attention kernels, `split_2D_jagged_multirow`, `concat_2D_jagged_multirow`, the LayerNorm, dropout and embedding kernels. 862 were buffer-only and 48 global-only. Even so, a GPU kernel hung before step 50: the NCCL all-reduce after work 238 never completed. The GPU scheduler firmware (MES) on two GPUs then stopped responding, and the host needed a power cycle. Earlier result, corrected: compiled at the benchmark shape, the attention kernels have no buffer ops, but training also compiles other variants of them that are hybrid. Run folder and evidence (`dmesg.wedge.txt`, `bufops_guard.tsv`): `rv_bufops_gbs4096_s600_20260927T050103Z`. | Keep buffer ops off. **Triton AMD team:** the fault is not limited to hybrid kernels; attach the guard log and dmesg. |
| T4 | int64 row offsets in LayerNorm / dropout kernels (commits `1a3dcd6`, `2d68965`) | The LayerNorm and dropout kernels were moved off block pointers (T9). Block pointers compute addresses in 64 bits; the plain-pointer version first computed `row × stride` in 32 bits, which overflows. | Yes | **Negligible: est. < 1 ms/step.** Adds one 64-bit multiply per row in memory-bound kernels, which take 84.5 ms/step in total (A8). Cannot be A/B-tested, because the 32-bit version is wrong. *Confidence: medium.* | **No: needed at this workload's size.** It restores the 64-bit addressing of the MI350X code. On-device test on GPU 0 used the dropout forward's store (`stride` 1,536, bf16) at 2.1 M rows, about the tokens per GPU per step here. The 32-bit version wrapped at the predicted row, 1,398,102. **701,898 rows went to the wrong place and 359 M elements were written outside the output buffer**, which is silent memory corruption. The 64-bit version wrote every row correctly. The LayerNorm kernels (stride 512) wrap only above 4.19 M rows, so for them the change is insurance at no cost. | None |
| T5 | Optional attention-backward VGPR cap (`HSTU_BWD_MAX_VGPR=256`) | B0 correctness experiment | **No** (0 = off) | **None while off.** If turned on: about +240 ms/step (+13%), scaled from the measured +85% per call. *Confidence: high.* | **Measured:** 256 cap → 358 spills, **203.9 ms** per call (+85%). Keep it off. | Keep off for performance runs |
| T6 | Optional weighted-LayerNorm backward pin (`WEIGHTED_LN_BWD_BLOCK_N`) | B0 NaN investigation | **No** (0 = default) | **None while off.** If turned on: small, since the weighted-LayerNorm backward is part of A8. Not measured. *Confidence: medium.* | Not applicable; off in the run | – |
| T7 | Position-embedding mask: `(a) and b < D` → `(a) & (b < D)` (commit `3f87a93`) | Triton 3.8 deprecates `and` on tensors | Yes | **None: 0 ms/step.** The compiled code is identical. *Confidence: high.* | **Could be reverted, but there is no reason to.** Both forms compile on Triton 3.8 (the old one with a deprecation warning) to identical AMDGCN. It is a style change, not a correctness fix. | None |
| T8 | PyTorch allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in the image, as on MI350X) is **cleared** by the MI450 run scripts | Expandable segments cannot map large blocks on this stack | Yes | **None expected: about 0 ms/step.** Changes only how PyTorch reserves memory, and kernels are unaffected. With 432 GiB per GPU, fragmentation has plenty of room. Not measured, because the revert fails at start-up. *Confidence: medium.* | **No.** With it set, training runs out of memory at start-up: allocating 64.85 GiB fails while 430.94 of 432 GiB is free. A standalone test tops out at 19 GiB with it set; with it cleared, 300 GiB allocates fine. Run folder: `rv_alloc_gbs4096_s600_20260927T064634Z`. | ROCm / PyTorch: fix expandable segments on gfx1250 |
| T9 | Attention, LayerNorm and dropout kernels ported off `tl.make_block_ptr` (commits `a7c30bb`, `6cace10`) | Newer Triton removed block pointers | Yes | **Low: est. < 1% of step, unmeasured.** Before Triton 3.8, the AMD backend's first compiler pass (`rewrite_tensor_pointer`, checked in the 3.4 and 3.5 sources) already turned block pointers into plain pointer arithmetic. The port therefore mostly reproduces what the compiler did. The remaining difference, 32- vs 64-bit offset math, is covered by T4. It cannot be A/B-tested, because the old API does not compile on 3.8. *Confidence: medium-low.* | **No.** Triton 3.8 no longer has `make_block_ptr`, so there is nothing to revert to. | Measure after re-autotune |

How the revisit was done: HSTU attention forward and backward called directly, exactly as in training. Batch 1,024, jagged lengths with mean fill 0.51 (2.16 M tokens), strided q/k/v views, sort-by-length on, one target per sequence. GPU 0, same image. Each configuration was first compiled without launching, to read its VGPR use, spills and memory instructions. Only configurations whose VGPR use is close to or below production's (654; the highest run was 685) were then run, one per process, and checked against production's output with kernel logs captured. All timed runs finished with no NaN or Inf and no GPU faults. Timings are per call, one HSTU layer; production was re-measured three times (109.9–110.0 ms). Scripts and logs are in the host folder `~/mi450_perf_20260926/triton_revert/`.

#### 4.3.1 Reverted on this host

Changes from the table above that were reverted and checked in end-to-end training on this host (R1 carries a stability caveat, below). The summary (§1), §4.4 and the §4.5 ladder use the step time after the revert. The per-component §4.1 breakdown comes from the run *before* the revert (`BLOCK_N` 64); only A4 changed.

| # | Change reverted | Why it was originally made | How it was reverted | Validation on this host (2026-09-27) | Performance gain | Remaining risk and follow-up |
|---|---|---|---|---|---|---|
| R1 | Attention **backward `BLOCK_N` 128 → 64** on gfx1250 with Triton ≥ 3.8 (commit `3e78d97`), now back to **128** | At `BLOCK_N=128` the backward uses 986 VGPRs (no spills). An address value held in an **extended VGPR** (index > 255) was corrupted and the kernel faulted on a bad store address. Recorded on the MI450 A0 bring-up: the standalone repro faulted every run, around iteration 125–150, and training faulted by batch 527. | Run-time override `HSTU_BWD_BLOCK_N=128`; everything else as in the measured run (4 warps, Triton 3.8.0+git7ff97e31). The pinned default in code is still 64. The compiled kernel was checked in the Triton cache: 986 VGPRs, 4 warps. | Full run, 4 GPUs, local batch 1,024, **600 steps** (~1,800 calls of the 986-VGPR kernel per GPU). No GPU fault, oops or reset in dmesg; no NaN. Loss identical to the `BLOCK_N` 64 run at steps 100, 150 and 200 (0.13918, 0.13856, 0.13895); 0.13791 at step 600. Run folder: `~/mi450_perf_20260926/bn128_gbs4096_s600_20260927T034620Z/`. A second 600-step run with pipelining on (T2) was also clean. **Later on 2026-09-27, 3 of 3 R1 runs faulted.** They were launched 15–16 h after boot, one with the exact `bn128` config. Each died on a GPU memory fault, named as `_hstu_attn_bwd` in two of the runs, shortly after step 100, with no NaN. Their losses matched the clean runs at steps 50 and 100 ([host record §8](ctheliosp-1b112-a37-1.md#8-2026-09-27-nan--fault-repro-r1-faults-and-cwsr-exposure)). | Attention backward **283.1 → 155.8 ms/step (−45%)**. Profiled step (steps 52–56) **1,860 → 1,735 ms (−6.7%)**. Wall step time at steps 100–200 (same token fill) 2,004 → 1,862 ms (−7%). Safer 128 variants measured earlier were slower than 64: 8 warps 123.8 ms per call, capped at 256 VGPRs 185.1 ms. | **The fault reproduces on this host. Whether a run survives depends on the host's memory state.** The likely mechanism is the known-faulty gfx1250 CWSR trap handler (`0f718b5e…`). Kernel memory compaction forces KFD queue evictions: 1,569 of 1,580 in one traced run. Each eviction saves and restores the 986-VGPR waves through that handler, which can corrupt an extended-VGPR address. This is not proven: there has been no run with the corrected handler, and a compiler-side cause is not excluded. **Until the corrected handler (`68c31ab2…`) is installed**, choose one: run R1 only on a freshly booted host, reduce compaction first, or use `BLOCK_N` 64. Before making 128 the default: the handler fix, then a longer soak (thousands of steps or a full convergence run) and the standalone repro for 500+ iterations. **Driver / KFD owners**: ship the corrected CWSR handler. **LLVM AMDGPU compiler team**: re-check the extended-VGPR corruption with the fixed handler (repro `scripts/repro_gfx1250_attn_bwd.py`). |

### 4.4 Reference: the MI350X trace the tool was calibrated on

The Primus source (`project_dlrm.py`) records a measured MI350X step for a close configuration: 8 GPUs, local batch 1,024, max seq 3,650, fill 0.6, **2.23 M tokens per GPU** (ours: 2.08 M). It is a "flydsl" run, so some kernels may differ from our Triton build. Treat this as a rough reference only.

| Component | MI350X reference ms/step | MI450 measured ms/step (current, after R1) | MI450 ÷ MI350X |
|---|---:|---:|---:|
| Dense GEMMs | 132 | 1,190 | 9.0× slower |
| Attention (fwd + bwd) | 131 | 202 (330 before R1) | 1.5× slower (2.5× before R1) |
| LayerNorm / dropout / elementwise / glue | 158 | 85 | faster |
| Embedding | 52 | 25 | faster (ID dedup) |
| Exposed collectives | 0.9 | 25 | 29× |
| **Step** | **472** | **1,735** (1,860 before R1) | **3.7× slower** (3.9× before R1) |

MI450 has more compute and memory bandwidth than MI350X, so each MI450 row should be *faster*. The GEMM and attention rows are where MI450 software currently falls short.

### 4.5 What fixing the MI450 stack would give (estimate)

| Fix (cumulative) | Est. step ms | Est. samples/s | vs today |
|---|---:|---:|---:|
| Before R1 (profiled, `BLOCK_N` 64) | 1,860 | 2,202 | 0.93× |
| **Today** (profiled, after R1) | **1,735** | **2,361** | **1.0×** |
| hipBLASLt fast kernels for all layouts (A1–A2, ~1.8 PF) | ~599 | ~6,840 | 2.9× |
| + pinned, asynchronous input copy (A6) | ~399 | ~10,270 | 4.3× |
| + attention at the projected efficiency (A4–A5) | ~258 | ~15,900 | 6.7× |
| Projection | 210 | 19,538 | 8.3× |

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
| B7 | **Attention model** | Causal FLOPs × fixed efficiency **0.246**, backward/forward **2.03**, both fitted on MI350X. The kernel-level `fav3_hstu` model needs the `origami` backend, which is not available. | Triton HSTU kernel on gfx1250: forward efficiency ≈ **0.11**, backward/forward ≈ **6.1** (**3.3** after R1) | Projection assumes MI350X attention efficiency on MI450 | Per-arch calibration, or a model of the actual Triton kernel; ship `origami` or make `fav3_hstu` work with gemmologist | projection tool |
| B8 | Sequence-length distribution | One mean fill + one std | Jagged lengths per sample. We derived std = 0.24 indirectly ([§3](#3-workload-configuration)). | Attention cost depends on E[L²]; an error here shifts attention time | Accept Σ L and Σ L² (or a histogram) per batch as input | projection tool |
| B9 | **Host-to-device input copy** | Not modelled; only accepts a measured value (`--h2d-ms`) | 200 ms/step (A6) | Blind to 12% of today's step | Model the input pipeline: bytes per step, pinned vs pageable, overlapped or blocking | projection tool |
| B10 | Embedding sharding and ID dedup | Row-wise sharding only; no ID deduplication; every lookup costs a full row | 8 table-wise + 3 column-wise tables; IDs deduplicated, so only ~50 k unique rows per step are exchanged, not 6.4 M | The projected 27 ms is close to measured by coincidence; it will not track changes | Support table-wise / column-wise plans and a dedup ratio | projection tool |
| B11 | **Collectives** | Embedding all-to-all only, with xGMI bandwidth. Exposed fraction 0 and 0.86 ms "sync" copied from MI350X. | Also a dense-gradient all-reduce (15 M FP32 params, 32 ms kernel time); Socket transport over loopback; 25 ms exposed | Collectives under-projected by ~24 ms here; far more at multi-node scale | Add the dense all-reduce, a transport choice (Socket / xGMI / RoCE) and an overlap model | projection tool |
| B12 | Dense optimizer, idle, launch overhead | Not modelled | ~7 ms/step | Small | Add an optimizer term | projection tool |
| B13 | Defaults tied to one MI350X trace | Defaults: 8 GPUs, global batch 8,192, max seq 3,650, fill 0.6, and MI350X-fitted efficiencies. `--trace-roles` prints hard-coded MI350X numbers as "measured". | Our run needs every workload flag in [§2.2](#22-projection-tool-used) overridden | Easy to project the wrong workload by accident | Separate workload defaults from per-arch calibration; add a trace-import mode that fills the measured column from a real trace | projection tool |

## 6. Limits of this analysis

- **Single run.** The per-component breakdown comes from one 200-step run, all within LR warm-up. All measured numbers are rank 0 over the 5 profiled steps 52–56 (1,860 ms/step). The current step time (1,735 ms) comes from the same steps of the 600-step R1 run; there only the attention backward changed. Step time varies from step to step with the jagged token count.
- **Microbenchmark conditions.** Microbenchmarks ran on an idle GPU 0 in a plain container. The weight-gradient rows use M = 1,048,576 tokens instead of ~2.1 M, because the plain container could not allocate more than ~20 GiB. TFLOP/s rates at this size are independent of M.
- **H2D root cause is open.** Why the in-training pageable copy runs at 1 GB/s instead of 73 GB/s is not known yet.
- **Tool calibration.** Several tool defaults are fitted to the MI350X trace, so the 210 ms projection is a calibrated estimate, not a pure first-principles number.
- **Estimates.** The §4.5 ladder is an estimate, not a measurement.
- **Revert checks are single runs.** Each §4.3 end-to-end check (R1, T2, T3, T8) is one run on one host (planned for 600 steps). A clean run lowers the risk but does not prove the underlying compiler bug is gone. One failure (T3) is enough to keep a workaround. R1 later faulted in 3 of 3 runs launched 15–16 h after boot. It is kept in the current configuration for performance, with the stability caveat in [§4.3.1](#431-reverted-on-this-host).

## 7. Files

| File | Content |
|---|---|
| [`traces/trace_step52.json.gz`](traces/trace_step52.json.gz) | Stitched 4-GPU trace, steps 52–56, used for every measured number above |
| [`ctheliosp-1b112-a37-1.md`](ctheliosp-1b112-a37-1.md) | Host, software stack and run record |
| [`hipBLASLt/`](hipBLASLt/README.md) | Request to the hipBLASLt team: all 33 training GEMMs as a `hipblaslt-bench` YAML, component map, per-GEMM timings |
