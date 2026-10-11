# MI450 Performance Optimizations

Performance work on the DLRM-v4 HSTU ranker (MLPerf yambda-5b) on **MI450 (gfx1250)**: where the
platform stands now, and each optimization applied to get there. Companion to
[`perf_opt.md`](perf_opt.md) (MI350X / B200) and to the host records (stack, hazards, TODOs):
[`ctheliosp-1b112-a37-2`](mi450_1P4G/ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md) (M1–M4, 2026-10-04) and
[`ctheliosp-1b112-a37-1`](mi450_1P4G/ctheliosp-1b112-a37-1/ctheliosp-1b112-a37-1.md) (M1 reproduction, T8, M5, 2026-10-10/11).

- **[Latest evaluation](#latest-evaluation)**: current measured end-to-end status.
- **[Blocked](#blocked-risky-operations-and-who-can-fix-them)**: the risky operations that block the next ~40 % of the step, and the likely fix owners.
- **[Optimization log](#optimization-log)**: one self-contained row per incremental optimization,
  each with its own A/B and the gin switch that turns it on.

**Throughput metric.** Global samples/s (`global_sps`, 4 ranks × local batch 1,024) and step time
(`step_ms`), both as the **median of the per-step logged values over steps 20–200**. This is the
MI350X headline rule in `perf_opt.md`. The trainer already excludes eval and checkpoint time.
Runs are 200 optimizer steps from `START_TS=0` (`DIE_AT_STEP=200`). No convergence result is
claimed here.

**Every optimization is a gin switch, all OFF by default** in
`generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin`. A run turns them on with a gin overlay
named by `$DLRM_GIN_OVERLAY`, parsed after the main gin. Ready-made overlays are in
`gin/mi450_hipblaslt/`: `m0_off`, `m1_lib`, `m2_override`, `m3_tn_hstu`, `m4_all`, `m1_t8_lib` (M1+T8)
and `m5_split_t` (M1+T8+M5). The hipBLASLt library dirs they name are `arbor_libs/rocm-10.2_nnonly_v{1,2,3}`
(mounted at `/armlibs`). The code is on the canonical branch
[`chcai/mi450`](https://github.com/chriscai-amd/training/tree/chcai/mi450), which merged the former
`chcai/mi450_hipblaslt_opt` in [`af6f033`](https://github.com/chriscai-amd/training/commit/af6f033).

---

## Latest evaluation

One row per host × node count, latest only. The two 1P4G hosts differ in firmware and driver, so
their numbers are not directly comparable.

| date | commit | hardware / host | nodes / GPUs | software stack | run config | throughput (median steps 20–200) | trace | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11 | [`7b53096`](https://github.com/chriscai-amd/training/commit/7b53096) on `chcai/mi450` (M5 [`a646eb1`](https://github.com/chriscai-amd/training/commit/a646eb1), T8 [`77c865f`](https://github.com/chriscai-amd/training/commit/77c865f), M1–M4 merged in [`af6f033`](https://github.com/chriscai-amd/training/commit/af6f033)); `BLOCK_N=128` is an env setting | 4× **MI450** (gfx1250, 256 CUs, 432 GiB HBM each, one XGMI hive)<br>host `ctheliosp-1b112-a37-1`<br>VBIOS `113-M4500001-900A` (not BKC 26.11.08), MES/KIQ fw `0x7e`, MEC `0x97e`<br>**amdgpu-dkms `7.1.0-2416495.el10` (CWSR handler fixed**, installed 2026-10-11; firmware package 2404163), kernel `6.16.1-0_fbk5`<br>amdgpu options `ip_block_mask=0xcff mes_log_enable=1 dyndbg=+plmf`; `noretry` and `gpu_recovery` at defaults | 1 / 4 | image `recommendation-gfx1250-20260910:triton-7ff97e` (`fc5040f7`, built 2026-09-26)<br>torch `2.11.0+rocm7.14.0a20260625`, triton `3.8.0`, torchrec `1.7.0a0+bf55480`, fbgemm_gpu nightly `2026.9.26`<br>stock hipBLASLt: `rocm-sdk-libraries-gfx1250 7.14.0a20260625`<br>**M1/T8/M5 hipBLASLt: host `/opt/rocm-10.2` (10.2.0, libhipblaslt 1.5, git `5911365c`, 2026-09-25)** via `arbor_libs/rocm-10.2_nnonly_v3/s103` | yambda-5b, MLPerf gin defaults, full model (11 tables, 3 HSTU layers, seq 4096)<br>local batch 1,024 / global 4,096, `START_TS=0`, 200 steps, seed 1<br>**`HSTU_BWD_BLOCK_N=128`**, `HSTU_BWD_MAX_VGPR=0`, `AMDGCN_USE_BUFFER_OPS=0`, `TRITON_FULL_AUTOTUNE=0`, allocator vars empty, eval off, `METRIC_LOG_FREQ=1`<br>gin overlay **`m5_split_t.gin`** (M1 + T8 + M5) | **8,415 global_sps, 486.8 ms/step** (ABBA mean of 486.3 / 487.2)<br>vs **2,977 / 1,376** all opts off (M0, 2026-10-10, same host and image, old driver 2404163)<br>**2.83× throughput, −65 % step time** | M0/M1 traces 2026-10-10 (`OUTPUT_TRACE=1`, 5 steps from step 52): M1 rank 0 = 1,021 ms GPU/step, of which hipBLASLt 525, HSTU attention 327. No trace of the current stack: since 2026-10-11 the torch profiler records no or partial GPU events on this host. | Stack is M1 + T8 + M5 + `BLOCK_N=128`, each A/B'd on its own (rows below). Long-run validation in progress: the first convergence run on the fixed driver went 11,581 steps fault-free, then the host crashed (fleet monitoring job, see the a37-1 record §12.8); no AUC yet. Loss at step 200 matches across all arms (0.138956 ± 1e-7). |
| 2026-10-04 | [`be2d249`](https://github.com/chriscai-amd/training/commit/be2d249) on `chcai/mi450_hipblaslt_opt` (trunk [`e4c612f`](https://github.com/chriscai-amd/training/commit/e4c612f) + gin-guarded MI450 opts) | 4× **MI450** (gfx1250, 256 CUs, 432 GiB HBM each, one XGMI hive)<br>host `ctheliosp-1b112-a37-2`<br>firmware **BKC 26.11.08**, VBIOS `113-M4500001-700E`<br>amdgpu-dkms `7.1.0-2413386.el10`, kernel `6.16.1-0_fbk5`<br>amdgpu loaded with `noretry=0 gpu_recovery=0 ip_block_mask=0xcff` | 1 / 4 | image `recommendation-gfx1250-20260910:triton-7ff97e` (`ea5cee73`, rebuilt 2026-10-04)<br>torch `2.11.0+rocm7.14.0a20260625`, triton `3.8.0+git7ff97e31`, torchrec `1.7.0a0+bf55480`, fbgemm_gpu nightly `2026.10.4`<br>stock hipBLASLt: `rocm-sdk-libraries-gfx1250 7.14.0a20260625`<br>**M1 hipBLASLt: ROCm 10.2 nightly `10.2.0a20260929`** `libhipblaslt.so.1.5`, NN Origami pool, NT catalog filtered to the stock stub | yambda-5b, MLPerf gin defaults, full model (11 tables, 3 HSTU layers, seq 4096)<br>local batch 1,024 / global 4,096, `START_TS=0`, 200 steps, seed 1<br>`HSTU_BWD_MAX_VGPR=0`, `AMDGCN_USE_BUFFER_OPS=0`, `TRITON_FULL_AUTOTUNE=0`, allocator vars empty, eval off<br>gin overlay **`m1_lib.gin`** (M1 on, M2–M4 off) | **2,841.1 global_sps, 1,441.7 ms/step**<br>vs **2,092.4 / 1,957.5** with all opts off (same host, boot, image, data)<br>**+35.8 % throughput, −26.3 % step time** | **pending**: the cumulative trace (`OUTPUT_TRACE=1`, 5 steps from step 52) needs a host power cycle first, because GPU 1 has been unusable since the M3 fault at 09:02 | Safe cumulative stack = **M1** (+M2, which is inert in training). M3/M4 are **unsafe in training** (GPU fault; see the log). One run per arm, not yet ABBA-repeated; the repeat was cut short by the M3 fault. Final loss at step 200 is identical across M0/M1/M2 (`0.13897`). |

---

## Blocked: risky operations and who can fix them

**Update 2026-10-11/12 (a37-1).** T8 + M5 took the six tall-K weight-gradient GEMMs from ~426 to
~29 ms/step using only the *stock* kernel sol 103, so the 10.2 tall-K kernels below are no longer
needed for them. What now blocks further gains and production use:

| Blocker | Effect | Seen | Likely fix owner | Unblocks when |
|---|---|---|---|---|
| ~~**Faulty CWSR trap handler** in amdgpu-dkms 2404163~~ **Resolved on a37-1 2026-10-11:** amdgpu-dkms 2416495 loads on VBIOS 900A with the fixed handler | `BLOCK_N=128` is back (+28.6 %, row M6). T8/M5 (sol 103) remain to be proven over a full convergence run. | resolved | — | A completed convergence run on 2416495 (in progress) |
| **GPU faults on host (SVM) memory during training**: GPU access to host pages the driver has just invalidated (~11 `svm_range_cpu_invalidate_pagetables` per second in normal training) | 2026-10-10: no-retry fault, `REMOVE_QUEUE` failed, queue reset, GPU 1 stuck until power cycle. 2026-10-11: retry faults, then a process memory-access fault. Plain M1 config: 1 fault in 5 runs on 10-10 (default `noretry`), 1 in 3 with `noretry=0`, then 0 in 17 runs (8 repro + 9 A/B arms) after the 10-11 reboot | 2026-10-10, 2026-10-11 | **KFD (SVM)**; recovery path with **MES / amdgpu** (ticket open, logs sent) | Root cause from KFD; MES confirms the reset escalation |
| Fault recovery | A stuck GPU after a queue reset is not escalated to a mode reset; on a37-2 a mode-2 reset hung SMU and wedged the host. Every fault costs a power cycle. | 2026-10-04, 2026-10-10 | **MES firmware / amdgpu / SMU (BFD)** | Working queue → mode reset escalation |
| **Host hard crash during training** (a37-1, 2026-10-11 20:09:48) | Lost a convergence run at step 11,581; no panic/kdump/SEL record. 1 s after the fleet monitoring job `amd_device_data.cli` wrote a kernel-exclusive host-bridge PCI config register | 2026-10-11 | **Fleet tooling (`orch`)**, host admin (kdump/SEL) | Job paused or fixed on GPU test hosts; crash evidence enabled |

The table below is the 2026-10-04 a37-2 state, still valid for the 10.2 tall-K kernels themselves.
Before T8/M5, after M1, **~625 ms/step (91 %) of the remaining hipBLASLt time** was six tall-K weight-gradient
GEMMs (K = T ≈ 2 M): G20, G19, G25, G29, G31, G27. That was **~40 % of the 1,442 ms step**, and
their combined lower bound is ~18 ms. Every route to speed them up with 10.2 kernels faulted a GPU.
Details and dmesg are in the host record:
[§0.3](mi450_1P4G/ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md#03-risky-operations-on-this-host)
and [§9](mi450_1P4G/ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md#9-hipblaslt-gemm-work-and-the-2026-10-04-host-hang).

| Blocked optimization | Risky operation | What happened (2026-10-04, a37-2) | Likely fix owner | Unblocks when |
|---|---|---|---|---|
| Tuned NT weight-gradient kernels (the direct fix for the 625 ms) | ROCm 10.2 **NT** stream-K kernel on K = T | MES hang; the GPU reset then hung in SMU, and the **whole host wedged** (reboot) | **hipBLASLt / TensileLite** (kernel fault); **SMU firmware / amdgpu (BFD)** (reset hang) | Fixed tall-K NT kernels in a gfx1250 hipBLASLt build |
| NT → NN re-expression of the weight gradients | 10.2 **NN** kernel on K = T | Page fault + MES `REMOVE_QUEUE` failure; GPU lost until power cycle | **hipBLASLt / TensileLite** | Same fix as above |
| **M3 / M4**: NT → TN re-expression (bench 1.08–1.15× each) | 10.2 **TN** tall-K kernels under jagged training T | Clean in the bench at fixed T, but **faulted GPU 1 in step 1 of training**: page fault, MES failures, kworkers in D state | **hipBLASLt / TensileLite** | Same fix; then also remove the transpose copies (287 ms/step in the bench) |
| Tuning sweeps, stream-K / split-K / GSU knobs on these shapes | Any run that selects a 10.2 tall-K kernel | The same fault class: a sweep has to execute the faulting kernels | **hipBLASLt / TensileLite** | Same fix |
| Fast iteration on any of the above | Recovering a faulted GPU without a reboot | The reset is hive-wide (all 4 GPUs, one XGMI hive) through the SMU mode-2 path that hung; **each fault costs a host power cycle** | **SMU firmware / amdgpu driver (BFD)** | A working per-GPU or hive reset |

Our side meanwhile (host record §0.4: T1–T3 go to the owning teams, T6–T7 are ours):
1. Send minimal repros, the single-GEMM command plus dmesg, to hipBLASLt and to BFD.
2. Keep M3/M4 gin-off.
3. Work on the transpose-free weight-gradient path in model code.

---

## Optimization log

One row per incremental optimization, **each with its own A/B**: same host, boot, image and
dataset; arms run back to back; the stated baseline is enabled. The first arm (M0) read the
227 GB dataset cold. Bench numbers come from the Arbor campaign `mi450-hipblaslt`: a paired
33-GEMM hipBLASLt bench at T = 2,097,152, same GPU, ABBA, measuring hipBLASLt ms/step. End-to-end
numbers come from the 200-step training A/B above.

Rows dated 2026-10-10 and later are on **`ctheliosp-1b112-a37-1`** with the stack in its
[Latest evaluation](#latest-evaluation) row (VBIOS 900A, amdgpu-dkms 2404163, host `/opt/rocm-10.2`
hipBLASLt, image `fc5040f7`, `HSTU_BWD_BLOCK_N=64`). Their A/Bs are ABBA, 200 steps per arm, run
back to back with [`a37-1_kit/run_ab.sh`](mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/a37-1_kit/run_ab.sh)
(live GPU-fault watch, MES evidence capture). Their GEMM column is single-GEMM `hipblaslt-bench`
timings on one GPU (one GEMM per process) at T = 2,097,152 unless stated, not the Arbor bench.

| date | main optimization | how it is applied (gin) | main commits | GEMM bench (hipBLASLt ms/step) | e2e perf (median steps 20–200) | notes |
|---|---|---|---|---|---|---|
| 2026-10-04 | **M1: hipBLASLt library swap, NN only.** ROCm 10.2 nightly `libhipblaslt.so.1.5` with its gfx1250 bf16 **NN Origami pool** (rocm-libraries #12271, ~425 stream-K tuned kernels). The image's 7.14 library has a 2-kernel stub there, which picks `MT32x16x32`. Its **NT catalog is filtered back to the stock stub** (sol 102/103), so no 10.2 NT tall-K kernel is reachable. | `apply_launch_env.hipblaslt_lib_dir = "/armlibs/rocm-10.2_nnonly_v1"` (overlay `m1_lib.gin`). The launcher process sets `LD_PRELOAD`, `LD_LIBRARY_PATH` and `HIPBLASLT_TENSILE_LIBPATH` before the ranks spawn. The container must mount the lib dir at `/armlibs` and `/opt/rocm-10.2` read-only. | [`86b88c1`](https://github.com/chriscai-amd/training/commit/86b88c1) (Arbor iter 5), gin guard [`5c1f080`](https://github.com/chriscai-amd/training/commit/5c1f080). Lib-dir manifest: Arbor `results/mi450-hipblaslt-a37-2/runs/mi450-hipblaslt/arbor_libs_rocm-10.2_nnonly_v1/` | **1,154.1 → 684.9 (1.685×)**, rounds 1.6849 / 1.6853; NN group 500 → 35 ms/step (UVQK fwd 349 → 25, out-proj fwd 128 → 6.6) | **2,092.4 → 2,841.1 global_sps (+35.8 %)**; step **1,957.5 → 1,441.7 ms (−26.3 %)**. Runs `1004_M0a` → `1004_M1`. | **Armed in training:** every rank maps `/opt/rocm-10.2/lib/libhipblaslt.so.1.5`; a G08-shaped `addmm` logs sol **951** (10.2 `MT256x256x64` SK3) vs **104** stock. Numerics: bench max rel err 1.66e-3 (unchanged); loss at step 200 identical (0.13897). Ship path: a newer `_rocm_sdk_libraries_gfx1250` wheel instead of an `LD_PRELOAD`. |
| 2026-10-04 | **M2: hipBLASLt tuning override.** Pins the NT weight gradients to stock sol 103 `MT256x128x64` (from 102 `MT32x16x32`) and the TN re-expressed keys to sol 1409. | `apply_hipblaslt_opts.tuning_override_file = ".../arbor_hipblaslt/tuning_override.csv"` (overlay `m2_override.gin`), exported as `HIPBLASLT_TUNING_OVERRIDE_FILE` per rank | [`c56bda3`](https://github.com/chriscai-amd/training/commit/c56bda3) (iter 6), [`80d0e62`](https://github.com/chriscai-amd/training/commit/80d0e62), [`bfa73ce`](https://github.com/chriscai-amd/training/commit/bfa73ce) | **684.7 → 502.7 (1.362×)** at the bench's fixed T | **No e2e gain:** 2,841.1 → 2,819.8 global_sps (−0.75 %, within run spread); 1,441.7 → 1,452.6 ms. Runs `1004_M1` → `1004_M2`. | **Expected:** override rows match exact `(m, n, k)`, and every row has K = 2,097,152. Training's T is jagged (1.59–2.53 M), so no row ever matches. To make it real, move the pin into the library catalog by size range (TODO). Kept off by default. |
| 2026-10-04 | **M3: TN weight gradients for the HSTU UVQK / output projections** (G20 2048×512×T, G19 512×1536×T). The NT `x.t() @ dz` becomes `mm(x.t().contiguous(), dz.t().contiguous().t())`, served by the tuned TN pool. | `apply_hipblaslt_opts.tn_wgrad_hstu = True` (overlay `m3_tn_hstu.gin`) → `HSTU_TN_WGRAD=hstu` | [`5e62ca3`](https://github.com/chriscai-amd/training/commit/5e62ca3) (iter 7), scopes [`e1b5a77`](https://github.com/chriscai-amd/training/commit/e1b5a77) | 502.8 → 465.4 (1.080×). The GEMMs drop to ~11 ms/step; the transpose copies add ~174 ms/step. Not claimable: no recapture yet | ❌ **FAULT, not measured.** Run `1004_M3` faulted **GPU 1 in its first training step**: no-retry page faults → MES `REMOVE_QUEUE`/`INVALIDATE_TLBS` failures → "Memory access fault by GPU". The launcher killed it; the GPU stays unusable until a power cycle. | **Unsafe in training; keep off.** In the bench (T = 2,097,152) the 10.2 TN tall-K kernels sol 1397/1409 ran cleanly. Under real jagged T (up to 2.53 M) the library picks other 10.2 TN tall-K solutions or larger operands, the same fault class as the 10.2 NN/NT tall-K hangs. Needs a hipBLASLt fix (host record TODO T1). |
| 2026-10-04 | **M4: TN weight gradients for the preprocessor MLP Linears** (G25 256×512×T ×3, G27/G29/G31 24/1024/512×256×T). `nn.Linear` → `TNWgradLinear`. | `apply_hipblaslt_opts.tn_wgrad_preproc = True` (overlay `m4_all.gin` = M1 + M2 + M3 + M4) → `HSTU_TN_WGRAD=all` | [`80d0e62`](https://github.com/chriscai-amd/training/commit/80d0e62) (iter 10), [`bfa73ce`](https://github.com/chriscai-amd/training/commit/bfa73ce) (iter 11) | 468.2 → 421.2 (1.112×), then 420.8 → 365.0 (1.153×). Not claimable: no recapture yet | **Not run.** Same 10.2 TN tall-K kernel class as M3. | **Unsafe in training; keep off** (same reason as M3). Bench cumulative M1..M4 = 1,154.1 → 364.6 ms/step (3.16×), of which 287 ms/step is transpose copies. |
| 2026-10-10 | **M1 reproduced on a37-1** (no code change; same `rocm-10.2_nnonly_v1` lib dir, built on a37-1's own `/opt/rocm-10.2`, whose NT stub sol 102/103 and GridBased table are byte-identical to a37-2's filtered catalog) | overlay `m1_lib.gin` (as above) | [`86b88c1`](https://github.com/chriscai-amd/training/commit/86b88c1), [`5c1f080`](https://github.com/chriscai-amd/training/commit/5c1f080) | All 33 model GEMMs, one per process, on all 4 GPUs, T = 2,097,152 and 2,533,005: no faults. NN fwd e.g. 2048×T×512 sol 951 4.09 ms | **ABBA: M0 2,970.4 / 2,983.3 → M1 3,843.5 / 3,833.4 global_sps (+28.9 %)**; step **1,378.9 / 1,373.0 → 1,065.7 / 1,068.5 ms (−22.4 %)**. Runs `M0a`, `M1a`, `M1b`, `M0b`. Repeated 2026-10-11: M1 1,096.5 / 1,098.0 ms. | Loss at step 200 identical (0.138956). Traces (rank 0 GPU ms/step): hipBLASLt 822.7 → 525.4; the NT wgrads on sol 102 (387) became the top item. **A 5th M1 run faulted GPU 1 at step 10** (host-memory page fault, `REMOVE_QUEUE` failed, GPU stuck until power cycle; see the a37-1 record §9). **`noretry=0` makes the long-K NT GEMMs 8–81 % slower** (G19 31.9 → 57.5 ms) and M1 ~1,700 ms/step: keep `noretry` at its default. |
| 2026-10-11 | **T8: size-range NT catalog.** Moves M2's pin into the library catalog: 19 GridBased entries (5 targets + guards) in the bf16 NT `.dat` so the six tall-K weight gradients (G20 2048×512, G19 512×1536, G25 256×512, G29 1024×256, G31 512×256, G27 24×256) select stock **sol 103** `MT256x128x64` at **any** jagged T instead of sol 102 `MT32x16x32`. No other shape changes (simulated over 132k shapes, 0 mismatches vs a compiled copy of the C++ matcher). | `apply_launch_env.hipblaslt_lib_dir = "/armlibs/rocm-10.2_nnonly_v2"` (overlay `m1_t8_lib.gin`) | [`77c865f`](https://github.com/chriscai-amd/training/commit/77c865f) (catalog, builder, report), overlay [`7b53096`](https://github.com/chriscai-amd/training/commit/7b53096) | Per call, sol 102 → 103: G20 57.0 → 24.0, G19 31.9 → 24.1, G29 29.7 → 23.4, G31 29.6 → 23.6, G27 28.4 → 22.8 ms. NT wgrads ~426 → ~288 ms/step. Numerics: 1000 zero-DY calls exact +0 and 200 random calls bit-identical (rel err ≤ 3.2e-3 vs fp32) per shape | **ABBA: M1 3,735.4 / 3,730.4 → M1+T8 4,433.0 / 4,423.8 global_sps (+18.6 %)**; step **1,096.5 / 1,098.0 → 924.0 / 925.9 ms (−15.7 %)**. Runs `M1e`, `M1T8a`, `M1T8b`, `M1f`. | Loss identical to 7 digits, no NaN, no faults. Sol 103 calls go from 3 to 12 per step: **sol 103 is the A0 DW-corruption kernel** under the faulty CWSR handler (A0: zero-DY failed at call 196; sol 102 passed 2×1000). Guarded ≤200-step A/Bs only on 2404163. Report: [`t8_catalog/T8_REPORT.md`](mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/t8_catalog/T8_REPORT.md). Supersedes M2. |
| 2026-10-11 | **M5: split-T weight gradients.** Sol 103 runs each tall-K NT wgrad on only 2–32 workgroups (no split-K). M5 views the T rows as s zero-copy chunks and issues **one strided-batched NT GEMM** (s× the workgroups, same stock kernel), sums the bf16 partials in fp32 and adds the leftover rows with an fp32 tail GEMM. Splits: G20 s=16, G19 s=10, G25/G29/G31 s=64, G27 s=128. The v3 catalog pins the batch-flattened GridBased keys to sol 103. | `apply_hipblaslt_opts.splitt_wgrad_hstu/_preproc = True`, `splitt_wgrad_splits = {...}`, lib dir `/armlibs/rocm-10.2_nnonly_v3/s103` (overlay `m5_split_t.gin`). Off = bit-identical. Import checks every s against the lib dir's `m5_splits.json`. | [`a646eb1`](https://github.com/chriscai-amd/training/commit/a646eb1) (code, v3 catalogs, tests, kit), splits [`7b53096`](https://github.com/chriscai-amd/training/commit/7b53096) | At T = 2,533,005 (largest observed), per call unsplit → split: G20 28.9 → 4.64, G19 29.1 → 3.57, G25 28.6 → 0.74, G29 28.2 → 1.29, G31 28.4 → 0.73, G27 27.6 → 0.35 ms. **NT wgrads ~342 → ~29 ms/step** (206 bench GEMMs, all on the expected kernel, no faults) | **ABBA: M1+T8 4,435.7 / 4,440.1 → M1+T8+M5 6,397.7 / 6,371.2 global_sps (+43.8 %)**; step **923.4 / 922.5 → 640.2 / 642.9 ms (−30.5 %)**. Runs `M1T8_A1`, `M1T8M5_B1`, `M1T8M5_B2`, `M1T8_A2`. | Loss matches (0.138956 ± 1e-7), no NaN, no faults. **Numerics:** bf16 partials cost ~1.41× the unsplit rounding error on independent chunks, more if chunk partials cancel. GPU checks per shape: 1000 zero-DY calls exact; 200 random calls bit-identical, 1.40–1.42× as modeled; synthetic worst-case cancellation 15–59× (model 108–497×), 0 bound violations. **In-training audit** (every 50 steps, all shapes): real cancellation q ≤ 1.13 → error ≤ 1.5× unsplit, all finite. Same sol-103 caveat as T8. Report: [`m5_split_t/M5_REPORT.md`](mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/m5_split_t/M5_REPORT.md). |
| 2026-10-11 | **M6: HSTU attention backward `BLOCK_N` 64 → 128** (the earlier "R1"), now that a37-1 runs the CWSR-fixed driver. At `BLOCK_N=128` the Triton `_hstu_attn_bwd` uses 986 VGPRs; under the faulty handler it faulted 3/3 at long uptime (a37-1 record §8), so it was pinned to 64. Not a hipBLASLt change. | Env `HSTU_BWD_BLOCK_N=128` (read in `ops/triton/triton_hstu_attention.py`; the gfx1250 default stays 64). Not yet a gin switch. | none (existing knob; the 64 pin is `3e78d97`) | n/a (Triton). Earlier a37-1 trace (2026-09-27, old stack): attention backward 283.1 → 155.8 ms/step | **ABBA on driver 2416495, on top of M1+T8+M5: 6,564.0 / 6,527.3 → 8,423.2 / 8,407.6 global_sps (+28.6 %)**; step **624.0 / 627.5 → 486.3 / 487.2 ms (−22.2 %)**. Runs `M5BN64_A1`, `M5BN128_B1`, `M5BN128_B2`, `M5BN64_A2`. | Loss at step 200 matches (0.1389564 ± 1e-7), no faults. **Requires a CWSR-fixed driver** (2416495 or later); never use it on 2404163. A convergence run with it reached step 11,581 fault-free before a host crash attributed to a fleet monitoring job (a37-1 record §12.8). |

### Not applied (recorded so they are not re-run)

| date | candidate | result | notes |
|---|---|---|---|
| 2026-10-04 | ROCm **10.1** hipBLASLt swapped into the image (Arbor iter 4) | 1.0001× in the bench (null) | Kernel selection changed on 30 GEMMs, yet the total did not move. The 1.56× seen when running the 10.1 stack **host-native** comes from the runtime, not from libhipblaslt (host record H6). |
| 2026-10-04 | 10.2 **NN with K = T** re-expression of G19/G20 (Arbor iter 8) | GPU 0 page fault + MES failure in the bench; reverted | The first tall-K fault inside the harness. |
| 2026-10-04 | Splitting the transpose copies into narrow column passes (Arbor iter 13) | REVERT | Did not beat the bar. The copies (287 ms/step in the bench) stay the top remaining cost. |
| 2026-10-11 | M5 on **sol 102** (`v3/s102`, the A0-safe 252-VGPR kernel) and the **mix** variant (G20/G19 on 103, rest on 102) | Bench only, not A/B'd | Splitting does not help sol 102 on G20/G19 (already 1,536–2,048 workgroups; 72.1 / 38.4 ms at T = 2.53 M); G25/G31/G27/G29 do improve (6.2 / 6.2 / 0.47 / 14.3 ms). Estimated s102 ≈ 906 ms/step, mix ≈ 700. Kept as fallbacks if sol 103 shows corruption (overlays `m5_split_t_s102.gin`, `m5_split_t_mix.gin`). |
| 2026-10-11 | amdgpu `noretry=0` (retry faults on), from the a37-2 bring-up recipe | Regression, removed | Long-K NT GEMMs 8–81 % slower; M1 ~1,700 vs ~1,090 ms/step; a fault became a retry-fault storm plus a process abort instead of a no-retry fault. |
| 2026-10-11 | M5 numerics via `bmm(out_dtype=float32)` (fp32 partials) | Not tried | Would select sol 1420 (1,020 VGPRs), never run on gfx1250 here. Unneeded: the measured real-gradient error is ≤ 1.5× with bf16 partials. |

### Reproduce

```bash
# host: amdgpu with the documented options; dataset at /home/chcai/data/mlperf_dlrm_v4
git clone -b chcai/mi450_hipblaslt_opt git@github.com:chriscai-amd/training.git
# kit: run_arm.sh + common.env + m*.env (the settings of host record §3, 200 steps, live GPU-fault watch)
bash run_arm.sh M0 <repo_root> m0_off.env    # all opts off
bash run_arm.sh M1 <repo_root> m1_lib.env    # DLRM_GIN_OVERLAY=.../gin/mi450_hipblaslt/m1_lib.gin
python3 analyze.py --base M0 runs/M0 runs/M1 # median global_sps / step_ms over steps 20-200
```

The kit (`/home/chcai/runs/mi450_e2e_ab/` on a37-2) is archived with the Arbor campaign in
`results/mi450-hipblaslt-a37-2/e2e_ab/`. Run logs: `runs/1004_M0a`, `1004_M1`, `1004_M2`, `1004_M3`.

**a37-1 (M1 reproduction, T8, M5).** Kit in
[`ctheliosp-1b112-a37-1/hipBLASLt/a37-1_kit/`](mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/a37-1_kit/);
`run_ab.sh` takes `ROOT` (repo `recommendation/`) and `LIBS` (repo `arbor_libs/`, mounted at `/armlibs`):

```bash
# host: strict preflight (driver build, options, VBIOS, no other GPU process)
./preflight.sh
# single GEMMs, one per process, stop at the first fault line (lib dir + union YAML lines)
./gemm_watch.sh <repo>/arbor_libs/rocm-10.2_nnonly_v3/s103 0 runs/bench <yaml> [lines...]
# e2e arms, 200 steps each, ABBA
./run_ab.sh M1T8_A1   m1_t8_lib.gin
./run_ab.sh M1T8M5_B1 m5_split_t.gin
./run_ab.sh M1T8M5_B2 m5_split_t.gin
./run_ab.sh M1T8_A2   m1_t8_lib.gin
python3 analyze.py runs/M1T8_A1_* runs/M1T8M5_B1_*   # median global_sps / step_ms over steps 20-200
```

T8 and M5 GPU checks (single GEMMs, zero-DY, random and cancellation numerics):
[`t8_catalog/gpu_check/`](mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/t8_catalog/gpu_check/) and
[`m5_split_t/gpu_check/README.txt`](mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/m5_split_t/gpu_check/README.txt).
