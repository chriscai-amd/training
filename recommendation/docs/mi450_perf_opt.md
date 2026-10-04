# MI450 Performance Optimizations

Performance work on the DLRM-v4 HSTU ranker (MLPerf yambda-5b) on **MI450 (gfx1250)**: where the
platform stands now, and each optimization applied to get there. Companion to
[`perf_opt.md`](perf_opt.md) (MI350X / B200) and to the host record
[`mi450_1P4G/ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md`](mi450_1P4G/ctheliosp-1b112-a37-2/ctheliosr-1b112-a37-2.md)
(stack, hazards, TODOs).

- **[Latest evaluation](#latest-evaluation)**: current measured end-to-end status.
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
`gin/mi450_hipblaslt/m{0..4}_*.gin`. The code is on branch
[`chcai/mi450_hipblaslt_opt`](https://github.com/chriscai-amd/training/tree/chcai/mi450_hipblaslt_opt).

---

## Latest evaluation

One row per hardware × node count, latest only.

| date | commit | hardware / host | nodes / GPUs | software stack | run config | throughput (median steps 20–200) | trace | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-04 | [`be2d249`](https://github.com/chriscai-amd/training/commit/be2d249) on `chcai/mi450_hipblaslt_opt` (trunk [`e4c612f`](https://github.com/chriscai-amd/training/commit/e4c612f) + gin-guarded MI450 opts) | 4× **MI450** (gfx1250, 256 CUs, 432 GiB HBM each, one XGMI hive)<br>host `ctheliosp-1b112-a37-2`<br>firmware **BKC 26.11.08**, VBIOS `113-M4500001-700E`<br>amdgpu-dkms `7.1.0-2413386.el10`, kernel `6.16.1-0_fbk5`<br>amdgpu loaded with `noretry=0 gpu_recovery=0 ip_block_mask=0xcff` | 1 / 4 | image `recommendation-gfx1250-20260910:triton-7ff97e` (`ea5cee73`, rebuilt 2026-10-04)<br>torch `2.11.0+rocm7.14.0a20260625`, triton `3.8.0+git7ff97e31`, torchrec `1.7.0a0+bf55480`, fbgemm_gpu nightly `2026.10.4`<br>stock hipBLASLt: `rocm-sdk-libraries-gfx1250 7.14.0a20260625`<br>**M1 hipBLASLt: ROCm 10.2 nightly `10.2.0a20260929`** `libhipblaslt.so.1.5`, NN Origami pool, NT catalog filtered to the stock stub | yambda-5b, MLPerf gin defaults, full model (11 tables, 3 HSTU layers, seq 4096)<br>local batch 1,024 / global 4,096, `START_TS=0`, 200 steps, seed 1<br>`HSTU_BWD_MAX_VGPR=0`, `AMDGCN_USE_BUFFER_OPS=0`, `TRITON_FULL_AUTOTUNE=0`, allocator vars empty, eval off<br>gin overlay **`m1_lib.gin`** (M1 on, M2–M4 off) | **2,841.1 global_sps, 1,441.7 ms/step**<br>vs **2,092.4 / 1,957.5** with all opts off (same host, boot, image, data)<br>**+35.8 % throughput, −26.3 % step time** | **pending**: the cumulative trace (`OUTPUT_TRACE=1`, 5 steps from step 52) needs a host power cycle first, because GPU 1 has been unusable since the M3 fault at 09:02 | Safe cumulative stack = **M1** (+M2, which is inert in training). M3/M4 are **unsafe in training** (GPU fault; see the log). One run per arm, not yet ABBA-repeated; the repeat was cut short by the M3 fault. Final loss at step 200 is identical across M0/M1/M2 (`0.13897`). |

---

## Optimization log

One row per incremental optimization, **each with its own A/B**: same host, boot, image and
dataset; arms run back to back; the stated baseline is enabled. The first arm (M0) read the
227 GB dataset cold. Bench numbers come from the Arbor campaign `mi450-hipblaslt`: a paired
33-GEMM hipBLASLt bench at T = 2,097,152, same GPU, ABBA, measuring hipBLASLt ms/step. End-to-end
numbers come from the 200-step training A/B above.

| date | main optimization | how it is applied (gin) | main commits | GEMM bench (hipBLASLt ms/step) | e2e perf (median steps 20–200) | notes |
|---|---|---|---|---|---|---|
| 2026-10-04 | **M1: hipBLASLt library swap, NN only.** ROCm 10.2 nightly `libhipblaslt.so.1.5` with its gfx1250 bf16 **NN Origami pool** (rocm-libraries #12271, ~425 stream-K tuned kernels). The image's 7.14 library has a 2-kernel stub there, which picks `MT32x16x32`. Its **NT catalog is filtered back to the stock stub** (sol 102/103), so no 10.2 NT tall-K kernel is reachable. | `apply_launch_env.hipblaslt_lib_dir = "/armlibs/rocm-10.2_nnonly_v1"` (overlay `m1_lib.gin`). The launcher process sets `LD_PRELOAD`, `LD_LIBRARY_PATH` and `HIPBLASLT_TENSILE_LIBPATH` before the ranks spawn. The container must mount the lib dir at `/armlibs` and `/opt/rocm-10.2` read-only. | [`86b88c1`](https://github.com/chriscai-amd/training/commit/86b88c1) (Arbor iter 5), gin guard [`5c1f080`](https://github.com/chriscai-amd/training/commit/5c1f080). Lib-dir manifest: Arbor `results/mi450-hipblaslt-a37-2/runs/mi450-hipblaslt/arbor_libs_rocm-10.2_nnonly_v1/` | **1,154.1 → 684.9 (1.685×)**, rounds 1.6849 / 1.6853; NN group 500 → 35 ms/step (UVQK fwd 349 → 25, out-proj fwd 128 → 6.6) | **2,092.4 → 2,841.1 global_sps (+35.8 %)**; step **1,957.5 → 1,441.7 ms (−26.3 %)**. Runs `1004_M0a` → `1004_M1`. | **Armed in training:** every rank maps `/opt/rocm-10.2/lib/libhipblaslt.so.1.5`; a G08-shaped `addmm` logs sol **951** (10.2 `MT256x256x64` SK3) vs **104** stock. Numerics: bench max rel err 1.66e-3 (unchanged); loss at step 200 identical (0.13897). Ship path: a newer `_rocm_sdk_libraries_gfx1250` wheel instead of an `LD_PRELOAD`. |
| 2026-10-04 | **M2: hipBLASLt tuning override.** Pins the NT weight gradients to stock sol 103 `MT256x128x64` (from 102 `MT32x16x32`) and the TN re-expressed keys to sol 1409. | `apply_hipblaslt_opts.tuning_override_file = ".../arbor_hipblaslt/tuning_override.csv"` (overlay `m2_override.gin`), exported as `HIPBLASLT_TUNING_OVERRIDE_FILE` per rank | [`c56bda3`](https://github.com/chriscai-amd/training/commit/c56bda3) (iter 6), [`80d0e62`](https://github.com/chriscai-amd/training/commit/80d0e62), [`bfa73ce`](https://github.com/chriscai-amd/training/commit/bfa73ce) | **684.7 → 502.7 (1.362×)** at the bench's fixed T | **No e2e gain:** 2,841.1 → 2,819.8 global_sps (−0.75 %, within run spread); 1,441.7 → 1,452.6 ms. Runs `1004_M1` → `1004_M2`. | **Expected:** override rows match exact `(m, n, k)`, and every row has K = 2,097,152. Training's T is jagged (1.59–2.53 M), so no row ever matches. To make it real, move the pin into the library catalog by size range (TODO). Kept off by default. |
| 2026-10-04 | **M3: TN weight gradients for the HSTU UVQK / output projections** (G20 2048×512×T, G19 512×1536×T). The NT `x.t() @ dz` becomes `mm(x.t().contiguous(), dz.t().contiguous().t())`, served by the tuned TN pool. | `apply_hipblaslt_opts.tn_wgrad_hstu = True` (overlay `m3_tn_hstu.gin`) → `HSTU_TN_WGRAD=hstu` | [`5e62ca3`](https://github.com/chriscai-amd/training/commit/5e62ca3) (iter 7), scopes [`e1b5a77`](https://github.com/chriscai-amd/training/commit/e1b5a77) | 502.8 → 465.4 (1.080×). The GEMMs drop to ~11 ms/step; the transpose copies add ~174 ms/step. Not claimable: no recapture yet | ❌ **FAULT, not measured.** Run `1004_M3` faulted **GPU 1 in its first training step**: no-retry page faults → MES `REMOVE_QUEUE`/`INVALIDATE_TLBS` failures → "Memory access fault by GPU". The launcher killed it; the GPU stays unusable until a power cycle. | **Unsafe in training; keep off.** In the bench (T = 2,097,152) the 10.2 TN tall-K kernels sol 1397/1409 ran cleanly. Under real jagged T (up to 2.53 M) the library picks other 10.2 TN tall-K solutions or larger operands, the same fault class as the 10.2 NN/NT tall-K hangs. Needs a hipBLASLt fix (host record TODO T1). |
| 2026-10-04 | **M4: TN weight gradients for the preprocessor MLP Linears** (G25 256×512×T ×3, G27/G29/G31 24/1024/512×256×T). `nn.Linear` → `TNWgradLinear`. | `apply_hipblaslt_opts.tn_wgrad_preproc = True` (overlay `m4_all.gin` = M1 + M2 + M3 + M4) → `HSTU_TN_WGRAD=all` | [`80d0e62`](https://github.com/chriscai-amd/training/commit/80d0e62) (iter 10), [`bfa73ce`](https://github.com/chriscai-amd/training/commit/bfa73ce) (iter 11) | 468.2 → 421.2 (1.112×), then 420.8 → 365.0 (1.153×). Not claimable: no recapture yet | **Not run.** Same 10.2 TN tall-K kernel class as M3. | **Unsafe in training; keep off** (same reason as M3). Bench cumulative M1..M4 = 1,154.1 → 364.6 ms/step (3.16×), of which 287 ms/step is transpose copies. |

### Not applied (recorded so they are not re-run)

| date | candidate | result | notes |
|---|---|---|---|
| 2026-10-04 | ROCm **10.1** hipBLASLt swapped into the image (Arbor iter 4) | 1.0001× in the bench (null) | Kernel selection changed on 30 GEMMs, yet the total did not move. The 1.56× seen when running the 10.1 stack **host-native** comes from the runtime, not from libhipblaslt (host record H6). |
| 2026-10-04 | 10.2 **NN with K = T** re-expression of G19/G20 (Arbor iter 8) | GPU 0 page fault + MES failure in the bench; reverted | The first tall-K fault inside the harness. |
| 2026-10-04 | Splitting the transpose copies into narrow column passes (Arbor iter 13) | REVERT | Did not beat the bar. The copies (287 ms/step in the bench) stay the top remaining cost. |

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
