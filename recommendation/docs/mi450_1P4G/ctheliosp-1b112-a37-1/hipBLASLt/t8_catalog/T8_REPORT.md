# T8: pin the bf16 NT weight-gradient GEMMs to sol 103 in the catalog itself

Status 2026-10-10: built and validated on the CPU only. **Nothing has run on a GPU.**

**Before any training use,** a human must run the single-GEMM checks in section 7. Sol 103 is the kernel behind the A0 corrupted-weight-gradient failures (section 6, risk R1).

## 0. Review findings and what was done

| # | Finding (reviewer) | Verdict after my own check | Action |
|---|---|---|---|
| S1 | The predicate list leaves out `streamKDynamicQueueSupported` and `uniformSummationOrderSupported` (semantics) | **Real, documentation only.** Both appear in `SolutionLibrary.hpp` L74-96 at 5911365c, combined with problem and task predicates. Neither can reject sol 102 or 103: both are SK0, and PyTorch never requests uniform summation order. | `gridmatch.nt_size_accepts` docstring now lists every unmodelled filter: these two, the task predicates and a user split-K. Also covered in section 1. |
| E1 | The speed-up per shape was never measured; G27 would run on only 2 workgroups (edit) | **Real.** `docs/mi450_perf_opt.md` L73: 684.7 → 502.7 is a bench total over the five pinned NT shapes plus the TN pins. No per-shape number for sol 103 exists except G25. | It can't be measured without a GPU. Section 7 step 1 records time per shape. `make_v2.py --drop MxN --out …_v3` builds a variant that leaves any shape on 102. Section 5 gives an estimate per shape from a37-1 data. |
| X1 | T8 moves the weight gradients onto the kernel behind the A0 corrupted-DW failures (selection) | **Real.** I read `.vgpr_count` from both `.co` files: sol 103 has **676** in the 10.2 and the stock build; sol 102 has **252**. `mi450_a0.md` L923-924 and `nan_triage_resume_20260920.md` L826 match the review. Sol-103 launches go from **3 to 12 per step** (`simulate_v1_v2.txt` §4b). | No catalog change can fix this without undoing T8. Instead, the risk is now in the v2 README and stock README, simulate §4b quantifies it, and the GPU plan (section 7) has a numerics stage before any A/B: zero-DY, 1000 calls per shape. Use T8 only for guarded A/Bs of 200 steps or fewer. |
| — | Build-report claim: "the bench measured a different binary" | **Wrong, corrected.** The M2 bench ran on the M1 lib (`mi450_perf_opt.md` L72-73). Its `tuning_override.csv` names the **10.2** sol-103 kernel (`Git Version: 5911365c`; the name has the 10.2-only `NTG0`/`RAP0` fields). | v2 runs exactly the measured binary; the README now says so. The **stock** sol 103 (the M0+T8 arm) is the unmeasured one, which the stock README now states. |
| — | README said KDTREE would leave "several shapes with no solution" | Loosely worded. That is true of the v1 table; with v2, KDTREE=1 would send G25 → 102 and G17 → 103 (simulate §5). | README reworded. All `TENSILE_*` knobs must stay unset. |
| E-notes | Wiring, `HIPBLASLT_TENSILE_LIBPATH`, index range, symlink paths in the stock README | Correct. | Stock README now says symlink targets must be container paths. Overlay `gpu_check/m1_t8_lib.gin` added; the repo is untouched. |
| S2-S4, X2-X6 | Notes | Correct, no defect. | Recorded in sections 1 and 6. |

## 1. Semantics: how the GridBased table selects

**Source and version.** All source references are to ROCm/rocm-libraries `projects/hipblaslt/tensilelite`, commit `5911365c1f5b81f6b2a9ad9515f731b4cfb979a4` (2026-09-23), checked out at `/home/chcai/t8_work/src/rl`. That is exactly the installed library: `libhipblaslt.so.1.5` reports `hipBLASLt git version: 5911365c` (for example `/home/chcai/a37-1_env/runs/m1_gpu0_NT_T2097152/g27.log`), so there is no version uncertainty for 10.2.

**Catalog row.** The NT catalog `TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib` contains:
- `ExactLogicLibrary`, with one row whose predicate is `GridBasedMatching` (`ExactLogicLibrary.hpp` L203, L304);
- under it, a `ProblemMatchingLibrary` with properties `[FreeSizeA, FreeSizeB, BatchSize, BoundSize]`. Each key is `[M, N, batch, K]` in hipBLASLt terms, and the distance is `GridBased`.

**Load** (`Serialization/MatchingLibrary.hpp` L58-97). The table is `std::sort`ed by key, then by `speed` descending. Keys are `std::array<int64,4>`. An `index` absent from `solutions` raises "Invalid solution index" (L283/L296).

**Search** (`PropertyMatching.hpp`: `findBestKeyMatch_GridBased` L965ff, binary path L1112-1430, used by default). With default env:
1. Find the M block: the first table M ≥ m, clamped to the largest M.
2. Inside that block, find the first N ≥ n, clamped.
3. Then match batch the same way.
4. Within that group, pick the entry nearest in K. `GridBasedDistance`, `Distance.hpp` L371-404, is `|K - gridK|` only.
5. Offer the entry to the solution's predicates. If they reject it, try the next entry by K distance in the group, then the next N group, then the next M block, then walk backwards, and finally fall through to the next library row.

**Batch flattening.** A key with batch > 1 is flattened once to batch 1 (`key[0]*=b` if m > n, else `key[1]*=b`; L1308-1330). This is on by default because `TENSILE_GRIDBASED_BATCH_EXP` is unset.

**Environment knobs.** `m_gridbasedKdTree` and `m_gridbasedBatchExp` both default to false (`Debug.hpp` L183-184), and are parsed with `strtol(..,0) != 0` (`Debug.cpp`). `GRIDBASED_TOPSOLS` is parsed but nothing reads it. `TENSILE_METRIC` would override the distance. None of these is set by `_env_bootstrap.py`, `run_ab.sh` or `gemm_watch.sh`.

**Per-solution filters** (`SolutionLibrary.hpp` L74-96, `softwarePredicate`): `problemPredicate && taskPredicate && streamKDynamicQueueSupported && uniformSummationOrderSupported`.

| Filter | Modelled? | Why it cannot reject sol 102 or 103 for these shapes |
|---|---|---|
| Problem predicates: LeadingFree0/1 ≥, WorkgroupNumberCheck, BufferLoad/Store offset checks, GSU-min-K | Yes, in `nt_size_accepts` | Accept for all target shapes at any K. For 103, the load-offset check depends only on lda/ldb × 64. |
| Task predicates: WorkspaceCheck, LaunchLimits | No | GSU = 1 for both solutions, so `calculateAutoGSU` (`ContractionSolution.cpp` L2131) returns 1 and no workspace is needed. |
| `streamKDynamicQueueSupported` | No | Applies only to SK4 or dynamic SK5 kernels on non-power-of-two XCD counts. Both solutions are SK0. |
| `uniformSummationOrderSupported` | No | True unless uniform summation order is requested by a matmul attribute. PyTorch does not set it. |
| User split-K (`problem.getParams().gsu()`) | No | Not used by training or by the checks below. |

If any of these did reject 103 for a target, the walk would land on that M block's `[M, 1048576]` guard, which maps to 102: today's behaviour, never a failure.

**Validation of the model.** `gridmatch.py` re-implements this. Reviewers compiled the C++ binary path verbatim (`/home/chcai/t8_work/review/harness.cpp`, built with `_GLIBCXX_DEBUG`) and fuzzed the two against each other: 0 mismatches.
- 135,000 queries on random tables plus v2.
- 98,112 queries on the v2 table only.
- A second, independent re-implementation also agreed.

**Stock image (7.14 nightly, 2026-06-25).** The GridBased search, distance and row walk are identical on the 7.14 branch tip apart from visibility macros. The 7.14 `softwarePredicate` is only `problemPredicate && taskPredicate`. Uncertainty remains: the diff was against the branch tip, not the exact nightly.

## 2. Entries added

19 entries take the table from 35 to 54. Nothing else changed: same `solutions` (102 and 103), row predicate, properties, distance and `{key, index}` entry layout, and no `speed` field. Keys are sorted and unique.

| Key [M,N,batch,K] | Sol | Role |
|---|---|---|
| [23,1,1,1] | 102 | guard: m in (16, 23] stays on 102 |
| [24,255,1,1] | 102 | guard |
| **[24,256,1,1]** | **103** | **G27** |
| [24,1048576,1,1] | 102 | guard: n > 256 |
| [511,1,1,1] | 102 | guard |
| [512,255,1,1] | 102 | guard |
| **[512,256,1,1]** | **103** | **G31** |
| [512,1535,1,1] | 102 | guard |
| **[512,1536,1,1]** | **103** | **G19** |
| [512,1048576,1,1] | 102 | guard |
| [1023,1,1,1] | 102 | guard |
| [1024,255,1,1] | 102 | guard |
| **[1024,256,1,1]** | **103** | **G29** |
| [1024,1048576,1,1] | 102 | guard |
| [2047,1,1,1] | 102 | guard |
| [2048,511,1,1] | 102 | guard |
| **[2048,512,1,1]** | **103** | **G20** |
| [2048,1048576,1,1] | 102 | guard |
| [1048576,1,1,1] | 102 | guard: m > 2048, previously clamped into a 102 block |

Each new (M,N,1) group has exactly one entry, so K never affects the choice, at any K and not just 1.0M-3.2M.

**Files.**
- `/home/chcai/training/arbor_libs/rocm-10.2_nnonly_v2/hipblaslt/library/gfx1250/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib`, sha256 `1053790f4c7ba8fe07b9f93b69baeaa02488035545b810168c828dc55630accc`. The rebuild gave a byte-identical catalog; only the README changed.
- Stock version: `/home/chcai/t8_work/stock_t8/gfx1250/<same name>`, sha256 `df62e167b4792227842d864dbc1d2043953b34f16bbf264846791925d4c60738`, with an identical table.

## 3. Simulation

Full output: `/home/chcai/t8_work/simulate_v1_v2.txt`. The stock-image catalogs (stock vs `stock_t8`) give byte-identical output: `simulate_stock_t8.txt`.

**Training bf16 NT GEMMs.** K was tested at 21 values (9 fixed plus 12 random in 1.0M-3.2M); union rows were tested at K = 1.0M, 1.59M, 2.097M, 2.53M and 3.2M.

| Group | m × n × K, batch | lda / ldb | Calls/step | v1 sol / WG | v2 sol / WG | Changed |
|---|---|---|---|---|---|---|
| G20 | 2048×512×T | 2048/512 | 3 | 102 / 2048 | **103 / 32** | yes |
| G19 | 512×1536×T | 512/1536 | 3 | 102 / 1536 | **103 / 24** | yes |
| G29 | 1024×256×T | 1024/256 | 1 | 102 / 512 | **103 / 8** | yes |
| G31 | 512×256×T | 512/256 | 1 | 102 / 256 | **103 / 4** | yes |
| G27 | 24×256×T | 24/256 | 1 | 102 / 16 | **103 / 2** | yes |
| G25 | 256×512×T | 256/512 | 3 | 103 / 4 | 103 / 4 | no |
| G17 | 516×512×1024 | 516/512 | 1 | 102 | 102 | no |
| G23 | 512×512×1024, batch 8 | 4096/4096 | 1 | 102 | 102 | no |

All 126 (shape, K) wgrad points land on 103 in v2. Launches per step: v1 is {102: 11, 103: 3}; v2 is {102: 2, 103: 12}. The f32 NT shapes (512,512,1024) and (1536,512,1024) use another catalog and are unaffected; so are the other bf16 NT catalogs.

**Every changed shape in the sweeps.**
- Generic sweep: 132,055 points over 49 m values × 49 n values × batch 1/2/3/4/8 × 11 K values. Exactly 12 (m,n,batch) combinations change, all 102 → 103 at every K:
  - the 5 targets;
  - 7 batched shapes that flatten onto a target: (24,32,b8), (24,64,b4), (24,128,b2), (512,256,b2), (512,512,b3), (512,768,b2), (1024,512,b2). None of these is in the workload.
- Random sweep of 200,000 points (m, n up to 2²¹; batch 1/2/4/8; K up to 2²²): 0 changes.

The general rule: a shape changes exactly when it is (m,n,1) = a target, or when batch-flattening it gives a target.

## 4. Load-validity checks

Run by `make_v2.py` on both builds; all pass:
- The catalog decompresses (zlib), and msgpack unpack then repack reproduces the bytes exactly.
- Solutions and every row's metadata are identical to the input catalog; only the table differs.
- Every key has 4 positive ints; keys are sorted and unique; entries have exactly `{key, index}`.
- No M block ends in a multi-batch N group (the forward-walk past-end hazard).
- Only indices 102 and 103 are referenced, and both are in `solutions`.
- For each kernel name, the name, its symbol and its `.kd` symbol are present in the gfx1250 ELF inside the zstd CCOB bundle (`llvm-readelf --syms`). This covers `/opt/rocm-10.2/.../<NT>.co` and the stock `.co`.
- The v2 directory tree matches v1: 405 symlinks with identical targets, and only 2 real files (README.txt and the catalog).
- `gridmatch --selftest`: 33/33.

Checked this session with `llvm-readelf --notes`:

| Kernel | `.vgpr_count` (both builds) |
|---|---|
| Sol 103 (MT256x128x64, SK0, GSU 1) | 676 |
| Sol 102 (MT32x16x32) | 252 |

v1 was not modified; the `--out` guard refuses v1 and anything outside `arbor_libs/rocm-10.2_nnonly_v<N>`.

## 5. Expected performance per shape (estimate, not measured)

Measured on a37-1 with v1 at T = 2,097,152 (`/home/chcai/a37-1_env/runs/m1_gpu0_NT_T2097152/summary.txt`):
- G25 on sol 103: 23.8 ms.
- On sol 102: G27 28.4 ms, G31 29.6 ms, G29 29.7 ms, G19 31.9 ms, G20 57.0 ms.

Sol 103 has no split-K. Each workgroup does a full 256×128 tile over all of K, and every target needs at most 32 workgroups on 256 CUs. So time per call should stay close to G25's 23.8 ms whatever the workgroup count. This is an inference, not a measurement.

| Shape | Calls/step | Sol 102 (ms) | Sol 103 estimate (ms) | Estimated saving (ms/step) |
|---|---|---|---|---|
| G20 | 3 | 57.0 | ≈24 | ≈100 |
| G19 | 3 | 31.9 | ≈24 | ≈24 |
| G29 | 1 | 29.7 | ≈24 | ≈6 |
| G31 | 1 | 29.6 | ≈24 | ≈6 |
| G27 | 1 | 28.4 | ≈24 | ≈4 |
| **Total** | | | | **≈140** |

The total is the same order as the bench's 182 ms (on a37-2). **G27 is the marginal case.** If step 1 shows it no faster on 103, drop it:

```
/home/chcai/.venv_tools/bin/python -I /home/chcai/t8_work/make_v2.py --drop 24x256 --out /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3
```

## 6. Remaining risks

- **R1, numerics (the main risk).** Sol 103 (676 VGPRs) is the A0 DW-corruption kernel. In A0's isolated zero-DY test it failed at call 196; sol 102 passed 2 × 1000 calls. The suspected cause is the faulty CWSR trap handler, which a37-1 still runs. T8 raises sol-103 exposure from 3 to 12 launches per step. Each launch is long: about 24 ms on 2-32 workgroups. There is one point in its favour: sol 103 has already run 3×/step for G25 in every M1 run, including a37-2's 14,000-step finite convergence run. The requirements:
  - pass section 7 step 3 first;
  - then only A/Bs of 200 steps or fewer, with a loss comparison and NaN/finite checks;
  - no long runs until the handler is fixed.
- **R2, faults.** Under the current driver (default module options), a GPU fault goes through the reset path that wedged a37-2. Sol 103 at K = T is already exercised by G25. The new shapes have the same kernel and the same uint32 strides. G20's `lda·K` passes 2³² once K exceeds 2,097,152, but with batch 1 that product only feeds the batch stride, which is multiplied by batch index 0. Sol 102 already runs G20 at K up to 2.53M with the same arguments, so R2 is low but non-zero. That is why step 1 runs one GEMM per process with a dmesg check after each.
- **R3, model vs C++.** The selection rests on `gridmatch.py` plus the reviewers' compiled copy of the C++ matcher. The real library has not loaded the file yet; step 1's `--print_kernel_info` closes this.
- **R4, environment.** All of these must stay unset: `TENSILE_GRIDBASED_KDTREE`, `TENSILE_GRIDBASED_BATCH_EXP`, `TENSILE_METRIC`, `TENSILE_SOLUTION_INDEX` and `HIPBLASLT_TUNING_OVERRIDE_FILE`. M2's override pins the same 103, so it is redundant rather than harmful. Keep `HIPBLASLT_TENSILE_LIBPATH` set explicitly: v1 and v2 have no `gfx1250v0` folder.
- **R5, stock arm (M0+T8).** Same table, but the 7.14 sol-103 binary was never measured at these shapes, and it has the same 676 VGPRs. To use it, the image's gfx1250 folder must be copied (or symlinked using container paths) with the edited file swapped in.
- **R6, inherited from v1.** The index mapping still assigns 102-525 to this file. Any explicit lookup of index 104-525 (an old override row or a TunableOp file) would find nothing.

## 7. GPU safety checks for a human to run, in order (I have NOT run any of these)

**Preconditions for every step:**
- `preflight.sh --allow-defaults` passes; `gemm_watch.sh` and `run_zero_dy.sh` both run it.
- No other KFD process is running.
- `AMDGCN_USE_BUFFER_OPS=0`.
- One GPU, one GEMM per process.
- **Stop everything at the first fault line** (exit code 3).

### Step 1: selection and speed, one GEMM per process, K = 1.6M, 2.097152M and 2.533005M

This uses `hipblaslt-bench` from `/opt/rocm-10.2/bin` via `gemm_watch.sh`, which:
- sets `HIPBLASLT_TENSILE_LIBPATH=<v2>/hipblaslt/library/gfx1250` and `LD_LIBRARY_PATH=/opt/rocm-10.2/lib`;
- passes `--print_kernel_info`;
- runs one bench process per YAML line with a 300 s kill timeout;
- diffs dmesg after each line.

The union.yaml lines are 27 G27, 31 G31, 29 G29, 25 G25, 19 G19 and 20 G20 (the targets), plus controls 17 G17 and 23 G23. Line 25 is a known-good control.

```
cd /home/chcai/a37-1_env
for T in 1600000 2097152 2533005; do
  T=$T ./gemm_watch.sh /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v2 0 \
      /home/chcai/t8_work/gpu_check/runs/v2_gpu0_NT_T$T \
      /home/chcai/mi450_perf_20260926/hipblaslt_req/union.yaml 25 27 31 29 19 20 17 23 || break
done
```

**Pass criteria:**
- Each `summary.txt` shows `rc=0 sol=103 … MT256x128x64 SK0` for lines 19, 20, 25, 27, 29 and 31, and `sol=102 MT32x16x32` for lines 17 and 23.
- Each `g*.log` has `--Solution index: 103` and `--kernel name: …MT256x128x64…SK0…`.
- No FAULT line.

Compare `us` against v1's `/home/chcai/a37-1_env/runs/m1_gpu0_NT_T2097152/summary.txt`. If G27 is not faster, rebuild without it (section 5).

### Step 2: hipblaslt-bench CPU-reference check at K = 1.6M

This uses `/home/chcai/t8_work/gpu_check/nt_verify.yaml`: the same six NT lines with `norm_check: 1`, `cold_iters: 1` and `iters: 1`.

```
cd /home/chcai/a37-1_env
T=1600000 ./gemm_watch.sh /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v2 0 \
    /home/chcai/t8_work/gpu_check/runs/v2_gpu0_verify_T1600000 \
    /home/chcai/t8_work/gpu_check/nt_verify.yaml 1 2 3 4 5 6
```

The lines run cheapest-first: G27, G31, G29, G25, G19, G20. Pass if every log shows sol 103 and a `hipblaslt-norm-error` column of about 1e-3 or less.

The CPU reference costs up to about 3.4 TFLOP (G20). If line 5 or 6 hits the 300 s timeout (rc 137, "bench rc" stop), that is CPU time, not a GPU problem. In that case skip those lines and rely on step 3.

### Step 3: A0-style repeated-call numerics, one shape per container

This runs `torch.mm(dY.t(), X)`, exactly the training call. First is `zero` mode: 1000 calls, every output word must be exactly +0 after every call. Then `rand` mode: 200 calls, each bit-identical to call 1, with call 1 within 2e-2 of an fp32 reference.
- The first call is profiled and must show an `MT256x128x64` kernel.
- The script `/home/chcai/t8_work/gpu_check/run_zero_dy.sh` does the guards: preflight, a live dmesg watch that kills the container, a 1800 s timeout and logs under `gpu_check/runs/`.
- It mounts `arbor_libs` at `/armlibs` and sets `LD_PRELOAD`, `LD_LIBRARY_PATH` and `HIPBLASLT_TENSILE_LIBPATH` the way `apply_launch_env` does.

```
cd /home/chcai/t8_work/gpu_check
for s in "24 256" "512 256" "1024 256" "512 1536" "2048 512" "256 512"; do
  bash run_zero_dy.sh 0 $s 2097152 zero 1000 || break
done
for s in "24 256" "512 256" "1024 256" "512 1536" "2048 512"; do
  bash run_zero_dy.sh 0 $s 2533005 rand 200 || break
done
```

The last zero-mode shape, "256 512", is G25: the sol-103 shape already in production. Any failure there also implicates today's M1, not just T8. Pass when every run ends with `PASS` and exits 0, with no fault.

**On any wrong word in zero mode,** do **not** proceed to training. That is the A0 symptom, and the answer is a v3 that drops the failing shape(s), or T8 waits for the handler fix.

### Step 4 (only if steps 1-3 pass): 200-step e2e A/B, M1 vs M1+T8, ABBA, one arm at a time

```
cd /home/chcai/a37-1_env
./run_ab.sh M1a  m1_lib.gin
./run_ab.sh M1T8a /t8/m1_t8_lib.gin -v /home/chcai/t8_work/gpu_check:/t8:ro
./run_ab.sh M1T8b /t8/m1_t8_lib.gin -v /home/chcai/t8_work/gpu_check:/t8:ro
./run_ab.sh M1b  m1_lib.gin
```

Check that:
- the step-200 loss is identical or within run spread;
- `train.log` has no NaN or inf;
- `fault.txt` is empty;
- a `HIPBLASLT_LOG_MASK` (or rocprof) recapture of one T8 step shows sols 103 and 102 per group, as in section 3.

## 8. Files

| Path | What it is |
|---|---|
| `/home/chcai/t8_work/make_v2.py` | Builder and validator. Modes: default v2, `--stock`, `--check DAT CO`, `--drop MxN --out …_vN`. |
| `/home/chcai/t8_work/gridmatch.py` | GridBased matcher model. The docstring lists the unmodelled filters. |
| `/home/chcai/t8_work/simulate.py` | Simulation; §4b is the exposure and workgroup table. |
| `/home/chcai/t8_work/simulate_v1_v2.txt` | Simulation output, v1 vs v2. |
| `/home/chcai/t8_work/simulate_stock_t8.txt` | Simulation output, stock vs stock_t8. |
| `/home/chcai/training/arbor_libs/rocm-10.2_nnonly_v2/` | The v2 lib dir. README updated with the risk text and kernel-provenance correction. |
| `/home/chcai/t8_work/stock_t8/` | Stock-image catalog and README. Not wired anywhere. |
| `/home/chcai/t8_work/gpu_check/nt_verify.yaml` | Step 2 input. |
| `/home/chcai/t8_work/gpu_check/zero_dy_loop.py` | Step 3 script. |
| `/home/chcai/t8_work/gpu_check/run_zero_dy.sh` | Step 3 wrapper with the guards. |
| `/home/chcai/t8_work/gpu_check/m1_t8_lib.gin` | Step 4 overlay. None of the `gpu_check` files have been run. |
| `/home/chcai/t8_work/review/` | Reviewers' C++ harness and fuzzer. |
