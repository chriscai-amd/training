# M5: split-T for the six tall-K NT weight gradients (DLRM-v4 HSTU, a37-1)

Status 2026-10-11: code, three v3 catalogs, CPU tests, catalog simulation and the GPU kit are done and reviewed. **Nothing has run on a GPU, and nothing is committed.** This file supersedes `DESIGN.md` where they differ.

## 1. What M5 is

Six NT weight-gradient GEMMs dominate the hipBLASLt time after T8. Each is `dW = a.t() @ b` with `a` [T, D] and `b` [T, O], bf16 and row-major. In hipBLASLt terms that is transA=N, transB=T, m=O, n=D, K=T, batch 1, and T is the jagged token count (1.59M to 2.53M).

| G | m×n | Calls/step | Call site |
|---|---|---|---|
| G20 | 2048×512 | 3 | `triton_addmm_bwd`: `x.t() @ duvqk` |
| G19 | 512×1536 | 3 | `HSTUComputeOutputFunction.backward`: `tn_wgrad_mm(y, dout)` |
| G25 | 256×512 | 3 | preprocessor `TNWgradLinear` 256→512 |
| G29 | 1024×256 | 1 | preprocessor `TNWgradLinear` 1024→256 |
| G31 | 512×256 | 1 | preprocessor `TNWgradLinear` 512→256 |
| G27 | 24×256 | 1 | preprocessor `TNWgradLinear` 24→256 |

**Why these calls are slow.** Under T8 every one of them runs on sol 103 (MT256x128x64) at about 24 ms per call, whatever m and n are. The reason is that it gets only 2 to 32 workgroups on a 256-CU GPU, and there is no split-K. Together they cost about 288 ms/step.

**How split-T works.**
1. Split T into s chunks of L = ⌊T/s⌋ rows, using zero-copy views: `a3 = a[:sL].view(s,L,D)` and `b3 = b[:sL].view(s,L,O)`.
2. Issue **one** strided-batched `torch.bmm(a3.transpose(1,2), b3)`. That is hipBLASLt NT with m=O, n=D, batch=s, K=L, lda=O, ldb=D, strideA=L·O, strideB=L·D. It runs on the same kernel with s times as many workgroups.
3. Sum the bf16 partials in fp32 with `part.sum(0, dtype=float32)`.
4. Add the leftover rows r = T − sL (r < s):
   - for r ≥ 2, one fp32 `addmm_`, which is NT on the SS catalog, sol 2821, MT32x32x16;
   - for r = 1, an elementwise fp32 `addr_`, with no GEMM.
5. Cast once to bf16, giving [D,O] contiguous. That is the same contract as `torch.mm(a.t(), b)`.

**Kernel selection.** hipBLASLt GridBased flattens batch: (m,n,b) becomes (m·b, n) if m > n, otherwise (m, n·b). The v3 catalogs pin those flattened keys to the intended solution and add guard rows so that nothing else changes.

**Output dtype.** I kept bf16-out bmm plus an fp32 sum and did not use `bmm(out_dtype=float32)`. The fp32-out call would land on the unmodified BB_SB catalog, sol 1420 (1020 VGPRs, never run on a37-1). bf16-out stays on the NT catalog that T8 already controls, with the two kernels A0 characterised. The cost is the numerics in section 6.

**Three variants**, all built by `make_v3.py`. Each lib dir is v2's symlink tree plus a new NT `.dat`, a README and `m5_splits.json`.

| Variant | Lib dir (container) | NT wgrad sol | Overlay |
|---|---|---|---|
| s103 (task variant a) | `/armlibs/rocm-10.2_nnonly_v3/s103` | 103 for all six, batch 1 and pinned s | `m5_split_t.gin` |
| s102 (task variant b) | `/armlibs/rocm-10.2_nnonly_v3/s102` | 102 for all six, batch 1 and every s in 2..256 | `m5_split_t_s102.gin` |
| mix (new) | `/armlibs/rocm-10.2_nnonly_v3/mix` | G20/G19 on 103, the other four on 102 | `m5_split_t_mix.gin` |

## 2. Gin switches

All are in `apply_hipblaslt_opts`, and `yambda_5b.gin` sets them all OFF. An environment variable that is already set wins over gin, as for the other options.

| gin binding | env var | Default | Meaning |
|---|---|---|---|
| `splitt_wgrad_hstu` | `HSTU_SPLITT_WGRAD` (`0`/`hstu`/`preproc`/`all`) | False | G20 and G19. Raises if combined with `tn_wgrad_hstu` (M3). |
| `splitt_wgrad_preproc` | (same variable) | False | G25, G29, G31 and G27. Raises if combined with `tn_wgrad_preproc` (M4). |
| `splitt_wgrad_splits` | `HSTU_SPLITT_WGRAD_SPLITS` (`MxN:s,…`) | `{}` | s per hipBLASLt m×n. Unlisted shapes keep the original call. |
| `splitt_min_rows` | `HSTU_SPLITT_MIN_ROWS` | 4096 | No split when L < this. |
| `splitt_audit_every` | `HSTU_SPLITT_AUDIT_EVERY` | 0 | Diagnostic (section 6). Keep it at 0 for timed runs. |
| (env only) | `HSTU_SPLITT_ALLOW_UNPINNED=1` | unset | Turns the import-time catalog check from a hard error into a warning. |

**Import-time catalog check (new).**
- `triton_addmm` reads `<lib dir>/m5_splits.json`, finding the lib dir from `HIPBLASLT_TENSILE_LIBPATH`, which `apply_launch_env` sets for every rank.
- It raises `ValueError` if a configured (shape, s) would not select the shape's mapped sol. Example: G19 s=12 on s103 would land on a guard row and run sol 102.
- If the lib dir has no manifest (stock, v1 or v2), it only logs that selection is unverified.
- If M5 is on but the split dict is empty, it warns "nothing is split".

**When it falls back.** The original call is issued in each of these cases:
- the shape is not configured;
- the operands are not bf16, not row-major, or not CUDA;
- D or O is not a multiple of 8, or a pointer is not 16-byte aligned;
- L < min_rows;
- L·max(D,O) ≥ 2³¹. This is a margin on the u32 batch stride, and it only refuses G20 at s=2 for T ≥ 2,097,152.

The first time a configured shape is refused, that is logged as "configured but not applied".

**Off is bit-identical.** With the variable unset or `0`, every new flag is False and the helper is None. `TNWgradLinear` is `nn.Linear`, and `tn_wgrad_mm` and `triton_addmm_bwd` issue the same aten calls as before. Tests check this with `torch.equal`, with `torch.bmm` patched to raise.

## 3. Files

**Repo worktree** `/home/chcai/training_opt/recommendation/generative_recommenders/`, uncommitted. The full patch, including new files, is `/home/chcai/m5_work/m5_code.diff` (1,186 lines).

| File | Change |
|---|---|
| `ops/triton/triton_addmm.py` (+286/−13) | Mode, split and min-rows parsing; `_splitt_catalog_check` (manifest); `splitt_wgrad_plan`; `_splitt_parts` / `splitt_wgrad_compute`; `splitt_wgrad_mm` (logs expected sol, GridBased key and preferred BLAS); `_splitt_audit`. Hooks: `tn_wgrad_mm(..., splitt=)`, `_TNWgradLinearFn.backward`, `TNWgradLinear.forward` (configured shapes only), `triton_addmm_bwd` via `globals().get`. |
| `ops/triton/triton_hstu_linear.py` | G19: `tn_wgrad_mm(y, dout, enabled=_TN_HSTU, splitt=_SPLITT_HSTU)` |
| `dlrm_v4/train/_env_bootstrap.py` | The 5 gin parameters, validation, and the M3/M4 exclusivity check |
| `dlrm_v4/train/gin/yambda_5b.gin` | All M5 bindings OFF |
| `gin/mi450_hipblaslt/m5_split_t.gin`, `m5_split_t_s102.gin`, `m5_split_t_mix.gin` (new) | Overlays for the three variants |
| `ops/tests/splitt_wgrad_test.py`, `dlrm_v4/train/tests/splitt_wgrad_bootstrap_test.py` (new) | CPU tests |

**Catalogs** under `/home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v3/{s103,s102,mix}/`. v1 and v2 are untouched.

**Work directory** `/home/chcai/m5_work/`:
- `make_v3.py`: builder and verifier, with `--sol 102|103`, `--map`, `--install`, and the new `--manifest-only`;
- `simulate_v3.py` and its output `simulate_v2_v3.txt`;
- `verify_{s103,s102,mix}.txt`;
- `proto/numerics_patterns.py` and `.out`;
- `gpu_check/`: `gen_bench.py`, the two `m5_bench_T*.yaml` files with their indexes, `splitt_loop.py`, `run_splitt_check.sh`, `README.txt`;
- `tests_out/pytest_m5_r2.txt`;
- `DESIGN.md`, corrected.

## 4. Review findings and what was done

| # | Finding | Verdict | Fix |
|---|---|---|---|
| R1 | The "1.41× rms" figure is only a best case | **Real.** Reproduced: drift and opposite-sign data give 57–459× (section 6). | Corrected in DESIGN and here. Added the error model sqrt(1+q²), a CPU test of it, the `--pattern drift/halves/users` checks in `splitt_loop.py`, and the in-training audit q. |
| R2 | The runtime s is not checked against the catalog | **Real.** G19 s=12 on s103 silently runs 102. | `m5_splits.json` per lib dir. Import raises on an unpinned s (override `HSTU_SPLITT_ALLOW_UNPINNED=1`). The per-shape log line shows the expected sol. Tested, including all three overlays against the installed manifests. |
| R3 | The mix variant was built but not installed | Real | Installed `v3/mix` (sha `32e7a28c…`, re-verified PASS, identical to the earlier build). Added `m5_split_t_mix.gin` and a `simulate_v3` section. |
| R4 | `splitt_loop.py` forced the BLAS backend | Real | It now uses the process default, as training does, and fails unless that is hipBLASLt (`--force-blas` exists only for experiments). |
| R5 | The kernel check was loose and a warning | Real | Call 1 must show exactly 1 bf16 NT `UserArgs` launch with the manifest's tile, exactly 1 fp32 MT32x32x16 tail iff r ≥ 2 (0 for r ≤ 1), and no rocBLAS or other GEMM. Any failure gives exit 2. |
| R6 | A silent rocBLAS fallback in training would go undetected | Real | Stage 3 greps `train.log` for `bgemm_internal_cublaslt error` (the string is confirmed in the image's `libtorch_hip.so`) and kernel-traces one B run. The log line also prints the preferred BLAS backend. |
| R7 | s102 speedup printed against the wrong baseline | Real | The timing line now names the batch-1 sol of the lib and says "NOT the T8 baseline" for 102. |
| R8 | The 1-row tail's layout is ambiguous (NN or NT) | Real but harmless | r=1 is now `acc.addr_` (an elementwise CUDA kernel, `addr_kernel_cuda`, confirmed in the image), so there is no GEMM. Its bench lines were removed (63 lines at T=2,533,005 instead of 70). The tests check the tail ops and their strides. |
| R9 | An unconfigured `TNWgradLinear` was rerouted when M5-preproc was on | Real (note) | It is rerouted only when (in, out) is configured; otherwise it is exactly `nn.Linear`. Tested. |
| R10 | `-e HSTU_SPLITT_WGRAD=…` with gin off is a silent no-op | Real (note) | Import now warns "nothing is split". |
| R11 | One split dict serves both scopes | Note, not fixed | No model shapes collide (the reviewer checked every NT group). Documented. |
| R12 | The README's reduction cost left out the tail and casts | Note | README updated: 3/4/6 extra launches for r=0, r=1 and r≥2, at most about 0.05 ms per call. |
| R13 | The 4 `output_backward_stages_test` failures | Pre-existing | The same 4 fail at HEAD (`NameError: tn_wgrad_mm` in an exec'd backward). Not touched. |

The catalog reviewer's notes still apply but need no change: s102 keeps the stub `[256,256,1,512]→103` for m=256, n≤256, which is no model problem; pins also affect any bf16 NT problem that flattens onto them, again none in the model union; there is no overflow in the C++ flatten (the largest key is 65,536).

## 5. Test results (CPU only: training image, no `--device`, worktree read-only, `arbor_libs` mounted read-only at `/armlibs`)

Output is in `/home/chcai/m5_work/tests_out/pytest_m5_r2.txt`. **72 passed and 4 failed; 173 subtests passed.**
- **M5 tests: 36 of 36 pass** (29 in `splitt_wgrad_test.py`, 7 in `splitt_wgrad_bootstrap_test.py`). They cover:
  - bit-exact against an emulation (tails 0, 1, 3, 4 and 63, plus D=24);
  - bf16 tolerance against `mm` and fp64;
  - tail op choice and strides;
  - the cancellation error model;
  - zero-dY gives +0;
  - zero-copy views;
  - every fallback, including the 2³¹ limit;
  - the parser;
  - the manifest check: pinned, unpinned, missing shape, no manifest, the import-time raise and the override;
  - dispatch through all three hooks;
  - the MLP gradient flow to fp32 master weights;
  - unconfigured layers staying `nn.Linear`;
  - the audit;
  - off-path bit-identity;
  - env modes and errors;
  - gin default, m0/m1/m2 and the three M5 overlays;
  - every overlay's s pinned by its installed catalog.
- **Existing `preprocess_backward_stages_test.py`: all pass.**
- **Existing `output_backward_stages_test.py`: 4 subtests fail.** They are the same 4 as at HEAD (`tests_out/pytest_head_baseline.txt`).

## 6. Numerics

Each bf16 partial P_i is rounded at its own magnitude. So the extra rms error is about u·‖sqrt(Σ P_i²)‖, against today's u·‖Σ P_i‖, where u is the bf16 unit roundoff. That gives:

**error ratio vs today ≈ sqrt(1 + q²),  q = ‖sqrt(Σ_i P_i²)‖ / ‖Σ_i P_i‖**

CPU study (`proto/numerics_patterns.out`): T = 2¹⁸, D=64, O=32, activations with mean 1. Each cell is the measured ratio, with the model's prediction in brackets.

| dY pattern | s=2 | s=8 | s=64 | s=256 |
|---|---|---|---|---|
| iid, mean 0 | 1.46 (1.37) | 1.41 (1.40) | 1.37 (1.36) | 1.37 (1.36) |
| sign per 512-token "user" | 1.51 (1.43) | 1.47 (1.47) | 1.52 (1.50) | 1.54 (1.50) |
| scale varying ×0.1..×10 along T | 1.40 | 1.50 | 1.46 | 1.47 |
| bias 0.3 | 1.44 | 1.07 | 1.01 | 1.00 |
| **drift −1→+1 along T** (total ≈ 0) | **451 (682)** | **328 (322)** | **111 (123)** | **57 (63)** |
| **opposite-sign halves** | **459 (835)** | **417 (388)** | **137 (142)** | **72 (67)** |

What this means:
- When chunk partials are independent, the cost is 1.4–1.5×, as designed.
- When the T-chunks carry large partials that cancel, the error relative to the small true gradient is 50–460× today's. The absolute error is still at most about u·Σ|P_i|.
- Larger s reduces this: with more, smaller chunks, each partial is smaller.
- Adam normalises per element, so this noise on near-zero gradient elements is not automatically harmless.
- Whether real dY/X cancel across contiguous token chunks (groups of users) is **unknown**. It depends on data order, for example whether batches are sorted by length or time.

How it will be measured:
1. `splitt_loop.py --pattern drift` checks that the GPU kernels follow the model. Its elementwise bound, 2⁻⁸(Σ|P_i| + |ref|) + 2⁻¹²(|A|ᵀ|B|), held with 0 violations on CPU for iid/drift/halves/users at s = 8 and 128.
2. `HSTU_SPLITT_AUDIT_EVERY=50` in one untimed B run logs the real q and the split-vs-original difference per shape.

If real q ≫ 1, there are three options: use a smaller s; use the fp32-out variant (`bmm(out_dtype=float32)`, which would need a BB_SB catalog edit and a first-ever safety check of sol 1420/1421); or restrict M5 to the shapes with q ≈ 1.

## 7. v3 catalogs and simulation

Each table goes from v2's 54 entries to 121.
- **Targets (28):**
  - the flattened keys of every candidate s;
  - the batch-1 keys G25 `[256,512]` and G29/G31 s=2/4 `[2048,256]`.
- **Guards (39):** each guard carries the old selection, so only the target (m,n) points change.

| Variant | Edits | NT `.dat` sha256 | Targets → sol |
|---|---|---|---|
| s103 | 67 added, 0 flipped | `a76b0b66…ca20` | all 103 |
| s102 | 67 added; 5 flipped 103→102 (T8 batch-1 pins of G20, G19, G29, G31, G27) | `b205ef00…30a4` | all 102. Strict guards: G25's region m ∈ (128,256], n > 256 is 102 at every s. |
| mix | 67 added; 3 flipped (G29, G31, G27 batch-1 → 102) | `32e7a28c…85cc` | G20 `[4096..32768, 512]` and G19 `[512, 3072..30720]` → 103; the rest → 102 (strict) |

Targets are `[M,N,1,1]`. The full edit list is in `verify_*.txt`, section 0, and in each lib dir's README.

**s that select the mapped sol** (from `m5_splits.json`, computed with gridmatch over T = 1.0M to 3.2M, lda=m, ldb=n):

| Variant | Shapes and s |
|---|---|
| s103 | G20 {2,4,8,16}; G19 {2,4,8,10,16,20}; G25 every s in 2..256 (v2 clamp); G29 {2,4,8,16,32,64}; G31 {2,4,8,16,32,64,128}; G27 {16,32,64,128,256}. Any other s goes to 102, and import now refuses it. |
| s102 | Every shape at every s in 1..256 → 102. |
| mix | G20 and G19 as in s103 on 103; the other four at every s → 102. |

**`simulate_v3.py`** (`simulate_v2_v3.txt`) gives **RESULT: PASS** for all three variants:
- **A. Install.** Each lib dir has the same 405 symlinks as v2 and the same as v1 apart from the NT catalog. The only regular files are the NT `.dat`, the README and `m5_splits.json`. The `.dat` equals the build, and `make_v2.validate` passes.
- **B. Mapped sol.** Every candidate s and T selects the mapped sol: 0 unexpected in each variant.
- **C. T8's `simulate.py` sweeps.** 66 changed lines for s103, 177 for s102 and 199 for mix. All of them flatten onto a pinned key or are a strict-region 103→102 change.
- **D. Exhaustive boundary grid** (151,704 points). The changed (m,n) are all pinned keys or strict 103→102: 21 for s103, 70 for s102, 78 for mix. The 300k random sweep, with batch up to 256, gives 0 / 38 / 38 differences, all explained the same way. In the model union file, nothing unintended changes: G17 and G23 (b=8) stay on 102.
- **E. Wgrad launches per step with each overlay's s.** s103 gives {103: 12}, each with 240–256 WGs. s102 gives {102: 12}. mix gives {103: 6, 102: 6}. T8 today is {103: 12}, with 2–32 WGs each.
- **F. Manifest.** `m5_splits.json` matches the simulation (variant, sha, solmap, and sol_by_s for s=1..256 across all 6 shapes), and every overlay s is pinned.

**Predicates and strides** (checked by the reviewers):
- Sols 102 and 103 are `StridedBatched` with no batch or stride predicate.
- Batch offsets are computed in 64 bits (`s_mul_hi_u32`/`s_mul_i32`). Only the u32 stride must fit.
- The worst accepted stride is G19 s=2 at T=2.53M, 1.95e9 < 2³¹.

## 8. Expected gain (inference, not measured)

The basis is measurements on a37-1 at T=2,097,152 with the default noretry:
- sol 103 takes about 23.8 ms per call at 2–32 WGs (no split-K, so each WG walks all of K);
- sol 102 takes G27 28.4, G31 29.6, G29 29.7, G19 31.9 and G20 57.0 ms.

**sol 103 model.** t ≈ ⌈WG_total/256⌉ · 23.8/s ms, because each WG does K/s, and with up to 256 WGs one wave runs in parallel.

| G (×calls) | s | WG_total | ms/call | ms/step |
|---|---|---|---|---|
| G20 (×3) | 8 | 32·8 = 256 | 23.8/8 = 2.98 | 8.93 |
| G19 (×3) | 10 | 24·10 = 240 | 2.38 | 7.14 |
| G25 (×3) | 64 | 4·64 = 256 | 0.372 | 1.12 |
| G29 (×1) | 32 | 8·32 = 256 | 0.744 | 0.74 |
| G31 (×1) | 64 | 4·64 = 256 | 0.372 | 0.37 |
| G27 (×1) | 128 | 2·128 = 256 | 0.186 | 0.19 |
| Reductions and tails | | | 12 × ≈0.05 | ≈0.6 |
| **s103 total** | | | | **≈19.1** (lower bound ≈18) |

**sol 102 model**, fitted to the five measured points with concurrency C ≈ 1536 WGs and a throughput floor WG₁·31.9/1536 ms per call:

t ≈ max(⌈WG₁·s/1536⌉ · 28.4/s, WG₁ · 31.9/1536)

Check: G20 at s=1 gives 2 × 28.4 = 56.8, against 57.0 measured.

| G (×calls) | s (overlay) | WG₁ | ms/call | ms/step |
|---|---|---|---|---|
| G20 (×3) | 4 | 2048 | max(6·28.4/4 = 42.6, 42.5) = 42.6 | 127.8 |
| G19 (×3) | 1 | 1536 | max(28.4, 31.9) = 31.9 | 95.7 |
| G25 (×3) | 4 | 256 | max(28.4/4 = 7.1, 5.3) = 7.1 | 21.3 |
| G29 (×1) | 2 | 512 | max(14.2, 10.6) = 14.2 | 14.2 |
| G31 (×1) | 4 | 256 | 7.1 | 7.1 |
| G27 (×1) | 64 | 16 | max(0.44, 0.33) = 0.44 | 0.44 |
| **s102 total** | | | | **≈266.5** (floor at optimal s ≈ 255) |

**mix:** G20 and G19 as in s103 (8.93 + 7.14) plus the 102 rows for the other four (21.3 + 14.2 + 7.1 + 0.44) plus about 0.6, giving **≈59.7 ms/step**.

**Per step, against today:**
- NT wgrad cost: v1 (M1) = 3·57 + 3·31.9 + 3·23.8 + 29.7 + 29.6 + 28.4 = **425.8 ms**; T8 (v2) = 12 × ≈24 = **≈288 ms**.
- The M1 step is 1,066 ms (measured ABBA, default noretry).
- M1+T8 ≈ 1066 − (425.8 − 288) ≈ **928 ms**. This is unmeasured; the T8 ABBA is still pending.

| Arm | NT wgrad ms/step | Est. step (ms) | vs M1+T8 | vs M1 | sol-103 launches/step |
|---|---|---|---|---|---|
| M1 (v1) | 425.8 | 1066 (measured) | | | 3 (G25) |
| M1+T8 (v2) | ≈288 | ≈928 | | +15 % | 12, 2–32 WG, ≈288 ms of 103 |
| **M1+T8+M5 s103** | ≈19 | **≈659** | **−269 ms, +41 % samples/s** | +62 % | 12, 240–256 WG, ≈18 ms of 103 |
| M1+M5 mix | ≈60 | ≈700 | −228 ms, +33 % | +52 % | 6, 240–256 WG, ≈16 ms |
| M1+M5 s102 | ≈266 | ≈906 | −22 ms, +2 % | +18 % | **0** |

**Caveats.**
- The s103 figure assumes the per-WG rate holds when every CU runs a 676-VGPR wave. G20 at s=8 streams about 52 GB through L2 in about 3 ms, so it may be HBM/L2-bound and slower. Stage 1 measures it.
- Step time scales with the real T, which spans 1.59M to 2.53M.
- s102 cannot beat about 255 ms because of 102's throughput floor. Its value is safety: it is 160 ms/step faster than M1 with no sol 103 at all, where M1 has 3 launches of 103 per step.

## 9. Risks

1. **CWSR / A0 corruption on sol 103 (main risk for s103 and mix).** Sol 103 (676 VGPRs) is the A0 DW-corruption kernel under a37-1's faulty CWSR handler. The memory notes say compaction evicts queues and that BLOCK_N=128 faulted 2 of 2 times.
   - s103 does the same 103 work as T8 (about 4,660 WG·ms per step). But it packs that into about 18 ms with every CU running 103, instead of 288 ms on 2–32 CUs.
   - If corruption scales with 103 wave-time, nothing changes. If it scales with the chance that a preempt/compaction event hits while 103 is resident, the window shrinks about 16×, but each event then saves and restores 8–128× more 676-VGPR waves. That is an untested regime.
   - The batched path (WG2 > 0) of 102 and 103 has never run on gfx1250. Byte offsets within one batch entry go beyond 2³¹, about 2.6 GB for G20 s=4 at T=2.53M.
   - Mitigation: batched zero-dY first (Stage 2), A/Bs of 200 steps or fewer, mix to halve the 103 launches, and s102 for no 103 at all.
2. **Numerics (section 6).** About 1.41× for chunk-independent data, up to hundreds of times on cancelling chunks. Real data is unmeasured. Use Stage 2 drift and the Stage 3 audit, and compare loss and grad-norm against the A arms.
3. **Kernel route.** bf16 `bgemm` must go through hipBLASLt; otherwise torch falls back to rocBLAS and warns once. The checks are Stage 2's strict kernel check, Stage 3's log grep and the kernel trace.
4. **Wrong s.** A non-pinned s is now rejected at import. Any `-e HSTU_SPLITT_WGRAD_SPLITS` override must use pinned values; on s102 and the 102 shapes of mix, any s from 2 to 256 is allowed.
5. **Silent fallback by layout or dtype.** If a call-site operand is not bf16 row-major, the original call runs. The "configured but not applied" log line shows this; expect 6 shapes split per rank.
6. **Side effects of the catalogs on non-model shapes.** Any bf16 NT problem whose flattened key equals a pin changes sol (for example batch-1 2048×256 → 103 in s103). None of these is in the model.
7. **Scope.** With mode `hstu`, the dict also applies to any `triton_addmm` backward of a matching (N, K). No collisions exist today.
8. **Estimates.** Every per-call number in section 8 is an inference.

## 10. GPU plan after the power cycle (nothing here has been run)

Preconditions:
- strict `preflight.sh` passes;
- the MES repro loop is **stopped**, so no other KFD process is running;
- `AMDGCN_USE_BUFFER_OPS=0`;
- one GEMM or shape per process with the dmesg watch.

Stop everything at the first exit 3. Full commands are in `/home/chcai/m5_work/gpu_check/README.txt`.

**Stage 1: gemm_watch single-GEMM bench of the batched shapes per s on one GPU; choose s.**
1. Run, from `/home/chcai/a37-1_env` and **without `T=`**:
   `./gemm_watch.sh <libdir> 0 <out> /home/chcai/m5_work/gpu_check/m5_bench_T{2097152,2533005}.yaml`
   Do it for s102 first, then s103. That is 40 and 63 lines; the `.index.txt` maps each line to shape, s, K and the expected kernel.
2. Check the kernel and sol of every line:
   - bf16 lines: MT32x16x32 on s102, MT256x128x64 on s103;
   - tail lines: MT32x32x16;
   - anything else: stop and re-check the catalog.
   The s=1 line of s103 is the T8 baseline.
3. Per shape and variant, pick s = argmin(bmm + tail + about 0.05 ms reduce). On 103, prefer fewer WGs when within 5 %. s must appear in the lib's pinned list (section 7).
4. Decide the arm(s): s103, mix (G20/G19 103 values plus the 102 values of the rest) and/or s102. Write the chosen s as an `HSTU_SPLITT_WGRAD_SPLITS` string.

**Stage 2: numerics checks through the real M5 function** (`run_splitt_check.sh <dev> <s103|s102|mix> <G..> <s> <T> <mode> [calls] [pattern]`). For each chosen (variant, shape, s):
1. `zero` 1000 calls at T=2097152. Every output and partial word must be +0. This is the A0 oracle on the batched kernels, at twice training's batched exposure.
2. `rand` 200 calls at T=2533005 with the iid pattern. Requirements:
   - bit-identical on every call;
   - 0 elementwise-bound violations;
   - rms ratio ≤ 1.6 against the batch-1 call of that lib;
   - the exit also requires the strict kernel check and a hipBLASLt default backend.
3. `rand` 50 calls with the drift pattern. The measured ratio must lie within 0.5–2× of sqrt(1+q²), with 0 bound violations. This shows the worst case on the real kernels.

Order: s102 shapes first (no 103), then the 103 shapes (G20 and G19 first, as they are shared by s103 and mix), then the other s103 shapes.

**Stage 3: e2e ABBA, M1+T8 against M1+T8+M5, with `run_ab.sh`** (200 steps each; the A overlay is `/t8/m1_t8_lib.gin -v /home/chcai/t8_work/gpu_check:/t8:ro`):
```
cd /home/chcai/a37-1_env
./run_ab.sh M1T8_A1   /t8/m1_t8_lib.gin -v /home/chcai/t8_work/gpu_check:/t8:ro
./run_ab.sh M1T8M5_B1 m5_split_t.gin   -e HSTU_SPLITT_WGRAD_SPLITS=<Stage-1 s103 choice>
./run_ab.sh M1T8M5_B2 m5_split_t.gin   -e HSTU_SPLITT_WGRAD_SPLITS=<same>
./run_ab.sh M1T8_A2   /t8/m1_t8_lib.gin -v /home/chcai/t8_work/gpu_check:/t8:ro
```
If 103 is judged too risky, use the B arms `m5_split_t_mix.gin` or `m5_split_t_s102.gin` with their own s strings.

After each B arm, `grep -n "M5\|bgemm_internal_cublaslt\|cublaslt error" train.log` must show:
- per rank, "split factors … verified against catalog <variant>";
- six "M5 split-T wgrad … expected sol …; preferred BLAS _BlasBackend.Cublaslt" lines;
- no "configured but not applied";
- no `bgemm_internal_cublaslt error`.

Two further untimed B runs:
- `OUTPUT_TRACE=1 ./run_ab.sh M1T8M5_TR m5_split_t.gin …`, then `trace_kernels.py`. Expect the `Cijk_Ailk_Bjlk_BBS_…MT256x128x64` launches at 12 per step with about 240–256 WGs (s103), and no rocBLAS.
- `-e HSTU_SPLITT_AUDIT_EVERY=50` to log the real chunk-cancellation q per shape. If q ≫ 1, go back to section 6's options before any longer run.

Compare `perf.txt` step time (expected about 928 → about 660 ms for s103), the loss curve, grad-norm and finiteness against the A arms. Do no long runs until the CWSR handler is fixed.
