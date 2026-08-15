# Performance Optimizations — MI350X HSTU / OneTrans (yambda-5b, bs=1024, TRITON)

Performance work for the 8× MI350X HSTU ranker on `yambda-5b` at `batch_size=1024`
with the **TRITON** HSTU kernel and bf16 training. Companion to
[`training_recipe.md`](./training_recipe.md) (environment + reproduction).

Throughput numbers are global samples/sec across 8 GPUs (`global_sps`), measured
at steady state (instantaneous, computed from consecutive logged steps).

---

## LN-dropout: multi-row, separated-RNG path on MI350

### What

`_ln_mul_dropout_*` has two kernel variants:

- **legacy** — single program per row, RNG fused inline (`_ln_mul_dropout_fwd`).
- **separated-RNG** — multiple rows per program, dropout mask precomputed once
  and reused by the backward (`_ln_mul_dropout_fwd_rng` /
  `_ln_mul_dropout_bwd_dx_du_rng`).

The separated path was previously gated to Blackwell only (`is_sm100_plus()`).
MI350X (`gfx950`) benefits from the same structure, so the gate now also enables
it on MI350.

### Where

| file | change |
|---|---|
| `ops/utils.py` | `is_amd_mi350()` (gfx950 detect) + `use_separated_rng_ln_mul_dropout()` gate |
| `ops/triton/triton_hstu_linear.py` | dispatch LN-dropout fwd to the separated-RNG path when the gate is true |

```python
# ops/utils.py
def use_separated_rng_ln_mul_dropout() -> bool:
    return is_sm100_plus() or is_amd_mi350()
```

### Perf

**+5.6% end-to-end → 14,222 global sps** (separated-RNG vs legacy fused, identical
config, full boost clocks — see the caveat below).

---

## Caveat — GPU clock lock can mask all perf changes

A node-level GPU clock lock will silently invalidate any benchmark on this
machine, so check it before trusting numbers.

During this work all 8 GPUs were stuck in **`perf_determinism`** performance
level at **sclk 1093 MHz** (DPM level 1) while the real max is **2200 MHz**
(level 2) — despite 100% utilization, ~370 W of power headroom (629 / 1000 W),
and low temps (~50 °C). This was **not** thermal/power throttling; it was
leftover node state from a prior job.

Effect: a **uniform ~1.87× slowdown of every Triton compute kernel**
(`2200 / 1093 ≈ 2.0×`), including kernels unrelated to any code change. It made
the LN-dropout fix above look like a regression until the clock state was found.

### Detect + fix

```bash
rocm-smi --showperflevel          # expect "auto", not perf_determinism/manual/low
rocm-smi -d 0 --showclocks        # expect sclk ~2000+ MHz under load
rocm-smi --setperflevel auto      # restore boost
```

`scripts/launch_slurm.sh` (worker phase) now logs the perf level + a live `sclk` sample on
every launch, auto-restores `auto` if it finds a `perf_determinism`/`manual`/`low`
lock, and warns (to reset from the host) if it lacks permission inside the
container. **Always sanity-check `sclk ≈ 2000+ MHz` before trusting a benchmark.**

---

## Evaluation log

Running record of measured runs — **one row per hardware platform / config**,
appended as new platforms and stacks are evaluated. All rows: yambda-5b, TRITON
HSTU kernel, bf16, aggregate average over steady-state steps.

| date | hardware / host | software stack (pinned) | run config | sharding plan (placement) | throughput | MFU / HFU | trace | notes — main gaps & things to try |
|---|---|---|---|---|---|---|---|---|
| 2026-08-14 | 8× **NVIDIA B200** (`sm_100`), 183 GiB HBM<br>driver 580.126.09<br>SLURM, 1 node<br>data on node-local NVMe | `nvcr.io/nvidia/pytorch:26.04-py3`<br>torch `2.12.0a0+0291f960b6.nv26.04`<br>CUDA 13.2 / cuDNN 9.21 / NCCL 2.29.7<br>Triton 3.6.0, torchrec 1.4.0, py 3.12.3<br>fbgemm_gpu src `10b77573`, repo `aed6a43` | **MLPerf config** — bs 1024/rank → 8192 global<br>`START_TS=0`, `HBM_CAP_GB` 90<br>`max_seq_len` 4096, no eval/ckpt<br>`NUM_TRAIN_TS=27`, 100 steps, ran to completion | **8 TW + 3 CW** pinned via `EMB_SHARDING_OVERRIDES`<br>`EMB_PLACEMENT=auto`<br><br>**HBM** (`fused`) — 422.5 GB, 0 DDR<br>`user_x_artist` CW 191.1<br>`user_x_album` CW 76.5<br>`item_x_hour` CW 76.5<br>`user_x_hour` TW 45.9<br>`item_id` TW 18.0<br>`album_id` TW 6.4<br>`user_x_is_organic` TW 3.8<br>`artist_id` TW 2.5<br>`uid` TW 1.9<br><br>**DDR** (`fused_uvm_caching`) — 28.0 HBM cache / 137.6 DDR<br>`user_x_artist_x_hour` TW 15.6 + **76.4 DDR**<br>`artist_x_hour` TW 12.5 + **61.2 DDR**<br><br>per-rank HBM: r0 **73.7** (peak), r1 71.1, r2 69.7,<br>r3 69.2, r4 65.3, r5 37.4, r6 35.0, r7 31.9<br>DDR only on r6 (76.4) and r7 (61.2) | avg **12,969** `global_sps`<br>min **10,818** / max **14,980**<br>**631.7** `ms/step` avg<br>steps 40–100 | **22.75%** / **9.10%**<br>511.8 / 204.7 tflops/gpu<br>`fill` 40.0% mean (33.8–42.6) | [`traces/b200/trace_step52.json.gz`](../traces/b200/trace_step52.json.gz)<br>8 ranks stitched, CPU+GPU<br>460,175 events, 8.7 MB<br>5 steps from 52; **crosses the<br>ts 23→24 window boundary** | |
| 2026-08-14 | 8× **AMD MI350X** (`gfx950`), 288 GB HBM3E<br>host `cv350-rck-g03-e16-08`<br>SLURM job 26010, 1 node<br>perf level `auto` on all 8 GPUs<br>data on **shared NFS** | `rocm/primus:v26.3`<br>torch `2.12.0+rocm7.2` / ROCm 7.2<br>torchvision 0.27.0, torchaudio 2.11.0<br>torchrec `v2026.06.01.00` (`bf554808`)<br>fbgemm_gpu nightly-rocm `2026.6.2`<br>(local gfx950 wheel), py 3.12<br>repo `13d6cf1` | **MLPerf config** — bs 1024/rank → 8192 global<br>`START_TS=0`, `HBM_CAP_GB` 260<br>`max_seq_len` 4096, no eval/ckpt<br>`NUM_TRAIN_TS=27`, 100 steps, ran to completion<br>`METRIC_LOG_FREQ=10`, `TRITON_FULL_AUTOTUNE=0`<br>train_samples 930,017 | **8 TW + 3 CW** chosen by the planner<br>`EMB_PLACEMENT=auto`, no overrides<br><br>**HBM** (`fused`) — 560.1 GB, **0 DDR**<br>(288 GB/GPU fits everything; no UVM tier)<br>`user_x_artist` CW 191.1<br>`user_x_album` CW 76.5<br>`item_x_hour` CW 76.5<br>`user_x_artist_x_hour` TW 76.5<br>`artist_x_hour` TW 61.2<br>`user_x_hour` TW 45.9<br>`item_id` TW 18.0<br>`album_id` TW 6.4<br>`user_x_is_organic` TW 3.8<br>`artist_id` TW 2.5<br>`uid` TW 1.9<br><br>per-rank HBM: r4 **95.9** (peak), r5 80.6, r0 73.7,<br>r1 71.1, r2 69.7, r3 69.2, r6 65.3, r7 37.4<br>mean 70.4, 0 DDR on every rank | avg **10,771** `global_sps`<br>min **5,856** / max **12,836**<br>**760.6** `ms/step` avg<br>steps 40–100<br><br>excl. the ts-boundary interval:<br>**12,522** sps / **654.2** `ms/step` | **19.87%** / **7.96%**<br>456.9 / 183.1 tflops/gpu<br>`fill` 40.3% mean (33.8–43.4)<br><br>excl. ts-boundary interval:<br>**21.52%** / **8.57%**<br>494.6 / 197.2 tflops/gpu | [`traces/mi350x/trace_step52.json.gz`](../traces/mi350x/trace_step52.json.gz)<br>8 ranks stitched, CPU+GPU<br>477,706 events, 8.5 MB<br>5 steps from 52, **same batches as B200** | **Gaps vs B200** — exact non-overlapping rank-0 decomposition, 5 traced steps on identical batches: **652.3 vs 558.1 ms/step, MI350X 16.9% slower**.<br><br>MI350X **loses 150.2 ms** on the components below and **wins 56.0 ms** back elsewhere; `150.2 − 56.0 = +94.2` ms/step, matching the measured wall to 0.00 ms. The % is each component's share of the **150.2 ms there is to reclaim**:<br><br>**MI350X slower — 150.2 ms/step total**<br>• **GEMM — 41%** (+61.3; 138.4 vs 77.1, i.e. 1.8× slower). Biggest single item.<br>• **GPU idle — 37%** (+55.2; 104.1 vs 48.9). 88.9 of the 104.1 is the unpinned input H2D stall: all threads idle between `aten::to`, ~2.4 GB/s effective on a 215.6 MB batch.<br>• **exposed comm — 13%** (+20.0; 27.0 vs 7.0). Total comm is 64.8 vs 31.4, so the a2a is both larger *and* less overlapped.<br>• **jagged / gather / index — 7%** (+11.0)<br>• **everything else — 2%** (+2.7; position emb, attn fwd, TBE, misc)<br><br>**MI350X faster — 56.0 ms/step total**<br>copy/memset 32.6, elementwise 11.2, attn bwd 6.9, LN/dropout 5.3.<br>**Don't bank the copy credit** — it is B200 paying *visibly* for the same unpinned input path (28.5 ms/step of device-side `Memcpy HtoD (Pageable -> Device)`) where ROCm hides the equivalent work in a host stall. B200 idles a further 36.8 ms/step there, so pinning helps both and **widens** the real gap MI350X has to close.<br><br>**Outside the trace** — the logged gap is 128.9 ms/step, so **+34.7** comes from window transitions on shared NFS (worst 10-step interval **1399 ms** vs B200 757).<br><br>**Things to try**, roughly by expected value:<br>1. **Pin the input path** — `pin_memory=True` on the streaming loaders (`train/utils.py` 1110/1185/1320/1331) + `non_blocking=True` in `Samples.to`. `Samples.pin_memory()` already exists and is **dead code** today. Targets ~85 ms/step.<br>2. **GEMM tuning** — the 1.8× GEMM gap is the largest single term. Try `TRITON_FULL_AUTOTUNE=1` (run used 0) and check hipBLASLt algo/tile selection for the HSTU linear shapes.<br>3. **Hide the a2a** — 27.0 ms/step is exposed; more comm streams or earlier sparse prefetch.<br>4. **Launch overhead** — 1123 kernels/step; fuse the small elementwise chains or capture the step with HIP graphs.<br>5. **Stage the dataset on node-local NVMe** — removes the window-transition spike (infra, not GPU).<br>6. **Attention bwd is 4.4× fwd** (B200 4.68×) — well above the 2–2.5× flash norm, so a bwd-kernel rewrite is a shared win rather than a platform gap.<br><br>MFU/HFU use the **2300 TFLOPS** dense-bf16 peak from `dlrm_v4/utils.py`, which is the right denominator here — the trace's `deviceProperties` report `AMD Instinct MI350X`, 256 CU, 288 GB. (The `submission_platform: MI355X` line in the MLPerf log is a hardcoded default in `mlperf_logging_utils.py`, **not** hardware detection.) |
