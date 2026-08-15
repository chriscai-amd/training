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

| date | hardware / host | software stack (pinned) | run config | sharding plan (placement) | throughput | MFU / HFU | trace |
|---|---|---|---|---|---|---|---|
| 2026-08-14 | 8× **NVIDIA B200** (`sm_100`), 183 GiB HBM<br>driver 580.126.09<br>SLURM, 1 node<br>data on node-local NVMe | `nvcr.io/nvidia/pytorch:26.04-py3`<br>torch `2.12.0a0+0291f960b6.nv26.04`<br>CUDA 13.2 / cuDNN 9.21 / NCCL 2.29.7<br>Triton 3.6.0, torchrec 1.4.0, py 3.12.3<br>fbgemm_gpu src `10b77573`, repo `aed6a43` | **MLPerf config** — bs 1024/rank → 8192 global<br>`START_TS=0`, `HBM_CAP_GB` 90<br>`max_seq_len` 4096, no eval/ckpt<br>`NUM_TRAIN_TS=27`, 100 steps, ran to completion | **8 TW + 3 CW**, pinned to the bs-512 plan via<br>`EMB_SHARDING_OVERRIDES`; `EMB_PLACEMENT=auto`<br><br>**HBM** (`fused`) — 422.5 GB, 0 DDR<br>`user_x_artist` CW 191.1<br>`user_x_album` CW 76.5<br>`item_x_hour` CW 76.5<br>`user_x_hour` TW 45.9<br>`item_id` TW 18.0<br>`album_id` TW 6.4<br>`user_x_is_organic` TW 3.8<br>`artist_id` TW 2.5<br>`uid` TW 1.9<br><br>**DDR** (`fused_uvm_caching`) — 28.0 HBM cache / 137.6 DDR<br>`user_x_artist_x_hour` TW 15.6 + **76.4 DDR**<br>`artist_x_hour` TW 12.5 + **61.2 DDR**<br><br>per-rank HBM: r0 **73.7** (peak), r1 71.1, r2 69.7,<br>r3 69.2, r4 65.3, r5 37.4, r6 35.0, r7 31.9<br>DDR only on r6 (76.4) and r7 (61.2) | avg **12,969** `global_sps`<br>min **10,818** / max **14,980**<br>**631.7** `ms/step` avg<br>steps 40–100 | **22.75%** / **9.10%**<br>511.8 / 204.7 tflops/gpu<br>`fill` 40.0% mean (33.8–42.6) | [`trace/b200/trace_step52.json.gz`](../trace/b200/trace_step52.json.gz)<br>8 ranks stitched, CPU+GPU<br>460,175 events, 8.7 MB<br>5 steps from 52; **crosses the<br>ts 23→24 window boundary** |
