# DLRM-v4 on MI450: hipBLASLt GEMM request

For the hipBLASLt team. This folder contains every hipBLASLt GEMM the MLPerf DLRM-v4 benchmark (Yambda-5B, HSTU ranker) runs in training on MI450 (gfx1250), with timings and a ready-to-run `hipblaslt-bench` YAML.

| File | Content |
|---|---|
| [`dlrmv4-BF16-allgemms-union.txt`](dlrmv4-BF16-allgemms-union.txt) | 33 unique GEMMs as `hipblaslt-bench` YAML lines, captured from training |
| [`dlrmv4_gemm_component_map.md`](dlrmv4_gemm_component_map.md) | Which model component each GEMM belongs to, with calls per step, measured time, TFLOP/s, and the solution index and kernel used in training |
| [`capture_to_yaml.py`](capture_to_yaml.py) | Script that turns `HIPBLASLT_LOG_MASK=32` logs into the YAML |
| [`../performance_analysis.md`](../performance_analysis.md) | Full step-time analysis of this workload on MI450 (hipBLASLt is §4.1 A1–A3 and §4.2) |

## Summary

- **hipBLASLt is about two-thirds of the training step:** ~1,160–1,190 ms of a 1,735 ms step per GPU.
- **97% of that time is in two operand layouts that fall back to slow kernels on gfx1250:**
  - **NN** (`Cijk_Ailk_Bljk`, 502 ms/step);
  - **NT** (`Cijk_Ailk_Bjlk`, weight gradients, 626 ms/step).
- They run at **0.5–77 TFLOP/s**, mostly on the `MT32x16x32` tile (solutions 104 and 102).
- In the **TN** layout (`Cijk_Alik_Bljk`), the library reaches **0.9–1.36 PFLOP/s** on the same GPU.
- Re-expressing the slow GEMMs in TN gives the same math 12–90× faster. So the hardware can do it; the NN/NT kernels or the heuristic are missing.

## The ask

1. **Fast bf16 NN kernels (`Ailk_Bljk`) for gfx1250** with a very large N (tokens, ~2 M) and M, K in 256–2,048.
   - Priority 1 is `2048 × T × 512` with a bf16 bias epilogue: 347 ms/step.
   - Priority 2 is `512 × T × 1536` with beta = 1: 129 ms/step.
   - A fast NN kernel exists: `256 × T × 512` NN gets MT256x256x64 (solution 105) at 472 TFLOP/s. But `512 × T × 256`, `1024 × T × 256` and the two shapes above all get the MT32x16x32 fallback (solution 104). That points to a gap in tuning or heuristic coverage.
2. **Fast bf16 NT weight-gradient kernels (`Ailk_Bjlk`) for a tall K (K = T ≈ 2 M) and a small output** (M × N from 24 × 256 to 2,048 × 512).
   - The output has only a few tiles, so these need **split-K / stream-K**. Example: `256 × 512 × T` gets MT256x128x64 (solution 103). That is 4 workgroups on 256 CUs, which runs at 15 TFLOP/s.
   - The rest get MT32x16x32 at 0.5–70 TFLOP/s.
   - The `24 × 256 × T` weight gradient takes 48.7 ms for 26 GFLOP of work (0.5 TFLOP/s). Its memory lower bound is 0.16 ms.
3. **Packaging note: benchmark results can be wrong.** The `hipblaslt-bench` in `_rocm_sdk_devel/bin` has an RPATH to `_rocm_sdk_devel/lib/libhipblaslt.so.1`. That copy's kernel library (16 MB) lacks the tuned gfx1250 kernels PyTorch uses from `_rocm_sdk_libraries_gfx1250` (26 MB). Without an `LD_PRELOAD`, even the fast TN GEMMs fall back to MT32x16x32. For example, `512 × T × 2048` TN runs at 82 TFLOP/s in the bench against 1,242 TFLOP/s in PyTorch. Solution indices from PyTorch logs, e.g. 173, report "NO solution found".

## Priority list

Measured in the bench at T = 2,097,152 (GPU 0, idle), with the gfx1250 hipBLASLt preloaded. The lower bound is the slower of two limits: 1,500 TFLOP/s of compute, or the operand traffic at 7.3 TB/s (device copy bandwidth measured on this GPU). It is a target, not a promise.

| # | M × N × K | layout | Component, pass | calls/step | TFLOP/s now | ms/step now | lower bound ms/step |
|---|---|---|---|---:|---:|---:|---:|
| 1 | 2048 × T × 512 | NN (+bias) | HSTU UVQK projection, fwd (+ recompute) | 6 | 76 | 346.6 | 17.6 |
| 2 | 2048 × 512 × T | NT | HSTU UVQK projection, wgrad | 3 | 70 | 187.2 | 8.8 |
| 3 | 512 × 1536 × T | NT | HSTU output projection, wgrad | 3 | 57 | 173.9 | 6.6 |
| 4 | 512 × T × 1536 | NN (beta = 1) | HSTU output projection, fwd | 3 | 77 | 128.6 | 6.6 |
| 5 | 256 × 512 × T | NT | Preprocessor MLP 256 → 512, wgrad | 3 | 15 | 112.2 | 1.3 |
| 6 | 1024 × 256 × T | NT | Preprocessor MLP 1024 → 256, wgrad | 1 | 21 | 52.6 | 0.7 |
| 7 | 512 × 256 × T | NT | Preprocessor MLP 512 → 256, wgrad | 1 | 11 | 51.4 | 0.4 |
| 8 | 24 × 256 × T | NT | Preprocessor MLP 24 → 256, wgrad | 1 | 0.5 | 48.7 | 0.2 |
| 9 | 1024 × T × 256 | NN | Preprocessor MLP 1024 → 256, dgrad | 1 | 73 | 15.1 | 0.7 |
| 10 | 512 × T × 256 | NN | Preprocessor MLP 512 → 256, dgrad | 1 | 72 | 7.6 | 0.4 |
| | | | **Top 10** | | | **1,124** | **~43** |

The remaining 23 GEMMs total 31 ms/step. They are already on good kernels or are tiny (1,024-row candidate MLP).

For the model, rows 1–10 are ~65% of the step. Bringing them to ~1 PFLOP/s would cut the step from 1,735 ms to roughly 650–700 ms (estimate, [performance_analysis.md §4.5](../performance_analysis.md#45-what-fixing-the-mi450-stack-would-give-estimate)).

## How to reproduce

Image `recommendation-gfx1250-20260910:triton-7ff97e`:
- PyTorch `2.11.0+rocm7.14.0a20260625`, HIP `7.14.60850`;
- hipBLASLt from ROCm 7.14, `_rocm_sdk_libraries_gfx1250`.

Host `ctheliosp-1b112-a37-1`: 4 × MI450 (gfx1250), 256 CUs, 432 GiB HBM per GPU. The image has no PyYAML, which `hipblaslt-bench --yaml` needs; the command below mounts the host's copy.

```bash
S=/opt/venv/lib/python3.12/site-packages
# The YAML ships with iters: 0 (same convention as the DeepSeek union); set timing iterations first.
sed 's/cold_iters: 0, iters: 0/cold_iters: 3, iters: 10/' dlrmv4-BF16-allgemms-union.txt > timed.yaml
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --security-opt seccomp=unconfined -v $PWD:/kit \
  -v /usr/lib64/python3.12/site-packages/yaml:/pyext/yaml:ro -e PYTHONPATH=/pyext \
  -e HIP_VISIBLE_DEVICES=0 \
  -e LD_PRELOAD=$S/_rocm_sdk_libraries_gfx1250/lib/libhipblaslt.so.1 \
  recommendation-gfx1250-20260910:triton-7ff97e \
  $S/_rocm_sdk_devel/bin/hipblaslt-bench --yaml /kit/timed.yaml
```

The `LD_PRELOAD` line is required to see the same kernels as PyTorch (ask 3).

Single GEMM, priority 1, with its kernel name:

```bash
hipblaslt-bench -m 2048 -n 2097152 -k 512 --lda 2048 --ldb 512 --ldc 2048 --ldd 2048 \
  --transA N --transB N --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r --bias_vector --bias_source d --bias_type bf16_r \
  --cold_iters 3 --iters 10 --print_kernel_info
```

Two edits were made to the logged lines for the YAML:
- **Token dimension** normalised to 2,097,152. The capture had 1.59–2.53 M. Priorities 1 and 2 run at the same rate at the smallest, canonical and largest T (75 and 68–69 TFLOP/s), so the normalisation does not change the picture.
- **`aux_type` set to `d_type`.** The bench rejects `aux_type: f32_r` with bf16 outputs ("Invalid aux type f32_r").

### How the GEMMs were captured

A 12-step training run on 4 GPUs, local batch 1,024, run with:

```
HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=<dir>/hipblaslt_bench_%i.log
```

This gave 2,736 logged calls; the same 33 shapes appear on every rank. `python capture_to_yaml.py <dir> groups.json union.yaml` rebuilds the YAML.

Run configuration: `HSTU_BWD_BLOCK_N=128`, `AMDGCN_USE_BUFFER_OPS=0`, and `PYTORCH_CUDA_ALLOC_CONF` cleared. These are the settings of the current performance configuration; they affect only Triton kernels, not GEMM shapes.

The per-GEMM ms/step agrees with the training profiler trace: 1,155 ms/step in the bench against 1,190 ms/step in the trace, where T varies from step to step. The kernel names in the component map come from that trace.
