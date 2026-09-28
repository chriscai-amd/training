# DLRM-v4 (Yambda-5B HSTU ranker) training — GEMM → component map

33 unique GEMMs. datatypes: {'bf16': 24, 'f32': 9}. Companion YAML: [`dlrmv4-BF16-allgemms-union.txt`](dlrmv4-BF16-allgemms-union.txt).
Precision: token path (HSTU projections, preprocessor MLPs) = bf16 in/out, f32 compute;
candidate-side MLP on 1,024 rows = f32. No fp8/mxfp8 in this workload.

Conventions, same as the DeepSeek map:

- **M, N, K and layout are hipblaslt-bench (column-major) values**: layout = `transA transB`. PyTorch's row-major `Y[rows, out] = X[rows, in] · W` appears as M = out, N = rows.
- **T = tokens per GPU per step.** The model is jagged: T ranged 1.59–2.53 M in the capture (mean ~2.08 M). It is normalised to **2,097,152** here.
- **pass**: fwd, dgrad (input gradient) or wgrad (weight gradient).

Extra columns (not in the DeepSeek map):

- **calls/step**: calls per GPU per training step.
- **ms/call, TFLOP/s**: `hipblaslt-bench` on one idle MI450 (gfx1250) at T = 2,097,152, cold_iters 3, iters 10, with the gfx1250 hipBLASLt preloaded (see [README](README.md#how-to-reproduce)).
- **ms/step**: ms/call × calls/step.
- **sol**: the `solution_index` PyTorch used in training, from the log.
- **kernel in training**: the kernel name from the profiler trace.

Sum over all GEMMs: **1,155 ms/step** in the bench, vs 1,190 ms/step of hipBLASLt kernel time in the training trace. The step is 1,735 ms, so hipBLASLt is **about two-thirds of the step**.

## HSTU UVQK projection (512 → 2048), `torch.addmm(bias, x, W[512,2048])` ×3 layers  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (+bias; also recomputed in bwd) | 2048 | T | 512 | bf16 | NN | 6 | 57.77 | **76** | **346.6** | 104 | `Ailk_Bljk` MT32x16x32 |
| dgrad | 512 | T | 2048 | bf16 | TN | 3 | 3.24 | 1,357 | 9.7 | 173 | `Alik_Bljk` MT128x256x128 |
| wgrad | 2048 | 512 | T | bf16 | NT | 3 | 62.40 | **70** | **187.2** | 102 | `Ailk_Bjlk` MT32x16x32 |

## HSTU output projection (1536 → 512), `torch.addmm(x, y, W[1536,512])` ×3 layers, residual via beta = 1  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (beta = 1) | 512 | T | 1536 | bf16 | NN | 3 | 42.86 | **77** | **128.6** | 104 | `Ailk_Bljk` MT32x16x32 |
| dgrad | 1536 | T | 512 | bf16 | TN | 3 | 3.59 | 918 | 10.8 | 235–237 | `Alik_Bljk` MT256x256x128 |
| wgrad | 512 | 1536 | T | bf16 | NT | 3 | 57.97 | **57** | **173.9** | 102 | `Ailk_Bjlk` MT32x16x32 |

## Preprocessor MLP output layers (256 → 512), `nn.Linear` ×3 (content / additional-embedding / action MLPs)  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (+bias) | 512 | T | 256 | bf16 | TN | 3 | 1.10 | 501 | 3.3 | 173 | `Alik_Bljk` MT128x256x128 |
| dgrad | 256 | T | 512 | bf16 | NN | 3 | 1.17 | 472 | 3.5 | 105 | `Ailk_Bljk` MT256x256x64 |
| wgrad | 256 | 512 | T | bf16 | NT | 3 | 37.40 | **15** | **112.2** | 103 | `Ailk_Bjlk` MT256x128x64 (4 workgroups, no split-K) |

## Preprocessor MLP input layer (1024 → 256), `nn.Linear`  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (+bias) | 256 | T | 1024 | bf16 | TN | 1 | 1.52 | 723 | 1.5 | 153 | `Alik_Bljk` MT96x256x128 |
| dgrad | 1024 | T | 256 | bf16 | NN | 1 | 15.10 | **73** | **15.1** | 104 | `Ailk_Bljk` MT32x16x32 |
| wgrad | 1024 | 256 | T | bf16 | NT | 1 | 52.61 | **21** | **52.6** | 102 | `Ailk_Bjlk` MT32x16x32 |

## Preprocessor MLP input layer (512 → 256), `nn.Linear`  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (+bias) | 256 | T | 512 | bf16 | TN | 1 | 0.82 | 670 | 0.8 | 141 | `Alik_Bljk` MT64x256x128 |
| dgrad | 512 | T | 256 | bf16 | NN | 1 | 7.60 | **72** | **7.6** | 104 | `Ailk_Bljk` MT32x16x32 |
| wgrad | 512 | 256 | T | bf16 | NT | 1 | 51.44 | **11** | **51.4** | 102 | `Ailk_Bjlk` MT32x16x32 |

## Preprocessor MLP input layer (24 → 256), `nn.Linear` (action encoder)  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (+bias) | 256 | T | 24 | bf16 | TN | 1 | 0.43 | 60 | 0.4 | 236 | `Alik_Bljk` MT256x240x64 |
| dgrad | 24 | T | 256 | bf16 | NN | 1 | 0.55 | 47 | 0.6 | 104 | `Ailk_Bljk` MT32x16x32 |
| wgrad | 24 | 256 | T | bf16 | NT | 1 | 48.66 | **0.5** | **48.7** | 102 | `Ailk_Bjlk` MT32x16x32 |

## Contextual-feature projection, `torch.baddbmm` over 8 contextual positions (batch 1,024)  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (batch_count 8, beta = 1) | 512 | 1024 | 512 | bf16 | NN | 1 | 0.070 | 61 | 0.07 | 104 | `Ailk_Bljk` MT32x16x32 |
| dgrad (batch_count 8) | 512 | 1024 | 512 | bf16 | TN | 1 | 0.010 | 425 | 0.01 | 165 | `Alik_Bljk` MT128x128x128 |
| wgrad (batch_count 8) | 512 | 512 | 1024 | bf16 | NT | 1 | 0.072 | 60 | 0.07 | 102 | `Ailk_Bjlk` MT32x16x32 |

## Postprocessor time-feature combiner (516 → 512), `nn.Linear` (batch 1,024)  [bf16]  (3)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd (+bias) | 512 | 1024 | 516 | bf16 | TN | 1 | 0.006 | 86 | 0.01 | 127 | `Alik_Bljk` MT64x32x256 |
| dgrad | 516 | 1024 | 512 | bf16 | NN | 1 | 0.016 | 33 | 0.02 | 104 | `Ailk_Bljk` MT32x16x32 |
| wgrad | 516 | 512 | 1024 | bf16 | NT | 1 | 0.029 | 19 | 0.03 | 102 | `Ailk_Bjlk` MT32x16x32 |

## Candidate-side MLP (1536 → 512 → 512, task head 512 → 1), `nn.Linear` (batch 1,024)  [f32]  (9)
| pass | M | N | K | dtype | layout | calls/step | ms/call | TFLOP/s | ms/step | sol | kernel in training |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| fwd 1536→512 (+bias) | 512 | 1024 | 1536 | f32 | TN | 1 | 0.032 | 50 | 0.03 | 905 | `S_B` MT32x32x16 |
| dgrad 1536→512 | 1536 | 1024 | 512 | f32 | NN | 1 | 0.030 | 54 | 0.03 | 903 | `S_B` MT32x32x16 |
| wgrad 1536→512 | 1536 | 512 | 1024 | f32 | NT | 1 | 0.028 | 58 | 0.03 | 902 | `S_B` MT32x32x16 |
| fwd 512→512 (+bias) | 512 | 1024 | 512 | f32 | TN | 2 | 0.014 | 38 | 0.03 | 905 | `S_B` MT32x32x16 |
| dgrad 512→512 | 512 | 1024 | 512 | f32 | NN | 2 | 0.013 | 40 | 0.03 | 903 | `S_B` MT32x32x16 |
| wgrad 512→512 | 512 | 512 | 1024 | f32 | NT | 2 | 0.023 | 23 | 0.05 | 902 | `S_B` MT32x32x16 |
| fwd task head (beta = 1) | 1 | 1024 | 512 | f32 | NN | 1 | 0.013 | 0.1 | 0.01 | 903 | `S_B` MT32x32x16 |
| dgrad task head | 512 | 1024 | 1 | f32 | NN | 1 | 0.004 | 0.2 | 0.00 | 903 | `S_B` MT32x32x16 |
| wgrad task head | 512 | 1 | 1024 | f32 | NN | 1 | 0.022 | 0.05 | 0.02 | 903 | `S_B` MT32x32x16 |

---
# Summary by layout (token-path GEMMs, T = 2,097,152)
| layout | role | GEMMs | ms/step | TFLOP/s range | kernels picked |
|---|---|---:|---:|---|---|
| **NN** (`Ailk_Bljk`) | forward of `addmm(b, x, W)`, dgrad of `nn.Linear` | 6 | **502** | 47–77; one exception at 472 (M = 256, K = 512) | MT32x16x32 fallback (sol 104) except M = 256 (sol 105, MT256x256x64) |
| **NT** (`Ailk_Bjlk`) | every wgrad (K = T) | 6 | **626** | 0.5–70 | MT32x16x32 fallback (sol 102), or MT256x128x64 without split-K (sol 103) |
| **TN** (`Alik_Bljk`) | forward of `nn.Linear`, dgrad of `addmm(b, x, W)` | 6 | 27 | 60–1,357 | tuned tiles (sol 141, 153, 173, 235–237) |

The large TN GEMMs on the token path reach 0.5–1.36 PFLOP/s; the NN and NT ones run at 0.5–77 TFLOP/s. The layout is what differs, not the shape. When the same math is re-expressed in the TN layout on the same GPU, it runs 12–90× faster: 935–1,979 TFLOP/s for the UVQK, output-projection and 256 × 512 wgrad shapes ([performance_analysis.md §4.2](../performance_analysis.md#42-evidence-hipblaslt-speed-on-mi450-depends-on-operand-layout-1290)).
