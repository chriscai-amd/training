# Step-101 replay after the second AC cycle, 2026-09-18

**The NaN remains unresolved.** After the user AC-cycled the host around
07:24 UTC, four direct replays of the captured step 101 completed with finite
gradients and byte-exact forward outputs. The first pair had a large finite
gradient difference **before** the layer-1 GEMM. The second pair, with every
restoration verified and complete selected-operation tensors saved, had identical
GEMM input/output bytes. Two isolated layer-norm replays also matched their
captured outputs. These are usable replay controls; they do not reproduce the
original NaN or establish a fix.

The [original report](nan_replay_20260918.md) records training, capture and the
06:25:50 MES replay hang. This report supersedes its pre-recovery status.
Small evidence files are checked in under
[evidence/replay_post_ac_20260918_0730/](evidence/replay_post_ac_20260918_0730/).
Full model captures and operation tensors remain outside Git and locally stored.
At **07:53:59 UTC**, the host was idle: no `/dev/kfd` owner, both replay services
inactive with exit 0, GFX 0%, and 0.2 GB VRAM in use. The investigation's kernel
log follower was stopped after the final log was saved.

## Stack and preserved failure

| Component | Exact tested value |
|---|---|
| Container | `triton-7ff97e-20260910` |
| Image | `recommendation-gfx1250-20260910:triton-7ff97e` |
| Image ID | `sha256:7dcd1177dac976c9af2da3e1eeaa9e2d119827879c1873c2dda97ac3b0b287d7` |
| Triton | `7ff97e310935b4a79794878dbc911f9af25d38d9`; distribution `3.8.0+git7ff97e31` |
| Torch / HIP | `2.11.0+rocm7.14.0a20260625` / `7.14.60850` |
| Configuration | Batch 1024, seed 1, `START_TS=0`, embedding row scale 0.25 |
| Existing workarounds | `AMDGCN_USE_BUFFER_OPS=0`, BLOCK_N=64 pin, pipeline clamp |
| Post-AC boot | `893e2301-265e-40d5-8fb9-0d7693d7d375`; `gpu_recovery=0`, `halt_if_hws_hang=0` |

The Torch matmul health probe was finite and the Triton JIT add had maximum
error zero ([health log](evidence/replay_post_ac_20260918_0730/health.log)). All
six original capture files were read and SHA-256 verified after AC, completing
at **07:32:48 UTC**; all **151,508,784,424 bytes** matched the existing manifest
([verification](evidence/replay_post_ac_20260918_0730/artifact_verification.json)).
This establishes local artifact integrity across the reboot, not an off-host backup.

Original artifact:
`/home/chcai/dlrm_data/nan_replay_long_20260918_0520/capture_step_000101`.
It contains complete pre-forward boundaries 100 and 101, both input batches,
dense/fused optimizer state and RNG, including **150,042,474,496 embedding
bytes** and the undo journal needed to reconstruct boundary 100.

The original run stopped at `backward_before_clip`: forward loss
`0.13949690759181976`, predictions, labels and weights were finite, but **38
parameter gradients were nonfinite**. Layer 2 was healthy. Layer 1's input-norm
weight/bias gradients were bad while its UVQK/output parameter gradients were
finite; layer 0 and earlier preprocessing gradients were bad. Clipping and the
dense optimizer update had not run. Fused embedding updates can occur during
backward; the saved pre-step state is independent of those writes.

The capture retains bad-gradient names, not the original bad-gradient values or
intermediate operation tensors. An after-backward scan cannot exclude a later
wild store into an earlier gradient. This limits attribution from that pattern.

## Completed replay findings

| Experiment | Result | Scope of the evidence |
|---|---|---|
| Initial post-AC pair, `replay_report.json` | Two direct attempts, exit 0; exact forward output/loss and finite parameter gradients | Repeat 0 restored bytes verified; repeat 1 restored bytes were not independently checked |
| Initial boundary summaries | `duvqk` maximum differs by 12,841.5× between repeats | Difference exists at the GEMM input; complete finite tensors from this pair were not persisted |
| Save-all pair, `saved_operations_report.json` | Two direct attempts, exit 0; all large/small state bytes, controls and RNG verified each time; exact forward and finite gradients | Four complete finite operation dumps retained: GEMM and LN for each repeat |
| CPU comparison of saved GEMMs | All six input/output tensors and raw backing-storage pairs match | **12,043,722,752 logical bytes**, zero differing elements or rows |
| Isolated saved LN replay | Two fresh-input GPU attempts, exit 0; all three full-output hashes equal the capture and each other | Tests the complete LN operation on the saved finite control inputs |
| LN FP64 reference | Rows 0–15, all 512 columns: 8,192 `d_x` elements equal after BF16 casting | Full parameter-gradient FP64 reductions were not computed; reported weight/bias references are partial row contributions |

Evidence: [initial report](evidence/replay_post_ac_20260918_0730/replay_report.json),
[initial boundary summaries](evidence/replay_post_ac_20260918_0730/boundaries_initial.jsonl),
[verified save-all report](evidence/replay_post_ac_20260918_0730/saved_operations_report.json),
[GEMM byte comparison](evidence/replay_post_ac_20260918_0730/gemm_repeat0_vs_repeat1_comparison.json),
and [isolated LN replay](evidence/replay_post_ac_20260918_0730/ln_operation_replay.json).

| Layer-1 absolute maximum | Initial repeat 0 | Initial repeat 1 | Save-all repeats 0 and 1 |
|---|---:|---:|---:|
| `dz` / `duvqk`, before GEMM | `1.722946763e-7` | `0.002212524414` | `1.722946763e-7` |
| GEMM `d_normed_x` | `7.543712854e-8` | `0.000602722168` | `7.543712854e-8` |
| UVQK weight gradient | `1.525878906e-5` | `0.017211914063` | `1.525878906e-5` |
| LN `d_x` | `3.070454113e-9` | `2.706050873e-5` | `3.070454113e-9` |

In the saved pair, the identical `dz` U/V/Q/K block maxima were respectively
`1.722946763e-7`, `1.367880031e-9`, `1.695007086e-7` and `4.329194780e-10`.
The tensor has shape `[1959558, 2048]`, BF16, stride `[2048, 1]`:
**8,026,349,568 bytes**. Complete tensors are under
`/home/chcai/dlrm_data/nan_replay_post_ac_20260918_0730/boundaries_saved/`.
The four `.pt` files total **36,158,352,092 bytes**; all were read and hashed in
the [operation capture manifest](evidence/replay_post_ac_20260918_0730/operation_captures_manifest.json).
They remain local-only and should accompany the complete failure capture in
off-host artifact storage.

The first pair's change is much larger than the one-ULP differences in healthy
step-3 replay. However, its second restoration was unverified and no complete
finite operation dump survives that pair. Full verification and persistence
were both added for the second pair; the divergence disappeared. Their effects
on allocation, synchronization, timing and process history are not separated.
No kernel fix or controlled causal A/B was applied.

## What the source audit establishes

The other host's finite incoming gradients around `6.9e37` motivate checking
extreme finite values before NaN. Its tensors have not been inspected here, so
its proposed jagged-split culprit is a hypothesis to test on this host.

Here the fused path in
[triton_hstu_preprocess_and_attention.py](../../generative_recommenders/ops/triton/triton_hstu_preprocess_and_attention.py)
allocates `duvqk`, makes U/V/Q/K column views, fills DQ/DK/DV in attention
backward, and fills DU with `aten.silu_backward`. The nearby `.split()` calls
are ordinary views. Then
[triton_addmm_bwd](../../generative_recommenders/ops/triton/triton_addmm.py)
uses **`torch.mm(x.T, dz)`** and **`torch.mm(dz, w.T)`** for the weight and input
gradients, followed by weighted layer norm. Despite its wrapper name, this GEMM
dispatches through PyTorch/ROCm BLAS.

No intentional unwritten DU/DK/DV region was found in the selected eager pinned
path. DQ is a load/add/store accumulator, but `_bwd_pre_hook` zeros it; the
pinned configs retain that hook and the installed Triton calls it on cached and
singleton launches too. DK/DV accumulators start at zero and all valid rows are
stored. Global sequence offsets and head addressing widen to int64. The remaining
sequence-local products fit int32 at the configured history limit 8192. The
PyTorch SiLU TensorIterator path splits launches when element count or byte
extent exceeds its 32-bit indexing limit.

These are source-level coverage checks, not executed-kernel proofs. They assume
valid monotonic offsets spanning the tensor, valid sequence-sort permutations,
and correct compiler/runtime behavior. They do not rule out corrupted metadata,
miscompilation, stale reads, wild writes, or execution-history dependence. If
testing unwritten outputs, poison DU/DK/DV while retaining DQ's required zero
initialization; check DU only after SiLU has executed. Initialization changing
the symptom would still require an explanation.

The replay audit found all 81 saved parameter gradients were `None`, and restore
explicitly restores gradient presence; missing `zero_grad()` is not a demonstrated
bug. Inputs are rebuilt as fresh KJTs and default RNG is restored last. TorchRec
DDP reducer iterations/buckets and allocator/stream/driver history are not
snapshotted. DDP uses static-graph and bucket-view behavior that can change
allocation/scheduling across repeats; no gradient-scaling bug was identified.

There are two different tests to complete. **Known-input numerical replay**
restores values/layouts and compares an operation against a reference; the saved
finite LN control now supports that. **Execution-history reproduction** must
recreate the conditions that produced the original bad tensors. Four finite
direct replays do not establish that condition or clear attention, SiLU, GEMM,
layer norm, TorchRec, compiler, driver or hardware.

## Exact resume commands

Run GPU commands only when the previous workload has released `/dev/kfd`; keep
one GPU workload at a time. The replay tools restore the captured environment
and keep model-source/stack checks enabled. These commands use fresh report paths
and preserve the original capture. The host data directory is mounted as
`/data/mlperf_dlrm_v4` inside the paired container.

CPU-only full GEMM comparison, which can also be used on a future divergent pair:

```bash
task_data=/data/mlperf_dlrm_v4/nan_replay_post_ac_20260918_0730
task_prefix=$task_data/boundaries_saved/boundary-2946-1789717407684763312-current
task_stamp=$(date -u +%Y%m%d_%H%M%S)
docker exec -w /workspace/recommendation \
  -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
  triton-7ff97e-20260910 python -B scripts/replay_backward_boundary.py \
  "$task_prefix-r0-s101-0003-triton_addmm_bwd-finite.pt" \
  --compare-other "$task_prefix-r1-s101-0003-triton_addmm_bwd-finite.pt" \
  --compare-only --report "$task_data/gemm_comparison_$task_stamp.json"
```

Repeat the complete saved LN operation with fresh input backing storage on each
attempt and the bounded FP64 control comparison already validated:

```bash
task_data=/data/mlperf_dlrm_v4/nan_replay_post_ac_20260918_0730
task_stamp=$(date -u +%Y%m%d_%H%M%S)
docker exec -w /workspace/recommendation triton-7ff97e-20260910 \
  python -B scripts/replay_backward_boundary.py \
  "$task_data/boundaries_saved/boundary-2946-1789717407684763312-current-r0-s101-0004-triton_weighted_layer_norm_bwd-finite.pt" \
  --gpu --repeats 2 --rows 0:16 \
  --context /data/mlperf_dlrm_v4/nan_replay_long_20260918_0520/capture_step_000101/context.json \
  --report "$task_data/ln_operation_$task_stamp.json"
```

For a full CPU LN reference, omit `--gpu` and replace `--rows 0:16` with
`--rows all --max-rows 0`. Only that all-row mode computes complete norm
weight/bias reductions. To test the saved GEMM, replace the dump's
`0004-triton_weighted_layer_norm_bwd` with `0003-triton_addmm_bwd`; full GPU GEMM
replay was not part of this completed validation.

The next full-model experiment can compare step 100 → 101 with the direct
controls. It starts from captured boundary 100, executes its update, then runs
101 without replacing the intermediate state:

```bash
task_run=/data/mlperf_dlrm_v4/nan_replay_resume_$(date -u +%Y%m%d_%H%M%S)
docker exec triton-7ff97e-20260910 mkdir -p "$task_run"
docker exec -w /workspace/recommendation \
  -e NAN_BACKWARD_TARGET=_stu_layers.1. -e NAN_BACKWARD_ABS_THRESHOLD=1e20 \
  -e NAN_BACKWARD_SAVE_ALL=1 \
  triton-7ff97e-20260910 python -u scripts/replay_training_step.py \
  /data/mlperf_dlrm_v4/nan_replay_long_20260918_0520/capture_step_000101 \
  --mode previous --repeats 2 --verify-every-repeat \
  --backward-probe-dir "$task_run/boundaries" --report "$task_run/report.json"
```

For fresh-process direct controls, run the same command twice with
`--mode current --repeats 1` and a different `task_run` each time. For the
same-process comparison use `--mode current --repeats 2`. Retain all restore and
intermediate-state differences. An extreme/nonfinite boundary stop exits 86
after the operation dump is persisted; exit 0 alone does not establish equality.

If the failure still depends on training history, the next recorder integration
should capture attention/SiLU boundaries in the live run and preserve the existing
pre-step CPU snapshot when a boundary probe stops. That integration is proposed,
not implemented by these replay-only probes. The next decisive artifact is a
complete anomalous operation input/output pair, followed by an isolated reference
comparison and a controlled component A/B.
