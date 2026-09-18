# Shared MI450 A0/B0 investigation code, September 18

The reconciliation starts from pushed A0 commit `de75cf8` and incorporates the
saved B0 work based on `a6b7c50`. The seven A0 commits are a linear continuation;
only two trainer hook regions conflicted. The result preserves A0's full-state
capture/replay, operation replay, configurable Docker base image, and docs,
and adds B0's operation tripwire, standalone reproducers, and evidence.

**The NaN remains unresolved on both hosts.** This integration does not establish
a shared culprit or satisfy the 3,000-optimizer-step validation requirement.
The detailed evidence remains in the [A0 record](mi450_a0/mi450_a0.md) and
[B0 record](mi450_b0/mi450_b0.md).

## Runtime settings

Both A0 and B0 report `gfx1250`; the architecture string cannot select an
investigation arm. The shared code retains the existing `BLOCK_N=64` pin for
HIP gfx1250 with Triton >= 3.8. The register cap is now explicitly opt-in.

| Setting | A0 current baseline | B0 capped investigation |
|---|---|---|
| `HSTU_BWD_MAX_VGPR` | Unset or `0` | **`256`** |
| `HSTU_BWD_BLOCK_N` | Unset, effective 64 | Unset, effective 64 |
| `TRITON_FULL_AUTOTUNE` | `0` | `0`; full autotuning bypasses the cap |
| `AMDGCN_USE_BUFFER_OPS` | `0` | `0` |
| `TRITON_ALLOW_PIPELINING` | `0` | `0` |
| `PYTORCH_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF` | Empty | Empty |
| `HSA_ENABLE_COREDUMP` | `0` | `0` |

Set these before importing model/Triton modules; decorators construct the
kernel configs at import time. Other architectures and older Triton retain
their existing pinned configurations. The B0 cap suppresses the captured
attention-gradient corruption but has already failed longer training runs.

Earlier B0 source snapshots defaulted to 256. For those runs, an absent
`HSTU_BWD_MAX_VGPR` does not mean uncapped. Read the saved effective kernel
configuration and explicitly choose 256 or 0 when replaying with shared code.
The attention capture runner respects an explicit environment value; otherwise
it uses a captured value when present, then the shared default of 0. The
specialized LN-from-attention runner still requires cap256 during preparation
and checks that it is effective.

The hosts keep their respective prepared images and native builds. Their
reported Python stacks match (Torch `2.11.0+rocm7.14.0a20260625`, HIP
`7.14.60850`, Triton `7ff97e310935b4a79794878dbc911f9af25d38d9` / 3.8.0),
but the image names and image IDs differ. Preserve the configurable
`BASE_IMAGE` in `Dockerfile.amdprimus0815`; a base image alone does not include
the installed FBGEMM/TorchRec/MLPerf additions or the custom Triton build.

## Choosing a diagnostic

All diagnostic paths remain disabled unless explicitly enabled. To run a clean
baseline, unset `NAN_REPLAY_DIR`, `NAN_TRIPWIRE_DIR`, `NAN_CAPTURE_STEP`, and
`NAN_CAPTURE_ON_FIRST`, and set `NAN_MODULE_PROBE=0` and `NAN_TRIPWIRE=0`.

| Diagnostic | Enable control | Implementation / artifact |
|---|---|---|
| A0 full training state | `NAN_REPLAY_DIR=/fresh/directory` | `scripts/nan_replay_capture.py`; complete pre-step inputs, dense/sparse weights, optimizer state, RNG |
| A0 module probe | `NAN_MODULE_PROBE=1` | `scripts/nan_module_probe.py`; module boundary reports |
| A0 legacy loss/operation probe | `NAN_TRIPWIRE=1` plus `scripts/autoload` on `PYTHONPATH` | `scripts/nan_tripwire.py`; legacy catch dumps |
| A0 lightweight snapshot | `NAN_CAPTURE_STEP` or `NAN_CAPTURE_ON_FIRST` | `scripts/nan_capture.py`; lightweight diagnostic snapshot |
| B0 custom-autograd operation capture | `NAN_TRIPWIRE_DIR=/output/directory` | `generative_recommenders.dlrm_v4.train.nan_tripwire`; operation inputs, saved context, outputs, metadata |

The two `nan_tripwire.py` files have distinct import namespaces and activation
keys. No renaming is required. Captures have different formats and must use
their corresponding replay entry point: `replay_training_step.py` for A0 full
state, `replay_nan_dump.py` for the legacy A0 dump, and
`replay_nan_tripwire.py` for the B0 operation payload. The compact attention
prefix format is accepted by `repro_hstu_attention_capture.py` only.

The audit found a preexisting legacy A0 counter difference: wrapping both
`Tensor.backward` and `torch.autograd.backward` advances its `STEP` twice for
one `Tensor.backward()` call. B0 and the full-state recorder use the trainer's
global step. Use trainer global-step logs when comparing hosts; legacy probe
numbers are not directly comparable. Its first-input gradient hooks can also
observe accumulated contamination and do not by themselves identify a producer.
The reconciliation preserves that legacy behavior and its historical evidence.

Prefer one primary recorder per experiment so its timing and failure behavior
are clear. Combining diagnostics adds instrumentation and does not guarantee
two capture files. Full-state replay retains its existing single-rank
requirement and rejection of `NAN_MODULE_PROBE=1`. Its A0 capture contains
150,042,474,496 embedding bytes, before other state and snapshot overhead;
the roughly 79 GiB free on B0 at reconciliation time cannot hold an equivalent
capture. Allocate sufficient backing storage before enabling it on B0.

## Trainer ordering and termination

The merged trainer performs exactly one forward, backward, and optimizer step
on the finite path. Its diagnostic order is:

1. Apply LR warmup, begin B0 tripwire, clear gradients, move the sample.
2. A0 full-state `before_step`, then optional module-probe installation.
3. Forward; A0 `after_forward`; lightweight capture and module-probe pump/pause;
   B0 loss observation and optional immediate forward check.
4. Backward; A0 `after_backward`; B0 dense-gradient observation/check.
5. Gradient clipping, optimizer step, B0 parameter observation/check.
6. Metrics; deferred B0 check only at its configured metric boundary; A0
   module-probe polling every step.

A0 full-state capture therefore has priority at completed forward/backward
boundaries: it persists before raising `SystemExit`. If B0 subsequently detects
an operation anomaly or a large finite value, it saves its own report/payload
and raises its diagnostic exception. An exception inside forward/backward can
prevent either boundary hook from running. A B0 magnitude-only failure does
not automatically freeze A0 state. Deferred B0 checks intentionally occur
after the optimizer and must match `METRIC_LOG_FREQ`; immediate checks occur
before clipping. These are instrumentation modes, not equivalent schedules.

Module/capture script imports now derive their location from `utils.py`, as the
full-state hook already did. This supports either host's repo mount and isolated
worktrees without requiring `/workspace/recommendation`.

## Replay provenance and retained work

A0 replay checks hashes of captured `generative_recommenders` sources by
default. The merged `train/utils.py` and attention configuration differ, so old
A0 captures correctly reject the shared tree. Use a worktree containing the
exact captured source for a baseline replay, including any recorded local
patches. Use `--allow-source-change` only for an intentional comparison and
retain the `changed_source` / `changed_model_source` report. The source guard
has not been weakened. Capture manifests also record script hashes; the
current replay guard enforces the model-source subset only.

The original B0 checkout at `/home/chcai/training` and its local edits remain
untouched. The reconciliation is in `/home/chcai/training-mi450-reconcile`,
branch `chcai/mi450-a0-b0-reconcile-20260918`. The shared branch for both hosts
is `chcai/mi450_a0`. The existing B0 container still
mounts the original checkout. Validation uses a separate candidate copy under
`/tmp/mi450-reconcile-validation/recommendation` inside that container.

The B0 persistence bundle remains at
`/home/chcai/runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/wrapup_20260918/`.
It includes the original patch/source snapshots, capture hashes, continuation
notes, pending checker/staged-replay proposals, and verified native archives.
Large captures stay at their recorded paths. The experimental fused checker
and output-backward stage runner remain saved proposals outside this merge.
No reset, reboot, or long training run was performed during reconciliation.

## Validation

Run from `recommendation/` in the prepared image:

```bash
python scripts/test_nan_replay.py
python -m unittest generative_recommenders.dlrm_v4.train.tests.nan_tripwire_test generative_recommenders.dlrm_v4.train.tests.replay_nan_tripwire_test
python -m unittest generative_recommenders.dlrm_v4.train.tests.nan_diagnostics_compat_test
```

The first suite validates full-state byte restoration, aliasing, optimizer and
RNG replay without initializing CUDA. The second validates the operation
capture/replay semantics. The compatibility suite executes the actual trainer
step statements with lightweight stand-ins to test hook ordering, termination,
and diagnostic modes without a distributed training setup; it also checks the
pinned config selection for A0/B0 and other supported targets.

Completed against the reconciled source on September 18:

- **84 CPU tests passed:** 3 A0 full-state replay, 40 B0 operation capture/replay,
  21 trainer/config compatibility, and 20 existing checkpoint/eval-cadence tests.
- **25/25 GPU calls passed** using the real 32-sequence attention prefix with
  `HSTU_BWD_MAX_VGPR=256` on B0. The actual selected config contains
  `llvm_fn_attrs=amdgpu-num-vgpr=256`, `BLOCK_N=64` and the production pre-hook.
- A fresh module import with the cap **unset** selects `BLOCK_N=64` and no VGPR
  attribute on gfx1250. This preserves A0's existing config selection; it is
  not an A0 hardware run.
- All **127 Python files** in the tested container copy hash-match the candidate;
  the new compatibility test was separately run on the host. Syntax and
  whitespace checks passed. All 12 original B0 snapshot files remain unchanged.

Logs, the GPU report, and source manifest are under
`/home/chcai/runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/reconcile_20260918/`.
The first two CPU suite outputs are retained in the session tool record; the
compatibility and existing-trainer logs are also saved in that directory.
No A0 GPU run, long training test, full-state B0 capture, or archive-restoration
test was performed. The short component replay does not establish a NaN fix.
