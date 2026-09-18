# NaN retest after AC cycle — 2026-09-18

**The early NaN and the previous step-211 hang did not reproduce in four fresh
uninstrumented runs on the intended image/Triton pairing.** Three runs completed
20 training steps each; a fourth completed 319. All 379 logged training losses
were finite, all four processes exited 0, and GPU memory returned to about
0.2 GB between runs. This is a bounded negative result, not a fix or convergence
claim. The separate START_TS=150 NaN near step 1340 was not tested.

## Stack and recovery

The user performed an AC cycle before this series. The amdgpu module was not
loaded after boot; it was loaded with `gpu_recovery=0 halt_if_hws_hang=0`.
The latter avoids the deliberate debug halt loop diagnosed in
[the hang handoff](hang_triage_20260918.md); it does not fix a MES timeout.

- Image: `recommendation-gfx1250-20260910:triton-7ff97e`.
- Container: `triton-7ff97e-20260910`.
- Triton git HEAD: `7ff97e310935b4a79794878dbc911f9af25d38d9`.
- Torch: `2.11.0+rocm7.14.0a20260625`; HIP: `7.14.60850`.
- `HIPBLASLT_TENSILE_LIBPATH` points to the image's `library/gfx1250` directory.
- `AMDGCN_USE_BUFFER_OPS=0`; existing BLOCK_N and pipeline workarounds retained.

The initial `probe_stack.py` check passed: 4096² matmul finite, Triton addition
maximum error 0. The base image's stock Triton 3.6.0 was not used.

## Results

All runs use batch 1024, START_TS=0, seed 1, embedding row scale 0.25,
`expandable_segments:True`, and per-step loss logging. Interim evaluation and
checkpoint writes are disabled (`CKPT_PATH=`); final evaluation is capped at
one batch. Run names are unique, so no old checkpoint is resumed. Runs finish
normally rather than being killed at the last training step.

| Run | Windows | Training steps | Non-finite loss | Exit | Last training loss |
|---|---:|---:|---:|---:|---:|
| baseline01 | 8 | 20 | 0 | 0 | 0.13042372465133667 |
| baseline02 | 8 | 20 | 0 | 0 | 0.13042329251766205 |
| baseline03 | 8 | 20 | 0 | 0 | 0.13042370975017548 |
| extended01 | 21 | 319 | 0 | 0 | 0.1397104114294052 |
| probe01, incomplete embedding coverage | 8 | 20 | 0 | 0 | 0.1304224729537964 |
| probe02, actual TBE weights checked | 8 | 20 | 0 | 0 | 0.1304231435060501 |

The longest run lasted 12m53s including initialization and final evaluation.
It crossed step 211 without stopping. Continuous dmesg capture recorded no MES
completion timeout, ring-full error, or GPU memory fault during these runs;
CPU workqueue-duration warnings were present. The short baselines differed by
at most 4.36e-6 in their logged full-precision losses.

The MLPerf `run_stop` status is `aborted` because this short test does not reach
the holdout accuracy target. It is separate from process exit status and is
not a NaN or execution failure.

## Diagnostic corrections

The original module probe could miss the actual embedding weights and could
misclassify a producer. It has been repaired to:

- Traverse TorchRec's hidden `_lookups` list to reach native FBGEMM TBE modules.
- Inspect actual table views in bounded chunks, and distinguish configured
  state from checks completed by executed hooks.
- Preserve input/weight state before the first bad output, including kwargs.
- Report bad weights even if the currently selected rows return finite output.
- Read/reuse a pinned host snapshot only after its CUDA completion event.
- Avoid materializing lazy awaitables and pause outside training forwards.

`probe01` completed with finite loss but reported `tbe_buffers=0`; it is not
evidence that embedding weights were checked. The hidden `_lookups` traversal
was added after that run. A focused injected-NaN test checks output-origin
classification, propagation, parameter and TBE corruption, delayed snapshots,
hidden module traversal, and error handling. Full-table scans perturb
scheduling, so instrumented results remain separate from the baseline rate.

**The repaired probe was validated on the real trainer in probe02.** All 20
steps completed with finite loss and exit 0, without a probe disable/error or
non-finite report. The event-confirmed coverage line is:

```text
completed coverage step=1 executed_modules=43/57 inspected_weight_tensors=80
inspected_weight_numel=37525861681 tbe_buffers_inspected=1/1
```

Those 80 tensors are the 69 dense parameter tensors and 11 embedding-table
views. The latter cover the full **37,510,618,624-element** fp32 TBE buffer.
The unexecuted registered embedding parameter owners are aliases used for
checkpointing; the native TBE views provide actual compute-path coverage.
Checks run before each training forward, so this does not claim inspection of
the last backward update's entire resulting state. Opaque lazy-embedding
awaitable output is explicitly marked incomplete; native TBE outputs and the
subsequent dense-module boundaries are checked. Optimizer state is not checked.

Both focused GPU self-tests passed, including the added hidden-lookup case in
`probe_selftest_v2.py`. Syntax and whitespace checks passed. At completion no
trainer or dataloader processes remained, GFX utilization was 0%, and reported
VRAM use was 0.2 GB. The runtime driver flags remain `gpu_recovery=0` and
`halt_if_hws_hang=0`.

The [source/capture review](nan_source_review_20260918.md) also corrects two
misleading assumptions in the older notes. The saved failing batches have
1024 samples and selected history length 1697, with heavily repeated embedding
IDs; MIN_HISTORY=4086 does not force 4086 selected UIH entries. The large step-8
normalized entropy is explained by an all-negative metric window's almost-zero
denominator. The training `REACHED AUC` message does not terminate MLPerf
training. Neither observation identifies the original NaN producer.

## Evidence and limits

Artifacts: `/home/chcai/mi450_logs/nan_post_ac_20260918_0347/`.
The directory contains each training/driver log, exact runner scripts,
`health.log`, `dmesg_live.log`, image/container and boot identity, a JSON
summary generator, and probe self-tests with their output.

This series does not establish that Triton alone fixed the older NaN: the
historical early-NaN runs used Triton 3.6.0, and image and boot state also differ.
One run passing step 211 does not exclude a nondeterministic MES timeout.
No fresh bad tensor was obtained, so the original NaN culprit remains unknown.
If it returns, the repaired probe can distinguish an existing bad embedding
weight from a finite-input module producing a bad output; the next A/B can then
target that component instead of swapping the entire kernel stack.
