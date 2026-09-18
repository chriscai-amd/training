# Full-state NaN capture and replay, 2026-09-18

**Updated 2026-09-18 after the second AC cycle:** the capture-enabled run caught
nonfinite gradients at step **101** and saved complete adjacent pre-step states.
The first direct replay hung with AMDGPU/MES failures. After recovery, the health
probe and four direct step-101 attempts completed: forward outputs matched the
capture exactly and gradients remained finite. The first replay pair showed an
unexplained roughly 10,000-fold change in the maximum gradient entering the
layer-1 GEMM. The next pair, with full restoration verification and saved
operation tensors, had byte-identical GEMM inputs and outputs. Two isolated
layer-norm replays also matched the saved outputs exactly. Neither the original
NaN nor its culprit is reproduced. See the [second-AC experiment and resume
record](nan_replay_post_ac_20260918_0730.md) for evidence and limitations.
The complete failure artifact passed another full SHA-256 check after recovery;
it and the operation tensor dumps are still stored locally only.

The post-AC long run **reproduced NaN at global step 775**, after 774 finite
losses, at batch 1024 / `START_TS=0`. The run was stopped at step 797
(`EXIT=137`, deliberate termination). No MES timeout or HSA GPU memory fault
was recorded. These results do not establish a fix or identify a culprit.

Evidence: `/home/chcai/mi450_logs/nan_long_20260918_0417/`, especially
`status.json`, `training.log`, and `dmesg.log`. The kernel also recorded corrected
CPU machine-check events on this boot, including before this run. There is no
evidence here connecting those corrected events to the NaN.

That run saved no complete preceding state. It cannot produce a one-step
reproducer of step 775. The follow-up therefore captures the actual training
state at every pre-forward boundary rather than relying on another finite run.

## Stack

- Container: `triton-7ff97e-20260910`.
- Image: `recommendation-gfx1250-20260910:triton-7ff97e`.
- Image ID: `sha256:7dcd1177dac976c9af2da3e1eeaa9e2d119827879c1873c2dda97ac3b0b287d7`.
- Triton: `7ff97e310935b4a79794878dbc911f9af25d38d9`, imported from
  `/opt/triton-custom/python/triton/__init__.py`.
- Torch: `2.11.0+rocm7.14.0a20260625`; HIP: `7.14.60850`.
- Batch 1024, seed 1, embedding row scale 0.25, `START_TS=0`;
  `AMDGCN_USE_BUFFER_OPS=0`, existing BLOCK_N=64 and pipeline clamp.
- `HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`.

## Capture contract

`NAN_REPLAY_DIR` enables the recorder in the streaming trainer. The boundary is
after learning-rate warmup, `zero_grad()`, and input transfer, but **before
forward**. It records:

- Both adjacent batches' UIH/candidate KJTs: values, lengths/offsets, keys,
  weights, stride, variable-stride metadata, and inverse indices.
- Every raw parameter and buffer, including nonpersistent buffers and hidden
  TorchRec lookup/TBE modules. Embedding aliases share one physical snapshot.
- All 11 embedding tables, the fused rowwise Adagrad accumulators, dense Adam
  state, optimizer groups, actual TBE kernel LR, `iter`, `iter_cpu`, and controls.
- Python, NumPy, Torch CPU and default CUDA RNG states, training flags,
  gradient presence, effective gin/environment configuration, source hashes,
  software versions, and observed predictions/labels/weights/losses.

The 150,042,474,496-byte embedding backing uses one CPU shadow. Every byte is
compared at each boundary; old bytes from each changed 16 KiB page go into a
rotating disk undo journal. This catches writes outside intended/touched rows.
The current shadow plus the journal reconstructs the preceding boundary.
Other state is small enough for complete copies of both boundaries (about
476 MB each in the forced capture), including scratch buffers whose shapes
change. Tensor aliases, shapes, strides, offsets and dtypes are retained.

GPU copies use bounded chunks; the full shadow stays pageable. Only on a
trigger is the full embedding base written to disk. An incomplete artifact has
no `COMPLETE.json` and cannot be replayed. Capture failures stop the run instead
of silently dropping evidence.

The recorder stops and saves on nonfinite forward outputs, nonfinite gradients
before clipping, or nonfinite floating model/optimizer state at a boundary.
Initial embedding state is scanned fully on CPU; after that, all changed pages
are scanned. This is valid because unchanged pages were already checked and the
recorder stops at its first hit. `NAN_REPLAY_FORCE_STEP=3` exercises persistence
with healthy outputs.

## Replay and validation

**Complete state capture does not guarantee a one-step NaN replay.** That would
require both complete relevant state and deterministic execution. This recorder
preserves model/input/optimizer/default-RNG state, but not allocator history,
concurrent execution timing or GPU/driver state. Healthy backward transitions
already show small numerical nondeterminism. Step-101 replay completed after the
second AC cycle with exact forward outputs and finite gradients, so repeatable
one-step NaN reproduction is still unproven. The separate large backward
discrepancy in the first post-recovery pair remains unexplained.

`scripts/replay_training_step.py` rebuilds the same model without dataloaders.
It initializes TorchRec's lazy input-distribution metadata from the captured
feature keys, restores the controls and complete bytes, and restores RNG last.

`--mode current` restores step N directly. `--mode previous` restores N−1,
executes forward/backward/clipping/optimizer, applies N's saved learning-rate
policy and clears gradients, then runs N **without replacing the intervening
weights, moments, counters or RNG**. Its report compares the intermediate
state against the captured N boundary and compares output bytes. These checks
test determinism; they do not assume it.

```bash
docker exec -w /workspace/recommendation triton-7ff97e-20260910 \
  python scripts/replay_training_step.py \
  /data/mlperf_dlrm_v4/nan_replay_validate_20260918_0504/capture_step_000003 \
  --mode both --repeats 2
```

Only run replay after the trainer has exited and released the GPU. This host
must not run simultaneous GPU workloads.

CPU validation covers unexpected writes outside exposed views, aliases across
dtypes, partial pages, journal rotation, truncated journals, rollback after a
partial write, exact NaN payloads/signed zero, optimizer restoration, RNG,
variable-stride KJTs and dynamic scratch storage. The retained regression command
is `python scripts/test_nan_replay.py` in the training image.

Forced captures with the actual full embedding storage are under
`/home/chcai/dlrm_data/nan_replay_validate_20260918_{0500,0504}/`; their run logs
are under the corresponding directories in `/home/chcai/mi450_logs/`.
Both saved step 3 with finite outputs and exited cleanly. Boundary-copy times
were generally about 19–20 seconds after initialization, with a 93-second
outlier; this prompted use of a reusable pinned transfer chunk.

The first GPU replay validation completed with exit 0:

| Check | Result |
|---|---|
| Restore current embedding state | All 150,042,474,496 bytes equal; zero mismatches |
| Restore previous state using undo | All embedding bytes and all small state equal |
| Direct step-3 replay, twice | All captured output bytes equal |
| Step 2 → step 3, twice | Both steps' output bytes equal; controls and RNG equal |
| Recreated intermediate embedding update | **Not bitwise deterministic:** 10 / 9 bytes differ in 9 pages |
| Other differing intermediate state | Rowwise Adagrad accumulator; timestamp-embedding Adam moments |

Report: `nan_replay_validate_20260918_0504/capture_step_000003/replay_report.json`.
The backward update therefore does **not** satisfy an assumption of bitwise
determinism on this stack, even though all replayed forward outputs matched.
The timestamp backward uses relaxed floating atomic additions in
`_add_embeddings_bwd_kernel`; different accumulation orders can produce such
differences. These healthy-state differences do not identify the NaN culprit.
A further replay with fresh inputs also matched both steps' output bytes, with
controls and RNG equal. It differed in 13 embedding bytes across 12 pages; all
12 reported weight differences were one FP32 ULP apart. The maximum absolute
differences were `4.33065e-8` in rowwise Adagrad, `3.55271e-13` in the timestamp
embedding's first Adam moment and `1.99053e-20` in its second moment. Its report
is `replay_fresh_report.json` in the same directory. The artifact thus restores
exactly, but the backward transition itself has small numerical nondeterminism.

## Capture-enabled long run

The follow-up used the same batch/seed/stack with 35 windows (3,133 expected
training steps, verified from all 35 anchor counts with dropped partial batches).
It started at **05:27:04 UTC** and stopped on nonfinite gradients at step 101,
with no runtime GPU fault during the capture run. The updated pinned-buffer
recorder took 13.90 and 12.26 seconds for the second and third pre-step snapshots.
Its independent service was
`mi450-nan-replay-long-20260918-0520.service`; logs, launch commands and an exact
source archive are in `/home/chcai/mi450_logs/nan_replay_long_20260918_0520/`.
Recorder status and captures are in
`/home/chcai/dlrm_data/nan_replay_long_20260918_0520/`.

On its first nonfinite state/output/gradient, the trainer saves and exits. The
supervisor was intended to wait for GPU release, then perform two direct and
two preceding-step replays. Its guard instead deferred replay because of a
service environment error; the actual manual replay and failure are below. The
recorder's `status.json` and the capture's `COMPLETE.json` identify successful
evidence capture; a deliberate nonfinite stop can make the training driver
return nonzero before it logs the bad loss. This run captured a real failure;
it did not complete the planned 3,133 steps.

## Step-101 capture and replay hang

The run stopped at `backward_before_clip` with **38 nonfinite parameter
gradients**, while loss (`0.13949690759`), predictions, labels and weights were
finite. Step 100's loss was `0.13982370496`. It stopped before gradient clipping
and the dense optimizer update; the fused embedding optimizer can update during
backward. The complete pre-forward artifact remains independent of those writes.

Artifact: `/home/chcai/dlrm_data/nan_replay_long_20260918_0520/capture_step_000101`.
`COMPLETE.json` and `state/manifest.json` confirm current boundary 101, previous
boundary 100, all **150,042,474,496** large-state bytes, and an undo journal of
20,278 changed pages (332,234,752 payload bytes). `capture.pt` contains both
frames' small state, inputs, controls, RNG and forward observations.

The gradient names narrow the next inspection boundary:

| Region | Parameter gradients at capture |
|---|---|
| HSTU layer 2 | All finite |
| HSTU layer 1 | Input norm weight/bias nonfinite; UVQK and output-side gradients finite |
| HSTU layer 0 and earlier | Layer 0's seven parameters and upstream preprocessing/positional parameters nonfinite |

These checks ran after backward, so they do not exclude a later wild store into
an earlier gradient. No individual bad-gradient values were saved in this
pre-step artifact.

The automatic supervisor deferred replay because its service PATH lacked
`amd-smi`, and its user lacked the interactive render/video access groups. This
was a guard failure, not evidence of a busy GPU. The guard now uses
`sudo -n /opt/rocm/bin/amd-smi monitor`; the original launcher is preserved as
`run.initial.sh`. Do not restart that training service against the existing
capture directory.

At **06:23:23 UTC**, a separate direct-replay service started after an idle GPU
check. It restored state, entered byte verification, and reached an embedding
forward warning at 06:25:39. At **06:25:50** the driver reported an
`INVALIDATE_TLBS` timeout, followed by failed `REMOVE_QUEUE` and `ADD_QUEUE`,
"MES might be in unrecoverable state" and "GPU recovery disabled."
No `replay_current_report.json` was produced. The log does not identify the exact
GPU operation that stopped progressing; the embedding warning is not attribution.

Host PID 83793/container PID 24250 was sampled before termination. All 251
userspace samples were in `rocr::core::InterruptSignal::WaitRelaxed`; the hot loop
uses `MWAITX` and polls an HSA signal. `halt_if_hws_hang` was already **0**.
This differs from the earlier step-211 kernel halt loop. SIGTERM stopped the main
thread, but another thread remained in GPU-memory teardown through
`amdgpu_gart_unbind` and MES fence/TLB polling. GPU usage remained 100% with
174.1 GB allocated. No reset or reboot was performed.

By **06:41 UTC**, the replay process had exited with status **143** following
SIGTERM, and the CPU regression process also finished with status 0 after a
shutdown delay. The GPU still reported **100% utilization / 174.0 GB**, with
continuing MES `WAIT_REG_MEM` timeouts during deferred cleanup. Process exit did
not restore device health; no further GPU workload was launched.

At **06:57 UTC**, VRAM had drained to **0.2 GB**, while GFX and power still read
100% and about 858 W. This later observation supersedes the earlier retained-
memory status. No post-hang compute probe has established recovery; monitor
readings alone do not settle it.

Hang evidence: `/home/chcai/mi450_logs/nan_replay_long_20260918_0520/replay_hang_0630/`.
The initial NaN capture had no runtime GPU fault/MES error. The subsequent replay
hang is a separate observed failure; a common cause is unproven.

## Extreme finite gradients and next replay

A user-provided screenshot from another MI450 host reports layer norm receiving
finite gradients as large as **6.9e37**. Its CPU reference reportedly confirms
legitimate BF16 overflow; about 550,000 extreme elements occupy 1,343 rows, while
other values are at most 2.94e-6. That session suspects a jagged split upstream.
Those artifacts have not been inspected on this host. The useful hypothesis is
that the first operation producing infinity can inherit already-corrupt finite
values.

Here, the suspect fused layer's immediate chain is different:

1. Attention writes `dq/dk/dv` into views of `duvqk`; SiLU backward writes `du`.
2. `triton_addmm_bwd` uses separate **`torch.mm`** calls for weight and input
   gradients. Its `dz @ w.T` result is `d_normed_x`.
3. Weighted layer-norm backward consumes `d_normed_x` and produces `d_x` and
   norm parameter gradients.

The nearby `.split()` calls are column views, not jagged split kernels. The
separate jagged split/concat backward path in `triton_jagged.py` remains an
upstream candidate if later evidence points there; it is not the immediate
producer of this layer's input-norm gradient.

`scripts/nan_backward_boundaries.py` now wraps the GEMM and weighted-layernorm
backward aliases in the fused preprocess module. It maps actual forward weight
pointers, including BF16 casts, to STU layers. For each selected call it:

- Saves complete, independent CPU copies of inputs **before the operation**,
  preserving storage aliases, strides and offsets with bounded transfers.
- Records finite absolute maxima, nonfinite counts, counts above a configurable
  threshold, affected row ranges and up to 16 element coordinates.
- Stops on the first nonfinite or extreme input/output after persisting a trusted
  `.pt` artifact. An input trigger stops before executing the operation; this
  shows inherited corruption rather than identifying that operation as producer.

Defaults are target `_stu_layers.1.`, threshold `1e20`, 64 MiB copy chunks and
32 GiB maximum capture per operation. The threshold is diagnostic; no gradients
are clipped or changed. CPU checks passed extreme finite `1e37` input detection,
pre-call snapshot fidelity despite input mutation, BF16 pointer attribution,
nonfinite outputs, strided/aliased scans/copies, layer filtering and unchanged
Python/NumPy/Torch RNG. Subsequent GPU validation captured finite calls and
replayed the isolated layer norm successfully; no bad operation was captured.
`NAN_BACKWARD_SAVE_ALL=1` additionally persists finite selected calls for tensor
comparison, while `--verify-every-repeat` checks restoration on every attempt.

The initial diagnostic command below was prepared before the second AC cycle.
For the completed runs and fresh-output resume commands, use the
[second-AC record](nan_replay_post_ac_20260918_0730.md).

```bash
docker exec -w /workspace/recommendation \
  -e NAN_BACKWARD_TARGET=_stu_layers.1. -e NAN_BACKWARD_ABS_THRESHOLD=1e20 \
  triton-7ff97e-20260910 python scripts/replay_training_step.py \
  /data/mlperf_dlrm_v4/nan_replay_long_20260918_0520/capture_step_000101 \
  --mode current --repeats 2 \
  --backward-probe-dir /data/mlperf_dlrm_v4/nan_replay_long_20260918_0520/boundaries \
  --report /data/mlperf_dlrm_v4/nan_replay_long_20260918_0520/replay_boundaries_report.json
```

Replay now persists progress before restoration, forward and backward, catches
probe stops with a dump path and actual frame/phase, and registers SIGUSR1 for
Python stack diagnostics. A completed anomaly dump exits 86; a finite dump in
save-all mode continues execution. Model-source
hash checks remain enabled; these diagnostic wrappers do not edit kernel source.

If the GEMM inherits extreme `duvqk`, move the same pre-call capture upstream
to attention/SiLU. If its finite, ordinary inputs produce extreme `d_normed_x`,
replay those GEMM inputs independently and compare suspicious rows in CPU FP64.
For a layer-norm output anomaly, compare against CPU FP64 using the captured
mean/rstd and check the destination dtype's range before blaming the kernel.

## Artifact backup

The complete failure artifact contains six files totaling **151,508,784,424
bytes (141.104 GiB)**. Every byte was read for the
[SHA-256 manifest](capture_step_000101_manifest.json), completed at 07:00:22 UTC.
The healthy step-3 control adds 151,223,107,709 bytes; both captures require
302,731,892,133 bytes before logs and software images.

Keep the complete capture directory, including base storage, `previous.undo`,
`capture.pt` and all three JSON metadata files, together in off-host object
storage or a backed-up file server. Preserve the ~17 MB log/source directory
with it: the exact source archive is important because the capture-time worktree
had uncommitted changes. GitHub should retain the diagnostic code, reports,
checksum manifest and eventual verified backup location. The 150 GB embedding
file exceeds both ordinary GitHub and Git LFS per-file limits.

The manifest explicitly says **`local_only` / `remote_backup_verified=false`**.
No remote backup has been created. Destination verification must compare every
file's size and SHA-256 before the backup is marked complete; keep the original
local artifact until then. See [the main report's backup plan](mi450_a0.md#28-capture-backup-and-retention)
for the retention units and GitHub limit references.

## Limits

This is a diagnostic for the current single-rank, DEVICE-only TBE configuration.
Active embedding caches, prefetch pipelines and raw embedding streaming are
rejected. The scripts do not represent arbitrary GPU virtual-address contents,
allocator history, streams, compiler caches, external generator objects or
driver state. Synchronization and copies change timing. A race can therefore
fail to reproduce even with identical mathematical inputs and state.

The rolling full shadow lives in host RAM until the process persists a trigger.
It is not crash recovery: an unkillable GPU hang, process death or power loss can
prevent a complete disk artifact. Dataset/metrics state is not needed for direct
model-step replay; captured batches are used instead of rerunning the loader.
Previous→current replay additionally assumes no evaluation or checkpoint work
changed state between those two training boundaries. The capture-enabled run
disables both. New dataloader iterations can still affect CPU RNG; the boundary
RNG comparison exposes any such difference rather than silently overwriting it.

The legacy `nan_capture.py` remains a partial diagnostic. It skips large
parameters and captures after forward; it must not be used as evidence that a
complete pre-step replay state was preserved.
