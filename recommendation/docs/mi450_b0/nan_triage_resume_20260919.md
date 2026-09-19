# NaN triage continuation, 2026-09-19

**Latest result, 07:11 UTC: the GPU health control fails at width 16.**
`index_select` reports **241/1,000** unequal iterations and `gather`
**251/1,000**; both width-4 arms pass. Kernel logs are unchanged and no failing
tensor payload was saved. The weighted-LN GPU controls and finer capture
remain unrun after this failed gate.

The training NaN remains unattributed. The strongest isolated numerical defect
is still weighted input-layer-norm backward with `BLOCK_N=8`: zero incoming
gradients produce NaNs with unchanged input-byte checkpoints. Its `BLOCK_N=1`
control passes, but training with BN1 and the attention VGPR cap still fails.
This continuation restores the prior context, prepares a smaller observation
of the weighted-LN failure, and records the intervening host reboot.

## Recovered evidence and current runtime

The recovered Codex investigation ended at **05:24:22 UTC**, source commit
`8f83cb5`. Its latest finer capture had stalled at attempted step **195**
after **194 completed optimizer iterations**, while copying backward flags
to the CPU. A later KFD `hqds` read caused a diagnostic driver oops, and a
ROCgdb attach was left stuck after failed automatic reset attempts. These
later faults do not identify the original stalled kernel.

By **06:52:19 UTC**, the host had rebooted from
`677a2df1-7c98-4785-92ea-9c8d559b05d4` to
`5769c84c-01bc-4bc4-88fb-d86335f24270`. Narrow process-identity checks found
all old launcher, worker and debugger PIDs absent. The old capture is now
annotated `interrupted_host_reboot`, retaining its original state file and
all stall evidence. Its 19 sampled losses remain finite, last sampled
**190 = 0.13864581286907196**; no payload or bound pass exists. Reboot
initiator, exact job termination time and launcher exit code are unknown.

The preserved `mi450-b0-0910-7ff-start0` container was restarted at
**06:50:51.503770295 UTC**. CPU-only checks verify Torch
`2.11.0+rocm7.14.0a20260625`, HIP `7.14.60850`, Triton `3.8.0` at exact commit
`7ff97e310935b4a79794878dbc911f9af25d38d9`, and all **143 Python source
hashes** in the initial continuation snapshot. CUDA was not initialized by
these checks. The current loaded driver reports `gpu_recovery=0` and
`halt_if_hws_hang=1`; VBIOS remains `113-M4500001-650D`.

During preparation, an unrelated host `tensile_client` GEMM benchmark owned
the single GPU, with successive worker PIDs. It was not launched by this
investigation. At **07:10:39 UTC**, privileged ownership checks found no
process on `/dev/kfd` or the render node: the benchmark had already exited,
so the instruction to stop it required no termination.
Its exact completion time and initiator are unknown. The other workload and
`mi450_c` container have not been changed by this investigation.

## Latest GPU health control

`health_20260919T071101021851Z/` ran **07:11:01–07:11:02.449157 UTC** and
exited **1** at the script's final assertion.

Checked-in evidence: [outcome and hashes](evidence/health_20260919T071101Z/outcome.json)
and [exact probe log](evidence/health_20260919T071101Z/run.log).

| Arm, execution order | Unequal iterations / total |
|---|---:|
| `index_select`, width 16 | **241 / 1,000** |
| `gather`, width 16 | **251 / 1,000** |
| `index_select`, width 4 | **0 / 1,000** |
| `gather`, width 4 | **0 / 1,000** |

Seed is 11; source has 2,347,656 rows of exact integer indices represented
as float32, and each iteration selects 16,384 indices. All output columns
should equal the selected row index. A bad iteration means at least one
unequal element; these are not counts of bad elements, rows or NaNs.

Privileged `/dev/kfd` and renderD128–135 ownership checks are empty before
and after, with no continuous ownership monitor. Boot is unchanged and
the before/after kernel logs are byte-identical, with **zero new lines**.
No investigator reset or reboot was issued.

The script generates source, random indices and expected values on GPU,
then checks equality on GPU. No failing tensors or CPU reference were
preserved. The result does not distinguish source construction, indexing,
the equality check or an underlying execution defect; it does not establish
NaNs or causality with training or the preceding `tensile_client` workload.
Width also changes allocation size, so vectorization is not isolated.
The next health diagnostic needs to save source/index/output values and
verify their exact mapping on CPU before attributing the corrupting primitive.

## What is established

| Observation | What it establishes |
|---|---|
| Full-size zero-DY helper BN8 fails at iterations 115 and 212, with BN1 passing 1,000 between them | An impossible numerical result in the isolated weighted-LN helper; checked input bytes remain unchanged |
| Portable helper BN8 fails at 90; direct DX and some subsequent helper controls pass | A positive portable case, with intermittent or execution-history dependence; direct passing runs do not clear DX |
| BN1 plus attention cap256 training reports NaN at sampled 60/70 | The two component mitigations are insufficient for training |
| Original step-51 output gradient GEMM receives already-huge finite `dout` | The selected GEMM has not been established as the corrupting producer |
| HSTU backward reuses saved mean/rstd when recomputing normalized X | The saved-statistics forward branch needs its own control; BN1 backward does not cover it |

The first weighted-LN kernel produces DX and the parameter partials; the
second reduces partials into final weight/bias gradients. Existing positive
helper captures observe results after both kernels. Register counts (BN8:
576; BN1: 124) and repeated bad columns are leads, not a compiler or hardware
attribution.

## Smaller observation before parameter reduction

The portable
[`repro_weighted_layer_norm_zero_dy.py`](../../scripts/repro_weighted_layer_norm_zero_dy.py)
now accepts `--check-partials-before-reduction`. It checks only the original
FP32 weight/bias partial arrays immediately after DX/partial production,
before the parameter reduction. The normal final DX and parameter-gradient
checks remain enabled. This is a helper-only option, mutually exclusive
with the existing `--check-before-reduction` full boundary check.

With 2,048 partial tiles on this GPU, both arrays total **8 MiB**. The new option
avoids the extra **2.35 GiB** DX scan at that boundary. It still changes
kernel ordering, memory traffic and timing. Reports and compact evidence
distinguish pre-reduction and final observations, record which check mode
ran, and identify a bad partial's program tile rather than pretending it
belongs to one unique input row.

All **27 focused CPU tests pass in the preserved container**: 21 backward
reproducer tests and six saved-statistics forward tests. New cases detect
bad partials even when final outputs appear clean, prove DX is not scanned
at the partial boundary, preserve independent samples across later writes,
check tile provenance, and verify option rejection and wrapper restoration.
This validates the diagnostic logic; the new option has no GPU result yet.

An independent CPU audit of all three saved failure payloads confirms that
both stored first-bad DX rows have `row % 8 = 3`, with weight-gradient
columns 5/263 and bias-gradient column 263 recurring. The portable failure's
first bad row belongs to a different program tile from the original failure
(the original mapping assumes 2,048 tiles, which its report did not record).
Saved compiler layouts map the same columns to different logical lanes
during accumulation and final partial stores. These observations are bounded
to the saved samples and do not identify a physical lane, bad register,
instruction or writer. The next partial observation can test whether the
parameter anomaly is already present before reduction.

## Prepared experiments

Use the exact runtime and mandatory environment in
[the B0 handoff](mi450_b0.md#13-mandatory-flags-and-boot-procedure). GPU
work must remain serial. Verify privileged `/dev/kfd`/render-node ownership;
`amd-smi process` previously missed a live training owner.

1. The four-arm health control has run and failed. Preserve a failing
   source/index/output, validate source and mapping on CPU, and determine
   whether the current health failure is reproducible. Re-establish GPU
   readiness before continuing the numerical controls and training capture.
2. After that gate passes, run the prepared forward control with `--statistics-mode saved`, then
   `recompute`, each in a fresh process. Preserve actual launch configuration.
   Explicit matching `--block-n`/`--num-warps` arms are needed to compare the
   same configuration: production's autotune key omits statistics mode.
   The current forward CLI accepts block sizes 1/8; production also offers
   2/4. If either wins autotuning, extend the diagnostic selector before
   claiming a comparison at that exact configuration.
3. Re-establish the positive BN8 helper case, then run
   `--check-partials-before-reduction`. Bad partials before reduction narrow
   the observed defect to the DX/partial-production boundary. Clean partials
   followed by bad final gradients narrow the interval to reduction or later
   writes. A passing intermittent control cannot identify a repair.
4. Test an explicit BN8 `--max-vgpr 256` arm if the positive control returns,
   preserving actual registers, spills and executable identity. That keeps
   row blocking fixed while testing the register cap separately from BN1.
5. Resume the original finer preprocess capture with cap256 + BN1, both
   class boundaries, preprocess internal stages on and output internal
   stages off. Use original projection/LN inputs and outputs to locate the
   remaining training failure before proposing another production fix.

The standalone commands, from the exact source snapshot, include:

```bash
python scripts/repro_weighted_layer_norm_zero_x.py \
  --source-root . --statistics-mode saved \
  --report /path/to/fresh/forward_saved.json

python scripts/repro_weighted_layer_norm_zero_dy.py \
  --stage helper --block-n 8 --rows 2468813 --repeat 1000 \
  --check-partials-before-reduction \
  --report /path/to/fresh/backward_partials.json
```

These commands are prepared, not completed GPU results. The **3,000
uninstrumented optimizer-step** validation requirement remains unmet.

## Local artifacts

Root:
`/home/chcai/runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/continue_20260919T065146Z/`.

- `reboot_observation/`: process/boot observations, container inspection and
  annotation hash manifest. The original run retains its old state separately.
- `verified_stack_cpu.json`, `source.tar.gz`, `source_sha256.json`: verified
  initial continuation stack/source; 143 matching Python files.
- `source_partials.tar.gz`, `source_partials_sha256.json`,
  `partials_probe.patch`: source containing the new partial observation.
- `weighted_ln_container_cpu_tests.log` and corresponding command JSON:
  the 27 passing CPU tests.
- `analyze_weighted_ln_failure_patterns_cpu.py`,
  `weighted_ln_failure_patterns_cpu.json` and `.md`: reproducible,
  GPU-disabled audit of the saved failures and compiler layouts, including
  artifact hashes and explicit sample/mapping limits.
- `prepared_capture_command.json`: exact next finer-capture command and
  prerequisites, explicitly marked not run.
- `health_20260919T071101021851Z/`: failed health log, exact command, terminal
  state/outcome, empty ownership snapshots, unchanged boot IDs, identical
  kernel logs and empty ordered delta. No failing tensor payload exists.
- `continuation_outcome.json`: current failed-health status; its earlier
  GPU-waiting version is preserved separately.
- `run_control.py`: exact-environment launcher for one health, forward or
  backward arm; refuses to launch when privileged GPU ownership checks find
  another process. Use `--describe` to inspect the command without launching.

Initial forward snapshot inside the container:
`/tmp/mi450-b0-continue-20260919-0653/recommendation`.
The newer partial-observation snapshot is
`/tmp/mi450-b0-continue-partials-20260919/recommendation`; pass it explicitly
to the launcher's `--source` option for the new backward arm.
