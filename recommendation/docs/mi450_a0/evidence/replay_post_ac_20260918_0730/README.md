# Second-AC replay evidence, 2026-09-18

These small reports and logs accompany
[the experiment record](../../nan_replay_post_ac_20260918_0730.md).
The original runtime directories are
`/home/chcai/mi450_logs/nan_replay_post_ac_20260918_0730/` and
`/home/chcai/dlrm_data/nan_replay_post_ac_20260918_0730/`.

| Evidence | Purpose |
|---|---|
| `metadata.json`, `health.log` | Image, Triton, boot and successful compute check after recovery |
| `artifact_verification.json` | Full post-AC hash verification of all six original failure-capture files |
| `run_boundaries.sh`, `replay_report.json`, `replay.log`, `boundaries_initial.jsonl` | Two direct replays; finite gradients, exact forward outputs, large backward-max discrepancy |
| `run_saved_boundaries.sh`, `saved_operations_report.json`, `replay_saved.log`, `boundaries_saved.jsonl` | Two direct replays with full restoration verification every repeat and saved finite operations |
| `gemm_repeat0_vs_repeat1_comparison.json` | Complete logical-byte and raw-storage equality for the saved GEMM pair, including U/V/Q/K summaries |
| `run_ln_operation.sh`, `ln_operation_replay.json`, `ln_operation_replay.log` | Two isolated LN GPU replays; complete output hashes match the capture; FP64 checks cover only the stated rows |
| `operation_captures_manifest.json` | Sizes and SHA-256 hashes of the four local operation dumps, totaling 36,158,352,092 bytes |
| `diagnostic_source_manifest.json` | Source hashes for the diagnostics and hashes of runtime source archives |
| `dmesg.log`, `wrapup_status.json` | Kernel record and final process/device status; experiment services and kernel logger stopped |

Validation completed before persistence: three retained CPU capture/restoration
regressions passed with `python -B scripts/test_nan_replay.py`; the new analyzer's
CPU self-tests passed for GEMM/LN references, partial reductions, BF16 overflow,
logical-byte comparisons, unused storage, aliases and fresh restoration. Syntax
checks passed for all three changed diagnostic scripts. The actual saved GPU
replay and isolated LN operation results are recorded above. The analyzer's
isolated GEMM GPU mode has not yet been exercised on these dumps.

Git includes **no tensor payloads**. Neither the original full-state capture nor
the four operation dumps has a verified off-host backup. Preserve the existing
local artifacts and source archives until a destination copy passes the full
manifest checks. Hashes establish file integrity, not deterministic NaN replay.

The checked-in kernel log has trailing line whitespace removed. The original
unmodified log remains in the runtime log directory.
