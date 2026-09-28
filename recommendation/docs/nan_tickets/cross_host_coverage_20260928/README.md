# Cross-host coverage of the gfx1250 CWSR ticket

Reviewed September 28, 2026 against repository `8cf681dead52574301cc68bdfa37442626d333b9`, after a fast-forward pull from `402f7d0`. Local A0 investigation updates and the frozen ticket were preserved. This supplement assesses retained repository evidence; it does not report new remote-host observations or GPU tests.

**Use one shared component ticket for the faulty CWSR prologue on local A0, B0, 1P4G a37-1, and the inspected RCK cluster nodes.** The matching handler bytes and controlled A0/B0 observations establish shared component applicability. The RCK cluster adds strong full-model corroboration. Attribution of every training NaN, memory fault, hang or host loss remains broader than the evidence.

The original [ticket](../session_20260928_driver_ticket_v1/TICKET.md), [portable reproducer](../session_20260928_driver_ticket_v1/reproducer/README.md) and [tested fix](../session_20260928_driver_ticket_v1/fix/README.md) remain unchanged. This supplement should accompany that attachment when reporting cross-host scope.

| Environment | Component evidence | Coverage of the observed failures | Remaining qualification |
| --- | --- | --- | --- |
| **Local A0 investigation host** | Original handler directly audited; ordinary-instruction, actual-trap, register-relocation and corrected-driver counterfactuals completed. | **Confirmed reproduced corruption and prevention.** The constructed DW replay matches every historical output word, including all 1,030 differences and 640 NaNs. Corrected driver preserves every output in the quartet. | The original training interrupt history was not retained. The separate corrected-driver STU1 step-583 inconsistency and runtime/MES stalls remain open. |
| **B0, a30-4** | Same original handler and same three sentinel bank-copy routes reproduced on B0. Six unequal-bank cases generate NaNs in a compiled LayerNorm kernel; nine controls remain zero. | **Confirmed shared component and controlled numerical consequence.** The ticket is directly relevant to B0. | The ticket's corrected driver has not run on B0. The B0-local staged candidate is a different implementation and was not loaded. Broad B0 training repair and exact historical LN coordinates remain unproved. |
| **1P4G a37-1** | Host report identifies the exact original handler in the el10/7.1.0 driver. Compaction-induced queue quiesces and high-VGPR attention faults are recorded. | **Shared component applies; the three reported memory faults are plausible consequences.** Two faults name attention; the first is an aperture violation before the step-50 log. No NaN was reported in those runs. | No corrected-driver comparison or explicit bank sentinel on that host. Passing runs lack comparable eviction measurements; SDMA0/TCP versus ROCr attribution is unresolved. |
| **1P4G a37-2** | 14,000 finite steps and AUC 0.7514703274 are reported, but its handler hash and eviction exposure were not recorded. | **Applicability is conditional on handler identity.** Successful convergence does not rule out an intermittent component defect. | Establish the actual handler first. Lower exposure from fresh boot/more host RAM is an inference, and the successful old stack differs from a37-1's reprovisioned stack. |
| **A0 RCK cluster** | Original handler matches on j19-1, j19-9, j19-11 and h21-15. The cluster builder reconstructs the exact tested corrected payload. j19-9 records its corrected load and training. | **Strong corroboration for the same component ticket and likely coverage of the localized preprocessor-DW NaN.** Stock failures occur at loss step 111, localized backward step 123, and bound-run loss step 354. Corrected CSV has 2,089 consecutive finite losses. | One incomplete corrected run. Host/boot/build-label or binding differences prevent a fully isolated training comparison. No numerical outcome is retained for h21-15. MES wedges and node losses have separate evidence. |

## Why this is a shared component claim

The vulnerable image is **5,656 bytes**, SHA-256 `0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290`. The tested corrected image is **5,672 bytes**, SHA-256 `68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80`.

The original prologue reads encoded `v1` through inherited SRC0 and restores it through inherited DST before clearing the bank selectors. With unequal banks, it copies lane zero between physical registers. A live destination can hold numerical data or an address, so the mechanism can produce finite errors, NaNs, or invalid addressing depending on the interrupted code and values. High register count alone is insufficient: the selected banks and the liveness of the destination matter.

The A0 proof isolates this software sequence: 1,536 predicted copies in 2,048 original-prologue calls, zero copies in 2,048 corrected calls, and zero in 2,048 no-prologue controls. Actual original-driver traps produce exactly the three predicted copies; saved corrected-driver traps preserve all 14 cases. The new portable adapter ran on A0's original driver on September 28. The same underlying host/shader ran on A0's corrected driver on September 23; a fresh corrected-driver execution of the portable adapter remains pending.

B0 reproduces the same bank routes on its hardware. Its compiled-LN test adds an arithmetic bridge: a temporary NaN is copied into a live zero-DY register and contaminates the parameter-gradient partials. A0 subsequently executed the B0-published LN native text with an earlier X-load intervention and removed all 3,584 DX and two partial-DW NaNs under the corrected driver. That broadens the demonstrated mechanism, while remaining an A0 execution rather than a B0 corrected-driver result.

The cluster native comparison independently parses only literal constants from its builder, applies the documented prologue/branch edits to the original bytes, and obtains the exact corrected hash above. The cluster kernel module and the A0 kernel module are different builds. Their shared handler establishes repair applicability; each target still requires its own source/kernel build and loaded-image validation. The A0 `.ko` must not be treated as a drop-in module for other kernels.

Sources: [A0 record snapshot](sources/a0_record.md.txt), [B0 record snapshot](sources/b0_record.md.txt), [a37-1 record snapshot](sources/a37_1_record.md.txt), [cluster record snapshot](sources/cluster_record.md.txt), and the three [independent assessments](reviews/).

## What the new cluster evidence adds

The j19-11 tripwire identifies `_hstu_transducer._input_preprocessor._additional_embedding_mlp.2.weight`, shape `[512,256]`: exactly one nonfinite dense gradient among 69, with finite reported inputs, on all four ranks. This is the parameter implicated by A0 attempt 405. The summary has no raw per-rank tensor bytes, so it does not establish the same 640-NaN coordinates or reconstruct the cluster failure byte for byte.

The checked-in observations have different capture cutoffs:

| Record | Defensible statement |
| --- | --- |
| j19-1 stock default CSV | Steps 1–110 finite; 111–113 NaN. |
| j19-11 stock default plus tripwire | Loss CSV finite through 122; tripwire summary reports a backward-gradient failure at 123. |
| j19-9 stock bound CSV | Steps 1–353 finite; 354–356 NaN. Binding did not guarantee finite training. |
| j19-9 corrected default CSV | **All 2,089 retained rows finite**, ending 02:34:19.393 on September 27. |
| Later corrected monitor status | Reports finite through **2,108**, at 02:34:49.434; trainer still alive. |
| Cluster document headline | Reports **2,128** at a later checkpoint; those additional raw/status observations are absent from this checkout. Do not label 2,128 independently verified. |

The reviewer independently decoded 4,575 CSV rows and 394 telemetry rows. In the corrected 50-minute window, telemetry gives 2,232,010.56 NUMA PTE updates per minute and an 18,466 ms increase in summed process/device eviction-duration counters. These demonstrate continued activity consistent with exposure. They do not count CWSR entries or identify an interrupted wave's banks. The stock 2.5-million/minute headline uses a different window including startup, so it does not establish equal exposure.

The default comparison changes physical host and boot. Stock j19-1/j19-11 records FBGEMM `2026.9.25`, while corrected j19-9 records `2026.9.26`; this may be a build-date label, but runtime byte equality is unrecorded. The same-host stock j19-9 comparison has the matching label and changes memory binding. Therefore “only the handler changed” is supported as a saved module-construction claim, with audit limits; it is too strong as a description of the training experiment. The cluster module files and full audit/relocation objects were not committed, so this review validates the retained audit summaries and handler reconstruction, not a fresh extraction of those remote modules.

The evidence supports a likely shared culprit for the localized cluster NaN and a promising corrected training result. It does not establish completed convergence, repeated prevention, or the remote run's current state on September 28.

Exact raw snapshots are retained under [cluster_evidence/](cluster_evidence/); the calculations and source hashes are in [CLUSTER_ASSESSMENT.json](reviews/CLUSTER_ASSESSMENT.json).

## What the ticket does not yet cover

- **1P4G wild-address faults and host halt:** the handler can corrupt an address, but those particular faults lack a corrected counterfactual and an interrupted-register witness. Fault-client attribution and whether a core dump itself caused the largest compaction burst remain unresolved.
- **MES non-response, recovery and reboot hangs:** the cluster records `INVALIDATE_TLBS`/`REMOVE_QUEUE` non-responses and `halt_if_hws_hang=1` behavior. These identify a separate progress/recovery problem; the CWSR bank fix has not been shown to repair it. Corrected boot `MEM_RESERVED_INFO` entries named `VM_PAGE_FAULT` are reserved-memory metadata, not runtime faults.
- **Known int32 row-offset bugs:** the source requests overflowing address arithmetic and needs its separate int64 correction. Normalizing CWSR banks cannot repair that calculation.
- **Buffer-op hangs, expandable-segment allocation failures, capacity OOMs, fabric/RCCL issues and compiler/performance limitations:** retain their own reproducers and acceptance criteria. The current evidence does not attribute all of them to this prologue.
- **A0 STU1 step 583:** the corrected driver still had one finite conditional-bound inconsistency. Its missing raw operands/output require a new targeted capture.

The latest a37-1 document has causal headlines stronger than its detailed limitations. Use its explicit “not proven” assessment for the faults. Its run ledger also places the first aperture fault before step 50, while following prose says all three failed runs logged matching steps 50/100. Only the two later named-attention failures have step-100 observations in that ledger. The BLOCK_N=64 VGPR count is also inconsistent between the host narrative and performance report until exact kernel/shape identities are joined. These discrepancies do not invalidate the matching handler or A0 proof; they limit the host-specific claims.

## Reproducer coverage and the remaining acceptance tests

The existing portable reproducer is the appropriate shared component regression. It needs Python 3.11+, gfx1250, CWSR enabled, and target-compatible HIP/HSA libraries. It does not require the training data, full model, compaction workload or AutoNUMA to create the explicit trap. On the original handler, `--expect original` requires exactly three copies among 14 cases. On a corrected handler, `--expect fixed` requires zero copies, with all initialization, MODE/EXEC and return-PC checks valid.

This gives the driver team a direct test of the bank-preservation defect even when a training run remains finite. Natural queue-save exposure and training-failure attribution need additional measurements:

1. **B0:** build and audit the ticket's correction against the B0 source, establish a corrected-driver boot, then run the same sentinel and compiled-LN matrices. The separate 5,736-byte B0 candidate used DST=SRC0 with waits and was tested in userspace/staged only; its results must not be presented as execution of the ticket's 5,672-byte correction.
2. **1P4G a37-1:** build against the actual 7.1.0 el10 source/kernel, run stock/corrected sentinels, then compare repeated identical BLOCK_N=128 runs under measured compaction/eviction exposure. Alternate BLOCK_N=64/128 controls and retain native metadata, fault records and completed process outcomes. Reconcile SDMA0/TCP versus ROCr attribution. The explicit-trap kit closes the missing portable-reproducer deliverable; it does not replace a natural-save/compaction experiment.
3. **1P4G a37-2:** measure its handler before assigning it to the affected set. A future successful run should retain eviction/compaction telemetry and exact runtime identities; driver version alone is insufficient.
4. **RCK cluster:** retain the corrected run's final outcome, kernels/logs and AUC if reached; repeat on matched host/software/settings with continued measured exposure. Run the portable sentinel on cluster stock/corrected images and retain a tensor capture if nonfinite values return. A separate MES/recovery issue should carry its own logs.
5. **Driver integration:** original SP3 generation, the affected-family configuration, bank-state ABI, MODE-write hazards, workaround timing and nested-trap requirements still need vendor validation. The A0 source/native and live-out proofs remain available in the main ticket.

The ticket can be filed now for the established component defect, with B0 and cluster corroboration and a37-1 listed as suspected impact. Full training repair across every environment should remain an acceptance goal rather than a claim already proved.

## Review provenance

`SOURCES.json` pins every copied source/evidence file and the repository revision. The A0 record snapshot includes preserved local investigation updates; other host snapshots match the pulled revision. `reviews/` contains the independently scoped B0, 1P4G and cluster reviews. Original absolute paths inside those receipts identify their source evidence. No remote system, driver or GPU was modified for this assessment. The original ticket/archive hashes remain unchanged.
