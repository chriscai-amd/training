The reviewed ticket supports a concrete CWSR prologue defect and a feasible
correction that prevents the reproduced register and DW failures. This review
found no additional source-level disqualifier. The remaining questions separate
production integration requirements from claims that the retained evidence
cannot establish. The reviewed ticket version and input hashes are recorded in
[PROVENANCE.json](PROVENANCE.json).

1. **Component attribution and the hardware contract.** A reviewer may ask
   whether the fault belongs to the KFD prologue, a second-level handler, MES,
   or a hardware trap-entry contract. The exact ordinary-instruction sequence
   predicts the observed SRC0-to-DST copy; the inline three-arm experiment,
   real-handler sentinel, victim-register relocation and handler-only module
   correction provide mutually reinforcing counterfactuals. The module audit
   checks allocated sections and relocation/symbol equivalence, beyond a
   version-string comparison. This strongly localizes the repair to the
   prologue. The vendor should confirm the architectural rule for inherited
   bank selectors on A0 trap entry. If an official contract promises bank zero
   there, ownership could instead be described as an A0 contract violation
   requiring this software workaround. The source already saves inherited
   bank bits later in the CWSR path, consistent with their being meaningful.
   No further available CPU inspection can substitute for that vendor contract.

2. **The expected interrupt route.** `S_TRAP 3` exercises a software trap; it
   does not prove that an identical software trap occurred during training.
   The additional native check establishes the useful narrower fact: for the
   normal entry at byte 0 with `WAVE_START` clear, the branch at byte 24 targets
   the prologue at byte 60. The complete straight-line prologue runs before
   trap/save classification and second-level dispatch. The byte-4 restore
   entry and the wave-start return at byte 56 are separate paths. Source
   coordinates are original lines 240–271, with dispatch following them.
   Thus the defect is in a common non-wave-start entry path, rather than in
   code specific to the chosen software trap ID. This is a conditional code
   path proof, not a direct trace of the hardware-selected TBA or a certificate
   of the historical training event. Historical trigger/timing cannot be
   recovered by more CPU checks of these artifacts.

3. **Preserving the VMEM workaround.** A reviewer should confirm the erratum's
   required address, scope, cache policy and timing constraints. The added
   native check finds prefetch words `ee17406e 00040000 00000001` originally
   and `ee17406e 00040000 ffffc101` after correction: the opcode/control words,
   scalar-base encoding and vector operand `v1` are retained; the signed
   address offset changes from zero to `-63`. Saved `k` plus vector `63-k`
   plus offset `-63` gives the intended address zero for all saved six-bit
   values. The original unequal-bank sequence did not reliably zero its
   SRC0-selected prefetch operand. These byte checks and the existing bank
   matrix settle the address/control-field concern. They do not prove that
   the additional scalar instructions, MODE writes, or changed instruction
   spacing satisfy an undocumented workaround deadline or hazard rule.
   That remains a vendor erratum/ISA review item.

4. **Scratch registers, flags and inherited state.** A reviewer may notice that
   `TTMP2` leaves the patch holding saved bank bits rather than zero, or ask
   about SCC/VCC, inactive lanes and MODE restoration. The separate completed
   live-out review establishes that both differing temporary values are
   overwritten before use on every following dispatch path; `TTMP3` and
   `TTMP14` already match. The patch uses non-carry vector subtraction,
   restores EXEC_LO and all six modified bank bits, and leaves SRC2 and other
   MODE bits alone. Existing sentinel/inline evidence exercises register and
   MODE/EXEC preservation. This review does not duplicate that analysis.
   Its remaining boundary is the existing scratch ABI and normal instruction
   ordering; nested/reentrant observation and special-register hazard timing
   still require vendor review. No additional ordinary def-use check closes
   those hardware questions.

5. **Exact production payload and supported family.** The `.asm` alone is not
   what this driver loads. The selected array is `cwsr_trap_gfx12_1_0_hex`,
   chosen by `kfd_cwsr_init`; its compiled size selection appears in
   `kgd2kfd_device_init`. The fix package supplies both source and the exact
   tested header bytes, and identifies `CHIP_GC_12_0_3` as the affected source
   family. The generic `CHIP_GFX12` output is unchanged. A reviewer should
   retain the explicit caveat that the tested payload used LLVM plus 84
   preserved VOP3 encodings; original SP3 acceptance was not tested. Given the
   vendor's assembler/generation environment, a CPU regeneration and full
   native comparison can close this integration gap. Repeating the existing
   LLVM round trip cannot establish SP3 acceptance.

6. **What “the fix works” establishes.** The strongest result is prevention of
   the demonstrated bank-copy failure and all bytes of the constructed DW
   failure, including its complete historical 1,030-word difference pattern.
   Register relocation with traps retained gives an independent numerical
   counterfactual. The retained corrected run is on a different boot, seven
   recorded parameters match, and complete historical parameter capture is
   unavailable. Loaded-module binding uses the pinned module audit and
   build-ID/srcversion receipts, not a readback of every live executable byte.
   Fresh original-wrapper success does not turn the older corrected run into
   a fresh corrected-wrapper execution. The ticket properly exposes these
   boundaries. A vendor-regenerated driver and fresh matched regression can
   strengthen production confidence; another CPU receipt cannot replace that
   execution.

7. **Historical attribution and broader training coverage.** Exact reproduction
   of the bad gradient is a strong causal fingerprint, but the trap schedule
   was constructed and the original interrupt timeline was not retained.
   The inline result demonstrates that executing the software sequence can
   cause the copy outside the handler; “no explicit trap” is not a complete
   absence-of-interrupt certificate. These facts support the prologue defect
   without proving it was the sole source of all historical anomalies. The
   ticket's unresolved corrected-driver STU1 step-583 result should remain
   separate. No extra CPU proof over the retained data can recreate its
   missing operands or the missing historical interrupt timeline.

The useful additional CPU checks for this review are complete: normal-entry
branch routing, straight-line prologue coverage and exact prefetch field
comparison. Their machine-readable results are in `PROVENANCE.json`; the local
checker is an audit implementation, not another driver-team runtime entrypoint.
The next production sign-off items are vendor trap/erratum/hazard confirmation,
affected-family SP3 regeneration and regression on that generated driver.
No GPU action, module read or mutation, bulk payload scan, or change to the
frozen fix/reproducer package occurred in this review.
