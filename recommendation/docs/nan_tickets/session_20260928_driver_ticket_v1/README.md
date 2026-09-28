# MI450 gfx1250 CWSR corruption: driver-ticket attachment

Start with [TICKET.md](TICKET.md). It identifies the faulty first-level KFD CWSR sequence, explains the physical-register copy, and connects the register reproducer to the exact historical gradient corruption. The proposed source/header patch and exact tested native payload are included.

This is a local attachment prepared for driver-team review. The demonstrated cause is the handler's use of inherited, unequal source/destination VGPR banks before it normalizes them. The tested correction prevents this corruption in the register and numerical comparisons.

| Item | Entry point |
| --- | --- |
| Issue description and causal evidence | [TICKET.md](TICKET.md) |
| Small executable reproducer, original/fixed commands | [reproducer/README.md](reproducer/README.md) |
| Proposed implementation, native comparison and integration guidance | [fix/README.md](fix/README.md) |
| Scratch-register and special-register preservation | [LIVEOUT.md](LIVEOUT.md) |
| Common trap-entry proof and remaining driver-review questions | [validation/driver_review/DRIVER_REVIEWER_QUESTIONS.md](validation/driver_review/DRIVER_REVIEWER_QUESTIONS.md) |
| Fresh execution of the portable wrapper on the original driver | [validation/portable_original/runtime/result/verdict.json](validation/portable_original/runtime/result/verdict.json) |
| Independent review of that execution | [reviews/PORTABLE_RESULT_REVIEW.json](reviews/PORTABLE_RESULT_REVIEW.json) |
| Historical DW bytes and compact evidence receipts | [EVIDENCE_COPIES.json](EVIDENCE_COPIES.json) |
| Exact attachment inventory | [MANIFEST.json](MANIFEST.json) |

From the extracted attachment root, these checks use CPU only:

```sh
sha256sum -c SHA256SUMS
python3 -B verify_evidence.py
python3 -B reproducer/run_repro.py --check-only
python3 -B fix/verify_fix.py
```

Use Python 3.11 or newer for the full set, plus `patch`. No model, training data, PyTorch, container or driver build is needed for the primary reproducer. Its prebuilt shader is 32,152 bytes; each case returns a 160-byte packet. A GPU run requires MI450/gfx1250 with CWSR enabled and compatible HIP/HSA libraries; the reproducer README gives the commands. `--expect original` succeeds only when the three specific corruptions are observed. `--expect fixed` succeeds only when all 14 cases preserve their tags. Both require the full packet, MODE/EXEC and return-PC checks to pass.

The portable wrapper completed its original-driver run on September 28, 2026. Saved September 23 corrected-driver observations use the same exact underlying host/shader, and their packets are included and decoded by `--check-only`. A fresh run of the portable wrapper on the corrected driver remains pending. The DW evidence contains the complete healthy/corrupted reference output bytes and all 1,030 differing coordinates; the verifier independently compares all 131,072 BF16 words. It checks the recorded 16 original and 16 corrected output mappings, without rereading the omitted large training operands.

`fix/` and `reproducer/` retain their reviewed source inventories. `validation/` contains exact compact copies of the fresh runtime, lifecycle and gate receipts, including all output/prefill bytes. `reviews/` contains independent reviews. `VALIDATION_COPIES.json` records the source path, size and SHA-256 of each additional copied file. Absolute paths inside historical receipts identify the originating A0 evidence; they are provenance, not portable runtime dependencies. Large kernel history captures, driver modules and training-input archives remain on A0. Their prior audit receipts are included, and the attachment does not pretend to reexecute those audits without their inputs. Preparation scripts and historical checkers are provenance; use the four commands above for portable verification.

The patch includes both the assembly source and the embedded C array that KFD actually selects. The exact corrected native payload ran successfully in the driver. Its construction used LLVM plus preserved native words; original SP3 acceptance and production hazard/ABI/regression validation remain driver-team work. The seven DW trap events form a constructed replay, because the original training interrupt timeline was not retained. A separate finite STU1 DX inconsistency remains unresolved. These limits are stated in the ticket so the proven defect and tested prevention can be assessed precisely.
