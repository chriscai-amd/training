# gfx1250 KFD CWSR trap prologue corrupts a live VGPR when inherited source and destination banks differ

The first-level KFD CWSR handler uses encoded `v1` before normalizing inherited VGPR bank selectors in the `VMEM_ON_TRAP_ENTRY_WA` sequence at `L_NOT_WAVE_START`, in `amd/amdkfd/cwsr_trap_handler_gfx12.asm`. An unequal source/destination bank makes its attempted save/restore copy lane 0 between different physical registers. We reproduce this with a small register-tag shader, reproduce the exact historical BF16 gradient corruption with controlled trap insertion, and prevent both with a local handler correction. The attached fix package contains the proposed implementation and exact tested native payload.

The requested driver change is to preserve the incoming bank fields, normalize the banks before the first vector-register access, preserve and restore the same physical register, and restore the incoming bank fields and EXEC afterward. Please review the trap ABI and workaround requirements, regenerate the embedded handler with the driver team's assembler, and run the attached regression against that build.

| Tested environment | Identity |
| --- | --- |
| GPU | MI450 A0, gfx1250 |
| Kernel | Linux `6.14.0-37-generic` |
| Audited DKMS source | `amdgpu-7.1.1-2397345.24.04` |
| Original AMDGPU | `7.1.1.31300009`, srcversion `EADFD8EC13CF3E6B329194E` |
| Locally corrected AMDGPU | Same version, srcversion `AD83C153B9701570F12964E` |
| CWSR | Enabled |
| Matched real-handler firmware observations | MES and MES_KIQ `0x0000007b` |
| Current original-driver reproduction | September 28, 2026 |
| Saved corrected-driver comparison | September 23, 2026; complete output bytes re-audited September 27 |

The original prologue reads encoded `v1` lane 0 through inherited SRC0, writes encoded `v1` through inherited DST, then writes the saved value through DST. Its net register effect is:

```text
v[1 + 256*DST][0] = old v[1 + 256*SRC0][0]
```

When the banks differ, this overwrites an unrelated live register. The prefetch address also comes through SRC0, so the original sequence does not reliably create its intended zero address. These are software instruction semantics, corroborated by the native instruction audit and the ordinary-shader experiment below.

The native entry-path check shows that normal entry with `WAVE_START` clear executes this complete prologue before trap/save classification. The defect therefore lies on that common entry path, rather than in code specific to `S_TRAP 3`. The [driver-review questions and native proof](validation/driver_review/DRIVER_REVIEWER_QUESTIONS.md) also identify what the vendor must confirm about the bank-state contract and the workaround's timing requirements.

The primary reproducer is in [reproducer/](reproducer/). It uses the executed gfx1250 shader bytes and a Python standard-library HIP host; no training operands or PyTorch are required. Its README gives the runtime library paths, build instructions, exact invocation and the validation status of the portable wrapper. The two comparison modes are:

1. On a driver containing the original handler, run the sentinel with `--expect original`. This expects the three specific copies below; a successful classification means the defect was reproduced.
2. On a separately loaded corrected driver, run the identical sentinel with `--expect fixed`. Every tag must remain unchanged. Both modes also require intact MODE/EXEC, valid output initialization and exact trap return PCs.

**The packaged portable wrapper was run on A0 on September 28 at 00:54 UTC.** It returned `PASS_EXPECTED_ORIGINAL_SENTINEL`: 14 complete cases, exactly three predicted copies, and every per-case check passed. All 94 HIP calls completed, the process exited with status zero and was reaped, and the final all-device scan was idle. The attached [fresh run verdict](validation/portable_original/runtime/result/verdict.json), [controller result](validation/portable_original/execution/terminal.json) and [independent actual-result review](reviews/PORTABLE_RESULT_REVIEW.json) bind this run to the packaged host/native bytes and the original driver. The underlying exact host/shader also ran on the corrected driver on September 23; the portable adapter has not yet had a fresh corrected-driver run.

The interval before staging contained one known CPU8/MC60 record, status `0xd8202000000c080b`, syndrome `0x000000005a800001`, at monotonic seconds `199232.763916–199232.803189`. It matched an already-admitted diagnostic template. All three subsequent kernel-log intervals through launch, process exit and final consumption were empty. The numerical result does not imply general platform-health clearance.

The seven no-explicit-trap controls and four equal-bank trap cases remain unchanged on the original driver. Only these three explicit-trap cases change tags:

| Shader bank-setter immediate | Original-handler result |
| --- | --- |
| `0x05` | `v257[0]` tag `0x22000000` replaces `v1[0]` tag `0x11000000` |
| `0x81` | `v257[0]` tag `0x22000000` replaces `v513[0]` tag `0x33000000` |
| `0x85` | Same `v257[0]` → `v513[0]` copy |

These setter immediates are not the raw MODE-bank byte encoding. The attached shader and oracle retain the exact distinction. The separately retained matched sentinel comparison uses the same original host, shader, configuration, required runtime libraries and environment on the September 28 original driver and September 23 corrected driver; all 14 corrected cases preserve every tag. Its seven recorded driver parameters match the original run; a complete historical parameter capture is unavailable. The newly portable adapter rebinds paths/library pins and removes the unused training-environment requirement, as described in its README.

The component and numerical counterfactuals are:

| Experiment | Original handler | Corrected handler / control | What it establishes |
| --- | --- | --- | --- |
| Execute the prologue as ordinary shader instructions: 256 MODE-bank bytes × eight EXEC masks, 2,048 calls per arm | Exactly 1,536 predicted lane-0 copies | Corrected prologue: zero copies; no-prologue control: zero copies | The software sequence is sufficient to cause the copy, independently of exceptional trap entry |
| Real first-level trap, 14 tagged cases | Exactly the three copies above | All 14 cases unchanged on the corrected driver | The installed trap path exhibits the predicted defect and the correction prevents it |
| Historical Linear DW, original register layout, no explicit traps, four calls | Every output byte equals healthy baseline | Every output byte equals healthy baseline | Arithmetic baseline |
| Same DW, original layout, seven selected explicit traps, four calls | Every output byte equals the historical bad gradient | Every output byte equals healthy baseline | Handler correction removes the constructed numerical failure with unchanged arithmetic code |
| Same DW, vulnerable live registers relocated, traps retained, four calls | Every output byte equals healthy baseline | Every output byte equals healthy baseline | Removing the live register victims prevents the failure |

The fourth DW arm, relocated registers without explicit traps, also equals baseline on both drivers. The current original quartet comprises 16 calls and 2,097,152 BF16 output words checked. Each original-layout trap call exactly matches all 131,072 words of the historical bad gradient, including all **1,030 changed words: 640 NaNs, 384 extreme finite values and six smaller finite differences**. The other 12 current outputs and all 16 saved corrected outputs equal the healthy reference byte for byte. Native ELFs, complete 172-byte launch packets, input identities, libraries, prefills, process lifetimes and full outputs are independently joined. See [the current/corrected comparison](evidence/current_vs_corrected_DW.json), [the current matrix](evidence/current_DW_matrix.json) and [all difference coordinates](evidence/DW_all_1030_differences.json).

The fix saves and clears `MODE[17:12]` with `S_SETREG_B32`, accesses physical `v1` consistently, performs the workaround prefetch with an intended address of zero, restores `v1`, restores the saved bank fields and then restores EXEC. The independently audited native change replaces the 48-byte prologue with a 64-byte sequence and adjusts the displaced entry-branch target; all other handler instructions remain identical after the 16-byte insertion. The current-baseline module audit confines the executable change to the handler-size selections and the corresponding embedded handler data. Source, patch, native words and reproduction details are in [fix/](fix/).

The [scratch-register analysis](LIVEOUT.md) independently checks all four conservative paths through the following dispatch prefix: the differing temporary `TTMP2` and `TTMP15` values are overwritten before use. The source/native analysis also checks saved SCC, VCC, EXEC and MODE preservation. This establishes sequential dataflow correctness under the existing trap ABI; vendor hazard and nested-trap timing review remain necessary.

The corrected native payload has been exercised in the driver. The source-to-payload construction used LLVM plus preserved native instruction words; acceptance by the original SP3 assembler has not been established. Driver-team assembly, generated-header integration, ABI review and normal regression validation remain necessary before an upstream production merge. This is a feasible tested implementation for review, not a claim of an upstream-ready assembler build.

The numerical DW trap schedule is constructed; the original training interrupt timeline was not retained. These results establish a concrete corruption component and prevention of the reproduced failures. They do not establish that every training anomaly has the same cause. A separate finite STU1 step-583 DX inconsistency remains unresolved because its raw operands/output were not retained. The attached portable-wrapper validation record distinguishes fresh GPU execution from CPU packaging checks, and the corrected driver is not currently loaded on A0.

The compact attachment includes original evidence receipts, exact reference output bytes and all difference coordinates. The full current DW raw evidence is durably archived and verified: 44 mappings, complete raw hashing and full decompression of every distinct blob. Large training operands remain in the local evidence archive and are unnecessary for the primary register reproducer. No driver reload, reset or recovery action is implemented by the reproducer; driver comparisons use separately established test boots.
