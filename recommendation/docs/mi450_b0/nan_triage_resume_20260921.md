# MI450 B0 NaN continuation — September 21, 2026

**Wrapped up at the user's request after the 04:32 UTC stock-GEMM control.**
All GPU sessions are terminal and drained; the final direct owner scan is
complete and empty. The **04:20 UTC** explicit-trap experiment now reproduces
NaNs in the exact historical compiled LN kernel: six unequal-bank cases
produce only partial DW/DB column-263 NaNs, while nine original/wait/equal-bank
controls are zero. All 15 cases and 270 retained buffers pass independent audit.
The fresh original-boundary training capture instead aborted at **04:17 UTC**
after finite sampled step830, with illegal memory access and no payload.

**The original training attribution and validated repair remain open.** The
03:21 userspace repair comparison and 03:26 candidate driver build remain
complete, but the candidate is uninstalled. Matched native-bank and same-site
LN interventions are GPU-unrun. The user explicitly authorized publication
of this B0 documentation update; no driver activation or further experiments
are included in that request.

## Restored context and execution rules

The investigation checkout is `/home/chcai/training-mi450-nan-20260919`.
The IDE checkout at `/home/chcai/training` retains its historical body and
points its current summary here. Prior training and component evidence remains
in the [September 20 record](nan_triage_resume_20260920.md).

Required runtime: container `mi450-b0-0910-7ff-start0`, September 10 image,
Triton commit `7ff97e310935b4a79794878dbc911f9af25d38d9`, gfx1250,
Torch `2.11.0+rocm7.14.0a20260625`, HIP `7.14.60850`. Preserve the
[B0 runtime precautions](mi450_b0.md#13-mandatory-flags-and-boot-procedure).
Use the corrected direct process/thread character-device FD scan by device
number before and after GPU work. Empty `fuser` or `amd-smi` output alone is
insufficient; saved scans do not prove continuous exclusivity.

Run paths below are under
`/home/chcai/runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/`.
Current checkpoint JSON records the wrap-up. Earlier running-state summaries
are historical; terminal reports and their audits take precedence.

## Controlled B0 trap mechanism reproduced

Run `resume_20260921/b0_cwsr_controlled_20260921T022905021210Z/`
completed all **14 cases**, exit 0. Its independent CPU certificate passes
**445 checks**, zero issues, from all 28 raw before/after buffers, launch/API
records and frozen shader/source evidence. The three distinct register tags
are sampled in lanes 0, 1 and 16.

| Bank setter | No explicit trap | Explicit `s_trap 3` |
|---|---|---|
| `0x00`, `0x41`, `0x45`, `0x82` | Sampled tags unchanged | Sampled tags unchanged |
| `0x05` | Sampled tags unchanged | v257 lane 0 tag copied into v1 lane 0 |
| `0x81`, `0x85` | Sampled tags unchanged | v257 lane 0 tag copied into v513 lane 0 |

Every other sampled tag and both full guard regions remain unchanged. All
seven explicit-trap TTMP pairs equal the full expected return PC; all seven
no-explicit-trap TTMP pairs are zero. The 14 computed PCs share one inferred
shader load base. Boot, container state and kernel-log bytes are unchanged;
the saved direct-device ownership scans are complete and empty.

The local packaged first-level handler bytes match A0's published image and
the local generated header; packaged version/srcversion match the loaded
module metadata. The entry workaround reads and writes encoded `v1` before
clearing the VGPR bank fields. With unequal source/destination banks, the
read and restoration access different logical registers, matching these
controlled copies. The source is
`/usr/src/amdgpu-7.1.1-2389086.24.04/amd/amdkfd/cwsr_trap_handler_gfx12.asm`.

Evidence:

- [Independent certificate](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/b0_cwsr_controlled_20260921T022905021210Z/cpu_terminal_certificate/certificate.md), [complete certificate JSON](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/b0_cwsr_controlled_20260921T022905021210Z/cpu_terminal_certificate/certificate.json), and [terminal lifecycle audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/b0_cwsr_controlled_20260921T022905021210Z/root_terminal_lifecycle_audit.json).
- [Installed handler identity](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/b0_cwsr_installed_identity_cpu_20260921.json).

This demonstrates the controlled bank-copy mechanism on B0. It does not
attest in-memory handler bytes, cover the complete register file, or establish
that a trap caused an original B0 arithmetic/training failure. The API log
comes from the reviewed runner rather than an independent argument interceptor.

## Ordered DB → DW → DX control completed

Run
`resume_20260920/projection_ordered_private172_fixed460_selector_20260921_20260921T020746757743Z/`
completed **five calls**, all reported outputs exactly zero, exit 0; session
82922 was drained. The helper preserves DB reduction → DW GEMM → DX GEMM
order without intermediate synchronization. It uses N=1,959,558, zero BF16 DZ,
synthetic finite X and the retained real B0 projection weight.

The independent native audit passes **220 checks**. All ten launch/return
pairs join complete declared argument fields, input/output pointers, shapes,
strides and report iterations. The exact symbols map to these disk entries:

| Operation | Symbol family | Native M / N / K | Workgroups |
|---|---|---|---:|
| DW | candidate 102, `MT32x16x32` | 2048 / 512 / 1959558 | 2,048 |
| DX | candidate 172, `MT128x240x128` | 512 / 1959558 / 2048 | 460 |

The fixed460 setting applies to DX here; DW uses 2,048 groups. Candidate
numbers are disk catalog mappings, not independently measured runtime
selection indices. The audit does not identify the Torch DB kernel or attest
loaded ELF bytes. Complete output readbacks supply completion evidence
separately from asynchronous host launch returns.

Boot/catalog and ownership checks pass. Kernel logs add NIC warnings, with
no new GPU-fault candidates. This is a bounded passing sequence control,
not original failed training operands or identification of a corrupting writer.

Evidence: [native audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260920/projection_ordered_private172_fixed460_selector_20260921_20260921T020746757743Z/cpu_ordered_native_terminal_audit/audit.md), [numerical terminal audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260920/projection_ordered_private172_fixed460_selector_20260921_20260921T020746757743Z/cpu_ordered_terminal_validated/audit.json), [lifecycle audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260920/projection_ordered_private172_fixed460_selector_20260921_20260921T020746757743Z/root_lifecycle_terminal_audit.json).

## BN8 LayerNorm cap256 smoke completed

Run `resume_20260921/backward_20260921T023949035814Z/` completed **five calls**,
exit 0 at **02:39:54 UTC**, with `--stage helper --block-n 8 --rows 2468813
--repeat 5 --max-vgpr 256`. The **124-check** compiled-artifact audit confirms
the actual DX object has **256 VGPRs**, **1,144 private bytes** and **zero
bank setters** in both saved assembly and independent HSACO disassembly.
Its reported spill count is 286; ELF spill count is 287. These counters are
preserved separately. Reduction remains uncapped, with 13 VGPRs and no spills.

A separate **77-check** comparison finds that the complete reduction object
matches all six retained same-shape uncapped helpers, including the positive
`resume_20260919/weighted_ln_helper_bn8_r2468813_20260919T041107Z/report.json`.
That positive has clean checkpoint 80 followed by checkpoint-90 NaNs in DX
(3,072 elements), norm weight gradient (2), and norm bias gradient (1), with
all input byte guards unchanged. Its DX compiled hash is `eae3693f…`,
576 VGPRs, zero spills. The older raw-zero failures at inner calls 115/212
use different artifacts (`e34c0e52…`); they are not the matched helper baseline.

The cap changes scratch traffic and scheduling as well as bank usage.
This short pass does not establish suppression or a NaN fix.

Evidence: [compiled audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/backward_20260921T023949035814Z/cpu_compiled_terminal_audit/README.md), [matching reduction comparison](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/backward_20260921T023949035814Z/cpu_reduction_comparison/README.md).

## Fresh unchanged BN8 helper: 1,000-call nonreproduction

Run `resume_20260921/backward_20260921T030755301716Z/` completed at
**03:08:11.120946 UTC**, exit 0; root drained session **88104**. It used
the uncapped BN8 helper, N=2,468,813, seed 947, 1,000 calls, and checkpoints
every ten calls. The report contains no failed per-call zero probes across
all 100 checkpoints; all **600** full input byte flags are true.

The concise independent CPU report/lifecycle comparison passes **49 checks**.
Both complete `actual_norm_launches` objects equal the historical positive:
DX `eae3693f…`, 576 VGPRs/zero spills; reduction `1f333850…`, 13 VGPRs/zero
spills. Shape, seed, poisoning, setup checks, source SHA, launch configs and
input/snapshot sizes match. The source-root path differs and the fresh report
explicitly records `AMD_SERIALIZE_KERNEL=0` where the positive omits it;
all other reported environment values match.

Before/after direct ownership scans are complete and empty, and boot is
unchanged. All 2,951 kernel-message bodies match; ISO timestamps shift by
one microsecond, so the raw log files are not byte-identical. This CPU audit
checks reported GPU probes and guards, not retained full tensor bytes or
executed shader memory. It independently confirms the archived comparison,
not a new numerical oracle for the passing GPU run.

**The unchanged helper passing again makes a longer capped-only comparison
nonlocalizing.** It cannot establish that the cap suppresses the intermittent
failure. The next work moves to a targeted trap-entry repair candidate while
preserving the need for a natural arithmetic/training bridge.

Evidence: [report and terminal comparison](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/backward_20260921T030755301716Z/cpu_terminal_audit.json), [original report](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/backward_20260921T030755301716Z/report.json).

## Culprit boundary, repair work and remaining proof

The original B0 step-51 output-gradient GEMM receives already-huge finite
`dout`; the original output matches CPU computation. The preceding observed
output backward is bounded. The unobserved preprocess/projection/input-LN
interval and possible later writes remain the training target. Its row-388482,
column-26 fingerprint is consistent with LN spreading a spike, but original
projection DX, norm X/DY/statistics and raw parameter gradients are missing.

The [A0 record](../mi450_a0/mi450_a0.md#current-status) provides a stronger
natural-arithmetic bridge on that host: matched bank81 control FAIL203,
equal-bank PASS1000, with sampled trap-return PCs and donor arithmetic.
It is evidence for a concrete B0 experiment, not a substitute for B0's
missing producing-boundary capture. The local
[counterfactual plan](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/cwsr_natural_arithmetic_counterfactual_plan_20260921.md)
records static bank coverage and limitations.

An offline repair to the handler's bank-sensitive entry sequence now passes
the userspace comparison below. It has not been installed or validated in the
live privileged handler.
The required proof is a reviewed targeted change, a controlled failing
sentinel restored around the candidate, and a bridge to naturally failing
arithmetic or original training. A last-visible TTMP snapshot is not a
complete trap history. A passing uninstrumented training run is necessary
validation but cannot by itself establish that causal bridge.

The September 20 baseline already completed **3,000 optimizer steps**, with
all 300 sampled losses finite and final loss **0.14524586498737335**. It used
the same production code and attention-cap256/LN-BN1 settings that previously
failed at 60/70. No new production repair explains that pass. After a targeted
repair is established, repeat timestamp-zero training through all 3,000
optimizer steps with finite losses and no GPU fault.

## Fresh compact attention control — 1,000 calls pass

Run `resume_20260921/attention_uncapped_current_20260921T031326766614Z/`
completed at **03:13:42 UTC**, controller exit 0; root drained session **59551**.
It uses the 32-sequence real step-55 prefix, uncapped attention, the required
image/Triton, and the frozen production tree. The current diagnostic adds
per-call checks of input contents and exact failing-output retention.
All **1,000 calls and 1,000 checkpoint records pass**. There is no failing
tensor payload. The compiled attention kernel has **654 VGPRs, zero spills**,
hash `f299f55b…`; this differs from the historical positive's `9a3dec64…`
identity, and no executable-byte equivalence is claimed here.

The [terminal audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/attention_uncapped_current_20260921T031326766614Z/root_terminal_audit.json)
checks completion, all checkpoint records, requested input/failure checks,
unchanged boot and kernel-message bodies, and complete empty device-owner
snapshots. Passing tensor contents were not retained; the audit checks the
archived flag reports. Additional per-call input checks and pre-loop CPU
snapshots change timing. This control supplies neither a recurrent natural
failure nor evidence that a repair suppresses one.

Two preceding launch attempts are retained as diagnostic setup errors:
`attention_uncapped_current_20260921T031203244193Z/` rejected the newer
`--failure-dump` option in the frozen older script before execution;
`attention_uncapped_current_20260921T031230293294Z/` failed to import a flat
script dependency. The completed run uses the archived current diagnostic
with both frozen source and scripts directories on `PYTHONPATH`. Production
kernel files were not changed. These attempts are not arithmetic failures.

## Targeted entry repair — 512 GPU cases and full handler build

The [persisted patch](patches/cwsr_gfx1250_entry_bank_fix.patch) contains both the
driver assembly change and the regenerated gfx1250 handler array. Its
[build/validation notes](patches/README.md) describe the exact source target
and limits. No installed source, module or runtime was changed.

The repair temporarily makes DST equal to incoming SRC0 when zeroing and
restoring encoded `v1`. It restores DST around the prefetch and on exit,
preserving the same logical register and original MODE/EXEC. Existing TTMP2:3
hold the temporary bank fields on either side of their zero-address use.
Two vector no-ops precede every MODE write; two scalar wait states follow it.
Independent source review checks that subsequent handler paths overwrite
these free temporaries before use. The non-target `CHIP_GFX12` preprocessing
output is unchanged.

Run `resume_20260921/b0_cwsr_entry_fix_controlled_20260921T032103984103Z/`
completed **03:21:10.052441 UTC**, exit 0, root session **20557 drained**.
It compares the original and candidate sequences across all 256 bank settings:

| Sequence | Predicted cross-bank copies | Sampled tags preserved |
|---|---:|---:|
| Original | 192 | 64 |
| Candidate | 0 | 256 |

The [independent raw-buffer audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/b0_cwsr_entry_fix_controlled_20260921T032103984103Z/cpu_terminal_audit.md)
passes all 30 aggregate checks: all 1,024 before/after buffers, MODE, EXEC,
completion/reserved words and complete allocation guards. All 3,079 API records
and 512 launch/report joins agree. Boot/container identities, runtime hashes
and kernel-log bytes are unchanged; device-owner scans are complete and empty.
The runner's nine adversarial CPU tests and independent compiled review also
pass. Twelve register/lane tags are sampled per case; this is not a complete
register-file integrity test.

The shader substitutes ordinary SGPRs for TTMP scratch and injects no trap.
It directly tests the offending instruction sequence and the repair, but does
not validate asynchronous privileged entry, full context save/restore, or the
original training NaN. The earlier explicit-trap sentinel independently
establishes that the installed handler exhibits the predicted corruption.

The [full-handler build certificate](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/cwsr_entry_fix_candidate/full_handler_build_audit.json)
reproduces all **5,656 bytes** of the installed original image. The candidate
is **5,736 bytes**, below the 6,144-byte driver limit. It replaces the 48-byte
entry at `0x3c` with 128 bytes and retargets the initial restore branch by
80 bytes. The entire shifted suffix is byte-identical. Only the corresponding
header array changes. A separate SP3 compatibility build emits the two existing
prefetch instructions as their exact original words; the canonical patch
retains the original mnemonic syntax.

The isolated CPU module build **completed successfully**, building all nine
modules from a complete source copy with 16 compile jobs. The compiled
`amdgpu.ko` embeds exactly the 5,736-byte candidate handler; kernel vermagic
matches `6.14.0-37-generic`. Independent ELF inspection and complete imported/
exported symbol-CRC comparisons match the installed modules. This supports
replacing only `amdgpu`; the eight companion modules can remain installed.
The full installed source tree and installed module hashes are unchanged.

The [staged deployment module](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/cwsr_entry_fix_candidate/deployment/amdgpu.ko.zst)
is **6,383,454 bytes**, SHA-256
`a74f05f4fa27c1eeab667ebd4d7142b2ab184609781ce96fca04e2fa82266201`.
It is a separately stripped/compressed copy of the retained full build and is
unsigned. The original installed module and current initramfs are preserved
under `deployment/rollback/` with verified hashes and loaded identity. The
[build plan and evidence](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/cwsr_entry_fix_candidate/module_build_plan.md)
record the unprivileged staging procedure, source/byte audits and limitations.

Installation, module loading and reboot have not been performed. The restored
rule in [USER_RULES.md](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/USER_RULES.md)
prohibits reset/reboot/container restart. A live-driver test requires explicit
authorization for that operational change. A host reboot also interrupts
`mi450_c`, which has a standing leave-untouched rule. The candidate is now
concrete and reviewable; the privileged trap test and 3,000-step training
validation remain outstanding.

The [deployment/validation/rollback plan](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/cwsr_entry_fix_candidate/deployment_validation_plan.md)
covers same-kernel module and initramfs replacement, reboot/manual driver load,
the explicit-trap acceptance test, subsequent arithmetic/training checks and
rollback. Its state-changing commands have not been executed.

Later results should be appended only after checking authoritative terminal
reports and draining the live tool session. A polling timeout does not mean
the worker exited and is not a reason to restart it.

## Resumed debugging: compiled LayerNorm NaN mechanism

The user requested continued debugging without authorizing driver activation.
The installed driver, module and boot remain unchanged. The isolated candidate
repair remains staged. The later publication request applies to these
documentation files and their linked source patch.

Run `resume_20260921/ln_trap_arithmetic_20260921T042040470947Z/`
completed **04:20:41.485831 UTC**, exit 0; root session **34297 is drained**.
It uses the exact historical `eae3693f…` / HSACO `788c2b99…` weighted-LN DX
kernel, with 576 VGPRs and no spills. Each modified variant changes exactly
one four-byte instruction; metadata, branches, operands and register allocation
are otherwise preserved. Tiny N=8/grid1, N=16/grid2 and N=32/grid1 inputs cover
one tile, multiple workgroups and loop reuse. Inputs are zero DY and finite
X/W/statistics; the mathematical DX and FP32 DW/DB partials are all zero.

| Variant | Three shape results |
|---|---|
| Original kernel | All outputs zero |
| `V_NOP` at `0x6290` replaced by `S_WAIT_IDLE` | All outputs zero |
| Explicit `S_TRAP 3` at equal-bank `0x6278` | All outputs zero |
| Explicit `S_TRAP 3` at unequal-bank `0x6290` | Only partial DW/DB column 263 is NaN, in every launched workgroup |
| Explicit `S_TRAP 3` at unequal-bank `0x62a8` | Only partial DW/DB column 263 is NaN, in every launched workgroup |

All **15 cases** match the precomputed coordinate prediction. DX remains zero
throughout. All nine allocations per case retain complete before/after bytes;
inputs and guards are unchanged. The
[independent raw audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/ln_trap_arithmetic_20260921T042040470947Z/independent_raw_cpu_audit.json)
passes **757 checks**, reading all **270 complete buffers** and checking all
**723 HIP API records**, launch packets, source pins and lifecycle evidence.
Boot/container identities are unchanged, owner scans are complete and empty,
and no new kernel-log fault is reported by this experiment.

The [instruction-level evidence](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/ln_arithmetic_trap_bridge/arithmetic_evidence.json)
traces physical v257 lane 0 to DY row `(pid + k*grid)*8 + 3`, column 263.
At the selected late windows it remains live for the parameter-gradient
accumulators. Physical v1 contains the temporary result of
`V_DIV_SCALE_F32(512,512,0)`, which the published ISA defines as NaN; the later
division-fixup sequence ordinarily handles it. Setter `0x40` has SRC0 bank 0
and DST bank 1, so the defective trap entry copies v1 lane 0 into v257 lane 0.
The following arithmetic propagates it to precisely the observed partials.
The second selected site follows a sign flip of that donor temporary.

This establishes a controlled NaN-generation route inside an actual compiled
LN kernel. It is **explicit injection**, not a naturally captured trap during
the historical standalone failure or original training. The original/wait-only
controls use the same instruction location, while the equal-bank trap control
uses a nearby different location. A further matched same-site bank intervention
has not run. The historical helper's parameter NaNs include column 263,
but its additional weight-gradient column 5 and 3,072 bad DX elements remain
unexplained by this late-site injection. Matching columns or NaN bits alone
does not establish a shared original event.

## Fresh original-boundary training capture: illegal-address abort

Run `original_boundary_cap256_bn1_20260921_20260921T035509Z/` used the same
143 frozen source files, required image/Triton, timestamp zero, seed 1,
cap256/BN1 and `AMD_SERIALIZE_KERNEL=0`. The new outer controller replaces
the old fuser gate with the complete process/thread character-device FD scan;
the original capture launcher and production source are unchanged. It retains
the original projection and input-LN stage references, including LN inputs,
statistics, final gradients and original launch metadata. Copying happens
after backward, so later device writes can still affect retained bytes.

The run terminated **04:17:42.020770 UTC**, wrapper/launcher exit 1; root
session **40267 is drained**. All **83 logged losses from 10 through 830**
are finite, last **0.13871853053569794**. It then reports illegal memory access
and SIGABRT, saving **no tensor payload**. It did not reach the 1,000-step
diagnostic bound. The
[terminal audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/capture_control_20260921T035508530526Z/independent_cpu_terminal_audit.json)
passes all 35 provenance/lifecycle checks; these do not turn the failed
training run into a numerical pass.

The [fault context](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/capture_control_20260921T035508530526Z/independent_cpu_fault_context.json)
records ten printed read-side gfxhub page faults at 04:17:35 UTC, naming
Python PID 972822, PASID 404, ring 24/vmid 3 and AID1.XCD3. Six reported pages
span `0x12b699000` through `0x12b6e9000` at 64-KiB spacing. The error surfaced
in `SetDevice` during tensor copying; this asynchronous stack and the page
records contain no producing shader PC or symbol. No reset/reboot occurred,
and both terminal and subsequent direct owner scans are empty. The later
tiny LN experiment is a separate controlled result after this abort.

## Native DW103 bank intervention: static proof and sparse runner

The new B0-local matched pair copies the LDS address v283 to unused v676 on
every visit, then changes 15 selected address operands and 16 bank setters
in the equal-bank arm. Both arms share the copy and branch trampoline; all
ordinary arithmetic and the 688-VGPR hardware allocation are preserved.
The [independent certificate](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/dw103_equal_bank_bridge/independent_cfg_certificate.md)
covers all 699 indirect transfers, direct branches, physical bank flow,
inclusive VGPR tuples, SGPR/EXEC/SCC lifetimes and unchanged descriptors.
An initial root scan missed alternate indirect-jump mnemonics; that claim is
superseded by this complete raw-opcode and control-flow review.

The isolated direct-HIP runner consumes the exact 176-byte/33-field native
ABI and never modifies an installed catalog. Stock run
`dw103_equal_bank_bridge/runs/stock_20260921T043159687356Z/` completed
**04:32:41 UTC**, exit 0, root session **91977 drained**. Five sparse cases
match exact BF16 outer-product oracles, covering reduction rows 0, 63, 64,
2,324,350 and all four together; complete input hashes and output guards
pass. The [independent terminal audit](../../../../runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260921/dw103_equal_bank_bridge/runs/stock_20260921T043159687356Z/independent_cpu_terminal_audit.json)
passes **241 checks**, including all 655,360 output values, 40,960 output-guard
bytes, 163,840 input-guard bytes and 3,680 successful HIP call pairs. Full input
bodies were hashed by the pinned runner but not retained; reconstructed expected
hashes match its saved observations. All three ownership snapshots are empty,
boot/container identities are stable, and the kernel log is unchanged.
The matched control/equal GPU comparison remains **unrun** at wrap-up.
Passing sparse arithmetic validates the direct ABI and ordinary semantics,
not the original training cause or a natural failure rate.
