# MI450 gfx1250 CWSR register-bank corruption

Updated 2026-09-22 UTC after the second host AC cycle and the subsequent
CPU-only preparation reviews. The corrected CWSR
candidate was first-loaded on fresh boot
`6478859b-3092-4792-a933-e91c311f2cd1`, with version `7.1.1.31300009`,
srcversion `8336BBB74E6C4D53275A4E5` and CWSR enabled. Actual GPU tests now
show that it prevents the controlled register-bank corruption, LayerNorm
NaNs, tiny and full historical attention corruption, and the complete
historical step-405 DW corruption. Native kernels, inputs and launch
parameters match the original-driver comparisons.

| Corrected-driver test on this boot | Completed result |
|---|---|
| Health and sentinels | 4,000 exact health checks; all 14 tag/MODE/EXEC cases preserved and all seven explicit trap return PCs exact |
| LayerNorm and tiny attention | All 24 processes / 36 dispatches pass; all 3,586 reproduced LN NaNs and all 512 tiny-attention wrong DQ words absent |
| Full historical attention FAIL131 | All four arms / eight dispatches pass; all 1,087,635,456 DQ/DK/DV words positive zero |
| Historical step-405 DW | All four arms / 16 calls equal the complete ordinary baseline; all 1,030 historical differences corrected, including 640 NaNs |

Complete raw/native consumers and independent comparison reviews passed.
All producers exited and final all-device idle gates passed. The full
attention capture also has a verified lossless archive, with raw retained.
[Actual AC2 results, receipts and scope](evidence/current_20260919/CWSR_candidate_AC2_validated_20260922.json).

**End-to-end training is still unvalidated.** On the preceding candidate
boot `2001b9f5-abbc-4032-a56f-9d41a57de7f3`, health, LayerNorm and attention
also passed, but default-policy E2E1000 stalled during its first backward
pass in a shader-loading PM4 code-cache invalidation wait. It produced no
completed numerical verdict. The original-driver binding run had the same
observed stall path. This does not establish the stall's physical cause.
The AC2 E2E attempt with `HIP_ENABLE_DEFERRED_LOADING=0` passed its pre-HIP
worker gate but aborted during ROCm library initialization because a
registered ROCsolver/rocPRIM symbol could not be resolved. Training never
started. All processes exited, the private trace was removed, devices were
idle and kernel logs were unchanged.
[Preserved eager-startup failure and clean cleanup audit](evidence/current_20260919/CWSR_candidate_AC2_E2E_eager_abort_20260922.json).

The subsequent AC2 E2E attempt restored default deferred loading and
successfully preloaded the original FBGEMM `linearize_index_kernel` through
`hipFuncGetAttributes` in 24.225 ms. Training then stalled in the first
backward while loading `split_embedding_backward_codegen_find_long_segments`
from `fbgemm_gpu_tbe_training_backward.so`. All 21 runtime frames through
`hipLaunchKernel` match the preceding stall by library and file offset;
the first FBGEMM caller changed. The host packet and its ACQUIRE_MEM command
were unchanged across 160.522 seconds. No backward completion or resolved
numerical verdict was produced. This shows that preloading one kernel is
insufficient; it does not establish whether code-cache maintenance or an
earlier nonprogressing training kernel caused the wait.

After more than 15 minutes without journal progress, only the owned
supervisor received SIGINT through a verified pidfd. The worker and launcher
remain preserved; the final idle gate failed and the private trace is
stopped. Further GPU experiments require recovery through a new host AC
cycle. No driver reset, reload or recovery diagnostic was attempted.
[Targeted-preload stall evidence](evidence/current_20260919/CWSR_candidate_AC2_E2E_targeted_stall_20260922.json).

The earlier strict post-load MC60 stop is preserved as historical evidence.
The diagnostic continuation uses the already authorized narrow policy for
complete known CPU8/MC60 corrected-error records; arbitrary faults still
reject. One matching record preceded AC2 activation, and the completed
workload intervals contained no new MC60 reports. The numerical PASS results
do not certify platform health. The `VM_PAGE_FAULT` insertion inventory label
was separately adjudicated using source and all 309 insertion lines.
[Earlier activation and strict validation stop](evidence/current_20260919/CWSR_post_ac_candidate_activation_20260922.json),
[original frozen preparation](evidence/current_20260919/CWSR_candidate_validation_preparation_20260922.json).

A KFD `hqds` read on the older original-driver boot caused a separate SDMA
debug-dump NULL dereference. The corrected CWSR candidate retains that
diagnostic function; it must not be invoked. No live reload or GPU recovery
was used for AC2 validation.
[Historical stall and diagnostic-fault record](evidence/current_20260919/E2E_bind_stall_and_HQD_oops_20260922.json).

The component to repair is the AMDGPU/KFD **first-level CWSR trap handler**,
specifically `VMEM_ON_TRAP_ENTRY_WA` at `L_NOT_WAVE_START` in
[the installed gfx12 handler source](/usr/src/amdgpu-7.1.1-2397345.24.04/amd/amdkfd/cwsr_trap_handler_gfx12.asm:261).
The installed original module is
preserved on disk: `7.1.1.31300009`, srcversion `654C1DDE7A9A3E129BA9553`.

**This is a confirmed defect, not a demonstrated sole remaining E2E cause.**
The candidate E2E attempt used a handler already validated by component
tests, but stalled before a numerical verdict. The evidence does not yet
establish that this patch alone unblocks training, or that existing source
fixes and execution workarounds can be removed.

The handler reads encoded `v1` through the inherited SRC0 bank, temporarily
writes encoded `v1` through the inherited destination bank, then restores the
saved word through that destination bank. These instructions execute before
bank normalization. Unequal banks therefore copy lane 0 between different
physical registers. In the failing GEMM, those registers can hold an
accumulator, a matrix operand, or the persistent X load address.

| Selected bank setter | Observed copy | Possible effect in the retained DW kernel |
|---|---|---|
| `0x05` | `v257[0] → v1[0]` | Accumulator corruption |
| `0x81` | `v257[0] → v513[0]` | Matrix-operand corruption |
| `0x46` | `v513[0] → v257[0]` | Persistent load-address corruption |

These are setter encodings; they must not be confused with the raw MODE
field byte used by the inline test.

The critical original sequence saves `exec_lo`, forces lane 0 active, reads
`v1` into `ttmp15`, writes zero to `v1` for the prefetch workaround, and writes
`ttmp15` back to `v1`. Equal encoded register names do not imply equal
physical registers while inherited SRC0 and DST bank selectors differ.
Restoring EXEC cannot undo that cross-bank write. The patch normalizes the
bank selectors before the first VGPR access and restores their original
values after the last one; changing only the later CWSR save/restore loops
would leave this entry-path defect intact.

## Evidence identifying the defect

The retained controlled seven-event DW replay reproduces every byte of the
original attempt-405 gradient: 131,072 BF16 words, including 640 NaNs, 384
enormous finite values, and six smaller differences. Equalizing the bank
routing while retaining the traps restores the complete ordinary baseline.
Relocating live values away from physical v1/v257/v513 also prevents this
controlled corruption.

The same historical-input quartet now passes on the corrected driver,
including the unchanged `sink_control_trap` ELF. Every complete 262,144-byte
output in all 16 calls equals ordinary baseline SHA256
`da3cc5656e1b105fc638d5556399233140fd2e5542ef822b5d7e9b4d8275d11e`.
The four original `sink_control_trap` calls equal the historical corrupted
output; the other three original arms equal baseline. All 1,030 differing
BF16 words are corrected: 640 NaNs, 384 extreme finite values and six
smaller finite differences, including two sign flips. The complete consumer
rehashed 14,280,812,544 retained X/DY bytes. The independent reviewer reread
all 16 original and 16 corrected outputs and authenticated the large-input
hash joins without repeating that 14 GB scan. The arm name `sink_candidate`
denotes the earlier kernel register relocation; all four current arms use
the corrected driver.

The new ordinary-shader comparison directly executes the original and
corrected prologues without an explicit trap. It covers all 256 raw MODE
bank bytes and eight EXEC masks, with every lane of v1/v257/v513/v769 tagged.
All **6,144 calls** pass the frozen matrix: the original prologue produces
exactly **1,536 predicted cross-bank copies** in its 2,048 calls; the corrected
prologue and no-prologue control preserve all tags in 2,048 calls each. Full
MODE and EXEC are preserved. All packet words, prefills, native argument
blocks, library identities and lifecycle checks pass independent audit.
The body uses SGPR stand-ins for TTMP scratch operands and preserves the
installed readlane/writelane instruction encoding outside those operand
substitutions. This validates ordinary instruction behavior; it does not
exercise privileged trap-entry scheduling or install the correction.

[Complete inline audit](evidence/current_20260919/CWSR_inline_correction_20260921_evening.json),
[external process and driver checks](evidence/current_20260919/CWSR_inline_lifecycle_20260921_evening.json).

The original production attention HSACO now has a direct controlled
reproduction at its late packed-dS window. On finite synthetic inputs with
zero upstream gradient, four unequal-bank trap calls each produce exactly
128 predicted finite DQ words; the equal-bank trap and both NOP controls
return complete zero DQ/DK/DV in four calls each. The intervention follows
the original WMMA at `0x5154`, waits for its completion, and exposes the
physical **v1 lane 0 → v257 lane 0** route before the later DQ consumer.
The CPU oracle derives every expected word from serialized K bytes without
fitting or tolerance. The complete audit covers 393,216 BF16 output words,
416 retained buffer files, 832 guards, all sixteen 200-byte launch packets,
788 HIP calls and unchanged source/library identities. Processes exited;
the per-arm KFD-owner checks were idle and kernel logs were unchanged.
This establishes a controlled mechanism in the original attention kernel.
The unchanged quartet on the corrected driver now returns only zeros in all
16 calls, eliminating all 512 wrong DQ words across the four unequal-trap
calls. The larger historical-input replay is described below.
[Attention reproduction and controls](evidence/current_20260919/CWSR_attention_controlled_20260922_resume.json).

Full historical attention FAIL131 replay now reproduces every retained
output byte on its original inputs and geometry. Five prescribed unequal-bank
events (WGx102, waves 0–3 at K448/query640, and WGx43, wave 0 at
K448/query448) produce exactly all 80 historical wrong DQ words in both calls;
all remaining DQ/DK/DV words are zero. Equal-bank traps and both matched NOP
controls return only zeros. Independent audit covers all 1,087,635,456 output
words, 144 complete raw files, 288 guards, eight native launch packets and
596 HIP calls. Every worker exited, all-device idle checks passed, and no
new kernel-log lines appeared. All 13.05 GB of raw evidence has lossless,
decompression-verified archives. The schedule is a controlled sufficient
intervention; the actual historical trap sequence remains unobserved.
[Full historical-input attention replay](evidence/current_20260919/CWSR_attention_FAIL131_20260922_resume.json).

The same full historical-input quartet now passes on the corrected driver:
every DQ/DK/DV word is positive zero in all eight calls. Complete pinned
auditors scanned all 13,053,235,840 raw bytes. Independent AC2 review verified
the complete inventory and hashes against the prior independently scanned
corrected capture, then reread all 144 files' guards and the 80 historical
DQ coordinates in all eight outputs (75,008 bytes). It did not repeat the
13 GB scan. Every file also passed lossless decompression/hash verification.

The LayerNorm earlier X-load window now explains controlled whole-row NaNs
and partial DW column-5 corruption. Prior-loop zero-numerator division can
leave a NaN in v1's upper half; `V_MOV_B16 v1.l` preserves that upper half.
An unequal-bank trap before native `0x4680` at bank byte `0x80` copies v1
to v513, corrupting an X operand at column 5. N16/grid1 produces 512 DX
NaNs in row 11, while N56/grid1 produces 3,072 in rows
11/19/27/35/43/51. Both have a partial DW NaN at column 5 and zero DB.
N8/grid1 and N16/grid2 return zeros, as do every stock, NOP and equal-bank
control. The full 20-case audit covers 440 raw buffers, 271,360 output
elements and 1,280 HIP calls; inputs and guards are unchanged, all workers
exited, all-device checks were idle and all kernel deltas empty. This is A0
execution of the native text matching the published B0 executable hash.
Matching the synthetic 3,072 count does not recover the historical B0 row
coordinates or establish unique attribution. DW/DB here are FP32 partials.
[LayerNorm matrix and complete audit](evidence/current_20260919/CWSR_LN20_controlled_20260922_resume.json).

On the corrected driver the same 20-case LN matrix is entirely zero,
including both unequal-trap NaN cases: all 3,584 DX NaNs and both partial DW
NaNs are absent. The combined LN/tiny-attention consumer verified 856 raw
files, 7,366,528 bytes and 1,024 native pins with exact unique inventory.
Independent original/current rereads confirm byte-identical native ELFs,
geometry/scalars and 3,683,264 prefill bytes.

The projection kernel's controlled four-arm matrix also passes. A selected
first-panel trap with the original unequal banks produces exactly four
nonzero BF16 DX words at `(0,2)`, `(0,34)`, `(0,66)` and `(0,98)`, with bits
`bc08`, `3cdf`, `bc53` and `bd14`. The three matched NOP/equal-bank controls
return only positive zeros. All four calls within each arm are byte-identical;
every one of the 1,003,293,696 output words per call was independently scanned.
Complete final W/DZ bytes are unchanged across all arms, and all 212 native
launch-argument bytes match the reviewed packet apart from recorded dynamic
pointers. The full 64.22 GB of retained raw data has verified compressed
archives. This demonstrates the predicted copy's effect and prevention by
equalizing the banks at the injected trap; actual corrected-driver repair
and the spontaneous event's original trap history remain unvalidated.
[Projection reproduction and controls](evidence/current_20260919/CWSR_projection_controlled_20260922_resume.json).

The natural zero-DY comparison also completed: the matched original control
fails at call 103 with 512 finite wrong BF16 words; the relocated candidate
passes 1,000 calls with all 131,072,000 words positive zero. Complete native,
prefill, input and output audits pass. The control words fit the historical
v257 donor model exactly, although the numerical fit does not identify the
physical trap event. This supports the relocation workaround for the tested
kernel under natural exposure. This original-driver comparison does not by
itself validate the corrected driver or training stability.
[Natural comparison](evidence/current_20260919/DW_sink_natural_zero_20260921_evening.json).

## Observed spontaneous trigger

The new private kernel traces identify this initiator in every captured
eviction stack:

```text
Linux automatic NUMA balancing: task_numa_work
  → change_prot_numa / change_protection
  → MMU-notifier invalidation
  → svm_range_cpu_invalidate_pagetables
  → kgd2kfd_quiesce_mm
  → KFD queue eviction (SVM trigger)
```

The first 400-call NOP-only sweep has nine fresh predicted copies and three
fresh equal-bank observations. Each has one target-process SVM eviction and
restore pair; the other 388 calls have neither. The second sweep adds kernel
stacks: six fresh copies and four fresh equal-bank observations each have
one pair, with none in the other 390 calls. All ten eviction stacks contain
the chain above. Both runs retain every image and report no trace losses or
probe misses. Thus automatic NUMA protection scanning is an observed trigger
in these probes. Queue helper entries record attempts, not hardware context
save acknowledgments, and neither run recovers attempt405's original trap
history.

[First trace result](evidence/current_20260919/SVM_fresh_TTMP_20260921_evening.json),
[stack trace result](evidence/current_20260919/AutoNUMA_eviction_stacks_20260921_evening.json).

The process-local policy counterfactual now also passes complete independent
audits. Each arm executes the same 400 NOP-only cases on the original driver.
Global `kernel.numa_balancing` remains 1 throughout.

| Process policy | SVM eviction/restore pairs | Observed register copies |
|---|---:|---:|
| `--membind=all`, allowed nodes 0–1 | 0 | 0 |
| Default policy | 12 | 5 |
| `--balancing --membind=all`, allowed nodes 0–1 | 10 | 5 |

Plain binding clears eligibility for automatic NUMA scanning on ordinary
task-policy VMAs. Re-enabling balancing with the same allowed-node mask
restores SVM eviction stacks and copies. Actual physical page placement and
timing are not asserted identical. All 1,200 cases, 3,600 retained images and
7,230 HIP calls pass their audits, with no trace losses or probe misses.
Every captured eviction stack comes through `task_numa_work`.

The default arm retains two snapshot exceptions: one copied register has
an unchanged masked TTMP PC, and one unchanged-register case has an SVM pair
without a fresh PC. Both show why a single return-PC snapshot is not a
complete trap history. This is evidence for suppressing the observed trigger
in the probe, not a repair of the handler or validation for all training VMAs.
[Complete policy counterfactual](evidence/current_20260919/AutoNUMA_policy_counterfactual_20260921_evening.json).

## Corrected component and build

The [source patch](cwsr_gfx1250_bank_fix.patch) exactly produces the retained
candidate assembly source when applied to the installed source tree. Before
touching v1, it saves and clears the six MODE bits selecting DST/SRC0/SRC1
banks, then restores them after restoring v1. It leaves SRC2 and other MODE
fields unchanged. It preserves the intended zero-address speculative prefetch using
`k + (63 - k) - 63`, keeping the saved bank bits in a scratch TTMP pair.

The clear must use **`S_SETREG_B32`**, as the patch does. The documented
gfx1250 masked-immediate erratum makes `S_SETREG_IMM32_B32` unsuitable for
this partial MODE update. The earlier immediate-clear proposal was withdrawn.

The separately reconstructed executable handler is 5,672 bytes, SHA256
`68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80`.
Its build uses disassembly-derived LLVM assembly with explicitly preserved
instruction encodings; acceptance of the source patch by the original SP3
assembler has not been tested. The baseline reconstruction matches all
5,656 installed bytes. The existing
scratch module passed CPU-code, branch, read-only-object and relocation
comparisons. Building the source patch alone does not update the embedded
handler array: the audited regeneration/build procedure must also be used.
See the main record's
[handler build](evidence/current_20260919/CWSR_complete_handler_build.json)
and [module audit](evidence/current_20260919/CWSR_module_byte_audit.json).

Actual corrected-driver execution now passes the AC2 tests above. Earlier
live-reload evidence remains a separate unresolved incident: the September 22
rebuilt-baseline load has a durable `insmod` start and child identity after
successful original-module removal, followed by an observation gap and
reboot without a load-completion receipt. The following boot records fatal
processor/context-corrupt BERT errors, but no preserved stack identifies
the cause. The rebuilt baseline's complete ELF differs from the installed
original only in 20 build-ID bytes; the original additionally carries a
signature trailer, and signature enforcement is disabled. This rules out
an unintended code/data/relocation change in that baseline. The unchanged
installed original was restored with `modprobe` on the new boot and all
four data-movement health arms passed 1,000 calls. A live reload should not
be treated as a validated activation path for the candidate.
[Reload evidence](evidence/current_20260919/CWSR_reload_recovery_20260922_resume.json),
[complete baseline comparison](evidence/current_20260919/CWSR_baseline_loading_input_20260922_resume.json).
Natural-failure suppression and completed training remain separate required
checks; a trigger-avoidance workaround would not repair the handler. AC2 used
a single first load after a new AC cycle. The health adapter's documented
release-coordination incident preserved both versions and restored its exact
launched bytes; actual result terminals were unchanged, and independent
manifest and full native/raw reviews passed. Its provenance receipt is pinned
in the AC2 result record.

## E2E coverage and remaining uncertainty

September 22 adds matched process-policy evidence in the original projection
and uncapped attention kernels. Projection gives default FAIL159 / bind
PASS1000 / balancing-enabled bind FAIL704; attention gives FAIL131 /
PASS1000 / FAIL174. Complete retained failures and native/source/policy
identities pass CPU audit. AutoNUMA eviction stacks appear in both failing
policies and disappear under plain binding. Other allocation/unmap/teardown
SVM events remain. These comparisons support common trigger exposure;
binding is still a workaround and does not validate the handler correction.
[Complete two-component record](evidence/current_20260919/component_policy_counterfactuals_20260922.json).

The original attention FAIL85 is now exactly reproduced by a CPU model of
one packed K-to-dS copy, including every other zero in its full logical
DQ/DK/DV outputs. Independent native/LDS/compiler-layout analysis identifies
the corresponding v1-to-v257 lane-0 route at bank byte 0x40. Both newer
policy failures also have complete CPU reconstructions through the later
packed-dS window. The controlled synthetic GPU intervention described above
now reproduces and prevents its predicted corruption; the full FAIL131
historical-input intervention above also passes.

| Failure boundary | CWSR coverage established so far |
|---|---|
| A0 preprocessing Linear DW | Controlled copies reproduce the complete attempt-405 gradient. The corrected driver restores the entire baseline in all four unchanged ELF arms / 16 calls, correcting all 1,030 differences. Earlier relocation also protects controlled and bounded natural tests on the original driver. |
| B0 weighted LayerNorm | The earlier X-load trap on A0 using the published B0 native text produces partial DW column-5 NaNs and six full DX rows (3,072 NaNs). The corrected driver eliminates all NaNs in the full earlier-X-load matrix. The prior late-site test separately produces partial DW/DB column-263 NaNs. Exact historical B0 coordinates and validation on B0 hardware remain open. |
| Attention backward DQ | Five prescribed traps reproduce every historical FAIL131 output byte, including all 80 wrong DQ words. The corrected driver returns positive-zero DQ/DK/DV for the entire quartet / eight calls. The actual historical trap schedule remains unobserved. |
| Projection backward DX | The controlled unequal-bank trap produces exactly four predicted wrong DX words; equal-bank trap and both NOP controls preserve all-zero outputs across the complete original geometry. Corrected-driver repair and unique attribution of the spontaneous event remain open. |
| Historical int32 row-offset overflow | Independently proved source defect with int64 fixes already present. The CWSR correction cannot replace those fixes. |

These controlled results establish a cause of the reproduced NaNs and
corruption and prevention by the candidate in those tests. They do not prove
the unique cause of every historical NaN or identify every historical trap
event. Projection still awaits corrected-driver replay, and shared cause
cannot be inferred from extended-register use alone. See the
[A0 component evidence](mi450_a0.md#strongest-evidence) and the
[B0 LayerNorm limits](../mi450_b0/nan_triage_resume_20260921.md#resumed-debugging-compiled-layernorm-nan-mechanism).

The earlier 1,000-step A0 training pass used attention cap256, weighted-LN
BN1, workspace1 and private DW solution102 together. It did not use the
corrected handler and cannot show that one new component change is sufficient.
Finite loss alone is also insufficient: the retained one-bit-DY experiment
demonstrates finite arithmetic corruption from this same handler defect.

The resumed original-configuration default-policy E2E run completed all
1,000 steps with 68 scans and 42 operation endpoints per step and no alarm.
The complete saved frame passes CPU audit across 166 tensor views and
19,237,486,690 floating logical elements, with zero nonfinite or extreme
values. No cap256, LNBN1, workspace or private-solution workaround was used;
existing source fixes remain. Clean completion includes 1,000 explicit
optimizer updates, while the scheduled finite frame precedes update 1,000
and reflects 999 prior explicit dense updates.

The strict lifecycle audit remains **FAIL** solely for two corrected CPU
MC60 records. Separate review of the complete kernel delta finds exact
matches to a pre-AMDGPU-load event and no new GPU or uncorrected fault; it
does not identify the physical hardware cause or replace the strict verdict.
The trace has 186 evictions and 185 restores, including 166 NUMA protection
scans and 17 NUMA fault-driven migrations, with no losses or probe misses.
Processes exited, tracing was removed, and all capture bytes have a verified
lossless archive. The matched binding run stalled in the first backward's
shader-loading path and has no numerical verdict; balancing-enabled binding
remains unrun. This original-driver bounded numerical pass is
separate from corrected-driver validation and long-run stability.
[Complete E2E audit and kernel caveat](evidence/current_20260919/E2E_default1000_20260922_resume.json).

The loaded corrected handler and the sentinel/LN/attention/historical-DW
replays now pass. Practical E2E sufficiency remains open because neither
candidate first-backward stall produced a completed numerical verdict.

## Remaining stall components

The saved stack localizes the observed wait to registered-function
resolution and executable freezing, before that pending kernel is enqueued:

```text
FBGEMM registered host kernel -> hipLaunchKernel -> function resolution
  -> executable Freeze -> RegionMemory::Freeze
  -> GpuAgent::InvalidateCodeCaches -> AqlQueue::ExecutePM4 -> signal wait
```

This is an observation boundary, not proof of the physical stall cause.
The 21 matching HIP/HSA runtime frames and changed FBGEMM caller make the
following components the useful next targets:

| Component | Current evidence | Test that would narrow the cause |
|---|---|---|
| FBGEMM registration and deferred executable loading | The supported `linearize_index_kernel` attribute query completed, then `split_embedding_backward_codegen_find_long_segments` in the backward library reached the same Freeze wait. One queried kernel did not cover later code loading. | Query supported registered host keys across the relevant FBGEMM code objects before training; join the actual failing launch key, device and library binding to its prewarm receipt. |
| ROCr executable Freeze / code-cache PM4 completion | Retained ordinary host memory contains an eight-dword `ACQUIRE_MEM` IB, opcode `0x58`, flags `0x4381`, byte base `0x721c70f80000`, rounded span `0x6ae00` (437,760 bytes). Four captured regions are identical at two observations 160.522 seconds apart. | Establish whether prewarm itself waits before any training kernel, or whether the wait appears only after training begins. Current read/write indices, signal value and hardware residency were not observed. |
| Earlier asynchronous training execution | A later code-freeze wait can be the first host-visible wait after an earlier kernel stopped progressing. The pending FBGEMM kernel name alone does not identify that earlier work. | A separate first-backward run with `HIP_LAUNCH_BLOCKING=1` and complete launch entry/return records can locate the first nonreturning covered launch. Keep prewarm coverage identical to its asynchronous comparison. |
| AutoNUMA / KFD SVM invalidation and restore | The final targeted trace has 23 matched eviction/restore pairs: 21 NUMA and two fork pairs, with no recorded losses or probe misses. Two `svm_range_restore_work` workqueue-hog warnings appear during this run, without a new GPU fault/reset/Oops or MC60 in that interval. | Preserve target-process exposure and kernel continuity in the next comparison. These traces establish activity, not completed hardware saves or the cause of the stall. |
| DataLoader fork after HIP initialization | Eight `pt_data_worker` children were forked after the successful attribute query. Together with the training worker, nine processes retained GPU descriptors. The original-driver default E2E1000 also passed using this fork path. | A separately labeled `NUM_WORKERS=0` control can remove the forks while retaining the model and data order. Verify inputs; changed loader scheduling prevents a claim about the ordinary worker configuration. |

The earlier global eager-loading attempt is a separate initialization
failure: a registered ROCsolver/rocPRIM trampoline symbol was unresolved,
the worker aborted, and no training began. It is not evidence that eager
loading completed or fixed the wait. Keep `HIP_ENABLE_DEFERRED_LOADING`
unset in the proposed supported-host-key experiment.

The [targeted stall evidence](evidence/current_20260919/CWSR_candidate_AC2_E2E_targeted_stall_20260922.json)
pins the stack, packet, trace and preserved-process reviews. The
[packet review](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_targeted_host_packet_peer_v1/review.json)
also records that the preceding candidate's IB had the same opcode/control
flags but a different address and size. Only ordinary host mappings were
captured; I/O/PFN mappings, queue descriptor and signal storage were excluded.
No cache-state, current queue-index or completion-signal claim follows from
these retained bytes.

## Reproducer and review entry points

These are the exact retained packages behind the results and proposed next
tests. Their command files contain source/contract/receipt hash arguments;
their frozen boot IDs and used output paths are historical. They must be
derived into fresh reviewed packages before another GPU run.

| Purpose | Retained entry and required evidence |
|---|---|
| Corrected-driver health, 14 sentinels, LN20 and tiny attention, then full FAIL131 attention | [AC2 prerequisite package](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_prerequisites_v1/README.md), [command arrays](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_prerequisites_v1/COMMANDS.json). `run_health.py` and `run_downstream.py` bind the frozen producers; the component consumer, complete attention raw audit and verified archive are required. |
| Complete historical step405 DW quartet | [DW package](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_DW_v1/README.md), [command arrays](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_DW_v1/COMMANDS.json). `supervise.py --execute-quartet` runs four arms / 16 calls; `audit_completed.py` without `--arm` supplies the complete result. Supervisor success alone is explicitly pending that consumer. |
| Preserved single-kernel-preload E2E stall | [Targeted E2E package](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_targeted_E2E_v1/README.md), [command arrays](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_targeted_E2E_v1/COMMANDS.json). Seed 1, batch 1024, default NUMA policy, original asynchronous configuration and 1,000-step bound; actual completion was zero steps. |
| Natural zero-DY DW on the corrected driver | [Reviewed source preparation v3](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_natural_DW_package_v3/READY.json), [next-boot plan](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_natural_DW_package_v3/NEXT_BOOT_PLAN.md). No runtime staging or execution occurred. The old contract still requires a successful targeted-E2E predecessor, which does not exist; the new contract must instead follow completed historical DW. |
| Broader FBGEMM prewarm | [Helper and inventory description](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_fbgemm_prewarm_full_v1/README.md), [worker integration](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_fbgemm_prewarm_full_v1/INTEGRATION.md). Stage/pin `prewarm_fbgemm.py`, `prewarm_support.py` and selected `prewarm_manifest_v2.json` together; call after the actual same-worker pre-HIP gate release and before diagnostic training entry. |
| Separate serialization or no-fork control | [Installed-HIP serialization analysis](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_serialization_semantics_v1/RESULT.md), [independent ELF review](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_serialization_semantics_peer_v1/review.json), [DataLoader fork evidence](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_root_v1/TARGETED_STALL_FORK_CONTROL.md). Neither diagnostic has been launched. |

The broader prewarm CPU inventory covers 18 host libraries, 227 fatbinary
wrappers and 43,450 function registrations. The selected manifest uses 226
supported host keys and explicitly accounts for one shadowed module; 2,410
registered names without gfx1250 kernel descriptors are excluded. A query
uses the registered host OBJECT address, not the stub pointer stored there.
The author CPU audit checked all representatives, 2,912 saved mapping ranges,
2,034 instruction sequences and six success/failure fixtures. The subsequent
[root runtime-helper review](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_root_v1/fbgemm_helper_root_review.json)
passed; separate registration/constructor/duplicate-handling peer review
remains pending. No GPU helper run has occurred. A future helper PASS means
226 verified query receipts, not proof that 227 distinct Programs were
loaded, that a particular hardware cache state exists, or that E2E passed.

Installed HIP binary analysis and its independent review establish that
`HIP_LAUNCH_BLOCKING=1` contributes the same kernel-command after-enqueue
wait bit as `AMD_SERIALIZE_KERNEL=2`. Setting both adds no further such wait.
The wait finishes that command's host queue, not all streams, host threads
and copies. Function resolution and code freezing occur before the enqueue,
so this setting does not bypass or serialize Freeze itself. A diagnostic
must record the actual launch route, host thread, stream, function identity
and entry/return timestamps before interpreting the first missing return.

## Next steps

1. Complete the pending CPU registration/constructor/duplicate-handling
   reviews and derive the next experiment contracts. Preserve the current
   worker, evidence and blocked boot; no further GPU work, live reload,
   reset or `hqds`/SDMA debug read is admissible there. Recovery requires a
   new host AC cycle and verified candidate first load.
2. On that fresh boot, complete health and all 14 sentinels, LN/tiny-attention
   with complete consumers, full historical attention with verified archive,
   then the historical DW quartet and complete consumer. Use new paths and
   actual same-boot source, driver and prerequisite hashes.
3. Run natural zero-DY `sink_control` before E2E, using the unchanged
   vulnerable ELF and original inputs, for 1,000 calls or the first completed
   numerical failure. Either signed zero passes; every nonzero or nonfinite
   value fails. Require native/input/output,
   lifecycle and exposure consumers plus lossless archival of a failure.
   A clean run without a qualifying target AutoNUMA/SVM pair is
   exposure-inconclusive; even a qualifying pair is not a measured CWSR event.
4. Run the fully reviewed FBGEMM prewarm with normal asynchronous E2E and
   original loader settings. Do not also change serialization, worker count
   or NUMA policy. If prewarm or backward stalls, preserve the first pending
   phase/launch and saved stack; another GPU comparison needs another
   recovered boot.
5. Choose the separate serialized first-backward/update diagnostic or
   `NUM_WORKERS=0` control from that evidence, keeping other factors matched.
   A completed diagnostic is attributable only to its tested configuration.
   Corrected-driver projection replay remains an independent coverage gap.
6. After a complete candidate E2E numerical and lifecycle pass, test the
   intended duration, independent restarts/seeds, raw gradients, parameters
   and expected loss/accuracy. Remove individual workarounds only through
   separate comparisons; retain the independently required int64 source
   fixes. A bounded pass does not prove that every NaN cause is eliminated.

Full chronology and current follow-ups are in [mi450_a0.md](mi450_a0.md).
