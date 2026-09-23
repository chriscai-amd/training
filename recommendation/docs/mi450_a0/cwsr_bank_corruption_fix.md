# MI450 gfx1250 CWSR register-bank corruption

Updated 2026-09-23 UTC. **Investigation paused at the user's request.** No GPU
workload is running; the 06:48 UTC all-task device-descriptor scan was idle.
Witness v2 and its training integration are source-reviewed, with no actual
binding, staging or GPU result. The
[pause checkpoint](evidence/current_20260923/session_pause.json) and
[handoff](/home/chcai/handoff/nan_handoff.md) supersede earlier active-work instructions.

The pause's saved kernel delta contains one complete corrected CPU8/MC60
record and retained rsyslogd/AppArmor and suppression messages. Independent
replay passes the unchanged diagnostic policy; service-actor attribution and
platform-health clearance do not follow. Fresh live gates remain required.

The new boot `b685ba14-e2ad-4751-b755-2e7bba3b58fa` initially had AMDGPU absent. A package
reinstall changed the installed module to srcversion
`EADFD8EC13CF3E6B329194E`, restored the exact faulty 5,656-byte handler, removed
custom GL2 controls and changed retry-bit initialization. Disk and initramfs
unified MES are `0x7b`, not the reported testing `0x17d` or intended official
`0x7e`; prior run-specific MES identities remain unverified.

A fresh candidate against this current source has passed independent CPU
byte/symbol/relocation audit: srcversion `AD83C153B9701570F12964E`, SHA256
`78de19854e38e298f6418bddcfa39f44a797f811f0f4b9c585482f4c4dc2b2b4`.
The subsequent guarded first load completed at 00:23:14 UTC, with both loaded
MES attributes reporting `0x0000007b`; driver/CWSR, kernel-delta and idle
checks passed. Health/sentinels passed at 00:28 UTC: 4,000 exact checks,
14 preserved tag/MODE/EXEC cases and seven exact return PCs, with both
producers exited and all devices idle. One known corrected MC60 record
preceded health launch; none occurred during the producers or final checks.
Tests on `0x7b` are labeled with that version; no evidence establishes `0x7b` as defective or `0x7e`
as installed. The MES report is relevant to queue hangs and does not undo
the deterministic CWSR proof. Firmware and the other source changes must
remain separate variables in any causal comparison.
[Complete platform transition evidence](evidence/current_20260923/platform_transition.json).
The follow-up native/source audit confirms preserved KFD user-queue retry
programming and byte-identical selected MES add-queue code. Removed optional
GL2 writes remain a conditional confound: invocation during the stall is
unproved, and initialization instrumentation only read the registers.
[Detailed GL2/retry/PM4 comparison and independent review](evidence/current_20260923/GL2_retry_stall_source.json).
[Current first load and firmware receipt](evidence/current_20260923/current_first_load.json).
[Current health and register sentinels](evidence/current_20260923/current_health_sentinels.json).
[Independent retained-health review and MC60 timing](evidence/current_20260923/current_health_peer.json).

The current stack also passed the LN/tiny-attention matrix: 24 processes,
36 dispatches, all 856 raw files and 1,024 native pins verified by the
complete consumer. Full historical attention also passed four arms/eight
calls, with all 144 raw hashes equal to the prior corrected run and an
independently verified lossless archive of all 13,053,235,840 bytes.
[Current component result](evidence/current_20260923/current_components.json).
[Current historical attention and archive](evidence/current_20260923/current_full_attention.json).

Historical step-405 DW then passed on this current stack: all 16 full outputs
equal the ordinary baseline. The complete consumer rehashed 14,280,812,544
input bytes; the verified archive reconstructs every one of the 44 raw paths,
with raw retained. Two known corrected CPU8/MC60 records occurred after
attention, one during the final DW producer, under the existing diagnostic
exception. Three SVM restore-work CPU-hog warnings accompanied successful
completion; those warnings alone do not identify the earlier training hang.
[Current complete DW and archive](evidence/current_20260923/current_historical_DW.json).

The first current natural-DW attempt stopped before HIP initialization or
worker release because its trace checker rejected `1*` for the enabled
queue-eviction event with a stacktrace trigger. No numerical call ran.
The worker expired normally; worker and launcher are absent, global tracing
was restored, and final idle/kernel checks passed at 01:01 UTC. The frozen
failed attempt is retained in the fresh reviewed contract's predecessor chain.
[Pre-HIP failure and cleanup](evidence/current_20260923/natural_DW_preHIP_abort.json).

Natural DW v2 then completed all 1,000 calls on the unchanged vulnerable
`sink_control` kernel, with all 131,072,000 BF16 words positive zero. Complete
and independent repeat audits verified all 172 packet bytes per call,
output/prefill checks and 3,570,465,280 input bytes. The lossless trace has
1,087 records and 18 relevant AutoNUMA/SVM eviction–restore pairs in 18 call
envelopes, without loss or probe misses. Worker exit, final idle and unchanged
kernel logs passed at 01:11:58 UTC. Those windows establish relevant exposure,
not physical CWSR acknowledgement, an interrupted PC or bank state.
[Current natural-DW complete result](evidence/current_20260923/current_natural_DW.json).
The full-FBGEMM-prewarm E2E experiment completed all 1,000 steps and exited.
The complete 38.56 GB payload audit passed all 166 tensor views, with zero
nonfinite or extreme values. Strict trace-controller failure is retained:
seven new disabled TLS event entries changed its global-inventory comparison,
while existing controls and the complete private trace remained intact.
The separate exact TLS-inventory adjudication passed at 02:30:31 UTC with
fresh idle/driver/firmware/kernel gates and unchanged source authorities.
It preserves both strict failure receipts and `global_state_unchanged=false`.
Durable archival and independent complete decompression passed at 02:44:45
UTC, preserving all 38,556,636,485 raw bytes in a 19,557,008,930-byte archive.
The original/current comparison has matching settings, RNG and scan geometry
but finite scalar/loss/endpoint differences; the first scalar difference is
about `1.16e-10` in the step-1 timestamp-embedding gradient extrema.
[Completed distinct adjudication](evidence/current_20260923/current_E2E_TLS_adjudication.json).
[Verified archive and numerical comparison](evidence/current_20260923/current_E2E_archive_and_comparison.json).
Two known corrected CPU8/MC60 records matched the frozen diagnostic policy
and training continued. Numerical progress does not certify platform health.
[Current E2E numerical pass and preserved strict trace failure](evidence/current_20260923/current_E2E_numerical_and_trace_gate.json).

The current corrected-driver projection quartet also passed all four arms
and 16 calls, including the unequal-bank trap case. Every output BF16 word
is positive zero; the four previously corrupted positions are corrected.
The complete consumer checked all 212 packet bytes per call, 32.11 GB of
outputs and 32.11 GB of inputs, with verified lossless archives and complete
trace/lifecycle checks. All four saved independent reviews passed. The
whole-worker queue-helper counts do not identify physical CWSR during the
four GEMMs. Loaded driver and MES `0x7b` were unchanged.
[Current projection quartet and exact receipts](evidence/current_20260923/current_projection.json).

A fresh process without the prewarm helper then passed two backward/optimizer
steps on the same current stack. The complete scalar-journal, trace, process
exit and final idle/kernel checks passed, with no anomaly and empty kernel
deltas. The journal checks 367 finite summaries and 12 zero comparisons per
step; no full endpoint tensor audit was scheduled. Only tiny finite
timestamp-embedding gradient-extrema differences appear against the first
two full-prewarm steps. This weakens the claim that explicit prewarm is
required for every fresh process on this boot, but earlier prewarm and other
platform changes still prevent cold-boot or MES-only attribution.
[No-prewarm two-step result and independent review](evidence/current_20260923/current_no_prewarm_two_steps.json).

The subsequent no-prewarm run completed **1,000 backward passes and optimizer
updates**. Its original complete consumer passed at **05:23:09 UTC**, followed
by independent saved review of 68,000 scans, 42,000 endpoints, 367,000 finite
summaries and 12,000 zero comparisons. There was no journal alarm or anomaly
dump; no full endpoint tensor audit was scheduled. The complete private trace
and process/idle/driver/MES continuity checks passed. The training interval
contains three corrected MC60 records admitted by the existing diagnostic
policy, 173 rsyslogd AppArmor denials and 14 shared-printk suppression notices.
One further MC60 record predates the run. The suppression prefix does not
recover omitted messages or prove all were audit records; private ftrace
integrity passed its separate complete audit. No platform-health clearance.

All 379,000 full-prewarm/no-prewarm scalar-summary pairs were compared with
matching normalized geometry. There are 166,932 different summaries across
998 steps. The first timestamp-gradient difference is at most `8.73e-11`,
but the step-583 content-MLP second-Linear bias gradient maximum differs by
**66.476806640625**: 66.5 without prewarm versus 0.023193359375 with prewarm.
Finite gradient divergence remains under investigation. Summary equality
does not establish tensor equality, and same-boot history limits attribution.
[Completed no-prewarm 1,000-step audit and comparison](evidence/current_20260923/current_no_prewarm_1000_steps.json).

The follow-up complete bound analysis found **one conditional inconsistency
among 18,000 projection DX/DW/DB checks** across both 1,000-step runs. At
no-prewarm step 583, STU1 `DX = torch.mm(DZ, W.T)` has recorded max magnitude
0.0458984375 versus the recorded-input triangle bound 2.1943822503089905e-5,
a factor of 2,091.63. The next weighted-LN input repeats the DX extrema.
The independently pinned source has no scale or addend in that matrix product.
STU1 only has deferred reductions of its original tensors; the extra complete
pre/post snapshots belong to STU2. Scalar fidelity, consumed-operand continuity,
native execution and unordered writes therefore remain unresolved. Ordinary
roundoff under the stated FP32/BF16 model cannot close this gap, and later
LayerNorm sensitivity alone cannot explain it. The original absolute alarm
threshold was 1e6, so this finite case produced no raw failure capture.
The next training diagnostic has a targeted STU1 tensor and scalar witness
prepared, with GPU execution deferred until the user resumes.
[Conditional bounds and independent source/observation audit](evidence/current_20260923/current_STU1_DX583.json).

Witness v2 retains original W/DZ objects before the public call and original
DX after it. It adds no GPU operation, clone or boundary readback before a
trigger; it does prolong allocations and change allocator history. At the
existing post-backward flush, a conditional GEMM-bound trigger saves actual
scalar endpoints, the stacked batch and original reader rows, plus two
sequential compact CPU observations of complete W and one selected DZ/DX
row. An exact integer BF16 dot-product check tests the selected coordinate.
These are post-backward observations and cannot prove the bytes consumed by
the original GEMM. The trigger stops before explicit clipping/optimizer;
fused sparse updates can already occur within backward. The original 68
scans, 367 summaries, STU2 full capture and generic alarms are retained.

The frozen v2 runtime differs from v1 only in its 16 → 24 GiB reference cap.
Saved steps 720 and 721 require 17,316,032,512 and 19,720,481,792 bytes of
compact references, respectively. Both small output files remain capped at
16 MiB. Nineteen author CPU methods and 16 independent cap cases passed;
17 root and 26 independent training integration cases passed, with eight
additional controller/auditor cases covering preserved failure handling.
The unchanged small consumer also passed actual complete and scalar-only
bootstrap runs with generated artifacts in containers without GPU devices.
Two earlier CPU-autograd tests in a GPU-visible container opened device
descriptors despite a false CUDA-initialized flag; their failed isolation
receipts remain. They were reaped before the successful isolated reviews.
[Exact source, reviews, bootstrap and prospective decision table](evidence/current_20260923/current_STU1_witness_preparation.json).

The saved step-583 timing analysis found no nominal helper-span or complete
MC60 overlap; the closest preceding restore ended 1.143 s before scan30
enqueue. It uses an endpoint clock-hull assumption without a local clock
bracket. Host enqueue time is not GPU execution time, so CWSR involvement
remains unresolved. Matching operand extrema do not establish tensor equality.
The following LN bias extrema equal four times DX extrema after BF16 rounding;
that scalar pattern does not establish four corrupt rows or a bank route.

The original **400 NOP-only long-plateau probes** also passed. All tags remain
unchanged in every packet. Ten fresh in-plateau return PCs have matching
target AutoNUMA/SVM eviction–restore pairs, including mode `0x46` at ordinals
75 and 239. This adds natural preservation evidence for `v513 → v257` under
the corrected handler. The complete consumer audited 1,200 raw images,
2,410 HIP calls, private trace and saved lifecycle; independent saved-result
review reproduced the complete result. Zero explicit traps were requested. These observations
do not reconstruct original step101/405 events or supply physical CWSR
acknowledgments. Natural historical nonzero-DY DW subsequently passed all
1,000 calls with exact full-baseline equality, including first-call qualification.
Independent review verified every output/prefill word, full input bytes and
172 native argument bytes per call. Fourteen qualifying SVM pairs overlap
14 call envelopes, with no private trace loss.
[Complete natural nonzero-DY result](evidence/current_20260923/current_natural_nonzero_DW.json).
[Current plateau400 result and exposure limits](evidence/current_20260923/current_plateau400.json).

Storage cleanup completed at **04:06:36 UTC**, retiring exactly 18 obsolete
successful/requested snapshots and reclaiming **530,390,228,992 allocated
durable bytes**. The group contains eight finite boundary captures, one
requested healthy attention capture, two healthy forced-replay base storages,
six older finite diagnostic frames and the September 22 finite E2E archive.
Their reports, manifests, source/native records and historical failure
verdicts remain. The original retired snapshot bytes have no replacement
archive, so direct raw reruns and full rescans of those snapshots are no
longer available. The two base storages had different full hashes; this was
intentional retirement rather than duplicate-file consolidation. Both base
capture directories now carry verified `RETIRED.json` and `README_RETIRED.md`
notices while preserving their original compact metadata and reports.
All unique numerical failure evidence, current datasets, the long-step101
base and the current September 23 E2E raw capture/archive remain protected.
All 336 then-pending no-prewarm source pins were verified and contain no selected
capture reference. The full `/home/chcai` measurement at **06:48:06 UTC** is
**992,334,336,000 bytes**, below decimal 1 TB by **7,665,664,000 bytes**.
Cumulative reclamation is **897,569,296,384 durable allocated bytes** plus
**77,197,692,928 tmpfs bytes**.
[Exact retirement scope, retained evidence and allocation receipt](evidence/current_20260923/storage_cleanup.json).

The following completed results describe September 22 AC2. The corrected CWSR
candidate was first-loaded on fresh boot
`6478859b-3092-4792-a933-e91c311f2cd1`, with version `7.1.1.31300009`,
srcversion `8336BBB74E6C4D53275A4E5` and CWSR enabled. Actual GPU tests now
show that it prevents the controlled register-bank corruption, LayerNorm
NaNs, tiny and full historical attention corruption, and the complete
historical step-405 DW corruption. Native kernels, inputs and launch
parameters match the original-driver comparisons.

| Corrected-driver test on AC2 | Completed result |
|---|---|
| Health and sentinels | 4,000 exact health checks; all 14 tag/MODE/EXEC cases preserved and all seven explicit trap return PCs exact |
| LayerNorm and tiny attention | All 24 processes / 36 dispatches pass; all 3,586 reproduced LN NaNs and all 512 tiny-attention wrong DQ words absent |
| Full historical attention FAIL131 | All four arms / eight dispatches pass; all 1,087,635,456 DQ/DK/DV words positive zero |
| Historical step-405 DW | All four arms / 16 calls equal the complete ordinary baseline; all 1,030 historical differences corrected, including 640 NaNs |

Complete raw/native consumers and independent comparison reviews passed.
All producers exited and final all-device idle gates passed. The full
attention capture also has a verified lossless archive, with raw retained.
[Actual AC2 results, receipts and scope](evidence/current_20260919/CWSR_candidate_AC2_validated_20260922.json).

**Earlier corrected-driver runs did not complete training.** On the preceding candidate
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
were preserved; the final idle gate failed and the private trace was
stopped. That boot required recovery through a new host AC cycle. Its PIDs
are historical. No driver reset, reload or recovery diagnostic was attempted.
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
The original comparison module was `7.1.1.31300009`, srcversion
`654C1DDE7A9A3E129BA9553`; its authenticated scratch baseline is retained.
The September 22 reinstall replaced the installed file as described above.

**This is a confirmed defect, not a demonstrated sole remaining E2E cause.**
The earlier candidate E2E attempt stalled before a numerical verdict. The
September 23 corrected-driver full-prewarm run completed 1,000 numerical
steps, followed by successful two-step and 1,000-step no-prewarm runs. Changes to driver source, boot and
prewarm history prevent assigning the new training completion to the CWSR
patch alone. Existing independent source fixes remain necessary.

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
equalizing the banks at the injected trap. The September 23 corrected-driver
quartet also passes all four arms / 16 calls, correcting the four predicted
positions and returning positive zero in every output word. The spontaneous
event's original trap history remains unobserved.
[Projection reproduction and controls](evidence/current_20260919/CWSR_projection_controlled_20260922_resume.json).
[Current corrected-driver projection and complete archive/lifecycle audit](evidence/current_20260923/current_projection.json).

The September 23 corrected-driver projection TLS v3 attempt stopped before
trace setup or worker creation because its uppercase `TLS` trace name failed
the unchanged controller's lowercase-only validation. No numerical call ran.
The failure remains frozen; a separate postfailure gate confirms idle devices,
unchanged driver/MES, accepted kernel continuity and absent attempt resources.
Fresh v4 source preserves the native kernels and geometry, corrects generated
tags, and requires every actual staged plan to pass controller admission.
[Preserved preworker failure and completed postfailure gate](evidence/current_20260923/projection_v3_preworker_abort.json).

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
| Projection backward DX | The controlled unequal-bank trap produces exactly four predicted wrong DX words; equal-bank trap and both NOP controls preserve all-zero outputs across the complete original geometry. The corrected-driver quartet passes all 16 calls with every output word positive zero, including the four formerly corrupted positions. The spontaneous event's original trap history remains unobserved. |
| Historical int32 row-offset overflow | Independently proved source defect with int64 fixes already present. The CWSR correction cannot replace those fixes. |

These controlled results establish a cause of the reproduced NaNs and
corruption and prevention by the candidate in those tests. They do not prove
the unique cause of every historical NaN or identify every historical trap
event. Shared cause cannot be inferred from extended-register use alone. See the
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
Processes exited, tracing was removed, and all capture bytes were losslessly
archived and verified at completion. This old finite archive was retired in
the September 23 cleanup; its complete numerical audit and strict lifecycle
failure evidence remain, but the original raw snapshot cannot now be rescanned.
The matched binding run stalled in the first backward's
shader-loading path and has no numerical verdict; balancing-enabled binding
remains unrun. This original-driver bounded numerical pass is
separate from corrected-driver validation and long-run stability.
[Complete E2E audit and kernel caveat](evidence/current_20260919/E2E_default1000_20260922_resume.json).

The loaded corrected handler and the sentinel/LN/attention/historical-DW
replays pass. The September 23 full-prewarm E2E numerical pass adds bounded
training coverage; changed platform source, boot and preload prevent crediting
the CWSR correction alone with resolving the earlier first-backward stalls.

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
| MES firmware and changed driver initialization | The reported testing `0x17d` lacks a `remove_queue` hang fix; current disk/initramfs contain `0x7b`. Old logs contain `REMOVE_QUEUE` failures, but the latest AC2 stall lacks a direct MES failure record. The reinstall also changed GL2/retry source. | Pin loaded MES and driver bytes before every test. Compare the same source/CWSR/workload under verified firmware versions on separate recovered boots; a mixed source/firmware transition cannot isolate MES. |
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
| STU1 step-583 bound finding and next diagnostic | [Complete saved-bound analysis](evidence/current_20260923/current_STU1_DX583.json), [frozen witness v2 instructions](/home/chcai/mi450_logs/root_cause_20260919/session_20260923_STU1_postback_witness_source_v2/INTEGRATION.md), [prepared request/reviews](evidence/current_20260923/current_STU1_witness_preparation.json), [resume argv](/home/chcai/mi450_logs/root_cause_20260919/session_20260923_pause_wrap_v1/RESUME_COMMANDS.json). Historical STU1 raw was not retained. The prospective 1,000-step witness has no runtime or result and still requires actual binding/release and live gates. |
| Current natural nonzero-DY DW | [Completed source, consumer, peer and archive](evidence/current_20260923/current_natural_nonzero_DW.json). All 1,000 outputs match baseline; 14 qualifying SVM pairs. Five raw files were fully round-trip verified, archived and released. Restore and hash before raw replay; use fresh reviewed runtime paths. |
| Corrected-driver health, 14 sentinels, LN20 and tiny attention, then full FAIL131 attention | [AC2 prerequisite package](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_prerequisites_v1/README.md), [command arrays](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_prerequisites_v1/COMMANDS.json). `run_health.py` and `run_downstream.py` bind the frozen producers; the component consumer, complete attention raw audit and verified archive are required. |
| Complete historical step405 DW quartet | [DW package](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_DW_v1/README.md), [command arrays](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_DW_v1/COMMANDS.json). `supervise.py --execute-quartet` runs four arms / 16 calls; `audit_completed.py` without `--arm` supplies the complete result. Supervisor success alone is explicitly pending that consumer. |
| Preserved single-kernel-preload E2E stall | [Targeted E2E package](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_targeted_E2E_v1/README.md), [command arrays](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_targeted_E2E_v1/COMMANDS.json). Seed 1, batch 1024, default NUMA policy, original asynchronous configuration and 1,000-step bound; actual completion was zero steps. |
| Natural zero-DY DW on the corrected driver | [Current completed result](evidence/current_20260923/current_natural_DW.json): unchanged vulnerable kernel PASS1000 with 18 relevant SVM pairs. Fresh v2 follows completed current historical DW and clean pre-HIP abort cleanup; the old v3 package remains historical. |
| Broader FBGEMM prewarm | [Helper and inventory description](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_fbgemm_prewarm_full_v1/README.md), [worker integration](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_fbgemm_prewarm_full_v1/INTEGRATION.md). The September 23 worker passed all 226 selected queries and completed 1,000 numerical steps; full tensor audit, exact TLS-inventory adjudication and verified archive passed. Original strict trace failure is retained. |
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
also passed on September 23 across all 18 libraries and 43,450 registrations
([independent review](evidence/current_20260923/fbgemm_registration_review.json)).
The September 23 GPU helper run passed all 226 selected queries in the actual
training worker before the completed asynchronous E2E1000 run. The full
tensor audit, distinct TLS-inventory adjudication and archive verification
also passed; the original strict trace failure remains. Query success does
not prove 227 distinct Programs loaded or a particular hardware cache state.

Installed HIP binary analysis and its independent review establish that
`HIP_LAUNCH_BLOCKING=1` contributes the same kernel-command after-enqueue
wait bit as `AMD_SERIALIZE_KERNEL=2`. Setting both adds no further such wait.
The wait finishes that command's host queue, not all streams, host threads
and copies. Function resolution and code freezing occur before the enqueue,
so this setting does not bypass or serialize Freeze itself. A diagnostic
must record the actual launch route, host thread, stream, function identity
and entry/return timestamps before interpreting the first missing return.

## Next steps after the user resumes

1. Recheck boot, loaded driver, firmware, kernel continuity, device ownership
   and home allocation. The pause checkpoint is observation only, and later
   kernel messages are preserved separately. Any changed boot or low-level
   stack requires rebased reviewed prerequisites. Never read KFD `hqds` or
   `amdgpu_gpu_recover`, reset/reload a stalled boot, or reuse used output roots.
2. Use `REQUEST_PREPARED.json` in the STU1 training source package. Its source
   request passed validation and pure construction with 890 source/metadata
   pins, 125 runtime pins and 12 planned stage files. Actual predecessor
   admission, independent binding review, finalization, staging and fresh live
   gates remain undone. The saved resume argv stops at preparation; subsequent
   commands must be generated from the actual reviewed binding.
3. Run the reviewed 1,000-step no-prewarm STU1 witness diagnostic. Analyze any
   trigger using its actual scalar-stage and selected-row findings; preserve
   failures and incomplete artifacts. A clean run is bounded journal evidence
   and does not explain the original step-583 event. Full tensor/native
   attribution remains separate from post-backward observations.
4. Preserve the completed full-prewarm E2E tensor audit, no-prewarm two-step
   and 1,000-step journals, plateau400 and natural zero/nonzero-DY results.
   Restore and hash archived payloads when a future raw audit needs them.
   Keep the original strict TLS failure and all corrected MC60 records.
5. Pursue independent restarts/seeds, matched boot/source comparisons,
   official MES `0x7e` after an actual artifact is supplied, and B0 validation
   as separate experiments. Retain required int64/source fixes and isolate
   workaround-removal comparisons. A bounded pass does not establish that
   every NaN cause is eliminated.

Full chronology and current follow-ups are in [mi450_a0.md](mi450_a0.md).
