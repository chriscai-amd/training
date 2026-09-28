# Prior CWSR platform and experiment narrative

Historical snapshot preserved during the September 28 refresh.
Use [the maintained focused record](../../mi450_a0/cwsr_bank_corruption_fix.md) for current status.

# MI450 gfx1250 CWSR register-bank corruption

Updated 2026-09-23 UTC. **Investigation paused at the user's request.** No GPU
workload is running; the 06:48 UTC all-task device-descriptor scan was idle.
Witness v2 and its training integration are source-reviewed, with no actual
binding, staging or GPU result. The
[pause checkpoint](../../mi450_a0/evidence/current_20260923/session_pause.json) and
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
[Complete platform transition evidence](../../mi450_a0/evidence/current_20260923/platform_transition.json).
The follow-up native/source audit confirms preserved KFD user-queue retry
programming and byte-identical selected MES add-queue code. Removed optional
GL2 writes remain a conditional confound: invocation during the stall is
unproved, and initialization instrumentation only read the registers.
[Detailed GL2/retry/PM4 comparison and independent review](../../mi450_a0/evidence/current_20260923/GL2_retry_stall_source.json).
[Current first load and firmware receipt](../../mi450_a0/evidence/current_20260923/current_first_load.json).
[Current health and register sentinels](../../mi450_a0/evidence/current_20260923/current_health_sentinels.json).
[Independent retained-health review and MC60 timing](../../mi450_a0/evidence/current_20260923/current_health_peer.json).

The current stack also passed the LN/tiny-attention matrix: 24 processes,
36 dispatches, all 856 raw files and 1,024 native pins verified by the
complete consumer. Full historical attention also passed four arms/eight
calls, with all 144 raw hashes equal to the prior corrected run and an
independently verified lossless archive of all 13,053,235,840 bytes.
[Current component result](../../mi450_a0/evidence/current_20260923/current_components.json).
[Current historical attention and archive](../../mi450_a0/evidence/current_20260923/current_full_attention.json).

Historical step-405 DW then passed on this current stack: all 16 full outputs
equal the ordinary baseline. The complete consumer rehashed 14,280,812,544
input bytes; the verified archive reconstructs every one of the 44 raw paths,
with raw retained. Two known corrected CPU8/MC60 records occurred after
attention, one during the final DW producer, under the existing diagnostic
exception. Three SVM restore-work CPU-hog warnings accompanied successful
completion; those warnings alone do not identify the earlier training hang.
[Current complete DW and archive](../../mi450_a0/evidence/current_20260923/current_historical_DW.json).

The first current natural-DW attempt stopped before HIP initialization or
worker release because its trace checker rejected `1*` for the enabled
queue-eviction event with a stacktrace trigger. No numerical call ran.
The worker expired normally; worker and launcher are absent, global tracing
was restored, and final idle/kernel checks passed at 01:01 UTC. The frozen
failed attempt is retained in the fresh reviewed contract's predecessor chain.
[Pre-HIP failure and cleanup](../../mi450_a0/evidence/current_20260923/natural_DW_preHIP_abort.json).

Natural DW v2 then completed all 1,000 calls on the unchanged vulnerable
`sink_control` kernel, with all 131,072,000 BF16 words positive zero. Complete
and independent repeat audits verified all 172 packet bytes per call,
output/prefill checks and 3,570,465,280 input bytes. The lossless trace has
1,087 records and 18 relevant AutoNUMA/SVM eviction–restore pairs in 18 call
envelopes, without loss or probe misses. Worker exit, final idle and unchanged
kernel logs passed at 01:11:58 UTC. Those windows establish relevant exposure,
not physical CWSR acknowledgement, an interrupted PC or bank state.
[Current natural-DW complete result](../../mi450_a0/evidence/current_20260923/current_natural_DW.json).
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
[Completed distinct adjudication](../../mi450_a0/evidence/current_20260923/current_E2E_TLS_adjudication.json).
[Verified archive and numerical comparison](../../mi450_a0/evidence/current_20260923/current_E2E_archive_and_comparison.json).
Two known corrected CPU8/MC60 records matched the frozen diagnostic policy
and training continued. Numerical progress does not certify platform health.
[Current E2E numerical pass and preserved strict trace failure](../../mi450_a0/evidence/current_20260923/current_E2E_numerical_and_trace_gate.json).

The current corrected-driver projection quartet also passed all four arms
and 16 calls, including the unequal-bank trap case. Every output BF16 word
is positive zero; the four previously corrupted positions are corrected.
The complete consumer checked all 212 packet bytes per call, 32.11 GB of
outputs and 32.11 GB of inputs, with verified lossless archives and complete
trace/lifecycle checks. All four saved independent reviews passed. The
whole-worker queue-helper counts do not identify physical CWSR during the
four GEMMs. Loaded driver and MES `0x7b` were unchanged.
[Current projection quartet and exact receipts](../../mi450_a0/evidence/current_20260923/current_projection.json).

A fresh process without the prewarm helper then passed two backward/optimizer
steps on the same current stack. The complete scalar-journal, trace, process
exit and final idle/kernel checks passed, with no anomaly and empty kernel
deltas. The journal checks 367 finite summaries and 12 zero comparisons per
step; no full endpoint tensor audit was scheduled. Only tiny finite
timestamp-embedding gradient-extrema differences appear against the first
two full-prewarm steps. This weakens the claim that explicit prewarm is
required for every fresh process on this boot, but earlier prewarm and other
platform changes still prevent cold-boot or MES-only attribution.
[No-prewarm two-step result and independent review](../../mi450_a0/evidence/current_20260923/current_no_prewarm_two_steps.json).

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
[Completed no-prewarm 1,000-step audit and comparison](../../mi450_a0/evidence/current_20260923/current_no_prewarm_1000_steps.json).

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
[Conditional bounds and independent source/observation audit](../../mi450_a0/evidence/current_20260923/current_STU1_DX583.json).

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
[Exact source, reviews, bootstrap and prospective decision table](../../mi450_a0/evidence/current_20260923/current_STU1_witness_preparation.json).

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
[Complete natural nonzero-DY result](../../mi450_a0/evidence/current_20260923/current_natural_nonzero_DW.json).
[Current plateau400 result and exposure limits](../../mi450_a0/evidence/current_20260923/current_plateau400.json).

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
[Exact retirement scope, retained evidence and allocation receipt](../../mi450_a0/evidence/current_20260923/storage_cleanup.json).

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
[Actual AC2 results, receipts and scope](../../mi450_a0/evidence/current_20260919/CWSR_candidate_AC2_validated_20260922.json).

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
[Preserved eager-startup failure and clean cleanup audit](../../mi450_a0/evidence/current_20260919/CWSR_candidate_AC2_E2E_eager_abort_20260922.json).

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
[Targeted-preload stall evidence](../../mi450_a0/evidence/current_20260919/CWSR_candidate_AC2_E2E_targeted_stall_20260922.json).

The earlier strict post-load MC60 stop is preserved as historical evidence.
The diagnostic continuation uses the already authorized narrow policy for
complete known CPU8/MC60 corrected-error records; arbitrary faults still
reject. One matching record preceded AC2 activation, and the completed
workload intervals contained no new MC60 reports. The numerical PASS results
do not certify platform health. The `VM_PAGE_FAULT` insertion inventory label
was separately adjudicated using source and all 309 insertion lines.
[Earlier activation and strict validation stop](../../mi450_a0/evidence/current_20260919/CWSR_post_ac_candidate_activation_20260922.json),
[original frozen preparation](../../mi450_a0/evidence/current_20260919/CWSR_candidate_validation_preparation_20260922.json).

A KFD `hqds` read on the older original-driver boot caused a separate SDMA
debug-dump NULL dereference. The corrected CWSR candidate retains that
diagnostic function; it must not be invoked. No live reload or GPU recovery
was used for AC2 validation.
[Historical stall and diagnostic-fault record](../../mi450_a0/evidence/current_20260919/E2E_bind_stall_and_HQD_oops_20260922.json).

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
