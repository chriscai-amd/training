# Prior A0 status narrative

Historical snapshot preserved during the September 28 post-reboot refresh.
Statements saying current, pending or next refer to their original checkpoints.
Use [the maintained A0 status](../../mi450_a0/mi450_a0.md#current-status) for the latest result.

## Current NaN status

<a id="current-status"></a>

**Latest cross-host review, September 28:** the repository was pulled to
`8cf681dead52574301cc68bdfa37442626d333b9`, preserving local investigation
updates and the frozen ticket. The [coverage supplement](../cross_host_coverage_20260928/README.md)
finds the same reported faulty component on B0, 1P4G a37-1 and inspected RCK
cluster nodes. B0 has controlled register/LN reproduction but no corrected
driver run; a37-1's attention faults remain plausible shared impact without
a corrected comparison. The cluster independently retained 2,089 finite
corrected-driver loss rows, with continuing eviction/NUMA activity; host,
boot, FBGEMM build-label or binding differences limit training attribution.
a37-2's successful run lacks handler/exposure measurements. MES/host hangs,
allocator/capacity/fabric problems, source int32 bugs and the STU1 finding
remain outside the demonstrated repair. No remote or GPU action was taken
for this review; per-host source/kernel builds and runtime comparisons remain.

**The driver-team ticket attachment is complete and locally validated.**
The [ticket draft](../session_20260928_driver_ticket_v1/TICKET.md), [portable reproducer](../session_20260928_driver_ticket_v1/reproducer/README.md)
and [proposed source/header fix](../session_20260928_driver_ticket_v1/fix/README.md) are frozen with a complete
manifest. The [standalone archive](../mi450_gfx1250_cwsr_driver_ticket_20260928_v1.tar.gz) contains 238 files,
4,469,946 logical bytes; compressed size is 972,023 bytes,
SHA256 `91180cb5e688f742b69c032d65e0c1fc659ff6868fc780e44403bc0cf2ef06ab`. Every extracted file matches its source byte
for byte, and all four documented CPU checks pass from the extracted copy.
The independent evidence-verifier review passes 17 boundary cases, including
missing/duplicate evidence, duplicate output mappings and failed statuses.
No ticket was submitted and no Git commit or push was made.

**The packaged portable wrapper also reproduced the defect on the current
original driver.** At September 28 00:54 UTC it completed 14 cases and all
94 HIP calls: exactly three predicted lane-zero copies, eleven unchanged
cases, all 14 MODE/EXEC checks and all seven full return PCs correct.
Independent review rereads all 560 output words, 42 raw output/prefill files,
188 HIP journal rows, case/pointer/native/library joins and all four gates.
The producer, Docker execution and root controller exited and were reaped;
the final scan found no device owners. The actual outer session 6228 exited
zero and was drained. No GPU producer is pending at this checkpoint.
[Actual result peer](../session_20260928_driver_ticket_v1/reviews/PORTABLE_RESULT_REVIEW.json).
[Archive validation](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_ticket_finalize_root_v1/FINAL.json).

The before-staging interval includes one already-admitted CPU8/MC60 record,
status `0xd8202000000c080b`, syndrome `0x000000005a800001`, at monotonic seconds
`199232.763916–199232.803189`. All three subsequent portable-run intervals
have zero new kernel lines, MC60 records and hard faults. This is distinct
from the earlier pre-archive MC60 record below, and is no platform-health
clearance. The latest kernel state is
`session_20260928_ticket_repro_root_v1/execution/final_gate/kernel/state.json`,
SHA256 `6474c82ae6632e7e99f011b53122f135bd4dcdf640fdf3503d21e244cf3f984a`.

The [scratch-register proof](../session_20260928_driver_ticket_v1/LIVEOUT.md) shows that all four following
dispatch paths overwrite differing TTMP2/TTMP15 values before use. The
[common-entry proof and driver questions](../session_20260928_driver_ticket_v1/validation/driver_review/DRIVER_REVIEWER_QUESTIONS.md)
show that normal non-wave-start entry executes the complete prologue before
trap/save classification, and the corrected prefetch retains its opcode and
control fields while restoring the intended zero address. These strengthen
the source attribution; vendor bank-state contract, erratum/hazard and
original SP3 generation validation remain explicit production checks.

The corrected-driver counterfactual remains the saved September 23 result
with the exact underlying host/shader, plus the fully audited numerical DW
comparison. The portable adapter has not run freshly on the corrected driver.
The current boot still runs the original module. Further corrected-driver
execution and the unresolved STU1 tensor witness require the already-prepared
external fresh boot with AMDGPU absent throughout, followed by a reviewed
first load. The first-load package below remains boot-unbound and nonrunnable;
there was no live replacement, reset or reboot. The missing historical STU1
operands and training interrupt timeline cannot be reconstructed from these
CPU checks.

**The fresh original-driver sentinel and complete historical DW comparison
reproduce the predicted trap-handler corruption. All 4,000 health checks
pass, and the completed DW evidence is durably archived.**
The live boot is `f268e253-8b79-4be6-ae3c-0d8e8b2dfd4e`, loaded AMDGPU
srcversion `EADFD8EC13CF3E6B329194E`, with CWSR enabled and MES/MES_KIQ
`0x7b`. The completed producers are reaped and their final all-device scans
are idle. No GPU workload is running at this checkpoint. Fresh corrected-driver GPU
validation requires a fresh boot with AMDGPU absent throughout; the current
loaded boot cannot meet that condition. The completed driver-team ticket,
evidence index, portable reproducer and feasible fix are linked above.
Independent audit matches the
loaded GNU build-id to the installed module and binds its native code to the
September 23 original baseline. The tested corrected candidate
still matches that baseline, but **the correction is not currently loaded**.
[Fresh causal and native-module audit](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_causal_audit_v1/REVIEW.md).

**The user approved the exact fourth CPU8/MC60 tuple and serial diagnostic
resumption.** The active policy adds only status `0x98202000000c080b`,
syndrome `0x000000005a800001`, with the complete approved CPU, flags, PPIN
and IPID identity. Its 647 CPU checks pass. The original policy and its
rejections remain preserved. The admitted initial history contains 4,061
journal records and 156 exact MC60 records; its overlapping 2,278 ring lines
match byte for byte. The two startup informational matches have exact
historical dispositions, with no generic exemption for future deltas.
Kernel continuity v2 passes 40 CPU checks, including the five failures of
its preserved first revision. The active baseline controller v3 has 37
source pins and passes 29 CPU checks plus independent source review.
[Exact authorization](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_mc60_authorized_v1/AUTHORIZATION.json).
[Active policy and source receipt](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_mc60_authorized_v1/SOURCE.json).
[Policy integration review](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_active_integration_peer_v1/POLICY_REVIEW.json).
[Initial-history review](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_active_integration_peer_v1/INITIAL_HISTORY_REVIEW.json).
[Kernel-continuity review](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_kernel_continuity_peer_v2/REVIEW.json).
[Controller source review](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_loaded_baseline_source_peer_v3/REVIEW.json).

The September 28 health producer passed four gather/index-select arms with
1,000 exact CPU-oracle checks each, including initial and final source
readbacks. The 14-case sentinel completed all 94 HIP calls. Only explicit
trap modes `0x05`, `0x81` and `0x85` change tags: lane 0 copies `v257` into
`v1` for `0x05`, and into `v513` for `0x81`/`0x85`. All seven controls
without an explicit shader trap and all four equal-bank trap cases remain
unchanged. All 14 packets preserve MODE and EXEC; all seven explicit traps
return to the exact intended PC. Independent review checks all 560 output
words, 42 raw artifacts, native-call records and process identities. All
eight saved gates have exact journal/ring/policy joins, idle scans, and zero
new MC60 or hard-fault records.

The saved September 23 corrected run has zero tag changes with identical
host, configuration, shader, runtime libraries, required environment,
container/image and MES/MES_KIQ `0x7b`. All seven saved driver parameters
also match; a full historical parameter capture is unavailable. This
strengthens the controlled trap-handler cause and fix evidence on the
current native baseline. It does not establish spontaneous trap timing in
every training failure or repair the separate STU1 inconsistency.
[Actual result and corrected-run comparison](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_loaded_baseline_result_peer_v1/REVIEW.json).
[Exact stage, release and source links](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_loaded_baseline_result_peer_v1/STAGE_LINKS.json).

**The fresh step-405 DW quartet is complete.** All four `sink_control_trap`
outputs exactly match the historical corrupted output, including all 1,030
BF16 differences per call: 640 NaNs, 384 extreme finite values and six other
finite differences. The other twelve current outputs exactly match the
healthy baseline. The full matrix independently checks all sixteen complete
outputs and prefills and rehashes the retained inputs. All sixteen saved
arm gates and both matrix gates have no new MC60 records or hard faults.
The September 23 corrected run has all sixteen outputs equal to the healthy
baseline. Its native ELF, complete saved input hashes, packet geometry,
runtime libraries and controls match the current comparison; the native
module audit binds that corrected candidate to the current original source.
Together these results provide strong cause-and-fix evidence for the
controlled CWSR corruption. They do not establish the spontaneous interrupt
history of every training failure.
[Completed matrix and independent result review](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_original_DW_result_peer_v1/COMPLETE_RESULT_REVIEW.json).
[Current-original versus saved-corrected comparison](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_original_DW_result_peer_v1/CORRECTED_COMPARISON.json).

**Archival is complete with all 44 raw files retained.** The raw inventory
totals 14,290,718,927 bytes, including 14,289,201,152 tensor bytes. Two existing
input blobs are verified external dependencies; seven new blobs plus
metadata use 575,181 logical bytes and 618,496 allocated bytes, below the
1 GiB cap. The reviewed archiver freshly hashes every raw file and fully
decompresses all nine unique blobs. The independent result review checks
the exact mappings, process completion, current file fingerprints and gates
without rereading raw or compressed payloads.
[Archive result and independent metadata/stat review](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_original_DW_result_peer_v1/ARCHIVE_RESULT_REVIEW.json).

One complete, already-admitted CPU8/MC60 record appeared **after the GPU and
matrix comparisons and before archive launch**: status
`0x982000000012080b`, syndrome `0x000000005a000001`, monotonic timestamps
`198577.423158–198577.461634` seconds. This is an original policy template,
not the added fourth tuple. The archive interval adds no kernel lines.
The initial zero-MC review assumption and its correction are preserved;
there is no platform-health clearance.

The next first-load package is CPU-prepared. Fresh read-only checks verify
the exact candidate, installed original, dependencies, firmware and helper
inputs; fourteen guard checks pass. Candidate and original-rollback
templates exclude the current and five prior used boots. They intentionally
have no future UUID or runnable `contract.json`, and no activation has run.
An external operator must provide a fresh A0 boot with the existing AMDGPU
kernel-command-line blacklist retained and AMDGPU absent throughout. Root
must then verify complete early-boot evidence, bind the actual UUID, review
the final contract, and perform the controlled first load before fresh
corrected health/sentinel/DW checks. Independent source review passes with
seventeen pure guard cases, exact source/template joins and a byte-identical
reviewed wrapper; the package remains boot-unbound and nonrunnable.
[Prepared first-load package and exact remaining steps](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_next_first_load_source_v1/README.md).
[Read-only preparation receipt](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_next_first_load_source_v1/PREPARED.json).
[Independent first-load source review](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_next_first_load_peer_v1/REVIEW.json).
The pending STU1 witness requires rebinding after the corrected driver is
actually active. The September 27 source audit confirms all eight relevant
sources match both original training archives; chunk indexing and all finite
BF16 scalar decoding checks pass. No source-level explanation of the
step-583 inconsistency was found.
[STU1 source audit](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_STU1_source_audit_v1/AUDIT.json).

**The September 27 full-output re-audit independently confirms the saved DW
fix result.** It reads every word of all 16 original outputs and reconstructs
all 16 corrected outputs from the verified archive mappings: 4,194,304 BF16
output words in total. The four original `sink_control_trap` calls exactly
match the original training-capture member, including all 1,030 differences:
640 NaNs, 384 extreme finite values, and six smaller finite differences.
The other 12 original calls and every corrected call exactly match the healthy
baseline. The complete difference coordinates, native ELF joins and archive
provenance are retained. This is a fresh audit of saved GPU results, not a
new GPU experiment. Identical corrected outputs share one deduplicated archive
blob; the original and corrected runs used different boots and driver source
baselines, so the full source/native causal analysis remains relevant.
[Full output-byte review](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_DW_full_output_peer_v1/REVIEW.json).

The September 27 23:06 UTC checkpoint is historical: it found the device
idle while the two new MC60 records still blocked admission. The subsequent
explicit authorization and completed runs above supersede that pending
state. Its raw logs, rejected reports, seven rsyslogd segfaults and
suppression notices remain preserved.
[Historical checkpoint](/home/chcai/mi450_logs/root_cause_20260919/session_20260927_resume_root_v1/checkpoint/STATUS.json).

The **September 23 pause** checkpoint found no GPU device descriptors across 4,052 tasks,
with the then-loaded corrected driver and MES `0x7b`. The next STU1 diagnostic has
reviewed source and a prepared request, but no actual binding, released
contract, staging or training result. Resume from the
[pause checkpoint and exact preparation receipts](../../mi450_a0/evidence/current_20260923/session_pause.json)
and [current handoff](/home/chcai/handoff/nan_handoff.md).

The pause log adds one complete corrected CPU8/MC60 record, 43 rsyslogd
AppArmor denials, five notices totaling 101 suppressed callbacks and two
profile-replacement STATUS messages after the last archive gate. Independent
saved-log review passes the unchanged diagnostic policy. The full appended
bytes and changed service PID observations remain; no service actor is
identified. This is not platform-health clearance or a new launch gate.

**The gfx1250 first-level KFD CWSR trap handler is a confirmed corruption
source, and the tested corrected driver prevents the reproduced failures.**
The strongest new result is the complete historical step-405 Linear DW
comparison: all 16 corrected-driver calls equal the ordinary baseline,
removing all 1,030 reproduced historical BF16 differences, including 640
NaNs and 384 extreme finite values. LayerNorm, tiny attention, full
historical attention and register sentinels also pass on the corrected
driver with the original comparison kernels, inputs and launch geometry.
[Completed AC2 validation and exact evidence receipts](../../mi450_a0/evidence/current_20260919/CWSR_candidate_AC2_validated_20260922.json).

**A new finite inconsistency remains under investigation on the corrected
driver.** In the September 23 no-prewarm run, step 583 STU1 projection DX
has recorded maximum magnitude **0.0458984375**, while its recorded operands
imply `2048 × 0.04833984375 × 2.2165477275848389e-7 = 2.1943822503089905e-5`.
The output is about **2,092 times** that conditional bound. This is the only
exceedance among 18,000 DX/DW/DB bounds across both 1,000-step runs. Independent
source review confirms `DX = torch.mm(DZ, W.T)` with no scale or addend, and
the next weighted-LayerNorm input repeats the large DX extrema.

These are saved scalar observations, without retained STU1 operand/output
bytes or a native launch trace. Input mutation, a wrong output, an unordered
writer, and a scalar-observation fault remain possible. The result therefore
does not yet identify a physical culprit. All values were finite and below
the original `1e6` alarm threshold, which explains the retained no-alarm
journal verdict. Later LayerNorm amplification alone cannot explain this
earlier inconsistency. The targeted STU1 tensor/scalar witness v2 is now
frozen and independently reviewed. Its 24 GiB reference cap admits all 985
known compact geometries; the old 16 GiB cap would reject steps 720 and 721.
The two small artifacts remain capped at 16 MiB each, and no pre-trigger GPU
operation or boundary readback is added. Retaining references does change
allocator history. Nineteen author CPU tests and 16 independent capacity
cases passed. The actual complete and scalar-only consumer paths also
passed with generated artifacts in containers without GPU devices. Training
integration source review and CPU checks passed; the fresh GPU trial awaits
corrected-driver reactivation and a newly reviewed launch binding.
[Complete bounds, exact step-583 records and independent source review](../../mi450_a0/evidence/current_20260923/current_STU1_DX583.json).
[Witness source, CPU validation, integration review and remaining release steps](../../mi450_a0/evidence/current_20260923/current_STU1_witness_preparation.json).

The saved scalar pattern places no-prewarm DX extrema at −0.04248046875
and 0.0458984375; corresponding full-prewarm extrema are about ±5.1e-8.
Input extrema and shapes match, which does not establish equal tensor bytes.
Following weighted-LN bias extrema are exactly four times the DX extrema
after BF16 rounding; this does not identify four corrupt rows or a bank route.
Saved timing has no nominal overlap with a queue helper span or complete
MC60 record. The nearest preceding restore ends 1.143 s before the DX scan
enqueue. Clock conversion lacks a local bracket, and enqueue times are not
GPU execution times, so this neither proves nor excludes CWSR involvement.

**The September 23 corrected-driver E2E run completed 1,000 numerical steps.**
The September 23 full-prewarm producer exited, and the complete CPU audit
passed all 166 tensor views and 19,237,486,690 floating logical elements,
with zero nonfinite or extreme values. All 38,556,636,485 capture bytes were
hashed. The strict trace controller still reports failure: seven new disabled
TLS event entries made its global-inventory equality check fail. All existing
global controls and the private trace configuration were unchanged; the
complete private trace and final device-idle/kernel gate passed. A separate
exact TLS-inventory adjudication passed at **02:30:31 UTC**, after independent
review and fresh idle/driver/firmware/kernel gates. It retains both original
failure receipts and `global_state_unchanged=false`. Durable archival and
independent complete decompression passed at **02:44:45 UTC**: the
19,557,008,930-byte archive reproduces every original capture byte; the raw
capture is retained.
[Current numerical pass and preserved strict trace failure](../../mi450_a0/evidence/current_20260923/current_E2E_numerical_and_trace_gate.json).
[Distinct completed TLS adjudication](../../mi450_a0/evidence/current_20260923/current_E2E_TLS_adjudication.json).
[Verified archive and original/current numerical comparison](../../mi450_a0/evidence/current_20260923/current_E2E_archive_and_comparison.json).
**A subsequent fresh process passed 1,000 steps without the prewarm helper.**
The complete consumer passed at **05:23:09 UTC**, with 1,000 backward
completions and optimizer updates, no journal alarm and no anomaly dump.
Independent review verified all 68,000 scans, 42,000 operation endpoints,
367,000 finite scalar summaries and 12,000 zero comparisons. The complete
private trace, process exit, final idle and driver/MES/kernel continuity
checks passed under the existing diagnostic policy. This is complete
scalar-journal coverage, without a full endpoint tensor audit. The boot had
already run full prewarm, so cold-boot behavior remains unresolved.

The complete training interval retained **three corrected MC60 records**,
173 rsyslogd AppArmor denials and 14 shared-printk suppression notices; one
additional MC60 record predates the run. Independent review recomputed all
176 continuity intervals and the separate lifecycle aggregate. The
`kauditd_printk_skb` prefix does not identify every suppressed message or
prove they were all audit records. Private KFD tracing separately passed
its complete loss/count audit. No platform-health clearance follows.

The full-prewarm/no-prewarm comparison examined all **379,000 summary
pairs**. Normalized event counts and geometry match, but 166,932 summaries
differ across 998 steps. The earliest difference is a timestamp-embedding
gradient extremum at step 1, at most `8.73e-11`. Later differences can be
substantial: at step 583 the content-MLP second-Linear bias gradient maximum
is **66.5 without prewarm versus 0.023193359375 with prewarm**. These values
are finite; the difference requires investigation and cannot be dismissed
as tiny rounding noise. Equal summaries would not establish tensor equality.
These gradient observations precede norm clipping at 1.0; they are not
post-update measurements. The complete bound analysis above localizes an
earlier STU1 DX inconsistency within the step-583 backward pass.
[Completed no-prewarm 1,000-step audit and full scalar comparison](../../mi450_a0/evidence/current_20260923/current_no_prewarm_1000_steps.json).
The earlier two-step screen also passed; only tiny timestamp-gradient
extrema differed within that short comparison.
[Completed no-prewarm screen, independent review and scalar comparison](../../mi450_a0/evidence/current_20260923/current_no_prewarm_two_steps.json).
The earlier AC2 targeted-preload run completed zero steps in the retained
ROCm executable-freeze / PM4 code-cache-invalidation wait. Changed platform
source, boot and full prewarm prevent assigning the new completion to MES
or the CWSR correction alone. Unique attribution of every historical NaN and
long-run stability remain unproved.
[Retained AC2 E2E failure and complete trace](../../mi450_a0/evidence/current_20260919/CWSR_candidate_AC2_E2E_targeted_stall_20260922.json).

**The September 23 resume found a changed platform.** New boot
`b685ba14-e2ad-4751-b755-2e7bba3b58fa` initially had no loaded AMDGPU.
The September 22 reinstall left disk and initramfs unified MES at **`0x7b`**,
not the user-reported testing `0x17d` or intended official `0x7e`.
No local `0x7e` artifact was found. This does not establish a defect in
`0x7b`; tests on that version must be labeled accordingly. Prior per-run
MES versions were not pinned, so the report cannot retrospectively assign
`0x17d` to each old failure.
The bounded public lookup also found no matching image: both exact unified-MES
URLs returned 404, while the successfully retrieved linux-firmware `WHENCE`
snapshot contains no `gc_12_1_0` entry. Vendor/private releases remain outside
that lookup. [Public lookup receipts](../../mi450_a0/evidence/current_20260923/public_MES_lookup.json).

At that initial checkpoint the installed AMDGPU srcversion was
**`EADFD8EC13CF3E6B329194E`**, and its
embedded gfx1250 handler is byte-identical to the original faulty 5,656-byte
handler. The reinstall also removed custom persisting-GL2 controls and
changed `SH_MEM_CONFIG` retry-bit initialization. All 13 reviewed MES
source/API files are semantically unchanged. Firmware, driver source and
boot therefore changed independently; a new pass would not isolate MES.
The old initialization logs narrow the retry-bit observation: only VMID0
had `RETRY_DISABLE=1`; all 120 later reads across eight XCCs had zero.
This does not show a changed KFD user-VMID retry bit or establish final live
register state. [Source/log join and independent review](../../mi450_a0/evidence/current_20260923/retry_initialization_scope.json).

A further source/native comparison confirms that subsequent KFD user-queue
retry branches and masks are preserved; the selected MES add-queue function
is byte-identical. The removed GL2 code includes an optional residency/flush
write path, but no retained evidence shows it ran during the stall. Its
unconditional initialization instrumentation only reads registers, and the
threshold setter is a software shadow. An unresolved flush-register address
discrepancy matters only if that optional write path executed. The saved
ROCr packet requests GL2 invalidation, without identifying a hardware PC or
the cause of nonprogress. These findings narrow the source confounders;
they do not isolate firmware or prove the old stall's cause.
[GL2/retry source, native and independent peer evidence](../../mi450_a0/evidence/current_20260923/GL2_retry_stall_source.json).

A fresh CWSR-only candidate was built against the reinstalled source,
srcversion **`AD83C153B9701570F12964E`**, module SHA256
`78de19854e38e298f6418bddcfa39f44a797f811f0f4b9c585482f4c4dc2b2b4`.
Independent CPU audit proves the rebuilt baseline matches installed
executable/data content and the candidate changes only the corrected handler,
two handler-size immediate bytes, associated layout and metadata. It was
not loaded at that CPU checkpoint. A subsequent guarded first load completed
at **00:23:14 UTC**. Cached driver attributes report both MES and MES_KIQ
**`0x0000007b`**; the expected corrected driver/CWSR, idle and kernel-delta
checks passed. Health and sentinels then passed at **00:28 UTC**: 4,000
exact health checks, all 14 tag/MODE/EXEC cases and seven exact trap-return
PCs. Both producers exited and the final all-device idle check passed.
One known corrected MC60 record occurred between activation and health;
the actual workload and final intervals had no new MC60 or hard fault.
LN/tiny attention also passed all 24 processes and 36 dispatches; its complete
consumer verified 856 raw files and 1,024 native pins. Full historical
attention also passed four arms/eight calls. Its 144 raw files match the
prior corrected result; all 13,053,235,840 bytes are losslessly archived
and independently decompression-verified. Historical step-405 DW also passed:
all 16 full outputs equal the baseline, with complete input/native/lifecycle
audit and a verified lossless archive of all 44 files. Natural zero-DY DW
then passed all 1,000 calls with relevant SVM exposure, as detailed below.
A `0x7e` comparison remains pending the actual firmware artifact.
[Platform transition, exact module/source/firmware evidence and build audit](../../mi450_a0/evidence/current_20260923/platform_transition.json).
[Current first load and loaded firmware](../../mi450_a0/evidence/current_20260923/current_first_load.json).
[Current health and register sentinels](../../mi450_a0/evidence/current_20260923/current_health_sentinels.json).
[Independent retained-health review and MC60 timing](../../mi450_a0/evidence/current_20260923/current_health_peer.json).
[Current LayerNorm/tiny-attention complete result](../../mi450_a0/evidence/current_20260923/current_components.json).
[Current full historical attention and verified archive](../../mi450_a0/evidence/current_20260923/current_full_attention.json).
[Current complete historical DW and verified archive](../../mi450_a0/evidence/current_20260923/current_historical_DW.json).

The current DW interval retained two matching corrected CPU8/MC60 records
after attention, one during the final DW producer, and three SVM restore-work
CPU-hog warnings. The narrow diagnostic policy admitted the MC60 records;
there were no hard driver faults. All DW calls and lifecycle checks completed.
The workqueue warnings alone are therefore not a discriminator for the prior
persistent training stall. No platform-health clearance follows.

The first current natural-DW attempt stopped **before HIP initialization or
worker release** at 00:58 UTC. Its trace checker required literal `1` and
rejected the queue-eviction event's `1*` readback with the stacktrace trigger.
The held worker expired normally; worker and launcher are absent, the empty
private trace is preserved, global tracing was restored, and final idle/kernel
checks passed at 01:01 UTC. No numerical call ran. A fresh contract retained
that failed attempt and corrected the trace-readback check before natural DW.
Independent review matched the installed kernel build ID to runtime notes and
confirmed `1*` means enabled with soft mode from the stacktrace trigger.
[Pre-HIP failure and completed cleanup](../../mi450_a0/evidence/current_20260923/natural_DW_preHIP_abort.json).

**Current natural zero-DY DW passed with relevant SVM exposure.** The unchanged
vulnerable `sink_control` kernel completed all 1,000 calls: 131,072,000 BF16
words were positive zero. The complete consumer and independent repeat audit
verified all native packets, output/prefill checks and 3,570,465,280 input bytes.
The trace retained 1,087 records with no loss or probe misses; 18 qualifying
AutoNUMA/SVM eviction–restore pairs occurred within 18 call envelopes.
The worker exited, devices were idle, and kernel logs were byte-identical
through completion at 01:11:58 UTC. These are relevant exposure windows,
not observations of physical CWSR acknowledgement, interrupted PCs or banks.
[Complete natural-DW result and exposure limits](../../mi450_a0/evidence/current_20260923/current_natural_DW.json).

**The original 400 NOP-only plateau probes now pass on the current driver.**
All recorded tags remain unchanged in all 400 packets. Ten cases have fresh
in-plateau return PCs and matching target AutoNUMA/SVM eviction–restore pairs:
three under mode `0x00`, two under `0x05`, three under `0x86` and two under
`0x46`. The mode-46 cases, ordinals 75 and 239, add natural preservation
evidence for the reverse `v513 → v257` route associated with historical DW
address corruption. The complete consumer verified all 1,200 raw images,
2,410 HIP calls, trace integrity and saved lifecycle; independent saved-result
review reproduced the complete packet, tag, exposure and lifecycle results. There were zero explicit traps. Host helper events do
not acknowledge physical CWSR completion or reconstruct the historical
training failures; unchanged sampled PCs also do not prove no trap occurred.
[Current plateau packets, exposure and lifecycle result](../../mi450_a0/evidence/current_20260923/current_plateau400.json).
**Natural historical nonzero-DY DW also passed all 1,000 calls.** Its first
complete output qualified against the retained ordinary baseline, and every
subsequent BF16 word matched exactly. Independent review checked all 131,072,000
output words, complete prefills, 3,570,465,280 input bytes and all 172 native
argument bytes per call. Fourteen target AutoNUMA/SVM pairs overlap 14 call
envelopes; the full private trace has no loss. The completed worker and final
idle/kernel checks passed on the same corrected driver and MES `0x7b`.
[Complete nonzero-DY result and independent review](../../mi450_a0/evidence/current_20260923/current_natural_nonzero_DW.json).
[Remaining causal gaps and natural-probe criterion](../../mi450_a0/evidence/current_20260923/current_causal_gaps.json).

The full-FBGEMM-prewarm E2E experiment completed with the original seed 1,
batch 1024, asynchronous execution and DataLoader settings. All current
prerequisites and loaded MES `0x7b` are pinned. The complete numerical audit
passed at 02:06:51 UTC; the original strict trace-inventory failure is retained.
Two known corrected CPU8/MC60 records appeared during training, with continued
numerical progress. Independent interval reviews matched both complete records
to the frozen diagnostic policy, including the second record's Over/UECC flags.
Associated SVM/MCE workqueue-duration warnings and rsyslogd AppArmor denials
remain in the full logs. No platform-health clearance follows. At 01:37:45,
all 256 private trace buffers had zero drops, overruns and consumed events.
[Current E2E progress, kernel reviews and trace-buffer snapshot](../../mi450_a0/evidence/current_20260923/current_E2E_in_progress.json).
The final private trace has 3,045 records without loss, 176 target evictions
and 175 restores: 167 NUMA protection scans, six NUMA folio migrations, two
forks and a final exit. All prewarm/worker/journal joins passed independent
review. The seven new disabled trace entries are owned by the matching TLS
kernel module; no event-registration actor or time is inferred from that fact.
[Complete raw audit, final trace review and TLS inventory evidence](../../mi450_a0/evidence/current_20260923/current_E2E_numerical_and_trace_gate.json).

The original/current comparison preserves finite differences. Initial RNG,
all compared settings, event counts and all 1,000 scan geometries match.
Normalized scalar summaries differ at 998 steps, and all 20 sampled losses
differ. The earliest retained scalar difference is only the timestamp-embedding
gradient extrema after step-1 backward; its largest absolute difference is
`1.1641532182693481e-10`. The largest logged loss difference is
`4.144012928009033e-5` at step 750, about 0.0293% of the original value.
All endpoint layouts match, but 69 gradient, 65 parameter and 28 other logical
tensor hashes differ. These are finite differences, not proof of a NaN,
reduction nondeterminism, or a firmware cause.
[Exact comparison and independent earliest-divergence analysis](../../mi450_a0/evidence/current_20260923/current_E2E_archive_and_comparison.json).

**The complete `/home/chcai` allocation is below the requested 1 TB limit.**
The full measurement at **06:48:06 UTC** was **992,334,336,000 bytes**, leaving
**7,665,664,000 bytes** below the decimal 1,000,000,000,000-byte limit.
The earlier exact 18-file retirement completed at **04:06:36 UTC** and reclaimed
**530,390,228,992 allocated durable bytes**. It removed eight obsolete finite
boundary snapshots, one requested healthy attention capture, two healthy
forced-replay base storages, six old finite diagnostic frames and the old
September 22 finite E2E archive. Compact reports, manifests, source/native
evidence and historical failure verdicts remain. These original successful
snapshots were intentionally retired without replacement archives; direct
raw reruns or complete rescans of those snapshots are no longer available.
The two replay bases had different complete SHA256 values, so this operation
does not claim they were duplicates. Both retired replay directories now
contain `RETIRED.json` and `README_RETIRED.md` notices; their original compact
metadata and reports remain unchanged.

Cumulative storage maintenance reclaimed **897,569,296,384 allocated durable
bytes** and **77,197,692,928 tmpfs bytes**. Current MIN_HISTORY4086 arrays,
raw source data, required metadata maps, the long-step101 replay base,
unique NaN/failure captures, and the current September 23 E2E raw capture and
archive remain. The then-pending no-prewarm experiment's complete 336-source
closure has no reference to the retired snapshots. Earlier cleanup removed
two obsolete MIN_HISTORY64 arrays and two compression scratch copies after
dependency/ownership checks. Exact duplicate input files share inodes; the
earlier 54-file zero-block sparsification pass verified identical complete
before/after hashes. Its original and final metadata remain recorded.
All 38 selected old `gpucore.N.gpu` dumps are now losslessly archived and
independently decompression-verified; removing their raw copies reclaimed
19,216,740,352 net allocated bytes. `core` and `core.gpu` were excluded.
The completed projection's 24 temporary tensor files were subsequently
removed after fresh full raw/archive/decompression verification, releasing
64,219,217,920 tmpfs bytes. The complete current projection archives remain.
The earlier roughly 1.4 TB usage was dominated by accumulated experiment
captures: about 815 GB of root-cause experiments, 454 GB in three replay
directories and 103 GB of other trace/replay captures at the saved snapshot.
Dataset and preprocessing storage are additional. Duplicate snapshots and
allocated zero-filled tensor regions account for much of the reclaimed space.
[Exact cleanup paths, retained evidence and receipts](../../mi450_a0/evidence/current_20260923/storage_cleanup.json).

The first current controlled-projection attempt, TLS v3, stopped before trace setup
or worker creation: its generated uppercase `TLS` tag failed the unchanged
controller's lowercase-only validation. No projection numerical call ran.
The original failure is preserved. A fresh read-only postfailure gate passed:
devices are idle, driver/MES/kernel continuity is intact, and no resources
from that attempt remain. A fresh package corrects tag generation and adds
actual generated-plan admission checks.
[Preserved preworker failure and completed postfailure gate](../../mi450_a0/evidence/current_20260923/projection_v3_preworker_abort.json).

**The corrected-driver projection quartet subsequently passed in TLS v4.**
All four arms completed four calls each, with every one of the
16,052,699,136 output BF16 words positive zero. The unequal-bank trap arm
also returns zero at the four positions corrupted with the faulty handler.
The complete consumer verified all native packets, 32,105,398,272 output
bytes and 32,113,786,880 retained input bytes, plus lossless archives,
private traces and worker/device lifecycle checks. All four independent
saved-evidence reviews passed. Queue-helper counts cover the entire worker
lifetime and do not establish physical CWSR during the four GEMMs.
The four initial gates admitted five, zero, one and zero known corrected
MC60 records, respectively; all predated the corresponding trace-controller
start. No additional MC60 records or hard driver faults appeared between
each initial and final capture. This is not platform-health clearance.
[Current projection quartet and exact evidence](../../mi450_a0/evidence/current_20260923/current_projection.json).

The preceding AC2 boot was `6478859b-3092-4792-a933-e91c311f2cd1`. Its AMDGPU was
`7.1.1.31300009`, srcversion `8336BBB74E6C4D53275A4E5`, with CWSR enabled.
Candidate module SHA256 is
`7e9a2985cb07838caf44793751ff6ce0149b61062e39bf0a4d72716c7289e8ab`;
the installed original module was unchanged at that checkpoint. At the final investigation
checkpoint, worker PID 38536/start 342982, launcher PID 38529/start 342973
and eight DataLoader children were preserved. Only the owned supervisor
was interrupted and reaped; the final idle gate failed. Private tracing is
stopped and retained with its probes. The September 22 document refresh
confirmed that boot, driver and retained workers. These PID identities are
historical and must not be reused.

**The old stalled boot required a new host AC cycle.** The retained failure
excludes more launches, retries, reset, unload/reload and recovery diagnostics
on that boot. The earlier `hqds` read caused a separate SDMA diagnostic
NULL dereference, and the candidate retains that function. The September 23
resume uses fresh platform and output bindings; it does not reuse that
boot's failed launch contracts.
