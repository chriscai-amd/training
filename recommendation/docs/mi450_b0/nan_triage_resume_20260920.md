# MI450 B0 NaN continuation — September 20, 2026

Checkpoint includes the **04:32:45 UTC verified uninstrumented 3,000-step pass**
and completed component controls through **05:03:10 UTC**.
**The 3,000-step validation bound passed; the original culprit and fix remain
unresolved.** Earlier NaNs and hard faults under the same production settings
are still unexplained.
All GPU jobs are terminal and their sessions drained. The private candidate-172
test now passes **1,000 calls**. The user requested wrap-up again; the new
ordered DB→DW→DX control is **CPU-ready and GPU-unrun**. No further launch
followed that request.
Historical evidence remains in [the main B0 record](mi450_b0.md) and
[the September 19 continuation](nan_triage_resume_20260919.md).

## Carry-forward user rule

**Do not commit or push anything until the user explicitly instructs it.**
The user reaffirmed this rule on September 20. It supersedes earlier push
authorization and applies to all agents and subsequent continuations.
The user subsequently explicitly authorized committing and pushing the updated
B0 documentation. That authorization covers this main-document/continuation
update only; code and diagnostic artifacts remain outside this commit, and
future commits or pushes still require explicit instruction.

## Restored execution context

| Item | Verified / observed state |
|---|---|
| Worktree | `/home/chcai/training-mi450-nan-20260919`; reviewed remote changes fast-forwarded to `191adbc`, with no production-code changes from that update |
| Shared branch | `chcai/mi450`; this B0 documentation update is explicitly authorized for commit/push |
| Current host boot | `05c3c692-9f81-44ce-98cb-5bed8231635b`, started September 19 around 15:21 UTC; old stalled workers are gone; this investigation issued no manual reboot |
| Required container | `mi450-b0-0910-7ff-start0`, restarted September 20 at **02:10:26.244 UTC** |
| Image | September 10 base, unchanged `sha256:6e656de79e6c8d7f3b3db3690606c3e6cb0536ec871f0521735746015297c9a1` |
| Runtime | Torch `2.11.0+rocm7.14.0a20260625`, HIP `7.14.60850`, Triton `3.8.0`, exact commit `7ff97e310935b4a79794878dbc911f9af25d38d9` |
| Frozen source | `/tmp/mi450-b0-continue-partials-20260919/recommendation`; all **143/143** saved source hashes match, with no extra files |
| Driver observations | `gpu_recovery=-1`, `halt_if_hws_hang=0`, left unchanged; September 19 07:11 failure context recorded `gpu_recovery=0`, `halt_if_hws_hang=1` |

All artifact paths below are relative to
`/home/chcai/runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/`.
Runtime/source verification is under `resume_20260920/` in
`verified_stack_cpu.json`, `triton_commit.txt`, `source_partials_verification.json`,
`docker_inspect.json` and `boot_id.txt`.

Use the mandatory environment in [the B0 stack instructions](mi450_b0.md#13-mandatory-flags-and-boot-procedure):
buffer ops off, allocator settings empty, coredumps off, full autotuning off,
pipelining off. The health/LN controls use `AMD_SERIALIZE_KERNEL=0`, attention
cap256 and `WEIGHTED_LN_BWD_BLOCK_N=0`; the latter does not configure LN
forward. The native DW arm uses inherited BN1 and workspace1; no Triton
kernel runs in that native BLAS control. Its recorded hipBLASLt library
path is the unchanged gfx1250 catalog, with no tuning-file or solution-index override.
GPU jobs remain serial and owned by the root investigator. Leave `mi450_c`
untouched. Use privileged device ownership checks as well as `amd-smi`;
`amd-smi process` previously missed a live worker. Do not repeat the faulting
KFD `hqds` read or run the forbidden post-reboot setup script.

## Completed finer training capture: hard fault

`preprocess_stages_bn1_20260920_20260920T023107Z/` started **02:31:07.524 UTC**,
root session **96316**, on the verified current boot and required runtime.
Its archive retains the same **143** frozen source hashes. Configuration:
timestamp zero, batch 1024, attention cap256, weighted-LN BN1, both
`HSTUComputeOutputFunction` and `_HSTUPreprocessAndAttentionFunction`
observed, preprocess stages enabled, output internal stages disabled,
fused checker enabled, **42 GiB** capture cap and **1,000** diagnostic steps.

The run ended **02:57:08.201326 UTC**; its lifecycle wrapper ended
**02:57:08.297347 UTC**. Launcher and wrapper both exited **1**, and root
drained session **96316**. All **95 MLLOG loss samples, 10..950**, are finite.
The final sample is **0.1375616043806076** at **02:56:47.654 UTC**; no 960
sample follows. Watchdog detection at **02:57:02.035046345 UTC** reports
illegal GPU memory access, `ExchangeDevice` / `ProcessGroupNCCL::Watchdog`,
then the worker terminates with **SIGABRT**. This is asynchronous detection,
not the originating kernel stack. Last progress is `train_ts=26`,
`train_batch_idx=60`, with last logged warmup `gstep=950`. Archived source
prints batch progress before forward, so it does not establish the exact
attempted faulting step or last completed optimizer iteration.

No tensor payload was saved. Ten **printed** read-side GPU page-fault records
(`RW=0`) name Python host PID/TID **389858**, VMID **3**, PASID **268**, TCP
client, `AID1.XCD1`, with seven distinct page bases spanning
`0x1b9468000..0x1b94c9000`. `MORE_FAULTS=1` and a suppressed callback mean
ten is not the total number of failing accesses. The log has no shader PC or
kernel symbol. Kernel ISO timestamps/reporting do not establish cross-log
event ordering or a component attribution.

The terminal CPU audit independently hashes the source archive and matches
all **143** files to expected, prelaunch and loaded manifests; pinned source,
launcher and environment hashes are intact. Boot is unchanged, saved final
device ownership is empty, wrapper `errors` and `lifecycle_errors` are empty,
and no reset/timeout/oops appears in the ordered kernel delta. Final free disk
is **50.070 GiB**. These checks establish terminal disposition, not a training
repair: this run did not pass its **1,000-step diagnostic bound**. The later
uninstrumented 3,000-step pass is a separate run, recorded below.

Evidence: run `outcome.json`, and
`resume_20260920/capture_control_20260920T023107298945Z/terminal_cpu_audit.json`
with reusable `audit_terminal_cpu.py`. The audit retains all full-precision
loss samples, relevant log lines, fault records and source/evidence SHA256s.
Train-log SHA256 is
`75cbcc789c1cc7fb8fabe67045b265ffb7f98ab75626c4ce79cafe6e8c1325c6`;
ordered kernel delta SHA256 is
`d56033a0b1788d02ea80a02493c7cb4de0e370d6541dde980b7a91da23cd2e79`.

A separate frozen-prefix comparison, ending at **840**, found that this run
and September 18 upstream-magnitude training have **75** shared finite samples
through 750: **two identical**, **73 different**, first difference at step 20
by two float32 ULPs. Maximum absolute difference is **9.0152e-6 at 730**.
Their step-750 losses both display `0.14136` but differ in full precision.
This checks sampled loss trajectories only, without parameter/gradient equality
or causal attribution. Evidence: `resume_20260920/loss_prefix_comparison_cpu.json`
and `compare_loss_prefixes_cpu.py`; this prefix audit is separate from the
terminal audit through step 950 above.

The reviewed uninstrumented **3,000-step E2E controller** is preserved as
`resume_20260920/run_e2e_control.py`, with
`run_e2e_control.cpu_readiness.json`, `run_e2e_control.native_guard_readiness.json`
and `e2e_root_review.json`. It subsequently launched after the completed
postfault health, B0-shape projection-DX and checker controls below.

## Postfault health controls

The original GPU-comparison health run
`continue_20260919T065146Z/health_20260920T025832602479Z/` passes **4 × 1,000**,
exit **0** at **02:58:34.018836 UTC**; root drained session **88013**.
The independent CPU-reference run
`resume_20260920/health_cpu_postfault_20260920T030032397781Z/` also passes
**4 × 1,000**, exit **0** at **03:00:38.863931 UTC**; root drained session
**25665**. Every output column is checked on CPU each iteration, with index
range/stability checks and complete source checks before/after each arm.
Width-16 source checks cover **37,562,496** elements; width-4 checks cover
**9,390,624**. The exact copied CPU probe hash matches its manifest and the
prior frozen probe. The original GPU-comparison run has no per-run frozen
source manifest, so its current source hash is audit-time context only.

Both saved boot IDs match, and saved GPU ownership is empty before/after.
Original health kernel snapshots are byte-identical; the CPU-reference
snapshots differ only in ISO timestamps. Between the runs, three NIC
transceiver warnings appear, with no new GPU-fault message. This checks
gather/index health after the training fault; it does not attribute that
fault or establish an end-to-end repair. Evidence and input-file hashes:
`resume_20260920/postfault_health_cpu_audit.json` and reusable
`audit_postfault_health_cpu.py`.

The postfault ordinary BN8 backward helper,
`continue_20260919T065146Z/backward_20260920T030131808050Z/`, also passes
**1,000**, exit **0** at **03:01:46.988822 UTC**; root drained session **22642**.
All six input-byte guards pass at **100** checkpoints, with no output failure.
No partials-before-reduction check runs in this ordinary arm. DX remains the
same `eae3693f...` **576-register**, zero-spill kernel, full HSACO
`788c2b9991cee0cbb6955eb35359020f9d683f4f8863bdf02debd31ebd4eb5a6`,
as the prior standalone positive. Passing again after training does not
re-establish a positive control or identify a repair.
The adjacent `terminal_cpu_audit.json` also verifies unchanged boot, empty
saved ownership and identical normalized kernel messages. It verifies the
recorded GPU-checker outcome; passing tensors were not retained for CPU replay.

## Completed projection DX at the B0 captured shape

`resume_20260920/projection_dx_b0shape_postfault_20260920T030256944702Z/`,
started **03:02:57.029382 UTC**, completed **03:10:20.910571 UTC**, exit **0**;
root drained session **69292**. The same synthetic script uses the B0
step-51 row count **2,468,813**, zero BF16 DZ `[2468813,2048]`, finite W
`[512,2048]` and allocating `torch.mm(DZ,W.T)`. All **1,000** complete CPU
output-bit checks pass, as do all **11** full input-byte guards. Runtime is
**442.838 seconds**. All **1,000** native launch starts join successful
returns; actual kernel is `MT128x256x128`/LDSB0/SK3/TDMI3, with
**256 workgroups × 128 threads**, matching the earlier synthetic A0-shape
control rather than A0's controlled `MT128x240`/fixed460 positive.
Workspace/StreamK/tuning-file/solution-index overrides are unset. Frozen probe
SHA256 is `c43c806bb98e43d8be3c5187ad19e647bdefd0a51baea18bf5810018b4973199`.
Source is unchanged, boot IDs match and saved owners are empty before/after.
The kernel-message delta contains four NIC transceiver warnings and one
SVM-workqueue CPU-duration warning, with no GPU fault/reset. This tests the
captured shape with synthetic inputs; original data and allocation history
differ. The trace does not establish a native catalog index, executable byte
identity or independent argument-ABI proof. Evidence:
`terminal_cpu_audit.json`, `report.json`, `hip_trace.jsonl` and frozen `probe.py`
in the run directory.

## Completed full-size fused-checker index control

`resume_20260920/checker_index64_20260920T031223855761Z/` completed
**03:13:42.088834 UTC**, exit **0**; root drained session **29050**.
Both arms pass **100 calls**, with all **1,234,407 status bytes** compared
on CPU after every call. The contiguous BF16 input `[2468813,2048]` contains
**5,056,129,024 elements / 10,112,258,048 bytes**. The zero arm expects every
status byte zero. The sparse arm inserts NaN, infinity and exact finite
values above `1e20` at `2**31-1`, `2**31`, `2**32-1`, `2**32`, and the final
element. All five block locations and all expected flag bits are correct.
The final block has **2,048 valid lanes**. All **four** complete input-byte
guards match their bounded-chunk CPU initialization hashes.

The exact production kernel uses BLOCK4096, four warps, one stage and
`MAX_ABS=1e20`. Actual compiled hash is
`e285a0697c51543df409234babd05d7fb3e0f1363385109242ea29ed4e32c960`,
**154 registers**, zero spills; the recorded `Numel` argument is **i64**.
Retained HSACO SHA256 is
`df776b381e528fccaf7e630eef3a224749608fb4bc8267e705e53ff192a285be`.
Final copied runner SHA256 is
`41fca209910364a58defd08c0458c677c01f3d5c612a9b435889c1e8cbcbce06`;
it matches its manifest and report. Both source-file guards match the frozen
production checker. Boot is unchanged, saved owners are empty, and kernel
snapshots are byte-identical. No failure payload exists.

This is a diagnostic-path control, with full synthetic input/status CPU
oracles and explicit 32-bit boundary coverage. The source already uses
64-bit addressing; no source truncation bug was established or repaired.
The direct kernel control does not execute caller `reduce_status` or its
Torch reductions. The pass does not attribute the training NaN or clear all
checker layouts, concurrency, transient changes between input guards, or
training execution histories. Evidence and hashes are in
the run's `terminal_cpu_audit.json`, `report.json`, retained HSACO and source
manifest. The reusable runner is `resume_20260920/repro_fused_checker_index64.py`;
CPU readiness includes boundary/flag fixtures and retention fault injection.

## Completed uninstrumented 3,000-step training

Run `e2e_bn1_3000_20260920T031436710808Z/` completed at
**04:32:45.582339 UTC**, controller exit **0**; root session **1890 is drained**.
It launched **03:14:37.849823 UTC**, timestamp zero, batch1024, seed1, empty
checkpoint, attention cap256 and weighted-LN BN1. All tripwires, module probes,
replay/capture hooks, fused checks, internal stage observers and preload
tracing were disabled. Native BLAS/workspace/tuning settings stayed inherited
and unchanged. Prelaunch and terminal CPU checks both match all **143/143**
Python hashes, GIN hash, required September 10 image, Torch/HIP and exact
Triton commit. All **60** common production Python files also match the earlier
cap256/BN1 E2E run that produced NaNs at60/70.

All **300 MLLOG samples, 10..3000**, are finite. Final full-precision loss is
**0.14524586498737335**, at **3,072,000 samples**, train timestamp34/batch306.
The log explicitly records `die_at_step=3000 ... global_step=3000` after the
optimizer update and worker exit **42**. Multiprocessing returns raw launcher
exit **1**, as expected for this intentional worker exit; the controller
classifies the complete evidence as **BOUND_COMPLETE**. No scoped stop,
monitor error, lifecycle error, GPU-fault candidate or reset is recorded.
Boot is unchanged, and final/audit-time device owners are empty.

The independent `terminal_cpu_audit/report.json` verifies the complete
full-precision loss sequence, console sample coverage, exact bound/worker
markers, stable whole-log hash, source/GIN/stack and effective environment,
controller/input hashes, boot, kernel delta and owners. It reports
**verified_uninstrumented_3000_pass=true**, exit0. Audit SHA256:
`a76ea6e661ad48c403e418cdc181049b22cfdc612d006c3e5fb273cfe42cc871`.
Train-log SHA256:
`0fc64c1faca1ad3711e6e0ed35efe23cf6b2318e2bd7b21bb2108ad5c3d107b9`.
The terminal audit's files named `*.before` are fresh post-terminal CPU
runtime/source records, not overwritten training prelaunch evidence.

This is a bounded stability pass of existing settings, **not a newly
identified causal repair**. It does not erase the prior BN1/cap256 NaNs or
instrumented hard faults, and sampled losses do not inspect every tensor.
The culprit investigation continues with component controls below.

## Completed real-B0-weight projection DX controls

`resume_20260920/projection_dx_b0weight_shape_cycle_20260920T043356880290Z/`
completed **04:40:46.076033 UTC**, exit0; root session **17072 drained**.
All **1,000 complete CPU output-bit checks pass**, **200 per shape**, with
all **11** full W/DZ guards and the final compact weight-file guard matching.
Source and boot are unchanged, saved/current owners empty, no GPU fault
candidates in the kernel delta. This control uses actual finite step55 B0
UVQK W, exact zero DZ, a shared max-sized backing and allocating
`torch.mm(DZ[:rows],W.T)`. No workspace/StreamK/solution override was supplied.

The full terminal trace audit joins **1,000 launches, successful returns,
argument packets and report calls** without issue. The exact full
MT128x256x128 symbol stays constant, while geometry depends on active rows:

| Active rows | Workgroups |
|---|---:|
| 1,959,558 | 256 |
| 2,468,813 | 256 |
| 2,097,151 | 256 |
| 2,097,153 | 256 |
| 2,013,428 | 31,460 |

The initial partial geometry impression was incomplete; final evidence
retains per-shape native scheduling arguments. The frozen full trace is
**1,849,315 bytes**, SHA256
`b8ed4cdbe9fc7b1213a422ced6da8482acf08f74b86033772d8e2564c330d236`.
Evidence: run `root_terminal_gate.json` and
`cpu_native_audit/terminal_20260920T044110Z/{audit.md,audit.json,decoded_launches.json}`,
plus independent `cpu_native_audit/oracle_identity/` data provenance.
Loaded ELF identity is unobserved; this is still zero-DZ/synthetic-sequence
coverage, with preceding production DB/DW work omitted.

The short fixed-grid selector control
`resume_20260920/projection_dx_b0weight_fixed460_selector_20260920T044242645001Z/`
then passes **5/5**, **six** full input guards, exit0 at **04:43:16.658560 UTC**;
session **38149 drained**. At rows1,959,558, setting
`TENSILE_STREAMK_FIXED_GRID=460` gives **460 workgroups ×128 threads**, but the
same **MT128x256x128** symbol remains selected. All five launches return
success. Source/weight bytes and boot match, owners are empty, no GPU fault
candidates. `root_terminal_audit.json` retains the terminal checks; trace SHA256
`4e1fdb4e06911f6a5b1fe52c5a8ba6ef9d924f8e740f0192411deaab196b4e60`.
This verifies that fixed-grid460 alone did not select A0's MT128x240 kernel.
A private catalog subsequently selected candidate 172 successfully in the
short control below; no installed catalog was changed.

## Completed private candidate-172 selector

`resume_20260920/projection_dx_private172_fixed460_selector_20260920T045021585365Z/`
ran **04:50:21.673410–04:50:55.650302 UTC**, exit **0**; root session
**47414 is drained**. All **5/5** complete CPU zero-output checks and **six**
full W/DZ byte guards pass. Source, compact weight and all private catalog
bytes are unchanged; boot matches, saved/current device owners are empty,
and normalized kernel messages are identical. No failure payload exists.

The independent terminal audit joins all five launches, successful returns,
real argument packets and report calls without issue. Every launch uses the
exact candidate-172 **MT128x240x128** symbol, at **460 workgroups × 128 threads**.
All five calls use **1,959,558 rows**; this is a fixed-shape test. Arguments
are identical across calls: `numWG=skGrid=skTiles=460`,
`ItersPerTile=SKItersPerWG=16`, M512/N1959558/K2048, alpha1/beta0. W/DZ/output
pointers, shapes and strides join the numerical report. A lookup error500
precedes successful symbol resolution; all actual launch returns succeed.

The private catalog is under
`resume_20260920/projection_dx_solution172_private_catalog_20260920/gfx1250/`.
Exactly **one of 402 regular files** differs from the retained installed
catalog: the projection `PredictionMatching` candidate list changes from
`110..237` to `[172]`. All solution definitions, predicates, scheduling
parameters and native object bytes are unchanged. Runtime constructs
candidate 172's actual packet; no raw function-handle substitution is used.
The pre/post manifest verifier and inventory checks pass. Pinned manifest
SHA256: `9440ab8f4592d53c2da3d45b998abbbfc96b65a2d40fb639ed823840bc7945df`.
The launcher's required **inventory fingerprint** is a different hash:
`e0dcd6ba02ad18f0d45200170f3ed944781d75bbd3950ed9031534accf9c7816`.

**Interpretation:** B0 can now select the A0 projection-DX kernel name, and
basic numerical correctness passes. At the 04:50 wrap-up, the 1,000-call
extension had not run; its subsequent completed result is recorded below.
A0's controlled original-ELF and A-reuse-off failures occurred at calls
**45 and 516**, so five calls provide little evidence about recurrence.
The zero-DZ layout and hash match A0, but the retained W bytes and allocator
settings differ. B0 uses the verified step55 W and empty allocator settings;
The retained A0 A-reuse-off launch environment used A0's own retained W and
`expandable_segments:True`; that allocator certificate does not independently
verify the original-ELF arm's setting.
Complete A0 argument packets and loaded/executed ELF identity have not been
matched. This test neither identifies B0's training culprit nor establishes
a cross-host fix.

Evidence: run `root_terminal_audit.json`,
`cpu_native_audit/independent_terminal_selector/{audit.md,audit.json,decoded_launches.json,snapshot_manifest.json}`,
`a0_comparison.json` and `catalog_identity_check.json` in that audit directory.
Trace SHA256: `8d60e215bc82d444aba628536cd157abd1104a2ed7cca21cd692eca5ef3dddaa`.
The comparison is also summarized in
`resume_20260920/a0_projection_fixed460_comparison_20260920.md`.

**Recorded command for the subsequent 1,000-call run.** This used the earlier
launcher. Future launches must use the corrected device-FD guard described
below; the historical `fuser` check does not establish idle GPU ownership.

```bash
B0_RESUME=/home/chcai/runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260920
python3 "$B0_RESUME/projection_dx_solution172_private_catalog_20260920/prepare_private_projection_catalog.py" \
  --verify 9440ab8f4592d53c2da3d45b998abbbfc96b65a2d40fb639ed823840bc7945df
sudo -n fuser /dev/kfd /dev/dri/renderD*
python3 "$B0_RESUME/run_native_catalog_control.py" \
  --script "$B0_RESUME/repro_projection_dx_zero_shape_cycle_b0_weight.py" \
  --label projection_dx_private172_fixed460_1000 \
  --catalog "$B0_RESUME/projection_dx_solution172_private_catalog_20260920/gfx1250" \
  --catalog-fingerprint e0dcd6ba02ad18f0d45200170f3ed944781d75bbd3950ed9031534accf9c7816 \
  --fixed-grid 460 -- \
  --weight-file /validation/debug_nan/resume_20260920/b0_projection_weight_step55_20260920/uvqk_weight.pt \
  --row-cycle 1959558 --repeats 1000 --guard-every 100
```

## Completed candidate-172 1,000-call control

`resume_20260920/projection_dx_private172_fixed460_1000_20260920T045658730547Z/`
ran **04:56:58.817665–05:03:10.825545 UTC**, exit **0**, private catalog unchanged;
root session **42100 is drained**. All **1,000 complete CPU output-bit checks**,
all **11 full W/DZ guards** and the final compact weight-file guard pass.
The exact probe is unchanged from the short selector, SHA256
`2f657c853afcf5c396593fb05d1db2e3fba28b70ec1c6e1b970364843820422e`.

The frozen independent auditor joins **1,000 native begins, successful returns,
argument packets and report calls**, with no issue. Every launch uses the exact
candidate-172 MT128x240x128 symbol, **460 workgroups × 128 threads**, at fixed
N=1,959,558. All packets are identical, including observed W/DZ/output pointers;
M512/N1959558/K2048, alpha1/beta0 and candidate-specific scheduling match the
short selector. All **402** private catalog entries match before/after.
Boot is unchanged. The kernel delta has four NIC warnings and one SVM-workqueue
CPU-duration warning, with **no GPU-fault candidate or reset**.

Evidence: run `root_terminal_audit.json`,
`cpu_native_audit/independent_terminal_1000/{audit.md,audit.json,decoded_launches.json,snapshot_manifest.json}`,
and `gpu_owners.after_task_fd_audit.json`. Trace SHA256:
`88bd1f5f86e15bc0c0a3e4ca2203ca1b105bc4880cf582af8ca56fa81c841da3`.
The terminal audit passes all nine checks. The direct post-terminal FD scan is
empty; earlier `fuser` snapshots have the limitation described next.

This extends B0 coverage to **1,000 calls of the A0 kernel name and fixed grid**.
It still uses B0's W and allocator settings, zero DZ and no preceding DB/DW
work. Loaded ELF byte identity and complete A0 packet equivalence remain
unproved. It does not reproduce A0's error or identify the B0 training fix.
The next useful control adds the actual production helper's preceding work.

## Device-ownership check corrected

While session42100 was active, **both `sudo fuser -v` and `amd-smi process`
missed the known live worker PID497987**. Direct `/proc` inspection found its
open KFD/render descriptors. Host and container nodes had identical character
device numbers (**234:0** and **226:128**) but different filesystem identities:
container `st_dev=65`, inodes11/22; host `st_dev=6`, inodes1485/1487.

The new `resume_20260920/gpu_device_fd_owners.py` scans **every process and
thread's** `/proc/.../fd` entries and matches character-device `st_rdev`.
Permission errors and changing task identities fail closed. Observed ownership
is retained conservatively when a task exits during the scan. A real CPU-only
regression showed why scanning process leaders alone is insufficient: a live
thread retained an FD after its leader exited. The final scanner finds it.

`run_native_catalog_control_device_guard.py` derives from the prior launcher,
verifies the scanner hash, executes those exact bytes through interpreter stdin,
and archives those same bytes. It refused a second launch with **GPU_BUSY/75**
while the known worker was live. Full live scans found only that root worker;
post-terminal scans found no owner. These are discrete FD observations, not a
GPU reservation or proof of continuous exclusivity. **Earlier empty fuser or
amd-smi outputs alone do not establish exclusivity**, and this finding does
not prove historical contention caused any NaN.

Final scanner SHA256:
`3b4404c59a0823df10dd7b9c124f18f58a920b6ff72692ff6c9963a0d2a99d97`.
Final launcher SHA256:
`c9a68e9ec5a0c399404d35679506f737091920479ca18cb8a95afc9fe7f505c2`.
Independent CPU review and real leader-exit regression:
`resume_20260920/gpu_device_fd_guard_cpu_review.json`.
Live integration: `resume_20260920/device_guard_live_integration.json`, plus
the 1,000-call run's `gpu_owners.live_fd_audit.json` and
`gpu_owners.live_task_fd_audit.json`. Original launchers/evidence remain unchanged.
Other older controllers, including the prepared serialized training controller,
have not been retrofitted; use external direct thread-FD checks before/after
any future GPU run rather than relying on their original fuser-only guard.

## Prepared ordered DB→DW→DX control — GPU unrun

The new standalone
`resume_20260920/repro_preprocess_projection_ordered_zero_dz.py` embeds the
exact frozen `triton_addmm_bwd` function and executes, in production order:
`DB=sum(DZ,dim=0)`, `DW=mm(X.T,DZ)`, then `DX=mm(DZ,W.T)`. `is_y_1d=True`;
no synchronization, checking or readback is inserted between these operations.
DB/DW remain alive through DX, and all outputs stay alive through readback and
input guards. It uses fixed N1,959,558, exact-zero BF16 DZ, verified B0 W and
**synthetic finite CPU X**, not original normalized training bytes.

All complete outputs are checked against exact signed zero on CPU. The first
available bad output is retained before further device reads, with a **3 GiB**
artifact cap. Full X/DZ/W guards run at checkpoints and on numerical failure.
Final observations occur after the whole helper, so a wrong result initially
localizes the sequence rather than the first corrupting writer. Trace both
native GEMMs: a process-wide fixed460 setting does not prove DW uses 460 groups.

**20 CPU tests pass**, including all65,536 BF16 encodings, exact archive/helper
identity, operation order, tensor lifetime, actual-W small zero products and
injected read/guard/producer/write failures. Root and independent source review
found no launch blocker. One nonblocking diagnostic caveat remains: a later
serialization failure can leave `.tmp`, masking the original write exception
on retry; the earlier completed artifact remains intact.

Probe SHA256: `d47097e4f5807f79fbc0990a8d25c01ce48eee16e93ecdc2e5a52969fa83d93f`.
Frozen helper segment SHA256:
`39d492a690e80134befa73fe51d62dde5e2cdd1ec3497ef29dff54aa76f2ae13`.
Evidence: `repro_preprocess_projection_ordered_zero_dz.cpu_readiness.json`,
`test_ordered_projection_zero_dz_cpu.py` and
`repro_preprocess_projection_ordered_zero_dz.independent_review.md` under resume.

**Next launch after resuming — not executed at this wrap-up:**

```bash
B0_RESUME=/home/chcai/runs/mi450_b0_0910_7ff_start0_20260918/debug_nan/resume_20260920
python3 "$B0_RESUME/run_native_catalog_control_device_guard.py" \
  --script "$B0_RESUME/repro_preprocess_projection_ordered_zero_dz.py" \
  --label projection_ordered_private172_fixed460_selector \
  --catalog "$B0_RESUME/projection_dx_solution172_private_catalog_20260920/gfx1250" \
  --catalog-fingerprint e0dcd6ba02ad18f0d45200170f3ed944781d75bbd3950ed9031534accf9c7816 \
  --fixed-grid 460 -- \
  --weight-file /validation/debug_nan/resume_20260920/b0_projection_weight_step55_20260920/uvqk_weight.pt \
  --rows 1959558 --repeats 5 --guard-every 5
```

Audit both native launches per helper call and terminal status before extending
to 1,000 ordered calls. Neither the five-call nor the longer ordered run has
been launched. All completed GPU work is drained at this wrap-up.

## Completed queued status-reduction control

`resume_20260920/reduce_status_queue_20260920T044118204208Z/` passes
**1,000 calls /40 phase readbacks**, exit0 at **04:41:20.596466 UTC**;
root session **77683 drained**. The identical500-call retained/released-input
schedules all match the independent CPU truth table. Recorded allocation
reuse is **0** in the retained arm and **493** in the released arm. The seven
empty-input calls have no allocation to reuse. This exercises the real frozen
`reduce_status`, temporary lifetime and same-stream reuse at the production
status-vector length and nearby boundaries, including disabled-bound cases.

The source and required stack preflight pass, frozen files remain unchanged,
boot matches, owners are empty, and the kernel delta contains no GPU fault
candidates. There is no failure payload. Evidence: run `state.json`,
`result/result.json`, per-phase planned/queued/result files, source manifest
and kernel snapshots. This passes the reducer-only control; the full
`try_check/watch/check` caller is not exercised, and it does not identify the
training fault's writer.

## CPU follow-up performed during E2E

A frozen-prefix comparison of this E2E attempt with the September 19
cap256/BN1 run that produced NaNs at 60/70 verifies **all 60 common production
Python files are byte-identical**. The prior 140-file archive matches its
manifest. The only changed common file is the standalone zero-DY reproducer;
three added files are tests and a forward reproducer. The sampled losses at
10, 20 and 50 match bitwise; 30/40 differ by **11/7 float32 ULPs**, respectively.
Current 60/70 are finite. This rules out a new production Python patch as the
explanation for the longer clean prefix. The different boot, compiled/native
selection and execution histories remain confounders. Recorded environment
files do not contain all inherited keys, so an absent key is not proof it was
unset. Evidence: `resume_20260920/e2e_repeat_cpu_comparison.json`, with exact
complete-line log-prefix hashes, and reusable `audit_e2e_repeat_cpu.py`.

The bounded CPU source/IR audit covers **11 files**, each matching the frozen
training archive and loaded manifest. It finds no reachable global-address
int32 overflow or premature Tensor-owner release at **2,468,813 rows** in the
inspected LN, output-LN, attention, preprocess, addmm and jagged paths. Global
row products are widened before multiplication; remaining attention products
are within a sequence, and reduction scratch uses tile count rather than full
packed rows. A dropout counter crosses signed32 but is not a memory offset.
Historical retained IR corroborates i64 bases, without identifying the binary
executed at the after-950 fault. A wrapped relative i32 offset alone would not
erase a normal high 64-bit tensor base to the observed upper32=1 address.
No source patch is justified by this negative audit. Evidence:
`resume_20260920/production_backward_address_audit_20260920.{md,json}`.

The original synthetic projection-DX control remains **CPU-ready, GPU-unrun**:
`resume_20260920/repro_projection_dx_zero_shape_cycle.py`, SHA256
`b9f34e2779a50329307120aa60eb0551a634086901684a946058b90d8e88e048`.
It cycles row counts **1,959,558; 2,468,813; 2,097,151; 2,097,153; 2,013,428**
for **1,000 total calls**, with one max-sized zero-DZ backing, allocating
outputs and unchanged bounded W. Each call checks all output magnitude bits
on CPU; eleven guards cover complete inputs, including inactive backing.
The synthetic sequence crosses the 2**32-element boundary but does not
reproduce original training data/allocation history. CPU oracle, variable-size
retention and inactive-suffix corruption fixtures pass; independent review
found no concrete bug. The original synthetic arm remains unrun; the real-B0-weight derivative was
launched after the verified E2E pass, as recorded above.

The separate caller audit, `fused_checker_caller_audit_20260920.{md,json}`,
finds no reachable reduction indexing, stream or owner-lifetime defect. Its
status vector is only **1,234,407 bytes**, scalar flags stay retained through
blocking phase readback, and this streaming loop does not use the optional
three-stream training pipeline. Frozen sources for the after-730/750 faults
contain only the earlier Torch checker. Thus this fused implementation cannot
be the common executed checker across all five episodes; a longer clean
uninstrumented prefix alone cannot establish diagnostic causality.

A reducer-only control was prepared under
`resume_20260920/reduce_status_standalone_20260920/`: **1,000 calls**, identical
500-call retained/released-input arms, **40 phase readbacks**, known status
sentinels and ordinary same-stream allocation reuse. It executes exact frozen
`reduce_status` bytes, checks expected flags on CPU, and labels regenerated
failure input as intended data rather than observed former device contents.
It does not launch the large checker or full tripwire. Probe SHA256:
`e6aef2576f4eda0d5a03b1abc482ef7d7613f44e7fb54bf3875d5afc813e283c`.
CPU syntax/truth-table/schedule/failure-payload checks pass; the subsequent
GPU control passed as recorded above.

If training hard-faults without a payload, the separate
`resume_20260920/run_serialized_diagnostic_control.py` is also **CPU-ready,
GPU-unrun**, SHA256
`684a0c165387b0a3fcdbe5593dde6bd68c7e1c12b4c9bf2b1983c382c4bb629a`.
It preserves frozen configuration and process guards, uses serialization3 and
runtime logging0, accepts `--steps` in multiples of10, and labels outcomes as
serialized diagnostics with `uninstrumented_e2e_pass=false`. Eighteen mocked
CPU tests pass. Serialization can change or suppress the failure; it does not
guarantee kernel attribution. The previous log-level3 one-step diagnostic
emitted **1,059,713,438 bytes**, so that unbounded logging is unsuitable here.

## Native catalog identity and actual-weight preparation

A read-only CPU audit copied **402 files / 25,437,158 bytes** from B0's
hipBLASLt catalog. Every filename, size and SHA256 equals the retained A0
original inventory. Decoded disk candidates 102/103 match A0's complete
DW metadata, including argument ABI. Candidate 103 matches B0's traced
MT256x128x64 name; candidate 173 matches B0's traced MT128x256x128 projection.
The candidate numbers identify disk catalog entries, not measured runtime
selection IDs or loaded ELF identities.

| Disk candidate | Role / tile | Static VGPR | Static SGPR | LDS bytes |
|---|---|---:|---:|---:|
| 102 | Alternative DW, MT32x16x32 | 252 | 93 | 8,192 |
| 103 | B0-traced DW, MT256x128x64 | 676 | 73 | 49,152 |
| 172 | A0-style projection DX, MT128x240x128 | 1,024 | 88 | 230,144 |
| 173 | B0-traced projection DX, MT128x256x128 | 1,024 | 88 | 246,528 |

All report zero spills. Extracted candidate 173 metadata decodes **all 2,000
B0 projection argument packets** from both completed fixed-shape runs. All
41 declared fields end at byte 212; the 216-byte kernarg segment adds alignment
padding. Candidates 172/173 have identical field layouts, but different tile
sizes and StreamK schedule calculations. Replacing function handles without
regenerating scheduling arguments would not be a valid comparison. Fixed grid
460 is not a demonstrated solution pin. Hardware-aware ranking could explain
the different family at equal shape, but the selector's runtime inputs and
scores were not captured. The resource counts do not establish a cause.

Evidence: `resume_20260920/native_catalog_cpu/final_audit.json`,
`native_candidate_audit_20260920.md`, complete inventory and extracted
metadata. DW ELF SHA256 is
`1c7b790600e7dffdd4caaca8c1213c5480a81314ad3c13682eeb24335b8926f3`;
projection ELF SHA256 is
`b323216507600606e076ab3546e81a931809008627c77d7f0d5a85ffa6cf1542`.
These are extracted disk objects, not proven execution bytes. Function-symbol
sizes are zero, so retained per-function intervals explicitly include padding.

The stronger DX control was **CPU-ready before its 04:33:56 UTC launch**. The separate
`repro_projection_dx_zero_shape_cycle_b0_weight.py` accepts a compact original
B0 UVQK weight from requested step55/event4, storage7. The **2 MiB BF16
[512,2048]** weight is entirely finite, maximum **0.04833984375**; decoded,
ZIP-member and raw-offset bytes agree. Exact weight SHA256:
`46c71d0f0bf75afb6f2daf85aa38517d44ea284122c14137d973a03416752b40`.
Its standalone bundle is **2,100,969 bytes**, SHA256
`257772ab3f84b0eceff0057cbe74897024318e11e1925fb3426bc274da9a6fd0`.
Original capture stat is unchanged; only the selected member was hashed,
not the complete 10.56 GB capture.

The derivative retains the five-row cycle, 1,000 total calls, complete CPU
zero oracle and eleven full input guards. It additionally rejects wrong
weight shape/dtype/nonfinite values/hash before device access, guards exact
resident W bytes, and checks the compact source file again at exit. **15 CPU
tests and the inherited self-test pass**; independent review found no blocking
issue. Probe SHA256:
`2f657c853afcf5c396593fb05d1db2e3fba28b70ec1c6e1b970364843820422e`.
Artifacts: `resume_20260920/b0_projection_weight_step55_20260920/`, including
`uvqk_weight.pt`, `provenance.json`, `readiness.json` and a description-only
launch command. The original synthetic probes remain unchanged.

This weight reduces one difference from training; zero DZ, synthetic shape
sequence, omitted preceding DB/DW work and CPU readbacks still differ. It
cannot recover missing step51 projection/LN intermediates or guarantee the
A0 kernel. It subsequently passed all1,000 calls; the earlier synthetic fixed-shape
controls remain separate observations.

## Completed controls

| Experiment | Observed terminal result | What it establishes / limits |
|---|---|---|
| Original health probe, `continue_20260919T065146Z/health_20260920T021123727786Z/` | **4 × 1,000 pass**, exit **0**; normalized kernel messages identical, boot unchanged | The previous GPU-only width-16 failure does not recur in this execution; no CPU failure mapping or recovery cause is established |
| Independent CPU health probe, `resume_20260920/health_cpu_20260920T021444Z/` | **4 × 1,000 pass**, exit **0** at **02:14:51.638 UTC** | Every source element passes before/after each arm; indices remain valid/unchanged and all output columns match CPU row-index expectations. GPU equality also passes. D2H checks alter timing |
| Weighted-LN zero-input forward, saved statistics, `continue_20260919T065146Z/saved_20260920T021509329193Z/` | **1,000 pass**, exit **0** at **02:15:24.319 UTC**; final BN8 / 4 warps / 1 stage, **42 registers**, 0 spills | All 1,000 final helper launches use `8496a96fc3e9cd4758bbee0bb7e1dd801c415f34826d9332391ed2bd7725a518`; no output failures and every checkpoint input-byte guard passes |
| Weighted-LN zero-input forward, recomputed statistics, `continue_20260919T065146Z/recompute_20260920T021627081432Z/` | **1,000 pass**, exit **0** at **02:16:40.838 UTC**; final BN8 / 2 warps / 1 stage, **125 registers**, 0 spills | All 1,000 final helper launches use `8c9ede2504f312624261ba307f49f85054da457f44bd5d524b7c06b9b588a31a`; no output failures and every checkpoint input-byte guard passes |
| Synthetic native Linear DW, `resume_20260920/dw_synthetic_bounded_20260920T021801Z/` | **1,000 pass**, **11** full input guards, exit **0** at **02:18:56.374 UTC**, runtime **53.887 s** | All original BF16 output magnitude bits pass the CPU zero oracle. Native trace matches A0 DW196's full symbol and all declared nonpointer arguments; this synthetic control does not reproduce A0's saved data/allocation history |
| Weighted-LN ordinary BN8 backward helper, `continue_20260919T065146Z/backward_20260920T022118960208Z/` | **1,000 pass**, exit **0** at **02:21:34.173 UTC** | Zero output failures; all six input-byte guards pass at all **100** checkpoints; no partial boundary check in this arm |
| Weighted-LN BN8 backward with partial-only pre-reduction check, `continue_20260919T065146Z/backward_20260920T022208523741Z/` | **1,000 pass**, exit **0** at **02:22:23.959 UTC** | Both original `[2048,512]` parameter partials are clean at all **1,000** observations; final gradients and all six input guards pass. No recurrent anomaly was available to localize |
| Weighted-LN saved forward, explicit BN8/2 warps, `continue_20260919T065146Z/saved_20260920T022239070381Z/` | **1,000 pass**, exit **0** at **02:22:50.642 UTC**; **57 registers**, 0 spills | One eligible configuration and all 1,000 launches use `579059b0bbefea19e0f7b133fa3368ef8407e8d9053411336b31244f8597425e`; all output and checkpoint input guards pass |

The root investigator confirmed the first eight sessions drained at the
**02:22:50** checkpoint. Projection DX subsequently completed; finer training
then hard-faulted without a payload as documented above. The independent file audit
`resume_20260920/completed_controls_cpu_audit.json` verifies all terminal
reports, guards and unchanged boot IDs. Kernel messages are identical for
seven runs after timestamp normalization; the DW run adds one ionic NIC
transceiver warning, with no new GPU/reset message. Both BN8 backward runs
and the explicit saved-forward arm have byte-identical raw kernel logs.

The health probe uses seed 11, **2,347,656** float32 source rows and **16,384**
GPU-generated random indices per call, widths 16/4 and both `index_select`
and `gather`, without the optional width-256 warmup. Source construction is
GPU arange → expand → contiguous. Complete before/after source checks cover
**37,562,496** elements per width-16 arm and **9,390,624** per width-4 arm.
Its CPU oracle and artifact handling passed **16** adversarial CPU checks,
recorded in `resume_20260920/gather_cpu_reference_selftest.json`.
The successful execution has no first-failure tensor payload.

Both LN controls use **2,468,813 × 512 BF16** zero X, unit weight and zero
bias. Saved mode supplies mean=0/rstd=1000 and checks exact returned values;
recompute mode checks mean=0 and rstd≈1000 with the documented tolerance.
Final output checks cover every call; complete input guards run at checkpoints.
Production forward still autotunes all eligible configurations even with
`TRITON_FULL_AUTOTUNE=0`. Only each helper's final result is checked, so these
passes do not validate every overwritten autotune-candidate output.

The initially selected configurations differ. Training may reuse the forward-selected
configuration during saved-statistics recomputation because the autotune key
omits statistics mode. The additional explicit saved BN8/2-warps pass covers
that row/warp configuration for zero inputs. It does not validate nonzero
training statistics or the unobserved producer interval.

Both backward arms select DX `eae3693fabb2660493dcae5c19ddffa4c994f39d0ec835166df62d032dea3ae2`,
**576 registers**, zero spills, HSACO SHA-256
`788c2b9991cee0cbb6955eb35359020f9d683f4f8863bdf02debd31ebd4eb5a6`.
The audit compares these against the portable helper's September 19
iteration-90 failure: compiled hash and full HSACO match. The report's
`dx_matches_known_bn8_hash=false` compares against an older `e34c0e...`
source/debug identity. The existing `resume_20260919/ln_compiler_identity/`
audit found identical executable and allocated ELF sections between that
older identity and `eae3693f...`. It is not evidence of a kernel fix.
Current passing backward arms supply no bad boundary; they cannot assign
the historical defect to partial production or reduction.

The DW control uses bounded CPU-generated X `[2324351,256]`, zero DY
`[2324351,512]`, and the allocating `torch.mm(DY.T, X).T`; all 1,000 outputs
are BF16 `[256,512]`, stride `[1,256]`. Every call's complete CPU bit oracle
accepts signed zero and rejects nonzero/subnormal/nonfinite values. Guards
cover all **3,570,203,136** input bytes after upload and every 100 calls.

The independent file audit `cpu_trace_audit.json` joins all **1,000** unique
native launch begins to their returns, each with HIP success and a preceding
successful name lookup. One earlier lookup returns 500, followed by success;
there is no failed launch. `hipExtModuleLaunchKernel` uses global threads
`[512,1,1]` and local threads `[128,1,1]`, hence **four workgroups**. One full
`MT256x128x64`/LDSB1/SK0 symbol exactly matches A0's failed DW196 record.
All 1,000 captured **172-byte** argument packets are identical; all **33**
declared fields match A0 except A/B/C/D addresses, which join this run's
original X/DY and output tensors. The audit script is preserved beside it.

This establishes the named host launch and argument match. It does not infer
B0's catalog solution index, loaded ELF bytes or static register count from
the symbol. HIP launch return is asynchronous; terminal exit and per-call CPU
readback supply the completion evidence. Passing outputs were not retained,
so the file audit checks their CPU-oracle records and frozen source semantics.
Synthetic data, allocation history, readback timing and host context differ
from the A0 positive. DW also remains distinct from projection DX.

### Native projection DX: 1,000 exact-zero calls

`resume_20260920/projection_dx_a0shape_default_20260920T022418264452Z/`
completed **1,000 calls**, exit **0** at **02:30:11.350692 UTC**, in
**351.962 s**; root drained session **31194**. The sole producer is allocating
`torch.mm(DZ, W.T)`, with synthetic finite BF16 W `[512,2048]`, exactly-zero
DZ `[1959558,2048]`, and contiguous DX `[1959558,512]`. All output magnitude
bits pass the CPU oracle, and **11** complete input-byte guards match after
upload and every 100 calls. Every call copies **2,006,587,392 output bytes**
to CPU; passing tensors are not retained.

The separate `final_native_source_audit.json` verifies **1,000 sequential
launches and 1,000 successful returns**, all with a preceding successful
symbol lookup. The report and all 1,000 stdout CPU-check records agree;
frozen source and final source hashes match. All captured **212-byte argument
packets are identical**. The ordered W/DZ/DX pointer, shape and stride joins
match M512/N1959558/K2048, alpha1/beta0. Those field interpretations use the
retained A0 UserArgs layout; metadata for this new symbol has not been
independently extracted, and pointer reuse alone cannot identify a call.

The actual kernel is **`MT128x256x128` / LDSB0 / SK3 / TDMI3**, with **256
workgroups × 128 threads**. It differs from A0's controlled failing
`MT128x240x128` / fixed460 case. Runtime preference is
`_BlasBackend.Cublaslt`; workspace, tuning-file and StreamK overrides are
unset. This validates a bounded synthetic B0 control, not the A0 retained-W
case or the unobserved B0 training interval. No solution index, executable
identity or static register count is inferred from the symbol.

Boot IDs match and saved ownership checks are empty before/after. Comparing
normalized kernel-message bodies adds only the SVM workqueue CPU-duration
warning, with no new fault/reset message; the NIC message body already
exists in the preceding snapshot. Raw kernel ISO times can shift with
conversion and do not establish ordering against launcher completion.
`terminal_cpu_audit.json` records the terminal checks, while
`early_native_source_audit.json` remains preserved separately from the final
audit. At that earlier checkpoint neither culprit attribution nor a 3,000-step pass
existed; the subsequent verified E2E pass above supersedes the latter status.

## Step-51 row fingerprint

A bounded CPU mmap read of the original step-51 capture finds one dominant
negative value in `dout` row **388482**, column **26**, with a nearly cancelling
positive tail. No training/model replay or GPU operation is involved.

| Original row measurement | Value |
|---|---:|
| Dominant column 26 | `-9.97363801182069e25` |
| Other columns | **511 positive**, minimum `1.9007525093550322e23`, maximum `2.0188116714267733e23` |
| Mean positive / absolute dominant | `0.001956970327047382`; pure-centering ratio `1/511 = 0.0019569471624266144` |
| Row sum / absolute dominant | `1.1837121212121212e-5`; compatible with BF16 rounding of a zero-sum vector |
| Positive-tail variation | Population standard deviation / mean **0.9633%**, across **11** BF16 values |

This is consistent with a single huge effective incoming gradient passed
through LN's derivative, `DX = B*(e_j - 1/D - z_j*z/D)`, where
`z=(X-mean)*rstd` and B absorbs the spike, its LN weight and rstd. It is not
an attribution: a constant-tail centering model still differs at **325/512**
BF16 coordinates, and the actual z/DY are absent. The normalized-input term
could account for the small modulation, but that explanation has not been
tested against original operands; another producer could produce this row.

Layer-2 input-LN DW/DB have **only final dense-gradient descriptors and GPU
flags**: finite=true, within `1e20`=false. Their **40,966,148-byte** shared
gradient storage is absent from all **17** retained storages. Layer-2 UVQK
weight/bias and output-stage parameter flags are bounded. These scalar flags
do not reveal original corrupt coordinates or independently prove magnitudes.

The archived `HSTUComputeOutputFunction` saves attn/u, output-norm parameters
and statistics, output weight and dropout mask, allowing reconstruction of
the projection branch. Its forward output also adds **residual x**, which is
not saved; the forward result is absent as well. Thus the actual layer-1
output (=layer-2 input-LN X) cannot be independently reconstructed from this
payload. `saved_tensors[0]` is attention output, and `parent_outputs[2]` is
backward `dout`; neither is that missing residual input.

Step 51 used the original default LN policy: the frozen source lists BN1/BN8
and labels BN8 the batch-1024 winner, with no explicit BN1 override. Actual
input-LN launch metadata was not captured. The later explicitly pinned BN1
training NaNs at 60/70 are a **separate event**; this fingerprint does not
explain them. The next original capture should retain projection DX, input-LN
DY/X/mean/rstd/weight and raw parameter gradients, so the column-26 spike
hypothesis can be checked at the producing boundary.

Evidence: `resume_20260920/step51_ln_fingerprint/report.json`, reusable
`analyze.py`, and **15,579-byte** `row388482_cpu.pt` containing three original
rows and the simple comparison model. Stored rows round-trip byte-exactly;
CUDA remains uninitialized. No complete matrix was copied or duplicated.

## Culprit evidence and next experiments

The B0 step-51 output-gradient GEMM receives already-huge finite `dout`;
its original outputs match an independent CPU reference. Its preceding
observed output backward is bounded. The unobserved interval remains the
training localization target. Earlier split/LN captures show propagation
or expected overflow from huge inputs. Attention cap256 and weighted-LN BN1
repair isolated controls but together previously yielded training NaNs at 60/70.
The latest unchanged-production-code run passed 3,000 steps; the reason for
the differing outcome remains unknown.

The newly fetched [A0 evidence](../mi450_a0/mi450_a0.md#strongest-evidence)
isolates `_additional_embedding_mlp.2` raw weight-gradient corruption from
small original inputs at attempt 405. Its isolated zero-DY native
`torch.mm(DY.T, X).T` fails at call 196; all traced calls use solution **103**,
`MT256x128x64`, with **676 static VGPRs**. Alternative solution 102 passes
two separate 1,000-call controls. This is a concrete B0 experiment lead;
neither its traced kernel nor its cause transfers retrospectively to B0's
already-huge `dout` capture. A register-count difference alone is not a cause.

1. Postfault health and the previously positive ordinary BN8 backward helper
   passed; preserve a recurrent failure before changing its execution. The
   A0-shaped synthetic projection-DX control already passed
   1,000 calls with a different tile/grid from A0's positive. The control at the
   **B0 captured shape** also passed 1,000, and the direct checker passed
   100 zero/100 sentinel calls across 32-bit boundaries. Uninstrumented E2E session **1890** subsequently passed all 3,000 steps and
   is drained. The real-B0-weight shape-cycle session **17072** also passed1,000 and
   is drained. Fixed-grid460 alone retained candidate 173 in a five-call check;
   the private catalog then selected candidate 172 and passed five calls and
   a separate **1,000-call** run. The next direct control is the **GPU-unrun
   ordered DB→DW→DX helper**, starting with five calls and native trace audit.
   These synthetic passes cannot clear original
   operands, allocation history or the unobserved B0 training interval.
2. Retain the now-completed saved BN8/2-warps and partial-only backward
   controls as passing observations, with their zero-input limitations.
   If the BN8 backward failure returns, observe its parameter partials
   before reduction with `--check-partials-before-reduction`.
   Bad partials narrow the boundary to their producer; clean partials followed
   by bad final gradients narrow the later interval. A passing intermittent
   control cannot identify a repair.
3. The finer original preprocess-stage run ended after finite sampled 950,
   without a payload. Its fault does not supply a numerical producing boundary.
   A subsequent original capture should retain projection/norm inputs and
   outputs for CPU comparison, with attention cap256 and weighted-LN BN1.
   Bad packed DZ requires checking recomputed UVQK and attention/SiLU;
   bad normalized X requires the original saved-statistics forward inputs.
   Bounded fused outputs followed by huge `dout` require inspecting residual
   accumulation or later writes. An alarm or regenerated intermediate alone
   cannot identify the corrupting writer.
4. Once original valid inputs produce an independently verified wrong output,
   reduce that stage to a positive standalone reproducer, establish a fix
   with restored failing control, and validate uninstrumented timestamp-zero
   training through **all 3,000 optimizer steps**, finite losses and no GPU
   fault. A short clean probe or instrumented run is not that validation.

**Wrap-up:** all GPU work is completed and drained. No further launch followed
the user's wrap-up request. Both the maintained worktree B0 document and the
opening of the older IDE checkout's B0 document now carry the current summary;
older local text is preserved as historical context. The user subsequently
authorized committing and pushing these B0 documents. No production code or
host-local diagnostic artifacts are included. Further commits and pushes still
require explicit user instruction.

Keep later run results separate until their authoritative report and terminal
process status are verified. Re-poll a live handle after observation timeouts;
do not restart a GPU job merely because a polling call expired.
