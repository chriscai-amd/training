# MI450 A0 hang triage, 2026-09-18

**Update, 06:38 UTC:** a separate replay of the full step-101 NaN capture
reproduced a GPU/MES hang at 06:25:50. The original step-211 investigation below
is retained. See [the replay report](nan_replay_20260918.md#step-101-capture-and-replay-hang)
and `/home/chcai/mi450_logs/nan_replay_long_20260918_0520/replay_hang_0630/`.
The new replay waited in ROCr `InterruptSignal::WaitRelaxed` with the driver halt
flag already zero. After SIGTERM, GPU-memory teardown remained blocked in
AMDGPU/MES TLB/fence polling. Its triggering kernel and any connection to NaN
remain unproven.

Later status at **06:57 UTC**: the replay had exited 143 after SIGTERM and VRAM
had drained to 0.2 GB. GFX/power still read 100%/~858 W, and no post-hang compute
check had completed. The original step-211 and replay-hang records below are
historical observations; current next steps are in the replay report.

The existing START_TS=0, batch-1024 run is stuck in AMDGPU/MES TLB
invalidation. This investigation inspected the existing step-211 failure;
it did not produce an independent training repeat or reproduce a new NaN.

## Verified stack

- Image: `recommendation-gfx1250-20260910:triton-7ff97e`.
- Container: `triton-7ff97e-20260910`.
- Imported Triton: `/opt/triton-custom/python/triton/__init__.py`.
- Distribution: `3.8.0+git7ff97e31`.
- Git HEAD: `7ff97e310935b4a79794878dbc911f9af25d38d9`.
- Torch: `2.11.0+rocm7.14.0a20260625`; HIP: `7.14.60850`.
- Loaded amdgpu: `7.1.1.31300009`; DKMS: `7.1.1-2397345.24.04`.
- Required image workaround:
  `HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`.
- Keep `AMDGCN_USE_BUFFER_OPS=0`, the existing BLOCK_N=64 attention workaround,
  and existing pipeline clamp for the baseline.

The base image alone ships the wrong Triton, 3.6.0. A bare `3.8.0` version
string also does not establish this source pin.

## Evidence and attribution

Host PID **187832** is container PID **996**. Host PID 996 is an unrelated
kernel migration thread: container PID numbers must be translated using NSpid.
At inspection the target had SIGKILL pending (`ShdPnd=0x102`) and was running.
The archived training log ends at step 211, loss 0.13985688984394073, with no
non-finite loss. It had already been stopped before this investigation.

Reading process command lines/maps can block: ordinary `ps` blocked in
`__access_remote_vm`, and targeted perf recording blocked in `m_start`.
System-wide sampling with event synthesis disabled succeeded:

```bash
sudo perf record -a --synth no -e cpu-clock:k -F 49 -g \
  -o kernel.data -- sleep 5
sudo perf script -i kernel.data -F pid,tid,cpu,event,ip,sym,dso
```

All **245 target-thread samples** contain this call chain:

```text
schedule / __schedule
mes_v12_1_submit_pkt_and_poll_completion.constprop.0
mes_v12_1_inv_tlbs_pasid
gmc_v12_1_flush_gpu_tlb_pasid
amdgpu_gmc_flush_gpu_tlb_pasid
amdgpu_vm_flush_compute_tlb
amdgpu_gem_va_update_vm
amdgpu_gem_va_ioctl
drm_ioctl_kernel -> drm_ioctl -> amdgpu_drm_ioctl
__x64_sys_ioctl -> do_syscall_64
```

The installed DKMS source at
`/usr/src/amdgpu-7.1.1-2397345.24.04/amd/amdgpu/mes_v12_1.c:251`
contains an explicit `while (halt_if_hws_hang) schedule();` after a MES
completion timeout. `/sys/module/amdgpu/parameters/halt_if_hws_hang` was **1**.
This is strong evidence for the mechanism making the task unkillable.
The earlier claim that absence of recent MES errors excluded a MES hang is
superseded. KFD ioctl `0xc0184b0c` is WAIT_EVENTS, but those log lines were not
evidence of the blocked thread's current ioctl.

## Recovery attempt and current limitation

Changed the writable debug flag from **1 to 0** at runtime. No persistent
configuration file or driver binary was changed. The driver default in source
is 0. The task left the schedule loop, processed the pending kill, and entered
`exit_mm` / `exit_mmap` / KFD queue destruction. Teardown still encounters MES
TLB waits. Subsequent dmesg contains explicit `REMOVE_QUEUE`,
`INVALIDATE_TLBS`, and `MISC (WAIT_REG_MEM)` completion timeouts.

A fresh invocation of `scripts/probe_stack.py` failed to complete initialization:
it initially waited in `kfd_create_process -> __flush_workqueue`, while a KFD
cleanup worker was blocked in MES/GART TLB handling. Later it advanced into
runtime initialization but still encountered MES waits. SIGTERM was sent to
this diagnostic process (host PID 223354). No fresh training was launched on
the unhealthy device. A shared-host reboot was requested, not performed.

## What is and is not located

The immediate hang belongs with **AMDGPU/MES firmware and GPU virtual-memory
management**, and the debug halt flag explains the otherwise unkillable loop.
Disabling that flag is not a fix for the underlying completion timeout.
The triggering GPU operation, firmware defect, hardware condition, or preceding
kernel corruption is not yet isolated. This does not exonerate Triton as a
possible trigger and does not establish a shared cause with the early NaN.

After recovery, verify the exact pin and pass `probe_stack.py` before training.
Repeat at START_TS=0, batch 1024, scale 0.25, per-step loss logging, eval off,
with unique RUN_NAME values and sufficient windows to pass step 211. Do not use
the runner's default SMOKE mode without explicit overrides: it sets START_TS=150.
Keep `halt_if_hws_hang=0` and collect dmesg continuously before a repeat so the
first timeout is retained. A useful next A/B is expandable allocator segments
enabled versus disabled, because the observed stack is GPU VA mapping/TLB
invalidation; this is a hypothesis, not a demonstrated allocator defect.
If NaNs recur, use the module/parameter probes to separate a bad forward result
from embedding weights corrupted during the preceding fused backward update.

## Artifacts

Evidence directory: `/home/chcai/mi450_logs/triage_20260918_0322/`.

- `kernel.data`: successful system-wide kernel sample capture.
- `kernel_stacks.txt`, `target_stacks.txt`: decoded samples and target subset.
- `status_before.txt`, `stat_before.txt`: target before releasing debug halt.
- `dmesg_after_release.txt`: timeout evidence after releasing the loop.
- `after_release.data`: follow-up sampling during teardown.
- `perf.data`: unsuccessful targeted capture; do not use as evidence.

Original run evidence:
`/home/chcai/mi450_logs/hang_ts0_7ff97e_r1_20260918_025004/`.
