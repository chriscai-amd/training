#!/usr/bin/env bash
# Evidence capture for the MES team (2026-10-10 request: dmesg + amdgpu_mes_event_log,
# driver loaded with mes_log_enable=1 dyndbg=+plmf and GPU recovery NOT disabled).
#   fault_capture.sh <out_dir> <tag> [delay_s]
# Read-only: dmesg, debugfs reads (MES event log, MES rings, fence info), sysfs.
# Never writes to debugfs, never triggers a reset. Called by run_ab.sh /
# gemm_watch.sh at run start (tag=baseline) and on the first fault line
# (tag=fault_t0, then tag=fault_t30 after the driver's recovery attempt).
set -u
OUT=$1 TAG=$2 DELAY=${3:-0}
[ "$DELAY" -gt 0 ] && sleep "$DELAY"
D="$OUT/mes_capture_$TAG"; mkdir -p "$D"
date -u +%FT%T.%3NZ > "$D/captured_utc.txt"
sudo dmesg --time-format iso > "$D/dmesg.txt" 2>&1
for p in 0001:04:00.0 0002:04:00.0 0003:04:00.0 0004:04:00.0; do
  dd=/sys/kernel/debug/dri/$p; s=${p//:/-}
  sudo test -e "$dd/amdgpu_mes_event_log" && sudo cat "$dd/amdgpu_mes_event_log" > "$D/${s}_amdgpu_mes_event_log.bin" 2>"$D/${s}_mes_event_log.err"
  [ "$TAG" = baseline ] && continue
  sudo cat "$dd/amdgpu_fence_info" > "$D/${s}_amdgpu_fence_info.txt" 2>/dev/null
  for r in $(sudo ls "$dd" 2>/dev/null | grep -E '^amdgpu_ring_mes_'); do sudo cat "$dd/$r" > "$D/${s}_$r.bin" 2>/dev/null; done
done
{ for d in /sys/bus/pci/devices/000{1,2,3,4}:04:00.0; do echo "$(basename $d) busy=$(cat $d/gpu_busy_percent 2>/dev/null)% vram_used_MiB=$(( $(cat $d/mem_info_vram_used 2>/dev/null || echo 0) / 1048576 ))"; done
  for k in noretry gpu_recovery ip_block_mask mes_log_enable cwsr_enable; do echo "param $k=$(cat /sys/module/amdgpu/parameters/$k 2>/dev/null)"; done
  echo "kfd_procs: $(ls /sys/class/kfd/kfd/proc 2>/dev/null | tr '\n' ' ')"
  echo "D-state:"; ps -eo stat,pid,etime,comm | awk '$1 ~ /D/'; } > "$D/gpu_state.txt"
ls "$D" | grep -q mes_event_log.bin || echo "WARN: no amdgpu_mes_event_log in debugfs (mes_log_enable=$(cat /sys/module/amdgpu/parameters/mes_log_enable))" >> "$D/gpu_state.txt"
echo "$D"
