#!/usr/bin/env bash
# Read-only environment check for a37-1 GPU work. Run before every GPU job
# (and from launchers); exits non-zero if the host drifted from the baseline.
# Other users can swap the driver / reload amdgpu / reflash at any time, so
# never assume the last check still holds.
# Usage: preflight.sh [--allow-defaults]   (allow amdgpu default module options)
set -u
EXP_SRCVERSION=5E97B61A2D67C92B973CF3D      # amdgpu-dkms 2416495 (CWSR handler FIXED: gfx12.1 array 3bc22f60, verified 2026-10-11)
EXP_VBIOS=113-M4500001-900A
EXP_MES=0x0000007e
EXP_PARAMS="noretry=-1 gpu_recovery=-1 ip_block_mask=3327 mes_log_enable=1"   # 0xcff; defaults for noretry/recovery + MES log per MES team (2026-10-11)
fail=0; warn() { echo "WARN: $*"; }; bad() { echo "FAIL: $*"; fail=1; }

if [ ! -d /sys/module/amdgpu ]; then bad "amdgpu not loaded (run bringup.sh)"; echo "PREFLIGHT FAIL"; exit 1; fi
sv=$(cat /sys/module/amdgpu/srcversion)
[ "$sv" = "$EXP_SRCVERSION" ] || bad "amdgpu srcversion $sv != $EXP_SRCVERSION (driver changed: $(modinfo -F version amdgpu 2>/dev/null), dkms: $(dkms status amdgpu 2>/dev/null | tr '\n' ' '))"
for kv in $EXP_PARAMS; do
  k=${kv%%=*}; v=${kv#*=}; cur=$(cat /sys/module/amdgpu/parameters/$k 2>/dev/null)
  if [ "$cur" != "$v" ]; then
    if [ "${1:-}" = "--allow-defaults" ]; then warn "amdgpu $k=$cur (want $v)"; else bad "amdgpu $k=$cur (want $v; see /etc/modprobe.d/amdgpu-a37.conf)"; fi
  fi
done
n=0
for d in /sys/class/drm/card*/device; do
  vb=$(cat $d/vbios_version 2>/dev/null) || continue; [ -n "$vb" ] || continue
  n=$((n+1)); mes=$(cat $d/fw_version/mes_fw_version 2>/dev/null)
  [ "$vb" = "$EXP_VBIOS" ] || bad "$(basename $(readlink -f $d)) VBIOS $vb != $EXP_VBIOS (firmware reflashed)"
  [ "$mes" = "$EXP_MES" ] || bad "$(basename $(readlink -f $d)) MES fw $mes != $EXP_MES"
done
[ "$n" = 4 ] || bad "found $n GPUs, want 4"
procs=$(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)
[ "$procs" = 0 ] || bad "$procs KFD process(es) already on the GPUs: $(for p in /sys/class/kfd/kfd/proc/*; do ps -o user=,pid=,comm= -p $(basename $p); done | tr '\n' ';')"
others=$(who | awk '$1!="'"$USER"'"{print $1}' | sort -u | tr '\n' ' ')
[ -z "$others" ] || warn "other users logged in: $others"
[ "${AMDGCN_USE_BUFFER_OPS:-0}" = 0 ] || bad "AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS} (must be 0 on this host)"
recent=$(sudo -n dmesg 2>/dev/null | grep -c -E 'no-retry page fault|MES.*failed to respond|soft lockup|Oops|amdgpu.*GPU reset')
[ "$recent" = 0 ] || warn "$recent GPU fault/MES/lockup lines in dmesg this boot; inspect before running"
echo "boot $(uptime -s), driver $sv, params $(for kv in $EXP_PARAMS; do k=${kv%%=*}; printf '%s=%s ' $k $(cat /sys/module/amdgpu/parameters/$k); done)"
[ $fail = 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAIL"
exit $fail
