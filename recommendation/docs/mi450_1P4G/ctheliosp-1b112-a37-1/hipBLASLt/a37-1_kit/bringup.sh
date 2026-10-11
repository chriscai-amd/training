#!/usr/bin/env bash
# Restore the a37-1 working driver state after a boot. Safe by construction:
# it only LOADS amdgpu when it is not loaded (amdgpu is blacklisted at boot),
# and never unloads, reinstalls or resets anything. If amdgpu is already
# loaded with other options or another build, it stops and reports; fixing
# that needs an unload (oops => panic) and is a manual, agreed decision.
set -eu
EXP_SRCVERSION=892370C5AB08A22771F1DC5
OPTS="ip_block_mask=0xcff mes_log_enable=1 dyndbg=+plmf"   # MES team request: no gpu_recovery=0; noretry default
here=$(dirname "$(readlink -f "$0")")

if [ -d /sys/module/amdgpu ]; then
  echo "amdgpu already loaded; not touching it."
  exec "$here/preflight.sh"
fi
disk_sv=$(modinfo -F srcversion amdgpu)
if [ "$disk_sv" != "$EXP_SRCVERSION" ]; then
  echo "FAIL: on-disk amdgpu srcversion $disk_sv != $EXP_SRCVERSION; someone installed another build. Not loading."
  echo "      dkms: $(dkms status amdgpu | tr '\n' ' ')"
  echo "      Known: 2413386 / 2415424 fail to load on VBIOS 900A (PSP ID_LOAD_TOC, modprobe soft lockup)."
  exit 1
fi
echo "loading amdgpu $OPTS"
sudo modprobe amdgpu $OPTS
sudo systemctl start docker
sleep 5
exec "$here/preflight.sh"
