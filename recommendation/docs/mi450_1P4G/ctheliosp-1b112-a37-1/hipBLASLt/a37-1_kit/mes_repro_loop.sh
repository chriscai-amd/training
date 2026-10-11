#!/usr/bin/env bash
# Repeat the configuration that faulted GPU 1 on 2026-10-10 (run M1c: M1 lib swap,
# BLOCK_N=64, local batch 1024, 200 steps) until the first GPU fault, for the MES
# team's dmesg + amdgpu_mes_event_log request. Stops at the first fault or failure.
#   mes_repro_loop.sh [max_runs=8] [overlay=m1_lib.gin]   (m0_off.gin = stock control)
# Needs the post-power-cycle driver state (strict preflight: mes_log_enable=1,
# gpu_recovery default). Each fault leaves <run>/mes_capture_{baseline,fault_t0,fault_t30}.
set -u
MAX=${1:-8} OVL=${2:-m1_lib.gin}
ENV_DIR=/home/chcai/a37-1_env
"$ENV_DIR/preflight.sh" || { echo "strict preflight failed: power-cycle + /etc/modprobe.d/amdgpu-a37.conf first"; exit 2; }
for i in $(seq 1 "$MAX"); do
  echo "== repro run $i/$MAX ($OVL) $(date -u +%FT%TZ)"
  "$ENV_DIR/run_ab.sh" "mesrepro${i}_${OVL%.gin}" "$OVL"; rc=$?
  case $rc in
    0) continue ;;
    3) echo "FAULT reproduced in run $i -- evidence: $(ls -td $ENV_DIR/runs/mesrepro${i}_* | head -1)"; exit 3 ;;
    *) echo "run $i ended rc=$rc (not a GPU fault) -- stopping for inspection"; exit $rc ;;
  esac
done
echo "no fault in $MAX runs"
