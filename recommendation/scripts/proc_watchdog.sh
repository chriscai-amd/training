#!/usr/bin/env bash
# Kill the training run if the container's process count explodes.
#
# On 2026-09-17 a PYTHONPATH re-entrancy bug in scripts/autoload/sitecustomize.py
# spawned 14,131 offload-arch processes, which drove amdgpu into a kernel-level
# SVM hang (svm_migrate_vram_to_ram), wedged the container past `docker stop`,
# and cost an AC power cycle. The bug is fixed; this is the backstop, because the
# failure is silent -- no log output at all -- and only visible as a process count.
C="${CONTAINER:-triton-7ff97e-test}"
LIMIT="${LIMIT:-300}"
while true; do
  n=$(timeout 20 docker exec "$C" sh -c 'ls /proc | grep -c "^[0-9]"' 2>/dev/null) || { sleep 30; continue; }
  if [ "${n:-0}" -gt "$LIMIT" ]; then
    echo "WATCHDOG: $n procs > $LIMIT -- killing run at $(date -Is)"
    timeout 30 docker exec "$C" sh -c 'pkill -9 -f train_ranker; pkill -9 -f offload-arch' 2>/dev/null
    pkill -9 -f run_prod_convergence 2>/dev/null
    exit 1
  fi
  sleep 30
done
