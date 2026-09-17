#!/usr/bin/env bash
# Wait for any in-flight training run to release the GPU, then run the bounded
# bs=1024 production-shape smoke.
#
# The smoke exists because BATCH_SIZE=1024 has never run on this node: the
# memory fit is an estimate with ~1.5x spread on the activation term, and
# _add_embeddings_bwd_kernel's autotune key contains AUTOTUNE_B =
# prev_power_of_2(B) (triton_position.py), so B=1024 selects a config that no
# run has executed. 20 minutes here beats finding out hours into a multi-day run.
#
# Only ONE training process may hold this GPU at a time: a concurrent run
# changes occupancy and can wedge the node.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$HOME/mi450_logs}"
CONTAINER="${CONTAINER:-triton-7ff97e-test}"
mkdir -p "$OUT"

# Match the ACTUAL trainer, not anything whose command line merely contains the
# string. A bare `pgrep -f train_ranker` also matches monitor shells, log tails,
# and this script's own `bash -c`, so the wait never terminates. Anchoring on the
# launcher prefix (`docker exec ...` on the host, `python -m ...` inside) is the
# difference between waiting for the GPU and waiting forever.
trainer_running() { pgrep -f '^(docker exec|python -m).*train_ranker' > /dev/null; }

echo "=== waiting for GPU to free $(date -Is) ==="
while trainer_running; do sleep 20; done
sleep 10
docker exec "$CONTAINER" pkill -f train_ranker 2>/dev/null || true
sleep 5

echo "=== GPU free; starting bs=1024 production-shape SMOKE $(date -Is) ==="
SMOKE=1 SMOKE_BATCHES="${SMOKE_BATCHES:-200}" \
  "$REPO_ROOT/scripts/run_prod_convergence_1gpu.sh" "$OUT/prod_b1024_smoke.log"
