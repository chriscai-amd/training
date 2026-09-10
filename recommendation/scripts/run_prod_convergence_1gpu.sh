#!/usr/bin/env bash
# Production yambda_5b.gin convergence run at local BATCH_SIZE=1024 on ONE MI450,
# targeting the gin's own AUC_THRESHOLD default of 0.75.
#
# THE ONLY INTENDED MODEL DEVIATION IS EMBEDDING_ROW_SCALE.
#   Unscaled tables are 293,051,712 rows x 2048 B = 560.04 GiB (dim 512, fp32,
#   plus 1.09 GiB row-wise Adagrad state) = 1.30x the 432 GiB device. Production
#   only fits by sharding across 8-32 ranks. At 0.25: 73,262,927 rows = 140.01
#   GiB, which is the "~140 GB" figure in docs/mi450.md.
#   This DOES change the model, not just its footprint: the vocabulary is
#   quartered and ids are modulo-wrapped into it (configs.py:817-827,
#   datasets/yambda.py:1003). Accuracy is therefore NOT directly comparable to
#   the MLPerf RCP reference, which converges to AUC 0.7636-0.7875 at 62-80M
#   samples on FULL tables. No one has measured convergence at 0.25.
#
# EVERYTHING ELSE IS LEFT UNSET SO THE GIN SUPPLIES THE PRODUCTION DEFAULT:
#   NUM_WORKERS=4      PREFETCH_FACTOR=8   HISTORY_LENGTH=4086  MIN_HISTORY=4086
#   MAX_SEQ_LEN=4096   HSTU_NUM_LAYERS=3   HSTU_HAMMER_KERNEL=TRITON
#   START_TS=0         NUM_TRAIN_TS=299    NUM_TRAIN_BATCHES=0 (full windows)
#   NUM_EVAL_BATCHES=0 EVAL_EVERY_N_WINDOWS=0  EVAL_EVERY_DATA_PCT=0.001
#   AUC_THRESHOLD=0.75 HBM_CAP_GB=260      LR_WARMUP_STEPS=24000
#   DENSE_LR=SPARSE_LR=1e-6  GRAD_CLIP_NORM=1.0  SEED=1  METRIC_LOG_FREQ=50
#
#   HSTU_NUM_LAYERS=3 IS production depth. The 5 at configs.py:140 and 12 at
#   dlrm_hstu.py:83 are dead constructor defaults the gin binding overrides.
#
# WORLD_SIZE IS DERIVED (train_ranker.py: NNODES*GPUS_PER_NODE) and any env value
# is ignored. One GPU therefore means GLOBAL batch 1024, not the 8192 that the
# gin's tuned defaults (LR_WARMUP_STEPS, SKIP_EVAL_EPOCH_PCT) were derived for.
# The RCP recipe scales peak LR linearly with GBS (1e-6/2e-6/4e-6 at
# 8192/16384/32768), so the stock 1e-6 is arguably 8x high for GBS 1024. It is
# left at the production default deliberately; override $DENSE_LR/$SPARSE_LR if
# the loss misbehaves after warmup.
#
# NON-TRAJECTORY OPERATIONAL SETTINGS (these do not change what the model learns):
#   AMDGCN_USE_BUFFER_OPS=0  MANDATORY. Buffer ops wedge the node (defect [1] in
#                            docs/mi450.md) and cost a reboot.
#   SKIP_EVAL_EPOCH_PCT      yambda_5b.gin:757 states outright that the 0.025
#                            default is the GBS=8192 value and is "NOT correct
#                            for other GBS. Set this per run alongside
#                            BATCH_SIZE." At GBS 1024 the default resolves to
#                            step ~55,929, suppressing eval through the region
#                            where the 0.75 crossing is expected. The gin says
#                            this knob "never changes the training trajectory,
#                            only how often the model is measured."
#   CKPT_*                   A 40-90 h run on a node with live defects must be
#                            resumable. Each checkpoint carries the 140 GiB
#                            table, so KEEP_LAST_N is small to bound disk.
#
# EXPECTED COST: ~50,000-65,000 steps, 40-90 h wall clock, from the RCP reference
# converging at 62-80M samples. Confirm with SMOKE=1 before committing to it.
#
# Usage:
#   SMOKE=1 scripts/run_prod_convergence_1gpu.sh <log>   # bounded shape/throughput check
#   scripts/run_prod_convergence_1gpu.sh <log>           # full convergence run
#
# Runs inside the EXISTING container (default triton-7ff97e-test), which carries
# the Triton 7ff97e3109 build docs/mi450.md pins. A fresh `docker run` of the
# base image would get a different Triton.
set -uo pipefail

LOG=${1:?usage: run_prod_convergence_1gpu.sh <logfile> [extra docker -e flags...]}
shift || true
mkdir -p "$(dirname "$LOG")"

CONTAINER="${CONTAINER:-triton-7ff97e-test}"
RUN_NAME="${RUN_NAME:-conv_b1024_s025}"
PORT="${MASTER_PORT:-$((29800 + RANDOM % 150))}"

EXTRA=()
if [ "${SMOKE:-0}" = "1" ]; then
  # Bounded validation only, NOT a convergence attempt: does bs=1024 fit in HBM,
  # what is the real throughput at the production sequence shape, and does the
  # B=1024 autotune entry (triton_position.py AUTOTUNE_B) compile and run. That
  # entry is distinct from B=8/B=128 and has never been executed here.
  EXTRA+=( -e NUM_TRAIN_BATCHES="${SMOKE_BATCHES:-200}" -e NUM_EVAL_BATCHES=8 )
  EXTRA+=( -e START_TS=150 -e NUM_TRAIN_TS=1 )
  EXTRA+=( -e EVAL_EVERY_N_WINDOWS=1 -e EVAL_EVERY_DATA_PCT=0 -e AUC_THRESHOLD=1.0 )
  EXTRA+=( -e PROGRESS_EVERY=10 -e METRIC_LOG_FREQ=10 )
else
  EXTRA+=( -e CKPT_PATH="/data/mlperf_dlrm_v4/ckpt_${RUN_NAME}" )
  EXTRA+=( -e CKPT_STEP_FREQ="${CKPT_STEP_FREQ:-5000}" -e KEEP_LAST_N="${KEEP_LAST_N:-2}" )
  EXTRA+=( -e SKIP_EVAL_EPOCH_PCT="${SKIP_EVAL_EPOCH_PCT:-0.005}" )
  EXTRA+=( -e PROGRESS_EVERY=100 )
fi

echo "[prod-b1024] ${SMOKE:+SMOKE }container=$CONTAINER run=$RUN_NAME port=$PORT"
echo "[prod-b1024] scale=${EMBEDDING_ROW_SCALE:-0.25} batch=1024 log=$LOG"
echo "[prod-b1024] start $(date -Is)"

docker exec -w /workspace/recommendation \
  -e AMDGCN_USE_BUFFER_OPS=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HSA_ENABLE_COREDUMP=0 -e HSA_COREDUMP_PATTERN=/tmp/gpucore.%p.gpu \
  -e EMBEDDING_ROW_SCALE="${EMBEDDING_ROW_SCALE:-0.25}" \
  -e BATCH_SIZE=1024 \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 \
  -e MASTER_ADDR=localhost -e MASTER_PORT="$PORT" -e NCCL_SOCKET_IFNAME=lo \
  -e RUN_NAME="$RUN_NAME" \
  "${EXTRA[@]}" "$@" \
  "$CONTAINER" python -m generative_recommenders.dlrm_v4.train.train_ranker \
    --dataset yambda-5b --mode streaming-train-eval > "$LOG" 2>&1
rc=$?

echo "[prod-b1024] EXIT=$rc $(date -Is) -> $LOG"
if grep -qiE "out of memory|OutOfMemoryError" "$LOG" 2>/dev/null; then
  echo "!! OOM at bs=1024: $(grep -ioE '.{0,90}out of memory.{0,90}' "$LOG" | head -1)"
elif grep -qE 'train_loss=nan|"key": "train_loss", "value": NaN' "$LOG" 2>/dev/null; then
  # Match the LOSS being non-finite, not the substring "nan" anywhere in the log.
  # A bare `grep -i nan` also matches "[nan-hook]", "nan_tripwire", and any path
  # containing those, and reports a clean run as a defect. (Same trap as matching
  # "inf" and hitting every "INFO" line.)
  echo "!! NAN: $(grep -nE 'train_loss=nan' "$LOG" | head -1 | cut -c1-200)"
elif grep -qE "^Traceback|RuntimeError|Process [0-9]+ terminated" "$LOG" 2>/dev/null; then
  echo "!! CRASH: $(grep -E "RuntimeError|Error:" "$LOG" | head -1 | cut -c1-220)"
elif grep -q "run_stop" "$LOG" 2>/dev/null; then
  echo "-- completed: $(grep -oE 'Step [0-9]+ train_loss=[0-9.]+' "$LOG" | tail -1)"
  grep -oE "metric/(lifetime|window)_auc[^ ]*: [0-9.]+" "$LOG" | tail -2
else
  echo "?? did not complete: $(tail -3 "$LOG" | tr '\n' ' ' | cut -c1-300)"
fi
exit $rc
