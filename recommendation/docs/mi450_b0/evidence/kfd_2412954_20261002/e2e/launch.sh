#!/usr/bin/env bash
set -uo pipefail
cd /home/chcai/training/recommendation
RUN_DIR=/home/chcai/runs/kfd2412954_e2e_start0_20261002T033543Z
export CONTAINER=mi450-b0-0910-7ff-start0
export RUN_NAME=b1024_kfd2412954_start0_20261002T033543Z
export SMOKE=1
export SMOKE_BATCHES=3000
export MASTER_PORT=29919

# Trailing Docker env arguments override the runner's smoke defaults.
# Batches are uncapped per window; DIE_AT_STEP bounds cumulative updates.
# The intentional exit at global step 3000 is logged and has worker code 42.
scripts/run_prod_convergence_1gpu.sh "$RUN_DIR/train.log" \
  -e START_TS=0 -e NUM_TRAIN_TS=299 -e NUM_TRAIN_BATCHES=0 \
  -e SEED=1 -e CKPT_PATH= -e CKPT_STEP_FREQ=0 \
  -e IN_WINDOW_CKPT_FREQ=0 -e CKPT_TIME_INTERVAL_S=0 \
  -e DIE_AT_STEP=3000 \
  -e EVAL_EVERY_N_WINDOWS=0 -e EVAL_EVERY_DATA_PCT=0 \
  -e SKIP_EVAL_EPOCH_PCT=0 \
  -e PYTORCH_CUDA_ALLOC_CONF= -e PYTORCH_ALLOC_CONF= \
  -e AMD_SERIALIZE_KERNEL=0 -e DEBUG_NAN_HOOKS=0 \
  -e TRITON_FULL_AUTOTUNE=0 -e TRITON_ALLOW_PIPELINING=0 \
  -e HISTORY_LENGTH=4086 -e MIN_HISTORY=4086 -e MAX_SEQ_LEN=4096 \
  -e HSTU_NUM_LAYERS=3 -e HISTORY_STRATEGY=interleaved \
  -e STREAMING_SHUFFLE_FRACTION=0 -e STREAMING_SHUFFLE_SEED=0 \
  -e TRAIN_SPLIT_PERCENTAGE=1.0 -e HSTU_HAMMER_KERNEL=TRITON
rc=$?
printf '%s\n' "$rc" > "$RUN_DIR/launcher_exit_code.txt"
exit "$rc"
