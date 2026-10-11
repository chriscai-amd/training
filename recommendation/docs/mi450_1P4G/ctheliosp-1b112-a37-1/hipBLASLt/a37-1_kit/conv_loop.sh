#!/usr/bin/env bash
# Repeat the a37-1 convergence run (fixed CWSR driver robustness test) until told to stop.
#   conv_loop.sh [first_run_dir_to_wait_for]
# Each run: M1+T8+M5 (m5_split_t.gin) + HSTU_BWD_BLOCK_N=128, a37-2 AUC-0.75 recipe
# (LR 5e-7, warmup 24k, holdout window 299 every 560 steps after the skip threshold).
# A run PASSES when MLPerf logs run_stop status=success with eval_accuracy >= 0.75, no GPU
# fault line and no non-finite train_loss. The loop stops at the first run that does not pass.
# Stop it any time:  touch /home/chcai/a37-1_env/runs/STOP_CONV_LOOP   (checked between runs)
set -u
ENV_DIR=/home/chcai/a37-1_env
STOPF=$ENV_DIR/runs/STOP_CONV_LOOP
LOG=$ENV_DIR/runs/conv_loop.log
verdict() {   # $1 = run dir; prints a one-line verdict, returns 0 on PASS
  local d=$1 m=$1/mlperf.log
  local acc stop nonfin fault
  acc=$(grep -a '"key": "eval_accuracy"' "$m" 2>/dev/null | grep -o -E '"value": [0-9.eE+-]+' | awk '{print $2}' | tr '\n' ' ')
  stop=$(grep -a '"key": "run_stop"' "$m" 2>/dev/null | grep -o -E '"status": "[a-z]+"' | tail -1)
  nonfin=$(grep -a '"key": "train_loss"' "$m" 2>/dev/null | grep -c -i -E '"value": (nan|inf|-inf|null)')
  fault=$([ -s "$d/fault.txt" ] && head -1 "$d/fault.txt" || echo none)
  local last=$(grep -a -o -E 'Step [0-9]+ perf' "$d/train.log" 2>/dev/null | tail -1)
  echo "$(basename $d) | evals: ${acc:-none} | run_stop: ${stop:-none} | nonfinite losses: $nonfin | fault: $fault | last: ${last:-none}"
  [ "$fault" = none ] && [ "$nonfin" = 0 ] && echo "$stop" | grep -q success && \
    awk -v a="$acc" 'BEGIN{n=split(a,x," "); exit !(n>0 && x[n]>=0.75)}'
}
run_one() {
  STEPS=-1 TIMEOUT_S=36000 "$ENV_DIR/run_ab.sh" "CONV_M5_BN128" m5_split_t.gin \
    -e HSTU_BWD_BLOCK_N=128 -e DENSE_LR=5e-7 -e SPARSE_LR=5e-7 \
    -e EVAL_EVERY_DATA_PCT=0.001 -e EVAL_EVERY_N_WINDOWS=0 \
    -e SKIP_EVAL_EPOCH_PCT=0.022361844716419856 -e EVAL_HOLDOUT_TS=299 \
    -e EVAL_HOLDOUT_NUM_WINDOWS=1 -e AUC_THRESHOLD=0.75 \
    -e WEIGHTED_LN_BWD_BLOCK_N=0 -e NAN_TRIPWIRE=0 > "$ENV_DIR/runs/conv_last.console" 2>&1
}
first=${1:-}
if [ -n "$first" ]; then
  echo "$(date -u +%FT%TZ) waiting for $first" >> "$LOG"
  while [ ! -s "$first/exit-status.txt" ]; do sleep 60; done
  sleep 30
  v=$(verdict "$first"); ok=$?
  echo "$(date -u +%FT%TZ) $v -> $([ $ok = 0 ] && echo PASS || echo STOP)" >> "$LOG"
  [ $ok = 0 ] || exit 1
fi
i=1
while true; do
  [ -e "$STOPF" ] && { echo "$(date -u +%FT%TZ) stop file found, loop ends" >> "$LOG"; exit 0; }
  "$ENV_DIR/preflight.sh" > /dev/null || { echo "$(date -u +%FT%TZ) preflight FAILED, loop ends" >> "$LOG"; exit 2; }
  echo "$(date -u +%FT%TZ) repeat $i starting" >> "$LOG"
  run_one
  d=$(ls -td "$ENV_DIR"/runs/CONV_M5_BN128_* | head -1)
  sleep 30
  v=$(verdict "$d"); ok=$?
  echo "$(date -u +%FT%TZ) repeat $i: $v -> $([ $ok = 0 ] && echo PASS || echo STOP)" >> "$LOG"
  [ $ok = 0 ] || exit 1
  i=$((i + 1))
done
