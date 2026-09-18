#!/usr/bin/env bash
set -uo pipefail
run_dir=/home/chcai/mi450_logs/nan_replay_post_ac_20260918_0730
docker exec -w /workspace/recommendation \
 -e NAN_BACKWARD_TARGET=_stu_layers.1. -e NAN_BACKWARD_ABS_THRESHOLD=1e20 \
 -e NAN_BACKWARD_SAVE_ALL=1 \
 triton-7ff97e-20260910 python -u scripts/replay_training_step.py \
 /data/mlperf_dlrm_v4/nan_replay_long_20260918_0520/capture_step_000101 \
 --mode current --repeats 2 --verify-every-repeat \
 --backward-probe-dir /data/mlperf_dlrm_v4/nan_replay_post_ac_20260918_0730/boundaries_saved \
 --report /data/mlperf_dlrm_v4/nan_replay_post_ac_20260918_0730/saved_operations_report.json \
 > "$run_dir/replay_saved.log" 2>&1
replay_rc=$?
printf 'REPLAY_EXIT=%s\n' "$replay_rc" >> "$run_dir/replay_saved.log"
exit "$replay_rc"
