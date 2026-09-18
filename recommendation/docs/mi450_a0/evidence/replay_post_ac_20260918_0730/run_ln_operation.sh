#!/usr/bin/env bash
set -uo pipefail
run_dir=/home/chcai/mi450_logs/nan_replay_post_ac_20260918_0730
docker exec -w /workspace/recommendation triton-7ff97e-20260910 \
 python -B -u scripts/replay_backward_boundary.py \
 /data/mlperf_dlrm_v4/nan_replay_post_ac_20260918_0730/boundaries_saved/boundary-2946-1789717407684763312-current-r0-s101-0004-triton_weighted_layer_norm_bwd-finite.pt \
 --gpu --repeats 2 --rows 0:16 \
 --context /data/mlperf_dlrm_v4/nan_replay_long_20260918_0520/capture_step_000101/context.json \
 --report /data/mlperf_dlrm_v4/nan_replay_post_ac_20260918_0730/ln_operation_replay.json \
 > "$run_dir/ln_operation_replay.log" 2>&1
replay_rc=$?
printf 'REPLAY_EXIT=%s\n' "$replay_rc" >> "$run_dir/ln_operation_replay.log"
exit "$replay_rc"
