#!/usr/bin/env bash
set -eu
sudo -n python3 /home/chcai/mi450_logs/root_cause_20260919/assert_gpu_idle.py
sha256sum -c /home/chcai/mi450_logs/root_cause_20260919/step405_linear_single_dw_frozen_sources_v1.sha256
docker exec -w /workspace/recommendation \
 -e AMDGCN_USE_BUFFER_OPS=0 -e TRITON_FULL_AUTOTUNE=0 -e TRITON_ALLOW_PIPELINING=0 \
 -e HIPBLASLT_WORKSPACE_SIZE=1 -e HSTU_BWD_MAX_VGPR=256 -e WEIGHTED_LN_BWD_BLOCK_N=1 \
 triton-7ff97e-20260910 env -u PYTORCH_ALLOC_CONF -u CUBLASLT_WORKSPACE_SIZE -u TENSILE_STREAMK_DATA_PARALLEL -u TENSILE_STREAMK_FIXED_GRID \
 LD_PRELOAD=/data/mlperf_dlrm_v4/root_cause_20260919/trace_projection_hip_launches_v2.so \
 PROJECTION_HIP_LAUNCH_TRACE=/data/mlperf_dlrm_v4/root_cause_20260919/step405_linear_single_dw_zero_workspace1_argtrace1000_v1_hip.jsonl \
 python -B -u scripts/repro_linear_dw_saved_orientation.py \
 /data/mlperf_dlrm_v4/root_cause_20260919/step405_linear_saved_operands_root_v1.pt \
 --context /data/mlperf_dlrm_v4/root_cause_20260919/train_linear_projection_diagnostics_cap256_bn1_workspace1_start0_1000_v1/context.json \
 --report /data/mlperf_dlrm_v4/root_cause_20260919/step405_linear_single_dw_zero_workspace1_argtrace1000_v1.json \
 --failure-dump /data/mlperf_dlrm_v4/root_cause_20260919/step405_linear_single_dw_zero_workspace1_argtrace1000_v1_failure.pt \
 --gpu --repeats 1000
