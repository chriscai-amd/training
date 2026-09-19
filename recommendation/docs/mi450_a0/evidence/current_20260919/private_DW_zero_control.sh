#!/usr/bin/env bash
set -eu
sudo -n python3 /home/chcai/mi450_logs/root_cause_20260919/assert_gpu_idle.py
sha256sum -c /home/chcai/mi450_logs/root_cause_20260919/step405_linear_private_DW_frozen_sources_v1.sha256
docker exec -w /workspace/recommendation \
 -e AMDGCN_USE_BUFFER_OPS=0 -e TRITON_FULL_AUTOTUNE=0 -e TRITON_ALLOW_PIPELINING=0 \
 -e HIPBLASLT_TENSILE_LIBPATH=/data/mlperf_dlrm_v4/root_cause_20260919/dw_solution102_private_catalog_v1/gfx1250 -e HIPBLASLT_WORKSPACE_SIZE=1 -e HSTU_BWD_MAX_VGPR=256 -e WEIGHTED_LN_BWD_BLOCK_N=1 \
 triton-7ff97e-20260910 env -u PYTORCH_ALLOC_CONF -u CUBLASLT_WORKSPACE_SIZE -u TENSILE_STREAMK_DATA_PARALLEL -u TENSILE_STREAMK_FIXED_GRID -u TENSILE_SOLUTION_INDEX -u HIPBLASLT_TUNING_OVERRIDE_FILE \
 LD_PRELOAD=/data/mlperf_dlrm_v4/root_cause_20260919/trace_projection_hip_launches_v2.so \
 PROJECTION_HIP_LAUNCH_TRACE=/data/mlperf_dlrm_v4/root_cause_20260919/step405_linear_single_dw_zero_private_catalog_workspace1_argtrace1000_v1_hip.jsonl \
 python -B -u /data/mlperf_dlrm_v4/root_cause_20260919/run_single_DW_private_catalog_control_root_v1.py
