#!/usr/bin/env bash
set -eu
sudo -n python3 /home/chcai/mi450_logs/root_cause_20260919/assert_gpu_idle.py
sha256sum -c /home/chcai/mi450_logs/root_cause_20260919/linear_projection_private_DW_training_frozen_sources_v1.sha256
docker exec -w /workspace/recommendation \
  -e HIPBLASLT_TENSILE_LIBPATH=/data/mlperf_dlrm_v4/root_cause_20260919/dw_solution102_private_catalog_v1/gfx1250 \
  -e AMDGCN_USE_BUFFER_OPS=0 \
  -e HIPBLASLT_WORKSPACE_SIZE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e WEIGHTED_LN_BWD_BLOCK_N=1 -e HSTU_BWD_MAX_VGPR=256 -e HSTU_HAMMER_KERNEL=TRITON \
  -e TRITON_FULL_AUTOTUNE=0 -e TRITON_ALLOW_PIPELINING=0 \
  -e HSA_ENABLE_COREDUMP=0 -e EMBEDDING_ROW_SCALE=0.25 \
  -e BATCH_SIZE=1024 -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 \
  -e MASTER_ADDR=localhost -e MASTER_PORT=29888 -e NCCL_SOCKET_IFNAME=lo \
  -e START_TS=0 -e NUM_TRAIN_TS=299 -e NUM_TRAIN_BATCHES=0 \
  -e CKPT_PATH= -e EVAL_EVERY_N_WINDOWS=0 -e EVAL_EVERY_DATA_PCT=0 \
  -e NAN_BACKWARD_MAX_CAPTURE_GIB=64 -e GRAD_CLIP_NORM=1.0 \
  -e SEED=1 -e RUN_NAME=train_linear_projection_diagnostics_cap256_bn1_workspace1_privateDW_start0_1000_v1 \
  triton-7ff97e-20260910 env -u CUBLASLT_WORKSPACE_SIZE -u TENSILE_STREAMK_DATA_PARALLEL -u TENSILE_STREAMK_FIXED_GRID -u PYTORCH_ALLOC_CONF -u TENSILE_SOLUTION_INDEX -u HIPBLASLT_TUNING_OVERRIDE_FILE \
  LD_PRELOAD=/data/mlperf_dlrm_v4/root_cause_20260919/trace_projection_hip_launches_v2.so \
  PROJECTION_HIP_LAUNCH_TRACE=/data/mlperf_dlrm_v4/root_cause_20260919/train_linear_projection_diagnostics_cap256_bn1_workspace1_privateDW_start0_1000_v1_hip.jsonl \
  python -B -u scripts/capture_linear_projection_solution102_diagnostics.py \
  --directory /data/mlperf_dlrm_v4/root_cause_20260919/train_linear_projection_diagnostics_cap256_bn1_workspace1_privateDW_start0_1000_v1 \
  --steps 1000 --save-finite-step 1000 --abs-threshold 1e6
sha256sum -c /home/chcai/mi450_logs/root_cause_20260919/linear_projection_private_DW_training_frozen_sources_v1.sha256
