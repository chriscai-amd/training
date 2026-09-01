#!/usr/bin/env bash
# Single-GPU e2e smoke on amdprimus/amdprimus:0815-derived image with shrunk
# embedding tables (EMBEDDING_ROW_SCALE). Requires the prepared yambda-5b cache
# under $DLRM_DATA_PATH (Option A layout).
set -euo pipefail

IMG="${IMG:-recommendation-amdprimus0815:gfx1250}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLRM_DATA_PATH="${DLRM_DATA_PATH:-/home/chcai/dlrm_data}"
LOG="${LOG:-$REPO_ROOT/results/smoke_shrink_1gpu/train.log}"
mkdir -p "$(dirname "$LOG")"

GID_RENDER=$(getent group render | cut -d: -f3)
GID_VIDEO=$(getent group video | cut -d: -f3)

# ~140 GiB embedding weights at 0.25; leaves headroom on 432 GiB HBM.
export EMBEDDING_ROW_SCALE="${EMBEDDING_ROW_SCALE:-0.25}"
export HBM_CAP_GB="${HBM_CAP_GB:-400}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT:-29501}"
export WORLD_SIZE=1
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# Must match Option A cache dir hstu_cache_L4086/
export HISTORY_LENGTH="${HISTORY_LENGTH:-4086}"
export MIN_HISTORY="${MIN_HISTORY:-64}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
export HSTU_NUM_LAYERS="${HSTU_NUM_LAYERS:-3}"
export HSTU_HAMMER_KERNEL="${HSTU_HAMMER_KERNEL:-TRITON}"
export START_TS="${START_TS:-150}"
export NUM_TRAIN_TS="${NUM_TRAIN_TS:-1}"
export NUM_TRAIN_BATCHES="${NUM_TRAIN_BATCHES:-5}"
export NUM_EVAL_BATCHES="${NUM_EVAL_BATCHES:-2}"
export EVAL_EVERY_N_WINDOWS="${EVAL_EVERY_N_WINDOWS:-1}"
export EVAL_EVERY_DATA_PCT="${EVAL_EVERY_DATA_PCT:-0}"
export AUC_THRESHOLD="${AUC_THRESHOLD:-1.0}"
export RUN_NAME="${RUN_NAME:-smoke_shrink_1gpu}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# gfx1250: the AMD backend's buffer-op pass group hangs the GPU in the jagged
# split/concat kernels on tensors above its 2 GiB narrowing cutoff. Disabling it
# makes the reproducer pass; see docs/mi450.md. Set to 1 to reproduce the hang.
export AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"
# A GPU memory-access fault writes a coredump the size of allocated VRAM (9-36 GB
# observed). HSA_ENABLE_COREDUMP=0 does NOT suppress it on this runtime, so also
# steer the file out of the repo -- the runtime's own message points at
# HSA_COREDUMP_PATTERN. Both are set: the first in case a newer runtime honours
# it, the second so the dump never lands in the working tree either way.
export HSA_ENABLE_COREDUMP="${HSA_ENABLE_COREDUMP:-0}"
export HSA_COREDUMP_PATTERN="${HSA_COREDUMP_PATTERN:-/tmp/gpucore.%p.gpu}"
export DLRM_DATA_PATH
export PYTHONPATH="/workspace/recommendation:${PYTHONPATH:-}"

echo "[smoke] image=$IMG scale=$EMBEDDING_ROW_SCALE batch=$BATCH_SIZE data=$DLRM_DATA_PATH"
echo "[smoke] kernel=$HSTU_HAMMER_KERNEL AMDGCN_USE_BUFFER_OPS=$AMDGCN_USE_BUFFER_OPS"
echo "[smoke] log=$LOG"

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$GID_VIDEO" --group-add "$GID_RENDER" \
  --ipc=host --security-opt seccomp=unconfined \
  --shm-size=64g \
  -v "$REPO_ROOT:/workspace/recommendation" \
  -v "$DLRM_DATA_PATH:/data/mlperf_dlrm_v4" \
  -e DLRM_DATA_PATH=/data/mlperf_dlrm_v4 \
  -e EMBEDDING_ROW_SCALE -e HBM_CAP_GB \
  -e GPUS_PER_NODE -e NNODES -e NODE_RANK -e MASTER_ADDR -e MASTER_PORT -e WORLD_SIZE \
  -e BATCH_SIZE -e NUM_WORKERS -e PREFETCH_FACTOR \
  -e HISTORY_LENGTH -e MIN_HISTORY -e MAX_SEQ_LEN -e HSTU_NUM_LAYERS \
  -e HSTU_HAMMER_KERNEL -e START_TS -e NUM_TRAIN_TS \
  -e NUM_TRAIN_BATCHES -e NUM_EVAL_BATCHES \
  -e EVAL_EVERY_N_WINDOWS -e EVAL_EVERY_DATA_PCT -e AUC_THRESHOLD \
  -e RUN_NAME -e NCCL_SOCKET_IFNAME -e PYTORCH_CUDA_ALLOC_CONF -e PYTHONPATH \
  -e AMDGCN_USE_BUFFER_OPS -e DEBUG_NAN_HOOKS \
  -e HSA_ENABLE_COREDUMP -e HSA_COREDUMP_PATTERN \
  -e AMD_SERIALIZE_KERNEL -e HIP_LAUNCH_BLOCKING -e TRITON_PRINT_AUTOTUNING \
  -e AMD_LOG_LEVEL \
  -w /workspace/recommendation \
  "$IMG" \
  python -m generative_recommenders.dlrm_v4.train.train_ranker \
      --dataset yambda-5b \
      --mode streaming-train-eval \
  2>&1 | tee "$LOG"
