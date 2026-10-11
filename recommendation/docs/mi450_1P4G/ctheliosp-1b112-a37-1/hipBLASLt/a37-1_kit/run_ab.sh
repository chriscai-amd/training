#!/usr/bin/env bash
# One e2e A/B arm on a37-1 (docs/mi450_perf_opt.md method, a37-1 safety rules).
#   run_ab.sh <arm_name> <overlay.gin basename|path> [extra docker -e/-v args...]
# 200 steps from START_TS=0, local batch 1024 (global 4096), seed 1, eval off,
# METRIC_LOG_FREQ=1, OUTPUT_TRACE=0, HSTU_BWD_BLOCK_N=64, buffer ops off.
# A live kernel-log watch kills the container at the first GPU-fault line.
set -uo pipefail
ARM=$1 OVL=$2; shift 2
ENV_DIR=/home/chcai/a37-1_env
ROOT=${ROOT:-/home/chcai/training_opt/recommendation}
LIBS=${LIBS:-/home/chcai/training_opt/arbor_libs}
DATA=/home/chcai/dlrm_data
IMAGE=${IMAGE:-recommendation-gfx1250-20260910:triton-7ff97e}
STEPS=${STEPS:-200}
TIMEOUT_S=${TIMEOUT_S:-3600}
GIN_DIR=/workspace/recommendation/generative_recommenders/dlrm_v4/train/gin/mi450_hipblaslt
[[ $OVL == /* ]] || OVL=$GIN_DIR/$OVL
RUN_NAME=${ARM}_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=$ENV_DIR/runs/$RUN_NAME
PAT='no-retry page fault|retry page fault|MES.*fail|failed to respond|ring .* timeout|GPU reset|soft lockup|Oops|BUG:|Memory access fault|VM_L2_PROTECTION'
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"

"$ENV_DIR/preflight.sh" --allow-defaults > preflight.txt || { cat preflight.txt; exit 2; }
git -C "$ROOT" rev-parse HEAD > repo-commit.txt
git -C "$ROOT" status --porcelain > repo-status.txt
docker image inspect "$IMAGE" --format '{{.Id}}' > image-id.txt || exit 1
cat /proc/sys/kernel/random/boot_id > boot-id.txt
echo "$OVL" > overlay.txt
cp "/home/chcai/mi450_perf_20260926/collective-socket.env" collective.env
sudo dmesg > dmesg.before.txt
CNAME=mi450ab_$RUN_NAME
# Live fault watch: first matching kernel line kills the container.
# On a fault: MES event log + dmesg at once, then again 30 s later (after the
# driver's queue-reset / recovery attempt), detached so it outlives this script.
# The live log drops the per-ioctl / per-mapping / IH dyndbg chatter (4.3 GB in 7 min
# otherwise); the full ring buffer is still saved by fault_capture.sh at the fault.
# The watch polls that file once a second (a pipe-based grep -m1 waited on dmesg -w
# and never fired on 10-11).
NOISE='kfd_ioctl:|gpuvm_|kfd_mem_detach|amdgpu_ih_process|was emitted|svm_range_from_addr|Interrupt not for Node'
"$ENV_DIR/fault_capture.sh" "$RUN_DIR" baseline >/dev/null
sudo dmesg -w --time-format iso | grep -a --line-buffered -v -E "$NOISE" > dmesg.live.txt &
( while sleep 1; do
    grep -a -m1 -E "$PAT" dmesg.live.txt > fault.txt 2>/dev/null && break
  done
  "$ENV_DIR/fault_capture.sh" "$RUN_DIR" fault_t0 >/dev/null
  setsid "$ENV_DIR/fault_capture.sh" "$RUN_DIR" fault_t30 30 >/dev/null 2>&1 < /dev/null &
  echo "FAULT -> killing $CNAME" >> fault.txt; docker kill "$CNAME" >/dev/null 2>&1 ) &
WATCH=$!
trap 'sudo pkill -f "dmesg -w --time-format iso" 2>/dev/null; kill $WATCH 2>/dev/null' EXIT

STOCK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
timeout --signal=KILL "$TIMEOUT_S" docker run --rm --name "$CNAME" \
  --network=bridge --ipc=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v "$ROOT":/workspace/recommendation \
  -v "$DATA":/data/mlperf_dlrm_v4 \
  -v "$LIBS":/armlibs:ro -v /opt/rocm-10.2:/opt/rocm-10.2:ro \
  -v "$RUN_DIR":/evidence \
  --env-file "$RUN_DIR/collective.env" \
  -e PYTORCH_ALLOC_CONF= -e PYTORCH_CUDA_ALLOC_CONF= \
  -e HIPBLASLT_TENSILE_LIBPATH="$STOCK" \
  -e AMDGCN_USE_BUFFER_OPS=0 -e TRITON_FULL_AUTOTUNE=0 \
  -e HSA_ENABLE_LIGHTWEIGHT_COREDUMP=1 -e HSA_COREDUMP_PATTERN=/evidence/gpucore.%p \
  -e HSTU_HAMMER_KERNEL=TRITON -e HSTU_BWD_MAX_VGPR=0 -e HSTU_BWD_BLOCK_N=64 \
  -e EMBEDDING_ROW_SCALE=1.0 -e PYTHONUNBUFFERED=1 \
  -e DLRM_DATA_PATH=/data/mlperf_dlrm_v4 \
  -e GPUS_PER_NODE=4 -e NNODES=1 -e NODE_RANK=0 \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29500 \
  -e SEED=1 -e START_TS=0 -e BATCH_SIZE=1024 -e DIE_AT_STEP="$STEPS" \
  -e EVAL_EVERY_N_WINDOWS=0 -e EVAL_EVERY_DATA_PCT=0 \
  -e METRIC_LOG_FREQ=1 -e OUTPUT_TRACE="${OUTPUT_TRACE:-0}" \
  -e RUN_NAME="$RUN_NAME" \
  -e MLPERF_LOGGING=1 -e MLPERF_LOG_PATH=/evidence/mlperf.log -e MLPERF_SUBMISSION_PLATFORM=MI450 \
  -e DLRM_GIN_OVERLAY="$OVL" \
  "$@" \
  -w /workspace/recommendation \
  "$IMAGE" \
  bash -c 'env | sort > /evidence/environment.txt; exec python -m generative_recommenders.dlrm_v4.train.train_ranker --dataset yambda-5b --mode streaming-train-eval' \
  > train.log 2>&1
RC=$?
echo "docker/timeout exit: $RC" > exit-status.txt
sudo dmesg > dmesg.after.txt
python3 "$ENV_DIR/analyze.py" . | tee perf.txt
if [ -s fault.txt ]; then echo "GPU FAULT during $RUN_NAME:"; cat fault.txt; exit 3; fi
grep -q "Step $STEPS " train.log || grep -q "Step $((STEPS-1)) " train.log || { echo "run did not reach step $STEPS (rc=$RC)"; exit 4; }
exit 0
