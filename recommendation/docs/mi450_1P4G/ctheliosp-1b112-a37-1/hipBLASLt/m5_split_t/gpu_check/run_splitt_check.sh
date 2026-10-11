#!/usr/bin/env bash
# M5 numerics check (splitt_loop.py), ONE shape x s per container/process, on one GPU, with the a37-1
# guards: preflight, no other KFD process, live dmesg watch (polling, as run_ab.sh) that kills the
# container at the first fault line and captures MES evidence. A HUMAN runs this after the
# gemm_watch.sh single-GEMM stage passed. NOT run by the M5 builder.
#   run_splitt_check.sh <device 0-3> <s103|s102|mix> <G20|G19|G25|G29|G31|G27> <s> <T> <zero|rand> [calls=1000] [pattern=iid|drift|halves|users]
# The expected kernel (sol -> macro tile) comes from the lib dir's m5_splits.json.
# Exit 3 = fault seen (stop everything), 2 = refused / wrong kernel, 1 = numerical failure.
set -uo pipefail
DEV=$1 VAR=$2 SHAPE=$3 S=$4 T=$5 MODE=$6 CALLS=${7:-1000} PATTERN=${8:-iid}
case $VAR in s103|s102|mix) ;; *) echo "variant must be s103|s102|mix"; exit 2;; esac
ENV_DIR=/home/chcai/a37-1_env
ROOT=${ROOT:-/home/chcai/training/recommendation}
IMAGE=${IMAGE:-recommendation-gfx1250-20260910:triton-7ff97e}
LIB=rocm-10.2_nnonly_v3/$VAR
OUT=/home/chcai/m5_work/gpu_check/runs/v3${VAR}_g${DEV}_${SHAPE}_s${S}_T${T}_${MODE}${PATTERN:+_$PATTERN}_$(date -u +%Y%m%dT%H%M%SZ)
PAT='no-retry page fault|retry page fault|MES.*fail|failed to respond|ring .* timeout|GPU reset|soft lockup|Oops|BUG:|Memory access fault|VM_L2_PROTECTION|amdgpu: .*hang'
NOISE='kfd_ioctl:|gpuvm_|kfd_mem_detach|amdgpu_ih_process|was emitted|svm_range_from_addr|Interrupt not for Node'
mkdir -p "$OUT"; cd "$OUT"
"$ENV_DIR/preflight.sh" --allow-defaults > preflight.txt || { cat preflight.txt; exit 2; }
[ "$(ls /sys/class/kfd/kfd/proc | wc -l)" = 0 ] || { echo "another KFD process is running; stop"; exit 2; }
git -C "$ROOT" rev-parse HEAD > repo-commit.txt; git -C "$ROOT" diff > repo-diff.txt
CNAME=m5chk_$(basename "$OUT")
sudo dmesg > dmesg.before.txt
"$ENV_DIR/fault_capture.sh" "$OUT" baseline >/dev/null
sudo dmesg -w --time-format iso | grep -a --line-buffered -v -E "$NOISE" > dmesg.live.txt &
( while sleep 1; do
    grep -a -m1 -E "$PAT" dmesg.live.txt > fault.txt 2>/dev/null && break
  done
  "$ENV_DIR/fault_capture.sh" "$OUT" fault_t0 >/dev/null
  setsid "$ENV_DIR/fault_capture.sh" "$OUT" fault_t30 30 >/dev/null 2>&1 < /dev/null &
  echo "FAULT -> killing $CNAME" >> fault.txt; docker kill "$CNAME" >/dev/null 2>&1 ) &
WATCH=$!
trap 'sudo pkill -f "dmesg -w --time-format iso" 2>/dev/null; kill $WATCH 2>/dev/null' EXIT
D=/armlibs/$LIB
timeout --signal=KILL 1800 docker run --rm --name "$CNAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES="$DEV" \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /home/chcai/training/arbor_libs:/armlibs:ro -v /opt/rocm-10.2:/opt/rocm-10.2:ro \
  -v "$ROOT":/workspace/recommendation:ro -v /home/chcai/m5_work/gpu_check:/m5chk:ro \
  -e LD_PRELOAD=$D/libhipblaslt.so.1 -e HIPBLASLT_TENSILE_LIBPATH=$D/hipblaslt/library/gfx1250 \
  -e AMDGCN_USE_BUFFER_OPS=0 -e HSA_ENABLE_LIGHTWEIGHT_COREDUMP=1 -e HIPBLASLT_LOG_MASK=32 -e HIPBLASLT_LOG_FILE=/tmp/hbl_%i.log -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -w /workspace/recommendation \
  "$IMAGE" bash -c "export LD_LIBRARY_PATH=$D:\${LD_LIBRARY_PATH:-}; exec python /m5chk/splitt_loop.py --shape $SHAPE --s $S --t $T --mode $MODE --pattern $PATTERN --calls $CALLS" \
  > check.log 2>&1
RC=$?
sleep 2; sudo dmesg > dmesg.after.txt
tail -4 check.log
if [ -s fault.txt ]; then echo "GPU FAULT:"; cat fault.txt; exit 3; fi
exit $RC
