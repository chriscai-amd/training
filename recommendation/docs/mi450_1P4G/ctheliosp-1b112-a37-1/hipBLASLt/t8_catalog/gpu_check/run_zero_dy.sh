#!/usr/bin/env bash
# T8 numerics check, ONE shape per container/process, on one GPU, with the a37-1 guards
# (preflight, idle GPUs, live dmesg watch that kills the container on the first fault line).
# A HUMAN runs this after the hipblaslt-bench checks pass. Not run by the T8 builder.
#   run_zero_dy.sh <device 0-3> <m> <n> <k> <zero|rand> [calls=1000] [lib=rocm-10.2_nnonly_v2]
# Exit 3 = fault seen (stop everything), 2 = refused, 1 = numerical failure.
set -uo pipefail
DEV=$1 M=$2 N=$3 K=$4 MODE=$5 CALLS=${6:-1000} LIB=${7:-rocm-10.2_nnonly_v2}
ENV_DIR=/home/chcai/a37-1_env
IMAGE=${IMAGE:-recommendation-gfx1250-20260910:triton-7ff97e}
OUT=/home/chcai/t8_work/gpu_check/runs/${LIB}_g${DEV}_${M}x${N}x${K}_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)
PAT='no-retry page fault|retry page fault|MES.*fail|failed to respond|ring .* timeout|GPU reset|soft lockup|Oops|BUG:|Memory access fault|VM_L2_PROTECTION|amdgpu: .*hang'
mkdir -p "$OUT"; cd "$OUT"
"$ENV_DIR/preflight.sh" --allow-defaults > preflight.txt || { cat preflight.txt; exit 2; }
[ "$(ls /sys/class/kfd/kfd/proc | wc -l)" = 0 ] || { echo "another KFD process is running; stop"; exit 2; }
CNAME=t8chk_$(basename "$OUT")
sudo dmesg > dmesg.before.txt
( sudo dmesg -w --time-format iso | tee dmesg.live.txt | grep -a -m1 -E "$PAT" > fault.txt \
    && { echo "FAULT -> killing $CNAME" >> fault.txt; docker kill "$CNAME" >/dev/null 2>&1; } ) &
WATCH=$!
trap 'sudo pkill -f "dmesg -w --time-format iso" 2>/dev/null; kill $WATCH 2>/dev/null' EXIT
D=/armlibs/$LIB
timeout --signal=KILL 1800 docker run --rm --name "$CNAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES="$DEV" \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /home/chcai/training/arbor_libs:/armlibs:ro -v /opt/rocm-10.2:/opt/rocm-10.2:ro \
  -v /home/chcai/t8_work/gpu_check:/t8:ro \
  -e LD_PRELOAD=$D/libhipblaslt.so.1 -e HIPBLASLT_TENSILE_LIBPATH=$D/hipblaslt/library/gfx1250 \
  -e AMDGCN_USE_BUFFER_OPS=0 -e HSA_ENABLE_COREDUMP=0 -e PYTHONUNBUFFERED=1 \
  "$IMAGE" bash -c "export LD_LIBRARY_PATH=$D:\${LD_LIBRARY_PATH:-}; exec python /t8/zero_dy_loop.py --m $M --n $N --k $K --mode $MODE --calls $CALLS" \
  > check.log 2>&1
RC=$?
sleep 1; sudo dmesg > dmesg.after.txt
tail -3 check.log
if [ -s fault.txt ]; then echo "GPU FAULT:"; cat fault.txt; exit 3; fi
exit $RC
