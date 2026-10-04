#!/usr/bin/env bash
# E2 route (b), HUMAN-RUN ONLY: HIPBLASLT_LOG_MASK=32 capture of the PATCHED
# testbed's GEMM calls from a synthetic-input yambda DlrmHSTU step
# (scripts/e2_synthetic_capture.py), then capture_to_yaml.py -> union.
#
#   GPU=2 OUT=/home/chcai/runs/e2_synth_$(date -u +%Y%m%dT%H%M%SZ) \
#     bash scripts/e2_synthetic_capture.sh
#
# Then review $OUT/union.txt and $OUT/capture_table.txt. If it holds the six
# TN wgrad keys and none of the six NT keys, copy it to
# arbor_hipblaslt/recapture_union.txt and name $OUT in the DONE line.
#
# Safety, same as validator/bench.sh: the pack's host_hazard.sh preflight
# (driver options, no HOST_HAZARD lock) and a live kernel-log watch that kills
# the container on a fault signature and writes the lock. GPU 0 is refused.
# It runs on the image's STOCK hipBLASLt: no ARM_LIB_DIR and no 10.2 tall-K
# kernels. Call keys do not depend on the library.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK="${PACK:-/home/chcai/Arbor/packs/mi450/hipblaslt}"
IMAGE="${IMAGE:-recommendation-gfx1250-20260910:triton-7ff97e}"
GPU="${GPU:?set GPU (not 0)}"
OUT="${OUT:?set OUT (an empty host dir)}"
STEPS="${STEPS:-2}"
S=/opt/venv/lib/python3.12/site-packages

fail() { echo "[e2] FAIL: $*" >&2; exit 1; }
[ "$GPU" != "0" ] || fail "GPU 0 is out of service (03:18 MES fault)"
case "$REPO" in /home/chcai/training|/home/chcai/training/*) fail "canonical tree";; esac
BENCH_OUTPUT="$OUT/capture_meta.json"   # host_hazard.sh names it in the lock
# shellcheck source=/dev/null
source "$PACK/validator/host_hazard.sh"
hazard_preflight
mkdir -p "$OUT"; [ -z "$(ls -A "$OUT")" ] || fail "$OUT not empty"

CNAME="arbor_e2_synth_$$"
hazard_watch "$CNAME" & WATCH=$!
trap 'kill $WATCH 2>/dev/null || true' EXIT
set +e
docker run --rm --name "$CNAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -e ROCR_VISIBLE_DEVICES="$GPU" -e PYTORCH_CUDA_ALLOC_CONF= -e PYTORCH_ALLOC_CONF= \
  -e HIPBLASLT_TENSILE_LIBPATH=$S/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250 \
  -e HIPBLASLT_LOG_MASK=32 -e HIPBLASLT_LOG_FILE=/out/hipblaslt_bench_%i.log \
  -e HSTU_HAMMER_KERNEL=TRITON -e AMDGCN_USE_BUFFER_OPS=0 -e HSTU_BWD_MAX_VGPR=0 \
  -e TRITON_FULL_AUTOTUNE=0 -e HSTU_BWD_BLOCK_N=64 \
  -e STEPS="$STEPS" -e BATCH_SIZE="${BATCH_SIZE:-1024}" \
  -e HSTU_TN_WGRAD="${HSTU_TN_WGRAD:-all}" \
  -v "$REPO":/workspace/recommendation:ro -v "$OUT":/out \
  -w /workspace/recommendation "$IMAGE" \
  python3 scripts/e2_synthetic_capture.py --out /out/summary.json \
  > "$OUT/container.log" 2>&1
RC=$?
set -e
kill $WATCH 2>/dev/null || true
hazard_check "e2_synthetic_capture"
[ "$RC" = 0 ] || fail "driver exited $RC, see $OUT/container.log"

# capture_to_yaml.py is protected pack data: run a COPY with STEPS/RANKS fixed.
sed -e "s/^STEPS, RANKS = 12, 4$/STEPS, RANKS = $STEPS, 1/" \
    "$PACK/eval/capture_to_yaml.py" > "$OUT/capture_to_yaml_copy.py"
grep -q "^STEPS, RANKS = $STEPS, 1$" "$OUT/capture_to_yaml_copy.py" || fail "STEPS/RANKS edit did not apply"
python3 "$OUT/capture_to_yaml_copy.py" "$OUT" "$OUT/groups.json" "$OUT/union.txt" \
  | tee "$OUT/capture_table.txt"
{
  echo "{\"testbed_head\": \"$(git -C "$REPO" rev-parse HEAD)\","
  echo " \"testbed_dirty\": $(git -C "$REPO" status --porcelain | wc -l),"
  echo " \"gpu\": $GPU, \"steps\": $STEPS, \"image\": \"$IMAGE\","
  echo " \"time_utc\": \"$(date -u +%FT%TZ)\", \"host\": \"$(hostname)\"}"
} > "$OUT/capture_meta.json"
echo "[e2] done: $OUT/union.txt (review before copying to arbor_hipblaslt/recapture_union.txt)"
