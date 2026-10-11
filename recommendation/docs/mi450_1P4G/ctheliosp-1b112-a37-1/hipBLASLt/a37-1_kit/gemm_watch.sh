#!/usr/bin/env bash
# Watched single-GEMM runner (a37-1 rules: one GEMM per process, idle GPU,
# stop at the first fault line before the next GEMM).
# Usage: gemm_watch.sh <lib_dir> <device> <out_dir> <yaml> [line numbers...]
#   lib_dir : dir with hipblaslt/library/gfx1250 (e.g. arbor_libs/rocm-10.2_nnonly_v1);
#             "stock10.2" = /opt/rocm-10.2/lib unmodified (NOT allowed: full NT pool)
#   yaml    : hipblaslt-bench YAML, one "- { function: matmul ... }" per line
#   T=<n>   : env, replaces 2097152 in the line (default keep)
# Exit 3 = fault seen (stop everything), 2 = preflight refused.
set -u
LIB=$1 DEV=$2 OUT=$3 YAML=$4; shift 4
BENCH=/opt/rocm-10.2/bin/hipblaslt-bench
here=$(dirname "$(readlink -f "$0")")
PAT='no-retry page fault|retry page fault|MES.*fail|failed to respond|ring .* timeout|GPU reset|soft lockup|Oops|BUG:|Memory access fault|VM_L2_PROTECTION|amdgpu: .*hang'
[ "$LIB" = stock10.2 ] && { echo "refusing: unfiltered 10.2 NT pool"; exit 2; }
[ -f "$LIB/hipblaslt/library/gfx1250/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib" ] || { echo "no NT catalog in $LIB"; exit 2; }
mkdir -p "$OUT"
"$here/preflight.sh" --allow-defaults > "$OUT/preflight.txt" || { cat "$OUT/preflight.txt"; exit 2; }
lines=("$@"); [ ${#lines[@]} -gt 0 ] || lines=($(seq 1 $(wc -l < "$YAML")))
for n in "${lines[@]}"; do
  [ "$(ls /sys/class/kfd/kfd/proc | wc -l)" = 0 ] || { echo "another KFD process appeared; stop"; exit 2; }
  y="$OUT/g$(printf %02d $n).yaml"
  sed -n "${n}p" "$YAML" | sed -e 's/iters: 0 }/iters: 20 }/' -e "s/cold_iters: 0/cold_iters: 3, print_kernel_info: true/" \
      ${T:+-e "s/2097152/$T/g"} > "$y"
  before=$(sudo dmesg | wc -l)
  HIPBLASLT_TENSILE_LIBPATH="$LIB/hipblaslt/library/gfx1250" LD_LIBRARY_PATH=/opt/rocm-10.2/lib \
    timeout --signal=KILL 300 "$BENCH" --yaml "$y" --device "$DEV" --print_kernel_info > "${y%.yaml}.log" 2>&1
  rc=$?
  sleep 1
  sudo dmesg | tail -n +$((before + 1)) > "${y%.yaml}.dmesg"
  sol=$(grep -a -m1 -o -E 'Solution index:? *[0-9]+|solution_index[^0-9]*[0-9]+' "${y%.yaml}.log" | grep -o -E '[0-9]+$')
  us=$(awk -F, '/^ *[NT],[NT],/{print $NF}' "${y%.yaml}.log" | tail -1)
  printf "line %2d rc=%d sol=%s us=%s %s  %s\n" "$n" "$rc" "${sol:-?}" "${us:-?}" "$(grep -a -o -E 'MT[0-9x]+_[^ ]*SK[0-9]' "${y%.yaml}.log" | head -1 | grep -o -E '^MT[0-9x]+|SK[0-9]$' | tr '\n' ' ')" \
    "$(sed -E 's/.*M: ([0-9]+), N: ([0-9]+), K: ([0-9]+).*transA: (.), transB: (.), batch_count: ([0-9]+).*a_type: ([a-z0-9_]+).*/\4\5 \1x\2x\3 bc\6 \7/' "$y")" | tee -a "$OUT/summary.txt"
  if grep -a -q -E "$PAT" "${y%.yaml}.dmesg"; then
    echo "FAULT after line $n -- stopping"; grep -a -E "$PAT" "${y%.yaml}.dmesg" | head -5 | tee -a "$OUT/summary.txt"
    "$here/fault_capture.sh" "$OUT" fault_t0 >/dev/null; setsid "$here/fault_capture.sh" "$OUT" fault_t30 30 >/dev/null 2>&1 < /dev/null &
    exit 3
  fi
  [ $rc = 0 ] || { echo "bench rc=$rc on line $n -- stopping (inspect ${y%.yaml}.log)"; exit 4; }
done
echo DONE | tee -a "$OUT/summary.txt"
