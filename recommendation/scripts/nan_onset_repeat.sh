#!/usr/bin/env bash
# Measure the DISTRIBUTION of the nan onset step over N identical short runs.
#
# Why this exists: run d (no capture hook) nan'd at step 11; run e (capture hook,
# which adds a per-step device sync) nan'd at step 5 -- on the SAME batch that run
# d computed as 0.13889. Same input, different output. That is either real compute
# nondeterminism or an instrumentation artifact, and the two runs differed in
# instrumentation so we could not tell which. This script removes that confound by
# running N times with BYTE-IDENTICAL config.
#
# Config is deliberately run-d baseline: per-step loss logging only, NO nan_capture
# hook. The "Step N train_loss=" line already reveals onset, so nothing beyond
# stock logging is added.
#
# NUM_TRAIN_TS=8 because windows 0..7 cover ~19 steps (w2/w3/w4 are dropped by
# drop_last: 885/121/41 anchors < batch 1024), which is past both observed onsets.
# Startup is dominated by per-window enumeration, so 8 windows instead of 80 takes
# run-to-trigger from ~12 min to ~1 min.
#
# EVAL MUST BE OFF, and shortening the horizon is exactly why. eval_interval_steps
# = round(EVAL_EVERY_DATA_PCT * total_train_anchors / batch). Windows 0..7 hold only
# 24,757 anchors, so the 0.001 default gives round(0.024) = 0, clamped to 1 -- eval
# after EVERY step, against a ~2.06M-sample holdout that does not shrink with the
# horizon. Observed cost: step 1 took 81 s, then 24 min of back-to-back eval with
# no step 2. Setting both cadence knobs to 0 disables eval entirely and skips
# building the eval dataloader (yambda_5b.gin ~line 649). Eval would also perturb
# the measurement, since it syncs between steps.
#
# TRITON_CACHE_DIR: the container ALREADY has a warm cache at /root/.triton (57 MB,
# ~2400 files) which survives `docker restart`. Pointing TRITON_CACHE_DIR at a fresh
# empty dir throws that away and forces a full recompile + autotune -- observed to
# cost >9 min on step 2, far worse than the ~160 s it was meant to save. So we seed
# the persistent /data cache from the warm one on first use. /data is preferred over
# /root/.triton only because it also survives container RE-CREATION, not just restart.
set -uo pipefail
cd /home/chcai/training/recommendation

N=${N:-10}
TAG=${TAG:-onset}
# Windows 0..7 yield 20 steps; past run d's step-11 and run e's step-5 onsets.
STOP_AT_STEP=${STOP_AT_STEP:-20}
MAX_WAIT=${MAX_WAIT:-1200}
RUNID=${RUNID:-$(date +%m%d_%H%M%S)}
OUT=${OUT:-runs/nan_onset_$RUNID}
CACHE=/data/mlperf_dlrm_v4/triton_cache
CACHE_HOST=/home/chcai/dlrm_data/triton_cache
# Seed once from the container's warm cache; skip if already populated.
if [ "$(find "$CACHE_HOST" -type f 2>/dev/null | wc -l)" -lt 2000 ]; then
  echo "[onset] seeding Triton cache from container /root/.triton ..."
  docker exec "${CONTAINER:-triton-7ff97e-test}" bash -c \
    "mkdir -p $CACHE && cp -an /root/.triton/cache/. $CACHE/cache/ 2>/dev/null; true"
  echo "[onset] cache now: $(du -sh "$CACHE_HOST" 2>/dev/null | cut -f1)"
fi
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf 'run\tonset_step\tlast_step\tstatus\telapsed_s\n' > "$SUMMARY"

echo "[onset] N=$N out=$OUT"
echo "[onset] host uptime at start: $(uptime -p)"

for i in $(seq 1 "$N"); do
  LOG="$OUT/run$(printf '%02d' "$i").log"
  t0=$(date +%s)
  # UNIQUE RUN_NAME PER RUN -- do not remove. RUN_NAME drives
  # CKPT_PATH=/data/mlperf_dlrm_v4/ckpt_$RUN_NAME, and the trainer AUTO-RESUMES
  # from whatever it finds there. Leaving the default (conv_b1024_s025) meant run
  # 1 of 0917g wrote an end-of-window checkpoint at train_ts=7, and run 1 of 0917h
  # then read it, logged "Resume target already reached (end_ts=8, start_ts=8) --
  # skipping straight to final eval", ran ZERO train steps, and burned 17 min at
  # 1334 W on the 2,058,204-anchor holdout. Every later run in a series would have
  # done the same, so the whole series would have been silently empty. The default
  # is also the PRODUCTION convergence run's checkpoint dir, which these throwaway
  # runs have no business writing into.
  RN="nanonset_${RUNID}_$(printf '%02d' "$i")"

  # GPU health gate: never start a run on a wedged GPU -- the result would be noise.
  vram=$(timeout 60 amd-smi monitor 2>/dev/null | awk 'NR==2{print $(NF-1)}')
  if ! timeout 60 amd-smi monitor >/dev/null 2>&1; then
    echo "[onset] run $i: amd-smi not responding, GPU likely wedged -- STOPPING"; break
  fi
  echo "[onset] run $i/$N start (vram=$vram)"

  CKPT_STEP_FREQ=100000 KEEP_LAST_N=1 RUN_NAME="$RN" \
  scripts/run_prod_convergence_1gpu.sh "$LOG" \
    -e NUM_TRAIN_TS=8 \
    -e EVAL_EVERY_DATA_PCT=0 -e EVAL_EVERY_N_WINDOWS=0 \
    -e PROGRESS_EVERY=1 -e METRIC_LOG_FREQ=1 \
    ${PROBE:+-e NAN_MODULE_PROBE=1} \
    -e TRITON_CACHE_DIR="$CACHE" \
    >> "$OUT/driver.log" 2>&1 &
  launcher=$!

  # Cut the run the moment the answer is in. Letting it exit on its own costs a
  # ~130 s end-of-window checkpoint plus a final eval over the full ~2.06M-sample
  # holdout -- neither of which can change the onset, and the eval alone was
  # >3 min of dead time per run in 0917g run01. Stop on: nan (onset found),
  # STOP_AT_STEP reached (clean, training over), or an amdgpu fault.
  waited=0
  while [ "$waited" -lt "$MAX_WAIT" ]; do
    kill -0 "$launcher" 2>/dev/null || break
    if grep -qE 'Step [0-9]+ train_loss=nan' "$LOG" 2>/dev/null; then
      # The probe reports one step late by design (it reads the previous step's
      # async D2H). Give it a grace window to print before we kill the process,
      # otherwise the nan run dies with its report still unflushed.
      if [ -n "${PROBE:-}" ] && ! grep -q 'NON-FINITE MODULE REPORT' "$LOG" 2>/dev/null; then
        echo "[onset] run $i: nan seen, waiting ${PROBE_GRACE:-120}s for probe report"
        pg=0
        while [ "$pg" -lt "${PROBE_GRACE:-120}" ]; do
          grep -q 'NON-FINITE MODULE REPORT' "$LOG" 2>/dev/null && break
          kill -0 "$launcher" 2>/dev/null || break
          sleep 5; pg=$((pg+5))
        done
      fi
      echo "[onset] run $i: nan seen, cutting early"; break
    fi
    cur=$(grep -oE 'Step [0-9]+ train_loss=' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    if [ -n "$cur" ] && [ "$cur" -ge "$STOP_AT_STEP" ]; then
      echo "[onset] run $i: reached step $cur with no nan, cutting early"; break
    fi
    if grep -qE 'Memory access fault|MES might be in unrecoverable' "$LOG" 2>/dev/null; then
      echo "[onset] run $i: amdgpu fault in log, cutting"; break
    fi
    sleep 5; waited=$((waited+5))
  done
  [ "$waited" -ge "$MAX_WAIT" ] && echo "[onset] run $i: MAX_WAIT ${MAX_WAIT}s hit, cutting"
  scripts/stop_training.sh >/dev/null 2>&1
  wait "$launcher" 2>/dev/null
  # Drop this run's checkpoint dir so a rerun cannot resume and /data cannot fill.
  docker exec "${CONTAINER:-triton-7ff97e-test}" \
    rm -rf "/data/mlperf_dlrm_v4/ckpt_$RN" >/dev/null 2>&1

  # A run that resumed to completion logs no steps and is NOT evidence of anything.
  if grep -q 'Resume target already reached' "$LOG" 2>/dev/null; then
    echo "[onset] run $i: RESUMED-TO-DONE, no train steps -- config bug, ABORTING"
    printf '%d\tNA\tNA\tRESUME_BUG\t0\n' "$i" >> "$SUMMARY"; break
  fi

  # First step whose loss is nan, and the last step reached.
  onset=$(grep -oE 'Step [0-9]+ train_loss=nan' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+' | head -1)
  last=$(grep -oE 'Step [0-9]+ train_loss=' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
  [ -z "$onset" ] && onset="none"
  [ -z "$last" ] && last="0"

  if grep -qE 'Memory access fault|MES might be in unrecoverable' "$LOG" 2>/dev/null; then
    status="FAULT"
  elif [ "$onset" != "none" ]; then
    status="nan"
  elif [ "$last" = "0" ]; then
    status="NO_STEPS"
  else
    status="clean"
  fi

  t1=$(date +%s)
  printf '%d\t%s\t%s\t%s\t%d\n' "$i" "$onset" "$last" "$status" "$((t1-t0))" >> "$SUMMARY"
  echo "[onset] run $i: onset=$onset last=$last status=$status ($((t1-t0))s)"

  scripts/stop_training.sh >/dev/null 2>&1
  sleep "${BETWEEN_RUNS_S:-30}"
done

echo; echo "[onset] === SUMMARY ==="; column -t "$SUMMARY"
echo; echo "[onset] onset histogram:"
awk -F'\t' 'NR>1{c[$2]++} END{for(k in c) printf "  step %-6s %d\n", k, c[k]}' "$SUMMARY" | sort -k2 -n
