#!/usr/bin/env bash
# Reset the GPU to idle between bisect runs (docs/mi450.md [6]).
#
# After a memory-access fault the rank process stays wedged and its dataloader
# workers are orphaned, and both keep holding HBM -- a following run then starts
# under artificial memory pressure. Run this from a file rather than inline:
# an inline `pkill -f train_ranker` also matches the shell running it, so the
# cleanup kills itself partway through.
set -uo pipefail

self=$$
for pid in $(ls -1 /proc | grep -E '^[0-9]+$'); do
  [ "$pid" = "$self" ] && continue
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
  [ -z "$cmd" ] && continue
  case "$cmd" in
    *train_ranker*|*multiprocessing.spawn*|*resource_tracker*)
      kill -9 "$pid" 2>/dev/null
      ;;
  esac
done
sleep 5
echo "[cleanup] live python:"
ps -eo pid,stat,cmd 2>/dev/null | grep pytho[n] | grep -v defunct || echo "  none"
