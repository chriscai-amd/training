#!/usr/bin/env bash
# Stop the training run INSIDE the container and wait for the GPU to actually
# release its memory.
#
# WHY THIS EXISTS (2026-09-17 incident):
#   Killing the host-side `docker exec` client does NOT kill the process inside
#   the container -- it only detaches the client. A run "stopped" that way stays
#   alive, holding ~150-380 GB of HBM. A second run launched on top of it then
#   shares the GPU, and per docs/mi450_b0/mi450_b0.md section 3.6 concurrent GPU
#   work corrupts unrelated work on this host in up to 35% of calls, so BOTH
#   runs' results become worthless.
#
#   Killing two live trainers at once is also not instant: KFD queue teardown
#   took ~15 minutes, during which dmesg filled with
#     amdgpu: MES(N, 0) failed to respond to msg=MISC (WAIT_REG_MEM)
#     amdgpu: MES(0, 0) ring buffer is full.
#   and `docker stop` failed with "did not receive an exit event". That looks
#   like a dead GPU but is only teardown in progress. WAIT IT OUT -- do not
#   power-cycle, and do not launch anything until VRAM has actually dropped.
#
#   BUT (2026-09-17, second incident): VRAM being released is NOT proof the
#   scheduler recovered. In that incident VRAM fell 378 GB -> 0.2 GB and the
#   "failed to respond" messages stopped, yet
#     amdgpu: MES(0, 0) ring buffer is full.
#   kept firing continuously for 44 minutes (546 ring-full vs 120
#   failed-to-respond). The GPU was still wedged: amd-smi showed GFX 100% with
#   zero VRAM in use, and `amd-smi reset -G` hung in D state, after which
#   `amd-smi monitor` hung too. Only an AC power cycle cleared it.
#
#   POWER, NOT GFX%, IS THE RELIABLE TELL (2026-09-17, third incident).
#   During that wedge amd-smi read GFX 100% while drawing 857 W -- the IDLE draw
#   on this part -- with VRAM static and no process owning it (`fuser /dev/kfd`
#   empty, only unreaped zombies). A healthy training step reads GFX ~100% at
#   ~1450 W with VRAM growing. So GFX% alone cannot tell a wedge from real work;
#   check power and the dmesg MES count together:
#     amd-smi monitor | sed -n 2p            # ~857 W + GFX 100% = wedged
#     sudo dmesg -T | grep -c 'failed to respond'
#   The first amdgpu error in every wedge observed so far is
#     MES(0, 0) failed to respond to msg=INVALIDATE_TLBS
#   and everything after it (REMOVE_QUEUE, evict/restore, ring-full) is failed
#   recovery, not the cause. Do NOT run `amd-smi reset -G`: it hung in D state
#   and left `amd-smi monitor` hanging too. See docs/mi450_a0/mi450_a0.md 8.1.
#
#   So: if `ring buffer is full` is STILL being emitted after VRAM has dropped,
#   in-band recovery is exhausted and a power cycle IS required. Check with
#     sudo dmesg -T | grep -c 'ring buffer is full'
#   twice ~60s apart -- a rising count after VRAM is clear means wedged, a flat
#   count means recovered. GFX% (not power -- 856-860 W is normal idle on this
#   part) is the corroborating tell: 0% healthy, pinned 100% with no VRAM wedged.
#   This host is shared: get explicit user approval before any power cycle.
#
# Container PID 1 is `sleep infinity`, which never reaps, so the killed
# processes become zombies. That is expected and harmless; what matters is VRAM.
set -uo pipefail

CONTAINER="${CONTAINER:-triton-7ff97e-test}"
TIMEOUT="${TIMEOUT:-1200}"   # teardown genuinely can take ~15 min
VRAM_OK_GB="${VRAM_OK_GB:-5}"

echo "[stop] container=$CONTAINER"
pids=$(docker exec "$CONTAINER" ps -eo pid,args 2>/dev/null \
        | grep -E 'train_ranker|multiprocessing' | grep -v grep | awk '{print $1}')
if [ -z "$pids" ]; then
  echo "[stop] no trainer processes found"
else
  echo "[stop] killing container pids: $(echo $pids | tr '\n' ' ')"
  for p in $pids; do docker exec "$CONTAINER" kill -9 "$p" 2>/dev/null; done
fi

echo "[stop] waiting for VRAM to drop below ${VRAM_OK_GB} GB (timeout ${TIMEOUT}s)"
start=$(date +%s)
while :; do
  # $(NF-1), NOT $NF. `amd-smi monitor` row 2 ends "197.4/432.0 GB", so $NF is
  # the literal string "GB" and a[1]+0 was 0 on EVERY iteration -- this gate
  # exited instantly and never once waited for VRAM to drain. Every
  # "VRAM released: 0 GB after 0s" printed before 2026-09-17 was false.
  used=$(amd-smi monitor 2>/dev/null | awk 'NR==2{split($(NF-1),a,"/"); print a[1]+0}')
  now=$(date +%s); el=$((now-start))
  if [ -n "$used" ] && [ "${used%.*}" -lt "$VRAM_OK_GB" ] 2>/dev/null; then
    echo "[stop] VRAM released: ${used} GB after ${el}s"
    # SETTLE BEFORE ANY RELAUNCH. Tearing down a ~200 GB address space is TLB
    # invalidation work, and `MES(0,0) failed to respond to msg=INVALIDATE_TLBS`
    # is the first error in every wedge seen on this host. Run 3 of
    # nan_onset_0917i was launched 10 s after the previous trainer was killed
    # and wedged 26 s into its first forward. Not proven causal (an 11 s gap
    # survived earlier), but the cost of waiting is nothing.
    sleep "${SETTLE_S:-30}"
    exit 0
  fi
  if [ "$el" -ge "$TIMEOUT" ]; then
    echo "[stop] TIMEOUT after ${el}s, VRAM still ${used} GB"; exit 1
  fi
  sleep 15
done
