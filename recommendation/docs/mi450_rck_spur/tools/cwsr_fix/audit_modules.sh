#!/bin/bash
# Strip the candidate as DKMS does, rebuild an unchanged baseline the same way, then audit
# installed->baseline (build fidelity) and baseline->candidate (handler change only).
set -euo pipefail
KVER=${KVER:-$(uname -r)}
SRC=${SRC:-/usr/src/amdgpu-7.1.1-2398876.24.04}
W=${W:-/var/tmp/chcai/cwsrmod}
REPO=$(cd "$(dirname "$0")" && pwd)
CLEAN=(env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin HOME="$HOME" LANG=C TERM=dumb)
log(){ echo "[$(date -u +%T)] $*"; }

[ -f "$W/src/amd/amdgpu/amdgpu.ko" ] || { echo "candidate build missing"; exit 2; }
strip -g -o "$W/candidate_amdgpu.ko" "$W/src/amd/amdgpu/amdgpu.ko"
log "candidate stripped: $(stat -c %s "$W/candidate_amdgpu.ko") bytes"

if [ ! -f "$W/baseline_amdgpu.ko" ]; then
  rm -rf "$W/src_baseline"; cp -a "$SRC" "$W/src_baseline"
  log "building unchanged baseline"
  (cd "$W/src_baseline" && "${CLEAN[@]}" make KERNELVER="$KVER" >"$W/build_baseline.log" 2>&1) \
    || { echo "BASELINE BUILD FAILED"; tail -30 "$W/build_baseline.log"; exit 4; }
  strip -g -o "$W/baseline_amdgpu.ko" "$W/src_baseline/amd/amdgpu/amdgpu.ko"
  log "baseline stripped: $(stat -c %s "$W/baseline_amdgpu.ko") bytes"
fi
[ -f "$W/installed.ko" ] || zstd -dc "/lib/modules/$KVER/updates/dkms/amdgpu.ko.zst" >"$W/installed.ko"

for m in installed baseline candidate; do
  f=$W/${m}_amdgpu.ko; [ $m = installed ] && f=$W/installed.ko
  printf '%-9s srcversion=%s vermagic="%s" sha256=%s\n' $m \
    "$(modinfo -F srcversion "$f")" "$(modinfo -F vermagic "$f")" "$(sha256sum "$f" | cut -c1-16)"
  [ "$(modinfo -F depends "$f")" = "$(modinfo -F depends "$W/installed.ko")" ] || echo "  DEPENDS DIFFER for $m"
done

log "audit installed -> baseline"
python3 "$REPO/verify_module.py" "$W/installed.ko" "$W/baseline_amdgpu.ko" "$W/audit_installed_to_baseline.json" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(" handlers:", d["installed_handler"], d["candidate_handler"]); print(" differing allocated sections:", [(x["section"], x["differing_bytes_in_overlap"], x["installed_bytes"], x["candidate_bytes"]) for x in d["allocated_sections_differing"]]); print(" functions checked:", d["functions_checked"], "changed:", [(c["symbol"], c["n_different"]) for c in d["changed_functions"]])'
log "audit baseline -> candidate"
python3 "$REPO/verify_module.py" "$W/baseline_amdgpu.ko" "$W/candidate_amdgpu.ko" "$W/audit_baseline_to_candidate.json" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(" handlers:", d["installed_handler"], d["candidate_handler"]); print(" differing allocated sections:", [(x["section"], x["differing_bytes_in_overlap"], x["installed_bytes"], x["candidate_bytes"]) for x in d["allocated_sections_differing"]]); print(" functions checked:", d["functions_checked"], "changed:", [(c["symbol"], c["n_different"], c["different_offsets"]) for c in d["changed_functions"]])'
cmp -l "$W/installed.ko" "$W/baseline_amdgpu.ko" 2>/dev/null | wc -l | xargs echo " whole-file differing bytes installed(unsigned part) vs baseline:"
log "DONE"
