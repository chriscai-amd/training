#!/bin/bash
# Build an amdgpu.ko carrying the corrected gfx1250 CWSR handler, in a scratch copy of the
# DKMS tree, with a clean env so the DKMS Makefile picks the kernel's own gcc (not the login
# gcc-14). Strips it as DKMS does and verifies the handler is the audited candidate.
# Nothing is loaded or installed; run audit_modules.sh next for the full module audit.
set -euo pipefail
KVER=${KVER:-$(uname -r)}
SRC=${SRC:-/usr/src/amdgpu-7.1.1-2398876.24.04}
W=${W:-/var/tmp/chcai/cwsrmod}
REPO=$(cd "$(dirname "$0")" && pwd)
INSTALLED_KO=/lib/modules/$KVER/updates/dkms/amdgpu.ko.zst
CORRECTED_SHA=68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80
log(){ echo "[$(date -u +%T)] $*"; }

rm -rf "$W"; mkdir -p "$W"
log "generating candidate header"
python3 "$REPO/make_candidate_handler.py" "$INSTALLED_KO" "$SRC/amd/amdkfd/cwsr_trap_handler.h" "$W/gen" >"$W/gen.log" 2>&1
grep -q '"corrected_matches_audited_candidate": true' "$W/gen/candidate_handler.json" || { echo "candidate header mismatch"; cat "$W/gen.log"; exit 3; }

log "copying DKMS source tree"
cp -a "$SRC" "$W/src"
cp "$W/gen/cwsr_trap_handler.h.candidate" "$W/src/amd/amdkfd/cwsr_trap_handler.h"

log "building (clean env, CC=gcc -> kernel gcc) on $(nproc) cores"
cd "$W/src"
env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin HOME="$HOME" LANG=C TERM=dumb \
    make KERNELVER="$KVER" >"$W/build.log" 2>&1 || { echo "BUILD FAILED"; tail -40 "$W/build.log"; exit 4; }
KO="$W/src/amd/amdgpu/amdgpu.ko"
log "built $KO ($(stat -c %s "$KO") bytes)"
strip -g -o "$W/candidate_amdgpu.ko" "$KO"
log "stripped $W/candidate_amdgpu.ko ($(stat -c %s "$W/candidate_amdgpu.ko") bytes)"

env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin HOME="$HOME" LANG=C \
    python3 "$REPO/extract_cwsr.py" "$W/candidate_amdgpu.ko" "$W/verify" >"$W/verify.log" 2>&1
GOT=$(python3 - "$W/verify/cwsr_handlers.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
h = [x for x in d["handlers"] if x["symbol"] == "cwsr_trap_gfx12_1_0_hex"][0]
print(h["sha256"], h["bytes"])
PY
)
echo "built handler: $GOT"
case "$GOT" in
  ${CORRECTED_SHA}*) log "PASS: built module carries the audited corrected handler";;
  *) echo "FAIL: built handler is not the audited corrected image"; exit 5;;
esac
log "DONE. candidate $W/candidate_amdgpu.ko srcversion=$(modinfo -F srcversion "$W/candidate_amdgpu.ko") (not installed)"
