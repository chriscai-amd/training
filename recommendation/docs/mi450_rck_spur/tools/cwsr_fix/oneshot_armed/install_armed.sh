#!/bin/bash
# Arm a one-boot load of the corrected amdgpu on a node whose boot already keeps amdgpu out.
# Does not reboot.
set -euo pipefail
O=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$O")
EV=${EV:-/shared/chcai/mi450_a0_newcluster/evidence_$(hostname -s | sed 's/.*-\(j19-[0-9]*\)$/\1/')_oneshot}
CAND=${CAND:-/var/tmp/chcai/cwsrmod/candidate_amdgpu.ko}
D=/var/lib/chcai-cwsrfix
S="sudo -n"
mkdir -p "$EV"

if ! grep -qE "modprobe\.blacklist=[^ ]*amdgpu" /proc/cmdline && ! grep -rqsE "^blacklist amdgpu" /etc/modprobe.d; then
  echo "amdgpu is not kept out at boot on this node; the stock module would load first"; exit 3
fi
[ "$(modinfo -F srcversion "$CAND")" = D0FD14A94CA9A71E8CE9BB4 ] || { echo "unexpected candidate srcversion"; exit 3; }
rm -rf /tmp/chcai_cand_check
python3 "$REPO/extract_cwsr.py" "$CAND" /tmp/chcai_cand_check >/dev/null
grep -q 68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80 /tmp/chcai_cand_check/cwsr_handlers.json \
  || { echo "candidate does not carry the corrected handler"; exit 3; }
SHA=$(sha256sum "$CAND" | cut -d' ' -f1)
sed "s/@SHA@/$SHA/" "$O/load.sh.in" >/tmp/chcai_load.sh
bash -n /tmp/chcai_load.sh

$S install -d -m 755 -o root -g root "$D" "$D/receipts"
$S install -m 644 -o root -g root "$CAND" "$D/candidate_amdgpu.ko"
$S install -m 755 -o root -g root /tmp/chcai_load.sh "$D/load.sh"
$S install -m 644 -o root -g root "$O/README.txt" "$D/README.txt"
$S install -m 644 -o root -g root "$O/chcai-cwsrfix-load.service" /etc/systemd/system/chcai-cwsrfix-load.service
$S ln -sfn /etc/systemd/system/chcai-cwsrfix-load.service /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
$S touch "$D/ARMED"

echo "== verification"
$S sha256sum "$D/candidate_amdgpu.ko" | sed "s/^/installed: /"; echo "expected:  $SHA"
ls -la "$D" /etc/systemd/system/chcai-cwsrfix-load.service /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
cat /proc/sys/kernel/random/boot_id >"$EV/pre_boot_id.txt"
cat /sys/module/amdgpu/srcversion >"$EV/pre_srcversion.txt" 2>/dev/null || true
cp /tmp/chcai_load.sh "$EV/load.sh"
echo "$SHA  candidate_amdgpu.ko" >"$EV/candidate_sha256.txt"
echo "armed: next boot loads the corrected amdgpu once (pre boot_id $(cat "$EV/pre_boot_id.txt"))"
