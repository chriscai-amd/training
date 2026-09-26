#!/bin/bash
# Install the one-shot corrected-amdgpu boot on this node and arm it with grub-reboot.
# Does not reboot. Normal boots keep the default GRUB entry and the stock driver.
set -euo pipefail
O=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$O")
EV=${EV:-/shared/chcai/mi450_a0_newcluster/evidence_$(hostname -s | sed 's/.*-\(j19-[0-9]*\)$/\1/')_oneshot}
CAND=${CAND:-/var/tmp/chcai/cwsrmod/candidate_amdgpu.ko}
D=/var/lib/chcai-cwsrfix
ENTRY=chcai-cwsrfix-oneshot
KVER=$(uname -r)
S="sudo -n"
mkdir -p "$EV"

[ "$(modinfo -F srcversion "$CAND")" = D0FD14A94CA9A71E8CE9BB4 ] || { echo "unexpected candidate srcversion"; exit 3; }
rm -rf /tmp/chcai_cand_check
python3 "$REPO/extract_cwsr.py" "$CAND" /tmp/chcai_cand_check >/dev/null
grep -q 68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80 /tmp/chcai_cand_check/cwsr_handlers.json \
  || { echo "candidate does not carry the corrected handler"; exit 3; }
SHA=$(sha256sum "$CAND" | cut -d' ' -f1)

LINE=$($S grep -m1 -P "^\s+linux\s+/boot/vmlinuz-$KVER " /boot/grub/grub.cfg | sed -E 's/^\s+linux\s+//')
case "$LINE" in *"modprobe.blacklist=device_dax,dax_hmem "*) ;; *) echo "unexpected default cmdline: $LINE"; exit 3;; esac
NEW="${LINE/modprobe.blacklist=device_dax,dax_hmem /modprobe.blacklist=device_dax,dax_hmem,amdgpu } chcai_cwsrfix=1"
UUID=$($S grep -m1 -oP "search --no-floppy --fs-uuid --set=root \K\S+" /boot/grub/grub.cfg)
INITRD=/boot/initrd.img-$KVER
[ -f "/boot/vmlinuz-$KVER" ] && [ -f "$INITRD" ]

cat >/tmp/chcai_custom.cfg <<EOF
# chcai MI450 A0 CWSR-handler test: one-shot entry booted once via grub-reboot. Same kernel, initrd
# and cmdline as the default entry plus modprobe.blacklist=amdgpu and chcai_cwsrfix=1.
# Safe to delete; see /var/lib/chcai-cwsrfix/README.txt.
menuentry '$ENTRY' --id $ENTRY {
	load_video
	gfxmode \$linux_gfx_mode
	insmod gzio
	if [ x\$grub_platform = xxen ]; then insmod xzio; insmod lzopio; fi
	insmod part_gpt
	insmod ext2
	search --no-floppy --fs-uuid --set=root $UUID
	linux	$NEW
	initrd	$INITRD
}
EOF
grub-script-check /tmp/chcai_custom.cfg
sed "s/@SHA@/$SHA/" "$O/load.sh.in" >/tmp/chcai_load.sh
bash -n /tmp/chcai_load.sh

$S install -d -m 755 -o root -g root "$D" "$D/receipts"
$S install -m 644 -o root -g root "$CAND" "$D/candidate_amdgpu.ko"
$S install -m 755 -o root -g root /tmp/chcai_load.sh "$D/load.sh"
$S install -m 644 -o root -g root "$O/README.txt" "$D/README.txt"
$S install -m 644 -o root -g root "$O/chcai-cwsrfix-load.service" /etc/systemd/system/chcai-cwsrfix-load.service
$S ln -sfn /etc/systemd/system/chcai-cwsrfix-load.service /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
$S install -m 644 -o root -g root /tmp/chcai_custom.cfg /boot/grub/custom.cfg
$S grub-reboot "$ENTRY"

echo "== verification"
$S sha256sum "$D/candidate_amdgpu.ko" | sed "s/^/installed: /"; echo "expected:  $SHA"
$S grub-editenv /boot/grub/grubenv list
ls -la "$D" /boot/grub/custom.cfg /etc/systemd/system/chcai-cwsrfix-load.service /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
echo "one-shot linux line: $NEW"

cat /proc/sys/kernel/random/boot_id >"$EV/pre_boot_id.txt"
cat /sys/module/amdgpu/srcversion >"$EV/pre_srcversion.txt"
$S dmesg >"$EV/pre_dmesg_stock_boot.txt"
cp /tmp/chcai_custom.cfg "$EV/custom.cfg"; cp /tmp/chcai_load.sh "$EV/load.sh"
echo "$SHA  candidate_amdgpu.ko" >"$EV/candidate_sha256.txt"
echo "armed: next boot uses $ENTRY (pre boot_id $(cat "$EV/pre_boot_id.txt"))"
