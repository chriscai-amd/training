chcai MI450 A0 CWSR trap-handler test, 2026-09-25 (contact: chcai)

candidate_amdgpu.ko  amdgpu 7.1.1-2398876 rebuilt from this node's DKMS source; the only change is
                     the corrected gfx1250 CWSR trap handler (sha256 68c31ab2...,
                     srcversion D0FD14A94CA9A71E8CE9BB4).
load.sh              loads it; run by /etc/systemd/system/chcai-cwsrfix-load.service only when the
                     kernel cmdline has chcai_cwsrfix=1.
receipts/<boot_id>/  load log and dmesg from each such boot.

chcai_cwsrfix=1 is added only by the 'chcai-cwsrfix-oneshot' entry in /boot/grub/custom.cfg, which is
booted once via grub-reboot. Normal boots use the unchanged default entry and the stock driver.

To remove everything:
  rm -f /boot/grub/custom.cfg /etc/systemd/system/chcai-cwsrfix-load.service \
        /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
  rm -rf /var/lib/chcai-cwsrfix
