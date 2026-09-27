chcai MI450 A0 CWSR trap-handler test, 2026-09-26 (contact: chcai)

This node's boot keeps amdgpu blacklisted (kernel cmdline and /etc/modprobe.d), and spurd only
starts once /dev/kfd exists.

candidate_amdgpu.ko  amdgpu 7.1.1-2398876 rebuilt from this node's DKMS source; the only change is
                     the corrected gfx1250 CWSR trap handler (sha256 68c31ab2...,
                     srcversion D0FD14A94CA9A71E8CE9BB4).
load.sh              loads it with the stock options before spurd starts; run by
                     /etc/systemd/system/chcai-cwsrfix-load.service only while ARMED exists.
ARMED                present = load the candidate on the next boot. load.sh deletes it first,
                     so this happens on one boot only; later boots behave as before.
receipts/<boot_id>/  load log and dmesg from each such boot.

To remove everything:
  rm -f /etc/systemd/system/chcai-cwsrfix-load.service \
        /etc/systemd/system/multi-user.target.wants/chcai-cwsrfix-load.service
  rm -rf /var/lib/chcai-cwsrfix
