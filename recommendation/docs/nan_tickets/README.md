# NaN investigation tickets

The MI450 gfx1250 CWSR ticket contains the culprit analysis, portable register reproducer, exact tested source/header correction, raw evidence and independent reviews.

- [Driver ticket](session_20260928_driver_ticket_v1/TICKET.md)
- [Package guide and verification commands](session_20260928_driver_ticket_v1/README.md)
- [Standalone attachment](mi450_gfx1250_cwsr_driver_ticket_20260928_v1.tar.gz) (972,023 bytes)
- [Archive SHA-256](mi450_gfx1250_cwsr_driver_ticket_20260928_v1.tar.gz.sha256)
- [Cross-host coverage and supporting evidence](cross_host_coverage_20260928/README.md): assessment of local A0, B0, 1P4G a37-1/a37-2, and the A0 RCK cluster at repository `8cf681d`, including the remaining host-specific validation gaps.
- [Cross-host evidence attachment](mi450_gfx1250_cwsr_cross_host_20260928_v1.tar.gz) and [SHA-256](mi450_gfx1250_cwsr_cross_host_20260928_v1.tar.gz.sha256): 80 files, 408,252 compressed bytes, including the reviewed scope, source snapshots and retained cluster loss/telemetry records. Attach this alongside the original ticket package.

The archive and unpacked package are byte-identical to the reviewed September 28 attachment. The historical paths under `mi450_logs/root_cause_20260919` are compatibility symlinks so frozen evidence receipts continue to resolve. The archive has not been submitted externally.

The versioned packages describe their original preparation checkpoints. The later September 28 corrected-driver run completed 2,305 training steps and one full evaluation with AUC 0.5027790069580078, below the 0.75 target, before the requested cooperative stop. All associated processes exited and the final device scan was idle. The packages and their archives retain their original bytes.

See the [maintained A0 investigation record](../mi450_a0/mi450_a0.md#current-status) and [current evidence index](../mi450_a0/evidence/current_20260928/STATUS.json) for the later qualification, stopped-run results, exact evidence hashes and remaining limits, including the historical STU1 inconsistency.

Related investigation history is kept with this ticket collection:

- [Prior A0 status and evidence narrative](history/history_20260928_prior_status.md)
- [Prior CWSR platform and experiment narrative](history/history_20260923_cwsr_status.md)

These historical snapshots preserve their original checkpoint claims and link back to the maintained A0 records. They supplement the frozen attachments and are not part of either archive's checksum inventory.
