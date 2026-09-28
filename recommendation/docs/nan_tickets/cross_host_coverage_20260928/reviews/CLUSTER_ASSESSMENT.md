# A0 RCK cluster assessment against the frozen CWSR ticket

Reviewed repository HEAD `8cf681dead52574301cc68bdfa37442626d333b9` on September 28, 2026. This is a CPU review of checked-in evidence, not a current observation of the remote cluster. No supplied build/install/boot scripts were executed; no remote, GPU, module, commit, or frozen-ticket action occurred.

**The cluster evidence belongs in a supplement to the same CWSR component ticket.** It identifies the exact original and corrected handler images and adds full four-GPU training observations with the expected failure location and a corrected run beyond 2,000 finite steps. The A0 controlled register/trap/DW tests remain the strongest proof of mechanism and prevention. The cluster record corroborates applicability; its training comparison is not a fully matched single-variable experiment, and it does not prove all NaNs, stalls or node losses are fixed.

The frozen [TICKET.md](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_driver_ticket_v1/TICKET.md) has no cluster host/run/telemetry section. Preserve that attachment and add this separately scoped evidence. The source [cluster record](/home/chcai/training/recommendation/docs/mi450_rck_spur/mi450_rck_spur.md:17) was updated at September 27 02:35 UTC and explicitly says the corrected run was still in progress. It is not evidence of the run's state on September 28.

## Native identity and loaded-driver evidence

| Item | Evidence and assessment |
|---|---|
| Original handler | 5,656 bytes, SHA256 `0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290`, matching frozen A0 native bytes. Recorded for j19-1, j19-9, j19-11 and h21-15. |
| Corrected handler | 5,672 bytes, SHA256 `68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80`, matching frozen A0 native bytes. |
| Cluster module | Stock srcversion `ECD716E95BE41CD5A13F14E`; corrected `D0FD14A94CA9A71E8CE9BB4`. Corrected `.ko` SHA256 `d224829e40dd759028f6ec36573ffedbdb79ee3aae215e014106517b8f23ea18`, 40,295,376 bytes. These are different full modules from the A0 investigation host, despite identical relevant handler images. |
| Saved build audit | j19-1 and j19-9 logs report 34,004 functions checked; unchanged rebuild differs from unsigned installed module only in 20 build-ID bytes; candidate `.text` differs by two handler-size bytes at offsets 3442 and 3448 of `kgd2kfd_device_init`. |
| j19-9 corrected activation | Boot `7f22d2fb-18e1-4895-9b5a-a2c054f872b6`; load log records `insmod rc=0` at 01:27:16, corrected srcversion and CWSR enabled. Post-boot inspection reports four GPUs bound, AutoNUMA on and zero KFD processes before training. |
| j19-1 corrected activation | Armed installation and reboot request only. The record says shutdown hung; it supplies no corrected training result for that host. |
| h21-15 | Original-handler identity on el9 srcversion `390DDE7DEA1EBD99204E6D5`; no recovered training rows or corrected run. Different BKC and uncertain stepping; PCI ID/revision alone is insufficient. |

Native checks performed here are stronger than comparing names: the reviewer parsed only literal constants from [make_candidate_handler.py](/home/chcai/training/recommendation/docs/mi450_rck_spur/tools/cwsr_fix/make_candidate_handler.py:28), applied its 48-to-64-byte prologue replacement and byte-4 branch adjustment to the frozen original native image, and reproduced the frozen corrected image byte for byte. The supplied script was neither imported nor run. This confirms the proposed cluster payload is the same tested correction, including `S_SETREG_B32` and the zero prefetch address construction.

Saved module receipts: [j19-1 audit](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-1/corrected_module_audit.txt:1), [j19-9 audit](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_module_audit.txt:1), [j19-11 identity](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-11/cwsr_handlers.txt:1), [h21-15 identity](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/h21-15/cwsr_handlers.txt:1), [candidate installation hash](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/armed_install.txt:2), [load receipt](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_boot/load_log.txt:1), [post-boot inspection](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_boot/post_boot_inspect.txt:1).

**Limit:** the cluster `.ko` files, full JSON audits and relocation mappings are not in the checkout. This review authenticates the retained summaries and independently reconstructs the handler, but cannot independently re-extract the claimed cluster module or prove its complete executable/relocation change boundary. The saved `.rodata` overlap difference of 528,176 bytes includes shifted data; that aggregate count alone is not a byte-by-byte proof that only handler content changed. The full-module claims remain saved audit observations, unlike A0's directly retained native/module audits.

## Independently counted numerical observations

All 4,575 CSV rows were decoded; each file has consecutive unique steps beginning at 1 and nondecreasing observation timestamps.

| Run | Directly checked CSV evidence | Separate status or summary | Classification |
|---|---|---|---|
| j19-1 stock default | 113 rows; steps 1–110 finite, 111–113 `nan` | Nonfinite-stop record agrees: first loss NaN 111 | Actual nonfinite loss |
| j19-1 stock bound | 1,889 rows, all finite; last 05:02:20.650 | Later status reports finite through 1,912 at 05:02:55.693 | Finite prefix; later node loss in narrative, not a NaN observation |
| j19-11 stock reproA, default plus tripwire | 122 rows, all finite | Tripwire summary: backward failure at 123, inputs finite, one nonfinite dense gradient of 69, four reporting ranks | Actual localized nonfinite gradient; no loss row 123 |
| j19-11 stock reproB, bound | Six rows, all finite | Status `stall=true`, trainer alive, no nonfinite steps | Stall; not a NaN observation |
| j19-9 stock bound | 356 rows; steps 1–353 finite, 354–356 `nan` | Nonfinite-stop record agrees: first loss NaN 354 | Actual nonfinite loss under binding |
| j19-9 corrected default | **2,089 rows, all finite**, last 02:34:19.393 | Later status: **2,108** at 02:34:49.434, no nonfinite/stall/error/eval/stop, trainer alive | Corrected finite prefix beyond 2,000 steps; no final outcome |
| j19-9 stock bound tripwire | No loss CSV | MES wedge evidence and telemetry | No numerical failure localized |
| h21-15 | No recovered loss CSV | Narrative says node agent lost | No numerical outcome available |

The corrected headline **2,128 at 02:35:19** appears in the document, but the checked-in CSV and status stop earlier. It may be a later observation; this checkout does not independently support that exact count. A defensible ticket statement is: “2,089 consecutive finite losses independently checked, with a later monitor snapshot reporting 2,108; one run, still in progress at the retained checkpoint.” The j19-1 bound 1,912 figure is likewise supported by the later status, not by all corresponding raw loss rows.

Exact sources: [j19-1 default losses](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-1/control_default/loss.csv:111), [j19-1 bound losses](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-1/numabind/loss.csv:1889), [j19-1 bound status](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-1/numabind/status.json:5), [j19-11 tripwire summary](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-11/reproA/tripwire_summary.json:1), [reproB status](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-11/reproB/status.json:5), [j19-9 bound losses](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/long_numabind/loss.csv:354), [corrected losses](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_default/loss.csv:2089), [corrected status](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_default/status.json:5).

The localized parameter is `_hstu_transducer._input_preprocessor._additional_embedding_mlp.2.weight`, shape `[512,256]`, matching A0 attempt 405's DW location. This is useful corroboration. Only the small summary is retained here; no raw per-rank tripwire events or tensor bytes establish the cluster's exact NaN coordinates or reproduce A0's 640-NaN pattern. Four ranks reporting a failure are not four independent causal trials.

## Trigger telemetry and comparison limits

All 394 JSONL telemetry rows were parsed, checked for strictly increasing times and four GPU entries. The headline corrected trigger figures reproduce exactly:

- Corrected [rows 24–124](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_default/telemetry.jsonl:24), 01:44:25.537–02:34:29.409, span 3,003.872 seconds. Rates: 2,232,010.56 PTE updates/min, 1,368,619.31 hint faults/min, 196,153.27 migrated pages/min. Summed per-process/per-GPU `evicted_ms` increases **18,466 ms**. Four nonzero process sums at the end are 9,254, 14,699, 12,713 and 11,234 ms.
- Stock j19-1's **2,543,480.00 PTE updates/min** is correct for [the entire telemetry snapshot](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-1/control_default/telemetry.jsonl:1), 03:50:58.152–04:04:29.013, including startup. It is not the same interval definition as the corrected 50-minute training window. Using only its first post-step telemetry sample through its last sample gives 6,464,630.57/min over 120.145 seconds; neither short window establishes an equivalent dose of interrupts.
- Stock j19-1 binding has zero PTE-update delta throughout its 114-row snapshot. Its peak per-GPU VRAM figures match the document.
- Stock j19-9 binding has zero PTE-update delta in [rows 32–39](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/long_numabind/telemetry.jsonl:32), 20:39:10.940–20:42:41.219. The final nonzero process eviction sums are 7,166, 10,818, 6,877 and 7,963 ms. The last telemetry sample precedes the first NaN observation by **30.036 seconds**, so the exact intervening trigger history is unobserved.

These are host-wide NUMA counters and cumulative per-process/per-device eviction-duration counters. They establish continuing activity consistent with exposure, not a count of CWSR entries, the saved wave's MODE state, or the interrupt that corrupted a particular tensor. Summing device counters must not be described as elapsed wall time or silently converted to an interrupt count. Non-AutoNUMA eviction sources remain untraced. The stock bound NaN is enough to reject binding as a guarantee of finite training; its failure was not localized to the same DW by a successful bound tripwire run.

The document's “only the handler changed” needs two scopes:

1. **Module construction:** the saved audits support a narrowly changed handler plus size selections, with the retained-artifact limits above.
2. **Training experiment:** the records do not isolate that as the only changed variable. j19-1/j19-11 versus corrected j19-9 changes the physical host and boot. The [stock platform](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-1/control_default/platform.txt:18) records FBGEMM `2026.9.25`; the [corrected platform](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_default/platform.txt:18) records `2026.9.26`. This may be a build-date label, but byte equality was not captured. Same-host original j19-9 matches that latter label but [uses `NUMA_BIND=0,1`](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/long_numabind/platform.txt:16), while corrected uses none. j19-11 reproA also adds a tripwire and a 600-step bound. Recorded source commit, kernel, GPU firmware and listed parameters agree where stated, but a complete run-time library/configuration/data hash inventory is absent.

## Faults, halts and historical limits

The stock j19-9 [kernel excerpt](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/dmesg_wedge_excerpt.txt:6) contains exactly four `INVALIDATE_TLBS` MES non-responses at 20:57:47 and one `REMOVE_QUEUE` non-response at 21:00:55, followed by blocked queue-restore/KFD workers. The [telemetry](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/tripwire_numabind_wedge/telemetry.jsonl:28) has an identical eviction-counter suffix from 21:01:00.019 through 21:31:31.656, summed 14,227 ms, while GPU `0004` remains 100% busy. The separately retained [halt-clear observation](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/halt_clear_observation.txt:5) records five KFD processes becoming one within 20 seconds and the later node loss/reset.

That evidence supports the documented MES non-response plus driver halt behavior. It does not prove the origin of the firmware non-response, that CWSR caused it, or that the correction repairs it. The four-hour busy claim extends beyond the 21:31 telemetry via the 01:12 observation; it is not a continuous four-hour telemetry capture. j19-1's idle-GPU hung reboot lacks retained MES failures; do not assert the same underlying cause without a console trace.

The corrected [boot excerpt](/home/chcai/training/recommendation/docs/mi450_rck_spur/evidence/j19-9/corrected_boot/dmesg_excerpt.txt:8) has four `MEM_RESERVED_INFO` rows named `VM_PAGE_FAULT`; these are reserved-memory table entries, not four runtime VM faults. Its BERT lines report one previous-boot error record but explicitly skip its content. They do not identify a new corrected-handler fault, its detailed severity, or the cause of the previous reset. The full stock initialization comparison and complete corrected-training kernel interval are not in this checkout.

No completed corrected run, multi-hour result, final AUC, repeated same-host default comparison, cluster controlled sentinel/DW counterfactual, or successful victim-kernel control is retained. The h21-15 node's missing local logs provide no clean/failing training result. Cluster observations also cannot recover A0's missing step-583 STU1 operands or historical training interrupt timeline. Vendor SP3 acceptance, ABI/hazard review and normal driver integration validation remain separate production work.

## Reproducible review receipts

[CLUSTER_ASSESSMENT.json](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_cross_host_cluster_peer_v1/CLUSTER_ASSESSMENT.json) contains exact SHA256/size pins for all 69 cluster files, the frozen native images and ticket, decoded run counts, exact telemetry intervals/deltas, platform joins, native reconstruction and explicit limits. [analyze_cluster.py](/home/chcai/mi450_logs/root_cause_20260919/session_20260928_cross_host_cluster_peer_v1/analyze_cluster.py) reproduces the CPU analysis without importing supplied cluster tools.

Result at creation: `PASS_INDEPENDENT_CPU_ASSESSMENT_WITH_EVIDENCE_LIMITS`; JSON 47,683 bytes, SHA256 `3073ffdc57bdbc02837f78a58bf0295a7ac1e39a781d48ce0c558ceee5bf5cb1`.
