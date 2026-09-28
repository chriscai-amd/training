The frozen CWSR ticket applies to B0's demonstrated bank-copy component with
high confidence. B0 independently reproduced the same controlled trap defect
and a consequent LayerNorm NaN route. A corrected privileged driver and a
causal training repair have not been demonstrated on B0 in the latest record.
The ticket therefore covers a shared corruption component, while substantial
B0 failure attribution and validation remain open.

This assessment uses repository HEAD
`8cf681dead52574301cc68bdfa37442626d333b9`. All six reviewed B0 documentation/
patch files match that commit byte for byte. The A0 main document has preserved
local edits; its citations below refer to the pinned working file. The cited
A0 result certificates match HEAD. [B0_ASSESSMENT.json](B0_ASSESSMENT.json)
records every input hash, exact source coordinates, the source-patch check and
the distinction between locally checked evidence and B0-reported results.

| Question | Assessment | Evidence |
| --- | --- | --- |
| Does the faulty component apply on B0? | **Yes, strong component evidence.** The documented selected original handler and complete assembly/header fingerprints match A0. B0's 14-case trap sentinel reproduces the same three SRC0-to-DST lane-0 copies. | [B0 continuation, lines 39–75](/home/chcai/training/recommendation/docs/mi450_b0/nan_triage_resume_20260921.md:39); [source fingerprints, lines 32–36](/home/chcai/training/recommendation/docs/mi450_b0/patches/README.md:32). |
| Does that component produce arithmetic NaNs on B0? | **Yes, under controlled injection.** Six unequal-bank cases in the historical 576-VGPR LN kernel produce partial DW/DB column-263 NaNs; nine original/wait/equal-bank controls return zero. | [B0 compiled-LN experiment, lines 312–356](/home/chcai/training/recommendation/docs/mi450_b0/nan_triage_resume_20260921.md:312). |
| Does it explain the additional historical DX/column-5 failure? | **A concrete mechanism is supported on A0 using B0-published executable text.** Earlier-window injection reproduces whole-row DX/column-5 NaNs, and the corrected A0 driver removes them. Exact B0 historical coordinates/events remain unjoined. | [A0 native-text reconciliation, lines 816–827](/home/chcai/training/recommendation/docs/mi450_a0/mi450_a0.md:816); [corrected result certificate, lines 26–40](/home/chcai/training/recommendation/docs/mi450_a0/evidence/current_20260919/CWSR_candidate_AC2_validated_20260922.json:26). |
| Was a corrected driver tested on B0? | **No recorded live-driver counterfactual.** Its own candidate passed a userspace sequence test and CPU module build, then remained staged and uninstalled. | [B0 repair and deployment state, lines 222–299](/home/chcai/training/recommendation/docs/mi450_b0/nan_triage_resume_20260921.md:222). |
| Are B0 training and other failures covered as fixed? | **No.** The original producing boundary is missing; unchanged settings have both NaNs and a 3,000-step pass; later capture faults lack a producing shader PC. Separate source and platform defects remain distinct. | [B0 current scope, lines 29–71](/home/chcai/training/recommendation/docs/mi450_b0/mi450_b0.md:29); [training capture, lines 360–383](/home/chcai/training/recommendation/docs/mi450_b0/nan_triage_resume_20260921.md:360). |

The component identity is stronger than a shared `gfx1250` name. B0 reports
the selected 5,656-byte handler matching A0's published image, despite its
different source package `amdgpu-7.1.1-2389086.24.04`; A0's current frozen
source package is `amdgpu-7.1.1-2397345.24.04`. Both documented original
assembly and header hashes match the exact A0 frozen originals. I applied the
checked-in B0 patch to private copies of those originals, extracted its array,
and independently obtained the documented B0 candidate hash. This checks
source-level applicability without reading a B0 module.

| Handler | Bytes | SHA-256 |
| --- | ---: | --- |
| Shared original | 5,656 | `0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290` |
| Frozen A0 ticket correction | 5,672 | `68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80` |
| B0's separately staged candidate | 5,736 | `1fda473155dc0a850039d2a018378facbb789cc0488891b26cdf18bbefef80b0` |

These are two distinct repairs of the same original component. The A0 ticket
clears/restores six bank bits and compensates the prefetch address. B0's patch
temporarily makes DST equal to incoming SRC0, restores DST around prefetch,
and uses vector no-ops/scalar waits around MODE writes. Its source-derived
candidate replaces `[60,108)` with `[60,188)` and moves the restore target
4044→4124; all other instruction bytes match after the shift. The B0 512-case
comparison found 192 predicted copies in the original sequence and none in
256 candidate cases, but substituted ordinary SGPRs for TTMPs and injected
no trap. This is useful repair evidence, not a loaded-handler result.
[B0 patch design and scope, lines 8–30](/home/chcai/training/recommendation/docs/mi450_b0/patches/README.md:8).
Source applicability also does not establish that A0's compiled kernel module
is a validated B0 deployment artifact.

The B0 LN arithmetic bridge is specific. At setter `0x40`, the handler copies
physical `v1[0]` into `v257[0]`, replacing a live zero-DY value at column 263
with a legitimate temporary NaN from `V_DIV_SCALE_F32(512,512,0)`. The following
arithmetic contaminates the parameter partials while DX stays zero. The equal-
bank trap control was at a nearby different location; a matched same-site
bank intervention was still unrun. This late-site experiment did not explain
the historical additional DW column 5 or 3,072 bad DX elements.
[Instruction chain and limitations, lines 338–356](/home/chcai/training/recommendation/docs/mi450_b0/nan_triage_resume_20260921.md:338).

The later A0 reconciliation substantially narrows that residual mechanism.
Six local kernels contain the 22,528-byte executable matching B0's published
`.text` SHA `77a9a6b4dbc5a08fe188d589ec1cebb302835819a74462f055629bd14181b11e`.
An earlier X-load window copies a NaN surviving in v1's upper half into v513,
then LayerNorm spreads it through the row. On A0, N16/grid1 gives 512 DX NaNs
in row 11; N56/grid1 gives 3,072 in rows 11/19/27/35/43/51, plus two partial DW
column-5 NaNs across the cases. The actual corrected A0 driver removes all
3,584 DX and two partial DW NaNs. The locally retained independent
[AC2 component review](/home/chcai/mi450_logs/root_cause_20260919/session_20260922_ac2_results_peer_v1/health_components_review.json)
and complete consumer support that counterfactual. This used an earlier A0
module build with the same corrected handler; it was not execution on B0.
The exact original B0 ELF and historical full output coordinates were not
recovered. Matching a synthetic count of 3,072 is insufficient to identify the
historical bad rows or number of traps.

B0's training record remains materially different from A0's captured DW
failure. At B0 step 51, the output-gradient GEMM receives already-huge finite
`dout` and agrees with CPU computation. The missing interval is upstream
preprocess/projection/input-LN, or a later overwrite. Original projection DX,
LN inputs/statistics and raw parameter gradients were not retained.
[B0 boundary, lines 163–176](/home/chcai/training/recommendation/docs/mi450_b0/nan_triage_resume_20260921.md:163).
The September 20 3,000-step finite pass used settings that had previously
NaNed at 60/70; no new repair explains that pass. The September 21 capture
later aborted after finite sampled step 830 with illegal access and no tensor
payload. Both observations can coexist with an exposure-dependent CWSR defect;
neither identifies its role in the original training event.

The independent int32 row-stride overflow fixed in `triton_layer_norm.py`
remains a separate established B0 source defect. The CWSR ticket does not
supersede that correction or establish a fix for historical attention/overlap
faults, width-16 health-copy failures or unlocalized page faults.
[B0 source bug, line 1979](/home/chcai/training/recommendation/docs/mi450_b0/mi450_b0.md:1979)
and [health/training distinction, lines 2391–2408](/home/chcai/training/recommendation/docs/mi450_b0/mi450_b0.md:2391).

The next evidence needed for a B0 repair claim is a B0-loaded corrected-handler
sentinel, the matched LN interventions and a matched natural/training
counterfactual with retained producing-boundary data. The documented 3,000-step
acceptance run should follow a causal intervention; another isolated clean run
of unchanged settings cannot establish the repair. B0 documents also report a
local SP3 compatibility build requiring the original prefetch words; this is
a tooling lead from that host, not SP3 validation of the frozen A0 patch.

The linked B0 runtime certificates and raw-data roots are absent on this A0
host. Their runtime findings above are attributed to the checked-in B0 terminal
record. I read and hashed the local A0 reconciliation receipts, without
repeating their tensor scans. The last retained historical-recovery attempt
was blocked by remote authentication; no new remote attempt occurred here.

Selected evidence hashes, with the complete inventory in the JSON:

| Evidence | SHA-256 |
| --- | --- |
| B0 September 21 continuation | `7d8bf18eb1906c7b4e626840cd61c094ad6878f61d5e4868b87c5d214f242543` |
| Checked-in B0 patch | `eea1556000417d48fad867de22c531e2c560fad38934cfceef9afb94e3df64e2` |
| A0 original-driver LN20 certificate | `95e9af9a8fabe7263a29d42dbcb35237ca1b00831fc0031c9c98077e57ca3edb` |
| A0 AC2 corrected result certificate | `9a791fd0000390083bb3307ffbd1c965721675b70ba2b5024bf10144ef0bb047` |
| Independent A0 AC2 component review | `7845987b4ceef81468c599a1c03a3140e6505b120834965855d37004f0e0c913` |
| Historical B0 recovery receipt | `0e72232773e7163704ae5da46990b032f61603bb770d17539d618bedb838a1a8` |

No GPU, remote-host or module action, bulk tensor read, frozen-package edit,
Git commit or push was performed for this assessment.
