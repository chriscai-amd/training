# State preservation after the corrected CWSR prologue

The corrected prologue's temporary `TTMP2` and `TTMP15` values do not survive into a use by the following handler. The [source/native receipt](validation/liveout/LIVEOUT_REVIEW.json) and [independent review](reviews/LIVEOUT_REVIEW.json) reconstruct both native handlers and check all four conservative paths through the following dispatch prefix.

| State on leaving the prologue | Original | Corrected | First subsequent use |
| --- | --- | --- | --- |
| `TTMP2` | Zero | Saved `MODE[17:12]` | Full overwrite from `TTMP12` before any read, at original PC `0xa8`, corrected PC `0xb8` |
| `TTMP15` | Source-bank lane-zero word | Physical `v1[0]` | Unconditional `S_GETREG_B32` overwrite at original PC `0x74`, corrected PC `0x84`, before any branch |
| `TTMP3` | Zero | Zero | Already identical |
| `TTMP14` | Incoming `EXEC_LO` | Incoming `EXEC_LO` | Already identical |

All four paths through the host-trap and save-context tests reach the `TTMP2` overwrite without reading either differing temporary. Native branch targets are decoded from signed immediates, avoiding stale symbolic label addresses in the corrected disassembly. This introduces no surviving dependence on `TTMP2` retaining its original zero value.

The original source captures incoming `STATE_PRIV`, including SCC, before this prologue. The added instructions preserve SCC and VCC; `V_SUB_NC_U32` is the non-carry form. The prologue restores `EXEC_LO`, leaves `EXEC_HI` untouched, and restores the six bank bits before dispatch. SRC2 and the remaining MODE bits are untouched. The affected family is wave32-only. Later context-save operations capture the restored EXEC and bank state. Detailed source coordinates and instruction bytes are retained in the two receipts and the [complete fix sources](fix/source/).

This is a sequential source/native dataflow proof under the existing trap scratch-register ABI and equal hardware-register responses at corresponding reads. It does not prove hazard timing, asynchronous exception timing equivalence, or safety for an external observer of transient TTMP values during nested traps. Those items and original SP3 acceptance remain explicit driver-team review requirements. The original checker is retained as provenance in `validation/liveout/analyze_liveout.py`; it refers to the historical A0 directory layout. This summary adapts the reviewed supplement to attachment-relative links without changing its findings.
