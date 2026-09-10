# gfx1250 A0: standalone extended-VGPR codegen reproducer, and a separate multi-stream fault

Two results from an attempt to build a standalone reproducer for `docs/mi450.md`
defect [3] (`train_loss=nan` at `BATCH_SIZE=128`), following
[raikonenfnu/training#3](https://github.com/raikonenfnu/training/issues/3).
They go to different teams, and the second is **not** the first.

1. **A standalone kernel that synthesizes issue #3's allocation** — `torch` +
   `triton` only, no repo import, no dataset. Previously `docs/mi450.md` recorded
   *"None, and three probes actively failed to find one"*. This closes that gap
   for the **codegen**, and is checkable without a GPU dispatch.
2. **A separate, reproducible page fault** under high register pressure with
   concurrent streams — which the controls show is **not** the extended-VGPR
   mechanism, and which is very likely a runtime/driver issue rather than an
   LLVM one.

**The value-victim hypothesis was not confirmed.** The idea under test was that
[6] (page fault) and [3] (`nan`) are one LLVM defect with different victims —
address-critical vs arithmetic. Under clean launch conditions the hazard-present
kernel produced bit-exact results across 200 iterations × 6 repeats. That is a
negative result for the hypothesis, reported as such.

## Result 1 — the allocation, reproduced standalone (for LLVM/AMDGPU)

```bash
export AMDGCN_USE_BUFFER_OPS=0
python scripts/repro_gfx1250_extvgpr_value.py --compile-only --show-asm
```

Compiles only — never dispatches, so it cannot fault or wedge a node. Default
config (`BLOCK_M=128 BLOCK_N=128 BLOCK_K=32 NACC=4 warps=4`), Triton
`7ff97e3109`, `.vgpr_count: 602`, **`.vgpr_spill_count: 0`**:

```
v_wmma*=64  s_set_vgpr_msb=101  extended regs referenced=283

v594: def line 23 -> use line 689, 66 s_set_vgpr_msb mode changes in between
   v_and_b32_e32 v82 /*v594*/, 31, v0               <-- lane-id mask, BEFORE the WMMAs
   ...
   v_lshlrev_b32_e32 v2 /*v514*/, 2, v82 /*v594*/   <-- read back AFTER, to form an address
```

Same shape as `v257` in issue #3: a lane-derived value in an **extended** VGPR,
live across the whole WMMA region under mutable `S_SET_VGPR_MSB` state, cheap
enough that LLVM could have rematerialized it instead of carrying it. The 66
mode changes are the static count; the loop runs `NDOT` times, so dynamically it
is thousands.

**Spilling is not involved** — this configuration spills zero registers. Pushing
to `BLOCK_N=256/NACC=4` does reach the 1024-VGPR ceiling that defect [6]'s
failing config hits, but it spills 255 registers, and a value spilled to scratch
is not live in an extended VGPR at all. The script refuses to run such a
configuration rather than reporting a vacuous PASS. `--sweep` measured 60
configurations to find ones that genuinely carry the hazard; 27 do.

### Static A/B on identical LLVM IR

| `llc -mcpu=gfx1250` | `s_set_vgpr_msb` | extended refs | `.vgpr_count` | spills |
|---|---|---|---|---|
| (default) | 101 | 415 | 602 | 0 |
| `-mattr=-1024-addressable-vgprs` | **0** | **0** | 256 | 654 |

The workaround removes the mechanism structurally. To A/B it at runtime rather
than only under `llc`, the AMD Triton backend needs one line — it currently
hard-codes an empty target-feature string. In
`python/triton/backends/amd/compiler.py`, `make_amdgcn`:

```python
-        features = ''
+        features = os.environ.get('TRITON_AMD_TARGET_FEATURES', '')
```

Verified to reach codegen: with `TRITON_AMD_TARGET_FEATURES=-1024-addressable-vgprs`,
`s_set_vgpr_msb` drops 101 → 0 and `.vgpr_count` 602 → 256.

## Result 2 — a multi-stream fault that is NOT this mechanism (for the runtime team)

```bash
python scripts/repro_gfx1250_extvgpr_value.py --occupancy 16 --streams 4 --iters 200
```

Rates over 6 repeats of 200 iterations each, identical launch parameters:

| Configuration | VGPRs | spills | ext regs | PASS | FAULT | wrong numbers |
|---|---|---|---|---|---|---|
| baseline (occ 16 × 4 streams) | 602 | 0 | 283 | 3 | **3** | 0 |
| `--remat` | 602 | 0 | yes | 1 | 0 | **5** |
| `--no-wmma` | 648 | 0 | 374 | 3 | 0 | **3** |
| `TRITON_AMD_TARGET_FEATURES=-1024-addressable-vgprs` | 256 | 654 | **0** | 1 | **5** | 0 |
| low pressure (`--nacc 2 --block-n 64`) | 234 | 0 | **0** | 6 | 0 | 0 |
| occ 16 × **1** stream | 602 | 0 | 283 | 6 | 0 | 0 |
| occ **4** × 4 streams | 602 | 0 | 283 | 6 | 0 | 0 |
| plain `torch.mm`, 4 and 8 streams | low | 0 | 0 | 6 | 0 | 0 |

**Why this is not the extended-VGPR hazard**, in the order the controls rule it out:

- `-mattr=-1024-addressable-vgprs` — issue #3's own fix, which eliminates every
  extended register — makes it **worse** (5/6 faults vs 3/6). A fix for the
  mechanism cannot increase the failure rate of the mechanism.
- `--no-wmma` still fails, so WMMA is not required. (Note this is a *weak*
  control on its own: with no `v_wmma` the analyzer has no region to span and is
  silent by construction. But it still uses 374 extended registers.)
- The faulting address is nowhere near the data. Buffers occupied
  `0x774e86a00000`–`0x774ea7200000`; the fault was at `0x774fa86fc000`, roughly
  4 GiB past the end of every one of them. A lane offset corrupted in a register
  would land in or near a buffer, not gigabytes outside it.
- Both **high register pressure** and **stream concurrency** are necessary, and
  neither alone is sufficient: 234 VGPRs passes, one stream passes, occupancy 4
  passes.

What that leaves is: a Triton kernel with high register pressure or heavy
scratch, massively oversubscribed (4096 workgroups × 4 concurrent streams),
faults in a region outside its own allocations. Plain `torch.mm` at the same
stream count does not, so it is not generic concurrency — but the evidence points
at scratch/private-memory backing under concurrent dispatch, which is a runtime
concern, not a codegen one.

## Caveats on the harness

Recorded because each of them produced a wrong answer first:

- **Register numbers are not live ranges.** The asm printer annotates extended
  references as `v2 /*v770*/`, but LLVM recycles register numbers; two
  references to `v770` on either side of a WMMA chain are usually unrelated
  values. The analyzer now separates destination from source operands and
  requires a write before, a read after, and no rewrite between. The earlier
  reference-counting version reported a hazard that did not exist — the
  "spanning" value it found was an accumulator's init on one side and an
  unrelated reuse of the same number on the other.
- **Concurrent streams need separate output buffers.** Sharing them made the
  harness report corruption that was its own aliasing, in controls that were
  actually clean.
- **Match `--iters` across controls.** An apparent nwit-dependence was an
  artifact of comparing a 100-iteration control against a 200-iteration test.
- **Single runs prove nothing here.** Every rate in this document is 6 repeats;
  the baseline is 3/6, so a one-shot run has an even chance of showing either
  answer.
