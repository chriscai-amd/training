# M5: split-T for the six tall-K NT weight gradients

Status 2026-10-11: design, CPU-built catalogs and a CPU-only prototype. **Nothing has run on a GPU.**
Update 2026-10-11 (later): section 3 is IMPLEMENTED in the worktree (uncommitted; patch in
`/home/chcai/m5_work/m5_code.diff`), with the pure functions `splitt_wgrad_plan` / `splitt_wgrad_compute`
split out for CPU tests, a one-time log when a configured shape is refused, and overlays
`gin/mi450_hipblaslt/m5_split_t.gin` (v3/s103) / `m5_split_t_s102.gin`. CPU tests:
`/home/chcai/m5_work/tests_out/pytest_m5.txt`. Catalog simulation: `simulate_v2_v3.txt`. GPU kit:
`gpu_check/README.txt` (supersedes `gpu_kit/`).
Update 2026-10-11 (review fixes): **`/home/chcai/m5_work/M5_REPORT.md` is the current summary and supersedes
this file where they differ.** Changes: (1) the 1.41x numerics figure below is the chunk-independent best
case, NOT a bound -- where chunk partials cancel the split error is ~sqrt(1+q^2)x today's (q up to ~800
measured on synthetic drift data; M5_REPORT section 6); (2) every lib dir now carries `m5_splits.json` and
`triton_addmm` refuses at import an s the catalog does not pin; (3) a third catalog `v3/mix` (G20/G19 on
103, the rest on 102) is installed with overlay `m5_split_t_mix.gin`; (4) a 1-row tail is an elementwise
fp32 `addr_` (no GEMM); (5) M5-preproc only reroutes configured `TNWgradLinear` shapes; (6) optional
`splitt_audit_every` diagnostic.

---

## 0. Decisions

| Item | Decision |
|---|---|
| Math | Treat X [T,·] and dY [T,·] as s contiguous chunks of L = ⌊T/s⌋ rows (zero-copy `view`). One `torch.bmm(a3.transpose(1,2), b3)` gives **bf16 partials**. Then `sum(0, dtype=float32)`. The tail of r = T − sL (< s) rows is added with one **fp32** `addmm_`. Cast once to the original dtype. |
| Output dtype | **bf16-out bmm + fp32 sum**, not `bmm(out_dtype=float32)`. The f32-out path hits the unmodified BB_SB catalog, which always gives sol 1420 (1020 VGPRs, GRVW 1/1). Section 5 has the reasons. |
| Numerics cost | rms error ≈ sqrt(1+q²)× today's, q = ‖sqrt(Σ P_i²)‖/‖Σ P_i‖: **1.41× only when chunk partials are independent** (q≈1); ≫1 where they cancel (CORRECTED, see M5_REPORT §6). The f32-out path would be 1.00×. |
| Switch | gin `apply_hipblaslt_opts.splitt_wgrad_{hstu,preproc,splits}` sets env `HSTU_SPLITT_WGRAD` (`0`/`hstu`/`preproc`/`all`), `HSTU_SPLITT_WGRAD_SPLITS` (`MxN:s,…`) and `HSTU_SPLITT_MIN_ROWS`. All OFF by default. When off, the code is bit-identical to today. |
| Catalogs | Two lib dirs are **built and installed**: `arbor_libs/rocm-10.2_nnonly_v3/s103` and `/s102`. They come from `/home/chcai/m5_work/make_v3.py --sol 103|102 [--install]`; `--map G20=103,…` builds a per-shape mix. v1 and v2 are unchanged. |
| s per shape | Comes from gin per shape. The catalogs cover these candidates: G20 {2,4,8,16}, G19 {2,4,8,10,16,20}, G25 {4,8,16,32,64,128}, G29 {2,4,8,16,32,64}, G31 {4,8,16,32,64,128}, G27 {16,32,64,128,256}. In s102, every s from 2 to 256 selects 102. |
| Estimate (unmeasured) | s103 about 19 ms/step for all 12 calls, against about 288 ms under T8. s102 about 263 ms/step with no 103 at all. A mix (G20/G19 on 103, the rest on 102) about 55 ms. |

---

## 1. Call sites: the six NT wgrads

Paths are relative to `/home/chcai/training_opt/recommendation/generative_recommenders/`. Line numbers are at worktree HEAD be2d249.

In hipBLASLt terms, every one of these is `torch.mm(a.t(), b)` with:
- `a` = [T, D] and `b` = [T, O], both bf16, dense row-major (`eval/eval_operand_layout.py`: lda/ldb = width);
- result [D, O] contiguous bf16;
- hipBLASLt problem: transA=N, transB=T, **m = O (width of b), n = D (width of a)**, K = T, lda = O, ldb = D, batch 1, catalog `BB_BB_HA_Bias_SAV … Ailk_Bjlk`.

The result flows back to an fp32 master parameter through autocast's `.to(bf16)` (`ToCopyBackward`). So the contract is shape [D,O], contiguous, `dtype == a.dtype`.

| G | hipBLASLt m×n | Calls/step | Call site | a, b | How dW is computed today | M3/M4 hook |
|---|---|---|---|---|---|---|
| G20 | 2048×512 | 3 | `ops/triton/triton_addmm.py:1713-1731` `triton_addmm_bwd`, line 1727 `dw = torch.mm(x.t(), dz)`. Called from `ops/triton/triton_hstu_preprocess_and_attention.py:263`, inside the backward of the fused preprocess+attention Function (training uses `hammer_kernel="TRITON"`, prefill False; `ops/hstu_compute.py:336`). | a = `normed_x` [T,512], b = `duvqk` [T,2048] | NT mm | M3, line 1723: `globals().get("_TN_HSTU")` gives the TN copy at line 1725. **The body must stay self-contained:** `dlrm_v4/train/tests/preprocess_backward_stages_test.py:235-252` execs only this function's AST with `{"torch": torch}`. |
| G19 | 512×1536 | 3 | `ops/triton/triton_hstu_linear.py:2385`, in `HSTUComputeOutputFunction.backward` (class at 2167): `d_output_weight = tn_wgrad_mm(y, dout, enabled=_TN_HSTU)` | a = `y` [T,1536] (saved, or recomputed by the norm bwd), b = `dout` [T,512] | `tn_wgrad_mm` (`triton_addmm.py:1644-1653`): `torch.mm(a.t(), b)` when TN is off | M3 via `enabled=_TN_HSTU`; imported at `triton_hstu_linear.py:33-37` |
| G25 | 256×512 | 3 | `modules/preprocessors.py:167,181,203`: the 2nd `TNWgradLinear` (hidden 256 → output 512) of the content, additional and action MLPs | a = dY [T,512], b = X [T,256] | Off: plain `nn.Linear` (`triton_addmm.py:1700-1701` returns `super().forward`), so aten `addmm` backward, `mm_mat2_backward` → `grad.t().mm(input)` (NT, as captured) | M4: `TNWgradLinear.forward` 1700-1710 → `_TNWgradLinearFn.backward` 1676-1687, dW at 1684 `tn_wgrad_mm(dy2, x2)` |
| G29 | 1024×256 | 1 | `preprocessors.py:175` additional-MLP layer 1 (512·2 → 256) | a = dY [T,256], b = X [T,1024] | same | M4 |
| G31 | 512×256 | 1 | `preprocessors.py:162` content-MLP layer 1 (512 → 256) | a = dY [T,256], b = X [T,512] | same | M4 |
| G27 | 24×256 | 1 | `preprocessors.py:198` action-MLP layer 1 (24 → 256) | a = dY [T,256], b = X [T,24] | same | M4 |

**How the shapes map.** The forward/dgrad rows in `groups.json` fix in/out widths: TN (256,T,512) means out 256 / in 512, and NN (24,T,256) means in 24 / out 256. The NT wgrad then has m = in and n = out, which matches every G above.

**Wiring order.** `train_ranker.py:236-238` parses gin, then calls `apply_hipblaslt_opts()` per rank before the heavy imports, so module-level env reads in `triton_addmm.py` see the gin values. `apply_launch_env` (the lib dir swap) runs in the launcher. torch.compile and CUDA graphs are not used on these paths (grep: no `torch.compile` in `dlrm_v4/train`).

---

## 2. Split math and the output contract

Given `a` [T,D] and `b` [T,O] (bf16, row-major), and s taken from the per-shape map:

```
L = T // s ; main = s*L ; r = T - main          (0 <= r < s)
a3 = a[:main].view(s, L, D)   b3 = b[:main].view(s, L, O)          # zero-copy, data_ptr unchanged
part = torch.bmm(a3.transpose(1, 2), b3)        # [s, D, O] bf16
```

The bmm is a hipBLASLt NT problem: m=O, n=D, batch=s, K=L, lda=O, ldb=D, strideA=L·O, strideB=L·D, ldc=O, strideC=D·O. That is the same catalog as today, now batched. Torch takes the transposed view as transB=T without a copy: strides (L·D, 1, D).

```
acc = part.sum(dim=0, dtype=torch.float32)      # [D, O] fp32, one reduction kernel
if r: acc.addmm_(a[main:].t().float(), b[main:].float())   # tail
dW = acc.to(a.dtype)                            # contiguous [D, O]: same shape/dtype/strides as torch.mm(a.t(), b)
```

**The tail.** It is exact, with no padding and no copy of the full tensors. r ≤ s−1 ≤ 255 rows are cast to fp32 (at most 255×2048×4 B ≈ 2 MB) and go through one fp32 NT GEMM: m=O, n=D, K=r, beta=1. That lands in the SS NT catalog, whose **only** solution is sol 2821 (MT32x32x16, 222 VGPRs). The workload's fp32 NT GEMMs already run on it. This keeps the tail off sol 103 and leaves the bf16 catalogs alone.

**Fallback.** The helper returns None, and the caller issues the original call unchanged, when any of these holds:
- s ≤ 1, or the shape is not in the map;
- the tensor is not CUDA, or the dtype is not bf16 for both operands;
- the tensor is not 2-D, or `stride() != (width, 1)`;
- D % 8 or O % 8 is nonzero, or `data_ptr % 16` is nonzero. With D, O multiples of 8, every batch base is 16 B-aligned, which 103's GRVW=8 loads need;
- L < `HSTU_SPLITT_MIN_ROWS` (default 4096);
- **L·max(D,O) ≥ 2³¹** (section 6). Only G20 at s=2 with T ≥ 2,097,152 trips this.

**Output contract.**
- shape [D,O], contiguous, `dtype == a.dtype`;
- one bf16 rounding of an fp32 value, as in today's kernel store, but the partials are bf16 (section 5.2);
- deterministic run to run: SK0 and GSU1 kernels, fixed reduction order.

The downstream path (autocast `ToCopyBackward` → fp32 `.grad`, DDP/optimizer) sees an identical tensor type. The CPU test checks this through a custom Function to an fp32 master weight.

**Autograd.** The helper is called only inside existing `backward()` bodies, under no-grad unless `create_graph` is set, which training never does. No new `autograd.Function` is needed.

**torch.compile and graphs.** These paths are not compiled. If they were, the helper is pure aten (view/bmm/sum/addmm), shape-dynamic in T, with one Python branch on `r`. It is not CUDA-graph-safe across varying T, and neither is today's code.

**Memory.** Transient extras are `part` (s·D·O bf16) plus `acc` (D·O fp32). The worst candidate is G20 s=16: 32 MB + 4 MB. That is under 40 MB per call and freed immediately.

**Prototype.** `/home/chcai/m5_work/proto/test_splitt.py` ran in the training image on CPU, with no `--device`, torch 2.11.0+rocm7.14. Output is in `proto/test_splitt.out`.
- Bit-exact against an exact emulation on integer data, with tails 0/3/4 and s=3..16, including D=24.
- Zero-copy views confirmed.
- Every fallback path confirmed.
- Autograd to an fp32 master weight: relative error 2.3e-3 against fp64.

---

## 3. Interception: exact code (all OFF by default)

### 3.1 `dlrm_v4/train/_env_bootstrap.py`, `apply_hipblaslt_opts` (line 86)

Add these parameters:

```python
    splitt_wgrad_hstu: bool = False,        # OPT M5: G20, G19 split-T (strided-batched bmm + fp32 reduce)
    splitt_wgrad_preproc: bool = False,     # OPT M5: G25, G27, G29, G31
    splitt_wgrad_splits: Optional[Dict[str, int]] = None,   # {"2048x512": 8, ...}; key = hipBLASLt MxN
    splitt_min_rows: int = 4096,
```

Body, after the HSTU_TN_WGRAD lines:

```python
    for scope, tn, st in (("hstu", tn_wgrad_hstu, splitt_wgrad_hstu), ("preproc", tn_wgrad_preproc, splitt_wgrad_preproc)):
        if tn and st:
            raise ValueError(f"apply_hipblaslt_opts: tn_wgrad_{scope} and splitt_wgrad_{scope} are exclusive")
    smode = {(False, False): "0", (True, False): "hstu", (False, True): "preproc",
             (True, True): "all"}[(bool(splitt_wgrad_hstu), bool(splitt_wgrad_preproc))]
    _set_env("HSTU_SPLITT_WGRAD", smode)
    if smode != "0":
        sp = dict(splitt_wgrad_splits or {})
        for k, v in sp.items():
            if not re.fullmatch(r"[1-9]\d*x[1-9]\d*", str(k)) or int(v) < 1:
                raise ValueError(f"splitt_wgrad_splits: bad entry {k!r}: {v!r}")
        _set_env("HSTU_SPLITT_WGRAD_SPLITS", ",".join(f"{k}:{int(v)}" for k, v in sorted(sp.items())))
        _set_env("HSTU_SPLITT_MIN_ROWS", str(int(splitt_min_rows)))
```

This needs `import re` and `Dict`. As with the TN switch, a pre-set environment variable wins.

### 3.2 `gin/yambda_5b.gin`, after line 51

```
# M5 split-T NT weight gradients. Needs lib dir rocm-10.2_nnonly_v3/s103 or /s102 (docs: m5 DESIGN)
apply_hipblaslt_opts.splitt_wgrad_hstu = False
apply_hipblaslt_opts.splitt_wgrad_preproc = False
apply_hipblaslt_opts.splitt_wgrad_splits = {}
```

Add the same three `= False/{}` lines to m0..m4 overlays for explicitness. New overlays go in `gin/mi450_hipblaslt/m5_s103.gin` and `m5_s102.gin`; drafts are at `/home/chcai/m5_work/gpu_kit/`.

### 3.3 `ops/triton/triton_addmm.py`, after line 1641

```python
# OPT M5 (split-T, /home/chcai/m5_work/DESIGN.md): "0" (default) = original calls.
_SPLITT_MODE: str = os.environ.get("HSTU_SPLITT_WGRAD", "0").strip().lower()
if _SPLITT_MODE not in ("0", "1", "all", "hstu", "preproc"):
    raise ValueError(f"HSTU_SPLITT_WGRAD={_SPLITT_MODE!r}")
_SPLITT_ON: bool = torch.version.hip is not None and _SPLITT_MODE != "0"
_SPLITT_HSTU: bool = _SPLITT_ON and _SPLITT_MODE in ("1", "all", "hstu")
_SPLITT_PREPROC: bool = _SPLITT_ON and _SPLITT_MODE in ("1", "all", "preproc")
if (_SPLITT_HSTU and _TN_HSTU) or (_SPLITT_PREPROC and _TN_PREPROC):
    raise ValueError("HSTU_SPLITT_WGRAD and HSTU_TN_WGRAD overlap in scope")

def _parse_splitt(spec: str) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    for item in filter(None, (x.strip() for x in spec.split(","))):
        shape, s = item.split(":")
        m, n = (int(v) for v in shape.lower().split("x"))
        if m < 1 or n < 1 or not 1 <= int(s) <= 4096:
            raise ValueError(f"HSTU_SPLITT_WGRAD_SPLITS entry {item!r}")
        out[(m, n)] = int(s)
    return out

_SPLITT_SPLITS = _parse_splitt(os.environ.get("HSTU_SPLITT_WGRAD_SPLITS", "")) if _SPLITT_ON else {}
_SPLITT_MIN_ROWS: int = int(os.environ.get("HSTU_SPLITT_MIN_ROWS", "4096"))
_SPLITT_SEEN: set = set()

def splitt_wgrad_mm(a: torch.Tensor, b: torch.Tensor) -> Optional[torch.Tensor]:
    """a.t() @ b (a [T,D], b [T,O] token-major) as s strided-batched NT GEMMs over T-chunks + fp32 sum.
    hipBLASLt sees m=O n=D batch=s K=T//s (flattened GridBased key: (O*s,D) if O>D else (O,D*s)).
    Returns None if not applicable; the caller then issues the original call."""
    if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[0] or not a.is_cuda:
        return None
    T, D = a.shape
    O = b.shape[1]
    s = _SPLITT_SPLITS.get((O, D), 1)
    if (s <= 1 or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
            or a.stride() != (D, 1) or b.stride() != (O, 1) or D % 8 or O % 8
            or a.data_ptr() % 16 or b.data_ptr() % 16):
        return None
    L = T // s
    if L < _SPLITT_MIN_ROWS or L * max(D, O) >= (1 << 31):   # hipBLASLt batch strides are u32 kernel args
        return None
    main = s * L
    if (O, D, s) not in _SPLITT_SEEN:
        _SPLITT_SEEN.add((O, D, s))
        logger.info("M5 split-T wgrad %dx%d s=%d (T=%d, tail=%d)", O, D, s, T, T - main)
    part = torch.bmm(a[:main].view(s, L, D).transpose(1, 2), b[:main].view(s, L, O))
    acc = part.sum(dim=0, dtype=torch.float32)
    if main < T:
        acc.addmm_(a[main:].t().float(), b[main:].float())
    return acc.to(a.dtype)

_splitt_wgrad_mm_hstu = splitt_wgrad_mm if _SPLITT_HSTU else None
```

This uses the module's `logging` logger; add one if missing.

**`tn_wgrad_mm` (1644).** New signature `tn_wgrad_mm(a, b, enabled=True, splitt=False)`. Its first statements:

```python
    if splitt and a.is_cuda:
        dw = splitt_wgrad_mm(a, b)
        if dw is not None:
            return dw
```

The rest is unchanged. With `splitt=False`, the path is identical.

**`_TNWgradLinearFn.backward` (1684).**
- Change to `dw = tn_wgrad_mm(dy2, x2, enabled=_TN_PREPROC, splitt=_SPLITT_PREPROC)`.
- The explicit `enabled=_TN_PREPROC` keeps M3-only (`hstu`) from turning on TN here once M5-preproc routes through this Function.
- When M4 alone is on, `_TN_PREPROC` is True: same as today.

**`TNWgradLinear.forward` (1701).**
- Change to `if not ((_TN_PREPROC or _SPLITT_PREPROC) and x.is_cuda): return super().forward(x)`.
- When off, it is exactly `nn.Linear`.
- When on, forward is the same `F.linear`. dx = `dy2.mm(w)` and db = `dy2.sum(0)` are the same ops aten's addmm backward issues.

**`triton_addmm_bwd` (1723-1727).** This must stay self-contained, for the AST test:

```python
    _splitt = globals().get("_splitt_wgrad_mm_hstu")
    dw = _splitt(x, dz) if _splitt is not None and dz.is_cuda else None
    if dw is None:
        if globals().get("_TN_HSTU", False) and dz.is_cuda:
            dw = torch.mm(x.t().contiguous(), dz.t().contiguous().t())
        else:
            dw = torch.mm(x.t(), dz)
```

In the test namespace this resolves to the original `torch.mm`.

### 3.4 `ops/triton/triton_hstu_linear.py`

- Import `_SPLITT_HSTU` at lines 33-37.
- Line 2385 becomes `tn_wgrad_mm(y, dout, enabled=_TN_HSTU, splitt=_SPLITT_HSTU)`.

### 3.5 Bit-identity when off

`HSTU_SPLITT_WGRAD` unset or `"0"` means:
- `_SPLITT_*` are False and `_splitt_wgrad_mm_hstu` is None;
- every changed line reduces to the current expression and the same aten call;
- the parser is not run.

The only behavioural addition is the `ValueError` on an invalid env value. Unit tests to add, which can run on CPU:
- the AST test above;
- parser and validation cases;
- a CPU version of `proto/test_splitt.py`, with the CUDA guard monkeypatched.

---

## 4. Catalog: flattened keys, today's selection, and the v3 entries

GridBased flattening rule (`PropertyMatching.hpp` L1308-1330, modelled by `gridmatch.py`): `(m,n,b)` becomes `(m·b, n)` if m > n, else `(m, n·b)`. The first pass with the unflattened key lands on a batch-1 entry, which forces a restart with the flattened key.

Selection, from gridmatch with the real predicates (lda=m, ldb=n), for T ∈ {1.0M, 1.59M, 2.097M, 2.533M, 3.2M} and K = T//s:

| Shape (m×n) | s | Flattened key | v2 today | v3/s103 | v3/s102 | WG on 103 / on 102 |
|---|---|---|---|---|---|---|
| G20 2048×512 | 2, 4, 8, 16 | (4096,512) (8192,512) (16384,512) (32768,512) | 102 | 103 | 102 | 64/128/256/512 · 4096…32768 |
| G19 512×1536 | 2, 4, 8, 10, 16, 20 | (512,3072) (512,6144) (512,12288) (512,15360) (512,24576) (512,30720) | 102 | 103 | 102 | 48…480 · 3072…30720 |
| G25 256×512 | 4…128 | (256,2048) … (256,65536) | **103** (clamp onto stub [256,256,1,512]) | 103 | 102 | 16…512 · 1024…32768 |
| G29 1024×256 | 2…64 | (2048,256) … (65536,256) | 102 | 103 | 102 | 16…512 · 1024…32768 |
| G31 512×256 | 4…128 | (2048,256) … (65536,256) (**same keys as G29 at s/2**) | 102 | 103 | 102 | 16…512 · 1024…32768 |
| G27 24×256 | 16…256 | (24,4096) … (24,65536) | 102 | 103 | 102 | 32…512 · 256…4096 |

The per-T detail is in `verify_s103.txt` / `verify_s102.txt` section 1.

**Builder.** `/home/chcai/m5_work/make_v3.py` starts from the **v2** table, which has 54 entries.

1. **Pins.** Each flattened key `[M,N,1,1]` maps to the variant's sol. The six batch-1 shapes are pinned too: already 103 in v2, and flipped to 102 in s102, where G25 batch-1 gets a new pin.
2. **Guards, so nothing else changes.**
   - New M blocks (4096…65536) get `[M-1,1,1,1]` and `[M,SENT,1,1]`. Both take the old value, which is 102 everywhere above m = 2048.
   - Inside each block, `[M,N-1,1,1]` takes the old selection.
   - `[M,SENT,1,1]` is added when the new key would become the block's clamp target. For G25 that is `[256,1048576]`.
   - Every guard splits only single-entry old groups, so it is K-independent. The builder checks this.
3. **strict mode** (automatic for any shape mapped to 102). Guards around 102 pins are themselves 102. So in s102, G25's region (m ∈ (128,256], n > 256) is 102 at every s, including s not in the list. s102 therefore never selects 103 for any of the six shapes, batched or not, for s = 2..256 (section 1b of the report).

**Results.** Each table goes from 54 to 121 entries.

| Variant | Edits | sha256 of NT .dat |
|---|---|---|
| s103 | 67 added | `a76b0b66…ca20` |
| s102 | 61 added + 6 flips (T8's five 103 pins and G25 batch-1 → 102) | `b205ef00…30a4` |

Verification is in `verify_{s103,s102,mix}.txt`, and every report says PASS.
- `make_v2.validate`: msgpack/zlib round trip, identical solutions/rows/predicates, sorted unique keys, no multi-batch last N group, sol 102/103 kernel and `.kd` symbols present in the 10.2 `.co`.
- **Boundary grid** of 151,704 points (every table key coordinate ±1 × K boundaries):
  - s103: exactly 21 (m,n) change, all of them pinned flattened keys, 102 → 103;
  - s102: only 103 → 102 changes (pins and G25's strict region);
  - mix: 78 (m,n) change. 102 → 103 happens only on pinned G20/G19 keys; everything else is 103 → 102 (pins and the strict G25 region).
- **300k random sweep** (m,n up to 2²¹, b up to 256, K up to 2²²):
  - s103: 0 differences;
  - s102 and mix: 38 each. All are 103 → 102: G25's strict region, or (24,32,b8), which flattens onto the pinned (24,256). There is no new 103.
- Workload union file: no unintended change. G17 (516×512×1024) and G23 (512×512, b8; flattens to (512,4096) → guard 102) stay 102.
- In s103, any s not in the candidate list falls back to 102: slower, but correct. The exception is G25, which v2 already clamps to 103.

**Lib dirs, installed.** `/home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v3/{s103,s102}/`. Each is a v2 copy: 405 symlinks into /opt/rocm-10.2, plus the new NT `.dat` and a README. In containers they are `/armlibs/rocm-10.2_nnonly_v3/sXXX` (run_ab.sh mounts `$LIBS` at /armlibs). Rebuild with `make_v3.py --sol N --install` after removing the dir.
- A per-shape mix needs `--map …` and a new name.
- G29 at s and G31 at 2s share keys, so their sols must agree. The builder enforces this.

**Superseded.** `m5_catalog.py`, `m5_catalog.txt` and `v3_candidate/` are from an earlier partial pass and are replaced by `make_v3.py`.

---

## 5. bf16-out bmm + fp32 sum, versus `bmm(out_dtype=float32)`

### 5.1 What each path selects

Decoded with msgpack from `/opt/rocm-10.2/lib/hipblaslt/library/gfx1250/`. v1 and v2 symlink the BB_SB catalog unchanged.

- **f32-out:** `torch.bmm(..., out_dtype=torch.float32)`, which is CUDA-only (`aten::bmm.dtype`). bf16 A/B with f32 C/D means `TensileLibrary_BB_SB_HA_Bias_SAV_UA_Type_BS_HPA_Contraction_l_Ailk_Bjlk…`. That catalog is GridBased with 2 solutions and table `[1,1,1,1]→1421, [128,128,1,128]→1420`. Every M5 problem, at every s and T, and every batch-1 wgrad, selects **sol 1420**:
  - MT256x128x64, GRVW A/B = **1/1**, SK0, GSU1;
  - **1020 VGPRs** (`notes_bbsb.txt`), more than 103's 676;
  - never run or measured on a37-1.

  The alternative 1421 is MT16x16x64, GRVW 1/1, 222 VGPRs. Steering to it would need a second edited catalog and an unknown, likely poor, GRVW-1 performance.
- **bf16-out:** the BB_BB NT catalog. That is the table T8 already controls, with exactly the two kernels A0 characterised (103 at 676 VGPRs, 102 at 252) and the measured timings.

### 5.2 Numerics

CPU emulation of the GPU arithmetic: bf16 products are exact in fp32, accumulation is fp32, and the store rounds once. T = 2²⁰, compared against fp64. Output is in `proto/test_splitt.out`.

| Data | today, rel rms | f32 partials | bf16 partials (chosen) |
|---|---|---|---|
| zero-mean x and dY (maximum cancellation) | 1.67e-3 | 1.00× | 1.41× at s = 4…256 |
| silu x, biased dY | 1.66e-3 | 1.00× | 1.17× (s=4) down to 1.00× (s=256) |
| 2 % of tokens carry dY | 1.67e-3 | 1.00× | 1.41× |

Max-abs relative error goes from at most 3.5e-3 to at most 4.5e-3. The extra error is one more bf16 rounding (partials), and it does not grow with s: Σ‖P_i‖² ≈ ‖ΣP_i‖² for weakly correlated chunks. That is well inside bf16 gradient noise, and the step goes to fp32 Adam.

**CORRECTION (review, 2026-10-11).** The paragraph above holds only for chunk-independent data. Each partial
is rounded at its own magnitude, so the extra rms error is ≈ u·‖sqrt(Σ P_i²)‖ and the ratio to today's is
≈ sqrt(1+q²) with q = ‖sqrt(Σ P_i²)‖/‖Σ P_i‖. With dY drifting from −1 to +1 along T (partials large,
total ≈ 0) the CPU study measures 451× (s=2), 328× (s=8), 111× (s=64), 57× (s=256); opposite-sign halves
459×/417×/137×/72×; per-512-token-user sign flips and chunk-varying scale stay at 1.4–1.5×
(`proto/numerics_patterns.out`). Absolute error stays ≤ ~u·Σ|P_i|. Whether real gradients cancel across
T-chunks is unknown; `splitt_audit_every` logs q in training, and `splitt_loop.py --pattern drift` checks the
GPU kernels against the model.

**Decision:** bf16-out. It is controllable through the same catalog, uses lower-VGPR kernels, and has measured baselines. The cost is a 1.41× rms gradient rounding error.

Revisit f32-out only if the GPU numerics stage (section 9) shows a loss or grad-norm deviation. That would need a BB_SB v3 edit and a single-GEMM safety check of 1420/1421 first.

---

## 6. Kernel and predicate checks for strided batches

**Batch-stride arithmetic.** In Tensile `KernelWriterAssembly.computeLoadSrd` (5911365c, L5097-5141), the batch offset is `s_mul_u64_u32(Stride, WorkGroup2)` into a 64-bit tileStart. It is then scaled to bytes with `scalarMultiply64Bpe` and added to the 64-bit SRD base. The product is never truncated.

Only the stride itself is a u32 kernel argument (`strideA1`, `strideB1` in `k103` metadata, `notes_bbsb.txt`), so the requirement is **L·max(D,O) < 2³²**.

For the candidates at T = 2,533,005:

| Shape, s | Max stride (elements) |
|---|---|
| G20, s=4 | 1.30e9 |
| G19, s=2 | 1.95e9 |
| G20, s=2 | 2.59e9 |

All are under 2³² = 4.29e9. The helper still refuses ≥ 2³¹ as a margin against signed handling anywhere on the host side. That disables only G20 s=2, for T ≥ 2,097,152, and leaves G20's useful s (4–16) valid up to T ≈ 4.19M.

**Buffer limits.** SRD limits use 64-bit `ShadowLimit`. Within a batch, lda·K is smaller than today's batch-1 lda·T, which sol 103 already handles.

**Predicates.** The modelled problem predicates all accept every candidate: `StridedBatched: true` in both 102 and 103, WorkgroupNumberCheck (worst 102 case 32,768 WG, far below 16.7M), and the BufferLoad offset checks, which depend on lda·DU only. GSU=1, so there is no workspace.

**Alignment.** The batch base offsets are L·D·2 B, a multiple of 16 B because D % 8 = 0. The helper also checks base pointer % 16.

**Not yet exercised on gfx1250.** The WG2 > 0 (batched) path of sols 102/103. The model's only batched bf16 NT GEMM today is G23 (b=8, on 102). The GPU stage must test batched 103 and 102 specifically, including zero-DY.

**Torch route.** `bgemm<bf16>` on ROCm goes to hipBLASLt when the preferred backend is Cublaslt, which G23 in the capture shows. If hipBLASLt returns no algorithm, torch silently falls back to rocBLAS. The GPU stage must confirm the kernel names: profiler output and the `print_kernel_info` lines.

---

## 7. Per-call and per-step estimates (inference, not measured)

**Basis.** a37-1 at T = 2,097,152:
- sol 103 ≈ 23.8 ms per call (G25), independent of m and n, with at most 256 WGs at 1 WG/CU (676 VGPRs);
- sol 102: G27 28.4 ms (16 WG), G31 29.6 (256), G29 29.7 (512), G19 31.9 (1536), G20 57.0 (2048).

**Model.**
- 103: t ≈ ⌈WG/256⌉ · 23.8 ms / s.
- 102: t ≈ max(⌈WG·s/C⌉ · 28.4/s, WG₁ · 31.9/1536) with concurrency C ≈ 1536. This fits all five 102 points: G20 = 2 waves × 28.4.
- Reduction: s·D·O bf16 read, under 17 MB per call, plus 2–6 small launches, ≈ 0.02 ms/call.

| Shape ×calls | T8 today (103) | s103 (s) | ms/call | s102 (s) | ms/call |
|---|---|---|---|---|---|
| G20 ×3 | 24 | 8 (256 WG) | ≈3.0 (FLOP floor ≈1.9, HBM floor ≈1.3) | 4 | ≈43 (or s=1: 57) |
| G19 ×3 | 24 | 10 (240 WG) | ≈2.4 | 1 | 31.9 |
| G25 ×3 | 23.8 | 64 | ≈0.37 | 4–6 | ≈5–7 |
| G29 ×1 | 24 | 32 | ≈0.74 | 2–3 | ≈10–14 |
| G31 ×1 | 24 | 64 | ≈0.37 | 4–6 | ≈5–7 |
| G27 ×1 | 24 | 128 | ≈0.19 | 64–96 | ≈0.3–0.45 |
| **Per step** | **≈288** | | **≈19** | | **≈263** |

For comparison:
- M1 (v1): ≈426 ms/step.
- Mix (G20/G19 on s103-split, the rest on s102-split): ≈55 ms/step, with sol-103 launches cut from 12 to 6.

The lower bound for all six is about 18 ms/step. s103 should be close to it unless L2/HBM contention at full-chip occupancy slows the per-WG rate (G20 reads ≈52 GB through L2 per call). The 102 numbers are least certain, because C and the throughput floor are fitted from 5 points.

---

## 8. CWSR / A0 exposure (sol 103 = the A0 DW-corruption kernel on a37-1's faulty handler)

| | 103 launches/step | 103 WG·ms/step | 103 wall time/step | Resident 103 WGs during it |
|---|---|---|---|---|
| T8 (v2) | 12 | ≈4,660 | ≈288 ms | 2–32 |
| M5 s103 | 12 | ≈4,660 (same work) | ≈18 ms | ≈240–256 (every CU) |
| Mix | 6 | ≈4,030 | ≈16 ms | ≈256 |
| M5 s102 | **0** | 0 | 0 | — |

How the risk moves under s103:
- If corruption scales with 103 wave-time, M5 changes nothing.
- If it scales with the chance that a preemption or compaction event lands while 103 is resident, the window shrinks about 16×, but each hit then saves and restores about 8–128× more 676-VGPR waves. That is a different, untested regime.
- A0 per-call failure rates (call 196) may not carry over to batched launches.

The memory note matters too: queue eviction by compaction is the trigger, and BLOCK_N=128 faulted 2/2.

**Recommendation.** Treat s103 like T8: only guarded ≤200-step A/Bs after the batched zero-DY stage passes. s102 is the variant to use if 103 stays unsafe.

---

## 9. GPU kit and bench plan (after the power cycle, a37-1 rules: one GEMM per process, live dmesg watch)

1. **Single GEMM, both lib dirs.** Run
   `gemm_watch.sh /home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v3/s103 <dev> <out> /home/chcai/m5_work/gpu_kit/m5_bench.yaml`
   and the same with `/s102`. There are 44 lines (index in `m5_bench.index.txt`):
   - per shape: the s=1 baseline;
   - every candidate s as the exact strided-batched problem (M=O, N=D, K=T/s, stride_a=O·K, stride_b=D·K, stride_c=O·D, batch_count=s);
   - the fp32 tail GEMM.

   G20 s=2 (stride 2³¹) is left out on purpose. Check the sol per line: MT256x128x64 on s103, MT32x16x32 on s102, MT32x32x16 for the tail. Record µs per line. Pick s per shape per variant: the minimum µs, then prefer fewer WGs on 103.
2. **Numerics, torch, one shape per process.** Adapt `/home/chcai/t8_work/gpu_check/zero_dy_loop.py` and `run_zero_dy.sh` to call the M5 helper with the chosen (shape, s), for both variants:
   - zero-DY, 1000 calls: every word must be +0. These are the batched WG2 > 0 kernels;
   - random: each call bit-identical to call 1, and rel-rms against fp64 at most 1.6× the batch-1 kernel's (section 5.2 predicts 1.41×);
   - print the profiler kernel names on call 1. This also catches a silent rocBLAS fallback.
3. **E2E.** Implement section 3, set the chosen s in `m5_s103.gin` / `m5_s102.gin` (or a mix overlay built with `make_v3.py --map`), and run:
   `run_ab.sh M5s102 /m5/m5_s102.gin -v /home/chcai/m5_work/gpu_kit:/m5:ro`
   and the same for s103. 200 steps each, against the T8 and M1 arms. Compare step time, loss curve and grad finiteness. If loss deviates beyond the T8 arm, go to the f32-out fallback in section 5.

---

## 10. Files

| Path | What |
|---|---|
| `/home/chcai/m5_work/DESIGN.md` | This document |
| `/home/chcai/m5_work/make_v3.py` | Catalog builder and verifier (`--sol 102\|103`, `--map`, `--install`) |
| `/home/chcai/m5_work/verify_s103.txt`, `verify_s102.txt`, `verify_mix.txt` | Full edit lists, selections and sweeps |
| `/home/chcai/m5_work/build/{s103,s102,mix}/…NT.dat.zlib` | Built catalogs |
| `/home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v3/{s103,s102}/` | Installed lib dirs (README inside) |
| `/home/chcai/m5_work/proto/test_splitt.py`, `test_splitt.out` | CPU prototype, contract tests, numerics study |
| `/home/chcai/m5_work/gpu_kit/m5_bench.yaml`, `m5_bench.index.txt`, `gen_yaml.py` | hipblaslt-bench lines for gemm_watch.sh |
| `/home/chcai/m5_work/gpu_kit/m5_s103.gin`, `m5_s102.gin` | Draft run_ab overlays |
| `/home/chcai/m5_work/notes_bbsb.txt`, `notes_102.txt`, `k103.txt`, `tmp_ss/` | Kernel metadata: VGPRs 1420=1020, 1421=222, 103=676, 102=252, 2821=222 |
