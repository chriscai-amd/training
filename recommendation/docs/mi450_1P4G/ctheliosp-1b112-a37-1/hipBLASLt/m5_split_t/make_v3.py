#!/usr/bin/env python3
"""M5 (split-T NT weight gradients): build + verify the v3 bf16 NT GridBased catalog. CPU-only.

M5 replaces each tall-K NT wgrad  dW = a.t() @ b  (a [T,D], b [T,O], hipBLASLt m=O n=D batch 1 K=T)
by one strided-batched GEMM  bmm(a3.transpose(1,2), b3)  (hipBLASLt m=O n=D batch=s K=T//s) plus an
fp32 reduction over s. hipBLASLt GridBased flattens batch (PropertyMatching.hpp L1308-1330):
key (m,n,s) -> (m*s, n) if m > n else (m, n*s). v2 sends those flattened keys to guard rows (102),
except G25's (256, 512*s), which clamps onto the stub's [256,256,1,512] -> 103.

Variants (one catalog per variant; same table structure, only the target index differs):
  --sol 103   v2 table (T8 batch-1 pins kept) + every M5 flattened key -> 103
  --sol 102   v2 table with the six batch-1 wgrad pins set to 102 (T8 undone; G25 batch-1 pinned 102)
              + every M5 flattened key -> 102. With this catalog no bf16 NT wgrad (batched or not) uses 103.
  --map G20=103,G19=103,G25=102,...   per-shape mix (flattened-key collisions G29 s <-> G31 2s
              must agree; checked). Batch-1 pins follow the shape's sol.
Guards: every added key K=(M,N) gets neighbour entries whose index is the OLD selection there, so the
only (m,n) points whose selection changes are the pinned keys themselves (verified exhaustively on
the boundary grid below + random sweep).

Usage
  python -I make_v3.py --sol 103                 build catalog into /home/chcai/m5_work/build/s103/ + verify
  python -I make_v3.py --sol 102 --install       also create .../arbor_libs/rocm-10.2_nnonly_v3/s102/
                                                 (v2 tree: same symlinks, new NT .dat, README,
                                                 m5_splits.json = per-shape sol for every s 1..256,
                                                 checked by triton_addmm at import)
  python -I make_v3.py --sol 103 --manifest-only (re)write m5_splits.json of an installed variant
"""
import argparse
import hashlib
import itertools
import math
import os
import random
import re
import shutil
import sys

sys.path.insert(0, "/home/chcai/t8_work")
import gridmatch as g  # noqa: E402
import make_v2 as mv  # noqa: E402

V2 = "/home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v2"
V3 = "/home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v3"
V2_DAT = os.path.join(V2, mv.GFX_REL, mv.NT_DAT)
V1_DAT = os.path.join(mv.V1, mv.GFX_REL, mv.NT_DAT)
BBSB_DAT = ("/opt/rocm-10.2/lib/hipblaslt/library/gfx1250/TensileLibrary_BB_SB_HA_Bias_SAV_UA_Type_BS_"
            "HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib")
UNION = ("/home/chcai/training/recommendation/docs/mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/"
         "dlrmv4-BF16-allgemms-union.txt")
SENT = 1 << 20
TS = [1_000_000, 1_590_000, 2_097_152, 2_533_005, 3_200_000]

# (name, hipBLASLt m = O = width of torch operand b, n = D = width of a, calls/step, candidate splits)
SHAPES = [("G20", 2048, 512, 3, (2, 4, 8, 16)),
          ("G19", 512, 1536, 3, (2, 4, 8, 10, 16, 20)),
          ("G25", 256, 512, 3, (4, 8, 16, 32, 64, 128)),
          ("G29", 1024, 256, 1, (2, 4, 8, 16, 32, 64)),
          ("G31", 512, 256, 1, (4, 8, 16, 32, 64, 128)),
          ("G27", 24, 256, 1, (16, 32, 64, 128, 256))]


def flat(m, n, b):
    return (m * b, n) if m > n else (m, n * b)


def wg(mt, m, n, b):
    return math.ceil(m / mt[0]) * math.ceil(n / mt[1]) * b


class Sel:
    """Pre-sorted table + gridmatch binary path (identical to g.select, without re-sorting per call)."""

    def __init__(self, cat, table_only=False):
        self.cat = cat
        self.table_only = table_only   # True: every solution accepts (pure table semantics)
        self.table = g.sort_table(g._norm_entries(g.gridbased_rows(cat)[0][2]))
        self._acc = {}

    def __call__(self, m, n, b, k, lda=None, ldb=None):
        key = (lda, ldb)
        if key not in self._acc:
            self._acc[key] = ((lambda i, p: True) if self.table_only
                              else g.nt_size_accepts(self.cat, lda=lda, ldb=ldb))
        r = g._binary(self.table, [m, n, b, k], self._acc[key], [m, n, b, k], 1, False, None)
        return r[0] if r else None


def pins_for(solmap):
    """{(M,N): sol} = batch-1 wgrad keys + all flattened M5 keys."""
    pins, owner = {}, {}
    for nm, m, n, _, ss in SHAPES:
        for key in [(m, n)] + [flat(m, n, s) for s in ss]:
            want = solmap[nm]
            if key in pins and pins[key] != want:
                raise SystemExit("flattened-key collision %s: %s wants %d, %s wants %d" % (
                    key, owner[key], pins[key], nm, want))
            pins[key] = want
            owner.setdefault(key, []).append(nm)
    return pins, owner


def build_table(v2cat, pins, strict=False):
    """v2 table + pins + guards. Guard index = OLD (v2) selection at the guard key; the old group a
    guard splits must be single-entry (K-independent), else abort."""
    old = Sel(v2cat, table_only=True)   # guards follow the table; predicates are checked by the sweeps
    t2 = g.gridbased_rows(v2cat)[0][2]
    have = {tuple(e["key"]): e["index"] for e in t2}
    blocks = {}
    for (M, N, B, K) in have:
        blocks.setdefault(M, set()).add(N)
    new = dict(have)
    log = []

    def old_sel_const(m, n):
        r = {old(m, n, 1, k) for k in (1, 64, 512, 65536, 1_000_000, 2_097_152, 3_200_000)}
        if len(r) != 1:
            raise SystemExit("old selection at (%d,%d) is K-dependent: %s" % (m, n, r))
        return r.pop()

    for (M, N), sol in sorted(pins.items()):
        k1 = (M, N, 1, 1)
        if k1 in new:
            if new[k1] != sol:
                log.append(("flip", k1, new[k1], sol))
                new[k1] = sol
            continue
        if M not in blocks:
            # new M block: everything m in (prevM, M-1] keeps its old result through an [M-1,1] block,
            # which is only valid if the old selection there is one constant for every n.
            prevM = max([x for x in blocks if x < M] or [0])
            vals = {old_sel_const(m, n) for m in (prevM + 1, M - 1, M) for n in (1, 255, 256, 257, 512,
                                                                                   1536, 4096, N - 1, N, N + 1, SENT, SENT + 1)}
            if len(vals) != 1:
                raise SystemExit("new block M=%d: old selection not constant %s" % (M, vals))
            v = vals.pop()
            for gk in [(M - 1, 1, 1, 1), (M, SENT, 1, 1)]:
                if gk not in new:
                    new[gk] = v
                    log.append(("guard", gk, None, v))
            blocks[M - 1] = {1}
            blocks[M] = {SENT}
        # N-side guards inside block M
        gk = (M, N - 1, 1, 1)
        if gk not in new and (M, N - 1) not in pins:
            new[gk] = 102 if strict and sol == 102 else old_sel_const(M, N - 1)
            log.append(("guard", gk, None, new[gk]))
        if N > max(blocks[M]):                       # would become the clamp target for n > N
            gk = (M, SENT, 1, 1)
            if gk not in new:
                new[gk] = 102 if strict and sol == 102 else old_sel_const(M, SENT + 1)
                log.append(("guard", gk, None, new[gk]))
        new[k1] = sol
        log.append(("target", k1, None, sol))
        blocks[M] |= {N - 1, N}
    table = [{"key": list(k), "index": v} for k, v in sorted(new.items())]
    return table, log


def build(solmap, outdir, strict=False):
    v2 = g.load_catalog(V2_DAT)
    pins, owner = pins_for(solmap)
    table, log = build_table(v2, pins, strict)
    v3 = g.load_catalog(V2_DAT)
    pos = g.gridbased_rows(v3)[0][0]
    v3["library"]["rows"][pos]["library"]["table"] = table
    os.makedirs(outdir, exist_ok=True)
    dat = os.path.join(outdir, mv.NT_DAT)
    with open(dat, "wb") as f:
        f.write(mv.pack(v3))
    return v2, v3, dat, pins, owner, log


def verify(v2, v3, pins, owner, solmap, out, strict=False):
    P = lambda *a: print(*a, file=out)  # noqa: E731
    s2, s3 = Sel(v2), Sel(v3)
    sm = {s["index"]: s["sizeMapping"]["macroTile"] for s in v3["solutions"]}
    ok = True
    P("\n== 1. M5 problems (m, n, batch=s, K=T//s; lda=m, ldb=n): v2 vs v3, T in %s ==" % TS)
    for nm, m, n, cps, ss in SHAPES:
        b1 = {s3(m, n, 1, T) for T in TS}
        want = solmap[nm]
        ok &= b1 == {want}
        P("  %s m=%d n=%d calls/step=%d | batch-1: v2 %s  v3 %s | WG/batch: 102=%d 103=%d" % (
            nm, m, n, cps, sorted({s2(m, n, 1, T) for T in TS}), sorted(b1),
            wg(sm[102], m, n, 1), wg(sm[103], m, n, 1)))
        for s in ss:
            r2 = {s2(m, n, s, T // s, m, n) for T in TS}
            r3 = {s3(m, n, s, T // s, m, n) for T in TS}
            good = r3 == {want}
            ok &= good
            P("    s=%-4d flat key %-15s K=T/s %8d..%8d  v2=%-6s v3=%-6s WG(v3 sol)=%-6d %s" % (
                s, "(%d,%d)" % flat(m, n, s), TS[0] // s, TS[-1] // s, sorted(r2), sorted(r3),
                wg(sm[want], m, n, s), "OK" if good else "FAIL"))
    P("\n== 1b. every s in 2..256 (also non-candidates), T=2,097,152: sol selected in v3 ==")
    for nm, m, n, _, ss in SHAPES:
        by = {}
        for s_ in range(2, 257):
            by.setdefault(s3(m, n, s_, 2_097_152 // s_, m, n), []).append(s_)
        P("  %s: %s" % (nm, "; ".join("sol %s for s in %s" % (k, v if len(v) < 40 else "%d values (%d..%d)" % (len(v), v[0], v[-1]))
                                    for k, v in sorted(by.items()))))
        if strict and solmap[nm] == 102:
            ok &= set(by) == {102}           # a 102-mapped shape is 102 at EVERY s
    P("\n== 2. bf16 NT GEMMs of the workload union file, v2 vs v3 ==")
    for line in open(UNION):
        if not line.startswith("- {"):
            continue
        f = dict(re.findall(r"(\w+): ([^,}]+)", line))
        if (f["transA"], f["transB"], f["a_type"]) != ("N", "T", "bf16_r"):
            continue
        m, n, k, b = int(f["M"]), int(f["N"]), int(f["K"]), int(f["batch_count"])
        a, c = s2(m, n, b, k, int(f["lda"]), int(f["ldb"])), s3(m, n, b, k, int(f["lda"]), int(f["ldb"]))
        intended = b == 1 and (m, n) in pins
        if a != c and not intended:
            ok = False
        P("  m=%-5d n=%-5d b=%d k=%-8d v2=%s v3=%s%s" % (m, n, b, k, a, c,
          "" if a == c else ("  changed (intended: batch-1 pin)" if intended else "  !! UNINTENDED")))

    P("\n== 3. exhaustive boundary grid (batch 1; batched problems reduce to their flattened key) ==")
    keys = [tuple(e["key"]) for e in g.gridbased_rows(v2)[0][2]] + [tuple(e["key"]) for e in g.gridbased_rows(v3)[0][2]]
    Ms = sorted({max(1, x + d) for k in keys for x in [k[0]] for d in (-1, 0, 1)} | {SENT + 7})
    Ns = sorted({max(1, x + d) for k in keys for x in [k[1]] for d in (-1, 0, 1)} | {SENT + 7})
    Ks = sorted({max(1, k[3] + d) for k in keys for d in (-1, 0, 1)} | {1_000_000, 2_097_152})
    changed = {}
    pts = 0
    for m in Ms:
        for n in Ns:
            for k in Ks:
                pts += 1
                a, c = s2(m, n, 1, k), s3(m, n, 1, k)
                if a != c:
                    changed.setdefault((m, n), set()).add((a, c))
    # allowed: pinned keys; with --sol 102 (strict) also any 103 -> 102 change (the variant's purpose)
    bad = sorted(x for x, v in changed.items() if x not in pins and not (strict and all(c == 102 for _, c in v)))
    ok &= not bad
    P("  %d points (%d m x %d n x %d K); %d (m,n) changed, all pinned keys: %s" % (
        pts, len(Ms), len(Ns), len(Ks), len(changed), "YES" if not bad else "NO -> %s" % bad[:20]))
    for (m, n), v in sorted(changed.items()):
        P("    (%d,%d) %s  [%s]" % (m, n, sorted(v), ",".join(owner.get((m, n), ["?"]))))

    P("\n== 4. random sweep 300k (m,n log-uniform 1..2^21, b in 1,2,3,4,8,16,32,64,128,256, K 1..2^22) ==")
    rnd = random.Random(11)
    diff = []
    for _ in range(300_000):
        m = int(2 ** rnd.uniform(0, 21)); n = int(2 ** rnd.uniform(0, 21))
        b = rnd.choice([1, 1, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256]); k = int(2 ** rnd.uniform(0, 22))
        a, c = s2(m, n, b, k), s3(m, n, b, k)
        if a != c:
            fk = flat(m, n, b) if b > 1 else (m, n)
            diff.append((m, n, b, k, a, c, fk in pins or (strict and c == 102)))
    ok &= all(x[-1] for x in diff)
    P("  %d differ; all explained by a pinned (flattened) key: %s" % (len(diff), all(x[-1] for x in diff)))
    for x in diff[:20]:
        P("    m=%d n=%d b=%d k=%d v2=%s v3=%s pinned=%s" % x)

    P("\n== 5. general rule check: every (m, n, b) whose flattened key is a pin, sampled ==")
    cnt = 0
    for (M, N), sol in sorted(pins.items()):
        for b in (2, 4, 8):
            for m, n in ((M // b, N), (M, N // b)):
                if m * b == M and m > n and flat(m, n, b) == (M, N) or n * b == N and m <= n and flat(m, n, b) == (M, N):
                    r = s3(m, n, b, 100_000)
                    cnt += 1
                    ok &= r == sol
    P("  %d pre-images (m,n,b) of pinned keys all select the pinned sol: %s" % (cnt, ok))
    return ok


def bbsb_report(out):
    P = lambda *a: print(*a, file=out)  # noqa: E731
    cs = g.load_catalog(BBSB_DAT)
    S = Sel(cs)
    P("\n== 6. fp32-out alternative: torch.bmm(..., out_dtype=float32) -> unmodified 10.2 BB_SB NT catalog ==")
    P("  table: %s" % [(tuple(e["key"]), e["index"]) for e in g.gridbased_rows(cs)[0][2]])
    for s_ in cs["solutions"]:
        z = s_["sizeMapping"]
        P("  sol %d MT%dx%dx%d GSU=%s SK=%s GRVW A/B=%s/%s WG=%s  %s" % (
            s_["index"], z["macroTile"][0], z["macroTile"][1], z["depthU"], z["globalSplitU"],
            z["streamK"], z["grvwA"], z["grvwB"], z["workGroup"], s_["name"][:60]))
    for nm, m, n, _, ss in SHAPES:
        got = {S(m, n, s, T // s) for s in ss for T in TS}
        P("  %s all candidate s, all T -> %s ; batch-1 -> %s" % (nm, sorted(got), sorted({S(m, n, 1, T) for T in TS})))


README = """\
rocm-10.2_nnonly_v3/{name}: hipBLASLt lib dir for M5 (split-T NT weight gradients), variant {name}.
Identical to rocm-10.2_nnonly_v2 (same symlinks into /opt/rocm-10.2) EXCEPT the bf16 NT catalog
{dat}
whose GridBased table = v2 table + {nadd} added / {nflip} flipped entries:
{entries}
Solution map per wgrad shape: {solmap}
Effect (gridmatch.py, see /home/chcai/m5_work/verify_{name}.txt): the batch-1 wgrads and every
batched M5 problem (m, n, batch=s) with s in the candidate list select the mapped sol at any K; no
other (m, n) changes (exhaustive boundary grid + 300k random sweep). Candidate s per shape:
{cands}
UNTESTED ON A GPU. Sol 103 (676 VGPRs) is the A0 DW-corruption kernel under the faulty CWSR handler.
Built by /home/chcai/m5_work/make_v3.py. sha256 {sha}
Keep TENSILE_GRIDBASED_KDTREE / _BATCH_EXP / TENSILE_METRIC / TENSILE_SOLUTION_INDEX unset.
"""


MANIFEST = "m5_splits.json"


def manifest(cat, name, solmap, sha):
    """Machine-readable selection summary read by triton_addmm at import: for each wgrad shape the
    mapped sol and the sol the installed catalog selects for (m, n, batch=s) at s = 1..256 (lda=m,
    ldb=n, every T in TS; None if K-dependent)."""
    S = Sel(cat)
    shapes = {}
    for nm, m, n, cps, ss in SHAPES:
        by = {}
        for s_ in range(1, 257):
            r = {S(m, n, s_, T // s_, m, n) for T in TS}
            by[str(s_)] = r.pop() if len(r) == 1 else None
        shapes["%dx%d" % (m, n)] = {"name": nm, "sol": solmap[nm], "calls_per_step": cps,
                                    "candidates": list(ss), "sol_by_s": by}
    return {"m5_manifest_version": 1, "variant": name, "nt_dat_sha256": sha, "solmap": solmap,
            "shapes": shapes, "built_by": "/home/chcai/m5_work/make_v3.py"}


def write_manifest(libdir, name, solmap):
    import json
    dat = os.path.join(libdir, mv.GFX_REL, mv.NT_DAT)
    sha = hashlib.sha256(open(dat, "rb").read()).hexdigest()
    man = manifest(g.load_catalog(dat), name, solmap, sha)
    for key, ent in man["shapes"].items():
        bad = [s_ for s_ in ent["candidates"] if ent["sol_by_s"][str(s_)] != ent["sol"]]
        if bad or ent["sol_by_s"]["1"] != ent["sol"]:
            raise SystemExit("manifest %s: %s candidates %s / batch-1 do not select sol %d" % (libdir, key, bad, ent["sol"]))
    with open(os.path.join(libdir, MANIFEST), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print("wrote %s (sha %s)" % (os.path.join(libdir, MANIFEST), sha[:12]))


def install(name, dat, log, solmap):
    out = os.path.join(V3, name)
    if os.path.lexists(out):
        raise SystemExit("%s exists; remove it by hand first" % out)
    tmp = out + ".tmp"
    if os.path.lexists(tmp):
        shutil.rmtree(tmp)
    for root, dirs, files in os.walk(V2, followlinks=False):
        rel = os.path.relpath(root, V2)
        dst_root = os.path.join(tmp, rel) if rel != "." else tmp
        os.makedirs(dst_root, exist_ok=True)
        for nm in sorted(dirs + files):
            src = os.path.join(root, nm)
            dst = os.path.join(dst_root, nm)
            if os.path.islink(src):
                os.symlink(os.readlink(src), dst)
                if nm in dirs:
                    dirs.remove(nm)
            elif os.path.isdir(src):
                continue
            elif os.path.relpath(src, V2) == os.path.join(mv.GFX_REL, mv.NT_DAT):
                shutil.copyfile(dat, dst)
            elif nm == "README.txt" and rel == ".":
                pass
            else:
                raise SystemExit("unexpected regular file in v2: %s" % src)
    sha = hashlib.sha256(open(dat, "rb").read()).hexdigest()
    ent = "\n".join("  %-24s %s -> %d%s" % (str(list(k)), kind, new, "" if old is None else " (was %d)" % old)
                    for kind, k, old, new in log)
    with open(os.path.join(tmp, "README.txt"), "w") as f:
        f.write(README.format(name=name, dat=os.path.join(mv.GFX_REL, mv.NT_DAT),
                              nadd=sum(1 for x in log if x[0] != "flip"),
                              nflip=sum(1 for x in log if x[0] == "flip"), entries=ent, solmap=solmap,
                              cands=", ".join("%s %s" % (nm, list(ss)) for nm, _, _, _, ss in SHAPES), sha=sha))
    shutil.move(tmp, out)
    write_manifest(out, name, solmap)
    print("installed %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol", type=int, choices=(102, 103))
    ap.add_argument("--map", help="per-shape, e.g. G20=103,G19=103,G25=102,G29=102,G31=102,G27=102")
    ap.add_argument("--name", help="variant name (default s<sol> or mix)")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--manifest-only", action="store_true",
                    help="(re)write m5_splits.json into the already installed lib dir of this variant")
    a = ap.parse_args()
    if bool(a.sol) == bool(a.map):
        raise SystemExit("give exactly one of --sol / --map")
    if a.sol:
        solmap = {nm: a.sol for nm, *_ in SHAPES}
        name = a.name or "s%d" % a.sol
    else:
        solmap = {k: int(v) for k, v in (x.split("=") for x in a.map.split(","))}
        if set(solmap) != {nm for nm, *_ in SHAPES} or set(solmap.values()) - {102, 103}:
            raise SystemExit("--map must give 102/103 for all of %s" % [nm for nm, *_ in SHAPES])
        name = a.name or "mix"
    if a.manifest_only:
        write_manifest(os.path.join(V3, name), name, solmap)
        return
    outdir = os.path.join("/home/chcai/m5_work/build", name)
    # strict: guards around every 102 pin are 102 too, so a 102-mapped shape never lands on 103 at an
    # uncovered s (needed for G25, whose v2 neighbourhood is 103). Only 103 -> 102 changes are added.
    strict = 102 in solmap.values()
    v2, v3, dat, pins, owner, log = build(solmap, outdir, strict)
    rep = os.path.join("/home/chcai/m5_work", "verify_%s.txt" % name)
    with open(rep, "w") as out:
        print("variant %s solmap %s\ncatalog %s\nsha256 %s\ntable %d -> %d entries" % (
            name, solmap, dat, hashlib.sha256(open(dat, "rb").read()).hexdigest(),
            len(g.gridbased_rows(v2)[0][2]), len(g.gridbased_rows(v3)[0][2])), file=out)
        print("\n== 0. table edits vs v2 ==", file=out)
        for kind, k, old, new in log:
            tag = ",".join(owner.get(k[:2], [])) if kind != "guard" else ""
            print("  %-7s %-24s -> %d%s %s" % (kind, list(k), new, "" if old is None else " (was %d)" % old, tag), file=out)
        print("\n== 0b. make_v2.validate ==", file=out)
        okv, _ = mv.validate(dat, mv.HOST_CO, ref_cat=v2, log=lambda s: print(s, file=out))
        ok = verify(v2, v3, pins, owner, solmap, out, strict)
        bbsb_report(out)
        print("\nRESULT: validate %s, selection %s" % ("PASS" if okv else "FAIL", "PASS" if ok else "FAIL"), file=out)
    print("%s: validate %s selection %s -> %s" % (name, okv, ok, rep))
    if a.install:
        if not (okv and ok):
            raise SystemExit("not installing: verification failed")
        install(name, dat, log, solmap)


if __name__ == "__main__":
    main()
