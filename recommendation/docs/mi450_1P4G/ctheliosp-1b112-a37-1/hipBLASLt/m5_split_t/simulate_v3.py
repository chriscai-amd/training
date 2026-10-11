#!/usr/bin/env python3
"""M5 catalog check (CPU-only): installed rocm-10.2_nnonly_v3/{s103,s102,mix} vs v2.

  /home/chcai/.venv_tools/bin/python -I /home/chcai/m5_work/simulate_v3.py > /home/chcai/m5_work/simulate_v2_v3.txt

Per variant:
  A. install check: symlink set (name -> target) identical to v2 and v1, the only regular files are the
     NT .dat.zlib + README, installed .dat == make_v3 build (sha256), make_v2.validate (round trip,
     sorted unique keys, indices exist, kernel + .kd symbols in the 10.2 .co).
  B. M5 problems as torch.bmm issues them (m=O n=D batch=s K=T//s lda=m ldb=n), every candidate s and
     every s in 2..256, T over the design range + jagged samples: v2 vs v3.
  C. simulate.py (T8's generic simulation: training batch-1 wgrads, model union file, generic sweep,
     200k random sweep, per-step exposure) with v2 as "old" and v3 as "new".
  D. make_v3.verify on the INSTALLED catalog (boundary grid 151,704 points, 300k random sweep incl.
     batch up to 256, union, pre-images of every pinned key) = the complete list of changed (m, n).
  E. per-step sol launches with the overlay s values (m5_split_t.gin / _s102.gin / _mix.gin).
  F. m5_splits.json (read by triton_addmm at import) == the selections computed here, sha matches.
"""
import hashlib
import io
import math
import os
import random
import re
import subprocess
import sys
from contextlib import redirect_stdout

sys.path.insert(0, "/home/chcai/m5_work")
sys.path.insert(0, "/home/chcai/t8_work")
import gridmatch as g  # noqa: E402
import make_v2 as mv  # noqa: E402
import make_v3 as m3  # noqa: E402

V1 = "/home/chcai/training_opt/arbor_libs/rocm-10.2_nnonly_v1"
V2 = m3.V2
V3 = m3.V3
OVERLAY_S = {   # from gin/mi450_hipblaslt/m5_split_t*.gin (hipBLASLt m x n -> s)
    "s103": {(2048, 512): 8, (512, 1536): 10, (256, 512): 64, (1024, 256): 32, (512, 256): 64, (24, 256): 128},
    "s102": {(2048, 512): 4, (256, 512): 4, (1024, 256): 2, (512, 256): 4, (24, 256): 64},
    "mix": {(2048, 512): 8, (512, 1536): 10, (256, 512): 4, (1024, 256): 2, (512, 256): 4, (24, 256): 64},
}
SOLMAP = {"s103": {nm: 103 for nm, *_ in m3.SHAPES}, "s102": {nm: 102 for nm, *_ in m3.SHAPES},
          "mix": {"G20": 103, "G19": 103, "G25": 102, "G29": 102, "G31": 102, "G27": 102}}
CALLS = {(m, n): c for _, m, n, c, _ in m3.SHAPES}


def tree(root):
    links, files = {}, set()
    for d, dirs, fs in os.walk(root, followlinks=False):
        for nm in dirs + fs:
            p = os.path.join(d, nm)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                links[rel] = os.readlink(p)
            elif os.path.isfile(p):
                files.add(rel)
    return links, files


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def install_check(name, P):
    d = os.path.join(V3, name)
    ok = True

    def chk(c, msg):
        nonlocal ok
        P(("  OK   " if c else "  FAIL ") + msg)
        ok &= bool(c)
    l3, f3 = tree(d)
    l2, f2 = tree(V2)
    l1, f1 = tree(V1)
    nt_rel = os.path.join(mv.GFX_REL, mv.NT_DAT)
    chk(l3 == l2, "symlink set (name -> target) == v2 (%d links)" % len(l3))
    chk(set(l3) | {nt_rel} == set(l1) | {nt_rel} and all(l3[k] == l1[k] for k in l3 if k in l1),
        "symlink set == v1 except the NT catalog (v1 symlinks it to the filtered stub copy: %s)"
        % ("symlink" if nt_rel in l1 else "regular file"))
    chk(f3 == {nt_rel, "README.txt", "m5_splits.json"},
        "regular files = NT .dat.zlib + README.txt + m5_splits.json (%s)" % sorted(f3))
    chk(all(os.path.exists(os.path.join(d, k)) for k in l3), "every symlink resolves")
    dat = os.path.join(d, nt_rel)
    built = os.path.join("/home/chcai/m5_work/build", name, mv.NT_DAT)
    chk(sha(dat) == sha(built), "installed NT .dat == build/%s (sha256 %s)" % (name, sha(dat)[:16]))
    chk(sha(dat) != sha(os.path.join(V2, nt_rel)), "differs from v2's NT .dat")
    okv, _ = mv.validate(dat, mv.HOST_CO, ref_cat=g.load_catalog(os.path.join(V2, nt_rel)), log=P)
    return ok and okv, dat


def m5_problems(name, s2, s3, sm, P):
    rnd = random.Random(20261011)
    Ts = m3.TS + sorted(rnd.randint(1_000_000, 3_200_000) for _ in range(8))
    bad = 0
    P("  T values: %s" % Ts)
    for nm, m, n, cps, ss in m3.SHAPES:
        want = SOLMAP[name][nm]
        P("  %s m=%d n=%d (calls/step %d), mapped sol %d" % (nm, m, n, cps, want))
        for s in (1,) + tuple(ss):
            r2 = sorted({s2(m, n, s, T // s, m, n) for T in Ts})
            r3 = sorted({s3(m, n, s, T // s, m, n) for T in Ts})
            good = r3 == [want]
            bad += not good
            fk = m3.flat(m, n, s) if s > 1 else (m, n)
            P("    s=%-4d key %-14s v2=%-7s v3=%-7s WG/launch(v3)=%-6d %s" % (
                s, "(%d,%d)" % fk, r2, r3, m3.wg(sm[r3[0]], m, n, s), "OK" if good else "UNEXPECTED"))
        by = {}
        for s in range(2, 257):
            by.setdefault((s2(m, n, s, 2_097_152 // s, m, n), s3(m, n, s, 2_097_152 // s, m, n)), []).append(s)
        for (a, c), lst in sorted(by.items()):
            P("    all s 2..256 @T=2097152: v2=%s v3=%s for %d s: %s" % (a, c, len(lst), lst if len(lst) <= 24 else "%s ... %s" % (lst[:6], lst[-3:])))
    return bad


def run_simulate(dat2, dat3):
    out = subprocess.run([sys.executable, "-I", "/home/chcai/t8_work/simulate.py", dat2, dat3],
                         capture_output=True, text=True, check=True).stdout
    # simulate.py labels its two catalogs v1/v2; here they are v2 (old) / v3 (new)
    out = re.sub(r"\bv2([=:])", r"v3\1", out)
    out = re.sub(r"\bv1([=:])", r"v2\1", out)
    out = out.replace("v1 vs v2", "v2 vs v3").replace("not on 103 in v2", "not on 103 in v3 (expected for s102)")
    out = re.sub(r"^  (v[12]) launches", lambda mo: "  %s launches" % {"v1": "v2", "v2": "v3"}[mo.group(1)], out, flags=re.M)
    return out


def exposure(name, s3, sm, P):
    T = 2_097_152
    tot, wgs = {}, {}
    for nm, m, n, cps, _ in m3.SHAPES:
        s = OVERLAY_S[name].get((m, n), 1)
        if s > 1 and (T // s) * max(m, n) >= 1 << 31:
            s = 1
        sol = s3(m, n, s, T // s, m, n)
        tot[sol] = tot.get(sol, 0) + cps
        wgs.setdefault(sol, []).append("%s:%dWG" % (nm, m3.wg(sm[sol], m, n, s)))
        P("  %s s=%-3d -> sol %d, %d WG/launch, %d launches/step" % (nm, s, sol, m3.wg(sm[sol], m, n, s), cps))
    P("  wgrad launches/step by sol: %s (T8/v2 today: {103: 12})" % dict(sorted(tot.items())))


def manifest_check(name, dat, s3, solmap, P):
    import json
    man = json.load(open(os.path.join(V3, name, "m5_splits.json")))
    ok = man["variant"] == name and man["nt_dat_sha256"] == sha(dat) and man["solmap"] == solmap
    for nm, m, n, _, ss in m3.SHAPES:
        ent = man["shapes"]["%dx%d" % (m, n)]
        for s_ in range(1, 257):
            r = {s3(m, n, s_, T // s_, m, n) for T in m3.TS}
            ok &= ent["sol_by_s"][str(s_)] == (r.pop() if len(r) == 1 else None)
        ok &= ent["sol"] == solmap[nm]
    over = {k: v for k, v in OVERLAY_S[name].items()}
    unp = [(k, v) for k, v in over.items() if man["shapes"]["%dx%d" % k]["sol_by_s"][str(v)] != man["shapes"]["%dx%d" % k]["sol"]]
    ok &= not unp
    P("  variant/sha/solmap/sol_by_s (s=1..256, 6 shapes) consistent: %s; overlay s all pinned: %s" % (ok, not unp))
    return ok


def main():
    P = print
    v2dat = os.path.join(V2, mv.GFX_REL, mv.NT_DAT)
    v2 = g.load_catalog(v2dat)
    s2 = m3.Sel(v2)
    sm = {s["index"]: s["sizeMapping"]["macroTile"] for s in v2["solutions"]}
    allok = True
    for name in ("s103", "s102", "mix"):
        solmap = SOLMAP[name]
        strict = 102 in solmap.values()
        P("#" * 100)
        P("# variant %s: %s  (v2 = %s)" % (name, os.path.join(V3, name), V2))
        P("#" * 100)
        P("\n== A. install check ==")
        ok, dat = install_check(name, P)
        v3 = g.load_catalog(dat)
        s3 = m3.Sel(v3)
        P("\n== B. M5 batched problems (lda=m, ldb=n, K=T//s) ==")
        bad = m5_problems(name, s2, s3, sm, P)
        P("  -> %d unexpected" % bad)
        ok &= bad == 0
        P("\n== C. simulate.py (T8 generic simulation), v2 vs v3/%s ==" % name)
        sim = run_simulate(v2dat, dat)
        P(sim)
        # every (m, n, b) that changed in simulate.py sections 3/4 must flatten onto a pinned key
        # (or, in s102, be a 103 -> 102 change of the strict G25 region)
        pins, owner = m3.pins_for(solmap)
        unexpl = []
        nchg = 0
        for mo in re.finditer(r"m=(\d+)\s+n=(\d+)\s+b=(\d+)\s+v2=(\d+) v3=(\d+)", sim):
            m, n, b, a, c = map(int, mo.groups())
            nchg += 1
            fk = m3.flat(m, n, b) if b > 1 else (m, n)
            if not (fk in pins and c == pins[fk] or strict and (a, c) == (103, 102)):
                unexpl.append((m, n, b, a, c))
        P("== C2. classification of the simulate.py sweep changes: %d changed (m,n,b) lines, %d not "
          "explained by a pinned flattened key%s: %s\n" % (
              nchg, len(unexpl), " or a 103->102 strict-region change" if strict else "", unexpl[:20]))
        ok &= not unexpl
        P("== D. make_v3.verify on the installed catalog (complete changed-(m,n) list) ==")
        buf = io.StringIO()
        okv = m3.verify(v2, v3, pins, owner, solmap, buf, strict=strict)
        P(buf.getvalue())
        ok &= okv
        P("== E. per-step wgrad launches with the overlay s (T=2,097,152) ==")
        exposure(name, s3, sm, P)
        P("\n== F. m5_splits.json vs this simulation ==")
        okm = manifest_check(name, dat, s3, solmap, P)
        ok &= okm
        P("\nVARIANT %s: %s\n" % (name, "PASS" if ok else "FAIL"))
        allok &= ok
    P("RESULT: %s" % ("PASS" if allok else "FAIL"))


if __name__ == "__main__":
    main()
