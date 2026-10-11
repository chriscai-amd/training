#!/usr/bin/env python3
"""T8 simulation (CPU-only): selected bf16 NT solution, v1 vs v2 catalog, using gridmatch.py
(faithful re-implementation of the 10.2 / 7.14 GridBased matcher, default env).
Usage: python -I simulate.py [v1.dat.zlib v2.dat.zlib]  > simulate_v1_v2.txt"""
import itertools
import random
import re
import sys

sys.path.insert(0, "/home/chcai/t8_work")
import gridmatch as g  # noqa: E402

NT_DAT = "hipblaslt/library/gfx1250/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib"
V1 = "/home/chcai/training/arbor_libs/rocm-10.2_nnonly_v1/" + NT_DAT
V2 = "/home/chcai/training/arbor_libs/rocm-10.2_nnonly_v2/" + NT_DAT
UNION = ("/home/chcai/training/recommendation/docs/mi450_1P4G/ctheliosp-1b112-a37-1/hipBLASLt/"
         "dlrmv4-BF16-allgemms-union.txt")
GROUPS = "/home/chcai/mi450_perf_20260926/hipblaslt_req/groups.json"
T = 2097152
WGRADS = [("G20", 2048, 512), ("G19", 512, 1536), ("G29", 1024, 256), ("G31", 512, 256),
          ("G27", 24, 256), ("G25", 256, 512)]
KS = [1_000_000, 1_400_000, 1_590_000, 1_800_000, 2_097_152, 2_300_000, 2_530_000, 2_800_000,
      3_200_000]

n_diff = 0


def main():
    global n_diff
    p1, p2 = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (V1, V2)
    c1, c2 = g.load_catalog(p1), g.load_catalog(p2)
    t1, t2 = g.gridbased_rows(c1)[0][2], g.gridbased_rows(c2)[0][2]
    mt = {s["index"]: "MT%dx%dx%d" % (*s["sizeMapping"]["macroTile"][:2], s["sizeMapping"]["depthU"])
          for s in c2["solutions"]}
    print("v1: %s (%d table entries)\nv2: %s (%d table entries)" % (p1, len(t1), p2, len(t2)))
    print("solutions: %s" % mt)
    print("matcher: gridmatch.py (rocm-libraries 5911365c GridBased binary path, default env: "
          "no KDTREE, batch flattening on); size predicates modelled by nt_size_accepts\n")

    acc_cache = {}

    def sel(tbl, cat, prob, lda=None, ldb=None, env=None):
        key = (id(cat), lda, ldb)
        if key not in acc_cache:
            acc_cache[key] = g.nt_size_accepts(cat, lda=lda, ldb=ldb)
        return g.select(tbl, prob, env, acc_cache[key])

    def row(label, prob, lda=None, ldb=None, expect=None):
        global n_diff
        a = sel(t1, c1, prob, lda, ldb)
        b = sel(t2, c2, prob, lda, ldb)
        flag = ""
        if a != b:
            n_diff += 1
            flag = "  <-- CHANGED"
        if expect is not None and b != expect:
            flag += "  !! expected %s" % expect
        print("  %-24s m=%-6d n=%-7d b=%-2d k=%-8d v1=%s v2=%s%s" % (label, *prob, a, b, flag))
        return a, b

    # 1) training NT wgrads over the design K range
    print("== 1. training NT wgrads (transA=N transB=T, batch 1, K=T jagged) ==")
    rnd = random.Random(20261010)
    ks = KS + sorted(rnd.randint(1_000_000, 3_200_000) for _ in range(12))
    bad = 0
    for name, m, n in WGRADS:
        for k in ks:
            a, b = row(name, (m, n, 1, k), expect=103)
            bad += b != 103
    print("  -> %d (shape,K) points, %d not on 103 in v2\n" % (len(WGRADS) * len(ks), bad))

    # 2) every bf16 NT GEMM in the model union (with logged lda/ldb), K=T at several T
    print("== 2. bf16 NT GEMMs in %s ==" % UNION.split("/")[-1])
    for line in open(UNION):
        if not line.startswith("- {"):
            continue
        f = dict(re.findall(r"(\w+): ([^,}]+)", line))
        if (f["transA"], f["transB"], f["a_type"]) != ("N", "T", "bf16_r"):
            continue
        m, n, k, b = int(f["M"]), int(f["N"]), int(f["K"]), int(f["batch_count"])
        kk = [k] if k != T else [1_000_000, 1_590_000, T, 2_530_000, 3_200_000]
        for kv in kk:
            row("union lda=%s ldb=%s" % (f["lda"], f["ldb"]), (m, n, b, kv),
                int(f["lda"]), int(f["ldb"]))
    print()

    # 3) generic NT sweep
    print("== 3. generic NT sweep (m,n in small/medium/large, batch 1/2/3/4/8, k in 1..4M) ==")
    vals = sorted(set([1, 2, 7, 8, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 96, 127, 128,
                       129, 192, 255, 256, 257, 384, 511, 512, 513, 516, 768, 1000, 1023, 1024,
                       1025, 1535, 1536, 1537, 2047, 2048, 2049, 3072, 4096, 8192, 16384, 65536,
                       1048575, 1048576, 1048577, 2097152]))
    kvals = [1, 8, 63, 64, 512, 1024, 4096, 65536, 1_000_000, 2_097_152, 4_194_304]
    pts = diffs = 0
    changed = {}
    for m, n in itertools.product(vals, vals):
        for b in (1, 2, 3, 4, 8):
            for k in kvals:
                pts += 1
                a = sel(t1, c1, (m, n, b, k))
                c = sel(t2, c2, (m, n, b, k))
                if a != c:
                    diffs += 1
                    changed.setdefault((m, n, b, a, c), []).append(k)
    for (m, n, b, a, c), kl in sorted(changed.items()):
        print("  m=%-6d n=%-7d b=%d  v1=%s v2=%s  at k=%s%s" % (
            m, n, b, a, c, "ALL" if len(kl) == len(kvals) else kl,
            "  (target)" if b == 1 and (m, n) in [(x[1], x[2]) for x in WGRADS] else
            "  (batch-flattened onto a target)"))
    n_diff += diffs
    print("  -> %d sweep points, %d differ, %d distinct (m,n,b)\n" % (pts, diffs, len(changed)))

    # 4) random dense sweep
    print("== 4. random sweep: 200k (m,n,b,k), m,n log-uniform 1..2^21 ==")
    rnd = random.Random(7)
    rd = {}
    for _ in range(200_000):
        m = int(2 ** rnd.uniform(0, 21)); n = int(2 ** rnd.uniform(0, 21))
        b = rnd.choice([1, 1, 1, 2, 4, 8]); k = int(2 ** rnd.uniform(0, 22))
        a = sel(t1, c1, (m, n, b, k)); c = sel(t2, c2, (m, n, b, k))
        if a != c:
            rd[(m, n, b, a, c)] = k
    for key in sorted(rd):
        print("  m=%-6d n=%-7d b=%d v1=%s v2=%s" % key)
    print("  -> %d random points differ\n" % len(rd))
    n_diff += len(rd)

    # 4b) exposure: per-step launches of each sol (training calls_per_step from groups.json) and
    #     workgroups per launch (no split-K: GSU=1 for both sols -> grid = tiles x batch)
    print("== 4b. per-step exposure (groups.json calls_per_step, K = T) ==")
    import json
    import math
    sm = {s["index"]: s["sizeMapping"] for s in c2["solutions"]}
    tot = {"v1": {}, "v2": {}}
    for i, gr in enumerate(json.load(open(GROUPS)), 1):
        if (gr["transA"], gr["transB"], gr["a_type"]) != ("N", "T", "bf16_r"):
            continue
        m, n, b = int(gr["m"]), int(gr["n"]), int(gr["batch_count"])
        k = T if gr["k"] == "T" else int(gr["k"])
        cps = gr["calls_per_step"]
        out = []
        for tag, tb, cat in (("v1", t1, c1), ("v2", t2, c2)):
            s = sel(tb, cat, (m, n, b, k), int(gr["lda"]), int(gr["ldb"]))
            mt = sm[s]["macroTile"]
            wg = math.ceil(m / mt[0]) * math.ceil(n / mt[1]) * b * max(1, sm[s]["globalSplitU"])
            tot[tag][s] = tot[tag].get(s, 0) + cps
            out.append("%s=sol%s/%dWG" % (tag, s, wg))
        print("  G%-2d m=%-5d n=%-5d b=%d k=%-7s calls/step=%-3g logged=%s  %s" % (
            i, m, n, b, gr["k"], cps, gr["solutions"], "  ".join(out)))
    for tag in ("v1", "v2"):
        print("  %s launches/step by sol: %s" % (tag, dict(sorted(tot[tag].items()))))
    print("  NB sol 103 = MT256x128x64 SK0, 676 VGPRs (A0 DW-corruption kernel, mi450_a0.md L923-924);"
          " sol 102 = MT32x16x32, 252 VGPRs\n")

    # 5) informational: env knobs that must stay unset
    print("== 5. informational: v2 with TENSILE_GRIDBASED_KDTREE=1 (must stay UNSET) ==")
    env = {"TENSILE_GRIDBASED_KDTREE": "1"}
    for name, m, n in WGRADS:
        for prob in [(m, n, 1, T)]:
            print("  %-4s %s  v1=%s v2=%s" % (name, prob, sel(t1, c1, prob, env=env),
                                             sel(t2, c2, prob, env=env)))
    for prob in [(516, 512, 1, 1024), (512, 512, 8, 1024)]:
        print("  ---- %s  v1=%s v2=%s" % (prob, sel(t1, c1, prob, env=env), sel(t2, c2, prob, env=env)))
    print("\nTOTAL differing points v1 vs v2 (sections 1-4): %d" % n_diff)


if __name__ == "__main__":
    main()
