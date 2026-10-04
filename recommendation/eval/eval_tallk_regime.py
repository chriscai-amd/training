#!/usr/bin/env python3
"""Hazard-regime audit of a finished hipBLASLt bench session (no GPU work).

Constraint 3b / roadmap F7: a stream-K (SK>0) kernel from a NON-image library
(ARM_LIB_DIR) on a tall-K problem (K >= --kmin, default 262144) is the regime
that faulted GPU 0 twice on 2026-10-04 (NT G19 host-native 01:05; NN G19/G20
iteration 8, 03:18). This reads the arm.json files of a session and lists, per
side (control / candidate), every timed GEMM in that regime, plus the per-call
split between a re-expression's GEMM and its declared transpose copies.

It reads only files bench.sh already wrote. It answers "will the NEXT session's
CONTROL run a forbidden kernel?" before anyone spends a GPU arm finding out.

usage: eval_tallk_regime.py <session_dir>/hipblaslt_bench [--kmin N] [--json out]
exit 0 = no side in the regime, 3 = at least one side in it, 2 = unreadable.
"""
import argparse, glob, json, os, re, sys


def sk_of(name):
    m = re.search(r"_SK(\d+)_", name or "")
    return int(m.group(1)) if m else None


def tile_of(name):
    m = re.search(r"Cijk_(A\w{3})_(B\w{3})_.*?_(MT\d+x\d+x\d+)", name or "")
    return "%s_%s %s" % m.groups() if m else (name or "")[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--kmin", type=int, default=262144)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {"kmin": a.kmin, "sides": {}}
    for side in ("control", "candidate"):
        try:
            prob = json.load(open(os.path.join(a.bench_dir, side, "problems.json")))
        except Exception as e:  # noqa: BLE001
            print("unreadable %s problems.json: %s" % (side, e)); return 2
        stock = prob.get("lib_dir") in (None, "")
        lines = {g["key"]: g for g in prob["gemms"]}
        copies = {c["key"]: c for c in prob.get("copies", [])}
        arms = sorted(glob.glob(os.path.join(a.bench_dir, "arms", "*", "arm.json")))
        arms = [p for p in arms if json.load(open(p)).get("side") == side]
        hits, split = [], {}
        for key, g in lines.items():
            K = int(g["line"]["K"])
            names = set()
            ms = []
            for p in arms:
                r = json.load(open(p))["gemms"].get(key) or {}
                if r.get("kernel_name"):
                    names.add(r["kernel_name"])
                if r.get("ms") is not None:
                    ms.append(r["ms"])
            sks = {sk_of(n) for n in names}
            if K >= a.kmin and not stock and any(s and s > 0 for s in sks):
                hits.append({"key": key, "K": K, "M": g["line"]["M"], "N": g["line"]["N"],
                             "layout": g["line"]["transA"] + g["line"]["transB"],
                             "kernels": sorted(tile_of(n) + " SK%s" % sk_of(n) for n in names)})
            if g.get("orig") and g["orig"] != key:
                cms = {}
                for ck, c in copies.items():
                    if c["orig"] == g["orig"]:
                        v = [json.load(open(p))["copies"].get(ck, {}).get("ms") for p in arms]
                        v = [x for x in v if x is not None]
                        cms[ck] = round(sum(v) / len(v), 3) if v else None
                gm = round(sum(ms) / len(ms), 3) if ms else None
                tot = gm + sum(x for x in cms.values() if x) if gm is not None else None
                split[g["orig"]] = {"gemm_ms_per_call": gm, "copies_ms_per_call": cms,
                                    "copy_share": round(1 - gm / tot, 3) if tot else None}
        out["sides"][side] = {"lib_dir": prob.get("lib_dir"), "lib_sha256": prob.get("lib_sha256"),
                              "n_arms": len(arms), "tallk_streamk_nonstock": hits,
                              "reexpress_split": split}
    bad = [s for s, v in out["sides"].items() if v["tallk_streamk_nonstock"]]
    out["verdict"] = "IN_FORBIDDEN_REGIME: " + ",".join(bad) if bad else "CLEAR"
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
