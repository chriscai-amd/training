#!/usr/bin/env python3
"""Current per-GEMM headroom from a finished bench session (no GPU work).

Headroom(G) = side's measured ms/step (GEMM + declared copies, as bench.sh
priced it) - the registry's doc_lower_bound_ms_per_step. Ranks the GEMMs so
the next iteration attacks the largest REMAINING headroom, not the documented
one. Ratios inside one session are what the bench may claim; the absolute
ms/step here is only used for ordering within the same session.

usage: eval_headroom.py <bench_output.json> [--side candidate|control]
       [--registry <pack>/eval/gemm_registry.json] [--top N] [--json out]
"""
import argparse, json, sys

REG = "/home/chcai/Arbor/packs/mi450/hipblaslt/eval/gemm_registry.json"


def load_registry(path):
    r = json.load(open(path))
    rows = r.get("gemms", r) if isinstance(r, dict) else r
    if isinstance(rows, dict):
        rows = [dict(v, id=k) for k, v in rows.items() if isinstance(v, dict)]
    return {g["id"]: g for g in rows if isinstance(g, dict) and "id" in g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_output")
    ap.add_argument("--side", default="candidate")
    ap.add_argument("--registry", default=REG)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json")
    a = ap.parse_args()
    try:
        d = json.load(open(a.bench_output))
        pg = d.get("per_gemm") or d["result"]["per_gemm"]
        reg = load_registry(a.registry)
    except Exception as e:  # noqa: BLE001
        print("unreadable:", e, file=sys.stderr)
        return 2
    rows = []
    for gid, g in pg.items():
        ms = g.get(f"{a.side}_ms_per_step")
        if ms is None:
            continue
        lb = (reg.get(gid) or {}).get("doc_lower_bound_ms_per_step", 0.0) or 0.0
        k = (g.get(f"{a.side}_kernel") or [[None, ""]])[0][1] or ""
        rows.append({"id": gid, "layout": g.get("layout"),
                     "reexpressed_as": g.get("reexpressed_as"),
                     "ms_per_step": round(ms, 2), "lower_bound": lb,
                     "headroom": round(ms - lb, 2),
                     "kernel": k.split("_SN_")[0][:60]})
    rows.sort(key=lambda r: -r["headroom"])
    tot = sum(r["headroom"] for r in rows)
    print(f"side={a.side} total ms/step={sum(r['ms_per_step'] for r in rows):.1f} "
          f"headroom={tot:.1f}")
    for r in rows[:a.top]:
        print(f"{r['id']:4} {str(r['layout']):3} {r['ms_per_step']:8.2f} "
              f"lb {r['lower_bound']:5.1f} head {r['headroom']:7.2f} "
              f"({100*r['headroom']/tot:4.1f}%) {r['kernel']}"
              + (f" [reexpr {r['reexpressed_as']}]" if r["reexpressed_as"] else ""))
    if a.json:
        json.dump({"side": a.side, "total_headroom": tot, "rows": rows},
                  open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
