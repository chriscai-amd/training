#!/usr/bin/env python3
"""Torch transpose rate by column count, read from bench.sh arm.json files.

No GPU work: reads the `copies` block that validator/arm_runner.py writes for
each declared re-expression copy ({rows, cols, ms, GBps}, timed as
src.t().contiguous() on the arm's GPU) and tabulates GB/s by `cols`, plus the
cost per bit of column index (1 / rate / log2(cols)) that decides how a
[T, C] -> [C, T] transpose is best factored into narrow passes
(generative_recommenders/ops/triton/triton_addmm.py feature_major).

May claim: the rate torch reached for each declared [rows, cols] shape in the
named arms, on that GPU. May NOT claim: rates for undeclared shapes, or a
training-step effect.

usage: eval_copy_rates.py <arm.json> [<arm.json> ...] [--out table.json]
"""
import argparse
import json
import math
import statistics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    by_cols = {}
    for p in a.arms:
        for k, v in (json.load(open(p)).get("copies") or {}).items():
            if "GBps" in v:
                by_cols.setdefault(int(v["cols"]), []).append(float(v["GBps"]))
    table = {}
    for c in sorted(by_cols):
        med = statistics.median(by_cols[c])
        bits = math.log2(c) if c > 1 else float("nan")
        table[c] = {"n": len(by_cols[c]), "GBps_median": round(med, 1),
                    "GBps_min": round(min(by_cols[c]), 1),
                    "GBps_max": round(max(by_cols[c]), 1),
                    "ms_per_GB_per_bit": round(1e3 / med / bits, 4) if bits else None}
        print(f"cols {c:5d}  n={len(by_cols[c]):3d}  median {med:7.1f} GB/s  "
              f"[{min(by_cols[c]):.0f}..{max(by_cols[c]):.0f}]  "
              f"cost/bit {table[c]['ms_per_GB_per_bit']}")
    if a.out:
        json.dump({"arms": a.arms, "by_cols": table}, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
