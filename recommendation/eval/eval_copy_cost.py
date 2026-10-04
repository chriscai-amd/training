#!/usr/bin/env python3
"""Copy-vs-GEMM split of every re-expressed GEMM in a finished bench session
(no GPU work).

For each re-expression `Gxx` the bench priced `Gxx.r0` (the replacement GEMM)
plus `Gxx.cN` (declared transposes, timed as torch `src.t().contiguous()`).
This reads the candidate arms' arm.json files and reports, per GEMM and in
total (weighted by calls/step):

  * GEMM ms/step vs copy ms/step and the copy share;
  * each copy's achieved GB/s (read + write) against a copy roofline
    (default 7.3 TB/s, perf section 5.1 B1 measured device-copy bandwidth);
  * what the copies would cost at the roofline, or at a given rate
    (--rate-tbps), i.e. the headroom that is TRANSPOSE BANDWIDTH and that
    no GEMM lever can reach.

It prices nothing new: the projection is arithmetic on bytes moved and is
labelled as such. Its use is to say where the remaining hipBLASLt-side time
is (GEMM vs transpose) and whether a faster transpose is worth a harness
channel (the runner can only price torch's .t().contiguous()).

usage: eval_copy_cost.py <iter_dir or bench_output.json> [--rate-tbps 3.0]
       [--roofline-tbps 7.3] [--json out]
"""
import argparse, glob, json, os, statistics, sys


def arm_files(root):
    hb = os.path.join(root, "hipblaslt_bench", "arms")
    out = {}
    for p in sorted(glob.glob(os.path.join(hb, "*", "arm.json"))):
        arm = os.path.basename(os.path.dirname(p))
        if arm.startswith("W"):
            continue  # warmups are not timed into W
        out[arm] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rate-tbps", type=float, default=3.0,
                    help="hypothetical transpose rate for the projection")
    ap.add_argument("--roofline-tbps", type=float, default=7.3)
    ap.add_argument("--json")
    a = ap.parse_args()
    root = a.path if os.path.isdir(a.path) else os.path.dirname(a.path)
    try:
        bo = json.load(open(os.path.join(root, "bench_output.json")))
        pg = bo.get("per_gemm") or bo["result"]["per_gemm"]
    except Exception as e:  # noqa: BLE001
        print("unreadable bench_output.json:", e, file=sys.stderr)
        return 2
    arms = arm_files(root)
    cand = {}
    for arm, p in arms.items():
        d = json.load(open(p))
        if d.get("side") != "candidate":
            continue
        cand[arm] = d
    if not cand:
        print("no candidate arm.json found under", root, file=sys.stderr)
        return 2

    weights = {k: v.get("calls_per_step", 1) for k, v in pg.items()}
    reex = sorted({k.split(".")[0] for d in cand.values()
                   for k in list(d.get("gemms", {})) + list(d.get("copies", {}))
                   if "." in k})
    rows, tot = [], {"gemm": 0.0, "copy": 0.0, "copy_roof": 0.0, "copy_rate": 0.0,
                     "bytes_gb": 0.0}
    for gid in reex:
        w = weights.get(gid, 1)
        g_ms = [d["gemms"][f"{gid}.r0"]["ms"] for d in cand.values()
                if f"{gid}.r0" in d.get("gemms", {}) and "ms" in d["gemms"][f"{gid}.r0"]]
        ckeys = sorted({k for d in cand.values() for k in d.get("copies", {})
                        if k.startswith(gid + ".c")})
        copies = []
        for ck in ckeys:
            ms = [d["copies"][ck]["ms"] for d in cand.values()
                  if ck in d.get("copies", {}) and "ms" in d["copies"][ck]]
            any_c = next(d["copies"][ck] for d in cand.values() if ck in d.get("copies", {}))
            nbytes = 2 * any_c["rows"] * any_c["cols"] * 2  # bf16 read + write
            m = statistics.mean(ms)
            copies.append({"key": ck, "rows": any_c["rows"], "cols": any_c["cols"],
                           "ms": round(m, 4), "GBps": round(nbytes / (m * 1e-3) / 1e9, 1),
                           "pct_of_roofline": round(100 * nbytes / (m * 1e-3) / (a.roofline_tbps * 1e12), 1),
                           "ms_at_roofline": round(nbytes / (a.roofline_tbps * 1e12) * 1e3, 4),
                           "ms_at_rate": round(nbytes / (a.rate_tbps * 1e12) * 1e3, 4),
                           "bytes_gb": nbytes / 1e9})
        gm = statistics.mean(g_ms) if g_ms else float("nan")
        cm = sum(c["ms"] for c in copies)
        r = {"id": gid, "calls_per_step": w,
             "gemm_ms_per_call": round(gm, 4), "copy_ms_per_call": round(cm, 4),
             "gemm_ms_per_step": round(gm * w, 3), "copy_ms_per_step": round(cm * w, 3),
             "copy_share_pct": round(100 * cm / (cm + gm), 1) if gm == gm else None,
             "copy_ms_per_step_at_roofline": round(w * sum(c["ms_at_roofline"] for c in copies), 3),
             "copy_ms_per_step_at_rate": round(w * sum(c["ms_at_rate"] for c in copies), 3),
             "copies": copies}
        rows.append(r)
        tot["gemm"] += r["gemm_ms_per_step"]
        tot["copy"] += r["copy_ms_per_step"]
        tot["copy_roof"] += r["copy_ms_per_step_at_roofline"]
        tot["copy_rate"] += r["copy_ms_per_step_at_rate"]
        tot["bytes_gb"] += w * sum(c["bytes_gb"] for c in copies)

    W = bo.get("hipblaslt_ms_per_step")
    Wc = bo.get("control_hipblaslt_ms_per_step")
    print(f"session {bo.get('run_id')}  candidate W {W} ms/step  control W {Wc}")
    print(f"{'id':5} {'calls':>5} {'gemm ms/st':>10} {'copy ms/st':>10} {'copy%':>6} "
          f"{'@roof':>7} {'@rate':>7}  copies (cols:GB/s)")
    for r in sorted(rows, key=lambda r: -r["copy_ms_per_step"]):
        cs = " ".join(f"{c['cols']}:{c['GBps']:.0f}" for c in r["copies"])
        print(f"{r['id']:5} {r['calls_per_step']:>5} {r['gemm_ms_per_step']:>10.2f} "
              f"{r['copy_ms_per_step']:>10.2f} {r['copy_share_pct']:>6} "
              f"{r['copy_ms_per_step_at_roofline']:>7.2f} {r['copy_ms_per_step_at_rate']:>7.2f}  {cs}")
    print(f"TOTAL re-expressed: GEMM {tot['gemm']:.1f} ms/step, copies {tot['copy']:.1f} ms/step "
          f"({tot['bytes_gb']:.1f} GB moved/step, {tot['bytes_gb'] / max(tot['copy'], 1e-9):.2f} TB/s avg)")
    print(f"  copies at {a.roofline_tbps} TB/s roofline: {tot['copy_roof']:.1f} ms/step; "
          f"at {a.rate_tbps} TB/s: {tot['copy_rate']:.1f} ms/step (ARITHMETIC, not a measurement)")
    if W:
        for lab, c in (("roofline", tot["copy_roof"]), (f"{a.rate_tbps} TB/s", tot["copy_rate"])):
            Wn = W - tot["copy"] + c
            print(f"  candidate W with copies at {lab}: {Wn:.1f} ms/step "
                  f"-> ratio vs this control {Wc / Wn:.3f} (vs current {Wc / W:.3f})")
    res = {"run_id": bo.get("run_id"), "W": W, "control_W": Wc, "rows": rows, "total": tot,
           "rate_tbps": a.rate_tbps, "roofline_tbps": a.roofline_tbps,
           "candidate_arms": sorted(cand)}
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
