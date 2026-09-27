"""Build hipblaslt-bench YAML union + table rows from HIPBLASLT_LOG_MASK=32 logs.

Usage: python capture_to_yaml.py <run_dir with hipblaslt_bench_*.log> <groups.json> <union.yaml>
Capture: run training with -e HIPBLASLT_LOG_MASK=32 -e HIPBLASLT_LOG_FILE=<run_dir>/hipblaslt_bench_%i.log
(STEPS/RANKS below must match the capture, for calls-per-step).
"""
import sys, glob, collections, json
T = 2097152          # canonical tokens per GPU per step (captured range printed separately)
BIG = 100000
STEPS, RANKS = 12, 4
runs = sys.argv[1]
lines = []
for f in sorted(glob.glob(runs + "/hipblaslt_bench_*.log")):
    lines += [l.split() for l in open(f) if l.startswith("hipblaslt-bench")]

def parse(toks):
    d, i = {}, 1
    while i < len(toks):
        t = toks[i]
        if i + 1 < len(toks) and not toks[i + 1].startswith("-"):
            d[t.lstrip("-")] = toks[i + 1]; i += 2
        else:
            d[t.lstrip("-")] = True; i += 1
    return d

SHAPE = ("m", "n", "k", "lda", "ldb", "ldc", "ldd", "stride_a", "stride_b", "stride_c", "stride_d",
         "alpha", "beta", "transA", "transB", "batch_count", "bias_vector", "bias_source",
         "a_type", "b_type", "c_type", "d_type", "scale_type", "bias_type", "compute_type", "activation_type")
groups = collections.OrderedDict()
for toks in lines:
    d = parse(toks)
    tok = [int(d[x]) for x in ("m", "n", "k") if int(d[x]) > BIG]
    key = []
    for x in SHAPE:
        v = d.get(x, False)
        if x in ("m", "n", "k") and int(v) > BIG: v = "T"
        key.append(v)
    g = groups.setdefault(tuple(key), {"calls": 0, "tok": [], "sol": collections.Counter()})
    g["calls"] += 1; g["tok"] += tok; g["sol"][d.get("solution_index")] += 1

out = []
for key, g in groups.items():
    d = dict(zip(SHAPE, key))
    out.append(dict(d, calls_per_step=g["calls"] / (STEPS * RANKS),
                    tok_min=min(g["tok"]) if g["tok"] else None, tok_max=max(g["tok"]) if g["tok"] else None,
                    solutions=dict(g["sol"])))
json.dump(out, open(sys.argv[2], "w"), indent=1)

def yaml(d):
    v = lambda x: T if d[x] == "T" else int(d[x])
    bc = int(d["batch_count"])
    s = lambda x: int(d[x]) if bc > 1 else 0
    return ("- { function: matmul, M: %d, N: %d, K: %d, lda: %s, ldb: %s, ldc: %s, ldd: %s, "
            "stride_a: %d, stride_b: %d, stride_c: %d, stride_d: %d, alpha: %s, beta: %s, "
            "transA: %s, transB: %s, batch_count: %d, scaleA: 0, scaleB: 0, scaleAlpha_vector: false, "
            "gradient: false, use_e: false, bias_vector: %s, bias_source: d, a_type: %s, b_type: %s, "
            "c_type: %s, d_type: %s, scale_type: %s, bias_type: %s, aux_type: %s, compute_type: c_%s, "
            "activation_type: %s, flush: false, any_stride: true, rotating: 0, cold_iters: 0, iters: 0 }") % (
        v("m"), v("n"), v("k"), d["lda"], d["ldb"], d["ldc"], d["ldd"],
        s("stride_a"), s("stride_b"), s("stride_c"), s("stride_d"), d["alpha"], d["beta"],
        d["transA"], d["transB"], bc, "true" if d["bias_vector"] else "false",
        d["a_type"], d["b_type"], d["c_type"], d["d_type"], d["scale_type"], d["bias_type"], d["d_type"],
        d["compute_type"], d["activation_type"])
seen = []
for d in out:
    y = yaml(d)
    if y not in seen: seen.append(y)
open(sys.argv[3], "w").write("\n".join(seen) + "\n")
print(len(lines), "calls", len(groups), "groups", len(seen), "unique yaml lines")
for d in out:
    print(d["m"], d["n"], d["k"], d["transA"] + d["transB"], d["a_type"], "bias" if d["bias_vector"] else "-",
          "beta", d["beta"][:1], "bc", d["batch_count"], "calls/step %.2f" % d["calls_per_step"], d["tok_min"], d["tok_max"], d["solutions"])
