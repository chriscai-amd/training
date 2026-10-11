"""GPU kernel time per optimizer step by category from a torch profiler trace (rank file)."""
import json, sys, collections, re
def cat(n):
    if n.startswith("Cijk_"): return "hipBLASLt"
    if "nccl" in n.lower() or "rccl" in n.lower(): return "RCCL"
    if re.search(r"hstu|attn|_hstu_attn", n, re.I): return "HSTU attention (triton)"
    if re.search(r"elementwise|vectorized|copy|transpose|cat|index|gather|scatter|reduce|fill", n, re.I): return "torch elementwise/copy/reduce"
    if re.search(r"tbe|embedding|split_embedding|fbgemm", n, re.I): return "FBGEMM/TBE"
    if re.search(r"triton|_kernel$|layer_norm|_ln_|addmm|silu", n, re.I): return "other triton"
    return "other"
for p in sys.argv[1:]:
    d = json.load(open(p))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    steps = [e for e in ev if e.get("ph") == "X" and re.match(r"ProfilerStep#\d+", e.get("name", ""))]
    nstep = max(1, len({e["name"] for e in steps}))
    tot = collections.Counter(); top = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op", "Kernel"):
            c = cat(e["name"]); tot[c] += e.get("dur", 0); 
            if c == "hipBLASLt": top[re.sub(r"_ISA1250.*", "", e["name"])[:90]] += e.get("dur", 0)
    print(f"== {p}  steps={nstep}  (ms/step, GPU kernel time summed over streams)")
    for c, v in tot.most_common(): print(f"  {c:32s} {v/1e3/nstep:9.1f}")
    print(f"  {'TOTAL':32s} {sum(tot.values())/1e3/nstep:9.1f}")
    for k, v in top.most_common(8): print(f"    {v/1e3/nstep:8.1f}  {k}")
