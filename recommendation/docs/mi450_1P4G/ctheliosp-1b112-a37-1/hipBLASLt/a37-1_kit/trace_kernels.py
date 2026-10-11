"""Per-kernel calls and mean duration per step from a torch profiler trace."""
import json, sys, collections, re
p = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
d = json.load(open(p)); ev = d["traceEvents"]
nstep = len({e["name"] for e in ev if e.get("ph") == "X" and re.match(r"ProfilerStep#\d+", e.get("name", ""))}) or 1
tot = collections.Counter(); cnt = collections.Counter(); grid = {}
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "kernel":
        n = e["name"]; k = re.sub(r"_(ISA1250|MI16x16x1).*", "", n) if n.startswith("Cijk_") else n[:80]
        tot[k] += e["dur"]; cnt[k] += 1; grid.setdefault(k, e.get("args", {}).get("grid"))
print(f"{'ms/step':>8} {'calls/step':>10} {'us/call':>9}  kernel  [grid]")
for k, v in tot.most_common(top):
    print(f"{v/1e3/nstep:8.1f} {cnt[k]/nstep:10.1f} {v/cnt[k]:9.1f}  {k}  {grid[k]}")
