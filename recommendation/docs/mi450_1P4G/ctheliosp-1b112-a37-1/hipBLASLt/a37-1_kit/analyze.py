"""Median global_sps / step_ms over steps 20-200 (rank 0 perf lines), last loss,
and which hipBLASLt library each rank mapped. Usage: analyze.py <run_dir> [...]"""
import re, statistics, sys, os

PERF = re.compile(r"Step (\d+) perf: .*?global_sps=([\d.]+) step_ms=([\d.]+)")
LOSS = re.compile(r'"key": "train_loss".*?"value": ([-\d.eE+naif]+).*?"step_num"?: ?(\d+)|loss[=: ]+([-\d.eE+]+)')

def summarize(d):
    steps = {}
    for line in open(os.path.join(d, "train.log"), errors="replace"):
        m = PERF.search(line)
        if m and int(m.group(1)) not in steps:      # first rank to log a step
            steps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    sel = [v for s, v in steps.items() if 20 <= s <= 200]
    last = max(steps) if steps else None
    out = {"run": os.path.basename(os.path.abspath(d)), "steps_logged": len(steps), "last_step": last}
    if sel:
        out["global_sps"] = statistics.median(v[0] for v in sel)
        out["step_ms"] = statistics.median(v[1] for v in sel)
        out["n"] = len(sel)
    losses = re.findall(r'"key": "train_loss"[^}]*"value": ([^,}]+)', open(os.path.join(d, "mlperf.log"), errors="replace").read()) if os.path.exists(os.path.join(d, "mlperf.log")) else []
    out["last_loss"] = losses[-1].strip() if losses else None
    env = open(os.path.join(d, "environment.txt"), errors="replace").read() if os.path.exists(os.path.join(d, "environment.txt")) else ""
    out["overlay"] = (re.search(r"DLRM_GIN_OVERLAY=(\S+)", env) or [None, None])[1]
    log = open(os.path.join(d, "train.log"), errors="replace").read()
    out["lib_swap"] = "hipBLASLt library swap ->" in log
    return out

for d in sys.argv[1:]:
    s = summarize(d)
    print(" ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
