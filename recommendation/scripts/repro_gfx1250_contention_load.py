#!/usr/bin/env python3
"""Concurrent-load generator: mimics the e2e's GPU residency + occupancy."""
import os, sys, time, torch
gib = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
dev = torch.device("cuda")
ballast = torch.empty(int(gib * (1 << 30)) // 2, device=dev, dtype=torch.bfloat16)
ballast.fill_(1.0)
print(f"# load: {gib} GiB resident, pid={os.getpid()}", flush=True)
a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
idx = torch.randint(0, 1 << 20, (1 << 22,), device=dev)
tbl = torch.randn(1 << 20, 256, device=dev, dtype=torch.bfloat16)
t0 = time.time()
while time.time() - t0 < secs:
    for _ in range(20):
        a = torch.nn.functional.silu(a @ b)
        a = a / (a.abs().amax() + 1e-6)
        _ = torch.index_select(tbl, 0, idx).sum(1)
    torch.cuda.synchronize()
print("# load done", flush=True)
