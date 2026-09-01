"""Deterministic, fixed-shape attention check: PyTorch ref vs Triton.

Replaces the hypothesis-driven unit test for bisection purposes. Two reasons:
the test's assert_close aborts the process on 3.8.0 while formatting its diff,
which hides the numbers, and hypothesis picks different shapes per run. Here the
comparison happens on CPU copies and the stats are printed for the forward and
each backward tensor separately, so we can tell which kernel is wrong.
"""

import argparse
import os

os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")

import torch

import fbgemm_gpu  # noqa: E402,F401  registers the fbgemm ops the ref path needs
from generative_recommenders.common import HammerKernel, set_dev_mode  # noqa: E402
from generative_recommenders.ops.hstu_attention import hstu_mha  # noqa: E402

# torch.testing.assert_close defaults for bfloat16.
ATOL, RTOL = 1e-5, 0.016
SKIP_BACKWARD = False


def stats(name, ref, real):
    r = ref.detach().float().cpu()
    x = real.detach().float().cpu()
    diff = (r - x).abs()
    bad = (diff > ATOL + RTOL * r.abs()).sum().item()
    finite = torch.isfinite(x).all().item()
    verdict = "OK" if bad == 0 and finite else "WRONG"
    print(f"    {name:<9} {verdict:<5} bad {bad:>7}/{r.numel():<8} "
          f"max|diff| {diff.max().item():.3e}  all_finite {finite}")
    return bad == 0 and finite


def run(batch_size, heads, max_uih_len, max_targets, attn_dim, hidden_dim,
        has_multiple_targets, has_max_attn_len, contextual_seq_len, dtype,
        seed=0):
    set_dev_mode(True)
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    alpha = 1.0 / (attn_dim ** 0.5)

    lengths = torch.randint(max_uih_len + 1, size=(batch_size,), device=dev)
    num_targets = torch.randint(1, max_targets + 1, size=(batch_size,), device=dev)
    lengths = lengths + num_targets + contextual_seq_len
    max_seq_len = max_uih_len + max_targets + contextual_seq_len
    max_attn_len = max_uih_len // 5 if has_max_attn_len else 0

    seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=dev)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(seq_offsets[-1].item())

    def mk(dim):
        return (torch.empty((total, heads, dim), dtype=dtype, device=dev)
                .uniform_(-0.1, 0.1).requires_grad_())

    q, k, v = mk(attn_dim), mk(attn_dim), mk(hidden_dim)

    def call(kernel, q, k, v):
        return hstu_mha(
            max_seq_len=max_seq_len, alpha=alpha, q=q, k=k, v=v,
            seq_offsets=seq_offsets, causal=True,
            num_targets=num_targets if has_multiple_targets else None,
            dropout_pr=0.0, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len, kernel=kernel,
            enable_tma=False,
        )

    def phase(label):
        torch.cuda.synchronize()
        print(f"    -- {label}", flush=True)

    phase("ref forward")
    ref_out = call(HammerKernel.PYTORCH, q, k, v)
    dout = torch.randn_like(ref_out)
    phase("ref backward")
    ref_out.backward(dout)
    ref = {"out": ref_out, "dq": q.grad.clone(), "dk": k.grad.clone(),
           "dv": v.grad.clone()}

    q2 = q.detach().clone().requires_grad_()
    k2 = k.detach().clone().requires_grad_()
    v2 = v.detach().clone().requires_grad_()
    phase("triton forward")
    real_out = call(HammerKernel.TRITON, q2, k2, v2)
    torch.cuda.synchronize()
    ok = stats("forward", ref["out"], real_out)
    if SKIP_BACKWARD:
        return ok
    phase("triton backward")
    real_out.backward(dout.detach().clone())
    torch.cuda.synchronize()
    for name, grad in (("dq", q2.grad), ("dk", k2.grad), ("dv", v2.grad)):
        ok &= stats(name, ref[name], grad)
    return ok


CASES = [
    # name,               B  H  uih  tgt  aD  hD  mt     mal    ctx
    ("tiny",              4, 1,  20,  20, 16, 16, False, False, 0),
    ("multi_target",      4, 1,  20,  20, 16, 16, True,  False, 0),
    ("max_attn_len",      4, 1, 100,  20, 16, 16, False, True,  0),
    ("contextual",        4, 1, 100,  20, 16, 16, False, False, 10),
    ("d64",               4, 1, 128,  20, 64, 64, False, False, 0),
    ("d128",              4, 2, 256,  20, 128, 128, False, False, 0),
    ("asym_dims",         4, 2, 128,  20, 32, 128, False, False, 0),
    ("heads4_all",        8, 4, 256, 512, 64, 64, True,  True,  10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--skip-backward", action="store_true")
    ap.add_argument("--fw-config", default=None,
                    help="override the pinned forward config, e.g. "
                         "'m=128,n=32,stages=2,warps=8,nonkdim=16,kpack=2'; "
                         "omit nonkdim/kpack to drop those AMD knobs entirely")
    args = ap.parse_args()
    global SKIP_BACKWARD
    SKIP_BACKWARD = args.skip_backward
    dtype = getattr(torch, args.dtype)

    import triton
    print(f"triton {triton.__version__} | dtype {args.dtype} | "
          f"HSTU_ATTN_PLAIN_K={os.environ.get('HSTU_ATTN_PLAIN_K', '<auto>')}")
    from generative_recommenders import common
    from generative_recommenders.ops.triton import triton_hstu_attention as tha
    print(f"_NO_PIPELINING = {common._NO_PIPELINING}")
    print("  fw configs num_stages:",
          [c.num_stages for c in tha._hstu_attn_fwd.configs])
    print("  bw configs num_stages:",
          [c.num_stages for c in tha._hstu_attn_bwd.configs])

    if args.fw_config:
        spec = dict(kv.split("=") for kv in args.fw_config.split(","))
        cfg = {
            "BLOCK_M": int(spec["m"]),
            "BLOCK_N": int(spec["n"]),
            "waves_per_eu": int(spec.get("waves", 0)),
            "USE_TLX": False,
            "NUM_BUFFERS": 1,
            "NUM_MMA_WARPS_PER_GROUP": 1,
            "NUM_MMA_GROUPS": 1,
        }
        if "nonkdim" in spec:
            cfg["matrix_instr_nonkdim"] = int(spec["nonkdim"])
        if "kpack" in spec:
            cfg["kpack"] = int(spec["kpack"])
        override = triton.Config(cfg, num_stages=int(spec.get("stages", 2)),
                                 num_warps=int(spec.get("warps", 8)))
        for name in ("_hstu_attn_fwd", "_hstu_attn_fwd_persistent"):
            fn = getattr(tha, name, None)
            if fn is not None and hasattr(fn, "configs"):
                fn.configs = [override]
                print(f"    overrode {name} configs -> {cfg}, "
                      f"stages={override.num_stages}, warps={override.num_warps}")

    failed = []
    for case in CASES:
        name = case[0]
        if args.case and args.case != name:
            continue
        print(f"\n[{name}]")
        try:
            if not run(*case[1:], dtype=dtype):
                failed.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"    EXCEPTION {type(exc).__name__}: {str(exc)[:300]}")
            failed.append(name)

    print("\n=== failed cases:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
