#!/usr/bin/env python3
"""Reproduce the gfx1250 fused-RNG layer-norm/dropout backward fault.

This is defect 4c in docs/mi450.md.  Run one mode per process because the
dropout-on case is expected to terminate with a recoverable GPU memory-aperture
violation on AMD Triton 3.8.0+git4cff872c.

    AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_fused_rng.py
    AMDGCN_USE_BUFFER_OPS=0 python scripts/repro_gfx1250_fused_rng.py --dropout 0
    AMDGCN_USE_BUFFER_OPS=0 TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
        python scripts/repro_gfx1250_fused_rng.py
    AMDGCN_USE_BUFFER_OPS=0 TRITON_HIP_USE_COEXEC_SCHEDULER=0 \
        python scripts/repro_gfx1250_fused_rng.py

The script deliberately bypasses the gfx1250 separated-RNG workaround in
``ops/utils.py`` so it launches the faulty fused-RNG kernel directly.  The
coexec knob requires its release/3.8.x backport (ROCm/triton #11272).
"""

import argparse
import os

import torch
import triton

from generative_recommenders.ops.triton import triton_hstu_linear as hstu_linear


N = 18187
D = 512


def check_results(x, u, weight, bias, mean, rstd, y, dy, dx, du, dw, db,
                  recomputed_y, dropout):
    """Check the kernel against its closed-form backward using the saved stats."""
    scale = 1.0 / (1.0 - dropout) if dropout else 1.0
    grad_u = dy[:, :D].float() * (y[:, :D] != 0) * scale
    grad_x = dy[:, D:2 * D].float() * (y[:, D:2 * D] != 0) * scale
    grad_y = dy[:, 2 * D:].float() * (y[:, 2 * D:] != 0) * scale
    xhat = (x.float() - mean[:, None]) * rstd[:, None]
    ln = xhat * weight.float() + bias.float()
    expected_du = grad_y * ln + grad_u
    norm_grad = grad_y * u.float()
    weighted_grad = norm_grad * weight.float()
    c1 = (xhat * weighted_grad).sum(dim=1, keepdim=True) / D
    c2 = weighted_grad.sum(dim=1, keepdim=True) / D
    expected_dx = grad_x + (weighted_grad - xhat * c1 - c2) * rstd[:, None]
    expected_dw = (norm_grad * xhat).sum(dim=0)
    expected_db = norm_grad.sum(dim=0)

    pairs = (
        ("dx", dx.float(), expected_dx),
        ("du", du.float(), expected_du),
        ("dweight", dw.sum(dim=0), expected_dw),
        ("dbias", db.sum(dim=0), expected_db),
    )
    if recomputed_y is not None:
        pairs += (("y", recomputed_y, y),)
    failures = []
    for name, actual, expected in pairs:
        diff = (actual.float() - expected.float()).abs()
        bad = ~torch.isclose(
            actual.float(), expected.float(), rtol=2e-2, atol=5e-2
        )
        print(
            f"{name:>8}: mismatched={bad.sum().item()} max_abs={diff.max().item():.4e}",
            flush=True,
        )
        if bad.any():
            failures.append(name)
    if recomputed_y is not None:
        for index, name in enumerate(("u", "x", "y")):
            actual = recomputed_y[:, index * D:(index + 1) * D] != 0
            expected = y[:, index * D:(index + 1) * D] != 0
            print(
                f"mask-{name}: mismatched={(actual != expected).sum().item()}",
                flush=True,
            )
    if failures:
        raise AssertionError(f"incorrect outputs: {', '.join(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-warps", type=int, default=2, choices=(1, 2, 4, 8))
    parser.add_argument("--waves-per-eu", type=int, default=0, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--fast-dropout", action="store_true")
    parser.add_argument(
        "--compute-y", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
        raise RuntimeError("set AMDGCN_USE_BUFFER_OPS=0 before running this repro")
    if "TRITON_ALLOW_PIPELINING" in os.environ:
        raise RuntimeError("TRITON_ALLOW_PIPELINING must be unset for the row-7 setup")

    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} | hip {torch.version.hip}", flush=True)
    print(f"triton {triton.__version__} | arch {props.gcnArchName}", flush=True)
    print(
        "coexec="
        f"{os.environ.get('TRITON_HIP_USE_COEXEC_SCHEDULER', '<default>')} "
        "expert-scheduling="
        f"{os.environ.get('TRITON_HIP_USE_EXPERT_SCHEDULING', '<default>')}",
        flush=True,
    )
    print(
        f"N={N} D={D} dropout={args.dropout} num_warps={args.num_warps} "
        f"waves_per_eu={args.waves_per_eu} fast_dropout={args.fast_dropout} "
        f"compute_y={args.compute_y}",
        flush=True,
    )

    torch.manual_seed(0)
    x = torch.randn((N, D), device="cuda", dtype=torch.bfloat16)
    u = torch.randn_like(x)
    weight = torch.randn((D,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)

    training = args.dropout > 0.0
    seed = 1234
    block_d = triton.next_power_of_2(D)
    y = torch.empty((N, 3 * D), device="cuda", dtype=torch.bfloat16)
    mean = torch.empty((N,), device="cuda", dtype=torch.float32)
    rstd = torch.empty_like(mean)
    hstu_linear._ln_mul_dropout_fwd[(N,)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        D,
        1e-5,
        seed,
        args.dropout,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        SILU_U=False,
        BLOCK_D=block_d,
        TRAINING=training,
        CONCAT_U=True,
        CONCAT_X=True,
        MUL_U_ACTIVATION_TYPE="none",
        FAST_DROPOUT=args.fast_dropout,
        num_warps=args.num_warps,
    )
    torch.cuda.synchronize()
    assert block_d == D
    print("forward: PASS; launching fused-RNG backward", flush=True)

    dy = torch.randn_like(y)
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    sms = props.multi_processor_count
    grid = max(1, min(sms * 64, N // 4))
    dweight = torch.empty((grid, D), device="cuda", dtype=torch.float32)
    dbias = torch.empty_like(dweight)
    recomputed_y = torch.empty_like(y) if args.compute_y else None

    tensors = {
        "x": x,
        "u": u,
        "weight": weight,
        "bias": bias,
        "mean": mean,
        "rstd": rstd,
        "dy": dy,
        "dx": dx,
        "du": du,
        "dweight": dweight,
        "dbias": dbias,
    }
    if recomputed_y is not None:
        tensors["y"] = recomputed_y
    for name, tensor in tensors.items():
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        print(f"{name:>8}: [{begin:#x}, {end:#x})", flush=True)

    hstu_linear._ln_mul_dropout_bwd_dx_du[(grid,)](
        dx,
        du,
        dy,
        dweight,
        dbias,
        x,
        u,
        recomputed_y,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        du.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        recomputed_y.stride(0) if recomputed_y is not None else 0,
        D,
        1e-5,
        seed,
        args.dropout,
        N=N,
        SILU_U=False,
        BLOCK_D=block_d,
        TRAINING=training,
        CONCAT_U=True,
        CONCAT_X=True,
        MUL_U_ACTIVATION_TYPE="none",
        COMPUTE_Y=args.compute_y,
        FAST_DROPOUT=args.fast_dropout,
        num_warps=args.num_warps,
        waves_per_eu=args.waves_per_eu,
    )
    torch.cuda.synchronize()
    check_results(
        x, u, weight, bias, mean, rstd, y, dy, dx, du, dweight, dbias,
        recomputed_y, args.dropout,
    )
    print("backward: PASS", flush=True)


if __name__ == "__main__":
    main()
