#!/usr/bin/env python3
"""Check HSTU attention backward against its exact zero-gradient oracle.

Finite Q/K/V and identically zero dOut require dQ=dK=dV=0. This invokes only
the attention-backward wrapper, without a dataset, model, optimizer, or extra
CUDA stream. Q/K/V and the default gradients use the production packed
U,V,Q,K layout with row stride 2048. The normal kernel pre-hook must clear dQ
before its accumulation; outputs are deliberately initialized to a nonzero
sentinel (or NaN) before every call. --init empty skips these fills, leaving
the unused U slice and the initial dK/dV allocation contents untouched.

With --dout target, only each sequence's last row has a bounded random dOut.
The exact-zero oracle then applies to history dQ rows; target dQ and dK/dV
may be nonzero, but every gradient element must remain finite.

Examples, inside the verified gfx1250 environment:
  python scripts/repro_hstu_attention_zero_grad.py --repeat 3
  python scripts/repro_hstu_attention_zero_grad.py --total-rows 2571872
  python scripts/repro_hstu_attention_zero_grad.py --contiguous-dq
  python scripts/repro_hstu_attention_zero_grad.py --dq-dtype float32 --init nan
  python scripts/repro_hstu_attention_zero_grad.py --dout target --min-seq-len 2200 --seq-len 2768
  python scripts/repro_hstu_attention_zero_grad.py --dout target --init empty --input-scale 2

The default 1024 equal sequences of length 2512 contain 2,572,288 rows.
--total-rows 2571872 matches the captured row total, using near-equal lengths;
it does not claim to recreate the captured sequence-length distribution.
Each stdout line is JSON. Exit 1 means the zero-gradient oracle failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


HEADS = 4
DIM = 128
WIDTH = HEADS * DIM
PACKED_WIDTH = 4 * WIDTH
STATIC_MAX_SEQ_LEN = 4096


def emit(record: dict) -> None:
    print(json.dumps(record, allow_nan=False), flush=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=2512)
    parser.add_argument("--min-seq-len", type=int, default=None,
                        help="sample lengths uniformly from this minimum through --seq-len")
    parser.add_argument("--total-rows", type=int, default=None)
    parser.add_argument("--dout", choices=("zero", "target"), default="zero")
    parser.add_argument("--dout-scale", type=float, default=1e-5)
    parser.add_argument("--input-scale", type=float, default=0.125,
                        help="Q/K/V uniform half-range, or standard deviation with --input-normal")
    parser.add_argument("--input-normal", action="store_true",
                        help="sample Q/K/V from a zero-mean normal distribution instead of uniform")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--contiguous-dq", action="store_true")
    parser.add_argument("--dq-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--init", choices=("sentinel", "nan", "empty"), default="sentinel",
                        help="empty skips gradient-buffer fills; the production pre-hook still zeros dQ")
    parser.add_argument("--sentinel", type=float, default=7.0)
    parser.add_argument("--sample-bad-rows", type=int, default=8)
    parser.add_argument("--check-chunk-rows", type=int, default=16384)
    args = parser.parse_args()
    if args.batch_size < 1 or not 1 <= args.seq_len <= STATIC_MAX_SEQ_LEN:
        parser.error("batch-size must be positive and seq-len must be in [1, 4096]")
    if args.min_seq_len is not None:
        if not 1 <= args.min_seq_len <= args.seq_len:
            parser.error("min-seq-len must be in [1, seq-len]")
        if args.total_rows is not None:
            parser.error("min-seq-len and total-rows select different length generators; choose one")
    if not math.isfinite(args.dout_scale) or args.dout_scale <= 0:
        parser.error("dout-scale must be finite and positive")
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        parser.error("input-scale must be finite and positive")
    if args.repeat < 1 or args.check_chunk_rows < 1 or args.sample_bad_rows < 0:
        parser.error("repeat/chunk rows must be positive; sample-bad-rows must be nonnegative")
    if not math.isfinite(args.sentinel) or args.sentinel == 0:
        parser.error("sentinel must be finite and nonzero")
    if args.total_rows is not None and not (
        args.batch_size <= args.total_rows <= args.batch_size * args.seq_len
    ):
        parser.error("total-rows must be between batch-size and batch-size * seq-len")
    return args


def check_zero(torch, tensor, *, chunk_rows: int, sample_limit: int,
               zero_row_mask=None, require_zero: bool = True) -> dict:
    """Bound temporary memory even when every gradient element is wrong."""
    nonzero = nonfinite = bad_rows = 0
    samples = []
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = tensor[start : start + chunk_rows]
        not_finite = ~torch.isfinite(chunk)
        nonfinite += int(not_finite.sum().item())
        if require_zero:
            wrong = chunk != 0  # NaN and infinities also violate the zero oracle.
            if zero_row_mask is not None:
                wrong &= zero_row_mask[start : start + chunk_rows, None, None]
            nonzero += int(wrong.sum().item())
            wrong |= not_finite  # Also reject NaN/Inf outside the zero-oracle region.
        else:
            wrong = not_finite
        chunk_bad = int(wrong.sum().item())
        if not chunk_bad:
            continue
        rows = wrong.any(dim=2).any(dim=1)
        bad_rows += int(rows.sum().item())
        remaining = sample_limit - len(samples)
        if remaining > 0:
            indices = rows.nonzero().flatten()[:remaining].cpu().tolist()
            for local_row in indices:
                row = chunk[local_row]
                positions = wrong[local_row].nonzero()[:4].cpu().tolist()
                values = []
                for head, dim in positions:
                    value = float(row[head, dim].item())
                    values.append({
                        "head": head,
                        "dim": dim,
                        "value": value if math.isfinite(value) else str(value),
                    })
                samples.append({"row": start + local_row, "values": values})
    return {
        "elements": tensor.numel(),
        "zero_oracle": ("history_rows" if zero_row_mask is not None else "all_rows") if require_zero else "none",
        "nonzero_including_nonfinite": nonzero if require_zero else None,
        "nonfinite": nonfinite,
        "bad_rows": bad_rows,
        "sample_bad_rows": samples,
        "exact_zero": nonzero == 0 if require_zero else None,
        "passed": nonfinite == 0 and (not require_zero or nonzero == 0),
    }


def main() -> int:
    args = arguments()
    # Set compile-time controls before importing torch or the decorated kernel.
    for key, value in {
        "AMDGCN_USE_BUFFER_OPS": "0",
        "TRITON_FULL_AUTOTUNE": "0",
        "TRITON_ALLOW_PIPELINING": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "",
        "PYTORCH_ALLOC_CONF": "",
    }.items():
        os.environ[key] = value
    os.environ.setdefault("AMD_SERIALIZE_KERNEL", "0")
    os.environ.setdefault("HSA_ENABLE_COREDUMP", "0")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    import triton
    from generative_recommenders.common import (
        autotune_max_seq_len,
        set_static_max_seq_lens,
        set_use_runtime_max_seq_len,
    )
    from generative_recommenders.ops.triton import triton_hstu_attention as attention

    set_static_max_seq_lens([STATIC_MAX_SEQ_LEN])
    set_use_runtime_max_seq_len(False)
    configs = attention._hstu_attn_bwd.configs
    if len(configs) != 1 or configs[0].kwargs.get("SEQUENCE_PARALLEL") is not False:
        raise RuntimeError("expected one pinned SEQUENCE_PARALLEL=False backward config")
    config = configs[0]
    if config.pre_hook is not attention._bwd_pre_hook:
        raise RuntimeError("the pinned config must use the production dQ-zeroing pre-hook")
    original_pre_hook = config.pre_hook
    pre_hook_calls = 0

    def counted_pre_hook(nargs):
        nonlocal pre_hook_calls
        if nargs["SEQUENCE_PARALLEL"] is not False:
            raise RuntimeError("sequence parallelism unexpectedly enabled")
        original_pre_hook(nargs)  # Includes DQ.zero_(); no extra GPU operation here.
        pre_hook_calls += 1

    config.pre_hook = counted_pre_hook
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    if args.min_seq_len is None:
        total_rows = args.total_rows or args.batch_size * args.seq_len
        base_length, remainder = divmod(total_rows, args.batch_size)
        lengths = torch.full((args.batch_size,), base_length, dtype=torch.int64, device=device)
        if remainder:
            lengths[:remainder] += 1
        runtime_max_seq_len = base_length + bool(remainder)
    else:
        lengths = torch.randint(args.min_seq_len, args.seq_len + 1, (args.batch_size,),
                                dtype=torch.int64, device=device, generator=rng)
        total_rows = int(lengths.sum().item())
        base_length = int(lengths.min().item())
        runtime_max_seq_len = int(lengths.max().item())
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), lengths.cumsum(0)))
    num_targets = torch.ones(args.batch_size, dtype=torch.int64, device=device)
    _, sort_indices = torch.sort(lengths, descending=True, stable=False)

    # Keep the original RNG and draw order; changing the uniform scale alone
    # preserves the seed and downstream random dOut draws.
    packed = torch.empty((total_rows, PACKED_WIDTH), dtype=torch.bfloat16, device=device)
    if args.input_normal:
        packed.normal_(mean=0.0, std=args.input_scale, generator=rng)
    else:
        packed.uniform_(-args.input_scale, args.input_scale, generator=rng)
    _, v, q, k = (part.view(total_rows, HEADS, DIM) for part in packed.split(WIDTH, dim=1))
    dout = torch.zeros((total_rows, HEADS, DIM), dtype=torch.bfloat16, device=device)
    history_rows = None
    if args.dout == "target":
        target_rows = offsets[1:] - 1
        target_dout = torch.empty((args.batch_size, HEADS, DIM), dtype=torch.bfloat16, device=device)
        target_dout.uniform_(-args.dout_scale, args.dout_scale, generator=rng)
        if not bool(torch.isfinite(target_dout).all().item()) or not bool(target_dout.count_nonzero().item()):
            raise ValueError("target dOut must contain finite, nonzero values; adjust dout-scale")
        dout[target_rows] = target_dout
        history_rows = torch.ones(total_rows, dtype=torch.bool, device=device)
        history_rows[target_rows] = False
    gradients = torch.empty_like(packed)
    _, dv, packed_dq, dk = (
        part.view(total_rows, HEADS, DIM) for part in gradients.split(WIDTH, dim=1)
    )
    dq_dtype = getattr(torch, args.dq_dtype)
    dq_owner = None
    if args.contiguous_dq:
        dq = torch.empty((total_rows, HEADS, DIM), dtype=dq_dtype, device=device)
        dq_owner = dq
    elif dq_dtype != packed.dtype:
        dq_owner = torch.empty((total_rows, PACKED_WIDTH), dtype=dq_dtype, device=device)
        dq = dq_owner[:, 2 * WIDTH : 3 * WIDTH].view(total_rows, HEADS, DIM)
    else:
        dq = packed_dq
    # Avoid a false attribution if setup itself produced nonfinite inputs.
    inputs_finite = all(bool(torch.isfinite(t).all().item()) for t in (q, k, v))
    if not inputs_finite:
        emit({"event": "invalid_input", "inputs_finite": False})
        return 2

    properties = torch.cuda.get_device_properties(device)
    emit({
        "event": "configuration",
        "torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__,
        "arch": getattr(properties, "gcnArchName", properties.name),
        "source": attention.__file__,
        "batch_size": args.batch_size, "rows": total_rows,
        "length_min": base_length, "runtime_max_seq_len": runtime_max_seq_len,
        "length_generation": "uniform_random" if args.min_seq_len is not None else "near_equal",
        "requested_length_min": args.min_seq_len, "requested_length_max": args.seq_len,
        "dout_mode": args.dout, "dout_scale": args.dout_scale if args.dout == "target" else 0.0,
        "dq_zero_oracle_rows": total_rows - args.batch_size if args.dout == "target" else total_rows,
        "static_max_seq_lens": [STATIC_MAX_SEQ_LEN],
        "autotune_max_seq_len": autotune_max_seq_len(runtime_max_seq_len),
        "heads": HEADS, "dim": DIM, "alpha": DIM ** -0.5,
        "input_dtype": str(q.dtype), "dq_dtype": str(dq.dtype),
        "seed": args.seed,
        "input_distribution": "normal" if args.input_normal else "uniform",
        "input_scale": args.input_scale,
        "strides": {name: list(t.stride()) for name, t in (("q", q), ("k", k), ("v", v), ("dq", dq), ("dk", dk), ("dv", dv), ("dout", dout))},
        "storage_offsets": {name: t.storage_offset() for name, t in (("q", q), ("k", k), ("v", v), ("dq", dq), ("dk", dk), ("dv", dv))},
        "config": config.kwargs, "num_warps": config.num_warps, "num_stages": config.num_stages,
        "pre_hook": original_pre_hook.__name__, "inputs_finite": inputs_finite,
        "init": args.init, "sentinel": args.sentinel, "repeat": args.repeat,
        "env": {key: os.environ.get(key) for key in ("AMDGCN_USE_BUFFER_OPS", "TRITON_FULL_AUTOTUNE", "TRITON_ALLOW_PIPELINING", "AMD_SERIALIZE_KERNEL", "HSTU_BWD_BLOCK_N")},
    })

    fill = float("nan") if args.init == "nan" else args.sentinel
    failed = False
    for repeat in range(args.repeat):
        if args.init != "empty":
            gradients.fill_(fill)
            if dq_owner is not None:
                dq_owner.fill_(fill)
        before_hooks = pre_hook_calls
        torch.cuda.synchronize()
        started = time.perf_counter()
        attention.triton_hstu_attention_bwd(
            dout=dout, q=q, k=k, v=v, dq=dq, dk=dk, dv=dv,
            seq_offsets=offsets, num_targets=num_targets,
            N=runtime_max_seq_len, alpha=DIM ** -0.5,
            max_attn_len=0, contextual_seq_len=0,
            sort_by_length_indices=sort_indices, enable_tma=False, num_softmax_heads=0,
        )
        torch.cuda.synchronize()
        kernel_seconds = time.perf_counter() - started
        if pre_hook_calls == before_hooks:
            raise RuntimeError("backward returned without executing its dQ-zeroing pre-hook")
        results = {
            name: check_zero(
                torch, t, chunk_rows=args.check_chunk_rows, sample_limit=args.sample_bad_rows,
                zero_row_mask=history_rows if name == "dq" else None,
                require_zero=args.dout == "zero" or name == "dq",
            )
            for name, t in (("dq", dq), ("dk", dk), ("dv", dv))
        }
        failed = any(not result["passed"] for result in results.values())
        emit({
            "event": "result", "repeat": repeat,
            "status": "FAIL" if failed else "PASS",
            "kernel_seconds": kernel_seconds,
            "pre_hook_calls": pre_hook_calls - before_hooks,
            "gradients": results,
        })
        if failed:
            break
    emit({"event": "summary", "status": "FAIL" if failed else "PASS", "completed_repeats": repeat + 1})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
