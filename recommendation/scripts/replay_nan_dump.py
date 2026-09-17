"""Turn a `nan_tripwire` catch dump into a standalone reproducer.

`scripts/nan_tripwire.py` answers *which op* produced the first non-finite
value and *what it was called with*.  This script answers the two questions
that come next, and that decide whether the defect can be handed to the
LLVM/AMDGPU team at all:

  1. Does the captured call reproduce the non-finite value ON ITS OWN, from
     the dumped inputs, outside the training loop?
  2. Is it deterministic, or does it only fire on some launches?

Those answers select the handoff:

  reproduces every time      -> a pure standalone reproducer is possible.
                                Emit it, hand it over, done.
  reproduces intermittently  -> a soak reproducer.  Report the RATE with a
                                confidence interval, never a single run.
  never reproduces           -> the inputs are NOT sufficient; the defect
                                depends on surrounding state (occupancy,
                                allocator layout, concurrent dispatch).  Say
                                so plainly instead of shipping a test that
                                passes on the vendor's machine and wastes
                                their week.

That third outcome is a real and likely one here -- [3] has resisted three
standalone probes already -- so this script is written to report it as a
first-class result rather than to keep retrying until something breaks.

Usage:

  python scripts/replay_nan_dump.py /tmp/nan_dump/catch_step47_hstu_mha_backward.pt
  python scripts/replay_nan_dump.py <dump> --iters 200      # soak for the rate
  python scripts/replay_nan_dump.py <dump> --emit repro.py  # write the artifact
  python scripts/replay_nan_dump.py <dump> --show            # inspect only

Requires the repo on PYTHONPATH (it calls the real op).  The EMITTED script is
the thing that must be repo-free; see `--emit`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch


def _p(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _rehydrate(v: Any, device: str) -> Any:
    """Undo nan_tripwire's `conv()`: dicts tagged __tensor__ become tensors."""
    if isinstance(v, dict) and v.get("__tensor__"):
        t = v["data"].to(device)
        if v.get("requires_grad"):
            t = t.detach().requires_grad_(True)
        return t
    if isinstance(v, list):
        return [_rehydrate(x, device) for x in v]
    return v


def load(path: str, device: Optional[str] = None) -> Dict[str, Any]:
    """Load a dump. Tensors stay on CPU unless `device` is given.

    Inspection (`--show`) must work on a machine with no GPU -- moving tensors
    to cuda at load time would make reading a dump require the hardware that
    produced it.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if device is not None:
        hydrate(payload, device)
    return payload


def hydrate(payload: Dict[str, Any], device: str) -> Dict[str, Any]:
    payload["args_live"] = [_rehydrate(a, device) for a in payload.get("args", [])]
    payload["kwargs_live"] = {
        k: _rehydrate(v, device) for k, v in payload.get("kwargs", {}).items()
    }
    return payload


# ---------------------------------------------------------------------------
# reporting what was captured
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if isinstance(v, dict) and v.get("__tensor__"):
        return (
            f"Tensor{tuple(v['shape'])} {v['dtype'].replace('torch.','')}"
            f"{' grad' if v.get('requires_grad') else ''}"
        )
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v[:4]) + ("...]" if len(v) > 4 else "]")
    return repr(v)[:70]


def show(payload: Dict[str, Any]) -> None:
    _p(f"== catch: {payload['op']}.{payload['when']} at step {payload['step']} ==")
    _p("-- positional args:")
    for i, a in enumerate(payload.get("args", [])):
        _p(f"     [{i}] {_fmt(a)}")
    if payload.get("kwargs"):
        _p("-- keyword args:")
        for k, v in payload["kwargs"].items():
            _p(f"     {k}= {_fmt(v)}")
    launches = payload.get("recent_triton_launches") or []
    if launches:
        _p(f"-- {len(launches)} Triton launches recorded before the catch "
           f"(newest last):")
        for r in launches:
            scal = {
                k: v for k, v in r["args"].items()
                if not isinstance(v, dict) and v is not None
            }
            _p(f"     {r['kernel']}  grid={r['grid']} warps={r['num_warps']} "
               f"stages={r['num_stages']}")
            _p(f"        {scal}")
    else:
        _p("-- no Triton launches recorded "
           "(re-run the catch with NAN_TRIPWIRE_TRACE_LAUNCH=1)")


# ---------------------------------------------------------------------------
# resolving and replaying the op
# ---------------------------------------------------------------------------

OP_MODULES = (
    "generative_recommenders.ops.hstu_compute",
    "generative_recommenders.ops.hstu_attention",
    "generative_recommenders.ops.jagged_tensors",
    "generative_recommenders.ops.layer_norm",
    "generative_recommenders.ops.mm",
    "generative_recommenders.ops.position",
)


def resolve(op_name: str):
    import importlib

    for m in OP_MODULES:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        fn = getattr(mod, op_name, None)
        if fn is not None:
            # unwrap the tripwire if it is installed in this process
            return getattr(fn, "__wrapped__", fn), m
    raise SystemExit(f"could not resolve op {op_name!r} in {OP_MODULES}")


def _nonfinite_any(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.numel():
            return not bool(torch.isfinite(obj).all().item())
        return False
    if isinstance(obj, (tuple, list)):
        return any(_nonfinite_any(o) for o in obj)
    return False


def replay(payload: Dict[str, Any], iters: int, device: str) -> Tuple[int, int]:
    """Re-run the captured call `iters` times. Returns (n_bad, n_total)."""
    fn, mod = resolve(payload["op"])
    when = payload["when"]
    _p(f"-- replaying {mod}.{payload['op']} ({when}) x{iters} on {device}")

    args = payload["args_live"]
    kwargs = payload["kwargs_live"]

    # Sanity: the inputs we captured must themselves be finite, otherwise the
    # replay is testing propagation rather than production.
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor) and _nonfinite_any(a):
            _p(f"!! captured arg[{i}] is ALREADY non-finite -- this dump records "
               f"propagation, not production. Replay cannot isolate a producer.")
            return (-1, iters)

    n_bad = 0
    for it in range(iters):
        fresh = [
            a.detach().clone().requires_grad_(a.requires_grad)
            if isinstance(a, torch.Tensor) and a.is_floating_point()
            else a
            for a in args
        ]
        try:
            out = fn(*fresh, **kwargs)
        except Exception as e:
            _p(f"!! iteration {it}: op raised {type(e).__name__}: {e}")
            return (-2, iters)

        if when == "forward":
            bad = _nonfinite_any(out)
        else:
            # drive a backward and look at grad_input
            flat = [o for o in (out if isinstance(out, (tuple, list)) else [out])
                    if isinstance(o, torch.Tensor) and o.requires_grad]
            if not flat:
                _p("!! output does not require grad; cannot replay the backward")
                return (-3, iters)
            loss = sum(o.float().sum() for o in flat)
            grads = torch.autograd.grad(
                loss,
                [a for a in fresh if isinstance(a, torch.Tensor) and a.requires_grad],
                allow_unused=True,
            )
            bad = _nonfinite_any([g for g in grads if g is not None])

        if bad:
            n_bad += 1
            if n_bad == 1:
                _p(f"!! reproduced on iteration {it}")
    return (n_bad, iters)


def _wilson(k: int, n: int) -> Tuple[float, float]:
    """95% Wilson interval -- a rate from n runs is not a point estimate."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - s) / d), min(1.0, (c + s) / d))


# ---------------------------------------------------------------------------
# emitting the standalone artifact
# ---------------------------------------------------------------------------

EMIT_TEMPLATE = '''"""Standalone reproducer generated by scripts/replay_nan_dump.py.

Captured: {op}.{when} at training step {step}
Replay:   reproduced {n_bad}/{n_total} runs

This script is torch + triton only in its DEPENDENCIES ON INPUT DATA -- the
tensors are loaded from the accompanying .pt, so no dataset and no repo import
is needed to supply them.  It still calls the repo op.

TO FINISH THE HANDOFF, inline the kernel: the launch table below names the
exact kernel and configuration recorded immediately before the catch.  Copy
that @triton.jit function in verbatim, pin its constexprs to these values, and
delete the repo import.  Configuration is not cosmetic here -- num_warps and
the constexprs decide register allocation, and therefore whether the
extended-VGPR hazard ([6], issue #3) is present at all.

Recorded Triton launches (newest last):
{launch_table}

Run:
  AMDGCN_USE_BUFFER_OPS=0 python {out_name} --iters 200
"""

import argparse
import os
import sys

import torch

if os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
    raise SystemExit(
        "Set AMDGCN_USE_BUFFER_OPS=0. Buffer ops trigger an unrecoverable "
        "node hang on this part ([1] in docs/mi450.md) that costs a reboot."
    )

DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{dump_base}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.replay_nan_dump import load, replay, _wilson

    payload = load(DUMP, args.device)  # device given -> hydrated
    n_bad, n = replay(payload, args.iters, args.device)
    if n_bad < 0:
        print("-- replay could not run; see message above")
        return 2
    lo, hi = _wilson(n_bad, n)
    print(f"-- reproduced {{n_bad}}/{{n}} ({{100.0*n_bad/n:.1f}}%, "
          f"95% CI {{100*lo:.1f}}-{{100*hi:.1f}}%)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def emit(payload: Dict[str, Any], out_path: str, dump_path: str,
         n_bad: int, n_total: int) -> None:
    launches = payload.get("recent_triton_launches") or []
    if launches:
        rows = []
        for r in launches:
            scal = {
                k: v for k, v in r["args"].items()
                if not isinstance(v, dict) and v is not None
            }
            rows.append(
                f"  {r['kernel']}: grid={r['grid']} num_warps={r['num_warps']} "
                f"num_stages={r['num_stages']}\n     {scal}"
            )
        launch_table = "\n".join(rows)
    else:
        launch_table = ("  (none recorded -- re-run the catch with "
                        "NAN_TRIPWIRE_TRACE_LAUNCH=1 before inlining)")

    src = EMIT_TEMPLATE.format(
        op=payload["op"],
        when=payload["when"],
        step=payload["step"],
        n_bad=n_bad,
        n_total=n_total,
        launch_table=launch_table,
        dump_base=os.path.basename(dump_path),
        out_name=os.path.basename(out_path),
    )
    with open(out_path, "w") as f:
        f.write(src)
    _p(f"-- wrote {out_path}")
    _p(f"   keep {os.path.basename(dump_path)} beside it; it holds the inputs")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="a catch_*.pt written by nan_tripwire")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--show", action="store_true", help="inspect only, no replay")
    ap.add_argument("--emit", metavar="OUT.py", help="write a standalone script")
    args = ap.parse_args()

    # --show dispatches nothing, so it needs neither a GPU nor the guard.
    if not args.show and os.environ.get("AMDGCN_USE_BUFFER_OPS") != "0":
        raise SystemExit(
            "Set AMDGCN_USE_BUFFER_OPS=0 -- buffer ops hang the node ([1])."
        )

    payload = load(args.dump)
    show(payload)
    if args.show:
        return 0

    hydrate(payload, args.device)
    _p()
    n_bad, n = replay(payload, args.iters, args.device)
    _p()
    if n_bad < 0:
        _p("== VERDICT: replay inconclusive (see message above) ==")
        return 2
    lo, hi = _wilson(n_bad, n)
    _p(f"== reproduced {n_bad}/{n} ({100.0 * n_bad / n:.1f}%, "
       f"95% CI {100 * lo:.1f}-{100 * hi:.1f}%) ==")
    if n_bad == n:
        _p("== VERDICT: deterministic. A clean standalone reproducer is "
           "possible; inline the kernel and hand it over. ==")
    elif n_bad:
        _p("== VERDICT: intermittent. Ship as a SOAK reproducer and quote the "
           "rate with this interval, never a single run. ==")
    else:
        _p("== VERDICT: does NOT reproduce from these inputs alone. ==")
        _p("   The captured arguments are not sufficient: the defect depends on")
        _p("   surrounding state -- occupancy, allocator layout, or concurrent")
        _p("   dispatch -- none of which a single-call replay recreates.")
        _p("   Do NOT ship this as a reproducer; it would pass on their machine.")
        _p("   Next lever: raise occupancy/concurrency, not input fidelity.")

    if args.emit:
        emit(payload, args.emit, args.dump, n_bad, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
