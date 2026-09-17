"""NaN tripwire for the DLRM v4 ranker: name the op that first produces a
non-finite value, and dump enough state to rebuild a standalone reproducer.

Motivating defect: [3] in docs/mi450.md -- `train_loss=nan` at BATCH_SIZE=128,
roughly 3 runs in 4, clean at BATCH_SIZE=8, and finite 3/3 under
`HSTU_HAMMER_KERNEL=PYTORCH`.  `DEBUG_NAN_HOOKS` and module-level forward hooks
cannot name a producer, because the suspected producer sits *inside* a fused
autograd function (`hstu_preprocess_and_attention`) rather than at a module
boundary, and because a backward-side producer is invisible to forward hooks
entirely.

This instrument works at the *op* boundary in `generative_recommenders/ops/`,
and it checks both directions:

  forward   inputs all finite  ->  output non-finite   =>  this op's FORWARD
  backward  grad_out finite    ->  grad_in non-finite  =>  this op's BACKWARD

The backward half is the point.  It is implemented by threading each wrapped
op's inputs and outputs through identity autograd Functions, so the check runs
inside the autograd graph without changing any value.

Why this is staged, and why that matters
----------------------------------------
The defect is nondeterministic and allocator-sensitive.  A heavy instrument
allocates temporaries and can make it vanish -- an instrument that suppresses
the bug measures nothing.  So the checks arm in tiers:

  NAN_TRIPWIRE=1                  tier 1: per-step loss + grad scan only.
                                  One reduction per step.  Nearly free.
  NAN_TRIPWIRE_OPS=1              tier 2: per-op forward+backward tripwire.
                                  Heavy.  Use with ARM_STEP.
  NAN_TRIPWIRE_ARM_STEP=<n>       tier 2 stays dormant until step n, so the
                                  first n-1 steps allocate exactly as the
                                  uninstrumented run does.  Set this to
                                  (measured onset - 2).
  NAN_TRIPWIRE_DUMP=<dir>         on catch, write <dir>/catch_<step>_<op>.pt
                                  with every tensor arg (CPU clone) and every
                                  scalar arg.  This is the reproducer seed.
  NAN_TRIPWIRE_FATAL=1            raise on first catch instead of logging on.
  NAN_TRIPWIRE_TRACE_LAUNCH=1     record every Triton launch (kernel, grid,
                                  num_warps, num_stages, constexprs, tensor
                                  shapes/strides/dtypes) in a ring buffer, and
                                  print/dump the tail on a catch.  This is what
                                  makes the standalone reproducer mechanical
                                  rather than guesswork -- it gives the exact
                                  compile+launch configuration, which is what
                                  decides register allocation.
  NAN_TRIPWIRE_TRACE_DEPTH=<n>    ring buffer depth (default 24).

Install (no source edit needed):

  -e PYTHONPATH=/workspace/recommendation/scripts/autoload \
  -e NAN_TRIPWIRE=1

Usage:

  # tier 1, find the onset step
  NAN_TRIPWIRE=1 <train cmd>
  # tier 2, armed just before it
  NAN_TRIPWIRE=1 NAN_TRIPWIRE_OPS=1 NAN_TRIPWIRE_ARM_STEP=45 \
    NAN_TRIPWIRE_DUMP=/tmp/nan_dump <train cmd>

Reads no repo-private API beyond the public op names, so it survives refactors
of the kernels themselves.
"""

from __future__ import annotations

import functools
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

ENABLED = os.environ.get("NAN_TRIPWIRE", "0") == "1"
OPS_ENABLED = os.environ.get("NAN_TRIPWIRE_OPS", "0") == "1"
ARM_STEP = int(os.environ.get("NAN_TRIPWIRE_ARM_STEP", "0"))
DUMP_DIR = os.environ.get("NAN_TRIPWIRE_DUMP", "")
FATAL = os.environ.get("NAN_TRIPWIRE_FATAL", "0") == "1"
MAX_REPORTS = int(os.environ.get("NAN_TRIPWIRE_MAX_REPORTS", "8"))

# op module -> public function names to wrap.  These are the entry points the
# ranker actually calls; everything below them is Triton.
OPS_TO_WRAP: Dict[str, Tuple[str, ...]] = {
    "generative_recommenders.ops.hstu_compute": (
        "hstu_preprocess_and_attention",
        "hstu_compute_uqvk",
        "hstu_compute_output",
    ),
    "generative_recommenders.ops.hstu_attention": ("hstu_mha", "delta_hstu_mha"),
    "generative_recommenders.ops.jagged_tensors": (
        "concat_2D_jagged",
        "split_2D_jagged",
        "jagged_dense_bmm_broadcast_add",
        "hstu_split_l2_embeddings",
        "hstu_concat_l2_embeddings",
    ),
    "generative_recommenders.ops.layer_norm": (
        "layer_norm",
        "rms_norm",
        "swish_layer_norm",
    ),
    "generative_recommenders.ops.mm": ("addmm",),
    "generative_recommenders.ops.position": ("add_timestamp_positional_embeddings",),
}

STEP = 0          # bumped once per backward() call
_caught: List[str] = []
_arm_announced = False


def _p(msg: str) -> None:
    print(f"[nan-tripwire] {msg}", flush=True)


def _armed() -> bool:
    return OPS_ENABLED and STEP >= ARM_STEP


# ---------------------------------------------------------------------------
# finiteness
# ---------------------------------------------------------------------------


def _checkable(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel() > 0


def _nonfinite(t: torch.Tensor) -> bool:
    return not bool(torch.isfinite(t).all().item())


def _describe(t: torch.Tensor) -> str:
    """Summarise a bad tensor: how much of it is bad, and what the bad bits are.

    The bit pattern matters.  0x7fc00000 is a plain quiet NaN (produced by
    0*inf, inf-inf, ...), while an arbitrary payload usually means the bits
    came from somewhere they should not have.
    """
    flat = t.detach().reshape(-1)
    bad = ~torch.isfinite(flat)
    n_bad = int(bad.sum().item())
    idx = torch.nonzero(bad, as_tuple=False).flatten()[:4].tolist()
    bits = []
    if t.dtype in (torch.float32, torch.float64):
        as32 = flat.to(torch.float32)
        for i in idx:
            bits.append(f"0x{int(as32[i].view(torch.int32).item()) & 0xFFFFFFFF:08x}")
    n_nan = int(torch.isnan(flat).sum().item())
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"bad={n_bad}/{flat.numel()} (nan={n_nan} inf={n_bad - n_nan}) "
        f"first_idx={idx} bits={bits}"
    )


def _any_bad(vals) -> Optional[Tuple[str, torch.Tensor]]:
    for name, v in vals:
        if _checkable(v) and _nonfinite(v):
            return name, v
    return None


def _flatten(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    if isinstance(obj, torch.Tensor):
        return [(prefix or "out", obj)]
    if isinstance(obj, (tuple, list)):
        out: List[Tuple[str, Any]] = []
        for i, v in enumerate(obj):
            out.extend(_flatten(v, f"{prefix}[{i}]"))
        return out
    return []


# ---------------------------------------------------------------------------
# dumping the reproducer seed
# ---------------------------------------------------------------------------


def _dump(op: str, when: str, args: tuple, kwargs: dict, extra: dict) -> str:
    if not DUMP_DIR:
        return "<no NAN_TRIPWIRE_DUMP set>"
    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, f"catch_step{STEP}_{op}_{when}.pt")
    payload: Dict[str, Any] = {
        "op": op,
        "when": when,
        "step": STEP,
        "args": [],
        "kwargs": {},
        "extra": {},
        "recent_triton_launches": _launch_tail(),
    }

    def conv(v):
        if isinstance(v, torch.Tensor):
            return {
                "__tensor__": True,
                "data": v.detach().to("cpu", copy=True),
                "dtype": str(v.dtype),
                "shape": tuple(v.shape),
                "requires_grad": v.requires_grad,
            }
        if isinstance(v, (int, float, bool, str, type(None))):
            return v
        if isinstance(v, (tuple, list)):
            return [conv(x) for x in v]
        return repr(v)

    payload["args"] = [conv(a) for a in args]
    payload["kwargs"] = {k: conv(v) for k, v in kwargs.items()}
    payload["extra"] = {k: conv(v) for k, v in extra.items()}
    torch.save(payload, path)
    return path


def _report(op: str, when: str, detail: str, args=(), kwargs=None, extra=None) -> None:
    kwargs = kwargs or {}
    extra = extra or {}
    tag = f"{op}.{when}"
    _caught.append(tag)
    # Once a producer is named, everything downstream of it also goes
    # non-finite.  Cap the noise: the FIRST catch is the answer.
    if len(_caught) > MAX_REPORTS:
        if len(_caught) == MAX_REPORTS + 1:
            _p(f"(further catches suppressed after {MAX_REPORTS} reports)")
        return
    _p("=" * 72)
    _p(f"CAUGHT at step {STEP}: {tag} produced a non-finite value")
    _p(f"  {detail}")
    path = _dump(op, when, args, kwargs, extra)
    _p(f"  reproducer seed: {path}")
    tail = _launch_tail()
    if tail:
        _p(f"  last {len(tail)} Triton launches before the catch (newest last):")
        for r in tail:
            cx = {k: v for k, v in r["args"].items()
                  if not isinstance(v, dict) and v is not None}
            _p(f"    {r['kernel']}  grid={r['grid']} "
               f"warps={r['num_warps']} stages={r['num_stages']}")
            _p(f"       constexprs/scalars: {cx}")
    elif TRACE_LAUNCH:
        _p("  (launch recorder on, but no launches recorded)")
    else:
        _p("  (set NAN_TRIPWIRE_TRACE_LAUNCH=1 to capture the kernel configs)")
    _p("  python stack at the offending call:")
    for line in "".join(traceback.format_stack(limit=25)).rstrip().split("\n"):
        _p(f"    {line}")
    _p("=" * 72)
    if FATAL:
        raise RuntimeError(f"nan-tripwire: {tag} at step {STEP}")


# ---------------------------------------------------------------------------
# backward-side probe
# ---------------------------------------------------------------------------


class _GradProbe(torch.autograd.Function):
    """Identity forward; on backward, record whether the gradient is finite.

    Placed on both sides of a wrapped op.  `slot` is a one-element list that
    the output-side probe writes and the input-side probe reads, so the two can
    be compared without holding a reference to the op's own graph node.
    """

    @staticmethod
    def forward(ctx, t, slot, key):  # type: ignore[override]
        ctx.slot = slot
        ctx.key = key
        return t.view_as(t)

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        try:
            ctx.slot[ctx.key] = bool(torch.isfinite(g).all().item())
            ctx.slot[ctx.key + "_t"] = g
        except Exception:
            pass
        return g, None, None


def _probe(t, slot, key):
    if isinstance(t, torch.Tensor) and t.requires_grad and t.is_floating_point():
        return _GradProbe.apply(t, slot, key)
    return t


def _probe_tree(obj, slot, key):
    if isinstance(obj, torch.Tensor):
        return _probe(obj, slot, key)
    if isinstance(obj, tuple):
        return tuple(_probe_tree(v, slot, f"{key}[{i}]") for i, v in enumerate(obj))
    if isinstance(obj, list):
        return [_probe_tree(v, slot, f"{key}[{i}]") for i, v in enumerate(obj)]
    return obj


class _BackwardVerdict:
    """Compares the two probe slots once both have fired.

    Registered as a tensor hook on the op's first grad-requiring input so it
    runs *after* the op's backward has produced grad_in.
    """

    def __init__(self, op: str, slot: dict, args, kwargs):
        self.op = op
        self.slot = slot
        self.args = args
        self.kwargs = kwargs
        self.fired = False
        # set by the wrapper when this same call's forward already emitted a
        # non-finite output.  Its backward will then trivially be non-finite
        # too, which is a consequence and not an independent producer.
        self.suppressed = False

    def __call__(self, grad):
        if self.fired or self.suppressed:
            return grad
        self.fired = True
        out_ok = [v for k, v in self.slot.items() if k.startswith("out") and v is True]
        out_bad = [
            k for k, v in self.slot.items() if k.startswith("out") and v is False
        ]
        try:
            in_ok = bool(torch.isfinite(grad).all().item())
        except Exception:
            return grad
        if not in_ok and not out_bad and out_ok:
            # grad flowing INTO the op was finite; grad coming OUT is not.
            _report(
                self.op,
                "backward",
                "grad_output finite, grad_input NON-FINITE -- this op's "
                "backward is the producer.  " + _describe(grad),
                self.args,
                self.kwargs,
                {"grad_input": grad},
            )
        return grad


# ---------------------------------------------------------------------------
# Triton launch recorder
# ---------------------------------------------------------------------------
#
# Naming the op is only half the handoff.  The LLVM/AMDGPU team needs the
# *kernel* and the exact configuration it was compiled and launched with --
# grid, num_warps, num_stages, and every constexpr, because those decide
# register allocation and therefore whether the extended-VGPR hazard is even
# present.  Recovering that by reading the autotuner after the fact is
# guesswork; recording it at launch is not.
#
# We keep a small ring buffer of the most recent launches.  On a catch, the
# tail of that buffer is exactly the set of kernels that could have produced
# the bad value, with everything needed to relaunch them standalone.

TRACE_LAUNCH = os.environ.get("NAN_TRIPWIRE_TRACE_LAUNCH", "0") == "1"
TRACE_DEPTH = int(os.environ.get("NAN_TRIPWIRE_TRACE_DEPTH", "24"))
_launches: List[Dict[str, Any]] = []


def _arg_spec(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": tuple(v.shape),
            "stride": tuple(v.stride()),
            "dtype": str(v.dtype).replace("torch.", ""),
            "device": str(v.device),
        }
    if isinstance(v, (int, float, bool, str, type(None))):
        return v
    return repr(v)[:80]


def install_launch_recorder() -> bool:
    try:
        import triton
        from triton.runtime.jit import JITFunction
    except Exception as e:
        _p(f"launch recorder unavailable: {e}")
        return False

    orig_run = JITFunction.run

    def _recording() -> bool:
        # Independent of OPS_ENABLED: capturing kernel configs is useful on its
        # own, e.g. to diff the autotuner's choices between B=8 and B=128.
        return TRACE_LAUNCH and STEP >= ARM_STEP

    @functools.wraps(orig_run)
    def run(self, *args, **kwargs):
        if _recording():
            grid = kwargs.get("grid")
            rec: Dict[str, Any] = {
                "step": STEP,
                "kernel": getattr(self, "__name__", repr(self)),
                "grid": repr(grid)[:120],
                "num_warps": kwargs.get("num_warps"),
                "num_stages": kwargs.get("num_stages"),
                "args": {},
            }
            # Triton binds most parameters by name, but not always -- record
            # positional args too, or a config captured here will not match
            # the one that actually ran.
            for i, v in enumerate(args):
                rec["args"][f"pos{i}"] = _arg_spec(v)
            for k, v in kwargs.items():
                if k in ("grid", "warmup"):
                    continue
                rec["args"][k] = _arg_spec(v)
            _launches.append(rec)
            if len(_launches) > TRACE_DEPTH:
                del _launches[: len(_launches) - TRACE_DEPTH]
        return orig_run(self, *args, **kwargs)

    JITFunction.run = run  # type: ignore[method-assign]
    _p(f"launch recorder installed (depth {TRACE_DEPTH})")
    return True


def _launch_tail() -> List[Dict[str, Any]]:
    return list(_launches)


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------


def _wrap(op_name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _armed():
            return fn(*args, **kwargs)

        named_in = _flatten(list(args), "arg") + [
            (f"kw:{k}", v) for k, v in kwargs.items()
        ]
        bad_in = _any_bad(named_in)

        slot: Dict[str, Any] = {}
        verdict = _BackwardVerdict(op_name, slot, args, kwargs)

        # input side: hook the first grad-requiring float input so we see
        # grad_input after this op's backward runs.
        for a in list(args) + list(kwargs.values()):
            if (
                isinstance(a, torch.Tensor)
                and a.requires_grad
                and a.is_floating_point()
            ):
                try:
                    a.register_hook(verdict)
                except Exception:
                    pass
                break

        out = fn(*args, **kwargs)

        # forward verdict
        if bad_in is None:
            bad_out = _any_bad(_flatten(out, "out"))
            if bad_out is not None:
                verdict.suppressed = True
                _report(
                    op_name,
                    "forward",
                    f"all inputs finite, output {bad_out[0]} NON-FINITE -- "
                    f"this op's forward is the producer.  {_describe(bad_out[1])}",
                    args,
                    kwargs,
                    {"bad_output": bad_out[1]},
                )
        # output side probe feeds `slot` during backward
        out = _probe_tree(out, slot, "out")
        return out

    wrapper.__nan_tripwire__ = True  # type: ignore[attr-defined]
    return wrapper


def install_op_tripwire() -> List[str]:
    """Wrap the public ops and rebind every `from ... import name` binding.

    Rebinding matters: every model module in generative_recommenders/modules/
    binds these names at import time, so patching only the defining module
    would leave all real call sites untouched.
    """
    import importlib

    patched: List[str] = []
    originals: Dict[Tuple[str, str], Any] = {}

    for mod_name, fn_names in OPS_TO_WRAP.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            _p(f"skip {mod_name}: {e}")
            continue
        for fn_name in fn_names:
            orig = getattr(mod, fn_name, None)
            if orig is None or getattr(orig, "__nan_tripwire__", False):
                continue
            new = _wrap(fn_name, orig)
            originals[(mod_name, fn_name)] = orig
            setattr(mod, fn_name, new)
            patched.append(f"{mod_name}.{fn_name}")

    # rebind aliases created by `from ... import name`
    rebound = 0
    for other_name, other in list(sys.modules.items()):
        if other is None or not other_name.startswith("generative_recommenders"):
            continue
        for (mod_name, fn_name), orig in originals.items():
            if other_name == mod_name:
                continue
            if getattr(other, fn_name, None) is orig:
                setattr(other, fn_name, getattr(sys.modules[mod_name], fn_name))
                rebound += 1
    _p(f"op tripwire: wrapped {len(patched)} ops, rebound {rebound} aliases")
    return patched


# ---------------------------------------------------------------------------
# tier 1: per-step loss and gradient scan (cheap, always on when ENABLED)
# ---------------------------------------------------------------------------


def _install_step_counter() -> None:
    """Bump STEP on every backward() -- one per training step."""
    orig_tensor_backward = torch.Tensor.backward
    orig_autograd_backward = torch.autograd.backward

    def _bump():
        global STEP, _arm_announced
        STEP += 1
        if OPS_ENABLED and not _arm_announced and STEP >= ARM_STEP:
            _arm_announced = True
            _p(f"op tripwire ARMED at step {STEP}")

    @functools.wraps(orig_tensor_backward)
    def tensor_backward(self, *a, **k):
        _bump()
        _scan_loss(self)
        return orig_tensor_backward(self, *a, **k)

    @functools.wraps(orig_autograd_backward)
    def autograd_backward(tensors, *a, **k):
        _bump()
        t0 = tensors[0] if isinstance(tensors, (tuple, list)) and tensors else tensors
        if isinstance(t0, torch.Tensor):
            _scan_loss(t0)
        return orig_autograd_backward(tensors, *a, **k)

    torch.Tensor.backward = tensor_backward  # type: ignore[method-assign]
    torch.autograd.backward = autograd_backward  # type: ignore[assignment]


_loss_was_bad = False


def _scan_loss(loss: torch.Tensor) -> None:
    """One reduction per step.  This is what establishes the true onset."""
    global _loss_was_bad
    if not _checkable(loss) or loss.numel() > 16:
        return
    if _nonfinite(loss):
        if not _loss_was_bad:
            _loss_was_bad = True
            _p(f"!! FIRST NON-FINITE LOSS at step {STEP}: {_describe(loss)}")
            if not OPS_ENABLED:
                _p(
                    "   re-run with NAN_TRIPWIRE_OPS=1 "
                    f"NAN_TRIPWIRE_ARM_STEP={max(1, STEP - 2)} to name the op"
                )
            if FATAL:
                raise RuntimeError(f"nan-tripwire: loss non-finite at step {STEP}")


def check_gradients(model, step: Optional[int] = None) -> Optional[str]:
    """Name the first parameter whose .grad is non-finite.  Call after backward."""
    for name, p in model.named_parameters():
        if p.grad is not None and _checkable(p.grad) and _nonfinite(p.grad):
            _p(f"!! non-finite grad at step {step if step is not None else STEP}: "
               f"{name} {_describe(p.grad)}")
            return name
    return None


# ---------------------------------------------------------------------------
# deferred install: the ops modules may not be imported yet at sitecustomize
# time, so hook the import system and install once they land.
# ---------------------------------------------------------------------------


def _install_when_ready() -> None:
    import builtins

    orig_import = builtins.__import__
    done = {"v": False}

    def guarded_import(name, *a, **k):
        mod = orig_import(name, *a, **k)
        if done["v"]:
            return mod
        # install once the last op module we care about is importable
        if name.startswith("generative_recommenders.ops") or name.startswith(
            "generative_recommenders.modules"
        ):
            if all(m in sys.modules for m in OPS_TO_WRAP):
                done["v"] = True
                try:
                    install_op_tripwire()
                except Exception as e:
                    _p(f"install failed: {e}")
        return mod

    builtins.__import__ = guarded_import


def install() -> None:
    if not ENABLED:
        return
    _p(
        f"enabled. ops={OPS_ENABLED} arm_step={ARM_STEP} "
        f"dump={DUMP_DIR or '<off>'} fatal={FATAL}"
    )
    _install_step_counter()
    if TRACE_LAUNCH:
        install_launch_recorder()
    if OPS_ENABLED:
        _install_when_ready()


def summary() -> None:
    if _caught:
        _p(f"caught {len(_caught)} producer(s): {sorted(set(_caught))}")
    elif ENABLED:
        _p(f"no non-finite value observed in {STEP} steps")


if __name__ == "__main__":
    # self-test: a deliberately NaN-producing fake op, forward and backward.
    os.environ.setdefault("NAN_TRIPWIRE", "1")
    ENABLED = OPS_ENABLED = True
    ARM_STEP = 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class _M:
        pass

    fake = _M()

    def good_op(x):
        return x * 2

    class _BadBwd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, g):
            return g * float("nan")

    def bad_backward_op(x):
        return _BadBwd.apply(x)

    def bad_forward_op(x):
        return x * float("nan")

    _install_step_counter()
    for name, op in (
        ("good_op", good_op),
        ("bad_forward_op", bad_forward_op),
        ("bad_backward_op", bad_backward_op),
    ):
        wrapped = _wrap(name, op)
        x = torch.randn(8, 8, device=dev, requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
    summary()
    got = sorted(set(_caught))
    expected = ["bad_backward_op.backward", "bad_forward_op.forward"]
    print(f"self-test: got {got}")
    assert got == expected, f"expected {expected}"
    print("self-test PASS")
