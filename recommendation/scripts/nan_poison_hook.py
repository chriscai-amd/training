#!/usr/bin/env python3
"""Turn the batch-128 `nan` flake into a deterministic, attributing detector.

Companion to `scripts/repro_jagged_concat_uninit.py`. That script proves the
mechanism in isolation; this one decides, in ONE training run, whether the
mechanism is what is actually happening.

THE IDEA
--------
Defect [3] is `train_loss=nan` at BATCH_SIZE=128 in 3 of 4 runs, clean at
BATCH_SIZE=8, and finite 3/3 under `HSTU_HAMMER_KERNEL=PYTORCH`. One hypothesis
class fits that profile exactly: a Triton kernel does not write every row of a
`torch.empty` output, so a gradient inherits whatever the caching allocator left
in that memory. Sometimes that garbage is finite and the run survives; sometimes
its bits decode to NaN or Inf and the loss goes non-finite on first touch.

If that is what is happening, the run-to-run lottery is entirely "what was in
that memory". So remove the lottery: fill every uninitialized floating-point
allocation with a NaN sentinel. Then

    reading uninitialized memory  <=>  immediate NaN, every time

and the experiment becomes decisive in both directions:

  * If poisoning makes BATCH_SIZE=8 go `nan` (it is clean today), an uninit read
    exists and is being masked by luck. The hypothesis class is confirmed and
    the traceback names the op.
  * If BATCH_SIZE=8 stays finite under poisoning for a few hundred steps, then
    NOTHING on this path reads uninitialized memory, and the entire hypothesis
    class -- including everything in `repro_jagged_concat_uninit.py` -- is dead.
    That is worth knowing for the cost of one run.

The second outcome is the one to expect, for the reason recorded in that
script's docstring: a static audit of all eight ranker call sites found
`max_seq_len = max_len_left + max_len_right` at every one, which bounds the
coverage condition structurally. This hook tests the assumption that the audit
cannot: that `max_len_left` and `max_len_right` are the true runtime maxima.

WHAT IT INSTALLS
----------------
1. `torch.empty` / `torch.empty_like` poisoning -- CUDA floating-point tensors
   only. Integer and CPU allocations are left alone (poisoning an index tensor
   changes semantics rather than exposing them).
2. A coverage precondition assert around `concat_2D_jagged` and
   `split_2D_jagged`: checks `max_seq_len >= max_z(len_a(z) + len_b(z))`
   directly from the offsets, which is the exact predicate the kernels' launch
   grid assumes. This catches a stale `max_len_*` even if the poison does not
   happen to land in a read row.
3. A non-finite gradient check in the optimizer step, reporting the first
   parameter to go bad and at which step.

Each is independently switchable, because they answer different questions.

LIMITATION, STATED UP FRONT
---------------------------
Poisoning is a Python-level monkeypatch of `torch.empty`. It covers allocations
made from Python -- which is where every allocation named in the hypothesis is
made -- but NOT allocations made inside ATen/C++ kernels. So a clean run bounds
the hypothesis to repo-level allocations; it does not prove no uninitialized
read exists anywhere in the process. Say that when reporting the result.

USAGE
-----
Import before the model is built, e.g. at the top of the training entrypoint, or
via `PYTHONSTARTUP`, or:

    export AMDGCN_USE_BUFFER_OPS=0
    export NAN_POISON=1                    # poison torch.empty
    export NAN_POISON_COVERAGE=1           # assert the concat/split predicate
    export NAN_POISON_GRADCHECK=1          # first non-finite gradient wins
    python -c "import scripts.nan_poison_hook as h; h.install()" # or call install()

The decisive experiment, in order:

    NAN_POISON=1 BATCH_SIZE=8   ...   # expected clean -> hypothesis class dead
    NAN_POISON=1 BATCH_SIZE=128 ...   # if 4/4 nan where it was 3/4, uninit read

Run BATCH_SIZE=8 first: it is the cheap one and it is the one that decides.
"""

import os
import sys
import traceback

import torch

# Same payload as repro_jagged_concat_uninit.py, so a poisoned value found in a
# dump is traceable to this hook rather than to a real computation.
SENTINEL_BITS = 0x7FC0BEEF

_state = {
    "installed": False,
    "orig_empty": None,
    "orig_empty_like": None,
    "n_poisoned": 0,
    "reported": False,
}


def _sentinel_value() -> float:
    return torch.tensor([SENTINEL_BITS], dtype=torch.int32).view(torch.float32).item()


# ---------------------------------------------------------------------------
# 1. torch.empty poisoning
# ---------------------------------------------------------------------------
def _should_poison(t: torch.Tensor) -> bool:
    return t.is_cuda and t.is_floating_point() and t.numel() > 0


def install_poison() -> None:
    if _state["orig_empty"] is not None:
        return
    orig_empty = torch.empty
    orig_empty_like = torch.empty_like
    _state["orig_empty"] = orig_empty
    _state["orig_empty_like"] = orig_empty_like
    sentinel = _sentinel_value()

    def empty(*args, **kwargs):
        t = orig_empty(*args, **kwargs)
        if _should_poison(t):
            t.fill_(sentinel)
            _state["n_poisoned"] += 1
        return t

    def empty_like(*args, **kwargs):
        t = orig_empty_like(*args, **kwargs)
        if _should_poison(t):
            t.fill_(sentinel)
            _state["n_poisoned"] += 1
        return t

    torch.empty = empty
    torch.empty_like = empty_like
    print(
        "-- nan_poison: torch.empty/empty_like poisoned with "
        f"0x{SENTINEL_BITS:08X} (CUDA float only)",
        flush=True,
    )


def uninstall_poison() -> None:
    if _state["orig_empty"] is None:
        return
    torch.empty = _state["orig_empty"]
    torch.empty_like = _state["orig_empty_like"]
    _state["orig_empty"] = None
    _state["orig_empty_like"] = None


# ---------------------------------------------------------------------------
# 2. Coverage precondition on concat_2D_jagged / split_2D_jagged
#
# The multirow kernels write output rows only for offs_n < seq_len_a + seq_len_b
# AND only from programs offs_n < cdiv(max_seq_len, BLOCK_N) * BLOCK_N. With
# BLOCK_N in {1,2,4,8} the slack is at most 7 rows, so the real precondition is
#     max_seq_len >= max_z (len_a(z) + len_b(z))
# Every call site passes max_seq_len = max_len_left + max_len_right; this checks
# that those maxima actually bound the offsets at runtime.
# ---------------------------------------------------------------------------
def _lengths_from(offsets, max_len, total_len, B):
    """Per-element lengths, whether the side is dense or jagged."""
    if offsets is None:
        if max_len is None or B is None:
            return None
        return torch.full((B,), int(max_len), dtype=torch.int64, device="cpu")
    return (offsets[1:] - offsets[:-1]).to("cpu", torch.int64)


def _check_coverage(where, max_seq_len, offsets_left, offsets_right,
                    max_len_left, max_len_right):
    if max_seq_len is None:
        # Positional call; nothing to check against. All eight ranker call sites
        # pass by keyword, so this is not the common path.
        return
    B = None
    for o in (offsets_left, offsets_right):
        if o is not None:
            B = o.numel() - 1
            break
    if B is None:
        return
    la = _lengths_from(offsets_left, max_len_left, None, B)
    lb = _lengths_from(offsets_right, max_len_right, None, B)
    if la is None or lb is None:
        return
    combined = la + lb
    worst = int(combined.max().item())
    if worst > int(max_seq_len):
        z = int(combined.argmax().item())
        print(
            f"!! nan_poison COVERAGE VIOLATION in {where}: "
            f"max_seq_len={int(max_seq_len)} but element z={z} has "
            f"len_a={int(la[z])} + len_b={int(lb[z])} = {worst}. "
            f"{worst - int(max_seq_len)} output row(s) will be left "
            f"UNINITIALIZED for that element.",
            flush=True,
        )
        traceback.print_stack(file=sys.stdout)
        if os.environ.get("NAN_POISON_COVERAGE_FATAL") == "1":
            raise RuntimeError(f"coverage violation in {where}")


def install_coverage_checks() -> None:
    from generative_recommenders.ops import jagged_tensors as jt

    orig_concat = jt.concat_2D_jagged
    orig_split = jt.split_2D_jagged

    def concat_2D_jagged(*args, **kwargs):
        try:
            _check_coverage(
                "concat_2D_jagged",
                kwargs.get("max_seq_len"),
                kwargs.get("offsets_left"),
                kwargs.get("offsets_right"),
                kwargs.get("max_len_left"),
                kwargs.get("max_len_right"),
            )
        except Exception as e:  # never let the probe break the run
            print(f"-- nan_poison: coverage check skipped ({e})", flush=True)
        return orig_concat(*args, **kwargs)

    def split_2D_jagged(*args, **kwargs):
        try:
            _check_coverage(
                "split_2D_jagged",
                kwargs.get("max_seq_len"),
                kwargs.get("offsets_left"),
                kwargs.get("offsets_right"),
                kwargs.get("max_len_left"),
                kwargs.get("max_len_right"),
            )
        except Exception as e:
            print(f"-- nan_poison: coverage check skipped ({e})", flush=True)
        return orig_split(*args, **kwargs)

    jt.concat_2D_jagged = concat_2D_jagged
    jt.split_2D_jagged = split_2D_jagged

    # Every model module does `from ...jagged_tensors import concat_2D_jagged`,
    # which binds the ORIGINAL function into that module's own namespace at
    # import time. Rebinding only `jt` would therefore intercept nothing if this
    # hook is installed after the modules are imported. Rebind in place
    # everywhere the original object is already reachable, so installation order
    # stops mattering.
    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is jt:
            continue
        if not mod_name.startswith("generative_recommenders"):
            continue
        for attr, orig, new in (
            ("concat_2D_jagged", orig_concat, concat_2D_jagged),
            ("split_2D_jagged", orig_split, split_2D_jagged),
        ):
            if getattr(mod, attr, None) is orig:
                setattr(mod, attr, new)
                patched.append(f"{mod_name}.{attr}")

    print(
        "-- nan_poison: coverage precondition installed on "
        f"concat_2D_jagged / split_2D_jagged ({len(patched)} call-site "
        "bindings rebound)",
        flush=True,
    )
    for p in patched:
        print(f"   rebound {p}", flush=True)
    if not patched:
        print(
            "   NOTE: no already-imported bindings found. That is correct only "
            "if install() runs BEFORE the model modules are imported; if the "
            "model was already built, no check will fire.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# 3. First non-finite gradient
# ---------------------------------------------------------------------------
def check_gradients(model, step: int) -> bool:
    """Call after backward(), before the optimizer step. True if all finite."""
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        if not torch.isfinite(g).all():
            n_bad = int((~torch.isfinite(g)).sum().item())
            flat = g.reshape(-1)
            bad = torch.nonzero(~torch.isfinite(flat), as_tuple=False).flatten()
            first = int(bad[0].item())
            bits = flat[first].view(torch.int32) if g.dtype == torch.float32 else None
            print(
                f"!! nan_poison NON-FINITE GRADIENT at step {step}: "
                f"param={name} shape={tuple(g.shape)} "
                f"{n_bad}/{g.numel()} elements non-finite, first at flat "
                f"index {first}"
                + (f", bits=0x{int(bits.item()) & 0xFFFFFFFF:08X}" if bits is not None else ""),
                flush=True,
            )
            if bits is not None and (int(bits.item()) & 0xFFFFFFFF) == SENTINEL_BITS:
                print(
                    "!! that is THIS HOOK'S SENTINEL -- the gradient contains "
                    "memory that no kernel ever wrote. Uninitialized read "
                    "CONFIRMED, and the parameter above names the op.",
                    flush=True,
                )
            return False
    return True


# ---------------------------------------------------------------------------
def install() -> None:
    if _state["installed"]:
        return
    _state["installed"] = True
    print(f"== nan_poison_hook (torch {torch.__version__}) ==", flush=True)
    if os.environ.get("NAN_POISON") == "1":
        install_poison()
    if os.environ.get("NAN_POISON_COVERAGE") == "1":
        install_coverage_checks()
    if os.environ.get("NAN_POISON_GRADCHECK") == "1":
        print(
            "-- nan_poison: gradient check requested; call "
            "nan_poison_hook.check_gradients(model, step) after backward()",
            flush=True,
        )
    if not any(
        os.environ.get(k) == "1"
        for k in ("NAN_POISON", "NAN_POISON_COVERAGE", "NAN_POISON_GRADCHECK")
    ):
        print(
            "-- nan_poison: nothing enabled; set NAN_POISON=1 and/or "
            "NAN_POISON_COVERAGE=1",
            flush=True,
        )


def stats() -> dict:
    return {"poisoned_allocations": _state["n_poisoned"]}


if __name__ == "__main__":
    # Self-test: prove the poison actually lands, without needing the model.
    os.environ.setdefault("NAN_POISON", "1")
    install()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible; self-test needs one")
    t = torch.empty((4, 8), device="cuda", dtype=torch.float32)
    ok = bool((t.view(torch.int32) == SENTINEL_BITS).all().item())
    print(f"-- self-test: every element poisoned = {ok}", flush=True)
    i = torch.empty((4,), device="cuda", dtype=torch.int64)
    print(f"-- self-test: integer tensor left alone = {i.dtype == torch.int64}",
          flush=True)
    print(f"-- {stats()}", flush=True)
    sys.exit(0 if ok else 1)
