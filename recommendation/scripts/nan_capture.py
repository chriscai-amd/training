"""Capture the exact inputs + state of one training step, for offline replay.

Motivation: the ts=0 run goes non-finite at global step 11 and never recovers
(10 finite steps, then absorbing nan). Step 11 is cheap to reach (~2 min), so a
single-step capture turns every later attribution experiment from a multi-minute
training run into a seconds-long replay.

Enable with NAN_CAPTURE_STEP=<step> (or a comma list) and NAN_CAPTURE_DIR=<dir>.
Entirely env-gated: when NAN_CAPTURE_STEP is unset every entry point returns
immediately, so the default training path is unchanged.

Deliberately does NOT dump the full model state. At step 11 the LR is ~8.75e-10,
so weights are still essentially at initialization; what matters for replay is
the batch, the dense params, and WHICH embedding rows the batch indexes. Dumping
sharded embedding tables would be ~200 GB and is not needed to reproduce.
"""
import os
import time
from typing import Any, Dict, Optional

import torch

STEPS = {
    int(s) for s in os.environ.get("NAN_CAPTURE_STEP", "").replace(" ", "").split(",") if s
}
DIR = os.environ.get("NAN_CAPTURE_DIR", "/data/mlperf_dlrm_v4/nan_capture")
# Scan every step's loss terms from this step on (cheap: one .item() per term).
SCAN_FROM = int(os.environ.get("NAN_CAPTURE_SCAN_FROM", "1"))
ENABLED = bool(STEPS)

_MAX_PARAM_NUMEL = int(os.environ.get("NAN_CAPTURE_MAX_PARAM", str(20_000_000)))

# Capture the FIRST step whose loss goes non-finite, plus the step before it.
# This exists because the onset step is NOT stable: with SEED=1 pinned and
# identical data order, onset was step 11 in one run and step 5 in the next.
# A fixed NAN_CAPTURE_STEP therefore cannot be aimed -- it has to be triggered.
# Holding the previous step's blob gives the clean -> broken PAIR, which is what
# makes the trigger diffable.
ON_FIRST = os.environ.get("NAN_CAPTURE_ON_FIRST", "0") == "1"
# Stop building per-step blobs after this step (building costs ~0.2s + ~138 MiB
# each; onset has always been well inside 40).
HOLD_UNTIL = int(os.environ.get("NAN_CAPTURE_HOLD_UNTIL", "40"))
ENABLED = ENABLED or ON_FIRST

_prev_blob: Optional[Dict[str, Any]] = None
_prev_step: int = -1
_fired: bool = False
_scan_announced: bool = False


def _p(msg: str) -> None:
    print(f"[nan-capture] {msg}", flush=True)


def _kjt_to_dict(kjt: Any) -> Optional[Dict[str, Any]]:
    if kjt is None:
        return None
    out: Dict[str, Any] = {}
    for attr in ("keys",):
        try:
            out[attr] = list(kjt.keys())
        except Exception:
            pass
    for attr in ("values", "lengths", "offsets"):
        try:
            v = getattr(kjt, attr)()
            if isinstance(v, torch.Tensor):
                out[attr] = v.detach().cpu()
        except Exception:
            pass
    try:
        w = kjt.weights_or_none()
        if isinstance(w, torch.Tensor):
            out["weights"] = w.detach().cpu()
    except Exception:
        pass
    return out


def scan_terms(step: int, aux_losses: Dict[str, torch.Tensor]) -> Optional[str]:
    """Name the first non-finite loss term. Returns the term name, or None.

    This is the cheapest possible attribution: it runs BEFORE backward, so a
    non-finite term here proves the forward produced it rather than the
    optimizer or a previous step's update.
    """
    if not ENABLED or step < SCAN_FROM or not aux_losses:
        return None
    bad = []
    for name, v in aux_losses.items():
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        if not bool(torch.isfinite(v).all().item()):
            bad.append(f"{name}={v.detach().float().reshape(-1)[:4].tolist()}")
    if bad:
        global _scan_announced
        if not _scan_announced:
            _scan_announced = True
            _p(f"FIRST NON-FINITE at step {step}: {'; '.join(bad)}")
            finite = [n for n in aux_losses if not any(b.startswith(n + '=') for b in bad)]
            _p(f"step {step}: other terms finite: {finite}")
        return bad[0].split("=")[0]
    return None


def capture(
    step: int,
    sample: Any,
    model: Any,
    aux_losses: Optional[Dict[str, torch.Tensor]] = None,
    preds: Any = None,
    labels: Any = None,
    weights: Any = None,
    tag: str = "",
) -> None:
    """Serialize one step's inputs and dense state to NAN_CAPTURE_DIR."""
    blob = build_blob(step, sample, model, aux_losses, preds, labels, weights, tag)
    write_blob(blob, tag)


def build_blob(
    step: int,
    sample: Any,
    model: Any,
    aux_losses: Optional[Dict[str, torch.Tensor]] = None,
    preds: Any = None,
    labels: Any = None,
    weights: Any = None,
    tag: str = "",
) -> Dict[str, Any]:
    blob: Dict[str, Any] = {"step": step, "tag": tag}

    blob["uih_features_kjt"] = _kjt_to_dict(getattr(sample, "uih_features_kjt", None))
    blob["candidates_features_kjt"] = _kjt_to_dict(
        getattr(sample, "candidates_features_kjt", None)
    )

    # The index set the batch actually touches -- this is what lets a replay
    # slice the embedding tables instead of materializing all ~70M rows.
    touched: Dict[str, Any] = {}
    for fname in ("uih_features_kjt", "candidates_features_kjt"):
        d = blob.get(fname)
        if d and isinstance(d.get("values"), torch.Tensor):
            touched[fname] = torch.unique(d["values"])
    blob["touched_rows"] = touched

    for nm, t in (("preds", preds), ("labels", labels), ("weights", weights)):
        if isinstance(t, torch.Tensor):
            blob[nm] = t.detach().cpu()
        elif isinstance(t, dict):
            blob[nm] = {
                k: v.detach().cpu() for k, v in t.items() if isinstance(v, torch.Tensor)
            }

    if aux_losses:
        blob["aux_losses"] = {
            k: v.detach().cpu() for k, v in aux_losses.items() if isinstance(v, torch.Tensor)
        }

    dense: Dict[str, torch.Tensor] = {}
    skipped = 0
    try:
        for name, p in model.named_parameters():
            if p is None:
                continue
            if p.numel() > _MAX_PARAM_NUMEL:
                skipped += 1
                continue
            try:
                dense[name] = p.detach().float().cpu()
            except Exception:
                skipped += 1
    except Exception as e:
        _p(f"named_parameters failed: {e}")
    blob["dense_params"] = dense
    blob["dense_skipped_large"] = skipped

    blob["rng"] = {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    return blob


def write_blob(blob: Dict[str, Any], tag: str = "") -> None:
    t0 = time.time()
    os.makedirs(DIR, exist_ok=True)
    step = blob.get("step", -1)
    tag = tag or blob.get("tag", "")
    blob["tag"] = tag
    path = os.path.join(DIR, f"step{step}{('_' + tag) if tag else ''}.pt")
    torch.save(blob, path)
    sz = os.path.getsize(path) / (1 << 20)
    _p(
        f"step {step} tag={tag or '-'} saved {path} ({sz:.1f} MiB, "
        f"{len(blob.get('dense_params', {}))} dense params) in {time.time()-t0:.1f}s"
    )


def step_hook(
    step: int,
    sample: Any,
    model: Any,
    aux_losses: Dict[str, torch.Tensor],
    preds: Any = None,
    labels: Any = None,
    weights: Any = None,
) -> None:
    """Single entry point called before backward on every training step.

    Fixed-step mode (NAN_CAPTURE_STEP) and triggered mode (NAN_CAPTURE_ON_FIRST)
    are independent and may both be on.
    """
    global _prev_blob, _prev_step, _fired
    if not ENABLED:
        return

    cur: Optional[Dict[str, Any]] = None
    if ON_FIRST and not _fired and step <= HOLD_UNTIL:
        cur = build_blob(step, sample, model, aux_losses, preds, labels, weights)

    bad = scan_terms(step, aux_losses)

    if ON_FIRST and not _fired and bad:
        _fired = True
        if _prev_blob is not None:
            _p(f"trigger at step {step}; writing clean pair from step {_prev_step}")
            write_blob(_prev_blob, tag="last_clean")
        else:
            _p(f"trigger at step {step} but NO clean predecessor held "
               f"(onset at or before the first captured step)")
        if cur is not None:
            write_blob(cur, tag="first_bad")
        _prev_blob = None

    if ON_FIRST and not _fired:
        _prev_blob, _prev_step = cur, step

    # fixed-step mode
    if step in STEPS:
        b = cur if cur is not None else build_blob(
            step, sample, model, aux_losses, preds, labels, weights
        )
        write_blob(b, tag="pre_bwd")
