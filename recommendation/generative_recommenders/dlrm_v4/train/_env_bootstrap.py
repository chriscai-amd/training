"""Gin-driven env-var bootstrap.

Some env vars must be set *before* certain modules import (e.g. Triton's
`@triton.autotune` decorator reads `TRITON_FULL_AUTOTUNE` at module load
time, well before `gin.parse_config_file` runs in the default ordering).

`apply_env_bootstrap()` is `@gin.configurable`, so the gin file becomes the
canonical source of truth. `train_ranker.py` parses gin with
`skip_unknown=True` early in `_main_func`, calls this function to push the
bindings into `os.environ`, then does the heavy imports.
"""

import logging
import os
import re
from typing import Dict, Optional

import gin

logger: logging.Logger = logging.getLogger(__name__)


@gin.configurable
def apply_env_bootstrap(
    TRITON_FULL_AUTOTUNE: Optional[bool] = None,
) -> None:
    # A pre-set environment variable wins over the gin binding. The pinned
    # triton configs are MI350X-specific, so a different GPU arch (e.g. B200
    # sm_100) sets TRITON_FULL_AUTOTUNE=1 in the launcher environment to
    # re-enable the full autotune search WITHOUT editing this (AMD-default)
    # gin file. Cross-cluster launchers thus stay config-as-code via env.
    if "TRITON_FULL_AUTOTUNE" in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set TRITON_FULL_AUTOTUNE=%s (overrides gin binding)",
            os.environ["TRITON_FULL_AUTOTUNE"],
        )
    elif TRITON_FULL_AUTOTUNE is not None:
        os.environ["TRITON_FULL_AUTOTUNE"] = "1" if TRITON_FULL_AUTOTUNE else "0"
        logger.info("env bootstrap: TRITON_FULL_AUTOTUNE=%s", os.environ["TRITON_FULL_AUTOTUNE"])


# =============================================================================
# MI450 hipBLASLt optimizations (docs/mi450_perf_opt.md). Each is OFF unless its
# gin binding turns it on; the defaults live in gin/yambda_5b.gin and per-run
# overrides in a gin overlay named by $DLRM_GIN_OVERLAY (see parse_gin below).
# As with TRITON_FULL_AUTOTUNE, a variable already present in the environment
# wins over the gin binding and is logged.
# =============================================================================
_HIPBLASLT_STOCK_TENSILE = (
    "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/"
    "hipblaslt/library/gfx1250"
)


def _set_env(key: str, value: str) -> None:
    if key in os.environ and os.environ[key] != value:
        logger.info("env bootstrap: honoring pre-set %s=%s (overrides gin)", key, os.environ[key])
        return
    os.environ[key] = value
    logger.info("env bootstrap: %s=%s", key, value)


@gin.configurable
def apply_launch_env(hipblaslt_lib_dir: Optional[str] = None) -> None:
    """OPT M1 -- hipBLASLt library swap. MUST run in the launcher process BEFORE
    the rank processes are spawned: LD_PRELOAD only takes effect at process
    start, and spawned ranks inherit this environment.

    hipblaslt_lib_dir: a directory holding a coherent libhipblaslt.so.1 +
    libtensilelite-host / liborigami / librocroller and hipblaslt/library/gfx1250
    (on a37-2: the ROCm 10.2 hipBLASLt with its NN Origami pool, NT catalog
    filtered to the stock stub). None = stock image library."""
    if not hipblaslt_lib_dir:
        return
    d = hipblaslt_lib_dir.rstrip("/")
    if not os.path.isfile(f"{d}/libhipblaslt.so.1"):
        raise FileNotFoundError(f"apply_launch_env.hipblaslt_lib_dir: no libhipblaslt.so.1 in {d}")
    _set_env("LD_PRELOAD", f"{d}/libhipblaslt.so.1")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld.startswith(d + ":") and ld != d:
        os.environ["LD_LIBRARY_PATH"] = d + (":" + ld if ld else "")
    os.environ["HIPBLASLT_TENSILE_LIBPATH"] = f"{d}/hipblaslt/library/gfx1250"
    logger.info("env bootstrap: hipBLASLt library swap -> %s", d)


@gin.configurable
def apply_hipblaslt_opts(
    tuning_override_file: Optional[str] = None,
    tn_wgrad_hstu: bool = False,
    tn_wgrad_preproc: bool = False,
    splitt_wgrad_hstu: bool = False,
    splitt_wgrad_preproc: bool = False,
    splitt_wgrad_splits: Optional[Dict[str, int]] = None,
    splitt_min_rows: int = 4096,
    splitt_audit_every: int = 0,
) -> None:
    """Per-rank, before the heavy imports (ops/triton/triton_addmm.py reads
    HSTU_TN_WGRAD at import; hipBLASLt reads the override file at handle init).

    tuning_override_file -- OPT M2: HIPBLASLT_TUNING_OVERRIDE_FILE (exact-shape
        solution pins; rows keyed on K = 2,097,152 do not match jagged T).
    tn_wgrad_hstu        -- OPT M3: HSTU UVQK / output-projection weight
        gradients issued TN on feature-major copies (G20, G19).
    tn_wgrad_preproc     -- OPT M4: preprocessor MLP Linear weight gradients
        issued TN (G25, G27, G29, G31).
    splitt_wgrad_hstu    -- OPT M5: split-T NT weight gradients (one strided-
        batched bmm over s T-chunks + fp32 reduce) for G20, G19. Needs the
        lib dir rocm-10.2_nnonly_v3/s103 or /s102 (M1 swap) so the batched
        problems select a known kernel. Exclusive with tn_wgrad_hstu.
    splitt_wgrad_preproc -- OPT M5: same for G25, G27, G29, G31. Exclusive
        with tn_wgrad_preproc.
    splitt_wgrad_splits  -- OPT M5: {"MxN": s} keyed on the hipBLASLt m x n
        of the NT wgrad (e.g. "2048x512" = G20); unlisted shapes keep s = 1
        (original call).
    splitt_min_rows      -- OPT M5: skip the split when T // s < this.
    splitt_audit_every   -- OPT M5 diagnostic: every N-th split call per shape
        also runs the original call and logs the difference and the chunk
        cancellation factor (extra GEMM + device sync; 0 = off, keep 0 for
        timed runs). The configured s are checked at import against the lib
        dir's m5_splits.json (an s the catalog does not pin raises)."""
    if tuning_override_file:
        _set_env("HIPBLASLT_TUNING_OVERRIDE_FILE", tuning_override_file)
    mode = {(False, False): "0", (True, False): "hstu",
            (False, True): "preproc", (True, True): "all"}[(bool(tn_wgrad_hstu), bool(tn_wgrad_preproc))]
    _set_env("HSTU_TN_WGRAD", mode)
    for scope, tn, st in (("hstu", tn_wgrad_hstu, splitt_wgrad_hstu),
                          ("preproc", tn_wgrad_preproc, splitt_wgrad_preproc)):
        if tn and st:
            raise ValueError(
                f"apply_hipblaslt_opts: tn_wgrad_{scope} and splitt_wgrad_{scope} are exclusive")
    smode = {(False, False): "0", (True, False): "hstu", (False, True): "preproc",
             (True, True): "all"}[(bool(splitt_wgrad_hstu), bool(splitt_wgrad_preproc))]
    _set_env("HSTU_SPLITT_WGRAD", smode)
    if smode != "0":
        sp = dict(splitt_wgrad_splits or {})
        for k, v in sp.items():
            if not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", str(k)) or not 1 <= int(v) <= 4096:
                raise ValueError(f"apply_hipblaslt_opts.splitt_wgrad_splits: bad entry {k!r}: {v!r}")
        if int(splitt_min_rows) < 1:
            raise ValueError(f"apply_hipblaslt_opts.splitt_min_rows={splitt_min_rows!r}")
        _set_env("HSTU_SPLITT_WGRAD_SPLITS", ",".join(f"{k}:{int(v)}" for k, v in sorted(sp.items())))
        _set_env("HSTU_SPLITT_MIN_ROWS", str(int(splitt_min_rows)))
        if int(splitt_audit_every) < 0:
            raise ValueError(f"apply_hipblaslt_opts.splitt_audit_every={splitt_audit_every!r}")
        _set_env("HSTU_SPLITT_AUDIT_EVERY", str(int(splitt_audit_every)))


def parse_gin(gin_file: str, skip_unknown: bool = False) -> None:
    """Parse the dataset gin, then the optional overlay named by
    $DLRM_GIN_OVERLAY (later bindings win). One overlay per A/B arm turns the
    MI450 optimizations on or off without editing the main gin file."""
    gin.parse_config_file(gin_file, skip_unknown=skip_unknown)
    overlay = os.environ.get("DLRM_GIN_OVERLAY", "")
    if overlay:
        gin.parse_config_file(overlay, skip_unknown=skip_unknown)
        logger.info("gin overlay: %s", overlay)
