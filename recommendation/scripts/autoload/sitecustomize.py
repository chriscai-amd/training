"""Auto-install the NaN tripwire when NAN_TRIPWIRE=1.

Put this directory on PYTHONPATH; Python imports sitecustomize automatically
at interpreter start, before any user code, so the tripwire's step counter and
import hook are in place before generative_recommenders is imported.
"""
import os
import sys

# Re-entrancy guard -- load-bearing, do not drop.
#
# PYTHONPATH is inherited by every child process, and this file therefore runs
# in children too. Importing nan_tripwire imports torch, and torch's ROCm arch
# detection shells out to `offload-arch`, which is itself a Python console
# script. Without this marker that child re-runs sitecustomize, re-imports
# torch, spawns another offload-arch, and so on: the run never reaches training
# and instead builds an unbounded process chain. Observed once for real --
# 14,131 offload-arch processes, no log output, GPU idle. The marker is placed
# in os.environ (not a module global) precisely because it must cross the fork.
_GUARD = "_NAN_TRIPWIRE_ACTIVE"

if os.environ.get("NAN_TRIPWIRE", "0") == "1" and os.environ.get(_GUARD) != "1":
    os.environ[_GUARD] = "1"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import nan_tripwire

        nan_tripwire.install()
    except Exception as e:  # never break the run because the instrument failed
        print(f"[nan-tripwire] install failed: {e}", flush=True)
