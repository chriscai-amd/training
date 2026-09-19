"""Opt-in bounded training with selected HSTU backward boundary observations.

Uses train_ranker's normal spawn launcher and production streaming loop. Model,
data, optimizer, precision, allocation, and RNG configuration come from the
same gin/environment as normal training. This diagnostic requires a cold
single-GPU start at timestamp zero with cap256; it disables evaluation and
checkpointing and stops at a positive global train-step bound.

The explicit monitor_on_anomaly mode saves input bytes only after a trigger;
an output-stage artifact therefore cannot establish pristine replay inputs.
Use pristine mode for that stronger guarantee, at substantial D2H/CPU cost.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OPERATIONS = ("hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd")
EXTENDED_OPERATIONS = ("hstu_attention_bwd", "hstu_silu_bwd", "hstu_output_weight_mm")
ALL_OPERATIONS = (*DEFAULT_OPERATIONS, "triton_addmm_bwd", "triton_weighted_layer_norm_bwd",
                  *EXTENDED_OPERATIONS)
LIMITS = [
    "Only the selected layer(s) and operations are observed; a bad input establishes propagation into that boundary, not its producer.",
    "Unselected operations, attention backward internals, group-norm alternative, residual/autograd accumulation, embeddings and optimizer internals are not directly observed.",
    "No full model, optimizer, dataset or allocator state is captured; this is an operation-boundary diagnostic, not a complete training replay.",
    "Synchronization, reductions, allocations and any D2H copies alter timing; a finite run does not clear an intermittent failure.",
    "Sparse fused optimizer updates can already have occurred inside backward when the probe stops.",
]


def _rng_equal(left, right):
    import numpy as np
    import torch

    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_rng_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_rng_equal(a, b) for a, b in zip(left, right))
    return left == right


def _archive_sources(directory, context):
    """Archive the exact source bytes whose hashes appear in worker context."""
    entries = dict(context["source_sha256"])
    for path in sorted((ROOT / "generative_recommenders/dlrm_v4/train/gin").glob("*.gin")):
        entries[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    target = directory / "sources.tar.gz"
    temporary = target.with_suffix(".gz.tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for relative, expected in sorted(entries.items()):
            data = (ROOT / relative).read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Source changed while collecting provenance: {relative}")
            entry = tarfile.TarInfo(relative)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    os.replace(temporary, target)
    return {"path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "entries": entries}


def run_instrumented_loop(original, options, args, kwargs, *, probe_factory=None,
                          context_factory=None, archive_factory=None):
    """Install after model construction; directly wrap forward used by the loop.

    Injection points allow CPU tests without importing the production trainer or
    initializing CUDA. The original decorated loop still resolves all ordinary
    gin bindings; only explicit diagnostic run controls are passed here.
    """
    import torch
    from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
    from nan_replay_capture import configuration_context, write_json
    from nan_replay_state import capture_rng

    if probe_factory is None:
        if set(options["operations"]).intersection(EXTENDED_OPERATIONS):
            from nan_backward_training_extended import ExtendedTrainingBackwardBoundaryProbe
            probe_factory = ExtendedTrainingBackwardBoundaryProbe
        else:
            from nan_backward_training_probe import TrainingBackwardBoundaryProbe
            probe_factory = TrainingBackwardBoundaryProbe
    context_factory = context_factory or configuration_context
    archive_factory = archive_factory or _archive_sources
    if args:
        raise ValueError("Training diagnostic expects train_ranker's keyword loop invocation")
    model, metrics = kwargs["model"], kwargs["metric_logger"]
    if int(kwargs["rank"]) != 0 or int(metrics.global_step["train"]) != 0:
        raise ValueError("Training boundary diagnostic requires rank zero and a cold global step zero")
    if kwargs.get("resume_train_ts") is not None or not kwargs.get("resume_cold_start", False):
        raise ValueError("Training boundary diagnostic does not accept a resumed checkpoint")
    directory = Path(options["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    initial_rng = capture_rng()
    context = context_factory()
    context["limits"] = LIMITS
    context["training_diagnostic"] = dict(options)
    context["source_archive"] = archive_factory(directory, context)
    context["launch"] = "normal train_ranker.main spawn launcher; worker installs scoped loop wrapper after gin environment bootstrap"
    write_json(directory / "context.json", context)
    torch.save(initial_rng, directory / "initial_rng.pt")
    outcome = {"status": "installing", "requested_steps": options["steps"],
               "last_attempted_step": 0, "completed_steps": 0, "limits": LIMITS}
    write_json(directory / "outcome.json", outcome)
    probe = None
    original_forward = model.forward
    had_own_forward = "forward" in model.__dict__
    own_forward = model.__dict__.get("forward")
    wrapped_forward = None
    try:
        probe = probe_factory(model, directory / "boundaries", mode=options["capture_mode"],
                              selected_operations=options["operations"])
        if not _rng_equal(initial_rng, capture_rng()):
            raise BoundaryProbeError("Diagnostic installation changed default RNG state")
        context["installation_rng_unchanged"] = True
        write_json(directory / "context.json", context)

        @functools.wraps(original_forward)
        def forward(*forward_args, **forward_kwargs):
            if not model.training:
                raise BoundaryProbeError("Evaluation reached this bounded training diagnostic")
            step = int(metrics.global_step["train"]) + 1
            if step > options["steps"] or step != outcome["last_attempted_step"] + 1:
                raise BoundaryProbeError(f"Unexpected root training forward at step {step}")
            probe.set_attempt("training", 0, step)
            outcome["last_attempted_step"] = step
            return original_forward(*forward_args, **forward_kwargs)

        wrapped_forward = forward
        model.forward = forward
        outcome["status"] = "running"
        write_json(directory / "outcome.json", outcome)
        call_kwargs = {**kwargs, "start_ts": 0, "die_at_step": options["steps"],
                       "eval_every_n_windows": 0, "eval_every_data_pct": 0.0}
        original(**call_kwargs)
        outcome["status"] = "returned_before_bound"
        raise BoundaryProbeError("Training returned without reaching its requested global step bound")
    except BoundaryAnomalyError as error:
        outcome.update(status="anomaly", dump_path=str(error.dump_path), error=str(error))
        raise
    except SystemExit as error:
        completed = int(metrics.global_step["train"])
        if error.code == 42 and completed == options["steps"] and outcome["last_attempted_step"] == completed:
            outcome.update(status="bounded_complete", trainer_exit_code=42)
        else:
            outcome.update(status="unexpected_exit", error=str(error), trainer_exit_code=error.code)
        raise
    except BaseException as error:
        if outcome["status"] != "returned_before_bound":
            outcome["status"] = "error"
        outcome.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        if wrapped_forward is not None and model.__dict__.get("forward") is wrapped_forward:
            if had_own_forward:
                model.forward = own_forward
            else:
                del model.forward
        if probe is not None:
            probe.close()
        outcome.update(completed_steps=int(metrics.global_step["train"]),
                       seconds=time.monotonic() - started)
        write_json(directory / "outcome.json", outcome)


def diagnostic_worker(local_rank, world_size, node_rank, gpus_per_node,
                      master_addr, master_port, gin_file, mode):
    """Top-level spawn target: all worker patches are installed in the child."""
    if (local_rank, world_size, node_rank, gpus_per_node, mode) != (0, 1, 0, 1, "streaming-train-eval"):
        raise ValueError("Unsupported diagnostic topology or mode")
    options = json.loads(os.environ["NAN_TRAINING_OPTIONS"])
    import gin
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap

    # This same idempotent early parse is repeated by _main_func. It must occur
    # here before importing utils so its Triton decorators see the normal env.
    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()
    utils = importlib.import_module("generative_recommenders.dlrm_v4.train.utils")
    trainer = importlib.import_module("generative_recommenders.dlrm_v4.train.train_ranker")
    original = utils.streaming_train_eval_loop

    @functools.wraps(original)
    def instrumented(*args, **kwargs):
        return run_instrumented_loop(original, options, args, kwargs)

    utils.streaming_train_eval_loop = instrumented
    try:
        trainer._main_func(local_rank, world_size, node_rank, gpus_per_node,
                           master_addr, master_port, gin_file, mode)
    except SystemExit as error:
        path = Path(options["directory"]) / "outcome.json"
        if error.code == 42 and path.is_file() and json.loads(path.read_text()).get("status") == "bounded_complete":
            print(f"[training-boundary] completed requested {options['steps']} steps: {path}", flush=True)
            return
        raise
    finally:
        utils.streaming_train_eval_loop = original


def configure_environment(options, environment):
    """Reject conflicting inherited controls, then explicitly pin diagnostics."""
    if options["steps"] <= 0 or not math.isfinite(options["abs_threshold"]) or options["abs_threshold"] <= 0:
        raise ValueError("Steps and the finite absolute threshold must be positive")
    if (not options["operations"] or set(options["operations"]) - set(ALL_OPERATIONS)
            or len(set(options["operations"])) != len(options["operations"])):
        raise ValueError(f"Operations must be a nonempty selection from {ALL_OPERATIONS}")
    for key in ("NAN_REPLAY_DIR", "NAN_CAPTURE_STEP", "NAN_CAPTURE_ON_FIRST", "NAN_TRIPWIRE_DIR"):
        if environment.get(key):
            raise ValueError(f"Disable {key} before this isolated diagnostic")
    if environment.get("NAN_MODULE_PROBE") == "1" or environment.get("DEBUG_NAN_HOOKS", "0") != "0":
        raise ValueError("Disable NAN_MODULE_PROBE and DEBUG_NAN_HOOKS before this diagnostic")
    required = {"GPUS_PER_NODE": "1", "NNODES": "1", "NODE_RANK": "0", "START_TS": "0",
                "HSTU_BWD_MAX_VGPR": "256", "AMDGCN_USE_BUFFER_OPS": "0",
                "TRITON_FULL_AUTOTUNE": "0", "TRITON_ALLOW_PIPELINING": "0",
                "CKPT_PATH": "", "EVAL_EVERY_N_WINDOWS": "0", "EVAL_EVERY_DATA_PCT": "0",
                "DIE_AT_STEP": str(options["steps"]), "NUM_TRAIN_BATCHES": "0",
                "NAN_BACKWARD_SAVE_ALL": "0"}
    for key, value in required.items():
        if key in environment and environment[key] != value:
            raise ValueError(f"{key}={environment[key]!r} conflicts with required diagnostic value {value!r}")
    environment.update(required)
    environment["NAN_BACKWARD_TARGET"] = options["target"]
    environment["NAN_BACKWARD_ABS_THRESHOLD"] = str(options["abs_threshold"])
    environment["NAN_TRAINING_OPTIONS"] = json.dumps(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="New worker artifact directory; must not exist")
    parser.add_argument("--steps", type=int, required=True, help="Positive global train-step limit")
    parser.add_argument("--capture-mode", choices=("pristine", "monitor_on_anomaly"), required=True)
    parser.add_argument("--target", default="_stu_layers.1.", help="Parameter-name substring, or '*' for every STU layer")
    parser.add_argument("--operations", default=",".join(DEFAULT_OPERATIONS))
    parser.add_argument("--abs-threshold", type=float, default=1e20)
    parser.add_argument("--dataset", default="yambda-5b")
    args = parser.parse_args()
    options = vars(args)
    options["directory"] = str(Path(options["directory"]).resolve())
    options["operations"] = [name.strip() for name in args.operations.split(",") if name.strip()]
    if Path(options["directory"]).exists():
        parser.error("--directory must be new to keep provenance and captures separate")
    try:
        configure_environment(options, os.environ)
    except ValueError as error:
        parser.error(str(error))
    trainer = importlib.import_module("generative_recommenders.dlrm_v4.train.train_ranker")
    original_worker, original_argv = trainer._main_func, sys.argv
    trainer._main_func = diagnostic_worker
    sys.argv = [sys.argv[0], "--dataset", args.dataset, "--mode", "streaming-train-eval"]
    try:
        trainer.main()
    finally:
        trainer._main_func, sys.argv = original_worker, original_argv


if __name__ == "__main__":
    main()
