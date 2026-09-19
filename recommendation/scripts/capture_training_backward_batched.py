#!/usr/bin/env python3
"""Bounded uncapped training with batched output-backward endpoint checks.

This separate entrypoint selects the output-gradient GEMM and normalization
boundaries, defaults to STU layer 1, and explicitly requires an uncapped
attention configuration. It reuses the existing cold-start production-loop
wrapper and fault captures. Each boundary scans all logical floating values
and synchronizes before its scan; only endpoint transfers are batched.

Input faults stop before the selected operation. Output-fault input snapshots
are post-call observations and cannot establish pristine original-call replay.
No optimizer, forward-producer or residual/autograd boundary is added here.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
from pathlib import Path
import sys

import capture_training_backward as base


OUTPUT_OPERATIONS = ("hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd")
VARIANT = "uncapped_output_backward_batched_endpoints_v1"


def configure_environment(options, environment):
    if options["capture_mode"] != "monitor_on_anomaly":
        raise ValueError("Batched endpoint training requires monitor_on_anomaly")
    operations = options["operations"]
    if isinstance(operations, str) or not operations or set(operations) - set(OUTPUT_OPERATIONS):
        raise ValueError("Select only output-gradient GEMM and/or normalization")
    if environment.get("HSTU_BWD_MAX_VGPR", "0") != "0":
        raise ValueError("Uncapped batched endpoint training requires HSTU_BWD_MAX_VGPR=0")
    # Reuse every existing diagnostic conflict check on a private candidate.
    # The original helper requires cap256; this entrypoint deliberately changes
    # only that requirement. No cap256 value reaches the live environment.
    candidate = dict(environment)
    candidate.pop("HSTU_BWD_MAX_VGPR", None)
    recorded = {**options, "monitor_variant": VARIANT, "attention_vgpr_cap": 0}
    base.configure_environment(recorded, candidate)
    candidate["HSTU_BWD_MAX_VGPR"] = "0"
    environment.update(candidate)


def diagnostic_worker(local_rank, world_size, node_rank, gpus_per_node,
                      master_addr, master_port, gin_file, mode):
    if (local_rank, world_size, node_rank, gpus_per_node, mode) != (0, 1, 0, 1, "streaming-train-eval"):
        raise ValueError("Unsupported diagnostic topology or mode")
    options = json.loads(os.environ["NAN_TRAINING_OPTIONS"])
    if (options.get("monitor_variant") != VARIANT or options.get("attention_vgpr_cap") != 0
            or os.environ.get("HSTU_BWD_MAX_VGPR") != "0"):
        raise ValueError("Batched training worker lost its explicit uncapped provenance")
    import gin
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap

    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()
    utils = importlib.import_module("generative_recommenders.dlrm_v4.train.utils")
    trainer = importlib.import_module("generative_recommenders.dlrm_v4.train.train_ranker")
    from nan_backward_training_batched import BatchedTrainingBackwardBoundaryProbe

    original = utils.streaming_train_eval_loop

    @functools.wraps(original)
    def instrumented(*args, **kwargs):
        return base.run_instrumented_loop(original, options, args, kwargs,
                                         probe_factory=BatchedTrainingBackwardBoundaryProbe)

    utils.streaming_train_eval_loop = instrumented
    try:
        trainer._main_func(local_rank, world_size, node_rank, gpus_per_node,
                           master_addr, master_port, gin_file, mode)
    except SystemExit as error:
        path = Path(options["directory"]) / "outcome.json"
        if error.code == 42 and path.is_file() and json.loads(path.read_text()).get("status") == "bounded_complete":
            print(f"[batched-training-boundary] completed requested {options['steps']} steps: {path}", flush=True)
            return
        raise
    finally:
        utils.streaming_train_eval_loop = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="New worker artifact directory")
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--capture-mode", choices=("monitor_on_anomaly",), default="monitor_on_anomaly")
    parser.add_argument("--target", default="_stu_layers.1.")
    parser.add_argument("--operations", default=",".join(OUTPUT_OPERATIONS))
    parser.add_argument("--abs-threshold", type=float, default=1e6)
    parser.add_argument("--dataset", default="yambda-5b")
    args = parser.parse_args(argv)
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
        return trainer.main()
    finally:
        trainer._main_func, sys.argv = original_worker, original_argv


if __name__ == "__main__":
    main()
