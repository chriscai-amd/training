#!/usr/bin/env python3
"""Uncapped cold-start training with an upper-layer backward propagation monitor.

All seven operations of the upper STU layer are scanned before and after.
Only incoming output-GEMM inputs of the adjacent lower layer are scanned;
faults stop before that GEMM. Small scalar witnesses of both upper residual
paths accompany lower-layer captures. Default scope is layer2 -> layer1.
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

ALL_OPERATIONS = base.ALL_OPERATIONS
VARIANT = "uncapped_upstream_batched_layer2_to_layer1_v1"


def layer_selection(upper_layer, lower_layer):
    if (type(upper_layer) is not int or type(lower_layer) is not int
            or lower_layer < 0 or upper_layer != lower_layer + 1):
        raise ValueError("Select adjacent nonnegative upper/lower STU layer indices")
    return {f"_stu_layers.{upper_layer}.": list(ALL_OPERATIONS),
            f"_stu_layers.{lower_layer}.": ["hstu_output_grad_mm"]}


def configure_environment(options, environment):
    if options["capture_mode"] != "monitor_on_anomaly":
        raise ValueError("Upstream batched training requires monitor_on_anomaly")
    selection = layer_selection(options["upper_layer"], options["lower_layer"])
    if environment.get("HSTU_BWD_MAX_VGPR", "0") != "0":
        raise ValueError("Uncapped upstream training requires HSTU_BWD_MAX_VGPR=0")
    candidate = dict(environment)
    candidate.pop("HSTU_BWD_MAX_VGPR", None)
    recorded = {**options, "target": "*", "operations": list(ALL_OPERATIONS),
                "monitor_variant": VARIANT, "attention_vgpr_cap": 0,
                "layer_operation_selection": selection, "lower_witness_input_only": True,
                "residual_witness_scope": "upper original dout and preprocessing d_x scalar summaries retained per attempt; no elementwise addition or byte-identity check"}
    base.configure_environment(recorded, candidate)
    candidate["HSTU_BWD_MAX_VGPR"] = "0"
    environment.update(candidate)


def diagnostic_worker(local_rank, world_size, node_rank, gpus_per_node,
                      master_addr, master_port, gin_file, mode):
    if (local_rank, world_size, node_rank, gpus_per_node, mode) != (0, 1, 0, 1, "streaming-train-eval"):
        raise ValueError("Unsupported diagnostic topology or mode")
    options = json.loads(os.environ["NAN_TRAINING_OPTIONS"])
    if (options.get("monitor_variant") != VARIANT or options.get("attention_vgpr_cap") != 0
            or os.environ.get("HSTU_BWD_MAX_VGPR") != "0"
            or options.get("target") != "*" or os.environ.get("NAN_BACKWARD_TARGET") != "*"
            or options.get("operations") != list(ALL_OPERATIONS)
            or options.get("layer_operation_selection") != layer_selection(options["upper_layer"], options["lower_layer"])
            or options.get("lower_witness_input_only") is not True):
        raise ValueError("Upstream training worker lost its uncapped per-layer provenance")
    import gin
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap

    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()
    utils = importlib.import_module("generative_recommenders.dlrm_v4.train.utils")
    trainer = importlib.import_module("generative_recommenders.dlrm_v4.train.train_ranker")
    from nan_backward_training_upstream import UpstreamBatchedTrainingBackwardBoundaryProbe
    factory = functools.partial(UpstreamBatchedTrainingBackwardBoundaryProbe,
                                upper_layer=options["upper_layer"], lower_layer=options["lower_layer"])
    original = utils.streaming_train_eval_loop

    @functools.wraps(original)
    def instrumented(*args, **kwargs):
        return base.run_instrumented_loop(original, options, args, kwargs, probe_factory=factory)

    utils.streaming_train_eval_loop = instrumented
    try:
        trainer._main_func(local_rank, world_size, node_rank, gpus_per_node,
                           master_addr, master_port, gin_file, mode)
    except SystemExit as error:
        path = Path(options["directory"]) / "outcome.json"
        if error.code == 42 and path.is_file() and json.loads(path.read_text()).get("status") == "bounded_complete":
            print(f"[upstream-training-boundary] completed requested {options['steps']} steps: {path}", flush=True)
            return
        raise
    finally:
        utils.streaming_train_eval_loop = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="New worker artifact directory")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--capture-mode", choices=("monitor_on_anomaly",), default="monitor_on_anomaly")
    parser.add_argument("--upper-layer", type=int, default=2)
    parser.add_argument("--lower-layer", type=int, default=1)
    parser.add_argument("--abs-threshold", type=float, default=1e6)
    parser.add_argument("--dataset", default="yambda-5b")
    args = parser.parse_args(argv)
    options = vars(args)
    options["directory"] = str(Path(options["directory"]).resolve())
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
