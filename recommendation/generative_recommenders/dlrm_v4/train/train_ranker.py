# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
import os
import sys
import traceback

import gin
import torch
from torch import multiprocessing as mp
from torchrec.test_utils import get_free_port

# NOTE: heavy imports of generative_recommenders.dlrm_v4.* are deferred to
# inside _main_func so that gin-driven env-var bootstrap (see
# _env_bootstrap.apply_env_bootstrap) can run BEFORE the triton kernel
# modules evaluate their `@triton.autotune` decorators at module-load time.

logger: logging.Logger = logging.getLogger(__name__)


def _iter_tensors(obj: object):  # pyre-ignore[3]
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def _install_nonfinite_hooks(module: torch.nn.Module) -> None:
    """Log the first submodule whose forward emits a non-finite value.

    Diagnostic for the gfx1250 NaN: hooks fire child-before-parent, so the first
    line logged names the innermost producer. Each site reports whether the
    model's parameters were *already* non-finite, which separates a poisoned
    weight (NaN arrived via a previous backward) from a forward that produced
    NaN out of finite inputs. Gated by DEBUG_NAN_HOOKS because the scan forces a
    device sync on every module output.
    """
    reported: set[str] = set()
    # Counts root-module forwards so a report can be dated to a step. A table
    # that is already non-finite at forward 1 was poisoned at init or by the
    # sharding/init path, never by a backward.
    forwards: list[int] = [0]
    swept: list[bool] = [False]

    # Opt-in (DEBUG_NAN_HOOKS=2): sweeping every embedding table before the
    # first forward has itself wedged the GPU on gfx1250, so it is not on the
    # default diagnostic path.
    if os.environ.get("DEBUG_NAN_HOOKS") == "2":
        bad_at_init = [
            pname
            for pname, param in module.named_parameters()
            if param.is_floating_point() and not torch.isfinite(param.data).all()
        ]
        logger.warning(
            "[nan-hook] non-finite params before any forward: %d%s",
            len(bad_at_init),
            (" e.g. " + ", ".join(bad_at_init[:3])) if bad_at_init else "",
        )

    def _count_forward(
        _mod: torch.nn.Module, _inp: object, _out: object
    ) -> None:  # pyre-ignore[2]
        forwards[0] += 1

    module.register_forward_pre_hook(_count_forward)

    def _make_hook(name: str):  # pyre-ignore[3]
        def hook(
            _mod: torch.nn.Module, _inp: object, out: object
        ) -> None:  # pyre-ignore[2]
            if name in reported:
                return
            for idx, tensor in enumerate(_iter_tensors(out)):
                if not tensor.is_floating_point() or tensor.numel() == 0:
                    continue
                if torch.isfinite(tensor).all():
                    continue
                reported.add(name)
                # Sweeping every embedding table is expensive and has wedged
                # the GPU, so do it once for the whole run, on the first hit.
                if swept[0]:
                    params_note = "params not re-swept"
                else:
                    swept[0] = True
                    bad_params = [
                        pname
                        for pname, param in module.named_parameters()
                        if param.is_floating_point()
                        and not torch.isfinite(param.data).all()
                    ]
                    params_note = "non-finite params: %d%s" % (
                        len(bad_params),
                        (" e.g. " + ", ".join(bad_params[:3])) if bad_params else "",
                    )
                logger.error(
                    "[nan-hook] forward %d: non-finite output from %s (%s), "
                    "output #%d shape %s; %s",
                    forwards[0],
                    name,
                    type(_mod).__name__,
                    idx,
                    tuple(tensor.shape),
                    params_note,
                )
                return

        return hook

    for name, submodule in module.named_modules():
        submodule.register_forward_hook(_make_hook(name or "<root>"))


def _install_nonfinite_backward_hooks(module: torch.nn.Module) -> None:
    """Log the module that *manufactures* the first non-finite gradient.

    A NaN gradient propagates, so most reports are downstream noise. The
    producer is the module whose grad_output is finite while its grad_input is
    not: it was handed good gradients and returned bad ones. That is the
    criterion applied here, which on the gfx1250 NaN should name the offending
    Triton backward rather than everything below it.
    """
    reported: set[str] = set()

    def _all_finite(grads: object) -> bool:
        return all(
            torch.isfinite(t).all()
            for t in _iter_tensors(grads)
            if t.is_floating_point() and t.numel() > 0
        )

    def _make_hook(name: str):  # pyre-ignore[3]
        def hook(
            _mod: torch.nn.Module, grad_input: object, grad_output: object
        ) -> None:  # pyre-ignore[2]
            if name in reported:
                return
            out_ok = _all_finite(grad_output)
            in_ok = _all_finite(grad_input)
            if out_ok and not in_ok:
                reported.add(name)
                logger.error(
                    "[nan-bwd] PRODUCER: %s (%s) returned non-finite grad_input "
                    "from finite grad_output",
                    name,
                    type(_mod).__name__,
                )
            elif not out_ok and name not in reported:
                reported.add(name)
                logger.error(
                    "[nan-bwd] downstream: %s (%s) already received non-finite "
                    "grad_output",
                    name,
                    type(_mod).__name__,
                )

        return hook

    for name, submodule in module.named_modules():
        try:
            submodule.register_full_backward_hook(_make_hook(name or "<root>"))
        except (RuntimeError, AttributeError) as exc:
            logger.warning("[nan-bwd] cannot hook %s: %s", name, exc)


SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "movielens-1m": "movielens_1m.gin",
    "movielens-20m": "movielens_20m.gin",
    "movielens-13b": "movielens_13b.gin",
    "movielens-18b": "movielens_18b.gin",
    "streaming-400m": "streaming_400m.gin",
    "streaming-200b": "streaming_200b.gin",
    "streaming-100b": "streaming_100b.gin",
    "yambda-5b": "yambda_5b.gin",
}


def _main_func(
    local_rank: int,
    world_size: int,
    node_rank: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    gin_file: str,
    mode: str,
) -> None:
    # `local_rank` is the index handed out by mp.start_processes (0..gpus_per_node-1)
    # and indexes this node's GPUs. The GLOBAL rank is what every downstream
    # consumer wants (data sharding via StreamingWindowSampler, checkpoint I/O,
    # metrics), so derive it once and pass it through as `rank`. Only the CUDA
    # device must be node-local. Single-node (node_rank=0) → rank == local_rank,
    # exactly as before.
    rank = node_rank * gpus_per_node + local_rank
    device = torch.device(f"cuda:{local_rank}")
    logger.info(
        f"rank: {rank} (node_rank={node_rank} local_rank={local_rank}), "
        f"world_size: {world_size}, device: {device}"
    )
    # Phase 1: parse gin early with skip_unknown=True so env-bootstrap
    # bindings take effect BEFORE any module-level @gin.configurable
    # discovers itself. This is required because triton @triton.autotune
    # decorators in generative_recommenders.ops.triton.* read env vars at
    # module import time, and the heavy imports below pull those in.
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
    from generative_recommenders.dlrm_v4.train.mlperf_logging_utils import (
        get_mlperf_logger,
        mlperf_checkpoint_present,
    )

    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()

    # Cold-start vs resume, decided from the on-disk checkpoint BEFORE setup so
    # the one-time INIT/RUN markers fire on a genuine cold start only and are NOT
    # re-emitted on a resume relaunch — the MLPerf run (run_start..run_stop) spans
    # the resume as a single coherent event stream in one appended log file.
    mlperf_resume = mlperf_checkpoint_present(os.environ.get("CKPT_PATH", ""))
    # Rank-0-gated MLPerf logger, only for the streaming-train-eval path. `fresh`
    # truncates the log on cold start (one run_start per file) but appends on a
    # resume so the pre-crash events are preserved and continued.
    mlperf_logger = (
        get_mlperf_logger(rank=rank, fresh=not mlperf_resume)
        if mode == "streaming-train-eval"
        else None
    )
    # INIT_START fires before setup on a cold start only (resume continues the
    # already-open run, whose markers were emitted by the original process).
    mlperf_cold_start = mlperf_logger is not None and not mlperf_resume
    if mlperf_cold_start:
        mlperf_logger.event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)
        mlperf_logger.start(key=mlperf_logger.constants.INIT_START)

    # Phase 2: heavy imports. Triton kernel modules evaluate their autotune
    # decorators here, using the env vars set above.
    from generative_recommenders.dlrm_v4.checkpoint import load_dmp_checkpoint
    from generative_recommenders.dlrm_v4.train.utils import (
        cleanup,
        decorrelate_runtime_rng,
        eval_loop,
        make_model,
        make_optimizer_and_shard,
        make_train_test_dataloaders,
        seed_everything,
        setup,
        streaming_train_eval_loop,
        train_eval_loop,
        train_loop,
    )
    from generative_recommenders.dlrm_v4.utils import (
        MetricsLogger,
        get_gpu_peak_flops,
    )

    setup(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        device=device,
    )
    # Phase 3: re-parse to bind the @gin.configurables now that they are
    # registered. The earlier skip_unknown pass already consumed the
    # env-bootstrap binding, but bindings are idempotent so re-applying is
    # fine, and this pass is the one that actually wires up make_model,
    # make_train_test_dataloaders, etc.
    gin.parse_config_file(gin_file)

    # Seed all RNGs (gin-configurable $SEED) BEFORE make_model() so weight init
    # is reproducible run-to-run. Must follow the full parse above so the binding
    # is wired, and precede make_model() below.
    seed_everything(rank=rank)

    model, model_configs, embedding_table_configs = make_model()
    model, optimizer = make_optimizer_and_shard(
        model=model,
        device=device,
        world_size=world_size,
        local_world_size=gpus_per_node,
        embedding_table_configs=embedding_table_configs,
    )
    # Decorrelate forward-time stochasticity (HSTU dropout) per data-parallel
    # rank. MUST run after make_model() + make_optimizer_and_shard() so the
    # replicated dense weights and sharded embeddings stay init-identical across
    # ranks; this only offsets the global RNG by rank so dropout masks differ.
    decorrelate_runtime_rng(rank=rank)
    train_dataloader, test_dataloader = make_train_test_dataloaders(
        hstu_config=model_configs,
        embedding_table_configs=embedding_table_configs,
    )
    # TFLOPS/MFU reporting: query the model's static dense estimate +
    # current GPU's peak FLOPS. Both default to 0 if the model doesn't
    # expose get_num_flops_per_sample, in which case MetricsLogger silently
    # drops the tflops fields from the perf line.
    inner_model = model.module if hasattr(model, "module") else model
    if os.environ.get("DEBUG_NAN_HOOKS", "0") != "0":
        logger.warning("[nan-hook] installing non-finite output hooks (slow)")
        _install_nonfinite_hooks(inner_model)
        _install_nonfinite_backward_hooks(inner_model)
    num_flops_per_sample = (
        float(inner_model.get_num_flops_per_sample())
        if hasattr(inner_model, "get_num_flops_per_sample")
        else 0.0
    )
    gpu_peak_flops = get_gpu_peak_flops(
        "bf16" if getattr(model_configs, "bf16_training", True) else "fp32"
    )
    # Streaming fixed-holdout eval uses the dual fresh/cumulative metric sets:
    # window_* = fresh per-pass full-holdout, lifetime_* = cumulative across
    # passes (AUC via O(bins) histogram). Other modes keep the legacy single set.
    metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=train_dataloader.batch_size,
        window_size=2500,
        device=device,
        rank=rank,
        # Pass the live world_size so metric normalization is correct at any
        # node count; the gin's MetricsLogger.world_size default (=8) is only a
        # single-node fallback and would mis-normalize a multi-node run.
        world_size=world_size,
        num_flops_per_sample=num_flops_per_sample,
        gpu_peak_flops=gpu_peak_flops,
        model=model,
        eval_cumulative=(mode == "streaming-train-eval"),
        # Lifetime-AUC backend + bins/window come from gin (see yambda_5b.gin:
        # MetricsLogger.{train,eval}_lifetime_auc_mode / cumulative_auc_bins /
        # lifetime_auc_window), env-overridable. eval_cumulative stays explicit
        # because it is runtime-mode dependent, not a config knob.
    )
    # Capture streaming resume hint (None for cold start / non-streaming
    # checkpoints). For the streaming-train-eval mode, we forward this into
    # streaming_train_eval_loop so it can advance past the last completed
    # window OR re-enter the partial window and skip already-trained batches.
    resume_train_ts, resume_batch_idx_in_window, resume_split_contract, resume_cold_start = (
        load_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metrics,
            device=device,
            rank=rank,
        )
    )

    # MLPerf run markers: open the run exactly once. On a cold start emit
    # submission info + hyperparameters + INIT_STOP/RUN_START and mark the run as
    # started (persisted in the checkpoint via metrics.mlperf_run_started). On a
    # resume, load_dmp_checkpoint restored mlperf_run_started=True, so we skip the
    # markers and just continue the stream. `metrics.mlperf_run_started` guards a
    # double-emit even if cold/resume detection and the checkpoint ever disagree.
    if mlperf_cold_start and not metrics.mlperf_run_started:
        # Submission info + hyperparameters + INIT_STOP/RUN_START, all emitted by
        # the logger (optimizer names/LRs read from gin internally). Seed is the
        # value setup() resolved and exported to $SEED.
        mlperf_logger.log_run_start(
            global_batch_size=world_size * int(train_dataloader.batch_size),
            seed=int(os.environ.get("SEED", "1")),
        )
        metrics.mlperf_run_started = True
    # Pass the logger to the loop whenever MLPerf logging is enabled, so block /
    # eval / train_loss / run_stop events emit on BOTH a cold start and a resume.
    mlperf_run_active = mlperf_logger is not None and metrics.mlperf_run_started

    # train loop
    try:
        if mode == "train":
            train_loop(
                rank=rank,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "eval":
            # reinit metrics logger for eval
            metrics = MetricsLogger(
                multitask_configs=model_configs.multitask_configs,
                batch_size=train_dataloader.batch_size,
                window_size=1000,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            eval_loop(
                rank=rank,
                model=model,
                dataloader=test_dataloader,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "train-eval":
            train_eval_loop(
                rank=rank,
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=test_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "streaming-train-eval":
            streaming_train_eval_loop(
                rank=rank,
                model=model,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
                hstu_config=model_configs,
                embedding_table_configs=embedding_table_configs,
                resume_train_ts=resume_train_ts,
                resume_batch_idx_in_window=resume_batch_idx_in_window,
                resume_split_contract=resume_split_contract,
                resume_cold_start=resume_cold_start,
                # Only pass the logger when run boundaries were emitted, so the
                # loop never produces orphan block/eval events.
                mlperf_logger=mlperf_logger if mlperf_run_active else None,
            )
    except Exception as e:
        logger.info(traceback.format_exc())
        raise Exception(e)
    finally:
        # Graceful distributed teardown on both success and failure: barrier so
        # all ranks finish in lockstep, then destroy the process group (best-
        # effort) to avoid noisy TCPStore/NCCL shutdown warnings at exit.
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                logger.info("teardown barrier failed (non-fatal)")
            try:
                cleanup()
            except Exception:
                logger.info("teardown destroy_process_group failed (non-fatal)")


def get_args():  # pyre-ignore [3]
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_CONFIGS.keys(), help="dataset"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="mode",
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


def main() -> None:
    args = get_args()
    logger.info(args)
    assert args.dataset in SUPPORTED_CONFIGS, f"Unsupported dataset: {args.dataset}"
    assert args.mode in [
        "train",
        "eval",
        "train-eval",
        "streaming-train-eval",
    ], f"Unsupported mode: {args.mode}"
    # Distributed topology (single-node defaults reproduce the legacy behavior):
    #   GPUS_PER_NODE  local procs to spawn on THIS node (default: all visible GPUs)
    #   NNODES/NODE_RANK  multi-node fan-out, set by the SLURM launcher
    #   WORLD_SIZE     global rank count = NNODES * GPUS_PER_NODE
    #   MASTER_ADDR/PORT  rank-0 rendezvous; the port MUST match across nodes, so
    #                     honor it from the env when set and only fall back to a
    #                     random free port for the standalone single-node path.
    GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", 0)) or torch.cuda.device_count()
    NNODES = int(os.environ.get("NNODES", 1))
    NODE_RANK = int(os.environ.get("NODE_RANK", 0))
    WORLD_SIZE = NNODES * GPUS_PER_NODE
    MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
    MASTER_PORT = str(os.environ.get("MASTER_PORT") or get_free_port())
    gin_path = f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[args.dataset]}"
    logger.info(
        f"launching: nnodes={NNODES} node_rank={NODE_RANK} "
        f"gpus_per_node={GPUS_PER_NODE} world_size={WORLD_SIZE} "
        f"master={MASTER_ADDR}:{MASTER_PORT}"
    )

    mp.start_processes(
        _main_func,
        args=(
            WORLD_SIZE,
            NODE_RANK,
            GPUS_PER_NODE,
            MASTER_ADDR,
            MASTER_PORT,
            gin_path,
            args.mode,
        ),
        nprocs=GPUS_PER_NODE,
        join=True,
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
