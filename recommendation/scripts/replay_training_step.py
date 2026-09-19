#!/usr/bin/env python3
"""Replay a trusted full-state NaN capture without a dataset or dataloaders."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import time


def differences(a, b, path="", limit=30):
    """Small structured diff, including byte equality for tensors/RNG state."""
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return [path + ": tensor shape/dtype differs"]
        aa = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        bb = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return [] if torch.equal(aa, bb) else [path + ": tensor bytes differ"]
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return [] if a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes() else [path + ": array differs"]
    if type(a) is not type(b):
        return [path + f": type differs {type(a).__name__}/{type(b).__name__}"]
    if isinstance(a, dict):
        result = []
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                result.append(f"{path}/{k}: missing key")
            else:
                result.extend(differences(a[k], b[k], f"{path}/{k}", limit))
            if len(result) >= limit:
                break
        return result[:limit]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + ": sequence length differs"]
        result = []
        for i, (x, y) in enumerate(zip(a, b)):
            result.extend(differences(x, y, f"{path}/{i}", limit))
            if len(result) >= limit:
                break
        return result[:limit]
    return [] if a == b else [path + f": {a!r} != {b!r}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--mode", choices=("current", "previous", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--verify-every-repeat", action="store_true",
                        help="Compare all restored state bytes and controls before every attempt")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--backward-probe-dir", type=Path,
                        help="Capture first extreme/nonfinite GEMM or layer-norm backward boundary")
    parser.add_argument("--allow-source-change", action="store_true", help="For subsequent kernel bisection; record differences in report")
    args = parser.parse_args()
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not (args.capture / "COMPLETE.json").is_file():
        parser.error("capture is incomplete (missing COMPLETE.json)")
    context = json.loads((args.capture / "context.json").read_text())
    if context["world_size"] != 1:
        parser.error("only single-rank replay is implemented")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    probe_environment = {key: value for key, value in os.environ.items()
                         if key.startswith("NAN_BACKWARD_")}
    for key, value in context["environment"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    # Diagnostic choices belong to this replay, rather than the original run.
    os.environ.update(probe_environment)
    # This entry point invokes the model directly and never the training hook.
    os.environ.pop("NAN_REPLAY_DIR", None)
    os.environ.pop("NAN_REPLAY_FORCE_STEP", None)
    changed = []
    for name, expected in context["source_sha256"].items():
        if not name.startswith("generative_recommenders/"):
            continue
        path = root / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    if changed and not args.allow_source_change:
        parser.error(f"model source differs: {changed}; use --allow-source-change only for intentional bisection")

    # Environment must be restored before Triton decorator imports.
    import gin
    import torch
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap

    gin.parse_config(context["gin"], skip_unknown=True)
    apply_env_bootstrap()
    from generative_recommenders.dlrm_v4.train.utils import (
        setup, cleanup, seed_everything, make_model, make_optimizer_and_shard,
    )
    from generative_recommenders.dlrm_v4.utils import MetricsLogger  # noqa: F401 (gin registration)
    import triton
    from nan_replay_state import (
        collect_state, restore_controls, restore_step_policy,
        capture_rng, restore_rng, unpack_kjt, _walk_modules,
    )
    from nan_replay_storage import restore_snapshot, compare_snapshot, describe_tensor_state, load_snapshot_manifest
    from nan_replay_capture import (
        observe_forward, nonfinite_names, write_json, partition_state,
        restore_small_state, compare_small_state,
    )
    from nan_backward_boundaries import BoundaryProbeError, install as install_backward_probe

    versions = {"torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__}
    if any(versions[k] != context[k] for k in versions):
        parser.error(f"stack version mismatch: live {versions}")
    if context.get("triton_distribution", importlib.metadata.version("triton")) != importlib.metadata.version("triton"):
        parser.error("Triton distribution/commit mismatch")
    torch.set_num_threads(context.get("cpu_threads", torch.get_num_threads()))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    device = torch.device("cuda:0")
    setup(rank=0, world_size=1, master_addr="localhost", master_port=port, device=device)
    gin.parse_config(context["gin"])
    seed_everything(rank=0)
    model, _, table_configs = make_model()
    model, optimizer = make_optimizer_and_shard(
        model=model, device=device, world_size=1, local_world_size=1,
        embedding_table_configs=table_configs,
    )
    payload = torch.load(args.capture / "capture.pt", map_location="cpu", weights_only=False)
    inputs = {}
    report = {
        "capture": str(args.capture), "stage": payload["stage"],
        "versions": versions, "changed_source": changed, "attempts": [],
        "backward_probe_dir": str(args.backward_probe_dir) if args.backward_probe_dir else None,
        "replay_source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
            if path.parts[len(root.parts)] in ("generative_recommenders", "scripts")
        },
        "effective_environment": {
            key: os.environ.get(key) for key in (
                "HSTU_BWD_MAX_VGPR", "HSTU_BWD_BLOCK_N", "TRITON_FULL_AUTOTUNE",
                "AMDGCN_USE_BUFFER_OPS", "TRITON_ALLOW_PIPELINING",
                "AMD_SERIALIZE_KERNEL", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "attention_backward_configs": [
            {"kwargs": dict(config.kwargs), "num_warps": config.num_warps,
             "num_stages": config.num_stages,
             "pre_hook": getattr(config.pre_hook, "__name__", None)}
            for config in getattr(getattr(sys.modules.get(
                "generative_recommenders.ops.triton.triton_hstu_attention"
            ), "_hstu_attn_bwd", None), "configs", [])
        ],
    }
    report_path = args.report or args.capture / "replay_report.json"
    probe = None

    def phase(name):
        attempt["phase"] = name
        report["in_progress"] = attempt
        write_json(report_path, report)
        print(f"[replay] {mode} repeat={repeat}: {name}", flush=True)

    def describe_large_differences(refs, comparison):
        topology, views = describe_tensor_state(refs)
        manifest = load_snapshot_manifest(args.capture / "state")
        details = []
        for hit in comparison["first_mismatches"]:
            sid, byte = hit["storage_id"], hit["byte_offset"]
            bindings = [b for b in topology["bindings"] if b["storage_id"] == sid]
            if {b["dtype"] for b in bindings} != {"torch.float32"}:
                continue
            offset = byte // 4 * 4
            actual = bytes(views[sid][offset:offset + 4].cpu().tolist())
            with (args.capture / "state" / manifest["base_files"][sid]).open("rb") as f:
                f.seek(offset)
                expected = f.read(4)
            av, ev = struct.unpack("<f", actual)[0], struct.unpack("<f", expected)[0]
            ai, ei = struct.unpack("<I", actual)[0], struct.unpack("<I", expected)[0]
            details.append({"storage_id": sid, "byte_offset": offset,
                            "expected": repr(ev), "actual": repr(av),
                            "bit_distance_same_sign": abs(ai - ei) if (ai >> 31) == (ei >> 31) else None})
        return details

    def forward(which):
        phase(f"{which}_forward")
        if probe is not None:
            probe.set_attempt(mode, repeat, payload[which]["step"])
        batch = inputs[which]
        return model.forward(batch["uih"], batch["candidates"])

    def rebuild_inputs(which):
        # Bounds checking or the defect itself can mutate input storage. Each
        # attempt must start from fresh captured bytes, including KJT metadata.
        inputs[which] = {key: unpack_kjt(value, device)
                         for key, value in payload[which]["inputs"].items()}

    def prepare_lazy_layout(frame):
        # DlrmHSTU.preprocess feeds UIH keys followed by candidate keys into
        # its ShardedEmbeddingCollection. TorchRec creates input-dist metadata
        # on the first forward. Recreate that metadata without executing a
        # training forward or touching weights; the raw buffers are validated
        # and restored with all other state immediately afterwards.
        keys = frame["inputs"]["uih"]["keys"] + frame["inputs"]["candidates"]["keys"]
        for path, module in _walk_modules(model).items():
            if type(module).__name__ != "ShardedEmbeddingCollection":
                continue
            saved = frame["controls"]["modules"][path]
            initialized = "_features_order_tensor" in saved["buffers"]
            if initialized and module._has_uninitialized_input_dist:
                module._create_input_dist(input_feature_names=keys, ctx=module.create_context())
                module._has_uninitialized_input_dist = False
            elif not initialized and not module._has_uninitialized_input_dist:
                module._input_dists.clear()
                module._features_order = []
                module._feature_splits = []
                for name in list(module._buffers):
                    if name == "_features_order_tensor" or name.startswith(("_hash_size_cumsum_tensor_", "_hash_size_offset_tensor_")):
                        del module._buffers[name]
                        module._non_persistent_buffers_set.discard(name)
                module._has_uninitialized_input_dist = True

    def advance(output, frame):
        phase("previous_backward")
        sum(output[2].values()).backward()
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and nonfinite_names(p.grad)]
        if frame["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=frame["grad_clip_norm"])
        optimizer.step()
        return bad

    attempt = {"phase": "probe_setup", "step": payload["current"]["step"]}
    start = time.monotonic()
    try:
        if args.backward_probe_dir:
            probe = install_backward_probe(model, args.backward_probe_dir)
            report["backward_probe_config"] = {
                "target": probe.target, "abs_threshold": probe.abs_threshold,
                "save_all": probe.save_all,
                "chunk_bytes": probe.chunk_bytes, "max_capture_bytes": probe.max_bytes,
            }
        modes = ("current", "previous") if args.mode == "both" else (args.mode,)
        for mode in modes:
            if payload[mode] is None:
                if args.mode == "both":
                    report["previous_unavailable"] = True
                    write_json(report_path, report)
                    continue
                raise RuntimeError("capture has no previous boundary")
            for repeat in range(args.repeats):
                start = time.monotonic()
                attempt = {"mode": mode, "repeat": repeat, "step": payload["current"]["step"]}
                phase("restore")
                frame = payload[mode]
                prepare_lazy_layout(frame)
                restore_controls(model, optimizer, frame["controls"])
                refs, _ = collect_state(model, optimizer)
                large, small = partition_state(refs)
                restore_small_state(small, frame["small_state"])
                restore_snapshot(args.capture / "state", large, boundary=mode)
                torch.cuda.synchronize()
                rebuild_inputs(mode)
                if repeat == 0 or args.verify_every_repeat:
                    phase("verify_restored_bytes")
                    attempt["restore_large_comparison"] = compare_snapshot(args.capture / "state", large, boundary=mode)
                    attempt["restore_small_comparison"] = compare_small_state(small, frame["small_state"])
                    _, restored_controls = collect_state(model, optimizer)
                    attempt["restored_control_differences"] = differences(frame["controls"], restored_controls)
                    if not (attempt["restore_large_comparison"]["equal"] and attempt["restore_small_comparison"]["equal"]):
                        report["attempts"].append(attempt)
                        write_json(report_path, report)
                        raise RuntimeError("Restoration byte verification failed; replay is invalid")
                    if attempt["restored_control_differences"]:
                        raise RuntimeError("Restored controls differ; replay is invalid")
                restore_rng(frame["rng"])
                attempt["restored_rng_differences"] = differences(frame["rng"], capture_rng())
                if attempt["restored_rng_differences"]:
                    raise RuntimeError("Restored RNG differs; replay is invalid")
                if mode == "previous":
                    prev_output = forward("previous")
                    prev_observed = observe_forward(prev_output)
                    attempt["previous_output_differences"] = differences(payload["previous"]["observed"], prev_observed)
                    attempt["previous_nonfinite"] = nonfinite_names(prev_observed)
                    attempt["previous_bad_gradients"] = advance(prev_output, frame)
                    del prev_output
                    inputs.pop("previous")
                    rebuild_inputs("current")
                    optimizer.zero_grad()
                    restore_step_policy(model, optimizer, payload["current"]["controls"])
                    refs, controls = collect_state(model, optimizer)
                    large, small = partition_state(refs)
                    attempt["boundary_control_differences"] = differences(payload["current"]["controls"], controls)
                    attempt["boundary_rng_differences"] = differences(payload["current"]["rng"], capture_rng())
                    phase("compare_previous_transition")
                    attempt["boundary_state_comparison"] = compare_snapshot(args.capture / "state", large)
                    attempt["boundary_small_state_comparison"] = compare_small_state(small, payload["current"]["small_state"])
                    attempt["boundary_large_difference_details"] = describe_large_differences(large, attempt["boundary_state_comparison"])
                    # Preserve the state and RNG produced by the previous step.
                output = forward("current")
                phase("observe_current_forward")
                observed = observe_forward(output)
                attempt["output_differences"] = (
                    differences(payload["current"]["observed"], observed)
                    if "observed" in payload["current"] else None
                )
                attempt["nonfinite"] = nonfinite_names(observed)
                attempt["losses"] = {k: str(v.tolist()) for k, v in observed["losses"].items()}
                if payload["stage"] == "backward_before_clip":
                    phase("current_backward")
                    sum(output[2].values()).backward()
                    phase("scan_current_gradients")
                    attempt["bad_gradients"] = [name for name, p in model.named_parameters()
                                                if p.grad is not None and nonfinite_names(p.grad)]
                    if probe is not None:
                        # Record finite growth too: a finite gradient scan alone
                        # would miss the huge values preceding the first NaN.
                        # Reuse the bounded CPU diagnostic and avoid full-size
                        # abs/isfinite temporaries on the GPU.
                        from nan_backward_boundaries import _cpu_copy_tree, _source_specs, _summaries
                        profile = []
                        for name, parameter in model.named_parameters():
                            if parameter.grad is None:
                                continue
                            gradient = {name: parameter.grad}
                            copied, _ = _cpu_copy_tree(gradient, probe.chunk_bytes, probe.max_bytes)
                            profile.extend(_summaries(copied, probe.chunk_bytes,
                                                      _source_specs(gradient), probe.abs_threshold))
                            del copied
                        attempt["gradient_profile"] = profile
                del output
                torch.cuda.synchronize()
                attempt["seconds"] = time.monotonic() - start
                attempt["phase"] = "complete"
                report.pop("in_progress", None)
                report["attempts"].append(attempt)
                write_json(report_path, report)
                print("[replay] " + json.dumps(attempt, allow_nan=False), flush=True)
    except BoundaryProbeError as error:
        stopped_phase = attempt["phase"]
        attempt["phase"] = "boundary_probe_stopped"
        attempt["seconds"] = time.monotonic() - start
        dump_path = getattr(error, "dump_path", None)
        attempt["boundary_probe"] = {
            "status": "captured" if dump_path else "failed",
            "dump": str(dump_path) if dump_path else None,
            "error": str(error),
            "phase": stopped_phase,
            "attempt": dict(probe.attempt) if probe is not None and probe.attempt is not None else None,
        }
        report.pop("in_progress", None)
        report["attempts"].append(attempt)
        write_json(report_path, report)
        print("[replay] " + json.dumps(attempt, allow_nan=False), flush=True)
        raise SystemExit(86 if dump_path else 1) from error
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "phase": attempt["phase"]}
        write_json(report_path, report)
        raise
    finally:
        if probe is not None:
            probe.close()
        cleanup()
    print(f"[replay] COMPLETE report={report_path}", flush=True)


if __name__ == "__main__":
    main()
