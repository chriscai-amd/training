#!/usr/bin/env python3
"""Stress attention backward using the real step-55 tripwire inputs.

Reconstruct normalization and UVQK once, then reuse one input/output allocation
for attention-only calls. Every history dQ must be exactly zero, some target dQ
must be nonzero, and all dQ/dK/dV must be finite. This oracle does not validate
the nonzero gradients numerically.
Only scalar check results survive each iteration; host readback happens every
--check-every calls. Checks themselves change timing and memory traffic.
With --failure-dump, --check-every must be 1. This optional mode snapshots full
input backing storages on CPU before the loop and saves the exact failing
iteration's outputs/current inputs without another attention launch. The extra
pre-loop synchronization/copy changes timing; ordinary stress is unchanged.

Examples inside the matching training image:
  HSTU_BWD_MAX_VGPR=256 python scripts/repro_hstu_attention_capture.py capture.pt --repeat 25 --report capped.json
  HSTU_BWD_MAX_VGPR=0 python scripts/repro_hstu_attention_capture.py capture.pt --repeat 25
  python scripts/repro_hstu_attention_capture.py capture.pt --batch-size 32 --repeat 10
  python scripts/repro_hstu_attention_capture.py capture.pt --batch-size 32 --export-prefix prefix32.pt
  python scripts/repro_hstu_attention_capture.py prefix32.pt --repeat 25

A prefix crop reduces uploaded storages and reconstruction work, but retains
the capture's N=2768 normalization. HSTU_BWD_MAX_VGPR is consumed by the normal
production config. An explicit value overrides the capture environment; if
neither specifies it, the shared default is uncapped (0). Set 256 explicitly
for the B0 workaround arm and 0 explicitly for its uncapped control.
--export-prefix writes a compact CPU artifact and exits without initializing
CUDA. Its explicit attention-prefix format is accepted by this runner only;
it contains no original backward outputs and cannot validate a full replay.
Only load trusted local tripwire captures (torch.load uses Python pickle).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


PREFIX_FORMAT = "hstu_attention_prefix"
PREFIX_VERSION = "hstu_attention_prefix_v1"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--batch-size", type=int, help="use the first B captured sequences")
    parser.add_argument("--export-prefix", type=Path, help="export the selected inputs to a new CPU file, then exit")
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--check-input-contents", action="store_true",
                        help="clone Q/K/V/dout once and compare their bytes each call")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--failure-dump", type=Path,
                        help="save pristine inputs and exact failing outputs; requires --check-every 1")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.check_every < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("repeat, check-every and batch-size must be positive")
    if args.failure_dump:
        if args.check_every != 1:
            parser.error("--failure-dump requires --check-every 1 so the current outputs belong to the failing call")
        if args.export_prefix:
            parser.error("--failure-dump cannot be combined with --export-prefix")
        if args.failure_dump.exists() or args.failure_dump.resolve() == args.capture.resolve():
            parser.error("--failure-dump must name a new file distinct from the input capture")
        if args.report and args.failure_dump.resolve() == args.report.resolve():
            parser.error("--failure-dump and --report must name different files")
    return args


def cpu_view(torch, capture, desc):
    return torch.empty(0, dtype=getattr(torch, desc["dtype"].removeprefix("torch."))).set_(
        capture["storages"][desc["__tripwire_tensor__"]].untyped_storage(),
        desc["storage_offset"], tuple(desc["shape"]), tuple(desc["stride"]),
    )


def select_prefix(torch, capture, batch_size):
    """Crop descriptors and raw byte extents before decode_payload uploads them."""
    replay = copy.deepcopy(capture["payload"]["replay"])
    if replay["class"] != "_HSTUPreprocessAndAttentionFunction" or replay["direction"] != "backward":
        raise ValueError("requires a preprocess-and-attention backward capture")
    ctx = replay["ctx"]
    required = ("recompute_normed_x_in_backward", "recompute_uvqk_in_backward", "has_multiple_targets", "sort_by_length")
    if not all(ctx.get(name) for name in required) or ctx["num_softmax_heads"]:
        raise ValueError("requires the step-55 SiLU capture with targets, sorting and both recomputations")
    saved = replay["saved_tensors"]
    if len(saved) != 10 or len(replay["args"]) != 2 or replay["kwargs"]:
        raise ValueError("unexpected captured argument layout")
    offsets = cpu_view(torch, capture, saved[6])
    original_batch = offsets.numel() - 1
    batch = original_batch if batch_size is None else batch_size
    if not 1 <= batch <= original_batch:
        raise ValueError(f"batch-size must be in [1, {original_batch}]")
    lengths = offsets[1:] - offsets[:-1]
    if int(offsets[0]) != 0 or not bool((lengths > 0).all()):
        raise ValueError("invalid captured sequence offsets")
    if int(offsets[-1]) != saved[0]["shape"][0] or int(lengths.max()) > ctx["max_seq_len"]:
        raise ValueError("captured row count or maximum length is inconsistent")
    if not bool((cpu_view(torch, capture, saved[7]) == 1).all()):
        raise ValueError("the history-zero oracle requires exactly one target per sequence")
    original_sort = cpu_view(torch, capture, saved[9])
    if not torch.equal(torch.sort(original_sort).values, torch.arange(original_batch)):
        raise ValueError("captured sort indices are not a sequence permutation")
    rows = int(offsets[batch])
    for desc in (saved[0], saved[3], saved[4], replay["args"][1]):
        desc["shape"] = [rows, *desc["shape"][1:]]
    saved[6]["shape"] = [batch + 1]
    saved[7]["shape"] = saved[9]["shape"] = [batch]
    # Attention consumes only dout. Keep its positional index without uploading
    # the separate SiLU-branch gradient (2.6 GB in the full step-55 capture).
    replay["args"] = (None, replay["args"][1])

    # Decoding a small view of a large original storage would otherwise upload
    # the entire storage. Preserve aliases while trimming unused suffix bytes.
    extents = {}

    def visit(value):
        if isinstance(value, dict) and "__tripwire_tensor__" in value:
            index = value["__tripwire_tensor__"]
            dtype = getattr(torch, value["dtype"].removeprefix("torch."))
            end = value["storage_offset"] + 1
            if any(size == 0 for size in value["shape"]):
                end = value["storage_offset"]
            else:
                end += sum((size - 1) * stride for size, stride in zip(value["shape"], value["stride"]))
            extents[index] = max(extents.get(index, 0), end * torch.empty((), dtype=dtype).element_size())
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(replay)
    selected = {"payload": {"replay": replay}, "storages": {
        index: capture["storages"][index][:end] for index, end in extents.items()
    }}
    return selected, {"original_batch_size": original_batch, "batch_size": batch,
                      "rows": rows, "length_min": int(lengths[:batch].min()),
                      "length_max": int(lengths[:batch].max()),
                      "normalization_N": ctx["max_seq_len"],
                      "uploaded_storage_bytes": sum(extents.values())}


def export_prefix(torch, selected, dimensions, source, destination, environment, provenance):
    """Save only the selected CPU inputs; clone byte slices to trim storage."""
    if source.resolve() == destination.resolve():
        raise ValueError("export destination must differ from the source capture")
    replay = selected["payload"]["replay"]
    keys = ("module", "class", "direction", "ctx", "kwargs", "saved_tensors", "args",
            "cuda_autocast_enabled", "cuda_autocast_dtype", "runtime_state")
    replay = copy.deepcopy({key: replay[key] for key in keys if key in replay})
    descriptors = [*replay["saved_tensors"], replay["args"][1]]
    ids = {desc["__tripwire_tensor__"] for desc in descriptors}
    # Saving a narrow view alone would serialize its full original storage.
    storages = {index: selected["storages"][index].clone() for index in ids}
    if dimensions["batch_size"] != dimensions["original_batch_size"]:
        offsets = cpu_view(torch, selected, replay["saved_tensors"][6])
        sort = torch.argsort(offsets[1:] - offsets[:-1], descending=True, stable=False)
        sort_desc = replay["saved_tensors"][9]
        storages[sort_desc["__tripwire_tensor__"]] = sort.view(torch.uint8).clone()
        sort_desc.update(storage_offset=0, stride=[1])
    keep = {"__tripwire_tensor__", "shape", "stride", "dtype", "device", "storage_offset"}
    for desc in descriptors:
        for key in list(desc):
            if key not in keep:
                del desc[key]
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    source_hash = digest.hexdigest()
    provenance = dict(provenance)
    provenance.setdefault("source_capture_name", source.name)
    provenance.setdefault("source_capture_sha256", source_hash)
    provenance.setdefault("source_batch_size", dimensions["original_batch_size"])
    provenance.update(exported_from_name=source.name, exported_from_sha256=source_hash,
                      prefix_batch_size=dimensions["batch_size"], prefix_rows=dimensions["rows"],
                      normalization_N=dimensions["normalization_N"],
                      sort_policy="recomputed on CPU for a crop; runtime recomputes on GPU for a source prefix")
    artifact = dict(format=PREFIX_FORMAT, format_version=PREFIX_VERSION,
                    payload={"replay": replay}, storages=storages,
                    environment=environment, provenance=provenance)
    with destination.open("xb") as handle:
        torch.save(artifact, handle)
    return dict(path=str(destination), bytes=destination.stat().st_size,
                format=PREFIX_FORMAT, format_version=PREFIX_VERSION, provenance=provenance,
                cuda_initialized=torch.cuda.is_initialized())


def reconstruct(torch, replay, cropped):
    from generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention import (
        maybe_triton_addmm_fwd, triton_weighted_layer_norm_fwd,
    )
    ctx = SimpleNamespace(**replay["ctx"])
    x, gamma, beta, mean, rstd, weight, offsets, targets, bias, sort = replay["saved_tensors"]
    with torch.autocast("cuda", enabled=replay.get("cuda_autocast_enabled", False),
                        dtype=replay.get("cuda_autocast_dtype", torch.bfloat16)):
        normed, _, _ = triton_weighted_layer_norm_fwd(
            x=x, weight=gamma, bias=beta, eps=ctx.norm_eps, mean=mean, rstd=rstd,
        )
        uvqk = maybe_triton_addmm_fwd(x=normed, w=weight, y=bias)
    widths = [ctx.hidden_dim * ctx.num_heads] * 2 + [ctx.attn_dim * ctx.num_heads] * 2
    _, v, q, k = uvqk.split(widths, dim=1)
    q, k = (t.view(-1, ctx.num_heads, ctx.attn_dim) for t in (q, k))
    v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
    if cropped:
        _, sort = torch.sort(offsets[1:] - offsets[:-1], descending=True, stable=False)
    gradients = torch.empty_like(uvqk)
    _, dv, dq, dk = gradients.split(widths, dim=1)
    inputs = dict(q=q, k=k, v=v, dout=replay["args"][1], seq_offsets=offsets,
                  num_targets=targets, sort_by_length_indices=sort)
    outputs = dict(dq=dq.view_as(q), dk=dk.view_as(k), dv=dv.view_as(v))
    return ctx, inputs, outputs


def all_chunks(torch, tensor, predicate, chunk_rows=32768):
    return torch.stack([predicate(tensor[start:start + chunk_rows], start)
                        for start in range(0, tensor.shape[0], chunk_rows)]).all()


def history_zero(torch, tensor, history):
    return all_chunks(torch, tensor, lambda chunk, start:
                      ~((chunk != 0).flatten(1).any(1) & history[start:start + len(chunk)]).any())


def compiled_metadata(kernel):
    """Read compiled objects already cached by the normal launch; no warmup call."""
    while not hasattr(kernel, "device_caches") and hasattr(kernel, "fn"):
        kernel = kernel.fn
    records = []
    for device, entry in getattr(kernel, "device_caches", {}).items():
        for compiled in entry[0].values():
            metadata = getattr(compiled, "metadata", None)
            if metadata is not None:
                records.append(dict(device=device, name=compiled.name, hash=compiled.hash,
                                    n_regs=getattr(compiled, "n_regs", None),
                                    n_spills=getattr(compiled, "n_spills", None),
                                    metadata=metadata._asdict()))
    return records


FAILURE_COPY_CHUNK_BYTES = 64 << 20
FAILURE_MAX_BYTES = 64 << 30


def snapshot_failure_inputs(inputs, *, chunk_bytes=FAILURE_COPY_CHUNK_BYTES,
                            max_bytes=FAILURE_MAX_BYTES):
    """Optional pre-loop CPU snapshot preserving complete storage and aliases."""
    from nan_backward_boundaries import _cpu_copy_tree, _source_specs

    copied, size = _cpu_copy_tree(inputs, chunk_bytes, max_bytes)
    return {"inputs": copied, "storage_bytes": size, "source_specs": _source_specs(inputs)}


def save_failure_dump(path, pristine, inputs, outputs, *, iteration, failures,
                      configuration, compiled_kernels, report_path=None,
                      chunk_bytes=FAILURE_COPY_CHUNK_BYTES, max_bytes=FAILURE_MAX_BYTES):
    """Persist the existing failing buffers, never replaying/reconstructing them."""
    import torch
    from nan_backward_boundaries import _cpu_copy_tree, _source_specs, _summaries
    from replay_backward_boundary import compare_tensors, _compare_storages

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Failure dump already exists: {path}")
    if iteration < 1 or not failures or any(item["iteration"] != iteration for item in failures):
        raise ValueError("Failure records must all describe the current positive iteration")
    # First save the original current buffers. All comparisons below are CPU-only.
    current, size = _cpu_copy_tree({"inputs": inputs, "outputs": outputs}, chunk_bytes,
                                   max_bytes - pristine["storage_bytes"])
    logical, raw, storage_pairs = {}, {}, {}
    for name, before in pristine["inputs"].items():
        after = current["inputs"][name]
        logical[name] = compare_tensors(before, after, chunk_bytes=chunk_bytes,
                                        axis0="tensor index (Q/K/V/dOut axis 0 is query row)")
        pair = (before.untyped_storage()._cdata, after.untyped_storage()._cdata)
        if pair not in storage_pairs:
            identifier = str(len(storage_pairs))
            storage_pairs[pair] = identifier
            raw[identifier] = _compare_storages(before, after, chunk_bytes)
        logical[name]["storage_pair"] = storage_pairs[pair]
    comparisons = {
        "all_logical_input_bytes_unchanged": all(item["equal_logical_bytes"] and item["layout_equal"] for item in logical.values()),
        "all_input_backing_bytes_unchanged": all(item["equal"] for item in raw.values()),
        "logical_tensors": logical, "backing_storages": raw,
        "interpretation": "Backings include unused packed columns/padding; logical and backing changes are reported separately.",
    }
    summaries = _summaries(current["outputs"], chunk_bytes, _source_specs(outputs), 1e20)
    dq = current["outputs"]["dq"]
    offsets = pristine["inputs"]["seq_offsets"]
    targets = pristine["inputs"]["num_targets"]
    if (dq.ndim != 3 or offsets.ndim != 1 or int(offsets[0]) != 0
            or int(offsets[-1]) != dq.shape[0] or not bool((offsets[1:] > offsets[:-1]).all())
            or targets.shape != (offsets.numel() - 1,) or not bool((targets == 1).all())):
        raise ValueError("Failure summary requires valid offsets and one target per sequence")
    history = torch.ones(dq.shape[0], dtype=torch.bool)
    history[offsets[1:] - 1] = False
    zero_oracle = {"history_nonzero_elements": 0, "history_affected_rows": 0,
                   "history_max_abs_finite": 0.0, "sample_locations": []}
    rows_per_chunk = max(1, chunk_bytes // max(1, dq.shape[1] * dq.shape[2] * 32))
    for start in range(0, dq.shape[0], rows_per_chunk):
        tile = dq[start:start + rows_per_chunk]
        wrong = (tile != 0) & history[start:start + len(tile), None, None]
        zero_oracle["history_nonzero_elements"] += int(wrong.sum())
        zero_oracle["history_affected_rows"] += int(wrong.flatten(1).any(1).sum())
        finite_wrong = wrong & torch.isfinite(tile)
        if bool(finite_wrong.any()):
            zero_oracle["history_max_abs_finite"] = max(zero_oracle["history_max_abs_finite"], float(tile[finite_wrong].abs().max()))
        if len(zero_oracle["sample_locations"]) < 32:
            # Limit nonzero materialization, even if an entire allocation is bad.
            for flat_start in range(0, wrong.numel(), 4096):
                local = torch.nonzero(wrong.reshape(-1)[flat_start:flat_start + 4096]).flatten()
                for item in local[:32 - len(zero_oracle["sample_locations"])].tolist():
                    flat = flat_start + item
                    feature = flat % dq.shape[2]
                    head = (flat // dq.shape[2]) % dq.shape[1]
                    row = start + flat // (dq.shape[1] * dq.shape[2])
                    sequence = int(torch.searchsorted(offsets[1:], torch.tensor(row, dtype=offsets.dtype), right=True))
                    zero_oracle["sample_locations"].append({
                        "row": row, "sequence": sequence, "sequence_position": row - int(offsets[sequence]),
                        "head": head, "feature": feature, "value": str(dq[row, head, feature].item())})
                if len(zero_oracle["sample_locations"]) == 32:
                    break
    artifact = {
        "format": "hstu_attention_backward_failure", "format_version": 1,
        "iteration": iteration, "failures": failures, "configuration": configuration,
        "compiled_kernels": compiled_kernels,
        "pristine_inputs": pristine["inputs"], "inputs_at_failure": current["inputs"],
        "outputs": current["outputs"], "input_comparison": comparisons,
        "output_summaries": summaries, "dq_zero_oracle": zero_oracle,
        "source_input_specs": pristine["source_specs"], "source_output_specs": _source_specs(outputs),
        "storage_bytes": pristine["storage_bytes"] + size,
        "report_path": str(report_path) if report_path else None,
        "capture_timing": "inputs snapshotted before stress loop; current inputs and original outputs copied immediately after this failing call's checks; no subsequent attention call",
        "limitation": "Optional pre-loop CPU snapshot and per-call checking change execution timing and memory traffic.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"path": str(path), "bytes": path.stat().st_size, "iteration": iteration,
            "storage_bytes": artifact["storage_bytes"], "input_comparison": comparisons,
            "output_summaries": summaries, "dq_zero_oracle": zero_oracle,
            "capture_timing": artifact["capture_timing"]}


def run(args, report, publish):
    failure_path = getattr(args, "failure_dump", None)
    if failure_path and args.check_every != 1:
        raise ValueError("--failure-dump requires --check-every 1")
    # Respect explicit controls, especially HSTU_BWD_MAX_VGPR=0, before import.
    for name, value in {"AMDGCN_USE_BUFFER_OPS": "0", "TRITON_FULL_AUTOTUNE": "0",
                        "TRITON_ALLOW_PIPELINING": "0", "PYTORCH_CUDA_ALLOC_CONF": "",
                        "PYTORCH_ALLOC_CONF": "", "HSA_ENABLE_COREDUMP": "0"}.items():
        os.environ.setdefault(name, value)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import triton
    from scripts.replay_nan_tripwire import changed_capture_inputs, decode_payload, _restore_runtime_state

    capture = torch.load(args.capture, map_location="cpu", mmap=True, weights_only=False)
    if capture.get("format") == PREFIX_FORMAT and capture.get("format_version") == PREFIX_VERSION:
        environment = capture["environment"]
        provenance = capture["provenance"]
    else:
        if capture.get("format_version") != 1 or changed_capture_inputs(capture):
            raise ValueError("unsupported capture or tracked mutation before capture")
        environment = capture["report"]["environment"]
        provenance = {"source_step": capture["report"].get("step"),
                      "source_metadata": capture["report"].get("metadata")}
    for name, value in environment.items():
        if not name.startswith("NAN_TRIPWIRE_"):
            os.environ.setdefault(name, value)
    selected, dimensions = select_prefix(torch, capture, args.batch_size)
    if args.export_prefix:
        report["export"] = export_prefix(torch, selected, dimensions, args.capture,
                                         args.export_prefix, environment, provenance)
        report["status"] = "EXPORTED"
        publish({"event": "export", **report["export"]})
        return 0
    torch.cuda.set_device(0)
    from generative_recommenders import common
    common.set_static_max_seq_lens([4096])  # Known training state for the older step-55 capture.
    common.set_use_runtime_max_seq_len(False)
    runtime_state = selected["payload"]["replay"].get("runtime_state")
    _restore_runtime_state(runtime_state)
    from generative_recommenders.ops.triton import triton_hstu_attention as attention
    configs = attention._hstu_attn_bwd.configs
    if len(configs) != 1 or configs[0].pre_hook is not attention._bwd_pre_hook:
        raise RuntimeError("requires one pinned config with the unchanged production prehook")
    config = configs[0]
    with torch.no_grad():
        replay = decode_payload(selected, "cuda", replay_only=True)["replay"]
        source_batch = provenance.get("source_batch_size", dimensions["original_batch_size"])
        ctx, inputs, outputs = reconstruct(torch, replay, dimensions["batch_size"] != source_batch)
        amp = {"enabled": replay.get("cuda_autocast_enabled", False),
               "dtype": str(replay.get("cuda_autocast_dtype"))}
        del replay, selected, capture
        offsets = inputs["seq_offsets"]
        history = torch.ones(dimensions["rows"], dtype=torch.bool, device="cuda")
        history[offsets[1:] - 1] = False
        finite = lambda t: all_chunks(torch, t, lambda chunk, _: torch.isfinite(chunk).all())
        setup = {name + "_finite": finite(inputs[name]) for name in ("q", "k", "v", "dout")}
        setup["dout_history_zero"] = history_zero(torch, inputs["dout"], history)
        setup["one_target_per_sequence"] = (inputs["num_targets"] == 1).all()
        setup["every_target_dout_nonzero"] = (inputs["dout"][offsets[1:] - 1] != 0).flatten(1).any(1).all()
        setup["sort_permutation"] = (torch.sort(inputs["sort_by_length_indices"]).values ==
                                     torch.arange(dimensions["batch_size"], device="cuda")).all()
        setup_values = torch.stack(list(setup.values())).cpu().tolist()
        if not all(setup_values):
            raise ValueError("invalid captured/reconstructed input: " + str(dict(zip(setup, setup_values))))
        versions = {name: tensor._version for name, tensor in inputs.items()}
        snapshots = {name: tensor.clone() for name, tensor in inputs.items()
                     if args.check_input_contents or name not in ("q", "k", "v", "dout")}
        properties = torch.cuda.get_device_properties(0)
        report["configuration"] = dict(
            **dimensions, torch=str(torch.__version__), hip=torch.version.hip, triton=triton.__version__,
            source_provenance=provenance,
            arch=getattr(properties, "gcnArchName", properties.name), source=attention.__file__,
            context=vars(ctx), autocast=amp, captured_runtime_state=runtime_state,
            static_max_seq_lens=common.STATIC_MAX_SEQ_LENS,
            use_runtime_max_seq_len=common.USE_RUNTIME_MAX_SEQ_LEN,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            config=config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages,
            pre_hook=config.pre_hook.__name__,
            input_layouts={name: {"shape": list(t.shape), "stride": list(t.stride()),
                                  "storage_offset": t.storage_offset(), "dtype": str(t.dtype)}
                           for name, t in inputs.items()},
            env={name: os.environ.get(name) for name in ("HSTU_BWD_MAX_VGPR", "HSTU_BWD_BLOCK_N",
                 "AMDGCN_USE_BUFFER_OPS", "TRITON_ALLOW_PIPELINING", "TRITON_FULL_AUTOTUNE",
                 "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "HSA_ENABLE_COREDUMP",
                 "AMD_SERIALIZE_KERNEL", "TRITON_HIP_USE_EXPERT_SCHEDULING",
                 "TRITON_HIP_USE_COEXEC_SCHEDULER", "AMDGCN_SCALARIZE_PACKED_FOPS",
                 "TRITON_HIP_USE_IN_THREAD_TRANSPOSE")},
        )
        pristine = None
        if failure_path:
            if Path(failure_path).exists():
                raise FileExistsError(f"Failure dump already exists: {failure_path}")
            from generative_recommenders.dlrm_v4.train.nan_tripwire import _capture_runtime_state
            source_hash = hashlib.sha256()
            with args.capture.open("rb") as stream:
                for chunk in iter(lambda: stream.read(16 << 20), b""):
                    source_hash.update(chunk)
            pristine = snapshot_failure_inputs(inputs)
            report["configuration"].update(
                input_artifact=str(args.capture.resolve()), input_artifact_sha256=source_hash.hexdigest(),
                effective_runtime_state=_capture_runtime_state(), setup_checks=dict(zip(setup, setup_values)),
                source_sha256={str(Path(source).resolve()): hashlib.sha256(Path(source).read_bytes()).hexdigest()
                               for source in (__file__, attention.__file__)},
                failure_capture={"path": str(failure_path), "pristine_input_storage_bytes": pristine["storage_bytes"],
                                 "check_every": 1, "chunk_bytes": FAILURE_COPY_CHUNK_BYTES,
                                 "max_artifact_storage_bytes": FAILURE_MAX_BYTES,
                                 "timing": "additional CPU input snapshot before loop; inputs and outputs copied only after a failed call"},
            )
        publish({"event": "configuration", **report["configuration"]})
        pending = []
        started = time.monotonic()
        for iteration in range(1, args.repeat + 1):
            attention.triton_hstu_attention_bwd(
                **inputs, **outputs, N=ctx.max_seq_len, alpha=ctx.attn_alpha,
                max_attn_len=ctx.max_attn_len, contextual_seq_len=ctx.contextual_seq_len,
                enable_tma=ctx.enable_tma, num_softmax_heads=ctx.num_softmax_heads,
            )
            changed = [name for name, tensor in inputs.items() if tensor._version != versions[name]]
            if changed:
                if failure_path:
                    failures = [{"iteration": iteration, "check": name + "_version_unchanged"} for name in changed]
                    report["iterations_completed"] = iteration
                    report["compiled_kernels"] = compiled_metadata(attention._hstu_attn_bwd)
                    report["status"] = "FAIL"
                    report["checks"].append({"checked_through_iteration": iteration, "failures": failures,
                                             "elapsed_seconds": time.monotonic() - started})
                    report["failure_dump"] = save_failure_dump(
                        failure_path, pristine, inputs, outputs, iteration=iteration, failures=failures,
                        configuration=report["configuration"], compiled_kernels=report["compiled_kernels"],
                        report_path=args.report)
                    publish({"event": "failure_dump", **report["failure_dump"]})
                    return 1
                raise RuntimeError(f"tracked input mutation at iteration {iteration}: {changed}")
            flags = {name + "_finite": finite(tensor) for name, tensor in outputs.items()}
            flags["dq_history_zero"] = history_zero(torch, outputs["dq"], history)
            flags["dq_target_nonzero"] = (outputs["dq"][offsets[1:] - 1] != 0).any()
            for name, snapshot in snapshots.items():
                flags[name + "_unchanged"] = all_chunks(torch, inputs[name], lambda chunk, start, snapshot=snapshot:
                    (chunk.view(torch.uint8) == snapshot[start:start + len(chunk)].view(torch.uint8)).all())
            pending.extend((iteration, name, flag) for name, flag in flags.items())
            report["iterations_completed"] = iteration
            if iteration % args.check_every and iteration != args.repeat:
                continue
            values = torch.stack([flag for _, _, flag in pending]).cpu().tolist()
            failures = [{"iteration": step, "check": name}
                        for (step, name, _), passed in zip(pending, values) if not passed]
            check = dict(checked_through_iteration=iteration, failures=failures,
                         elapsed_seconds=time.monotonic() - started)
            report["checks"].append(check)
            report["compiled_kernels"] = compiled_metadata(attention._hstu_attn_bwd)
            pending.clear()
            report["status"] = "FAIL" if failures else ("PASS" if iteration == args.repeat else "RUNNING")
            publish({"event": "check", **check, "status": report["status"]})
            if failures:
                if failure_path:
                    report["failure_dump"] = save_failure_dump(
                        failure_path, pristine, inputs, outputs, iteration=iteration, failures=failures,
                        configuration=report["configuration"], compiled_kernels=report["compiled_kernels"],
                        report_path=args.report)
                    publish({"event": "failure_dump", **report["failure_dump"]})
                return 1
    return 0


def main():
    args = arguments()
    report = dict(capture=str(args.capture), repeat=args.repeat, check_every=args.check_every,
                  check_input_contents=args.check_input_contents, status="RUNNING",
                  iterations_completed=0, checks=[])

    def publish(record):
        print(json.dumps(record, default=str, allow_nan=False), flush=True)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")

    try:
        result = run(args, report, publish)
    except Exception as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        publish({"event": "error", "error": report["error"]})
        raise
    publish({"event": "summary", "status": report["status"],
             "iterations_completed": report["iterations_completed"],
             "compiled_kernels": report.get("compiled_kernels", [])})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
