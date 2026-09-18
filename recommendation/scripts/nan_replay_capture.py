"""Rolling full-state, pre-forward capture for the single-rank NaN investigation.

NAN_REPLAY_DIR enables this deliberately synchronous diagnostic. A single CPU
shadow plus a disk undo journal retains two adjacent boundaries, including all
embedding storage. NAN_REPLAY_FORCE_STEP exercises the failure path on a healthy
step. A completed capture stops training BEFORE a bad forward reaches backward.
This is a trusted local debugging artifact, not a portable model checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time

import torch

from nan_replay_state import collect_state, capture_rng, pack_kjt
from nan_replay_storage import (
    RollingTensorSnapshot, describe_tensor_state as _describe,
    validate_tensor_topology as _validate_topology,
)


def partition_state(refs):
    # Full copies of the small state permit shape changes in TBE's saved input
    # scratch attributes. Large embedding storage uses the rolling undo scheme.
    large, small = {}, {}
    for name, tensor in refs.items():
        (large if tensor.untyped_storage().nbytes() >= 1 << 30 else small)[name] = tensor
    return large, small


def pack_small_state(refs):
    topology, views = _describe(refs)
    size = sum(s["nbytes"] for s in topology["storages"])
    if size > 4 << 30:
        raise RuntimeError(f"Small replay state exceeds 4 GiB safety bound: {size}")
    return {"topology": topology, "bytes": [v.to("cpu", copy=True) for v in views]}


def restore_small_state(refs, saved):
    topology, views = _describe(refs)
    expected = saved["topology"]
    if {b["name"] for b in topology["bindings"]} != {b["name"] for b in expected["bindings"]}:
        raise ValueError("Small replay tensor names differ after restoring controls")
    if topology != expected:
        # Recreate changing scratch views AND their aliases while keeping the
        # live Tensor/Parameter objects held by modules and optimizers intact.
        raw = [torch.empty(s["nbytes"], dtype=torch.uint8, device=s["device"])
               for s in expected["storages"]]
        with torch.no_grad():
            for binding in expected["bindings"]:
                tensor = refs[binding["name"]]
                if str(tensor.dtype) != binding["dtype"] or binding["is_conj"] or binding["is_neg"]:
                    raise ValueError(f"Cannot rebuild small state view: {binding['name']}")
                tensor.set_(raw[binding["storage_id"]].untyped_storage(),
                            binding["storage_offset"], binding["shape"], binding["stride"])
        topology, views = _describe(refs)
    _validate_topology(expected, topology)
    for dest, source in zip(views, saved["bytes"]):
        dest.copy_(source)


def compare_small_state(refs, saved):
    topology, views = _describe(refs)
    if topology != saved["topology"]:
        return {"equal": False, "topology_differs": True}
    different, details = [], []
    for index, (view, expected) in enumerate(zip(views, saved["bytes"])):
        actual = view.cpu()
        if not torch.equal(actual, expected):
            different.append(index)
            bindings = [b for b in topology["bindings"] if b["storage_id"] == index]
            dtype_names = {b["dtype"] for b in bindings}
            item = {"storage_id": index, "mismatched_bytes": int((actual != expected).sum())}
            if len(dtype_names) == 1:
                dtype = getattr(torch, next(iter(dtype_names)).removeprefix("torch."))
                if dtype.is_floating_point:
                    a, b = actual.view(dtype), expected.view(dtype)
                    max_abs, count = 0., 0
                    for offset in range(0, a.numel(), 1 << 20):
                        aa, bb = a[offset:offset + (1 << 20)], b[offset:offset + (1 << 20)]
                        count += int((aa != bb).sum())
                        finite = torch.isfinite(aa) & torch.isfinite(bb)
                        if bool(finite.any()):
                            max_abs = max(max_abs, float((aa[finite].double() - bb[finite].double()).abs().max()))
                    item.update(dtype=str(dtype), mismatched_values=count, max_finite_abs_diff=max_abs)
            details.append(item)
    return {"equal": not different, "different_storage_ids": different,
            "details": details,
            "different_bindings": [b["name"] for b in topology["bindings"] if b["storage_id"] in different]}


def nonfinite_small_state(saved):
    hits, seen = [], set()
    for binding in saved["topology"]["bindings"]:
        dtype = getattr(torch, binding["dtype"].removeprefix("torch."))
        if not (dtype.is_floating_point or dtype.is_complex):
            continue
        identity = (binding["storage_id"], binding["dtype"], binding["storage_offset"],
                    tuple(binding["shape"]), tuple(binding["stride"]))
        if identity in seen:
            continue
        seen.add(identity)
        view = torch.empty(0, dtype=dtype).set_(
            saved["bytes"][binding["storage_id"]].untyped_storage(),
            binding["storage_offset"], binding["shape"], binding["stride"],
        ).reshape(-1)
        for chunk in view.split(1 << 22):
            if not bool(torch.isfinite(chunk).all()):
                hits.append(binding["name"])
                break
    return hits


def nonfinite_large_state(snapshot):
    dtypes = {}
    for binding in snapshot.topology["bindings"]:
        sid = binding["storage_id"]
        dtype = getattr(torch, binding["dtype"].removeprefix("torch."))
        if not dtype.is_floating_point:
            raise ValueError("Large replay state scanner requires floating embedding storage")
        if sid in dtypes and dtypes[sid] != dtype:
            raise ValueError("Large state has incompatible dtype aliases")
        dtypes[sid] = dtype
    hits = snapshot.nonfinite_floating_ranges(dtypes)
    names = []
    for hit in hits:
        aliases = []
        for binding in snapshot.topology["bindings"]:
            if binding["storage_id"] != hit["storage_id"] or "/parameter/" not in binding["name"]:
                continue
            width = torch.empty((), dtype=dtypes[hit["storage_id"]]).element_size()
            span = 1 + sum((size - 1) * stride for size, stride in zip(binding["shape"], binding["stride"]))
            begin = binding["storage_offset"] * width
            if begin <= hit["byte_offset"] < begin + span * width:
                aliases.append(binding["name"])
        names.append(f"large_storage/{hit['storage_id']}/byte/{hit['byte_offset']}" +
                     (" aliases=" + ",".join(aliases) if aliases else ""))
    return names


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported observation: {type(value)}")


def nonfinite_names(value, path=""):
    found = []
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and value.numel():
            if not bool(torch.isfinite(value).all().item()):
                found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite_names(item, f"{path}/{key}"))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found.extend(nonfinite_names(item, f"{path}/{i}"))
    return found


def observe_forward(output):
    return cpu_tree(dict(zip(("losses", "preds", "labels", "weights"), output[2:])))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def configuration_context():
    import gin
    import triton

    root = Path(__file__).resolve().parents[1]
    gin_text = gin.config_str()
    # Snapshot explicitly referenced configuration variables, including their
    # absence. Do not dump unrelated process credentials into artifacts.
    keys = set(re.findall(r"\.key\s*=\s*['\"]([A-Z][A-Z0-9_]*)['\"]", gin_text))
    source_hashes = {}
    pattern = re.compile(r"(?:os\.environ\.get|os\.getenv|os\.environ\[)\s*\(?\s*['\"]([A-Z][A-Z0-9_]*)['\"]")
    for subdir in ("generative_recommenders", "scripts"):
        for path in sorted((root / subdir).rglob("*.py")):
            data = path.read_bytes()
            keys.update(pattern.findall(data.decode(errors="replace")))
            source_hashes[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    for key in os.environ:
        if key.startswith(("TRITON_", "AMD", "HIP", "HSA_", "ROCM", "PYTORCH_", "TORCH_", "NCCL_", "CUBLAS_")):
            keys.add(key)
    keys.update(("DLRM_DATA_PATH", "GPUS_PER_NODE", "NNODES", "NODE_RANK", "OMP_NUM_THREADS", "LD_LIBRARY_PATH"))
    env = {key: os.environ.get(key) for key in sorted(keys)}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    return {
        "format_version": 1,
        "gin": gin_text,
        "environment": env,
        "source_sha256": source_hashes,
        "git_head": git_head,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "triton_distribution": importlib.metadata.version("triton"),
        "triton_path": triton.__file__,
        "device": torch.cuda.get_device_name(0),
        "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
        "cpu_threads": torch.get_num_threads(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": [
            "Full synchronization and D2H copies change timing and allocator history.",
            "Captures tensor state, optimizer controls and default RNGs; not arbitrary GPU address-space bytes or driver state.",
            "In-memory shadow requires the process to survive to persist a NaN capture; not crash/hang recovery.",
        ],
    }


class ReplayCapture:
    def __init__(self, model, optimizer, directory):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise RuntimeError("NaN replay capture currently requires world_size=1")
        if os.environ.get("NAN_MODULE_PROBE") == "1":
            raise RuntimeError("Use NAN_MODULE_PROBE=0 for initial full-state replay validation")
        self.model, self.optimizer = model, optimizer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.context = configuration_context()
        write_json(self.directory / "context.json", self.context)
        self.snapshot = None
        self.current = self.previous = None
        self.force_step = int(os.environ.get("NAN_REPLAY_FORCE_STEP", "0"))
        self.chunk_bytes = int(os.environ.get("NAN_REPLAY_CHUNK_MIB", "64")) << 20
        self.page_bytes = int(os.environ.get("NAN_REPLAY_PAGE_KIB", "16")) << 10

    def before_step(self, step, sample, grad_clip_norm):
        started = time.monotonic()
        torch.cuda.synchronize()
        live, controls = collect_state(self.model, self.optimizer)
        large, small = partition_state(live)
        if self.snapshot is None:
            self.snapshot = RollingTensorSnapshot(
                large, self.directory / "rolling", boundary_id=step,
                chunk_bytes=self.chunk_bytes, page_bytes=self.page_bytes,
            )
        else:
            self.snapshot.update(large, boundary_id=step)
        self.previous = self.current
        self.current = {
            "step": step, "controls": controls,
            "small_state": pack_small_state(small),
            "inputs": {"uih": pack_kjt(sample.uih_features_kjt), "candidates": pack_kjt(sample.candidates_features_kjt)},
            "grad_clip_norm": grad_clip_norm,
            "rng": capture_rng(),
        }
        # Copies and bookkeeping above must not consume default RNG state.
        elapsed = time.monotonic() - started
        small_bytes = sum(s["nbytes"] for s in self.current["small_state"]["topology"]["storages"])
        write_json(self.directory / "status.json", {
            "status": "running", "pre_step": step,
            "previous_step": self.previous["step"] if self.previous else None,
            "capture_seconds": elapsed, "updated_unix": time.time(),
            "large_state_bytes": self.snapshot.total_bytes,
            "small_state_bytes": small_bytes,
        })
        print(f"[nan-replay] pre_step={step} full_state_capture_seconds={elapsed:.3f}", flush=True)
        # Check all initial floating state, then every changed embedding page.
        # This prevents a delayed loss NaN from hiding an earlier bad update.
        bad = nonfinite_small_state(self.current["small_state"]) + nonfinite_large_state(self.snapshot)
        if bad:
            self.freeze("pre_forward_state", bad)

    def after_forward(self, losses, preds, labels, weights):
        self.current["observed"] = cpu_tree({"losses": losses, "preds": preds, "labels": labels, "weights": weights})
        bad = nonfinite_names(self.current["observed"])
        if bad:
            self.freeze("forward", bad)
        if self.current["step"] == self.force_step:
            self.freeze("forced_forward", [])

    def after_backward(self):
        # Check before global clipping can spread a single poisoned gradient.
        bad = [name for name, parameter in self.model.named_parameters()
               if parameter.grad is not None and nonfinite_names(parameter.grad)]
        if bad:
            self.freeze("backward_before_clip", bad)

    def freeze(self, stage, bad):
        torch.cuda.synchronize()
        target = self.directory / f"capture_step_{self.current['step']:06d}"
        target.mkdir(exist_ok=False)
        write_json(self.directory / "status.json", {
            "status": "saving", "step": self.current["step"], "stage": stage,
            "nonfinite": bad, "capture": str(target), "updated_unix": time.time(),
        })
        print(f"[nan-replay] freezing step={self.current['step']} stage={stage} nonfinite={bad}", flush=True)
        # A partial directory has no COMPLETE marker and cannot be replayed.
        payload = {"current": self.current, "previous": self.previous, "stage": stage, "nonfinite": bad}
        with (target / "capture.pt").open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        write_json(target / "context.json", self.context)
        self.snapshot.persist(target / "state", metadata={"stage": stage, "step": self.current["step"]})
        write_json(target / "COMPLETE.json", {"format_version": 1, "step": self.current["step"], "stage": stage})
        write_json(self.directory / "status.json", {
            "status": "captured", "step": self.current["step"], "stage": stage,
            "nonfinite": bad, "capture": str(target), "updated_unix": time.time(),
        })
        print(f"[nan-replay] CAPTURE_COMPLETE {target}", flush=True)
        # Propagates through multiprocessing as a deliberate diagnostic exit;
        # never swallow capture errors and quietly resume an uncaptured run.
        raise SystemExit(86 if bad else 0)


_recorder = None


def before_step(model, optimizer, step, sample, grad_clip_norm):
    global _recorder
    if _recorder is None:
        _recorder = ReplayCapture(model, optimizer, os.environ["NAN_REPLAY_DIR"])
    _recorder.before_step(step, sample, grad_clip_norm)
    return _recorder
