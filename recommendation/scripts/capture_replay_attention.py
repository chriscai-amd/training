#!/usr/bin/env python3
"""Capture the last HSTU layer's real backward during a full-state replay.

Use the replay_training_step.py CLI, including --backward-probe-dir. This adapter
substitutes its probe factory without changing the trainer or replay entry point.
The first fused-preprocess backward must belong to the numerically last STU layer;
forward context and the saved norm-weight view establish attribution.

Inputs are copied to independent CPU backing storages before backward executes.
The B0 tripwire checks/copies outputs immediately when that backward returns,
then its artifact's input storage references are replaced with the pristine
snapshots. Original device/stride/alias descriptors are preserved. Source input
version changes remain explicit metadata, separate from the restored input bytes.
Pre-step replay, these copies and immediate checking change execution history.

The resulting B0-format .pt works with replay_nan_tripwire.py and the compact
attention-prefix exporter repro_hstu_attention_capture.py. A completed finite
capture intentionally exits replay code 86, as does a captured anomaly. No
attention-zero oracle is assumed here; the component runner validates its actual
incoming gradients before applying that oracle.

Example:
  python scripts/capture_replay_attention.py /data/capture_step_000101 \
    --mode current --repeats 1 --verify-every-repeat --allow-source-change \
    --backward-probe-dir /data/attention_capture --report /data/attention_replay.json
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import threading


def _encoded_tensors(value):
    if isinstance(value, dict) and "__tripwire_tensor__" in value:
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _encoded_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _encoded_tensors(child)


class AttentionCaptureAdapter:
    def __init__(self, model, directory, *, function_class=None, layer_modules=None):
        import torch
        import nan_backward_boundaries as boundaries
        from generative_recommenders.dlrm_v4.train import nan_tripwire as tripwire_module
        from nan_replay_state import _walk_modules

        self.torch = torch
        self.boundaries = boundaries
        self.error_type = boundaries.BoundaryProbeError
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.abs_threshold = float(os.environ.get("NAN_BACKWARD_ABS_THRESHOLD", "1e20"))
        self.chunk_bytes = int(os.environ.get("NAN_BACKWARD_CHUNK_MIB", "64")) << 20
        self.max_bytes = int(os.environ.get("NAN_BACKWARD_MAX_CAPTURE_GIB", "32")) << 30
        self.save_all = True
        self.attempt = None
        self.closed = self.stopped = False
        self._contexts = {}
        self._local = threading.local()
        self._handles = []
        self._descriptors = {}
        self._pre_storages = {}
        self._last_report = None
        if not math.isfinite(self.abs_threshold) or self.abs_threshold <= 0 or min(self.chunk_bytes, self.max_bytes) <= 0:
            raise self.error_type("Attention capture threshold and copy limits must be positive")
        layers = layer_modules if layer_modules is not None else {
            path.replace("/", "."): module for path, module in _walk_modules(model).items()
            if hasattr(module, "_input_norm_weight") and hasattr(module, "_uvqk_weight")
        }
        numbered = []
        for name, module in layers.items():
            match = re.search(r"(?:^|\.)_stu_layers\.(\d+)$", name)
            if match:
                numbered.append((int(match[1]), name, module))
        if not numbered:
            raise self.error_type("Cannot identify numbered _stu_layers for attention capture")
        maximum = max(item[0] for item in numbered)
        last = [item for item in numbered if item[0] == maximum]
        if len(last) != 1:
            raise self.error_type("Multiple STU stacks have the same final-layer index; attribution is ambiguous")
        self.target = last[0][1]
        self.function_class = function_class or importlib.import_module(
            "generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention"
        )._HSTUPreprocessAndAttentionFunction
        owner = self

        class CaptureStopped(boundaries.BoundaryProbeError):
            def __init__(self, report, capture, reason):
                self.dump_path = Path(capture)
                self.report_path = Path(report)
                super().__init__(f"Attention {reason}; original inputs preserved; report={report}; capture={capture}")

        self.stop_type = CaptureStopped

        class ImmediateTripwire(tripwire_module.NaNTripwire):
            def _record(self, *args, **kwargs):
                super()._record(*args, **kwargs)
                try:
                    self.check("attention_backward")
                except (tripwire_module.CaptureComplete, tripwire_module.NonFiniteError,
                        tripwire_module.MagnitudeError) as error:
                    owner.stopped = True
                    report = json.loads(owner._last_report.read_text()) if owner._last_report else {}
                    if "capture_path" not in report:
                        raise boundaries.BoundaryProbeError(
                            f"Attention report has no complete payload: {report.get('capture_error', str(error))}"
                        ) from error
                    raise CaptureStopped(owner._last_report, report["capture_path"], report["reason"]) from error
                except Exception as error:
                    if isinstance(error, boundaries.BoundaryProbeError):
                        raise
                    raise boundaries.BoundaryProbeError(f"Attention capture failed: {error}") from error
                raise boundaries.BoundaryProbeError("Requested attention capture returned without stopping")

            def _dump(self, *args, **kwargs):
                path = super()._dump(*args, **kwargs)
                owner._last_report = path
                owner._replace_input_storages(path)
                return path

        self.tripwire = ImmediateTripwire(
            directory, function_pattern=r"_HSTUPreprocessAndAttentionFunction$",
            capture_limit_bytes=self.max_bytes, retain_payload=True,
            finite_chunk_elements=max(1, self.chunk_bytes // 8), max_abs=self.abs_threshold,
            capture_direction="backward", capture_function_pattern=r"_HSTUPreprocessAndAttentionFunction$",
        )
        self.source_sha256 = {}
        for source in (Path(__file__), inspect.getsourcefile(self.function_class)):
            if source and Path(source).is_file():
                self.source_sha256[str(Path(source).resolve())] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        try:
            self._install(numbered)
        except BaseException:
            self.close()
            raise

    def _install(self, numbered):
        cls = self.function_class
        for direction in ("forward", "backward"):
            if getattr(getattr(cls, direction), "_nan_tripwire", False):
                raise self.error_type("A tripwire already instruments the requested autograd class")
            self._descriptors[direction] = inspect.getattr_static(cls, direction)
        original_forward = cls.forward
        original_backward = cls.backward
        forward_signature = inspect.signature(original_forward)
        self.tripwire.patch_function(cls)
        instrumented_backward = cls.backward
        # The B0 implementation installs both directions. Remove its forward
        # probe: forward only records attribution, with no flags/payload refs.
        setattr(cls, "forward", self._descriptors["forward"])
        self.tripwire.handles = [item for item in self.tripwire.handles if item[1] == "backward"]

        for _, name, module in numbered:
            def before(_module, _args, _kwargs, layer=name):
                stack = getattr(self._local, "layers", [])
                stack.append(layer)
                self._local.layers = stack

            def after(_module, _args, _kwargs, _output, layer=name):
                stack = getattr(self._local, "layers", [])
                if not stack or stack[-1] != layer:
                    raise self.error_type("Unbalanced STU attribution hooks")
                stack.pop()

            self._handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self._handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))

        @functools.wraps(original_forward)
        def forward(ctx, *args, **kwargs):
            if self.attempt is None:
                return original_forward(ctx, *args, **kwargs)
            stack = getattr(self._local, "layers", [])
            if not stack:
                raise self.error_type("Fused attention forward outside a mapped STU layer")
            weight = forward_signature.bind(ctx, *args, **kwargs).arguments["norm_weight"]
            self._contexts[id(ctx)] = (stack[-1], self.boundaries._pointer(weight))
            return original_forward(ctx, *args, **kwargs)

        @functools.wraps(original_backward)
        def backward(ctx, *args, **kwargs):
            if self.attempt is None:
                return original_backward(ctx, *args, **kwargs)
            if self.stopped:
                raise self.error_type("Backward continued after attention capture")
            mapping = self._contexts.get(id(ctx))
            if mapping is None or mapping[0] != self.target:
                raise self.error_type(f"First fused backward is not the verified final layer {self.target}: {mapping}")
            if len(ctx.saved_tensors) < 2 or self.boundaries._pointer(ctx.saved_tensors[1]) != mapping[1]:
                raise self.error_type("Backward saved norm-weight view differs from its attributed forward")
            self._snapshot_inputs({"args": args, "kwargs": kwargs,
                                   "ctx": dict(ctx.__dict__), "saved_tensors": tuple(ctx.saved_tensors)})
            self.tripwire.metadata.update(layer=self.target, attribution="STU forward context and saved norm-weight view",
                                           source_sha256=self.source_sha256,
                                           pre_input_storage_bytes=sum(t.numel() for t in self._pre_storages.values()))
            return instrumented_backward(ctx, *args, **kwargs)

        setattr(cls, "forward", staticmethod(forward))
        setattr(cls, "backward", staticmethod(backward))

    def _snapshot_inputs(self, values):
        torch = self.torch
        copied, _ = self.boundaries._cpu_copy_tree(values, self.chunk_bytes, self.max_bytes)
        self._pre_storages = {}
        for (name, source), (copy_name, tensor) in zip(self.boundaries._tensors(values), self.boundaries._tensors(copied)):
            if name != copy_name:
                raise self.error_type("Pristine snapshot structure changed")
            key = (str(source.device), source.untyped_storage().data_ptr())
            storage = tensor.untyped_storage()
            self._pre_storages[key] = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))

    def _replace_input_storages(self, report_path):
        """Keep B0 GPU descriptors while substituting independent pre-call bytes."""
        torch = self.torch
        report = json.loads(report_path.read_text())
        if "capture_path" not in report:
            return
        capture_path = Path(report["capture_path"])
        capture = torch.load(capture_path, map_location="cpu", mmap=True, weights_only=False)
        payload = capture["payload"]
        pristine_ids = {}
        next_id = max(capture["storages"], default=-1) + 1
        trees = [payload["inputs"]]
        replay = payload["replay"]
        trees.extend(replay[field] for field in ("args", "kwargs", "ctx", "saved_tensors"))
        replaced = 0
        for tree in trees:
            for descriptor in _encoded_tensors(tree):
                key = (descriptor["device"], descriptor["storage_ptr"])
                if key not in self._pre_storages:
                    raise self.error_type("A replay input has no pristine pre-call backing storage")
                if key not in pristine_ids:
                    pristine_ids[key] = next_id
                    capture["storages"][next_id] = self._pre_storages[key]
                    next_id += 1
                descriptor["__tripwire_tensor__"] = pristine_ids[key]
                replaced += 1
        # Drop now-unreferenced post-call input storages; output aliasing retains
        # its own post-call storage instead of accidentally rewriting evidence.
        used = {item["__tripwire_tensor__"] for item in _encoded_tensors(payload)}
        capture["storages"] = {key: tensor for key, tensor in capture["storages"].items() if key in used}
        changed = []
        selected = next(event for event in report["events"] if event["sequence"] == report["selected_sequence"])
        for info in selected["inputs"]:
            info["source_version_after_operation"] = info.get("version_at_capture")
            info["source_version_changed_during_operation"] = info.get("version_changed")
            if info.get("version_changed"):
                changed.append(info["name"])
            # Replay's mutation guard describes the input bytes in the artifact.
            info["version_at_capture"] = info.get("version")
            info["version_changed"] = False if info.get("version") is not None else None
        report.update(
            capture_timing="pristine CPU input storage snapshots before backward; outputs copied immediately after backward",
            pristine_input_storage_bytes=sum(t.numel() for t in self._pre_storages.values()),
            pristine_input_descriptors=replaced,
            source_inputs_with_tracked_mutations=changed,
            capture_storage_bytes=sum(t.numel() for t in capture["storages"].values()),
            adapter_limitations=["Copies and checks change allocation/synchronization history.",
                                 "Input snapshots are pristine; original execution may still mutate inputs or unrelated storage.",
                                 "No all-history DQ-zero assertion is inferred from layer identity alone."],
        )
        capture["report"] = report
        temporary = capture_path.with_suffix(".pristine.tmp")
        with temporary.open("xb") as stream:
            torch.save(capture, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, capture_path)
        from nan_replay_capture import write_json
        write_json(report_path, report)
        directory_fd = os.open(capture_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._pre_storages.clear()

    def set_attempt(self, mode, repeat, step):
        if self.closed or self.stopped:
            raise self.error_type("Cannot reuse a closed/stopped attention capture adapter")
        self.attempt = {"mode": mode, "repeat": repeat, "step": step}
        self._contexts.clear()
        self.tripwire.capture_step = step
        self.tripwire.begin(step, mode=mode, repeat=repeat, replay_step=step)

    def close(self):
        if self.closed:
            return
        for direction, original in self._descriptors.items():
            setattr(self.function_class, direction, original)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._contexts.clear()
        self._pre_storages.clear()
        self.tripwire.end()
        self.tripwire.handles.clear()
        self.closed = True


def install(model, directory, **kwargs):
    return AttentionCaptureAdapter(model, directory, **kwargs)


def main():
    # Apply captured import-time environment before importing the probe/torch.
    # replay_training_step repeats the restoration and all original source guards.
    import replay_training_step
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return replay_training_step.main()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--backward-probe-dir", type=Path, required=True)
    # Recognize option values so options before the capture path do not get
    # mistaken for the positional path by parse_known_args.
    parser.add_argument("--mode", choices=("current", "previous", "both"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--verify-every-repeat", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-source-change", action="store_true")
    known, _ = parser.parse_known_args()
    context = json.loads((known.capture / "context.json").read_text())
    controls = {key: value for key, value in os.environ.items() if key.startswith("NAN_BACKWARD_")}
    for key, value in context["environment"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.environ.update(controls)
    import nan_backward_boundaries
    original = nan_backward_boundaries.install
    nan_backward_boundaries.install = install
    try:
        return replay_training_step.main()
    finally:
        nan_backward_boundaries.install = original


if __name__ == "__main__":
    main()
