#!/usr/bin/env python3
"""Keep complete final buffers for positive stages and ordinary raw failures.

Host-only extension of frozen v1/v2. The instrumented JIT remains unchanged.
Requires distinct --stage-dump, raw --failure-dump and optional --report paths.
A positive stage also writes STAGE_DUMP.pair.json linking the unchanged stage
artifact to complete final input/output buffers from the same kernel launch.
Inconsistent workspace decoding and forced-control labels remain explicit.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import torch

import attention_dq_narrow_stage_probe as base
import attention_dq_narrow_stage_probe_v2 as v2
import probe_attention_raw_failure as runner


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _write_pair(path, data, *, first):
    path = Path(path)
    if first and path.exists():
        raise FileExistsError(f"Pair report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_view(left, right):
    if left is None or right is None:
        return left is right
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    def signature(value):
        return (str(value.device), str(value.dtype), tuple(value.shape), tuple(value.stride()),
                value.storage_offset(), value.data_ptr(), value.untyped_storage().data_ptr(),
                value.untyped_storage().nbytes(), value.is_conj(), value.is_neg())
    return signature(left) == signature(right)


class NarrowStageControllerV3(v2.NarrowStageControllerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_context = None
        self.positive_output_capture = None
        self._verified_launch = None
        self.pair_path = Path(str(self.destination) + ".pair.json")

    def bind_raw_context(self, args, report, inputs, outputs, pristine, *, chunk_bytes, save=None):
        if self.raw_context is not None:
            raise ValueError("Narrow v3 supports one raw loop per controller")
        if args.keep_going or args.failure_dump is None:
            raise ValueError("Narrow v3 requires --failure-dump and rejects --keep-going")
        paths = [self.destination, self.pair_path, args.failure_dump, args.capture]
        if args.report is not None:
            paths.append(args.report)
        if len({Path(path).resolve() for path in paths}) != len(paths):
            raise ValueError("Stage, pair, raw output, report and source paths must be distinct")
        if self.pair_path.exists() or Path(args.failure_dump).exists():
            raise ValueError("Narrow v3 requires new pair and raw output paths")
        report.setdefault("iterations_completed", 0)
        self.raw_context = {"args": args, "report": report, "inputs": inputs, "outputs": outputs,
                            "pristine": pristine, "chunk_bytes": chunk_bytes,
                            "save": runner.save_failure_dump if save is None else save}
        self.pristine = pristine["inputs"]

    def _validate_kernel_views(self, kwargs):
        if self.raw_context is None:
            raise ValueError("Narrow v3 lacks current raw input/output context")
        inputs, outputs = self.raw_context["inputs"], self.raw_context["outputs"]
        names = {"q": "Q", "k": "K", "v": "V", "dout": "DOut",
                 "seq_offsets": "seq_offsets", "num_targets": "num_targets",
                 "sort_by_length_indices": "sort_by_length_indices"}
        for name, argument in names.items():
            if argument not in kwargs or not _same_view(inputs.get(name), kwargs[argument]):
                raise ValueError(f"Raw kernel input {argument} differs from retained caller view")
        for name in ("dq", "dk", "dv"):
            if name.upper() not in kwargs or not _same_view(outputs[name], kwargs[name.upper()]):
                raise ValueError(f"Raw kernel output {name.upper()} differs from retained caller view; copyback is unsupported")

    def launch(self, function, kwargs, *, grid):
        if self.positive is not None or self.positive_output_capture is not None:
            raise ValueError("Cannot launch after the first positive-stage buffer capture")
        self._validate_kernel_views(kwargs)
        self._verified_launch = self.calls + 1
        try:
            return super().launch(function, kwargs, grid=grid)
        except base.StageProbePositive:
            self.capture_positive_outputs()
            raise

    def capture_positive_outputs(self):
        if self.positive_output_capture is not None or self.raw_failure_capture is not None:
            raise ValueError("Only the first failure may own the complete output capture")
        if self.raw_context is None or self.positive is None:
            raise ValueError("Positive-stage capture requires current raw context and saved stage evidence")
        context = self.raw_context
        args, report = context["args"], context["report"]
        if (self.calls < 1 or self.calls != report.get("iterations_completed", 0) + 1
                or self._verified_launch != self.calls
                or self.positive.get("iteration") != self.calls or "first_failure_dump" in report):
            raise ValueError("Positive-stage iteration does not match the first current raw launch")
        stage_hash = _sha256(self.destination)
        stage = torch.load(self.destination, map_location="cpu", weights_only=False)
        if (stage.get("format") != "hstu_dq_narrow_stage_probe_v1"
                or stage.get("iteration") != self.calls or stage.get("selected_stage") != self.stage
                or stage.get("force_positive_control") is not self.force_positive_control
                or stage.get("workspace_spec") != self.spec):
            raise ValueError("Saved positive workspace does not match the current launch")
        raw = stage["workspace_raw_int32"]
        scan = v2.summarize_workspace(raw, self.spec)
        correlation = {
            "iteration": self.calls, "narrow_launch_count": self.calls,
            "workspace_path": str(self.destination.resolve()),
            "workspace_file_sha256": stage_hash,
            "workspace_sha256": hashlib.sha256(raw.contiguous().numpy().tobytes()).hexdigest(),
            "raw_output_path": str(Path(args.failure_dump).resolve()),
            "pair_path": str(self.pair_path.resolve()),
            "selected_stage": self.stage, "force_positive_control": self.force_positive_control,
            "decode_error": stage.get("decode_error"),
            "capture_kind": "forced_diagnostic_control" if self.force_positive_control else "positive_instrumented_stage",
            "capture_timing": "Raw kernel completed and workspace saved; full buffers copied before standard output checks, with no subsequent attention launch",
            "actual_kernel_views_match_retained_callers": True,
        }
        pair = {"format": "hstu_narrow_positive_pair_v3", "correlation": correlation,
                "workspace_scan": scan, "raw_dump_status": "pending"}
        _write_pair(self.pair_path, pair, first=True)
        summary = {"path": str(args.failure_dump), "pair_path": str(self.pair_path),
                   "iteration": self.calls, "selected_stage": self.stage,
                   "force_positive_control": self.force_positive_control,
                   "decode_error": stage.get("decode_error"), "correlation": correlation,
                   "raw_dump_status": "pending"}
        self.positive_output_capture = summary
        report["first_narrow_positive_output_capture"] = summary
        configuration = {**report["configuration"], "narrow_positive_stage_capture": copy.deepcopy(correlation),
            "narrow_host_capture_sources": {Path(path).name: _sha256(path)
                                             for path in (__file__, v2.__file__, base.__file__)}}
        failures = [{"iteration": self.calls,
                     "check": "forced_narrow_stage_control" if self.force_positive_control else "narrow_stage_positive"}]
        try:
            result = context["save"](args.failure_dump, context["pristine"], context["inputs"], context["outputs"],
                iteration=self.calls, failures=failures, configuration=configuration,
                compiled_kernels=stage["compiled_kernels"], report_path=args.report,
                chunk_bytes=context["chunk_bytes"], max_bytes=args.max_storage_gib << 30)
            if (result.get("iteration") != self.calls
                    or Path(result.get("path", "")).resolve() != Path(args.failure_dump).resolve()):
                raise ValueError("Raw saver returned a different positive-stage iteration or path")
            if _sha256(self.destination) != stage_hash:
                raise ValueError("Original stage artifact changed while saving complete buffers")
            raw_hash = _sha256(args.failure_dump)
            pair.update(raw_dump_status="complete", raw_file_sha256=raw_hash)
            summary.update(raw_dump_status="complete", raw_file_sha256=raw_hash)
            report["first_narrow_positive_raw_dump"] = result
        except BaseException as error:
            pair.update(raw_dump_status="failed", error=f"{type(error).__name__}: {error}")
            summary.update(raw_dump_status="failed", error=pair["error"])
            _write_pair(self.pair_path, pair, first=False)
            raise
        _write_pair(self.pair_path, pair, first=False)
        if self.publish is not None:
            self.publish({"event": "narrow_positive_full_buffers_saved", **summary})
        return result


def arguments(argv=None):
    options, raw_args = v2.arguments(argv)
    pair = Path(str(options.stage_dump) + ".pair.json")
    paths = [options.stage_dump, raw_args.failure_dump, raw_args.capture, pair]
    if raw_args.report is not None:
        paths.append(raw_args.report)
    if pair.exists() or len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("Pair report must be new and distinct from all other paths")
    return options, raw_args


@contextmanager
def capture_hooks():
    original_controller, original_save, original_loop = base.NarrowStageController, runner.save_failure_dump, runner.run_loop
    instance = None

    def factory(*args, **kwargs):
        nonlocal instance
        if instance is not None:
            raise ValueError("Narrow v3 supports one controller per invocation")
        instance = NarrowStageControllerV3(*args, **kwargs)
        return instance

    def save(*args, **kwargs):
        if instance is None:
            raise ValueError("Raw failure save arrived before controller creation")
        return instance.save_raw_failure(original_save, *args, **kwargs)

    def loop(args, report, publish, attention, config, inputs, outputs, ctx, history,
             pristine, expected_digests, *, chunk_bytes):
        if instance is None:
            raise ValueError("Raw loop arrived before controller creation")
        instance.bind_raw_context(args, report, inputs, outputs, pristine, chunk_bytes=chunk_bytes, save=original_save)
        return original_loop(args, report, publish, attention, config, inputs, outputs, ctx, history,
                             pristine, expected_digests, chunk_bytes=chunk_bytes)

    base.NarrowStageController, runner.save_failure_dump, runner.run_loop = factory, save, loop
    try:
        yield
    finally:
        base.NarrowStageController, runner.save_failure_dump, runner.run_loop = original_controller, original_save, original_loop


def main(argv=None):
    arguments(argv)
    with capture_hooks():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
