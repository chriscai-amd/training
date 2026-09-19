#!/usr/bin/env python3
"""Preserve the complete narrow workspace at the first raw output failure.

Uses the unchanged v1 instrumented kernel and its existing positive/control
capture path. Adds a host-side snapshot before the first raw failure dump even
when the global claim word is zero. --failure-dump is required; --keep-going is
rejected. Pass the existing narrow probe arguments and raw arguments after --.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import torch

import attention_dq_narrow_stage_probe as base
import probe_attention_raw_failure as runner


WORKSPACE_INTERPRETATION = (
    "Every saved workspace word is independently scanned on CPU. Zero global "
    "claim alone does not establish zero workspace. Even an entirely zero "
    "workspace records absence of retained diagnostic writes, not proof that "
    "the selected arithmetic stage was zero or correct during the kernel."
)


def summarize_workspace(raw, spec):
    if (raw.device.type != "cpu" or raw.dtype != torch.int32 or raw.ndim != 1
            or raw.numel() != spec["total_words"]):
        raise ValueError("Unexpected complete narrow workspace layout")
    if spec["global_words"] != base.GLOBAL_WORDS:
        raise ValueError("Unexpected narrow global header size")
    headers = raw[base.GLOBAL_WORDS:].view(spec["programs"], spec["program_stride_words"])
    counts = (headers != 0).sum(dim=1)
    unclaimed = headers[:, 0] == 0
    nonzero = int((raw != 0).sum())
    return {
        "scanned_words": raw.numel(), "scanned_bytes": raw.numel() * raw.element_size(),
        "global_words": raw[:base.GLOBAL_WORDS].tolist(),
        "global_first_claim_word": int(raw[0]),
        "global_claimed_program_count_word": int(raw[1]),
        "global_reserved_nonzero_words": int((raw[2:base.GLOBAL_WORDS] != 0).sum()),
        "nonzero_words": nonzero, "all_workspace_words_zero": nonzero == 0,
        "nonzero_program_header_words": int((headers[:, 0] != 0).sum()),
        "unclaimed_program_nonzero_words": int(counts[unclaimed].sum()),
        "per_program": [{"program_id": pid, "header_word": int(headers[pid, 0]),
                         "nonzero_words": int(counts[pid])}
                        for pid in range(spec["programs"])],
        "first_nonzero_words": [
            {"offset_words": int(index), "int32": int(raw[index]),
             "hex_uint32": f"0x{int(raw[index]) & 0xffffffff:08x}"}
            for index in torch.nonzero(raw).flatten()[:32].tolist()],
        "interpretation": WORKSPACE_INTERPRETATION,
    }


def _save_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Workspace artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class NarrowStageControllerV2(base.NarrowStageController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_failure_capture = None

    def launch(self, function, kwargs, *, grid):
        if self.raw_failure_capture is not None:
            raise ValueError("Cannot launch after the first raw failure workspace capture")
        return super().launch(function, kwargs, grid=grid)

    def save_raw_failure(self, original_save, path, pristine, inputs, outputs, *,
                         iteration, failures, configuration, compiled_kernels, **kwargs):
        if self.raw_failure_capture is not None or self.positive is not None:
            raise ValueError("Only the first failure may own the workspace capture")
        if self.report is not None and "first_failure_dump" in self.report:
            raise ValueError("A previous raw failure already owns the first-failure dump")
        if (self.workspace is None or self.spec is None or iteration < 1
                or iteration != self.calls or not failures
                or any(record.get("iteration") != iteration for record in failures)):
            raise ValueError("Raw failure iteration does not match the current narrow launch")
        if Path(path).resolve() == self.destination.resolve() or Path(path).exists():
            raise ValueError("Raw failure path must be new and distinct from workspace path")
        # Blocking readback freezes this launch's complete workspace before the
        # larger original output/input dump. It is saved regardless of word 0.
        raw = self.workspace.detach().cpu().clone()
        scan = summarize_workspace(raw, self.spec)
        workspace_hash = hashlib.sha256(raw.contiguous().numpy().tobytes()).hexdigest()
        try:
            decoded = base.decode_workspace(raw, self.spec, self.pristine,
                                            force_positive_control=self.force_positive_control)
        except ValueError as error:
            decoded = {"first_claimed_program": int(raw[0]) - 1 if int(raw[0]) else None,
                       "records": [], "decode_error": str(error)}
        correlation = {
            "iteration": iteration, "narrow_launch_count": self.calls,
            "raw_failure_path": str(Path(path).resolve()),
            "workspace_path": str(self.destination.resolve()),
            "workspace_sha256": workspace_hash,
            "first_raw_failure": True,
            "capture_timing": "After this call's raw checks; before first raw failure dump; no subsequent attention launch",
        }
        output_checks = {name + suffix for name in outputs for suffix in ("_finite", "_zero_oracle", "_target_nonzero")}
        trigger = ("raw_output_failure" if any(record.get("check") in output_checks for record in failures)
                   else "raw_failure_after_launch")
        payload = {
            "format": "hstu_dq_narrow_raw_failure_workspace_v2", "format_version": 2,
            "trigger": trigger, "iteration": iteration, "selected_stage": self.stage,
            "force_positive_control": self.force_positive_control,
            "workspace_spec": self.spec, "workspace_raw_int32": raw, "workspace_scan": scan,
            "correlation": correlation, "failures": copy.deepcopy(failures),
            "launch_metadata": self.launch_metadata, "diagnostic_metadata": self.metadata,
            "compiled_kernels": compiled_kernels,
            "source_capture": self.source_capture, "effective_input_digests": self.expected_digests,
            "configuration": configuration,
            "host_source_sha256": {Path(source).name: hashlib.sha256(Path(source).read_bytes()).hexdigest()
                                   for source in (__file__, base.__file__, runner.__file__)},
            "raw_failure_dump_completion": "Not yet observed: workspace is persisted first; consult matching raw artifact/report",
            "scope": WORKSPACE_INTERPRETATION, **decoded,
        }
        _save_new(self.destination, payload)
        summary = {**correlation, "path": str(self.destination), "trigger": trigger,
                   "selected_stage": self.stage, "workspace_scan": scan,
                   "decode_error": decoded.get("decode_error"),
                   "decoded_records": len(decoded["records"]),
                   "raw_failure_dump_status": "pending"}
        self.raw_failure_capture = summary
        if self.report is not None:
            self.report["first_narrow_raw_failure_workspace"] = summary
        if self.publish is not None:
            self.publish({"event": "narrow_raw_failure_workspace_saved", **summary})
        linked_configuration = {**configuration, "narrow_stage_workspace_capture": copy.deepcopy(correlation)}
        try:
            result = original_save(path, pristine, inputs, outputs, iteration=iteration,
                failures=failures, configuration=linked_configuration, compiled_kernels=compiled_kernels, **kwargs)
            if (result.get("iteration") != iteration
                    or Path(result.get("path", "")).resolve() != Path(path).resolve()):
                raise ValueError("Raw failure saver returned a different path or iteration")
        except BaseException as error:
            summary.update(raw_failure_dump_status="failed", raw_failure_dump_error=f"{type(error).__name__}: {error}")
            raise
        summary["raw_failure_dump_status"] = "complete"
        return {**result, "narrow_stage_workspace": copy.deepcopy(correlation)}


def validate_raw_options(options, raw_args):
    if raw_args.keep_going:
        raise ValueError("Narrow v2 rejects --keep-going to preserve the first failing launch workspace")
    if raw_args.failure_dump is None:
        raise ValueError("Narrow v2 requires --failure-dump to correlate the first raw failure")
    paths = [options.stage_dump, raw_args.failure_dump, raw_args.capture]
    if raw_args.report is not None:
        paths.append(raw_args.report)
    if len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("Workspace, raw failure, report and capture paths must be distinct")


def arguments(argv=None):
    options = base.arguments(argv)
    raw_args = runner.arguments(options.raw_args)
    validate_raw_options(options, raw_args)
    return options, raw_args


@contextmanager
def capture_hooks():
    """Scoped host-only extension of v1; the instrumented JIT is unchanged."""
    original_controller, original_save = base.NarrowStageController, runner.save_failure_dump
    instance = None

    def controller_factory(*args, **kwargs):
        nonlocal instance
        if instance is not None:
            raise ValueError("Narrow v2 supports exactly one controller per invocation")
        instance = NarrowStageControllerV2(*args, **kwargs)
        return instance

    def save(*args, **kwargs):
        if instance is None:
            raise ValueError("Raw failure save arrived before narrow controller creation")
        return instance.save_raw_failure(original_save, *args, **kwargs)

    base.NarrowStageController, runner.save_failure_dump = controller_factory, save
    try:
        yield
    finally:
        base.NarrowStageController, runner.save_failure_dump = original_controller, original_save


def main(argv=None):
    arguments(argv)  # Parse/validate keep-going, output paths and limits before any GPU initialization.
    with capture_hooks():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
