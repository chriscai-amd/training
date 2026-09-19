#!/usr/bin/env python3
"""Stress a saved HSTU output stage with resident inputs and full GPU checks.

Every call compares all outputs with the captured correct tensors. Only scalar
check results reach the CPU on a healthy iteration. Exact byte differences are
counted; finite deviations within --rtol/--atol are reported and may continue.
The first larger mismatch, nonfinite value, or value above --threshold stops the
loop and saves that call's existing tensors without rerunning the operation.
Load only trusted local boundary dumps and context files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import replay_hstu_output_boundary as replay


FIELDS = ("nonfinite_elements", "extreme_elements", "different_bytes",
          "different_numeric_elements", "outside_tolerance_elements",
          "max_abs_finite", "max_abs_difference_finite_pairs")


def norm_kernel_metadata(module):
    """Read compiled caches and selected configs without launching kernels."""
    if module is None:
        return {"operation_kind": "torch.mm", "triton_kernels": {}}
    from repro_hstu_attention_capture import compiled_metadata

    def config_info(config):
        return None if config is None else {
            "kwargs": config.kwargs, "num_warps": config.num_warps,
            "num_stages": config.num_stages, "num_ctas": getattr(config, "num_ctas", None)}

    records = {}
    for name in ("_ln_mul_dropout_bwd_dx_du_rng", "_ln_mul_dropout_bwd_dx_du", "_ln_mul_dropout_bwd_dwdb"):
        kernel = getattr(module, name, None)
        if kernel is None:
            continue
        cache = getattr(kernel, "cache", {})
        records[name] = {"compiled_variants": compiled_metadata(kernel),
                         "best_config": config_info(getattr(kernel, "best_config", None)),
                         "autotune_cache": {str(key): config_info(value) for key, value in cache.items()}}
    return {"operation_kind": "normalization_multiply_dropout_backward", "triton_kernels": records,
            "scope": "Cached variants and selected configs in this process; cache membership alone does not prove a variant produced a particular output."}


def compare_device_outputs(outputs, references, names, *, chunk_elements=1 << 24,
                           threshold=1e20, rtol=1e-2, atol=1e-12):
    """Bounded device reductions; exactly one scalar-matrix CPU readback."""
    torch = replay.common._torch()
    if len(outputs) != len(references) or len(outputs) != len(names):
        raise ValueError("Output count differs from captured reference")
    matrices, descriptors = [], []
    for name, value, reference in zip(names, outputs, references):
        if value is None or reference is None:
            if value is not reference:
                raise ValueError(f"Optional output {name} differs from captured presence")
            descriptors.append((name, None))
            continue
        if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"Output {name} shape, dtype, or device differs from reference")
        if not value.is_floating_point():
            raise ValueError("Only floating-point stage outputs are supported")
        totals = torch.zeros(7, device=value.device, dtype=torch.float64)
        for current, expected, _ in replay.common._paired_chunks(value, reference, chunk_elements):
            if not current.numel():
                continue
            current = current.resolve_conj().resolve_neg().contiguous()
            expected = expected.resolve_conj().resolve_neg().contiguous()
            different_bytes = (current.reshape(-1).view(torch.uint8) != expected.reshape(-1).view(torch.uint8)).sum()
            dtype = torch.float64 if current.dtype == torch.float64 else torch.float32
            observed, baseline = current.to(dtype), expected.to(dtype)
            finite = torch.isfinite(observed)
            finite_pair = finite & torch.isfinite(baseline)
            magnitude = observed.abs()
            difference = torch.where(finite_pair, (observed - baseline).abs(), 0.)
            counts = torch.stack((
                (~finite).sum(), (finite & (magnitude > threshold)).sum(), different_bytes,
                (observed != baseline).sum(),
                (finite_pair & (difference > atol + rtol * baseline.abs())).sum(),
            )).to(torch.float64)
            maxima = torch.stack((torch.where(finite, magnitude, 0.).amax(), difference.amax())).to(torch.float64)
            totals[:5] += counts
            totals[5:] = torch.maximum(totals[5:], maxima)
        descriptors.append((name, len(matrices)))
        matrices.append(totals)
    host = torch.stack(matrices).cpu().tolist() if matrices else []
    return {name: None if index is None else {
        field: int(host[index][i]) if i < 5 else replay.common._number(host[index][i])
        for i, field in enumerate(FIELDS)} for name, index in descriptors}


def classify(checks, *, strict_exact=False):
    labels, failed = {}, False
    for name, values in checks.items():
        if values is None:
            labels[name] = "not_returned"
        elif values["nonfinite_elements"]:
            labels[name], failed = "nonfinite", True
        elif values["extreme_elements"]:
            labels[name], failed = "extreme_finite", True
        elif values["outside_tolerance_elements"]:
            labels[name], failed = "finite_mismatch_outside_tolerance", True
        elif values["different_bytes"]:
            labels[name] = "finite_difference_within_tolerance"
            failed |= strict_exact
        else:
            labels[name] = "exact_bytes"
    return {"failed": bool(failed), "classifications": labels}


def failure_locations(outputs, references, *, chunk_bytes, threshold, rtol, atol):
    """CPU-only sample coordinates from the already captured failed outputs."""
    torch = replay.common._torch()
    samples = {}
    for name, actual in outputs.items():
        if actual is None:
            samples[name] = []
            continue
        expected = references[name]
        selected = []
        for left, right, origin in replay.common._paired_chunks(actual, expected, max(1, chunk_bytes // 128)):
            a, b = left.to(torch.float64), right.to(torch.float64)
            bad = ~torch.isfinite(a) | (a.abs() > threshold) | ((a - b).abs() > atol + rtol * b.abs())
            # Exact byte differences within tolerance are useful when strict mode stops.
            if not bool(bad.any()):
                lc, rc = left.contiguous().reshape(-1), right.contiguous().reshape(-1)
                bad = (lc.view(torch.uint8) != rc.view(torch.uint8)).reshape(-1, left.element_size()).any(1).reshape(left.shape)
            if bool(bad.any()):
                for flat in torch.nonzero(bad.reshape(-1)).flatten()[:32 - len(selected)].tolist():
                    selected.append({"index": replay.common._coordinate(flat, left.shape, origin),
                                     "actual": str(a.reshape(-1)[flat].item()),
                                     "captured_correct": str(b.reshape(-1)[flat].item())})
            if len(selected) == 32:
                break
        samples[name] = selected
    return samples


def save_failed_call(path, payload, pristine, inputs, outputs, *, iteration, checks,
                     configuration, chunk_bytes=64 << 20, max_bytes=96 << 30,
                     threshold=1e20, rtol=1e-2, atol=1e-12, gpu_references=None):
    """Copy existing buffers before any further operation; preserve raw storage."""
    torch = replay.common._torch()
    from nan_backward_boundaries import _cpu_copy_tree
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Failure artifact already exists: {path}")
    originals = {"inputs": pristine, "reference_outputs": payload["outputs"]}
    old_storages = {t.untyped_storage()._cdata: t.untyped_storage().nbytes()
                    for t in replay.common._tensors(originals)}
    current, copied_bytes = _cpu_copy_tree({"inputs": inputs, "outputs": outputs, "references": gpu_references}, chunk_bytes,
                                          max_bytes - sum(old_storages.values()))
    pristine_hashes = replay.input_digests(pristine, chunk_bytes=chunk_bytes, threshold=threshold)
    actual_hashes = replay.input_digests(current["inputs"], chunk_bytes=chunk_bytes, threshold=threshold)
    names = replay.OUTPUTS[payload["operation"]]
    actual_map = dict(zip(names, current["outputs"]))
    reference_map = dict(zip(names, payload["outputs"]))
    locations = failure_locations(actual_map, reference_map, chunk_bytes=chunk_bytes,
                                   threshold=threshold, rtol=rtol, atol=atol)
    reference_unchanged = None
    if current["references"] is not None:
        reference_unchanged = all(replay.same_digest(
            replay.common.output_digest(a, chunk_bytes, threshold),
            replay.common.output_digest(b, chunk_bytes, threshold))
            for a, b in zip(current["references"], payload["outputs"]))
    source_controls = payload.get("execution_controls") or payload.get("event", {}).get("execution_controls")
    used_controls = configuration.get("execution_controls", {}).get("effective") or source_controls
    artifact = {
        "format": "hstu_output_stage_stress_failure", "format_version": 1,
        "operation": payload["operation"], "stage": "stress_output_failure",
        "args": (), "kwargs": pristine, "outputs": current["outputs"],
        "pristine_inputs": pristine, "inputs_at_failure": current["inputs"],
        "reference_outputs": payload["outputs"], "execution_controls": used_controls,
        "source_execution_controls": source_controls,
        "resident_references_at_failure": current["references"],
        "resident_reference_bytes_unchanged": reference_unchanged,
        "iteration": iteration, "checks": checks, "failure_locations": locations,
        "configuration": configuration,
        "input_integrity": {"unchanged": actual_hashes == pristine_hashes,
                            "pristine": pristine_hashes, "at_failure": actual_hashes},
        "storage_bytes": sum(old_storages.values()) + copied_bytes,
        "capture_timing": "Existing failing outputs and current inputs copied after scalar checks; no subsequent stage execution.",
        "reference_scope": "kwargs are pristine saved inputs; inspect inputs_at_failure before attributing an output mismatch to computation.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "iteration": iteration, "input_bytes_unchanged": actual_hashes == pristine_hashes,
            "resident_reference_bytes_unchanged": reference_unchanged,
            "failure_locations": locations}


def stress_loop(function, inputs, references, *, operation, repeats, controls,
                failure_callback, progress=None, chunk_elements=1 << 24,
                threshold=1e20, rtol=1e-2, atol=1e-12, strict_exact=False):
    """No warmup or recovery rerun: each execution is checked and counted."""
    torch = replay.common._torch()
    names = replay.OUTPUTS[operation]
    records = []
    exact_iterations = rounding_iterations = 0
    initial_versions = {name: tensor._version for name, tensor in inputs.items() if isinstance(tensor, torch.Tensor)}
    for iteration in range(1, repeats + 1):
        start = time.monotonic()
        with replay.restored_execution_controls(controls) as effective, torch.no_grad():
            outputs = function(**inputs)
        checks = compare_device_outputs(outputs, references, names, chunk_elements=chunk_elements,
                                         threshold=threshold, rtol=rtol, atol=atol)
        classification = classify(checks, strict_exact=strict_exact)
        changed_versions = [name for name, version in initial_versions.items() if inputs[name]._version != version]
        if changed_versions:
            classification["failed"] = True
        record = {"iteration": iteration, "checks": checks, **classification,
                  "changed_input_versions": changed_versions, "seconds_including_checks": time.monotonic() - start}
        records.append(record)
        if classification["failed"]:
            # outputs remain live until the callback has preserved this exact invocation.
            retained = failure_callback(iteration, inputs, outputs, record, effective)
            if progress:
                progress(record)
            return {"status": "FAIL", "iterations_completed": iteration, "exact_iterations": exact_iterations,
                    "rounding_deviation_iterations": rounding_iterations, "iterations": records,
                    "failure_capture": retained, "execution_controls": effective}
        if any(label == "finite_difference_within_tolerance" for label in classification["classifications"].values()):
            rounding_iterations += 1
        else:
            exact_iterations += 1
        del outputs
        if progress:
            progress(record)
    return {"status": "PASS", "iterations_completed": repeats, "exact_iterations": exact_iterations,
            "rounding_deviation_iterations": rounding_iterations, "iterations": records,
            "execution_controls": effective}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--failure-dump", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--allocator-config", default=None)
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument("--threshold", type=float, default=1e20)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--strict-exact", action="store_true")
    parser.add_argument("--chunk-elements", type=int, default=1 << 24)
    parser.add_argument("--copy-chunk-mib", type=int, default=64)
    parser.add_argument("--max-resident-gib", type=int, default=64)
    parser.add_argument("--max-failure-gib", type=int, default=96)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.repeats, args.chunk_elements, args.copy_chunk_mib, args.max_resident_gib,
           args.max_failure_gib, args.cpu_threads) < 1:
        parser.error("Repeat counts and resource limits must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.threshold, args.rtol, args.atol)) or args.threshold == 0:
        parser.error("Threshold must be finite positive; tolerances finite nonnegative")
    paths = [args.dump.resolve(), args.context.resolve(), args.report.resolve(), args.failure_dump.resolve()]
    if len(set(paths)) != len(paths) or args.report.exists() or args.failure_dump.exists():
        parser.error("Report/failure paths must be new and distinct from source files and each other")
    return args


def main(argv=None):
    args = arguments(argv)
    root = Path(__file__).resolve().parents[1]
    context = json.loads(args.context.read_text())
    environment = replay.restore_replay_environment(context["environment"], args.allocator_config)
    changed = []
    for name, expected in context["source_sha256"].items():
        if name.startswith("generative_recommenders/"):
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(name)
    if changed and not args.allow_source_change:
        raise ValueError(f"Captured model source differs: {changed}; intentional comparison needs --allow-source-change")
    sys.path.insert(0, str(root))
    torch = replay.common._torch()
    torch.set_num_threads(args.cpu_threads)
    import triton
    versions = {"python": platform.python_version(), "torch": str(torch.__version__), "hip": torch.version.hip,
                "triton": triton.__version__, "triton_distribution": importlib.metadata.version("triton")}
    for name in ("torch", "hip", "triton", "triton_distribution"):
        if name in context and context[name] != versions[name]:
            raise ValueError(f"Stack mismatch {name}: captured {context[name]!r}, current {versions[name]!r}")
    payload = torch.load(args.dump, map_location="cpu", weights_only=False, mmap=True)
    pristine = replay.bind_inputs(payload)
    if payload.get("outputs") is None:
        raise ValueError("Stress replay requires captured correct outputs")
    operation = payload["operation"]
    module = None
    if operation == replay.MM:
        def function(**values):
            return (torch.mm(values["dout"], values["output_weight"].t()),)
    else:
        module = importlib.import_module("generative_recommenders.ops.triton.triton_hstu_linear")
        function = getattr(module, replay.NORM)
    configuration = {"source_dump": str(args.dump), "context": str(args.context), "operation": operation,
                     "versions": versions, "restored_environment": context["environment"], **environment,
                     "changed_model_source": changed, "repeats": args.repeats,
                     "threshold": args.threshold, "rtol": args.rtol, "atol": args.atol,
                     "strict_exact": args.strict_exact, "chunk_elements": args.chunk_elements,
                     "max_resident_storage_bytes": args.max_resident_gib << 30,
                     "max_failure_storage_bytes": args.max_failure_gib << 30,
                     "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                         Path(__file__).resolve(), root / "generative_recommenders/ops/triton/triton_hstu_linear.py")},
                     "checking": "All output elements each call; one CPU scalar matrix readback. No full tensor CPU copies during successful iterations.",
                     "finite_difference_policy": "Within-tolerance deviations are reported and continue unless strict-exact; larger finite mismatches stop separately from extreme/nonfinite values."}
    report = {"status": "INITIALIZING", "configuration": configuration, "iterations": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.report.with_name(args.report.name + ".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.report)

    save()
    try:
        chunk_bytes = args.copy_chunk_mib << 20
        resident, copied = replay.common.restore_raw_tree({"inputs": pristine, "references": payload["outputs"]},
                                                         device="cuda:0", chunk_bytes=chunk_bytes,
                                                         max_bytes=args.max_resident_gib << 30)
        torch.cuda.synchronize()
        report["resident_storage_bytes"] = copied
        expected_inputs = replay.input_digests(pristine, chunk_bytes=chunk_bytes, threshold=args.threshold)
        restored_inputs = replay.input_digests(resident["inputs"], chunk_bytes=chunk_bytes, threshold=args.threshold)
        if expected_inputs != restored_inputs:
            raise RuntimeError("Restored resident inputs differ from pristine logical/raw bytes")
        report["pristine_input_digests"] = expected_inputs
        report["resident_input_bytes_verified"] = True
        report["captured_output_digests"] = [replay.common.output_digest(t, chunk_bytes, args.threshold) for t in payload["outputs"]]
        actual_reference = [replay.common.output_digest(t, chunk_bytes, args.threshold) for t in resident["references"]]
        if not all(replay.same_digest(a, b) for a, b in zip(actual_reference, report["captured_output_digests"])):
            raise RuntimeError("Restored GPU reference outputs differ from captured correct outputs")
        if any(item and (item["nonfinite_count"] or item["extreme_count"]) for item in actual_reference):
            raise ValueError("Captured reference outputs are nonfinite or above the anomaly threshold")
        report["resident_reference_bytes_verified"] = True
        report["status"] = "RUNNING"
        save()

        def failed(iteration, inputs, outputs, checks, controls):
            report.update(status="SAVING_FAILURE", failing_iteration=iteration, failing_checks=checks)
            report["compiled_kernels"] = norm_kernel_metadata(module)
            save()
            return save_failed_call(args.failure_dump, payload, pristine, inputs, outputs,
                                    iteration=iteration, checks=checks,
                                    configuration={**configuration, "execution_controls": controls,
                                                   "compiled_kernels": report["compiled_kernels"]},
                                    chunk_bytes=chunk_bytes, max_bytes=args.max_failure_gib << 30,
                                    threshold=args.threshold, rtol=args.rtol, atol=args.atol,
                                    gpu_references=resident["references"])

        def progress(record):
            report["iterations"].append(record)
            report["iterations_completed"] = record["iteration"]
            save()
            print(json.dumps({"iteration": record["iteration"], "failed": record["failed"],
                              "classifications": record["classifications"],
                              "seconds_including_checks": record["seconds_including_checks"]}), flush=True)

        result = stress_loop(function, resident["inputs"], resident["references"], operation=operation,
                             repeats=args.repeats, controls=payload.get("execution_controls") or payload.get("event", {}).get("execution_controls"),
                             failure_callback=failed, progress=progress, chunk_elements=args.chunk_elements,
                             threshold=args.threshold, rtol=args.rtol, atol=args.atol, strict_exact=args.strict_exact)
        report.update(result)
        report["compiled_kernels"] = norm_kernel_metadata(module)
        if report["status"] == "PASS":
            after = replay.input_digests(resident["inputs"], chunk_bytes=chunk_bytes, threshold=args.threshold)
            report["resident_inputs_unchanged_after_loop"] = after == expected_inputs
            final_references = [replay.common.output_digest(t, chunk_bytes, args.threshold) for t in resident["references"]]
            report["resident_references_unchanged_after_loop"] = all(replay.same_digest(a, b) for a, b in zip(
                final_references, report["captured_output_digests"]))
            if not report["resident_inputs_unchanged_after_loop"]:
                report["status"] = "INPUT_MUTATION_AFTER_OUTPUT_CHECKS"
            elif not report["resident_references_unchanged_after_loop"]:
                report["status"] = "REFERENCE_MUTATION_AFTER_OUTPUT_CHECKS"
        save()
    except Exception as error:
        report.update(status="ERROR", error={"type": type(error).__name__, "message": str(error)})
        save()
        raise
    print(json.dumps({"status": report["status"], "report": str(args.report),
                      "iterations_completed": report.get("iterations_completed"),
                      "failure_capture": report.get("failure_capture")}, default=str), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
