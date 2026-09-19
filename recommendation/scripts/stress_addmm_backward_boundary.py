#!/usr/bin/env python3
"""Resident full-shape projection backward stress; load trusted local captures only.

Calls the original public triton_addmm_bwd, checking all dx/dw/db elements after
every call. The optional zero-dz oracle expects numerical zero independently of
thresholds, tolerances and captured outputs. No warmup or failure rerun occurs.
"""
from __future__ import annotations

import argparse
import copy
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

import replay_backward_boundary as common
import replay_hstu_output_boundary as helpers
import stress_hstu_output_boundary as checks_helper


OPERATION = "triton_addmm_bwd"
NAMES = ("d_normed_x", "d_uvqk_weight", "d_uvqk_bias")
SOURCE_FILES = (
    "scripts/stress_addmm_backward_boundary.py",
    "scripts/replay_backward_boundary.py", "scripts/replay_hstu_output_boundary.py",
    "scripts/stress_hstu_output_boundary.py", "scripts/nan_backward_boundaries.py",
    "generative_recommenders/ops/triton/triton_addmm.py",
    "generative_recommenders/ops/utils.py", "generative_recommenders/common.py",
)
LIMITS = [
    "This resident experiment preserves the full captured shape, not the original training allocator history or surrounding workload.",
    "Full input/reference bytes are checked at initial restore and at final/failure observation. Intermediate calls have version checks, not full byte checks. Undetected transient changes remain possible.",
    "Original captured finite outputs are comparison baselines, not independent exact mathematical oracles.",
    "A completed failing call preserves all three existing outputs without rerunning the operation. An exception before the public API returns may leave outputs unavailable.",
    "A failure identifies this public boundary; dispatch order is db sum, dw GEMM, dx GEMM. It does not by itself identify a BLAS kernel, instruction or hardware cause.",
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def storage_bytes(tree):
    sizes = {(str(t.device), t.untyped_storage()._cdata): t.untyped_storage().nbytes()
             for t in common._tensors(tree)}
    return sum(sizes.values())


def digests(tree, *, chunk_bytes, threshold):
    return helpers.input_digests(tree, chunk_bytes=chunk_bytes, threshold=threshold)


def reference_map(values):
    return dict(zip(NAMES, values))


def validate_controls(controls):
    keys = {"autocast", "float32_matmul_precision", "allow_tf32",
            "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"}
    if not isinstance(controls, dict) or set(controls) != keys:
        raise ValueError("Complete operation-local execution controls are required")
    if controls["float32_matmul_precision"] not in ("highest", "high", "medium"):
        raise ValueError("Invalid float32 matmul precision")
    if any(type(controls[name]) is not bool for name in keys - {"autocast", "float32_matmul_precision"}):
        raise ValueError("Matmul controls must be booleans")
    if not isinstance(controls["autocast"], dict) or set(controls["autocast"]) != {"cpu", "cuda"}:
        raise ValueError("Both CPU and CUDA autocast controls are required")
    for value in controls["autocast"].values():
        if (not isinstance(value, dict) or set(value) != {"enabled", "dtype"}
                or type(value["enabled"]) is not bool
                or value["dtype"] not in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64")):
            raise ValueError("Invalid autocast controls")


def validate_tensor(value, *, name, shape=None, dtype=None, device="cpu"):
    torch = common._torch()
    if (not isinstance(value, torch.Tensor) or value.device.type != device
            or value.layout != torch.strided or value.is_quantized
            or value.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64)
            or value.is_conj() or value.is_neg()):
        raise ValueError(f"{name} must be a real strided tensor on {device}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape differs from {tuple(shape)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} dtype differs from {dtype}")
    # Conservative nonoverlap proof, accepting ordinary slices and transposes.
    span = 1
    for stride, size in sorted((s, n) for s, n in zip(value.stride(), value.shape) if n > 1):
        if stride < span:
            raise ValueError(f"{name} has overlapping or unsupported strides")
        span += (size - 1) * stride


def validate_capture(payload):
    if payload.get("operation") != OPERATION or payload.get("format_version") != 1:
        raise ValueError("Requires version-1 triton_addmm_bwd capture")
    common.require_pristine_input_snapshot(payload)
    event = payload.get("event") or {}
    observation = event.get("input_observation") or {}
    if (payload.get("stage") != "finite" or event.get("operation_executed") is not True
            or observation.get("source") != "pristine_pre_call_cpu_snapshot"
            or observation.get("operation_started") is not False
            or observation.get("source_devices_synchronized_before_copy") is not True):
        raise ValueError("Requires a finite capture with explicit pristine pre-call input evidence")
    if any(event.get(key) for key in ("bad_inputs", "bad_outputs", "extreme_inputs", "extreme_outputs")):
        raise ValueError("Capture event reports an input/output anomaly")
    values = common.bind_inputs(payload)
    if type(values["is_y_1d"]) is not bool:
        raise ValueError("is_y_1d must be a boolean")
    if min(values["x"].shape) <= 0 or values["dz"].shape[1] <= 0:
        raise ValueError("Positive full dimensions are required")
    for name in ("x", "w", "dz"):
        validate_tensor(values[name], name=name)
    references = payload.get("outputs")
    if not isinstance(references, (tuple, list)) or len(references) != 3:
        raise ValueError("All three original finite output references are required")
    shapes = (values["x"].shape, values["w"].shape,
              (values["dz"].shape[1],) if values["is_y_1d"] else values["dz"].shape)
    for name, value, shape in zip(NAMES, references, shapes):
        validate_tensor(value, name=name, shape=shape, dtype=values["x"].dtype)
    controls = payload.get("execution_controls")
    validate_controls(controls)
    if event.get("execution_controls") != controls:
        raise ValueError("Capture and event execution controls differ")
    return values, tuple(references), copy.deepcopy(controls)


def require_finite(summary, description):
    for name, item in summary["inputs"].items():
        if "scalar" not in item and (item["nonfinite_count"] or item["extreme_count"]):
            raise ValueError(f"{description}/{name} contains nonfinite or above-threshold values")


def prepare_capture(payload, *, zero_oracle=False, chunk_bytes=64 << 20,
                    threshold=1e6, max_clone_bytes=32 << 30):
    values, references, controls = validate_capture(payload)
    original = {"inputs": values, "references": references}
    original_inputs = digests(values, chunk_bytes=chunk_bytes, threshold=threshold)
    original_references = digests(reference_map(references), chunk_bytes=chunk_bytes, threshold=threshold)
    require_finite(original_inputs, "original inputs")
    require_finite(original_references, "original references")
    transformed = original
    transformation = {"mode": "original_pristine_inputs", "zero_oracle": False}
    if zero_oracle:
        if any(item["enabled"] for item in controls["autocast"].values()):
            raise ValueError("Zero oracle requires captured autocast disabled to exclude finite-to-infinite narrowing")
        # Raw x/w preservation requires that no zeroed view share their backing.
        protected = {values[name].untyped_storage()._cdata for name in ("x", "w")}
        if any(t.untyped_storage()._cdata in protected for t in (values["dz"], *references)):
            raise ValueError("Zeroed dz/references must not share a complete backing with protected x/w")
        transformed, copied = common.restore_raw_tree(original, device="cpu", chunk_bytes=chunk_bytes,
                                                      max_bytes=max_clone_bytes)
        with common._torch().no_grad():
            transformed["inputs"]["dz"].zero_()
            for value in transformed["references"]:
                value.zero_()
        for name in ("x", "w"):
            before = digests({name: values[name]}, chunk_bytes=chunk_bytes, threshold=threshold)
            after = digests({name: transformed["inputs"][name]}, chunk_bytes=chunk_bytes, threshold=threshold)
            if before != after:
                raise RuntimeError(f"Zero transformation altered protected logical/raw {name} bytes")
        zero_tensors = (transformed["inputs"]["dz"], *transformed["references"])
        if any(bool((chunk != 0).any()) for value in zero_tensors
               for chunk in common._logical_chunks(value, max(1, chunk_bytes // 16))):
            raise RuntimeError("Zero transformation did not produce numerical zero")
        transformation = {
            "mode": "zero_dz_oracle_v1", "zero_oracle": True, "cloned_storage_bytes": copied,
            "zeroed_logical_input": "dz", "protected_logical_and_raw_inputs": ["x", "w"],
            "zero_reference_names": list(NAMES), "unused_backing_bytes": "Retained by full CPU raw clone followed only by logical zero_ operations",
            "oracle": "For finite x/w and exact zero dz, all dx/dw/db must be numerical zero under the captured matmul/reduction controls. Signed zeros pass; every nonzero or nonfinite output fails independently of threshold/tolerance.",
            "original_outputs": "Retained separately as original captured finite references; never used as zero-oracle expected values",
        }
    prepared_inputs = digests(transformed["inputs"], chunk_bytes=chunk_bytes, threshold=threshold) if zero_oracle else original_inputs
    prepared_references = digests(reference_map(transformed["references"]), chunk_bytes=chunk_bytes, threshold=threshold) if zero_oracle else original_references
    require_finite(prepared_inputs, "prepared inputs")
    require_finite(prepared_references, "prepared references")
    return {"original": original, "prepared": transformed, "controls": controls,
            "transformation": transformation, "original_input_digests": original_inputs,
            "original_reference_digests": original_references,
            "prepared_input_digests": prepared_inputs, "prepared_reference_digests": prepared_references}


def resource_plan(prepared):
    resident = storage_bytes(prepared["prepared"])
    # Public API creates dense dx/dw/db; is_y_1d=False aliases dz for db.
    refs = prepared["prepared"]["references"]
    outputs = sum(t.numel() * t.element_size() for t in refs[:2])
    if prepared["prepared"]["inputs"]["is_y_1d"]:
        outputs += refs[2].numel() * refs[2].element_size()
    static = storage_bytes({"original": prepared["original"], "prepared": prepared["prepared"]})
    return {"resident_input_and_reference_storage_bytes": resident,
            "expected_output_storage_bytes": outputs,
            "resident_plus_outputs_bytes_before_workspace": resident + outputs,
            "original_and_prepared_CPU_storage_bytes": static,
            "expected_complete_failure_storage_bytes": static + resident + outputs}


def compare_outputs(outputs, references, *, zero_oracle=False, chunk_elements=1 << 24,
                    threshold=1e6, rtol=1e-2, atol=1e-12, strict_exact=False):
    torch = common._torch()
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("Public addmm backward must return three tensors")
    if zero_oracle:
        # Integer magnitude-bit checks cannot lose subnormal nonzeros to GPU
        # floating-point flush modes and do not depend on resident references.
        # One scalar-matrix readback covers all three outputs.
        totals = []
        for value, reference in zip(outputs, references):
            if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError("Zero-oracle output shape, dtype or device differs from expected")
            total = torch.zeros(7, device=value.device, dtype=torch.float64)
            for chunk in common._logical_chunks(value, chunk_elements):
                # Explicit contiguous allocation also normalizes singleton
                # strides, which contiguous() alone is allowed to retain.
                current = torch.empty(chunk.shape, dtype=chunk.dtype, device=chunk.device).copy_(chunk)
                observed = current.to(torch.float64 if current.dtype == torch.float64 else torch.float32)
                finite, magnitude = torch.isfinite(observed), observed.abs()
                integer_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}[current.element_size()]
                magnitude_bits = (1 << (8 * current.element_size() - 1)) - 1
                nonzero = (current.reshape(-1).view(integer_dtype) & magnitude_bits) != 0
                counts = torch.stack(((~finite).sum(), (finite & (magnitude > threshold)).sum(),
                                      (current.reshape(-1).view(torch.uint8) != 0).sum(),
                                      nonzero.sum(), (finite & (magnitude > atol)).sum())).to(torch.float64)
                maximum = torch.where(finite, magnitude, 0.).amax().to(torch.float64)
                total[:5] += counts
                total[5:] = torch.maximum(total[5:], maximum)
            totals.append(total)
        host = torch.stack(totals).cpu().tolist()
        result = {name: {field: int(row[index]) if index < 5 else common._number(row[index])
                         for index, field in enumerate(checks_helper.FIELDS)} for name, row in zip(NAMES, host)}
    else:
        result = checks_helper.compare_device_outputs(outputs, references, NAMES, chunk_elements=chunk_elements,
                                                  threshold=threshold, rtol=rtol, atol=atol)
    classification = checks_helper.classify(result, strict_exact=strict_exact)
    if zero_oracle:
        classification = {"failed": False, "classifications": {}}
        for name, item in result.items():
            nonzero = item["different_numeric_elements"]
            item["nonzero_elements_including_nonfinite"] = nonzero
            classification["classifications"][name] = "zero_oracle_nonzero_or_nonfinite" if nonzero else "numerical_zero"
            classification["failed"] |= bool(nonzero)
    return result, classification


def new_path(path):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if os.path.lexists(target) or os.path.lexists(temporary):
        raise FileExistsError(f"Target and temporary must be new: {target}")
    return target, temporary


def atomic_new_save(path, writer):
    """No-clobber publication, file/directory fsync, owned-temp cleanup."""
    target, temporary = new_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    owned = False
    try:
        with temporary.open("xb") as stream:
            owned = True
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link atomically rejects a target created by a race.
        os.link(temporary, target)
        temporary.unlink()
        descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if owned:
            temporary.unlink(missing_ok=True)
        raise


def save_failure(path, *, payload, prepared, resident, outputs, record, configuration,
                 initial_integrity, chunk_bytes, max_capture_bytes, threshold, rtol, atol,
                 strict_exact=False):
    from nan_backward_boundaries import _cpu_copy_tree
    torch = common._torch()
    new_path(path)
    before = {"original": prepared["original"], "prepared": prepared["prepared"]}
    current = {**resident, "outputs": tuple(outputs)}
    before_bytes, current_bytes = storage_bytes(before), storage_bytes(current)
    total = before_bytes + current_bytes
    if total > max_capture_bytes:
        raise ValueError(f"Complete failure needs {total} storage bytes; budget {max_capture_bytes}")
    started = time.time()
    post, copied = _cpu_copy_tree(current, chunk_bytes, max_capture_bytes - before_bytes)
    if copied != current_bytes:
        raise RuntimeError("Failure-copy storage accounting changed")
    current_inputs = digests(post["inputs"], chunk_bytes=chunk_bytes, threshold=threshold)
    current_references = digests(reference_map(post["references"]), chunk_bytes=chunk_bytes, threshold=threshold)
    output_digests = digests(reference_map(post["outputs"]), chunk_bytes=chunk_bytes, threshold=threshold)
    saved_checks, saved_classification = compare_outputs(
        post["outputs"], prepared["prepared"]["references"],
        zero_oracle=prepared["transformation"]["zero_oracle"],
        chunk_elements=max(1, chunk_bytes // 128), threshold=threshold, rtol=rtol, atol=atol,
        strict_exact=strict_exact)
    integrity = {"inputs_match_prepared_logical_and_raw_bytes": current_inputs == prepared["prepared_input_digests"],
                 "resident_references_match_prepared_logical_and_raw_bytes": current_references == prepared["prepared_reference_digests"],
                 "initial_restore": initial_integrity,
                 "current_input_digests": current_inputs, "current_reference_digests": current_references,
                 "scope": "Complete logical values, scalars, layout, alias groups and raw backings at failure; does not prove absence of transient mutation during the call"}
    locations = checks_helper.failure_locations(reference_map(post["outputs"]),
                                                reference_map(prepared["prepared"]["references"]),
                                                chunk_bytes=chunk_bytes, threshold=threshold,
                                                rtol=0 if prepared["transformation"]["zero_oracle"] else rtol,
                                                atol=0 if prepared["transformation"]["zero_oracle"] else atol)
    original_metadata = {k: copy.deepcopy(v) for k, v in payload.items() if k not in ("args", "kwargs", "outputs")}
    artifact = {
        "format": "addmm_backward_resident_stress_failure_v1", "format_version": 1,
        "operation": OPERATION, "iteration": record["iteration"], "observed_record": copy.deepcopy(record),
        "original_capture_metadata": original_metadata,
        "original_capture": prepared["original"], "prepared_pre_call": prepared["prepared"],
        "current_at_failure": post, "transformation": prepared["transformation"],
        "original_input_digests": prepared["original_input_digests"],
        "original_reference_digests": prepared["original_reference_digests"],
        "prepared_input_digests": prepared["prepared_input_digests"],
        "prepared_reference_digests": prepared["prepared_reference_digests"],
        "current_output_digests": output_digests, "input_and_reference_integrity": integrity,
        "saved_output_checks_against_prepared_reference_or_zero": saved_checks,
        "saved_output_classification": saved_classification, "failure_locations": locations,
        "execution_controls": record["execution_controls"], "configuration": copy.deepcopy(configuration),
        "dispatch_order": ["d_uvqk_bias: sum(dz, dim=0)", "d_uvqk_weight: x.T @ dz", "d_normed_x: dz @ w.T"],
        "storage_bytes": total, "capture_budget_bytes": max_capture_bytes,
        "snapshot_started_unix_time": started, "snapshot_completed_unix_time": time.time(),
        "capture_timing": "Existing first-failing call outputs, inputs and references copied together before any subsequent public API call; no rerun",
        "limits": LIMITS,
    }
    atomic_new_save(path, lambda stream: torch.save(artifact, stream))
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": file_hash(path),
            "storage_bytes": total, "iteration": record["iteration"],
            "inputs_match_prepared_logical_and_raw_bytes": integrity["inputs_match_prepared_logical_and_raw_bytes"],
            "resident_references_match_prepared_logical_and_raw_bytes": integrity["resident_references_match_prepared_logical_and_raw_bytes"],
            "saved_output_classification": saved_classification, "failure_locations": locations}


def stress_loop(function, resident, *, repeats, controls, zero_oracle, failure_callback,
                final_integrity, progress=None, chunk_elements=1 << 24,
                threshold=1e6, rtol=1e-2, atol=1e-12, strict_exact=False):
    torch = common._torch()
    if not 1 <= repeats <= 1000:
        raise ValueError("Repeat count must be between 1 and 1000")
    inputs, references = resident["inputs"], resident["references"]
    versions = {name: value._version for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    reference_versions = [value._version for value in references]
    records, exact, rounding = [], 0, 0
    for iteration in range(1, repeats + 1):
        start = time.monotonic()
        with helpers.restored_execution_controls(controls) as effective, torch.no_grad():
            outputs = function(**inputs)
        checked, classification = compare_outputs(outputs, references, zero_oracle=zero_oracle,
                                                  chunk_elements=chunk_elements, threshold=threshold,
                                                  rtol=rtol, atol=atol, strict_exact=strict_exact)
        changed_inputs = [name for name, version in versions.items() if inputs[name]._version != version]
        changed_references = [name for name, value, version in zip(NAMES, references, reference_versions) if value._version != version]
        record = {"iteration": iteration, "checks": checked, **classification,
                  "changed_input_versions": changed_inputs, "changed_reference_versions": changed_references,
                  "execution_controls": effective, "zero_oracle": zero_oracle}
        if changed_inputs or changed_references:
            record["failed"] = True
        if iteration == repeats and not record["failed"]:
            # Retain this call's outputs until the final full integrity audit ends.
            record["final_integrity"] = final_integrity(resident)
            if not all(record["final_integrity"].values()):
                record["failed"] = True
                record["failure_detection"] = "final full-byte audit; mutation start time unknown"
        record["seconds_including_checks"] = time.monotonic() - start
        records.append(record)
        if record["failed"]:
            retained = failure_callback(resident, outputs, record)
            if progress:
                progress(record)
            return {"status": "FAIL", "iterations_completed": iteration, "iterations": records,
                    "exact_iterations": exact, "rounding_deviation_iterations": rounding,
                    "failure_capture": retained}
        if any(item["different_bytes"] for item in checked.values()) and not zero_oracle:
            rounding += 1
        else:
            exact += 1
        del outputs
        if progress:
            progress(record)
    return {"status": "PASS", "iterations_completed": repeats, "iterations": records,
            "exact_iterations": exact, "rounding_deviation_iterations": rounding,
            "final_integrity": records[-1]["final_integrity"]}


def run_stress(payload, prepared, *, function, failure_path, configuration, repeats=1000,
               device="cuda:0", chunk_bytes=64 << 20, chunk_elements=1 << 24,
               max_resident_bytes=32 << 30, max_capture_bytes=64 << 30,
               threshold=1e6, rtol=1e-2, atol=1e-12, strict_exact=False, progress=None):
    new_path(failure_path)
    if not 1 <= repeats <= 1000:
        raise ValueError("Repeat count must be between 1 and 1000")
    plan = resource_plan(prepared)
    if plan["resident_plus_outputs_bytes_before_workspace"] > max_resident_bytes:
        raise ValueError("Resident inputs/references plus expected outputs exceed the budget")
    if plan["expected_complete_failure_storage_bytes"] > max_capture_bytes:
        raise ValueError("Complete failure capture exceeds budget before dispatch")
    resident, copied = common.restore_raw_tree(prepared["prepared"], device=device,
                                              chunk_bytes=chunk_bytes, max_bytes=max_resident_bytes)

    def integrity(current):
        return {"inputs_match_prepared_logical_and_raw_bytes": digests(current["inputs"], chunk_bytes=chunk_bytes, threshold=threshold) == prepared["prepared_input_digests"],
                "references_match_prepared_logical_and_raw_bytes": digests(reference_map(current["references"]), chunk_bytes=chunk_bytes, threshold=threshold) == prepared["prepared_reference_digests"]}

    initial = integrity(resident)
    if not all(initial.values()):
        raise RuntimeError("Initial resident logical/raw bytes differ from prepared evidence")

    def failed(current, outputs, record):
        return save_failure(failure_path, payload=payload, prepared=prepared, resident=current,
                            outputs=outputs, record=record, configuration=configuration,
                            initial_integrity=initial, chunk_bytes=chunk_bytes,
                            max_capture_bytes=max_capture_bytes, threshold=threshold, rtol=rtol,
                            atol=atol, strict_exact=strict_exact)

    result = stress_loop(function, resident, repeats=repeats, controls=prepared["controls"],
                         zero_oracle=prepared["transformation"]["zero_oracle"], failure_callback=failed,
                         final_integrity=integrity, progress=progress, chunk_elements=chunk_elements,
                         threshold=threshold, rtol=rtol, atol=atol, strict_exact=strict_exact)
    return {**result, "resource_plan": plan, "resident_storage_bytes": copied, "initial_integrity": initial}


def loaded_blas_libraries():
    paths = set()
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/") and any(name in fields[5].lower() for name in ("hipblas", "rocblas", "tensile")):
                paths.add(fields[5])
    return [{"path": path, "sha256": file_hash(path) if Path(path).is_file() else None} for path in sorted(paths)]


def restore_environment(context):
    captured = context["environment"]
    previous_alias = os.environ.get("PYTORCH_ALLOC_CONF")
    if "PYTORCH_ALLOC_CONF" not in captured:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
    restored = helpers.restore_replay_environment(captured)
    mandatory = {"AMDGCN_USE_BUFFER_OPS": "0", "TRITON_FULL_AUTOTUNE": "0", "TRITON_ALLOW_PIPELINING": "0"}
    os.environ.update(mandatory)
    return {**restored, "mandatory_environment_overrides": mandatory,
            "mandatory_environment_differences": {name: {"captured": captured.get(name), "effective": value}
                                                   for name, value in mandatory.items() if captured.get(name) != value},
            "inherited_modern_allocator_alias_removed": previous_alias if "PYTORCH_ALLOC_CONF" not in captured else None,
            "effective_HIPBLASLT_TENSILE_LIBPATH": os.environ.get("HIPBLASLT_TENSILE_LIBPATH")}


def blas_preference():
    function = getattr(common._torch().backends.cuda, "preferred_blas_library", None)
    return {"observed_preference": str(function()) if function is not None else None,
            "scope": "PyTorch selection preference; not proof of an executed BLAS kernel"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--failure-dump", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--expect-rows", type=int, default=None)
    parser.add_argument("--zero-oracle", action="store_true")
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument("--threshold", type=float, default=1e6)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--strict-exact", action="store_true")
    parser.add_argument("--chunk-elements", type=int, default=1 << 24)
    parser.add_argument("--copy-chunk-mib", type=int, default=64)
    parser.add_argument("--max-resident-gib", type=int, default=32)
    parser.add_argument("--max-failure-gib", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args(argv)
    if (not 1 <= args.repeats <= 1000 or min(args.chunk_elements, args.copy_chunk_mib,
            args.max_resident_gib, args.max_failure_gib, args.cpu_threads) < 1
            or (args.expect_rows is not None and args.expect_rows < 1)):
        parser.error("Positive resource limits required; repeats must be 1..1000")
    if any(not math.isfinite(v) or v < 0 for v in (args.threshold, args.rtol, args.atol)) or args.threshold == 0:
        parser.error("Threshold must be finite positive; tolerances finite nonnegative")
    paths = [args.dump, args.context, args.report, args.report.with_name(args.report.name + ".tmp"),
             args.failure_dump, args.failure_dump.with_name(args.failure_dump.name + ".tmp")]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("Source, report, failure and temporary paths must all be distinct")
    try:
        new_path(args.report)
        new_path(args.failure_dump)
    except FileExistsError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = arguments(argv)
    root = Path(__file__).resolve().parents[1]
    context = json.loads(args.context.read_text())
    # Mandatory controls are applied after context restoration and before imports.
    restored_environment = restore_environment(context)
    torch = common._torch()
    torch.set_num_threads(args.cpu_threads)
    import triton
    versions = {"python": platform.python_version(), "torch": str(torch.__version__), "hip": torch.version.hip,
                "triton": triton.__version__, "triton_distribution": importlib.metadata.version("triton")}
    for name in ("torch", "hip", "triton", "triton_distribution"):
        if context.get(name) != versions[name]:
            raise ValueError(f"Captured/current stack differs for {name}")
    changed = []
    for name, expected in context["source_sha256"].items():
        if name.startswith("generative_recommenders/"):
            actual = file_hash(root / name) if (root / name).is_file() else None
            if actual != expected:
                changed.append({"path": name, "captured": expected, "current": actual})
    relevant = "generative_recommenders/ops/triton/triton_addmm.py"
    if relevant not in context["source_sha256"] or any(item["path"] == relevant for item in changed):
        raise ValueError("Relevant addmm source must match captured context exactly")
    if changed and not args.allow_source_change:
        raise ValueError(f"Whole-model source changes require --allow-source-change: {changed}")
    sources = {name: file_hash(root / name) for name in SOURCE_FILES}
    configuration = {"source_dump": {"path": str(args.dump), "bytes": args.dump.stat().st_size, "sha256": file_hash(args.dump)},
                     "context": {"path": str(args.context), "sha256": file_hash(args.context)},
                     "versions": versions, "source_sha256_before": sources, "whole_model_source_changes": changed,
                     "captured_environment": context["environment"], **restored_environment,
                     "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "public_api": "generative_recommenders.ops.triton.triton_addmm.triton_addmm_bwd",
                     "dispatch_order": ["db=sum(dz,dim=0)", "dw=x.T@dz", "dx=dz@w.T"],
                     "limits": LIMITS}
    report = {"status": "PREPARING", "configuration": configuration, "iterations": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_new_save(args.report, lambda stream: stream.write((json.dumps(report, indent=2) + "\n").encode()))

    def save_report():
        temporary = args.report.with_name(args.report.name + ".tmp")
        owned = False
        try:
            with temporary.open("x") as stream:
                owned = True
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.report)
            descriptor = os.open(args.report.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            if owned:
                temporary.unlink(missing_ok=True)
            raise

    try:
        payload = torch.load(args.dump, map_location="cpu", weights_only=False, mmap=True)
        prepared = prepare_capture(payload, zero_oracle=args.zero_oracle, chunk_bytes=args.copy_chunk_mib << 20,
                                   threshold=args.threshold, max_clone_bytes=args.max_resident_gib << 30)
        rows = prepared["prepared"]["inputs"]["x"].shape[0]
        if args.expect_rows is not None and rows != args.expect_rows:
            raise ValueError(f"Expected full M={args.expect_rows}; got {rows}")
        report.update(status="READY", shape_M=rows, transformation=prepared["transformation"],
                      resource_plan=resource_plan(prepared),
                      original_input_digests=prepared["original_input_digests"],
                      original_reference_digests=prepared["original_reference_digests"],
                      prepared_input_digests=prepared["prepared_input_digests"],
                      prepared_reference_digests=prepared["prepared_reference_digests"],
                      captured_execution_controls=prepared["controls"])
        save_report()
        sys.path.insert(0, str(root))
        module = importlib.import_module("generative_recommenders.ops.triton.triton_addmm")
        configuration["device"] = {"name": torch.cuda.get_device_name(0), "properties": str(torch.cuda.get_device_properties(0))}
        configuration["blas_selection_preference_before"] = blas_preference()
        configuration["blas_libraries_before"] = loaded_blas_libraries()

        def progress(record):
            report["iterations"].append(record)
            report.update(status="FAIL" if record["failed"] else "RUNNING", iterations_completed=record["iteration"])
            save_report()
            print(json.dumps({"iteration": record["iteration"], "failed": record["failed"],
                              "classifications": record["classifications"], "seconds": record["seconds_including_checks"]}), flush=True)

        result = run_stress(payload, prepared, function=getattr(module, OPERATION), failure_path=args.failure_dump,
                            configuration=configuration, repeats=args.repeats, chunk_bytes=args.copy_chunk_mib << 20,
                            chunk_elements=args.chunk_elements, max_resident_bytes=args.max_resident_gib << 30,
                            max_capture_bytes=args.max_failure_gib << 30, threshold=args.threshold,
                            rtol=args.rtol, atol=args.atol, strict_exact=args.strict_exact, progress=progress)
        report.update(result)
        report["source_sha256_after"] = {name: file_hash(root / name) for name in SOURCE_FILES}
        report["source_files_unchanged_during_run"] = report["source_sha256_after"] == sources
        report["blas_libraries_after"] = loaded_blas_libraries()
        report["blas_selection_preference_after"] = blas_preference()
        if report["status"] == "PASS" and not report["source_files_unchanged_during_run"]:
            report["status"] = "SOURCE_CHANGED_DURING_RUN"
        save_report()
    except BaseException as error:
        report.update(status="ERROR", error={"type": type(error).__name__, "message": str(error)},
                      failure_artifact_exists=args.failure_dump.is_file())
        save_report()
        raise
    print(json.dumps({"status": report["status"], "report": str(args.report),
                      "iterations_completed": report.get("iterations_completed"),
                      "failure_capture": report.get("failure_capture")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
