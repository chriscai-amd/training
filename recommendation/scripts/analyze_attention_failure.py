#!/usr/bin/env python3
"""CPU analysis of a saved attention history-DQ violation and its propagation.

This isolates only the erroneous DQ contribution from history rows with exactly
zero incoming attention dOut. It propagates that contribution through W_Q.T and
input layer norm using saved weights/statistics. It is not a full backward replay
or proof that this failure caused an earlier training NaN. Load trusted dumps only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import replay_backward_boundary as common
from replay_hstu_output_boundary import ReferenceSummary, input_digests
from repro_hstu_attention_capture import PREFIX_FORMAT, PREFIX_VERSION, cpu_view, select_prefix


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_cpu(tensor, name):
    torch = common._torch()
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ValueError(f"{name} must be a CPU strided tensor")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(failure, prefix, *, prefix_sha256, chunk_bytes, input_abs_limit):
    """Validate provenance, launch dimensions, pristine inputs, and zero oracle."""
    torch = common._torch()
    _require(failure.get("format") == "hstu_attention_backward_failure" and failure.get("format_version") == 1,
             "Expected hstu_attention_backward_failure format_version=1")
    _require(prefix.get("format") == PREFIX_FORMAT and prefix.get("format_version") == PREFIX_VERSION,
             "Expected the original compact attention prefix")
    configuration = failure["configuration"]
    expected_hash = configuration.get("input_artifact_sha256")
    _require(bool(expected_hash) and prefix_sha256 == expected_hash,
             "Prefix file SHA256 does not match the failure's recorded input artifact")
    selected, dimensions = select_prefix(torch, prefix, None)
    replay = selected["payload"]["replay"]
    ctx = replay["ctx"]
    _require(configuration.get("context") == ctx, "Failure launch context differs from the original prefix context")
    required = ("recompute_normed_x_in_backward", "recompute_uvqk_in_backward", "has_multiple_targets", "sort_by_length", "uvqk_bias_1d")
    _require(all(ctx.get(key) is True for key in required), "Unsupported context flags for the saved preprocess path")
    _require(ctx.get("num_softmax_heads") == 0 and ctx.get("enable_tma") is False,
             "Only the saved non-TMA SiLU attention path with zero softmax heads is supported")
    for key in ("num_heads", "hidden_dim", "attn_dim", "max_seq_len", "norm_BLOCK_D"):
        _require(isinstance(ctx.get(key), int) and ctx[key] > 0, f"Invalid context dimension {key}")
    _require(math.isfinite(ctx["attn_alpha"]) and math.isfinite(ctx["norm_eps"]) and ctx["norm_eps"] >= 0,
             "Invalid attention scale or norm epsilon")
    _require(ctx.get("max_attn_len", -1) >= 0 and ctx.get("contextual_seq_len", -1) >= 0,
             "Invalid attention mask context")
    runtime = replay.get("runtime_state")
    _require(runtime is not None and configuration.get("captured_runtime_state") == runtime,
             "Captured runtime state differs from the prefix or is absent")
    common_state = runtime.get("common", {})
    _require(configuration.get("use_runtime_max_seq_len") == common_state.get("USE_RUNTIME_MAX_SEQ_LEN"),
             "Effective runtime length mode differs from captured state")
    _require(configuration.get("static_max_seq_lens") == common_state.get("STATIC_MAX_SEQ_LENS"),
             "Effective static length buckets differ from captured state")
    _require(configuration.get("pre_hook") == "_bwd_pre_hook", "Original attention DQ-zero pre-hook is not recorded")
    if not configuration["use_runtime_max_seq_len"]:
        buckets = configuration["static_max_seq_lens"]
        _require(buckets and max(buckets) >= ctx["max_seq_len"], "Static buckets do not cover the attention normalization length")
    saved = [cpu_view(torch, selected, descriptor) for descriptor in replay["saved_tensors"]]
    x, gamma, beta, mean, rstd, weight, offsets, targets, bias, sort = saved
    n, d = x.shape
    h, a, v = ctx["num_heads"], ctx["attn_dim"], ctx["hidden_dim"]
    widths = [h * v, h * v, h * a, h * a]
    _require(ctx["norm_BLOCK_D"] >= d, "Norm tile is smaller than the input feature width")
    _require(tuple(weight.shape) == (d, sum(widths)) and tuple(bias.shape) == (sum(widths),), "Invalid saved UVQK projection shapes")
    _require(gamma.shape == beta.shape == (d,) and mean.shape == rstd.shape == (n,), "Invalid saved normalization shapes")
    before, after, outputs = failure["pristine_inputs"], failure["inputs_at_failure"], failure["outputs"]
    shapes = {"q": (n, h, a), "k": (n, h, a), "v": (n, h, v), "dout": (n, h, v),
              "seq_offsets": tuple(offsets.shape), "num_targets": tuple(targets.shape),
              "sort_by_length_indices": tuple(sort.shape)}
    for collection_name, collection in (("pristine_inputs", before), ("inputs_at_failure", after)):
        _require(set(collection) == set(shapes), f"Unexpected {collection_name} tensor fields")
        for name, shape in shapes.items():
            _tensor_cpu(collection[name], f"{collection_name}/{name}")
            _require(tuple(collection[name].shape) == shape, f"Invalid {collection_name}/{name} shape")
            if name in ("q", "k", "v", "dout"):
                _require(collection[name].is_floating_point(), f"{collection_name}/{name} must be floating point")
            else:
                _require(collection[name].dtype == torch.int64, f"{collection_name}/{name} must be int64")
    _require(set(outputs) == {"dq", "dk", "dv"}, "Expected dq, dk, dv outputs")
    for name, shape in (("dq", (n, h, a)), ("dk", (n, h, a)), ("dv", (n, h, v))):
        _tensor_cpu(outputs[name], name)
        _require(tuple(outputs[name].shape) == shape and outputs[name].is_floating_point(), f"Invalid {name} output")
        input_name = {"dq": "q", "dk": "k", "dv": "v"}[name]
        _require(outputs[name].dtype == before[input_name].dtype, f"{name} dtype differs from its corresponding input")
    for name, original in (("seq_offsets", offsets), ("num_targets", targets)):
        _require(torch.equal(before[name], original), f"Failure {name} differs from original prefix")
    actual_sort = before["sort_by_length_indices"]
    _require(torch.equal(torch.sort(actual_sort).values, torch.arange(offsets.numel() - 1)),
             "Failure sequence sort is not a permutation")
    sorted_lengths = (offsets[1:] - offsets[:-1])[actual_sort]
    _require(bool((sorted_lengths[1:] <= sorted_lengths[:-1]).all()), "Failure sequences are not sorted by descending length")
    original_dout = cpu_view(torch, selected, replay["args"][1])
    _require(torch.equal(before["dout"], original_dout), "Pristine attention dOut differs from original prefix")
    pristine_digests = input_digests(before, chunk_bytes=chunk_bytes, threshold=input_abs_limit)
    current_digests = input_digests(after, chunk_bytes=chunk_bytes, threshold=input_abs_limit)
    _require(pristine_digests == current_digests,
             "Attention inputs changed logically or in raw storage; the clean-input zero oracle cannot attribute this failure")
    bounds = {}
    for name in ("q", "k", "v", "dout"):
        summary = pristine_digests["inputs"][name]
        _require(summary["nonfinite_count"] == 0 and summary["extreme_count"] == 0,
                 f"Input {name} is nonfinite or exceeds the conservative input magnitude bound")
        bounds[name] = summary["max_abs_finite"]
    score_bound = a * bounds["q"] * bounds["k"] * abs(ctx["attn_alpha"])
    ds_bound = v * bounds["dout"] * bounds["v"]
    _require(math.isfinite(score_bound) and math.isfinite(ds_bound) and max(score_bound, ds_bound) < torch.finfo(torch.float32).max / 1024,
             "Conservative attention arithmetic bounds do not exclude overflow")
    history = torch.ones(n, dtype=torch.bool)
    history[offsets[1:] - 1] = False
    bad_history_dout = 0
    for start in range(0, n, 4096):
        chunk = before["dout"][start:start + 4096]
        bad_history_dout += int(((chunk != 0).flatten(1).any(1) & history[start:start + len(chunk)]).sum())
    _require(bad_history_dout == 0, "History dOut is not exactly zero: history DQ need not be zero")
    saved_summaries = {}
    for name, tensor in zip(("x", "norm_weight", "norm_bias", "mean", "rstd", "uvqk_weight"), saved[:6]):
        summary = common.output_digest(tensor, chunk_bytes, input_abs_limit)
        _require(summary["nonfinite_count"] == 0, f"Saved {name} is nonfinite")
        saved_summaries[name] = summary
    _require(bool((rstd > 0).all()), "Saved inverse standard deviations must be positive")
    return {
        "ctx": ctx, "saved": saved, "history": history, "widths": widths,
        "report": {"prefix_sha256_verified": True, "context_matches_prefix": True,
                   "runtime_state_matches_prefix": True, "original_pre_hook_recorded": True,
                   "input_logical_and_raw_bytes_unchanged": True, "input_digests": pristine_digests,
                   "sort_matches_prefix": torch.equal(actual_sort, sort),
                   "sort_validation": "descending lengths and a complete permutation; equal-length ties may differ after GPU re-sorting",
                   "input_abs_limit": input_abs_limit, "input_max_abs": bounds,
                   "conservative_qk_score_bound": score_bound, "conservative_dout_v_dot_bound": ds_bound,
                   "history_rows": int(history.sum()), "history_rows_with_nonzero_dout": bad_history_dout,
                   "one_target_per_sequence": True, "dimensions": dimensions,
                   "saved_consumer_tensor_summaries": saved_summaries,
                   "scope": "The actual pristine/failure inputs are equal and bounded. Each history query contributes only to its own zero-gradient output, hence its mathematical DQ is zero. This does not validate nonzero target/K/V gradients."},
    }


def analyze(failure, prefix, *, prefix_sha256, chunk_bytes=64 << 20, row_chunk=128,
            max_propagation_rows=65536, sample_rows=32, max_matrix_bytes=64 << 20,
            input_abs_limit=1e6):
    torch = common._torch()
    _require(min(chunk_bytes, row_chunk, sample_rows, max_matrix_bytes) > 0 and max_propagation_rows >= 0,
             "Invalid analysis limits")
    _require(math.isfinite(input_abs_limit) and input_abs_limit > 0, "Invalid input magnitude limit")
    started = time.monotonic()
    validated = validate(failure, prefix, prefix_sha256=prefix_sha256,
                         chunk_bytes=chunk_bytes, input_abs_limit=input_abs_limit)
    ctx, saved, history, widths = (validated[key] for key in ("ctx", "saved", "history", "widths"))
    x, gamma, beta, mean, rstd, weight = saved[:6]
    dq = failure["outputs"]["dq"]
    n, d = x.shape
    qwidth = widths[2]
    qstart = sum(widths[:2])
    row_bad = torch.zeros(n, dtype=torch.bool)
    row_max = torch.zeros(n, dtype=torch.float64)
    violations = nonfinite = above_extreme = 0
    values_sample = []
    for start in range(0, n, 4096):
        value = dq[start:start + 4096].to(torch.float64)
        selected = history[start:start + len(value), None, None]
        bad = (value != 0) & selected
        finite = torch.isfinite(value)
        violations += int(bad.sum())
        nonfinite += int((bad & ~finite).sum())
        above_extreme += int((bad & finite & (value.abs() > 1e20)).sum())
        row_bad[start:start + len(value)] = bad.flatten(1).any(1)
        row_max[start:start + len(value)] = torch.where(bad & finite, value.abs(), 0.).flatten(1).amax(1)
        if len(values_sample) < 64:
            for index in torch.nonzero(bad)[:64 - len(values_sample)].tolist():
                values_sample.append({"index": [start + index[0], *index[1:]],
                                      "actual": str(value[tuple(index)].item()), "expected": 0.0})
    ids = torch.nonzero(row_bad).flatten()
    dq_max = float(row_max.max())
    dout_max = validated["report"]["input_max_abs"]["dout"]
    row_details = []
    for row in ids.tolist():
        sequence = int(torch.searchsorted(saved[6][1:], row, right=True))
        row_details.append({"row": row, "sequence": sequence,
                            "sequence_position": row - int(saved[6][sequence]),
                            "max_abs_finite": float(row_max[row])})
    report = {"format": "attention_history_dq_propagation_analysis_v1", "iteration": failure["iteration"],
              "validation": validated["report"], "compiled_kernels": failure.get("compiled_kernels"),
              "captured_kernel_source_sha256": failure["configuration"].get("source_sha256"),
              "source_provenance": failure["configuration"].get("source_provenance"),
              "history_dq": {"violating_elements": violations, "violating_row_count": ids.numel(),
                             "violating_rows": ids.tolist(), "nonfinite_elements": nonfinite,
                             "row_details": row_details,
                             "finite_elements_above_1e20": above_extreme,
                             "max_abs_finite": dq_max, "actual_value_samples": values_sample,
                             "dq_error_to_global_dout_max_ratio": dq_max / dout_max if dout_max else None,
                             "ratio_scope": "Scale comparison with the global dOut maximum; mathematical history DQ is zero."},
              "interpretation": [
                  "This isolates the erroneous history-DQ contribution; valid target DQ and all DK/DV/DU contributions are excluded.",
                  "The consumer calculations below are CPU counterfactual contributions, not actual downstream GPU output captures.",
                  "Finite corruption does not by itself establish the producer of any earlier training NaN.",
                  "The FP64 path and the path with BF16 casting after each stage are separate numerical models; FP32 GPU arithmetic and reduction order are not reproduced.",
                  "Casting an isolated contribution is not necessarily equal to the difference between two rounded full gradients.",
                  "Delta UVQK weight gradients use a reconstruction of normed_x from saved x/mean/rstd/weight/bias, not an original normed_x tensor.",
                  "Original mean/rstd are retained for layer-norm derivatives; statistics are never recomputed."]}
    retained = {"all_violating_rows": ids.clone()}
    if not ids.numel():
        report.update(propagation={"status": "no history-DQ violation"}, seconds=time.monotonic() - started)
        return report, retained
    _require(d * qwidth * 8 * 3 <= max_matrix_bytes,
             "Q-block weight-gradient matrices exceed --max-matrix-mib")
    chosen = ids if not max_propagation_rows else ids[:max_propagation_rows]
    full = chosen.numel() == ids.numel()
    top = chosen[torch.argsort(row_max[chosen], descending=True)[:sample_rows]]
    sample_set = set(top.tolist())
    qweight = weight[:, qstart:qstart + qwidth].to(torch.float64)
    _require(max(d, qwidth) * 8 * 24 <= chunk_bytes, "One full-feature row exceeds the analysis chunk budget")
    block = min(row_chunk, max(1, chunk_bytes // (max(d, qwidth) * 8 * 24)))
    summaries = {name: ReferenceSummary(torch.bfloat16, scope="isolated history-DQ contribution", observed_available=False)
                 for name in ("delta_d_normed_x_fp64", "delta_d_input_x_fp64", "delta_d_input_x_from_bf16_projection")}
    cast_maxima = dict.fromkeys(summaries, 0.0)
    dw = torch.zeros((d, qwidth), dtype=torch.float64)
    dw_rounded_normed = torch.zeros_like(dw)
    db = torch.zeros(qwidth, dtype=torch.float64)
    dgamma = torch.zeros(d, dtype=torch.float64)
    dbeta = torch.zeros(d, dtype=torch.float64)
    samples = {name: [] for name in ("rows", "dq_actual", "delta_d_normed_x_fp64", "delta_d_normed_x_bf16",
               "delta_d_input_x_fp64", "delta_d_input_x_from_bf16_projection_fp64", "delta_d_input_x_stage_bf16")}
    with torch.no_grad():
        for start in range(0, chosen.numel(), block):
            rows = chosen[start:start + block]
            error = dq[rows].reshape(-1, qwidth).to(torch.float64)
            normalized_grad = error @ qweight.T
            normalized_grad_bf16 = normalized_grad.to(torch.bfloat16)
            xx = x[rows].to(torch.float64)
            rs = rstd[rows].to(torch.float64)[:, None]
            normalized = (xx - mean[rows].to(torch.float64)[:, None]) * rs
            normed_x = normalized * gamma.to(torch.float64) + beta.to(torch.float64)

            def ln_derivative(incoming):
                weighted = incoming * gamma.to(torch.float64)
                return (weighted - normalized * (normalized * weighted).mean(1, keepdim=True) - weighted.mean(1, keepdim=True)) * rs

            dx = ln_derivative(normalized_grad)
            dx_stage = ln_derivative(normalized_grad_bf16.to(torch.float64))
            for name, value in (("delta_d_normed_x_fp64", normalized_grad),
                                ("delta_d_input_x_fp64", dx),
                                ("delta_d_input_x_from_bf16_projection", dx_stage)):
                summaries[name].add(value, coordinate=lambda ij: [int(rows[ij[0]]), ij[1]])
                cast_value = value.to(torch.bfloat16).to(torch.float64)
                finite = torch.isfinite(cast_value)
                if bool(finite.any()):
                    cast_maxima[name] = max(cast_maxima[name], float(cast_value[finite].abs().max()))
            dw += normed_x.T @ error
            dw_rounded_normed += normed_x.to(torch.bfloat16).to(torch.float64).T @ error
            db += error.sum(0)
            dgamma += (normalized_grad_bf16.to(torch.float64) * normalized).sum(0)
            dbeta += normalized_grad_bf16.to(torch.float64).sum(0)
            selected = torch.tensor([i for i, row in enumerate(rows.tolist()) if row in sample_set], dtype=torch.int64)
            if selected.numel():
                for name, value in (("rows", rows), ("dq_actual", error),
                                    ("delta_d_normed_x_fp64", normalized_grad), ("delta_d_normed_x_bf16", normalized_grad_bf16),
                                    ("delta_d_input_x_fp64", dx), ("delta_d_input_x_from_bf16_projection_fp64", dx_stage),
                                    ("delta_d_input_x_stage_bf16", dx_stage.to(torch.bfloat16))):
                    samples[name].append(value[selected].clone())
    retained.update({name: torch.cat(value) for name, value in samples.items()})
    reductions = {"delta_uvqk_weight_qblock_fp64": dw,
                  "delta_uvqk_weight_qblock_from_bf16_normed_x_fp64": dw_rounded_normed,
                  "delta_uvqk_bias_qblock_fp64": db,
                  "delta_norm_weight_from_bf16_projection_fp64": dgamma,
                  "delta_norm_bias_from_bf16_projection_fp64": dbeta}
    reduction_summaries = {}
    for name, value in reductions.items():
        summary = ReferenceSummary(torch.bfloat16, scope="all violating history rows" if full else "partial violating history rows", observed_available=False)
        summary.add(value)
        reduction_summaries[name] = summary.result()
        cast_value = value.to(torch.bfloat16).to(torch.float64)
        finite = torch.isfinite(cast_value)
        reduction_summaries[name]["bf16_cast_max_abs_finite"] = float(cast_value[finite].abs().max()) if bool(finite.any()) else 0.0
        retained[name] = value
        retained[name.removesuffix("_fp64") + "_bf16"] = value.to(torch.bfloat16)
    report["propagation"] = {"status": "computed", "rows_propagated": chosen.numel(),
                             "all_violating_rows_included": full, "row_cap": max_propagation_rows,
                             "sample_rows": retained["rows"].tolist(), "q_column_range": [qstart, qstart + qwidth],
                             "uvqk_widths_u_v_q_k": widths, "other_uvqk_weight_bias_blocks": "zero for this isolated Q-only contribution",
                             "stage_references": {name: {**value.result(), "bf16_cast_max_abs_finite": cast_maxima[name]}
                                                  for name, value in summaries.items()},
                             "parameter_delta_references": reduction_summaries}
    report["seconds"] = time.monotonic() - started
    return report, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("failure", type=Path)
    parser.add_argument("--prefix", type=Path, help="Original prefix, overriding its stored path; SHA256 must match")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tensor-report", type=Path, help="CPU tensor sidecar; defaults alongside JSON")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--max-propagation-rows", type=int, default=65536, help="0 means all, still processed in bounded tiles")
    parser.add_argument("--sample-rows", type=int, default=32)
    parser.add_argument("--max-matrix-mib", type=int, default=64)
    parser.add_argument("--input-abs-limit", type=float, default=1e6)
    args = parser.parse_args()
    if min(args.cpu_threads, args.chunk_mib, args.row_chunk, args.sample_rows, args.max_matrix_mib) < 1 or args.max_propagation_rows < 0:
        parser.error("Analysis limits must be positive; --max-propagation-rows may be zero")
    torch = common._torch()
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_initialized():
        raise RuntimeError("This analyzer must start without CUDA initialized")
    rng = torch.get_rng_state().clone()
    failure = torch.load(args.failure, weights_only=False, map_location="cpu", mmap=True)
    prefix_path = args.prefix or Path(failure["configuration"]["input_artifact"])
    prefix_hash = file_sha256(prefix_path)
    prefix = torch.load(prefix_path, weights_only=False, map_location="cpu", mmap=True)
    report, tensors = analyze(failure, prefix, prefix_sha256=prefix_hash,
                              chunk_bytes=args.chunk_mib << 20, row_chunk=args.row_chunk,
                              max_propagation_rows=args.max_propagation_rows, sample_rows=args.sample_rows,
                              max_matrix_bytes=args.max_matrix_mib << 20, input_abs_limit=args.input_abs_limit)
    report_path = args.report or args.failure.with_suffix(".propagation.json")
    tensor_path = args.tensor_report or report_path.with_suffix(".tensors.pt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    _require(report_path.resolve() not in (args.failure.resolve(), prefix_path.resolve()) and
             tensor_path.resolve() not in (args.failure.resolve(), prefix_path.resolve(), report_path.resolve()),
             "Output files must be distinct from source artifacts and each other")
    torch.save({"format": "attention_history_dq_propagation_tensors_v1", "tensors": tensors}, tensor_path)
    report.update(failure_path=str(args.failure), failure_sha256=file_sha256(args.failure),
                  prefix_path=str(prefix_path), prefix_sha256=prefix_hash,
                  tensor_report=str(tensor_path), tensor_report_sha256=file_sha256(tensor_path),
                  cuda_initialized=torch.cuda.is_initialized(), rng_unchanged=torch.equal(rng, torch.get_rng_state()))
    _require(not report["cuda_initialized"] and report["rng_unchanged"], "CPU isolation or RNG invariant failed")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"report": str(report_path), "tensor_report": str(tensor_path),
                      "history_dq": report["history_dq"], "propagation": report["propagation"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
