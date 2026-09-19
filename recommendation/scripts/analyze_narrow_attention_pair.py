#!/usr/bin/env python3
"""CPU-only audit of a trusted v3 narrow workspace and same-launch raw buffers.

No Triton import, compilation, replay or GPU initialization. Positive-stage
artifacts require the complete v3 pair sidecar; ordinary raw failures use the
inherited v2 correlation. Inconsistent slots are retained as rejected evidence.
Bit correspondence is not proof of runtime provenance or an instruction cause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


MAGIC = 0x44514E31
STAGES = ("dqk_trans", "old_dq", "dot")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def path_key(path):
    value = str(Path(path).resolve())
    prefix = "/home/chcai/dlrm_data/"
    return "/data/mlperf_dlrm_v4/" + value[len(prefix):] if value.startswith(prefix) else value


def tensor_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def scalar(tensor):
    require(tensor.numel() == 1, "Expected scalar")
    width = 16 if tensor.dtype == torch.bfloat16 else 32
    integer = torch.int16 if width == 16 else torch.int32
    bits = int(tensor.reshape(()).view(integer)) & ((1 << width) - 1)
    mantissa = 7 if width == 16 else 23
    exponent = (bits >> mantissa) & 255
    fraction = bits & ((1 << mantissa) - 1)
    kind = ("nan" if fraction else "infinity") if exponent == 255 else (
        ("subnormal" if fraction else "zero") if exponent == 0 else "normal")
    return {"dtype": str(tensor.dtype), "bits": bits, "hex": f"0x{bits:0{width // 4}x}",
            "value": str(float(tensor)), "kind": kind}


def tensor_summary(tensor, row_chunk):
    require(tensor.device.type == "cpu" and tensor.dtype == torch.bfloat16, "Expected CPU BF16 tensor")
    result = {"shape": list(tensor.shape), "stride": list(tensor.stride()),
              "storage_offset": tensor.storage_offset(), "elements": tensor.numel(),
              "numeric_nonzero": 0, "nonzero_bits": 0, "negative_zero": 0,
              "nonfinite": 0, "abs_gt_1e6": 0, "abs_gt_1e20": 0, "max_abs_finite": 0.0}
    digest = hashlib.sha256()
    for begin in range(0, tensor.shape[0], row_chunk):
        part = tensor[begin:begin + row_chunk].contiguous()
        bits = part.view(torch.int16)
        finite = torch.isfinite(part)
        result["numeric_nonzero"] += int((part != 0).sum())
        result["nonzero_bits"] += int((bits != 0).sum())
        result["negative_zero"] += int((bits == -32768).sum())
        result["nonfinite"] += int((~finite).sum())
        result["abs_gt_1e6"] += int((part.float().abs() > 1e6).sum())
        result["abs_gt_1e20"] += int((part.float().abs() > 1e20).sum())
        if bool(finite.any()):
            result["max_abs_finite"] = max(result["max_abs_finite"], float(part[finite].float().abs().max()))
        digest.update(tensor_bytes(part))
    result["logical_sha256"] = digest.hexdigest()
    return result


def input_comparison(pristine, current, row_chunk):
    require(set(pristine) == set(current), "Input field sets differ")
    logical, backings, seen = {}, [], set()
    for name, before in pristine.items():
        after = current[name]
        if before is None or after is None:
            logical[name] = {"layout_equal": before is after, "bits_equal": before is after}
            continue
        require(before.device.type == after.device.type == "cpu", "Input is not on CPU")
        layout = (before.shape == after.shape and before.dtype == after.dtype
                  and before.stride() == after.stride() and before.storage_offset() == after.storage_offset())
        equal = layout
        if layout:
            for begin in range(0, before.shape[0], row_chunk):
                equal &= tensor_bytes(before[begin:begin + row_chunk]) == tensor_bytes(after[begin:begin + row_chunk])
        logical[name] = {"layout_equal": layout, "bits_equal": bool(equal)}
        key = (before.untyped_storage()._cdata, after.untyped_storage()._cdata)
        if key in seen:
            continue
        seen.add(key)
        a, b = before.untyped_storage(), after.untyped_storage()
        same = a.nbytes() == b.nbytes()
        if same:
            av = torch.empty(0, dtype=torch.uint8).set_(a, 0, (a.nbytes(),), (1,))
            bv = torch.empty(0, dtype=torch.uint8).set_(b, 0, (b.nbytes(),), (1,))
            for begin in range(0, a.nbytes(), 16 << 20):
                same &= torch.equal(av[begin:begin + (16 << 20)], bv[begin:begin + (16 << 20)])
        backings.append({"first_input": name, "pristine_bytes": a.nbytes(),
                         "final_bytes": b.nbytes(), "bits_equal": bool(same)})
    return {"logical": logical, "backings": backings,
            "all_logical_bits_unchanged": all(v["bits_equal"] for v in logical.values()),
            "all_backing_bits_unchanged": all(v["bits_equal"] for v in backings)}


def verify_pair(stage, raw, stage_id, raw_id, stage_path, raw_path, pair, pair_path, expect):
    require(raw.get("format") == "hstu_attention_backward_failure" and raw.get("format_version") == 1,
            "Unsupported raw artifact format")
    require(stage.get("selected_stage") in ("old_dq", "dot"), "Analyzer supports old_dq and dot")
    require(type(stage.get("force_positive_control")) is bool, "Missing boolean control label")
    forced = stage["force_positive_control"]
    require(expect == "auto" or forced == (expect == "control"), "Capture/control expectation mismatch")
    require(stage.get("iteration") == raw.get("iteration") and stage["iteration"] >= 1, "Iteration mismatch")
    require(raw.get("failures") and all(item.get("iteration") == stage["iteration"] for item in raw["failures"]),
            "Raw failure/control records do not match current iteration")
    require(stage.get("compiled_kernels") == raw.get("compiled_kernels"), "Compiled-kernel metadata mismatch")
    workspace_hash = hashlib.sha256(tensor_bytes(stage["workspace_raw_int32"])).hexdigest()
    if stage.get("format") == "hstu_dq_narrow_stage_probe_v1":
        require(stage.get("format_version") == 1, "Unsupported positive-stage version")
        require(pair is not None and pair.get("format") == "hstu_narrow_positive_pair_v3", "Positive stage requires v3 pair sidecar")
        require(pair.get("raw_dump_status") == "complete", "V3 raw dump is not marked complete")
        require(pair.get("raw_file_sha256") == raw_id["sha256"], "Raw artifact SHA256 mismatch")
        corr = pair["correlation"]
        require(corr == raw["configuration"].get("narrow_positive_stage_capture"), "Raw/sidecar correlation differs")
        require(corr.get("workspace_file_sha256") == stage_id["sha256"], "Stage-file SHA256 mismatch")
        require(corr.get("selected_stage") == stage["selected_stage"] and corr.get("force_positive_control") is forced,
                "Stage/control correlation mismatch")
        require(corr.get("decode_error") == stage.get("decode_error"), "Decode-error label mismatch")
        require(corr.get("actual_kernel_views_match_retained_callers") is True, "No actual-kernel-view validation")
        require(path_key(corr["pair_path"]) == path_key(pair_path), "Sidecar path mismatch")
        raw_key = "raw_output_path"
        kind = "forced_diagnostic_control" if forced else "positive_instrumented_stage"
        require(corr.get("capture_kind") == kind, "Capture-kind label mismatch")
    else:
        require(stage.get("format") == "hstu_dq_narrow_raw_failure_workspace_v2", "Unsupported workspace format")
        require(stage.get("format_version") == 2 and pair is None, "Unexpected v2 version or positive-stage sidecar")
        corr = stage["correlation"]
        require(corr == raw["configuration"].get("narrow_stage_workspace_capture"), "Raw/v2 workspace correlation differs")
        require(corr.get("first_raw_failure") is True, "No first-raw-failure correlation")
        raw_key, kind = "raw_failure_path", "ordinary_raw_failure"
    require(corr.get("iteration") == stage["iteration"] == corr.get("narrow_launch_count"), "Launch-count mismatch")
    require(corr.get("workspace_sha256") == workspace_hash, "Workspace raw-byte SHA256 mismatch")
    require(path_key(corr["workspace_path"]) == path_key(stage_path), "Workspace path mismatch")
    require(path_key(corr[raw_key]) == path_key(raw_path), "Raw path mismatch")
    return {"verified": True, "capture_kind": kind, "workspace_raw_sha256": workspace_hash,
            "correlation": corr, "path_alias_policy": "Exact paths after /home/chcai/dlrm_data to /data/mlperf_dlrm_v4 mapping"}


def validate_spec(stage, inputs, outputs):
    k, dq = inputs["k"], outputs["dq"]
    require(k.ndim == 3 and dq.shape == k.shape and k.dtype == dq.dtype == torch.bfloat16, "K/DQ shape or dtype mismatch")
    require(all(inputs[name].shape == k.shape and inputs[name].dtype == torch.bfloat16 for name in ("q", "v", "dout")),
            "Input tensor shape/dtype mismatch")
    require(all(value.shape == k.shape and value.dtype == torch.bfloat16 for value in outputs.values()),
            "Output tensor shape/dtype mismatch")
    offsets, order = inputs["seq_offsets"], inputs["sort_by_length_indices"]
    require(offsets.ndim == 1 and int(offsets[0]) == 0 and int(offsets[-1]) == k.shape[0]
            and bool((offsets[1:] > offsets[:-1]).all()), "Invalid sequence offsets")
    sequences, heads, dim = offsets.numel() - 1, k.shape[1], k.shape[2]
    if order is not None:
        require(order.shape == (sequences,) and torch.equal(torch.sort(order).values, torch.arange(sequences)), "Invalid sort permutation")
    spec, selected = stage["workspace_spec"], stage["selected_stage"]
    bm, bn = spec["block_m"], spec["block_n"]
    require(type(bm) is int and type(bn) is int and min(bm, bn) > 0, "Invalid tile dimensions")
    panels = [{"name": "k", "shape": [bn, dim], "dtype": "torch.bfloat16", "offset_words": 16, "words": bn * dim},
              {"name": selected, "shape": [dim, bm], "dtype": "torch.float32" if selected == "dot" else "torch.bfloat16",
               "offset_words": 16 + bn * dim, "words": dim * bm}]
    stride = 16 + bn * dim + dim * bm
    expected = {"programs": sequences * heads, "block_m": bm, "block_n": bn, "dim": dim,
                "selected_stage": selected, "global_words": 8, "header_words": 16,
                "program_stride_words": stride, "total_words": 8 + sequences * heads * stride,
                "bytes": 4 * (8 + sequences * heads * stride), "panels": panels}
    require(spec == expected, "Workspace spec differs from independently reconstructed layout")
    words = stage["workspace_raw_int32"]
    require(words.device.type == "cpu" and words.dtype == torch.int32 and words.shape == (spec["total_words"],), "Workspace tensor layout mismatch")
    return spec


def audit_workspace(stage, inputs, final_inputs, outputs, *, max_records, alpha):
    spec = validate_spec(stage, inputs, outputs)
    words, selected = stage["workspace_raw_int32"], stage["selected_stage"]
    slots = words[8:].reshape(spec["programs"], spec["program_stride_words"])
    offsets, order, heads = inputs["seq_offsets"], inputs["sort_by_length_indices"], inputs["k"].shape[1]
    claimed = torch.nonzero(slots[:, 0]).flatten().tolist()
    global_errors = []
    if int(words[1]) != len(claimed) or (claimed and int(words[0]) - 1 not in claimed) or (not claimed and int(words[0]) != 0):
        global_errors.append("global_and_per_program_claims_disagree")
    if bool((words[2:8] != 0).any()):
        global_errors.append("global_reserved_words_nonzero")
    records, samples = [], []
    total_observed = 0
    covered_final, covered_consistent, final_at_nonzero = 0, 0, 0
    for pid in range(spec["programs"]):
        slot = slots[pid]
        if int(slot[0]) == 0:
            count = int((slot != 0).sum())
            if count:
                records.append({"program_id": pid, "header": slot[:16].tolist(), "errors": ["unclaimed_slot_contains_writes"],
                                "nonzero_words": count, "locally_consistent": False, "coordinates_valid": False})
            continue
        errors = []
        record = {"program_id": pid, "header": slot[:16].tolist(), "errors": errors, "coordinates_valid": False}
        records.append(record)
        if int(slot[0]) != STAGES.index(selected) + 1 or int(slot[1]) != pid or int(slot[6]) != MAGIC or int(slot[7]) != int(stage["force_positive_control"]):
            errors.append("invalid_or_incomplete_completion_header")
            record["locally_consistent"] = False
            continue
        if bool((slot[9:16] != 0).any()):
            errors.append("reserved_slot_header_words_nonzero")
        sequence = pid // heads if order is None else int(order[pid // heads])
        head, start, end = pid % heads, int(offsets[sequence]), int(offsets[sequence + 1])
        query, key, length = map(int, slot[2:5])
        if length != end - start or not (0 <= query < length and 0 <= key < length):
            errors.append("invalid_sequence_coordinates")
            record["locally_consistent"] = False
            continue
        valid = min(spec["block_m"], length - query)
        record.update(sequence=sequence, head=head, query_start=query, key_start=key, sequence_start=start,
                      valid_query_rows=valid, coordinates_valid=True)
        if int(slot[5]) != valid:
            errors.append("query_mask_count_mismatch")
        panels = {}
        for panel in spec["panels"]:
            data = slot[panel["offset_words"]:panel["offset_words"] + panel["words"]]
            if panel["dtype"] == "torch.bfloat16":
                if bool(((data < 0) | (data > 65535)).any()):
                    errors.append(panel["name"] + "_invalid_BF16_high_bits")
                    continue
                value = data.to(torch.int16).view(torch.bfloat16)
            else:
                value = data.clone().view(torch.float32)
            panels[panel["name"]] = value.reshape(panel["shape"])
        if selected not in panels:
            record["locally_consistent"] = False
            continue
        observed = panels[selected][:, :valid]
        nonzero = observed != 0
        finite = torch.isfinite(observed)
        observed_bits = observed.view(torch.int16 if selected == "old_dq" else torch.int32)
        count = int(nonzero.sum())
        total_observed += count
        record.update(panel_numeric_nonzero=count, header_nonzero_count=int(slot[8]),
                      count_payload_consistent=count > 0 and count == int(slot[8]),
                      panel_nonzero_bits=int((observed_bits != 0).sum()),
                      panel_nonfinite=int((~finite).sum()),
                      panel_max_abs_finite=float(observed[finite].float().abs().max()) if bool(finite.any()) else None,
                      panel_abs_gt_1e20=int((observed.float().abs() > 1e20).sum()),
                      masked_tail_numeric_nonzero=int((panels[selected][:, valid:] != 0).sum()))
        if stage["force_positive_control"]:
            record["control_sentinel_only"] = count == 1 and bool(observed[0, 0] == 1)
        if not record["count_payload_consistent"]:
            errors.append("count_panel_inconsistency")
        if "k" in panels:
            expected = torch.zeros_like(panels["k"])
            current = torch.zeros_like(expected)
            nk = min(spec["block_n"], length - key)
            expected[:nk] = inputs["k"][start + key:start + key + nk, head]
            current[:nk] = final_inputs["k"][start + key:start + key + nk, head]
            record["K_panel_bits_match_pristine"] = torch.equal(expected.view(torch.int16), panels["k"].view(torch.int16))
            record["K_panel_bits_match_final_input"] = torch.equal(current.view(torch.int16), panels["k"].view(torch.int16))
            if not record["K_panel_bits_match_pristine"]:
                errors.append("K_panel_differs_from_pristine")
        final = outputs["dq"][start + query:start + query + valid, head].T
        final_bad = final != 0
        record.update(final_DQ_numeric_nonzero_in_region=int(final_bad.sum()),
                      final_DQ_nonzero_at_observed_nonzero=int((final_bad & nonzero).sum()),
                      observed_nonzero_but_final_zero=int((nonzero & ~final_bad).sum()),
                      observed_zero_but_final_nonzero=int((~nonzero & final_bad).sum()))
        if selected == "old_dq":
            record["observed_nonzero_exact_BF16_matches_final"] = int((nonzero & (observed.view(torch.int16) == final.view(torch.int16))).sum())
        else:
            candidate = (observed * torch.tensor(alpha, dtype=torch.float32)).to(torch.bfloat16)
            record["observed_nonzero_scaled_dot_candidate_bits_match_final"] = int((nonzero & (candidate.view(torch.int16) == final.view(torch.int16))).sum())
        record["locally_consistent"] = not errors
        covered_final += int(final_bad.sum())
        covered_consistent += int(final_bad.sum()) if not errors else 0
        final_at_nonzero += int((final_bad & nonzero).sum())
        for feature, query_offset in torch.nonzero(nonzero)[:max(0, max_records - len(samples))].tolist():
            value = observed[feature, query_offset]
            output = final[feature, query_offset]
            item = {"program_id": pid, "sequence": sequence, "head": head, "key_start": key,
                    "stage_tile_index": [feature, query_offset], "DQ_global_index": [start + query + query_offset, head, feature],
                    "stage_value": scalar(value), "final_DQ": scalar(output), "slot_locally_consistent": not errors,
                    "forced_diagnostic_value": stage["force_positive_control"],
                    "stage_numeric_equals_final": bool(value == output.float())}
            if selected == "old_dq":
                item["stage_BF16_bits_equal_final"] = item["stage_value"]["bits"] == item["final_DQ"]["bits"]
                item["stage_K_search_bits"] = item["stage_value"]["bits"]
            else:
                cast = value.to(torch.bfloat16)
                exact = scalar(cast.float())["bits"] == item["stage_value"]["bits"]
                item["dot_exactly_representable_as_BF16"] = exact
                item["stage_K_search_bits"] = scalar(cast)["bits"] if exact else None
                candidate = (value * torch.tensor(alpha, dtype=torch.float32)).to(torch.bfloat16)
                item["hypothetical_zero_old_scaled_dot_BF16"] = scalar(candidate)
                item["scaled_dot_candidate_bits_equal_final"] = scalar(candidate)["bits"] == item["final_DQ"]["bits"]
            samples.append(item)
    strict = not global_errors and all(record["locally_consistent"] for record in records)
    return {"global_words": words[:8].tolist(), "nonzero_words": int((words != 0).sum()),
            "all_workspace_words_zero": not bool((words != 0).any()), "global_errors": global_errors,
            "workspace_strictly_consistent": strict, "recorded_decoder_error": stage.get("decode_error"),
            "slots": records, "observed_numeric_nonzero": total_observed, "observed_samples": samples,
            "observed_samples_truncated": len(samples) < total_observed,
            "final_DQ_nonzero_in_decodable_regions": covered_final,
            "final_DQ_nonzero_in_locally_consistent_regions": covered_consistent,
            "final_DQ_nonzero_at_observed_nonzero": final_at_nonzero}


def final_samples(dq, offsets, row_chunk, limit):
    records = []
    for begin in range(0, dq.shape[0], row_chunk):
        for row, head, feature in torch.nonzero(dq[begin:begin + row_chunk] != 0)[:limit - len(records)].tolist():
            global_row = begin + row
            sequence = int(torch.searchsorted(offsets[1:], global_row, right=True))
            records.append({"DQ_global_index": [global_row, head, feature], "sequence": sequence, "head": head,
                            "final_DQ": scalar(dq[global_row, head, feature])})
        if len(records) == limit:
            break
    return records


def add_K_candidates(stage_samples, dq_samples, inputs, *, forced, max_candidates):
    requests = {}
    for item in stage_samples + dq_samples:
        for target, bits in (("final_K_bit_candidates", item["final_DQ"]["bits"]),
                             ("stage_K_bit_candidates", item.get("stage_K_search_bits"))):
            if bits is None or (target.startswith("stage_") and forced):
                continue
            requests.setdefault((item["sequence"], item["head"]), []).append((item, target, bits))
    offsets, k = inputs["seq_offsets"], inputs["k"]
    for (sequence, head), targets in requests.items():
        start, end = int(offsets[sequence]), int(offsets[sequence + 1])
        flat = k[start:end, head].contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
        sorted_indices = np.argsort(flat, kind="stable")
        sorted_bits = flat[sorted_indices]
        cache = {}
        for item, target, bits in targets:
            if bits not in cache:
                lo, hi = int(np.searchsorted(sorted_bits, bits)), int(np.searchsorted(sorted_bits, bits, side="right"))
                selected = sorted_indices[lo:min(hi, lo + max_candidates)]
                cache[bits] = {"same_sequence_head_exact_bit_match_count": hi - lo,
                               "sample_global_K_indices": [[start + int(i) // k.shape[2], head, int(i) % k.shape[2]] for i in selected],
                               "samples_truncated": hi - lo > max_candidates}
            item[target] = cache[bits]


def analyze(stage_path, raw_path, *, pair_path=None, expect="auto", max_records=4096, max_candidates=8, row_chunk=512):
    require(not torch.cuda.is_initialized(), "CUDA was already initialized")
    stage_path, raw_path = Path(stage_path), Path(raw_path)
    stage_id, raw_id = file_identity(stage_path), file_identity(raw_path)
    stage = torch.load(stage_path, map_location="cpu", mmap=True, weights_only=False)
    raw = torch.load(raw_path, map_location="cpu", mmap=True, weights_only=False)
    pair_path = Path(pair_path) if pair_path else Path(str(stage_path) + ".pair.json")
    pair = json.loads(pair_path.read_text()) if pair_path.exists() else None
    identity = verify_pair(stage, raw, stage_id, raw_id, stage_path, raw_path, pair, pair_path, expect)
    inputs, final_inputs, outputs = raw["pristine_inputs"], raw["inputs_at_failure"], raw["outputs"]
    require(set(outputs) == {"dq", "dk", "dv"}, "Unexpected output fields")
    comparison = input_comparison(inputs, final_inputs, row_chunk)
    summaries = {name: tensor_summary(value, row_chunk) for name, value in outputs.items()}
    input_summaries = {name: tensor_summary(inputs[name], row_chunk) for name in ("q", "k", "v", "dout")}
    alpha = float(raw["configuration"]["context"]["attn_alpha"])
    require(math.isfinite(alpha), "Nonfinite attention alpha")
    workspace = audit_workspace(stage, inputs, final_inputs, outputs, max_records=max_records, alpha=alpha)
    dq_records = final_samples(outputs["dq"], inputs["seq_offsets"], row_chunk, max_records)
    add_K_candidates(workspace["observed_samples"], dq_records, inputs,
                     forced=stage["force_positive_control"], max_candidates=max_candidates)
    workspace["final_DQ_nonzero_outside_decodable_regions"] = summaries["dq"]["numeric_nonzero"] - workspace["final_DQ_nonzero_in_decodable_regions"]
    require(workspace["final_DQ_nonzero_outside_decodable_regions"] >= 0, "Overlapping decoded DQ regions")
    require(not torch.cuda.is_initialized(), "Analyzer initialized CUDA")
    return {"format": "narrow_attention_same_launch_CPU_analysis_v1", "status": "ANALYZED",
            "artifacts": {"workspace": stage_id, "raw": raw_id, "pair": file_identity(pair_path) if pair else None},
            "analysis_source_sha256": file_identity(__file__)["sha256"], "pair_identity": identity,
            "iteration": stage["iteration"], "selected_stage": stage["selected_stage"],
            "forced_control": stage["force_positive_control"], "attention_alpha": alpha,
            "input_comparison": comparison, "pristine_input_summaries": input_summaries,
            "all_dout_numeric_zero": input_summaries["dout"]["numeric_nonzero"] == 0,
            "Q_V_dOut_numeric_zero": all(input_summaries[name]["numeric_nonzero"] == 0 for name in ("q", "v", "dout")),
            "reduced_zero_oracle_inputs_verified": all(input_summaries[name]["numeric_nonzero"] == 0 for name in ("q", "v", "dout"))
                and input_summaries["k"]["nonfinite"] == 0 and comparison["all_logical_bits_unchanged"],
            "forced_control_final_outputs_all_numeric_zero": all(value["numeric_nonzero"] == 0 for value in summaries.values())
                if stage["force_positive_control"] else None,
            "output_summaries": summaries, "workspace": workspace, "final_DQ_samples": dq_records,
            "final_DQ_samples_truncated": len(dq_records) < summaries["dq"]["numeric_nonzero"],
            "gpu_initialized": False, "gpu_execution": False,
            "scope": [
                "All workspace words and full output tensors are scanned; coordinate/provenance sample limits are explicit.",
                "A rejected count/panel slot remains rejected even when other slots are locally consistent. Global first claim is a race winner, not causal order.",
                "An entirely zero workspace means no retained writes; it does not prove the arithmetic stage was zero or correct.",
                "Old-DQ/final comparisons use BF16 bits. Dot values use FP32 bits; K matching requires exact BF16 representability, not rounding to a nearby K value.",
                "Scaled-dot BF16 is a hypothetical zero-old contribution, not a predicted final result: the old addend and later key-pass contributions are unobserved.",
                "Forced sentinels modify diagnostic panels only; they are not arithmetic failures. K matches are exact value correspondence, not runtime provenance.",
                "Input mutation, rejected slots, masked tails and truncated samples must be considered before attribution. No compiler/hardware instruction cause or original enormous-value origin is proved."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--pair", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expect", choices=("auto", "control", "natural"), default="auto")
    parser.add_argument("--max-records", type=int, default=4096)
    parser.add_argument("--max-k-candidates", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(not args.report.exists(), "Report destination already exists")
    require(min(args.max_records, args.max_k_candidates, args.row_chunk, args.threads) > 0, "Limits must be positive")
    require(args.report.resolve() not in {args.workspace.resolve(), args.raw.resolve(),
            (args.pair or Path(str(args.workspace) + ".pair.json")).resolve()}, "Report aliases an input")
    torch.set_num_threads(args.threads)
    rng = torch.get_rng_state().clone()
    try:
        report = analyze(args.workspace, args.raw, pair_path=args.pair, expect=args.expect,
                         max_records=args.max_records, max_candidates=args.max_k_candidates, row_chunk=args.row_chunk)
        status = 0
    except (ValueError, KeyError, RuntimeError) as error:
        report = {"format": "narrow_attention_same_launch_CPU_analysis_v1", "status": "REJECTED",
                  "error": f"{type(error).__name__}: {error}", "gpu_initialized": torch.cuda.is_initialized()}
        status = 2
    require(torch.equal(rng, torch.get_rng_state()), "CPU RNG changed")
    report["cpu_rng_unchanged"] = True
    report["gpu_visibility"] = {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "status": report["status"], "error": report.get("error"),
                      "iteration": report.get("iteration"), "selected_stage": report.get("selected_stage"),
                      "workspace_strictly_consistent": report.get("workspace", {}).get("workspace_strictly_consistent")}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
