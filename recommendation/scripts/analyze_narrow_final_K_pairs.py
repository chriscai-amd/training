#!/usr/bin/env python3
"""Independently search exact adjacent BF16 pairs in trusted final DQ and K.

Reads raw tensor bits, not the earlier analyzer's samples. Same-sequence/head
matches and any row progression are value correspondence only. No GPU work.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_narrow_attention_pair import file_identity, require, scalar


def analyze(path, expected_sha256):
    identity = file_identity(path)
    require(identity["sha256"] == expected_sha256, "Raw artifact SHA256 differs")
    require(not torch.cuda.is_initialized(), "GPU already initialized")
    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    require(raw["format"] == "hstu_attention_backward_failure", "Unsupported capture")
    dq, k = raw["outputs"]["dq"], raw["pristine_inputs"]["k"]
    require(dq.dtype == k.dtype == torch.bfloat16 and dq.shape == k.shape, "DQ/K layout mismatch")
    offsets = raw["pristine_inputs"]["seq_offsets"]
    rows, nonzero = {}, 0
    for begin in range(0, len(dq), 512):
        indices = torch.nonzero(dq[begin:begin + 512] != 0)
        nonzero += len(indices)
        require(nonzero <= 100000, "More than 100000 wrong values; this bounded pair search is unsuitable")
        for local, head, feature in indices.tolist():
            rows.setdefault((begin + local, head), []).append(feature)
    pairs, unpaired = [], []
    for (row, head), features in sorted(rows.items()):
        sequence = int(torch.searchsorted(offsets[1:], row, right=True))
        start, end = int(offsets[sequence]), int(offsets[sequence + 1])
        values = {feature: scalar(dq[row, head, feature]) for feature in features}
        kb = k[start:end, head].contiguous().view(torch.int16).numpy().view(np.uint16)
        pending = set(features)
        for feature in features:
            if feature not in pending:
                continue
            if feature + 1 not in pending or any(values[f]["kind"] != "normal" for f in (feature, feature + 1)):
                unpaired.append({"DQ_global_index": [row, head, feature], "sequence": sequence,
                                 "sequence_row": row - start, "value": values[feature]})
                pending.remove(feature)
                continue
            bits = [values[feature]["bits"], values[feature + 1]["bits"]]
            candidates = np.argwhere((kb[:, :-1] == bits[0]) & (kb[:, 1:] == bits[1])).tolist()
            pairs.append({"sequence": sequence, "head": head, "DQ_global_row": row,
                          "DQ_sequence_row": row - start, "DQ_features": [feature, feature + 1],
                          "bits": bits, "candidate_count": len(candidates),
                          "all_same_sequence_head_K_adjacent_pairs": [
                              {"sequence_row": r, "global_row": start + r, "features": [f, f + 1]} for r, f in candidates]})
            pending -= {feature, feature + 1}
    groups = []
    for seq_head in sorted({(v["sequence"], v["head"]) for v in pairs}):
        group = [v for v in pairs if (v["sequence"], v["head"]) == seq_head]
        # Retain all feature-2/3 row progressions satisfying dKrow=8*dDQrow;
        # no alignment or prior-block relationship is presumed.
        common = None
        for item in group:
            candidates = {v["sequence_row"] - 8 * item["DQ_sequence_row"] for v in item["all_same_sequence_head_K_adjacent_pairs"] if v["features"] == [2, 3]}
            common = candidates if common is None else common & candidates
        groups.append({"sequence": seq_head[0], "head": seq_head[1], "pair_count": len(group),
                       "DQ_sequence_rows": [v["DQ_sequence_row"] for v in group],
                       "K_feature_2_3_affine_candidates_Krow_equals_8_DQrow_plus": sorted(common or [])})
    require(not torch.cuda.is_initialized(), "GPU initialized")
    return {"format": "narrow_final_DQ_K_pair_CPU_search_v1", "raw_artifact": identity,
            "iteration": raw["iteration"], "final_DQ_numeric_nonzero": nonzero,
            "normal_adjacent_pairs": pairs, "unpaired_values": unpaired, "groups": groups,
            "matched_normal_pairs": sum(v["candidate_count"] > 0 for v in pairs),
            "unique_normal_pairs": sum(v["candidate_count"] == 1 for v in pairs),
            "analysis_source_sha256": file_identity(__file__)["sha256"], "gpu_execution": False,
            "scope": "Complete nonzero DQ scan for this bounded artifact; normal adjacent pairs compared with every adjacent K feature pair in the same sequence/head. Scalar or pair equality does not prove runtime provenance, a common production mechanism, or original enormous-value origin."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "Report already exists")
    torch.set_num_threads(4)
    rng = torch.get_rng_state().clone()
    result = analyze(args.raw, args.sha256)
    require(torch.equal(rng, torch.get_rng_state()), "CPU RNG changed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "nonzero": result["final_DQ_numeric_nonzero"],
                      "normal_pairs": len(result["normal_adjacent_pairs"]), "matched_pairs": result["matched_normal_pairs"],
                      "unique_pairs": result["unique_normal_pairs"], "unpaired": len(result["unpaired_values"]),
                      "groups": result["groups"]}))


if __name__ == "__main__":
    main()
