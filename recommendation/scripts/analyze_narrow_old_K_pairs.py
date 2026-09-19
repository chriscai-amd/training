#!/usr/bin/env python3
"""CPU search of old-DQ query-separated pairs against adjacent pristine K bits.

Uses independently reread saved workspace words. A match is value
correspondence; no instruction, temporal origin or production mapping follows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_narrow_attention_pair import file_identity, require, scalar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--workspace-sha256", required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "Report already exists")
    require(not torch.cuda.is_initialized(), "GPU initialized")
    torch.set_num_threads(4)
    identities = {"workspace": file_identity(args.workspace), "raw": file_identity(args.raw)}
    require(identities["workspace"]["sha256"] == args.workspace_sha256, "Workspace hash mismatch")
    require(identities["raw"]["sha256"] == args.raw_sha256, "Raw hash mismatch")
    stage = torch.load(args.workspace, map_location="cpu", mmap=True, weights_only=False)
    raw = torch.load(args.raw, map_location="cpu", mmap=True, weights_only=False)
    require(stage["selected_stage"] == "old_dq" and not stage["force_positive_control"], "Requires natural old-DQ")
    require(stage["iteration"] == raw["iteration"], "Iteration mismatch")
    inputs, dq = raw["pristine_inputs"], raw["outputs"]["dq"]
    spec = stage["workspace_spec"]
    require(spec["block_m"] == 32 and spec["dim"] == 128, "Requires observed 128x32 panel layout")
    rows = stage["workspace_raw_int32"][8:].reshape(spec["programs"], spec["program_stride_words"])
    selected = spec["panels"][1]
    records, singles = [], []
    normal_values, final_matches = 0, 0
    for pid in torch.nonzero(rows[:, 0]).flatten().tolist():
        slot = rows[pid]
        require(int(slot[0]) == 2 and int(slot[1]) == pid, "Incomplete/mismatched slot")
        sequence = int(inputs["sort_by_length_indices"][pid // dq.shape[1]])
        head = pid % dq.shape[1]
        start, end = map(int, inputs["seq_offsets"][sequence:sequence + 2])
        query, key = int(slot[2]), int(slot[3])
        bits = slot[selected["offset_words"]:selected["offset_words"] + selected["words"]].reshape(selected["shape"])
        require(bool(((bits >= 0) & (bits <= 65535)).all()), "Invalid panel bits")
        values = bits.to(torch.int16).view(torch.bfloat16)
        kb = inputs["k"][start:end, head].contiguous().view(torch.int16).numpy().view(np.uint16)
        for feature, q in torch.nonzero(values != 0).tolist():
            b = int(bits[feature, q])
            final = int(dq[start + query + q, head, feature].view(torch.int16)) & 65535
            require(b == final, "Saved old-DQ differs from same-coordinate final DQ")
            final_matches += 1
            if scalar(values[feature, q])["kind"] != "normal":
                singles.append({"program_id": pid, "stage_coordinate": [feature, q], "bits": b})
            else:
                normal_values += 1
        for feature in range(128):
            for q in range(16):
                if any(scalar(values[feature, x])["kind"] != "normal" for x in (q, q + 16)):
                    continue
                pair = [int(bits[feature, x]) for x in (q, q + 16)]
                matches = np.argwhere((kb[:, :-1] == pair[0]) & (kb[:, 1:] == pair[1])).tolist()
                records.append({"program_id": pid, "sequence": sequence, "head": head,
                    "query_start": query, "key_start": key, "DQ_feature": feature,
                    "query_offsets": [q, q + 16], "bits": pair,
                    "candidate_count": len(matches), "all_same_sequence_head_K_adjacent_pairs": [
                        {"sequence_row": r, "global_row": start + r, "row_relative_to_previous_key_block": r - (key - spec["block_n"]),
                         "features": [f, f + 1]} for r, f in matches],
                    "v1_feature0_formula_applicable": feature == 0 and q in (8, 10, 12),
                    "v1_feature0_formula_match": [key - spec["block_n"] + 8 * (q - 8), 2] in matches
                        if feature == 0 and q in (8, 10, 12) else None})
    require(2 * len(records) == normal_values, "Some normal values lack a query-separated normal partner")
    result = {"format": "old_DQ_query_separated_K_pair_CPU_search_v1", "artifacts": identities,
              "iteration": stage["iteration"], "normal_values": normal_values, "pairs": records,
              "non_normal_values": singles, "all_nonzero_old_final_bit_matches": final_matches,
              "matched_pairs": sum(item["candidate_count"] > 0 for item in records),
              "unique_pairs": sum(item["candidate_count"] == 1 for item in records),
              "source_sha256": file_identity(__file__)["sha256"], "gpu_execution": False,
              "scope": "Each normal pair joins query offsets q and q+16 at one DQ feature, then searches all adjacent K features in the same sequence/head. These exact saved-value matches do not establish a transfer instruction or runtime provenance."}
    require(not torch.cuda.is_initialized(), "GPU initialized")
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
