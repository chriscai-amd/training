#!/usr/bin/env python3
"""Decode CPU-captured arguments for the audited MT128x240x128 native kernel.

This reads files only. It never imports Torch or follows a captured pointer.
The inline layout is checked against the supplied ELF metadata. Indirect HBM
and external UserArgs modes are reported as envelopes, not guessed structures.
Nominal header bitfield names use related-build hipBLASLt revision 67811f1e;
that source revision is NOT pinned to the replay's loaded host library. Raw
fields are metadata-backed. Tree split predictions also match static inspection
of the actual 88b27358 runtime binary's streamKStaticSplit helper.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct


# (metadata name, offset, interpretation). Metadata calls pointers global_buffer.
FIELDS = [
    ("Gemm info", 0, "u32"), ("kernel info0", 4, "u32"),
    ("kernel info1", 8, "u32"), ("numWG", 12, "u32"),
    ("SizesFree0", 16, "u32"), ("SizesFree1", 20, "u32"),
    ("SizesFree2", 24, "u32"), ("SizesSum0", 28, "u32"),
    ("D", 32, "pointer"), ("C", 40, "pointer"),
    ("A", 48, "pointer"), ("B", 56, "pointer"),
    ("AddressWS", 64, "pointer"), ("AddressFlags", 72, "pointer"),
    ("strideD0", 80, "u32"), ("strideD1", 84, "u32"),
    ("strideC0", 88, "u32"), ("strideC1", 92, "u32"),
    ("strideA0", 96, "u32"), ("strideA1", 100, "u32"),
    ("strideB0", 104, "u32"), ("strideB1", 108, "u32"),
    ("alpha", 112, "f32"), ("beta", 116, "f32"),
    ("ItersPerTile", 120, "u32"), ("MagicNumberItersPerTile", 124, "u32"),
    ("MagicShiftItersPerTile", 128, "u32"), ("SKItersPerWG", 132, "u32"),
    ("skGrid", 136, "u32"), ("skTiles", 140, "u32"),
    ("AddressScaleAlphaVec", 144, "pointer"), ("bias", 152, "pointer"),
    ("biasType", 160, "u32"), ("StrideBias", 164, "u32"),
    ("activationAlpha", 168, "f32"), ("activationBeta", 172, "f32"),
    ("activationType", 176, "u32"), ("batchOffsetD", 180, "u64"),
    ("batchOffsetC", 188, "u64"), ("batchOffsetA", 196, "u64"),
    ("batchOffsetB", 204, "u64"),
]
SIZES = {"u32": 4, "f32": 4, "pointer": 8, "u64": 8}


def validate_metadata(path):
    raw = Path(path).read_bytes()
    text = raw.decode()
    entries = []
    for block in re.split(r"(?m)^      - ", text)[1:]:
        props = dict(re.findall(r"(?m)^(?: {8})?\.(\w+):[ \t]*([^\n]+)", block))
        if "offset" in props:
            entries.append(props)
    actual = [(p["name"].strip(), int(p["offset"]), int(p["size"])) for p in entries]
    expected = [(n, o, SIZES[t]) for n, o, t in FIELDS]
    if actual != expected:
        raise ValueError("metadata layout does not match the audited inline layout")
    for p, (_, _, typ) in zip(entries, FIELDS):
        if typ == "pointer":
            if p.get("value_kind", "").strip() != "global_buffer":
                raise ValueError("metadata pointer kind mismatch")
        elif p.get("value_type", "").strip() != typ:
            raise ValueError("metadata scalar type mismatch")
    names = re.findall(r"(?m)^    \.name:\s*(\S+)", text)
    if len(names) != 1 or "_MT128x240x128_" not in names[0]:
        raise ValueError("expected metadata for one MT128x240x128 kernel")
    version = re.search(r"KernArgsVersion:\s*(\d+)", text)
    if not version or int(version[1]) != 2:
        raise ValueError("only audited KernArgsVersion 2 is supported")
    segments = re.findall(r"(?m)^    \.kernarg_segment_size:\s*(\d+)", text)
    aligns = re.findall(r"(?m)^    \.kernarg_segment_align:\s*(\d+)", text)
    if segments != ["216"] or aligns != ["8"]:
        raise ValueError("metadata must declare kernarg size 216 and alignment 8")
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "kernel_name": names[0], "kernargs_version": 2,
            "last_field_end": 212, "aligned_kernarg_segment_size": 216}


def decode_capture(capture):
    if capture.get("status") != "captured":
        return {"status": "not_captured", "capture": capture}
    size, hx = capture.get("buffer_size"), capture.get("buffer_hex")
    if type(size) is not int or size < 0 or size > 4096 or not isinstance(hx, str):
        return {"status": "invalid_capture", "reason": "invalid size or hex type"}
    if len(hx) != 2 * size or re.fullmatch(r"[0-9a-fA-F]*", hx) is None:
        return {"status": "invalid_capture", "reason": "hex and declared size disagree"}
    raw = bytes.fromhex(hx)
    result = {"size_bytes": size, "sha256": hashlib.sha256(raw).hexdigest()}
    if size < 16:
        return dict(result, status="truncated_header")
    info, info0, info1, numwg = struct.unpack_from("<4I", raw)
    mode, count = info >> 30, info & 0x3fffffff
    result["header"] = {
        "bitfield_name_basis": "Nominal KernArgsVersion2 names from related-build67811f1e; runtime source revision unknown",
        "gemm_count": count, "arg_type": mode, "numWG": numwg,
        "internalArgs0_hex": hex(info0), "internalArgs1_hex": hex(info1),
        "gsu": info0 & 0x3fff, "gsuwgmrr": (info0 >> 14) & 1,
        "gsuc": (info0 >> 15) & 1, "stagger_upper16_raw": info0 >> 16,
        "wgm_low16_raw": info1 & 0xffff, "wgmxcc_bits16_21": (info1 >> 16) & 63,
        "wgmxccgroup_bits22_31": info1 >> 22,
    }
    if mode in (1, 2):
        if size < 24:
            return dict(result, status="truncated_indirect_envelope")
        result["device_argument_pointer"] = hex(struct.unpack_from("<Q", raw, 16)[0])
        result["indirect_mode"] = "internal_hbm" if mode == 1 else "external_user_args"
        return dict(result, status="indirect_body_not_decoded",
                    limitation="Captured pointer is never followed; body layout is not inferred.")
    if count != 1:
        return dict(result, status="unsupported_inline_gemm_count")
    if size not in (212, 216):
        return dict(result, status="truncated_inline" if size < 212 else "unexpected_inline_size")
    fields = {}
    for name, offset, typ in FIELDS:
        chunk = raw[offset:offset + SIZES[typ]]
        value = struct.unpack("<f" if typ == "f32" else "<I" if typ == "u32" else "<Q", chunk)[0]
        if typ == "pointer":
            value = hex(value)
        elif typ == "f32" and not math.isfinite(value):
            value = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        fields[name] = {"offset": offset, "type": typ, "raw_hex_le": chunk.hex(), "value": value}
        if name.startswith("batchOffset"):
            fields[name]["signed_value"] = struct.unpack("<q", chunk)[0]
    result.update(status="decoded_inline", fields=fields, tail_padding_hex=raw[212:].hex())
    vals = {k: v["value"] for k, v in fields.items()}
    tiles = ((vals["SizesFree0"] + 127) // 128) * ((vals["SizesFree1"] + 239) // 240) * vals["SizesFree2"]
    grid, iters = vals["skGrid"], vals["ItersPerTile"]
    # Tree SK3 with FULL_TILES=1: remainder plus one grid when > one full grid.
    sktiles = tiles % grid if grid else None
    if grid and (sktiles == 0 or tiles > grid):
        sktiles += grid
    result["geometry"] = {
        "macro_tile": [128, 240, 128], "logical_tiles": tiles,
        "computed_iters_per_tile": max(1, (vals["SizesSum0"] + 127) // 128),
        "tree_sk3_full_tiles_1_expected_skTiles": sktiles,
        "tree_sk3_full_tiles_1_expected_SKItersPerWG": sktiles * iters // grid if grid else None,
        "skGrid_equals_numWG": grid == numwg,
        "one_workgroup_per_logical_tile": grid == tiles,
    }
    return result


def launch_grid(row):
    grid, block = row.get("grid_or_global"), row.get("block_or_local")
    if not isinstance(grid, list) or not isinstance(block, list) or len(grid) != 3 or len(block) != 3:
        return None
    if any(type(v) is not int or v <= 0 for v in grid + block):
        return None
    if row.get("api") in ("hipExtModuleLaunchKernel", "hipHccModuleLaunchKernel"):
        if any(g % b for g, b in zip(grid, block)):
            return None
        return [g // b for g, b in zip(grid, block)]
    if row.get("api") == "hipModuleLaunchKernel":
        return grid
    return None


def analyze_trace(path, metadata, terminal=False):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if terminal and not stable:
        raise ValueError("trace changed while reading terminal snapshot")
    events = collections.Counter()
    variants, launches, returns, issues = {}, {}, {}, []
    for lineno, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            issues.append({"line": lineno, "error": str(exc)})
            continue
        event = row.get("event")
        events[event] += 1
        key = (row.get("pid"), row.get("launch_id"))
        if event == "launch_return":
            if key in returns:
                issues.append({"line": lineno, "error": "duplicate launch return"})
            returns[key] = row.get("hip_error")
        if event != "launch_begin" or row.get("name") != metadata["kernel_name"]:
            continue
        if key in launches:
            issues.append({"line": lineno, "error": "duplicate launch begin"})
        decoded = decode_capture(row.get("argument_capture", {}))
        variant_id = hashlib.sha256(json.dumps(decoded, sort_keys=True).encode()).hexdigest()
        variant = variants.setdefault(variant_id, {"decode": decoded, "launch_count": 0,
                                                  "first_launch": list(key), "last_launch": list(key)})
        variant["launch_count"] += 1
        variant["last_launch"] = list(key)
        wg = launch_grid(row)
        launches[key] = {"workgroups": wg, "variant_id": variant_id}
        if decoded.get("status") == "decoded_inline":
            agree = wg is not None and math.prod(wg) == decoded["header"]["numWG"]
            if not agree:
                issues.append({"line": lineno, "error": "launch grid disagrees with numWG"})
    if terminal and issues:
        raise ValueError(f"terminal trace has parse/integrity issues: {issues[:3]}")
    return {
        "schema": "projection_hip_arguments_v1", "trace": str(path.resolve()),
        "trace_sha256": hashlib.sha256(raw).hexdigest(), "trace_bytes": len(raw),
        "terminal_asserted_by_caller": terminal, "stable_during_read": stable,
        "metadata": metadata, "events": dict(events), "selected_launches": len(launches),
        "selected_returns": sum(k in returns for k in launches),
        "selected_successful_returns": sum(returns.get(k) == 0 for k in launches),
        "launch_workgroup_counts": dict(collections.Counter(str(v["workgroups"]) for v in launches.values())),
        "argument_variants": list(variants.values()), "issues": issues,
        "limitations": ["Host-captured arguments do not establish device memory contents.",
                        "Header bitfield names use related-build67811f1e source; runtime source match is not established.",
                        "Successful HIP launch return does not prove kernel completion or numerical correctness.",
                        "Exact symbol equality does not establish this process loaded identical ELF bytes.",
                        "No HBM or external UserArgs body is decoded; no captured pointer is followed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal", action="store_true", help="Caller attests writer has terminated")
    args = parser.parse_args()
    report = analyze_trace(args.trace, validate_metadata(args.metadata), args.terminal)
    report["decoder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("trace_sha256", "selected_launches", "selected_successful_returns", "launch_workgroup_counts")}))


if __name__ == "__main__":
    main()
