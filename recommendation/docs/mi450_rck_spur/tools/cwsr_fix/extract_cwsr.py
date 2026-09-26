#!/usr/bin/env python3
"""Extract CWSR trap-handler arrays from an amdgpu.ko[.zst|.xz] and compare
them with the MI450 A0 reference images recorded in
recommendation/docs/mi450_a0/evidence/current_20260919/CWSR_module_byte_audit.json.
"""
import hashlib
import json
import lzma
import os
import struct
import subprocess
import sys

KNOWN = {
    "0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290":
        "FAULTY original gfx1250 handler (5656 B; A0 7.1.1 srcversions 654C1DDE/EADFD8EC; B0 identical)",
    "68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80":
        "CORRECTED A0 candidate handler (5672 B; srcversions 8336BBB7/AD83C153)",
}
# v_readlane_b32 ttmp15, v1, 0 / v_writelane_b32 v1, ttmp15, 0 as installed on A0
READLANE_V1 = bytes.fromhex("7b0060d701010100")
WRITELANE_V1 = bytes.fromhex("010061d77b000100")


def load_module(path):
    raw = open(path, "rb").read()
    if path.endswith(".zst"):
        try:
            import zstandard  # type: ignore
            return zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 30)
        except ImportError:
            return subprocess.run(["zstd", "-dc", path], check=True, capture_output=True).stdout
    if path.endswith(".xz"):
        return lzma.decompress(raw)
    return raw


def parse_elf(blob):
    assert blob[:4] == b"\x7fELF" and blob[4] == 2, "expect ELF64"
    (e_shoff,) = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    secs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", blob, o)
        secs.append(dict(name_off=name, type=typ, off=off, size=size, link=link, entsize=entsize))
    shstr = secs[e_shstrndx]
    for s in secs:
        n = blob[shstr["off"] + s["name_off"]:]
        s["name"] = n[:n.index(b"\0")].decode()
    return secs


def symbols(blob, secs):
    for s in secs:
        if s["type"] != 2:  # SHT_SYMTAB
            continue
        strtab = secs[s["link"]]
        for i in range(s["size"] // 24):
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", blob, s["off"] + i * 24)
            n = blob[strtab["off"] + st_name:]
            yield n[:n.index(b"\0")].decode(errors="replace"), st_shndx, st_value, st_size


def main():
    path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    blob = load_module(path)
    secs = parse_elf(blob)
    result = {"module": path, "module_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
              "decompressed_bytes": len(blob),
              "decompressed_sha256": hashlib.sha256(blob).hexdigest(), "handlers": []}
    for name, shndx, value, size in symbols(blob, secs):
        if "cwsr_trap" not in name or size == 0 or shndx == 0 or shndx >= len(secs):
            continue
        sec = secs[shndx]
        data = blob[sec["off"] + value: sec["off"] + value + size]
        h = hashlib.sha256(data).hexdigest()
        entry = {"symbol": name, "section": sec["name"], "section_offset": value, "bytes": size,
                 "sha256": h, "known": KNOWN.get(h, "UNKNOWN"),
                 "readlane_v1_offsets": [i for i in range(0, len(data) - 7, 4) if data[i:i + 8] == READLANE_V1],
                 "writelane_v1_offsets": [i for i in range(0, len(data) - 7, 4) if data[i:i + 8] == WRITELANE_V1]}
        binp = os.path.join(out_dir, f"{name}.bin")
        open(binp, "wb").write(data)
        entry["dump"] = binp
        entry["head_hex"] = data[:160].hex()
        result["handlers"].append(entry)
    json.dump(result, open(os.path.join(out_dir, "cwsr_handlers.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "handlers"}, indent=1))
    for e in result["handlers"]:
        print(f"{e['symbol']:40s} {e['section']:10s} {e['bytes']:6d} B sha256={e['sha256'][:16]}.. "
              f"readlane_v1@{e['readlane_v1_offsets'][:4]} writelane_v1@{e['writelane_v1_offsets'][:4]} -> {e['known']}")


if __name__ == "__main__":
    main()
