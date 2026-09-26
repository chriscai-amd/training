#!/usr/bin/env python3
"""Compare a rebuilt (strip -g) amdgpu.ko against the installed one.

Reports allocated-section differences, the .text functions whose bytes changed, and the
gfx1250 CWSR handler identity of each module.

usage: verify_module.py <installed.ko> <candidate.ko> <out.json>
"""
import hashlib
import json
import struct
import sys

SHF_ALLOC = 0x2
STT_FUNC = 2


def elf(path):
    blob = open(path, "rb").read()
    assert blob[:4] == b"\x7fELF" and blob[4] == 2
    (shoff,) = struct.unpack_from("<Q", blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    secs = []
    for i in range(shnum):
        name, typ, flags, _addr, off, size, link, info, _align, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", blob, shoff + i * shentsize)
        secs.append(dict(name_off=name, type=typ, flags=flags, off=off, size=size, link=link))
    strtab = secs[shstrndx]
    for s in secs:
        raw = blob[strtab["off"] + s["name_off"]:]
        s["name"] = raw[:raw.index(b"\0")].decode()
    syms = []
    for s in secs:
        if s["type"] != 2:
            continue
        st = secs[s["link"]]
        for i in range(s["size"] // 24):
            n, info, _other, shndx, value, size = struct.unpack_from("<IBBHQQ", blob, s["off"] + i * 24)
            raw = blob[st["off"] + n:]
            syms.append((raw[:raw.index(b"\0")].decode(errors="replace"), info & 0xF, shndx, value, size))
    return blob, secs, syms


def section_bytes(blob, s):
    return b"" if s["type"] == 8 else blob[s["off"]:s["off"] + s["size"]]  # SHT_NOBITS


def diff_count(a, b):
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return 0
    return sum(1 for i in range(n) if a[i] != b[i])


def handler(blob, secs, syms, name="cwsr_trap_gfx12_1_0_hex"):
    for n, _t, shndx, value, size in syms:
        if n == name:
            s = secs[shndx]
            data = blob[s["off"] + value:s["off"] + value + size]
            return {"bytes": size, "sha256": hashlib.sha256(data).hexdigest()}
    return None


def main():
    inst_path, cand_path, out = sys.argv[1:4]
    A, B = elf(inst_path), elf(cand_path)
    rep = {"installed": inst_path, "candidate": cand_path,
           "installed_handler": handler(*A), "candidate_handler": handler(*B)}

    secs_a = {s["name"]: s for s in A[1] if s["flags"] & SHF_ALLOC}
    secs_b = {s["name"]: s for s in B[1] if s["flags"] & SHF_ALLOC}
    rep["allocated_sections_only_in_installed"] = sorted(set(secs_a) - set(secs_b))
    rep["allocated_sections_only_in_candidate"] = sorted(set(secs_b) - set(secs_a))
    differing = []
    for name in sorted(set(secs_a) & set(secs_b)):
        da, db = section_bytes(A[0], secs_a[name]), section_bytes(B[0], secs_b[name])
        if da != db:
            differing.append({"section": name, "installed_bytes": len(da), "candidate_bytes": len(db),
                              "differing_bytes_in_overlap": diff_count(da, db)})
    rep["allocated_sections_differing"] = differing

    def funcs(E):
        blob, secs, syms = E
        text = {i for i, s in enumerate(secs) if s["name"] == ".text"}
        out = {}
        for n, t, shndx, value, size in syms:
            if t == STT_FUNC and shndx in text and size:
                s = secs[shndx]
                out.setdefault(n, blob[s["off"] + value:s["off"] + value + size])
        return out
    fa, fb = funcs(A), funcs(B)
    changed = []
    for n in sorted(set(fa) & set(fb)):
        if fa[n] != fb[n]:
            a, b = fa[n], fb[n]
            offs = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
            changed.append({"symbol": n, "installed_bytes": len(a), "candidate_bytes": len(b),
                            "different_offsets": offs[:16], "n_different": len(offs)})
    rep["functions_checked"] = len(set(fa) & set(fb))
    rep["functions_only_in_one"] = len(set(fa) ^ set(fb))
    rep["changed_functions"] = changed
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
