#!/usr/bin/env python3
"""Build the audited corrected gfx1250 CWSR handler and a matching cwsr_trap_handler.h.

The corrected image is the installed 5656-byte handler with bytes [60, 108) replaced by the
64-byte L_NOT_WAVE_START sequence from cwsr_gfx1250_bank_fix.patch and the pc-4 branch to
L_RESTORE moved by the 16-byte growth, exactly as recorded in
recommendation/docs/mi450_a0/evidence/current_20260919/CWSR_complete_handler_build.json.
The result must hash to the audited candidate or nothing is written.

usage: make_candidate_handler.py <installed amdgpu.ko[.zst]> <cwsr_trap_handler.h> <out_dir>
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cwsr  # noqa: E402

SYMBOL = "cwsr_trap_gfx12_1_0_hex"
FAULTY_SHA = "0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290"
CORRECTED_SHA = "68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80"
MC = "/opt/rocm/lib/llvm/bin/llvm-mc"

ORIGINAL_REGION = (60, 108)
ORIGINAL_PIECES = {  # offset -> installed bytes that must be present before replacement
    60: "7e00fabe",          # s_mov_b32 ttmp14, exec_lo
    64: "8100febe",          # s_mov_b32 exec_lo, 1
    68: "7b0060d701010100",  # v_readlane_b32 ttmp15, v1, 0
    96: "010061d77b000100",  # v_writelane_b32 v1, ttmp15, 0
    104: "7a00febe",         # s_mov_b32 exec_lo, ttmp14
}
REPLACEMENT = [
    ("s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_MODE, 12, 6)", "012beeb8"),
    ("s_mov_b32 ttmp3, 0", "8000efbe"),
    ("s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 6), ttmp3", "012b6fb9"),
    ("s_mov_b32 ttmp14, exec_lo", "7e00fabe"),
    ("s_mov_b32 exec_lo, 1", "8100febe"),
    ("v_readlane_b32 ttmp15, v1, 0", "7b0060d701010100"),
    ("v_sub_nc_u32_e64 v1, 63, ttmp2", "010026d5bfdc0002"),
    ("global_prefetch_b8 v1, ttmp[2:3] offset:-63 scope:SCOPE_SE", "6e4017ee0000040001c1ffff"),
    ("v_writelane_b32 v1, ttmp15, 0", "010061d77b000100"),
    ("s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 6), ttmp2", "012b6eb9"),
    ("s_mov_b32 exec_lo, ttmp14", "7a00febe"),
]
BRANCH_PC, BRANCH_OLD, BRANCH_NEW = 4, 0xBFA003F1, 0xBFA003F5  # s_branch L_RESTORE: 4044 -> 4060


def installed_handler(module_path):
    blob = extract_cwsr.load_module(module_path)
    secs = extract_cwsr.parse_elf(blob)
    for name, shndx, value, size in extract_cwsr.symbols(blob, secs):
        if name == SYMBOL:
            s = secs[shndx]
            return blob[s["off"] + value: s["off"] + value + size]
    raise SystemExit(f"{SYMBOL} not found in {module_path}")


def header_array(text):
    m = re.search(r"static const uint32_t " + SYMBOL + r"\[\] = \{(.*?)\};", text, re.S)
    if not m:
        raise SystemExit(f"{SYMBOL} initializer not found in header")
    words = [int(w, 16) for w in re.findall(r"0x[0-9a-fA-F]{8}", m.group(1))]
    return m, b"".join(struct.pack("<I", w) for w in words)


def disassemble(data):
    try:
        hexs = " ".join(f"0x{b:02x}" for b in data)
        out = subprocess.run([MC, "--disassemble", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1250"],
                             input=hexs, capture_output=True, text=True, check=True).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.strip().startswith(".")]
    except (OSError, subprocess.CalledProcessError) as e:
        return [f"disassembly unavailable: {e}"]


def main():
    module_path, header_path, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    rep = {"module": module_path, "header": header_path}

    inst = installed_handler(module_path)
    rep["installed_bytes"] = len(inst)
    rep["installed_sha256"] = hashlib.sha256(inst).hexdigest()
    if rep["installed_sha256"] != FAULTY_SHA:
        raise SystemExit(f"installed handler is not the known faulty image: {rep['installed_sha256']}")

    text = open(header_path).read()
    m, hdr = header_array(text)
    rep["header_matches_installed"] = hdr == inst
    if hdr != inst:
        raise SystemExit("header array differs from the installed handler; source/module mismatch")

    for off, want in ORIGINAL_PIECES.items():
        got = inst[off: off + len(want) // 2].hex()
        if got != want:
            raise SystemExit(f"unexpected installed bytes at {off}: {got} != {want}")
    (branch,) = struct.unpack_from("<I", inst, BRANCH_PC)
    if branch != BRANCH_OLD:
        raise SystemExit(f"unexpected pc-{BRANCH_PC} branch 0x{branch:08x}")

    repl = b"".join(bytes.fromhex(h) for _, h in REPLACEMENT)
    lo, hi = ORIGINAL_REGION
    new = bytearray(inst[:lo] + repl + inst[hi:])
    struct.pack_into("<I", new, BRANCH_PC, BRANCH_NEW)
    new = bytes(new)
    rep["corrected_bytes"] = len(new)
    rep["corrected_sha256"] = hashlib.sha256(new).hexdigest()
    rep["corrected_matches_audited_candidate"] = rep["corrected_sha256"] == CORRECTED_SHA
    rep["original_region_disassembly"] = disassemble(inst[lo:hi])
    rep["replacement_region_disassembly"] = disassemble(new[lo:lo + len(repl)])
    if not rep["corrected_matches_audited_candidate"]:
        json.dump(rep, open(os.path.join(out, "candidate_handler.json"), "w"), indent=1)
        raise SystemExit(f"corrected handler hash {rep['corrected_sha256']} != audited {CORRECTED_SHA}")

    words = struct.unpack(f"<{len(new) // 4}I", new)
    body = "".join(f"\t0x{words[i]:08x}, 0x{words[i + 1]:08x},\n" for i in range(0, len(words) - 1, 2))
    if len(words) % 2:
        body += f"\t0x{words[-1]:08x},\n"
    new_text = text[:m.start(1)] + "\n" + body + text[m.end(1):]
    _, check = header_array(new_text)
    assert check == new, "regenerated header does not round-trip"

    open(os.path.join(out, "handler_installed.bin"), "wb").write(inst)
    open(os.path.join(out, "handler_corrected.bin"), "wb").write(new)
    open(os.path.join(out, "cwsr_trap_handler.h.candidate"), "w").write(new_text)
    rep["candidate_header_sha256"] = hashlib.sha256(new_text.encode()).hexdigest()
    json.dump(rep, open(os.path.join(out, "candidate_handler.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("disassembly")}, indent=1))
    print("original [60,108):", *rep["original_region_disassembly"], sep="\n  ")
    print("replacement [60,124):", *rep["replacement_region_disassembly"], sep="\n  ")


if __name__ == "__main__":
    main()
