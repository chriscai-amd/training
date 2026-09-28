"""Preserve installed VOP3 encodings while assembling the complete handler."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parent
V1 = LOGS / "session_20260921_CWSR_llvm_roundtrip_v1"
spec = importlib.util.spec_from_file_location("roundtrip", V1 / "build_roundtrip.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
helpers.ROOT = ROOT
write, run, identity = helpers.write, helpers.run, helpers.identity
LLVM = helpers.LLVM


def instructions(path):
    result = {}
    pattern = re.compile(r"^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            inst, address, words = match.groups()
            pc = int(address, 16)
            assert pc not in result
            result[pc] = (inst, b"".join(int(x, 16).to_bytes(4, "little") for x in words.split()))
    return result


def raw_instruction(inst, data):
    assert inst.split()[0] in ("v_readlane_b32", "v_writelane_b32", "v_add_nc_u32_e64", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32")
    assert len(data) in (8, 12)
    return ".long " + ", ".join(f"0x{int.from_bytes(data[i:i+4], 'little'):08x}" for i in range(0, len(data), 4)) + " // preserved: " + inst


def main():
    original_path = helpers.OLD / "installed_CWSR_gfx1250.bin"
    original = original_path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == helpers.EXPECTED
    old_rows = instructions(helpers.OLD / "installed_CWSR_gfx1250.dis")
    attempt = instructions(V1 / "baseline_disassemble.stdout")
    assert old_rows.keys() == attempt.keys()
    preserved = {}
    branches = []
    for pc, (inst, data) in old_rows.items():
        assert original[pc:pc + len(data)] == data
        other_inst, other_data = attempt[pc]
        if data != other_data:
            assert inst == other_inst
            assert len(data) == len(other_data) and len(data) in (8, 12)
            assert int.from_bytes(data, "little") ^ int.from_bytes(other_data, "little") == 1 << 57
            preserved[pc] = {"pc": pc, "instruction": inst, "installed_hex": data.hex(), "llvm_hex": other_data.hex(), "preserved_bit": 57}
        if inst.startswith("s_branch ") or inst.startswith("s_cbranch_"):
            target = pc + 4 + int.from_bytes(data[:2], "little", signed=True) * 4
            assert target in old_rows
            assert not 0x3c < target < 0x6c
            branches.append({"pc": pc, "target": target})
    assert len(preserved) == 84
    lines = (V1 / "baseline.s").read_text().splitlines()
    original_source = [".text"]
    for i in range(1, len(lines), 2):
        label, instruction = lines[i:i+2]
        pc = int(re.fullmatch(r"L_([0-9a-f]+):", label).group(1), 16)
        original_source += [label, raw_instruction(*old_rows[pc]) if pc in preserved else instruction]
    probe_path = helpers.FIX / "non_wave_start_compensated_address_probe_v2.s"
    block = probe_path.read_text().splitlines()[1:]
    assert len(block) == 11
    assert block[5] == old_rows[0x44][0]
    assert block[8] == old_rows[0x60][0]
    block[5] = raw_instruction(*old_rows[0x44])
    block[8] = raw_instruction(*old_rows[0x60])
    corrected = original_source[:original_source.index("L_003c:")] + ["L_003c:"] + block + original_source[original_source.index("L_006c:"):]
    builds = {}
    for name, source_lines in (("baseline", original_source), ("corrected", corrected)):
        source, obj, binary = [ROOT / (name + suffix) for suffix in (".s", ".o", ".bin")]
        write(source, "\n".join(source_lines) + "\n")
        run([LLVM / "llvm-mc", "--triple=amdgcn-amd-amdhsa", "--mcpu=gfx1250", "--filetype=obj", source, "-o", obj], name + "_assemble")
        run([LLVM / "llvm-objcopy", "--dump-section", f".text={binary}", obj], name + "_extract")
        run([LLVM / "llvm-objdump", "-d", "--mcpu=gfx1250", obj], name + "_disassemble")
        reloc = run([LLVM / "llvm-readelf", "-r", obj], name + "_relocations")
        assert "There are no relocations in this file." in reloc
        built = binary.read_bytes()
        builds[name] = {"source": identity(source), "object": identity(obj), "binary": identity(binary), "no_unresolved_relocations": True}
        if name == "baseline":
            assert built == original
            builds[name]["exact_installed_byte_match"] = True
        else:
            assert len(built) == len(original) + 16
            new_rows = instructions(ROOT / "corrected_disassemble.stdout")
            assert len(new_rows) == len(old_rows) + 3
            expected = bytearray(original[:0x3c] + built[0x3c:0x7c] + original[0x6c:])
            changed_branches = []
            for branch in branches:
                old_pc, old_target = branch["pc"], branch["target"]
                new_pc = old_pc + (16 if old_pc >= 0x6c else 0)
                new_target = old_target + (16 if old_target >= 0x6c else 0)
                immediate = (new_target - new_pc - 4) // 4
                assert -32768 <= immediate <= 32767
                expected[new_pc:new_pc + 2] = (immediate & 0xffff).to_bytes(2, "little")
                emitted = new_rows[new_pc][1]
                assert new_pc + 4 + int.from_bytes(emitted[:2], "little", signed=True) * 4 == new_target
                if new_pc - old_pc != new_target - old_target:
                    changed_branches.append({**branch, "new_pc": new_pc, "new_target": new_target, "new_immediate": immediate})
            assert bytes(expected) == built
            assert built[0x50:0x58] == original[0x44:0x4c]
            assert built[0x6c:0x74] == original[0x60:0x68]
            builds[name].update({"size_delta": 16, "all_other_instruction_bytes_preserved": True, "replacement_range": [0x3c, 0x7c], "original_replacement_range": [0x3c, 0x6c], "branch_targets_checked": len(branches), "branches_requiring_relocation": changed_branches, "instructions_in_replacement": [{"pc": pc, "instruction": row[0], "hex": row[1].hex()} for pc, row in new_rows.items() if 0x3c <= pc < 0x7c]})
    report = {"format": "CWSR_complete_LLVM_preserved_encoding_build_v2", "created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS_CPU_BUILD_ONLY", "builder": identity(Path(__file__)), "helper_source": identity(V1 / "build_roundtrip.py"), "failed_pure_LLVM_roundtrip": identity(V1 / "baseline_mismatch.json"), "installed_binary": identity(original_path), "installed_disassembly": identity(helpers.OLD / "installed_CWSR_gfx1250.dis"), "input_assembly": identity(V1 / "baseline.s"), "replacement_probe": identity(probe_path), "llvm_mc": identity(LLVM / "llvm-mc"), "preserved_VOP3_instruction_encodings": list(preserved.values()), "builds": builds, "limitations": ["LLVM assembly with 84 exact installed VOP3 instruction encodings emitted as .long. No semantic assumption about differing bit 57 is required.", "Disassembly-derived reconstruction, not compilation of original SP3 source.", "No handler installation or hardware execution; private early-VMEM workaround scheduling remains unvalidated.", "Original training NaNs are not yet linked to this repair hypothesis."]}
    write(ROOT / "complete_handler_build.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "baseline": builds["baseline"]["binary"], "corrected": builds["corrected"]["binary"], "branches": len(branches)}))


if __name__ == "__main__":
    main()
