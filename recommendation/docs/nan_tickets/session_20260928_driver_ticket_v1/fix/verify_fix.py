#!/usr/bin/env python3
"""Check the packaged fix on CPU; optionally rebuild only two 5.6 KB handlers.

Requires Python 3 and patch. --llvm-bin requires gfx1250-capable llvm-mc and
llvm-objcopy. All temporary writes go to a private system temporary directory.
No kernel-module build, GPU access, system changes, or original SP3 assembly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
PINS = {
    'baseline': (5656, '0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290'),
    'corrected': (5672, '68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80'),
}
ARRAY = re.compile(rb'(static const uint32_t cwsr_trap_gfx12_1_0_hex\[\] = \{)(.*?)(\};)', re.S)
DIS = re.compile(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def extract_array(raw):
    matches = list(ARRAY.finditer(raw))
    require(len(matches) == 1, 'Expected exactly one affected handler array')
    return b''.join(int(word, 16).to_bytes(4, 'little')
                    for word in re.findall(rb'0x[0-9A-Fa-f]+', matches[0][2]))


def verify_disassembly(path, raw):
    cursor = 0
    count = 0
    for line in path.read_text().splitlines():
        match = DIS.match(line)
        if not match:
            continue
        _, at, words = match.groups()
        payload = b''.join(int(word, 16).to_bytes(4, 'little') for word in words.split())
        require(int(at, 16) == cursor, f'Non-contiguous disassembly at {at}')
        require(raw[cursor:cursor + len(payload)] == payload, f'Disassembly mismatch at {at}')
        cursor += len(payload)
        count += 1
    require(cursor == len(raw), f'Incomplete disassembly: {path.name}')
    return count


def command(argv, cwd):
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60)
    require(done.returncode == 0, f'Command failed: {argv}\n{done.stdout}\n{done.stderr}')
    return dict(argv=argv, returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-bin', type=Path,
                        help='Directory containing gfx1250-capable llvm-mc and llvm-objcopy')
    args = parser.parse_args()
    manifest = ROOT / 'SHA256SUMS'
    require(manifest.is_file(), 'SHA256SUMS missing; use a complete fix package')
    manifest_count = 0
    manifest_names = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT / name
        require(path.is_file() and path.resolve().is_relative_to(ROOT), f'Invalid manifest path: {name}')
        require(name not in manifest_names, f'Duplicate manifest entry: {name}')
        require(sha(path.read_bytes()) == expected, f'Manifest mismatch: {name}')
        manifest_names.add(name)
        manifest_count += 1
    present = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file()}
    require(present - manifest_names == {'SHA256SUMS'}, 'Manifest does not cover every package file')
    require(manifest_count > 0, 'Empty package manifest')
    native = {}
    instruction_counts = {}
    for name, (size, expected) in PINS.items():
        raw = (ROOT / 'native' / (name + '.bin')).read_bytes()
        require((len(raw), sha(raw)) == (size, expected), f'Handler pin mismatch: {name}')
        native[name] = raw
        instruction_counts[name] = verify_disassembly(ROOT / 'native' / (name + '_disassemble.stdout'), raw)
    old, new = native['baseline'], native['corrected']
    require(old[:4] == new[:4] and old[8:60] == new[8:60] and old[108:] == new[124:],
            'Unexpected instruction change outside prologue/branch')
    require(struct.unpack_from('<I', old, 4)[0] == 0xbfa003f1, 'Original branch mismatch')
    require(struct.unpack_from('<I', new, 4)[0] == 0xbfa003f5, 'Corrected branch mismatch')
    original_header = (ROOT / 'source/original/cwsr_trap_handler.h').read_bytes()
    proposed_header = (ROOT / 'source/proposed/cwsr_trap_handler.h').read_bytes()
    require(extract_array(original_header) == old, 'Original header differs from native')
    require(extract_array(proposed_header) == new, 'Proposed header differs from native')
    old_match, new_match = ARRAY.search(original_header), ARRAY.search(proposed_header)
    require(original_header[:old_match.start(2)] == proposed_header[:new_match.start(2)]
            and original_header[old_match.end(2):] == proposed_header[new_match.end(2):],
            'Other header content changed')
    receipts = []
    with tempfile.TemporaryDirectory(prefix='verify_cwsr_fix_') as temporary:
        scratch = Path(temporary)
        target = scratch / 'drivers/gpu/drm/amd/amdkfd'
        target.mkdir(parents=True)
        for filename in ('cwsr_trap_handler_gfx12.asm', 'cwsr_trap_handler.h'):
            shutil.copyfile(ROOT / 'source/original' / filename, target / filename)
        receipts.append(command(['patch', '--batch', '--forward', '-p1', '-i',
                                 str(ROOT / 'proposed_fix.patch')], scratch))
        for filename in ('cwsr_trap_handler_gfx12.asm', 'cwsr_trap_handler.h'):
            require((target / filename).read_bytes() == (ROOT / 'source/proposed' / filename).read_bytes(),
                    f'Patch output mismatch: {filename}')
        if args.llvm_bin:
            for name in PINS:
                obj, binary = scratch / (name + '.o'), scratch / (name + '.bin')
                receipts.append(command([str(args.llvm_bin / 'llvm-mc'), '--triple=amdgcn-amd-amdhsa',
                                         '--mcpu=gfx1250', '--filetype=obj',
                                         str(ROOT / 'native' / (name + '.s')), '-o', str(obj)], scratch))
                receipts.append(command([str(args.llvm_bin / 'llvm-objcopy'), '--dump-section',
                                         '.text=' + str(binary), str(obj)], scratch))
                require(binary.read_bytes() == native[name], f'Rebuilt {name} handler differs')
    report = dict(status='PASS', manifest_files_checked=manifest_count,
                  complete_disassembly_instruction_counts=instruction_counts,
                  patch_applied_in_private_temporary_directory=True,
                  patch_output_matches_packaged_source=True,
                  header_contains_exact_tested_native_bytes=True,
                  only_prologue_and_branch_changed=True,
                  native_llvm_reassembly_checked=bool(args.llvm_bin),
                  native_pins={name: dict(bytes=size, sha256=digest) for name, (size, digest) in PINS.items()},
                  commands=receipts,
                  original_SP3_assembler_tested=False, module_rebuilt=False, GPU_actions=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
