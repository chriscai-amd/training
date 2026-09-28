#!/usr/bin/env python3
"""Bounded CPU dataflow check of the already frozen CWSR native/source fix.

No disassembler/build invocation, GPU access, or module read. Reads only small
source/native artifacts; writes only this supplement's new JSON receipt.
"""
from pathlib import Path
import hashlib
import json
import re
import struct

ROOT = Path(__file__).resolve().parent
FIX = ROOT.parent / 'session_20260928_driver_ticket_v1/fix'
DIS = re.compile(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)')
PINS = {
    'baseline': '0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290',
    'corrected': '68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80',
}
SOURCE_LINES = {
    0x6c: (291, 298), 0x74: (293, 300), 0x78: (309, 316),
    0x7c: (310, 317), 0x80: (311, 318), 0x88: (329, 336),
    0x8c: (330, 337), 0x90: (331, 338), 0x98: (333, 340),
    0xa0: (334, 341), 0xa8: (337, 344),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def parse(name):
    path = FIX / 'native' / (name + '.bin')
    binary = path.read_bytes()
    require(hashlib.sha256(binary).hexdigest() == PINS[name], 'Native pin mismatch')
    rows = {}
    cursor = 0
    for line in (FIX / 'native' / (name + '_disassemble.stdout')).read_text().splitlines():
        match = DIS.match(line)
        if not match:
            continue
        instruction, address, words = match.groups()
        raw = b''.join(int(word, 16).to_bytes(4, 'little') for word in words.split())
        pc = int(address, 16)
        require(pc == cursor and binary[pc:pc + len(raw)] == raw, 'Native/disassembly mismatch')
        rows[pc] = dict(pc=pc, instruction=instruction.strip(), hex=raw.hex(), size=len(raw))
        cursor += len(raw)
    require(cursor == len(binary), 'Incomplete native disassembly')
    return rows


def semantics(row):
    """Reject any unexpected instruction in this short checked dispatch prefix."""
    instruction = row['instruction']
    operation = instruction.split()[0]
    registers = re.findall(r'\bttmp\d+\b', instruction)
    if operation in ('s_and_not1_b32', 's_and_b32'):
        require(bool(registers), 'Missing scalar destination')
        reads, writes = registers[1:], registers[:1]
    elif operation == 's_getreg_b32':
        require(registers == ['ttmp15'], 'Unexpected GETREG target')
        reads, writes = [], registers
    elif operation in ('s_bitcmp0_b32', 's_bitcmp1_b32'):
        reads, writes = registers, []
    elif operation in ('s_cbranch_scc0', 's_cbranch_scc1', 's_setreg_imm32_b32'):
        require(not registers, 'Unexpected implicit scratch operand')
        reads, writes = [], []
    else:
        raise RuntimeError('Unreviewed instruction: ' + instruction)
    next_pc = row['pc'] + row['size']
    edges = [next_pc]
    if operation.startswith('s_cbranch'):
        word = struct.unpack('<I', bytes.fromhex(row['hex']))[0]
        require(word >> 16 in (0xbfa1, 0xbfa2), 'Unexpected branch encoding')
        signed_immediate = (word & 0xffff) - (0x10000 if word & 0x8000 else 0)
        # The disassembly's L_XXXX names retain original addresses after insertion.
        # Derive actual targets from instruction words instead of label spellings.
        edges.append(row['pc'] + 4 + 4 * signed_immediate)
    return reads, writes, edges


def check(name, offset):
    rows = parse(name)
    start, end = 0x6c + offset, 0xa8 + offset
    prefix = {}
    for pc in sorted(rows):
        if not start <= pc <= end:
            continue
        row = dict(rows[pc])
        reads, writes, successors = semantics(row)
        original_line, proposed_line = SOURCE_LINES[pc - offset]
        row.update(ttmp_reads=reads, ttmp_writes=writes,
                   successors=[] if pc == end else successors,
                   original_source_line=original_line, proposed_source_line=proposed_line)
        prefix[pc] = row
    require(len(prefix) == 11, 'Unexpected prefix size')
    require(prefix[end]['instruction'] == 's_and_b32 ttmp2, ttmp12, 0x4000', 'Changed merge write')
    paths = []

    def walk(pc, dirty, visited):
        require(pc in prefix and pc not in visited, 'Exit or cycle before overwrite')
        row = prefix[pc]
        require(not set(row['ttmp_reads']) & dirty, 'Live-out difference read before overwrite')
        dirty = dirty - set(row['ttmp_writes'])
        visited = visited + [pc]
        if pc == end:
            require(not dirty, 'Residual differing scratch value')
            paths.append(visited)
        else:
            for successor in row['successors']:
                walk(successor, dirty, visited)

    walk(start, {'ttmp2', 'ttmp15'}, [])
    require(len(paths) == 4, 'Expected four conservative conditional paths')
    return dict(prefix_start=start, common_ttmp2_overwrite=end,
                unconditional_ttmp15_overwrite=0x74 + offset,
                complete_handler_instruction_count=len(rows),
                prefix=list(prefix.values()), conservative_paths=paths,
                all_paths_overwrite_both_differences_before_read=True)


def main():
    require(pin(FIX / 'SHA256SUMS')['sha256'] ==
            'be7559b911e385d39c12cb0c4d347f8107cd7ac25c983479c9cdb8ba49ec6029',
            'Unexpected frozen package revision')
    baseline, corrected = check('baseline', 0), check('corrected', 16)
    for old, new in zip(baseline['prefix'], corrected['prefix']):
        require(old['hex'] == new['hex'] and old['instruction'] == new['instruction'],
                'Dispatch prefix changed beyond relocation')
        require([pc + 16 for pc in old['successors']] == new['successors'],
                'Dispatch CFG changed beyond relocation')
    sources = {variant: FIX / 'source' / variant / 'cwsr_trap_handler_gfx12.asm'
               for variant in ('original', 'proposed')}
    source = {variant: path.read_text().splitlines() for variant, path in sources.items()}
    require('= ttmp2' in source['proposed'][182] and '= ttmp3' in source['proposed'][183],
            'Unexpected EXEC save aliases')
    require('= ttmp15' in source['proposed'][185] and '= ttmp14' in source['proposed'][194],
            'Unexpected scratch aliases')
    for pc, (before, after) in SOURCE_LINES.items():
        require(source['original'][before - 1] == source['proposed'][after - 1],
                f'Following source instruction changed at old PC {pc}')
    report = dict(
        status='PASS_BOUNDED_SEQUENTIAL_LIVEOUT_DATAFLOW',
        scope='Affected non-wave-start prologue through common dispatch write, without GPU or module reads',
        checker=pin(Path(__file__).resolve()), frozen_fix_manifest=pin(FIX / 'SHA256SUMS'),
        sources={variant: pin(path) for variant, path in sources.items()},
        baseline=baseline, corrected=corrected,
        live_out={
            'ttmp2': dict(original='0', corrected='k = original MODE[17:12]',
                          result='Overwritten from ttmp12 HALT bit before any use on all four prefix paths'),
            'ttmp3': dict(original='0', corrected='0', result='No live-out difference'),
            'ttmp14': dict(original='Incoming EXEC_LO', corrected='Incoming EXEC_LO', result='No live-out difference'),
            'ttmp15': dict(original='Lane 0 of SRC0-selected encoded v1', corrected='Physical v1 lane 0',
                           result='Unconditionally overwritten with EXCP_FLAG_PRIV before any use or following branch'),
        },
        source_coordinates={
            'MODE_field_shift_and_size': dict(original=[142, 143], proposed=[142, 143]),
            'saved_STATE_PRIV_including_original_SCC': dict(original=[246], proposed=[246]),
            'prologue': dict(original=[261, 271], proposed=[261, 278]),
            'scratch_convention_comment': dict(original=[401, 402], proposed=[408, 409]),
            'second_level_transfer': dict(original=[467], proposed=[474]),
            'return_EXECZ_and_VCCZ_refresh': dict(original=[505, 506], proposed=[512, 513]),
            'return_saved_SCC_restore': dict(original=[510, 517, 518], proposed=[517, 524, 525]),
            'CWSR_EXEC_copy_into_ttmp2_ttmp3': dict(original=[543, 544], proposed=[550, 551]),
            'CWSR_original_MODE_bank_capture_then_clear': dict(original=[562, 566], proposed=[569, 573]),
        },
        preserved_state={
            'SCC': 'Prologue MOV/GETREG/SETREG and lane operations do not write SCC; V_SUB_NC_U32 does not write SCC. Current SCC and saved STATE_PRIV are equal at dispatch entry.',
            'VCC': 'No added instruction writes VCC; non-carry V_SUB_NC_U32 is required.',
            'EXEC': 'EXEC_LO is restored from ttmp14 before dispatch; EXEC_HI is not touched; WAVE32_ONLY applies.',
            'MODE': 'Only bits 17:12 are temporarily cleared and restored; SRC2 and other bits are untouched. Restored before later context-save bank capture.',
        },
        limitations=[
            'Source/native sequential def-use proof; not a hardware hazard or timing proof.',
            'Existing TTMP scratch ABI and normal ordered instruction execution are assumed. No new dependence on TTMP2 retaining zero survives this merge.',
            'Asynchronous/reentrant observation of transient trap temporaries is not modeled; vendor trap ABI/hazard review remains necessary.',
            'Original SP3 parser/assembler acceptance remains untested.',
            'No claim that all training NaNs or residual corrected-driver STU1 behavior are resolved.',
        ],
        original_SP3_assembler_tested=False, GPU_actions=False, module_reads=False, module_rebuilds=False,
        frozen_fix_changed=False)
    with (ROOT / 'LIVEOUT_REVIEW.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], original_paths=len(baseline['conservative_paths']),
                          corrected_paths=len(corrected['conservative_paths']),
                          receipt=pin(ROOT / 'LIVEOUT_REVIEW.json'))))


if __name__ == '__main__':
    main()
