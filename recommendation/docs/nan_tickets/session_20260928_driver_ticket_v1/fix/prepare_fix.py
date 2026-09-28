"""Package the already tested CWSR fix; no module rebuild, system/GPU action.

Reads only pinned sources, small handler binaries/disassembly and metadata.
Writes only this fresh fix evidence directory and a private patch-check tree.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
import struct
import subprocess

ROOT=Path(__file__).resolve().parent
B=ROOT.parents[1]
BUILD=B/'session_20260923_current_CWSR_build_v1'
NATIVE=B/'session_20260921_CWSR_llvm_roundtrip_v2'
PATTERN=re.compile(r'(static const uint32_t cwsr_trap_gfx12_1_0_hex\[\] = \{)(.*?)(\};)',re.S)
OLD_SHA='0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290'
NEW_SHA='68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def pin(path):
    path=Path(path);assert path.stat().st_size<=1<<20
    raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=sha(raw))
def checked(row):
    assert pin(row['path'])==row;return Path(row['path']).read_bytes()
def save(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not isinstance(raw,bytes):raw=(json.dumps(raw,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:stream.write(raw)
    return pin(path)
def array(raw):
    m=PATTERN.search(raw.decode());assert m
    return b''.join(int(w,16).to_bytes(4,'little') for w in re.findall(r'0x[0-9A-Fa-f]+',m[2]))
def instructions(path):
    result=[]
    for line in path.read_text().splitlines():
        m=re.match(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)',line)
        if m:
            inst,at,words=m.groups();raw=b''.join(int(w,16).to_bytes(4,'little') for w in words.split())
            result.append(dict(pc=int(at,16),instruction=inst,hex=raw.hex(),native_words=words.strip()))
    return result


def main():
    inventory=json.loads((BUILD/'source_inventory_before.json').read_text())
    build_inputs=json.loads((BUILD/'inputs.json').read_text())
    native=json.loads((NATIVE/'complete_handler_build.json').read_text())
    module_audit_path=B/'session_20260923_current_CWSR_build_audit_v1/REVIEW.json'
    audit=json.loads(module_audit_path.read_text())
    assert pin(BUILD/'source_inventory_before.json')['sha256']=='401da7c4df750d8fc87b50e66da5355248634101c9911f4dfa0800ece0a88282'
    assert pin(module_audit_path)['sha256']=='df6b21da005286a79048a95e1028f1f65b0584c69e95543cbcf1b509944cf3d1'
    originals={}
    for name in ('cwsr_trap_handler_gfx12.asm','cwsr_trap_handler.h','kfd_device.c'):
        key='amd/amdkfd/'+name
        originals[name]=checked(inventory[key])
    before=checked(native['builds']['baseline']['binary'])
    after=checked(native['builds']['corrected']['binary'])
    assert len(before)==5656 and sha(before)==OLD_SHA
    assert len(after)==5672 and sha(after)==NEW_SHA
    assert array(originals['cwsr_trap_handler.h'])==before
    tested_header=(BUILD/'cwsr_trap_handler.h').read_bytes()
    assert array(tested_header)==after
    assert before[:4]==after[:4] and before[8:60]==after[8:60] and before[108:]==after[124:]
    old_branch,new_branch=struct.unpack_from('<I',before,4)[0],struct.unpack_from('<I',after,4)[0]
    assert old_branch==0xbfa003f1 and new_branch==0xbfa003f5
    assert (4+4+4*(old_branch&65535),4+4+4*(new_branch&65535))==(4044,4060)
    oldrows=instructions(NATIVE/'baseline_disassemble.stdout')
    newrows=instructions(NATIVE/'corrected_disassemble.stdout')
    for raw,rows in ((before,oldrows),(after,newrows)):
        assert len({r['pc'] for r in rows})==len(rows)
        cursor=0
        for row in rows:
            payload=bytes.fromhex(row['hex']);assert row['pc']==cursor and raw[cursor:cursor+len(payload)]==payload
            cursor+=len(payload)
        assert cursor==len(raw)
    # Keep the original two-word-per-line style. The tested module used a
    # one-word-per-line equivalent; all complete C hex tokens must match.
    oldtext=originals['cwsr_trap_handler.h'].decode();m=PATTERN.search(oldtext)
    words=struct.unpack('<'+str(len(after)//4)+'I',after)
    body='\n'+''.join('\t'+', '.join(f'0x{v:08x}' for v in words[i:i+2])+',\n' for i in range(0,len(words),2))
    proposed_header=(oldtext[:m.start(2)]+body+oldtext[m.end(2):]).encode()
    assert array(proposed_header)==after
    assert re.findall(rb'0x[0-9A-Fa-f]+',proposed_header)==re.findall(rb'0x[0-9A-Fa-f]+',tested_header)
    candidate_asm=(BUILD/'cwsr_trap_handler_gfx12.asm').read_bytes()
    patch=[]
    for name,new in [('cwsr_trap_handler_gfx12.asm',candidate_asm),('cwsr_trap_handler.h',proposed_header)]:
        target='drivers/gpu/drm/amd/amdkfd/'+name
        patch.extend(difflib.unified_diff(originals[name].decode().splitlines(True),new.decode().splitlines(True),
                                        fromfile='a/'+target,tofile='b/'+target,n=3))
        save(ROOT/'source/original'/name,originals[name])
        save(ROOT/'source/proposed'/name,new)
    patch_pin=save(ROOT/'proposed_fix.patch',''.join(patch).encode())
    # Apply the complete artifact in an isolated source tree, never /usr/src.
    scratch=ROOT/'patch_check/drivers/gpu/drm/amd/amdkfd';scratch.mkdir(parents=True)
    for name in ('cwsr_trap_handler_gfx12.asm','cwsr_trap_handler.h'):
        (scratch/name).write_bytes(originals[name])
    command=['patch','--batch','--forward','-p1','-i',patch_pin['path']]
    executed=subprocess.run(command,cwd=ROOT/'patch_check',capture_output=True,text=True,timeout=30)
    assert executed.returncode==0,(executed.stdout,executed.stderr)
    assert (scratch/'cwsr_trap_handler_gfx12.asm').read_bytes()==candidate_asm
    assert (scratch/'cwsr_trap_handler.h').read_bytes()==proposed_header
    assert array((scratch/'cwsr_trap_handler.h').read_bytes())==after
    # Retain complete small native payloads and source/disassembly receipts.
    copies=[]
    for name in ['baseline.bin','corrected.bin','baseline.s','corrected.s',
                 'baseline_disassemble.stdout','corrected_disassemble.stdout',
                 'baseline_assemble.command.json','corrected_assemble.command.json',
                 'baseline_extract.command.json','corrected_extract.command.json',
                 'baseline_disassemble.command.json','corrected_disassemble.command.json',
                 'complete_handler_build.json']:
        original=pin(NATIVE/name)
        copies.append(dict(original=original,packaged=save(ROOT/'native'/name,checked(original))))
    evidence=[]
    for name,path in [('module_audit.json',module_audit_path),('module_BUILD.json',BUILD/'BUILD.json'),
        ('module_build_inputs.json',BUILD/'inputs.json'),('module_build.py',BUILD/'build.py'),
        ('native_builder.py',NATIVE/'build_preserved.py'),
        ('prior_causal_audit.json',B/'session_20260927_causal_audit_v1/AUDIT.json')]:
        original=pin(path);evidence.append(dict(original=original,packaged=save(ROOT/'evidence'/name,checked(original))))
    save(ROOT/'evidence/kfd_device.c',originals['kfd_device.c'])
    replacement=dict(original=[r for r in oldrows if 60<=r['pc']<108],corrected=[r for r in newrows if 60<=r['pc']<124])
    save(ROOT/'native/prologue.json',replacement)
    text=[]
    for label,rows in replacement.items():
        text += [label.upper()+': byte offset; instruction; native 32-bit words']
        text += [f"0x{r['pc']:04x}: {r['instruction']} // {r['native_words']}" for r in rows]
        text += ['']
    save(ROOT/'native/prologue.txt','\n'.join(text).encode())
    report=dict(status='PASS_PATCH_APPLIES_AND_GENERATES_EXACT_TESTED_HANDLER_WORDS',
        preparer=pin(Path(__file__).resolve()),patch=patch_pin,
        original_source_pins=[inventory['amd/amdkfd/'+name] for name in originals],
        tested_candidate_source=dict(assembly=pin(BUILD/'cwsr_trap_handler_gfx12.asm'),header=pin(BUILD/'cwsr_trap_handler.h')),
        module_build_inputs=pin(BUILD/'inputs.json'),original_design_patch=build_inputs['patch'],
        native_copies=copies,evidence_copies=evidence,original_handler=native['builds']['baseline']['binary'],
        corrected_handler=native['builds']['corrected']['binary'],
        exact_change=dict(original_prologue_range=[60,108],corrected_prologue_range=[60,124],
                          branch_byte_offset=4,old_branch_hex='bfa003f1',new_branch_hex='bfa003f5',
                          old_branch_target=4044,new_branch_target=4060,other_instruction_bytes_unchanged=True),
        patch_application=dict(command=command,working_directory=str(ROOT/'patch_check'),returncode=executed.returncode,
                               stdout=executed.stdout,stderr=executed.stderr,exact_expected_source_bytes=True),
        all_C_hex_tokens_equal_tested_header=True,header_format_only_two_words_per_line=True,
        complete_native_disassemblies_reconstructed_and_verified=True,
        original_SP3_assembly_validated=False,module_rebuilt=False,GPU_actions=False,system_files_changed=False,
        module_provenance_from_pinned_prior_audit=audit['modules'],
        limits=['Native payloads were copied and completely rechecked; large module payload hashes and builds were not repeated.',
                'Source assembly hunk is a feasible proposal. Original SP3 parser/assembler acceptance is not established.',
                'The embedded uint32_t array is the exact payload already built and tested in the corrected module.',
                'See parent ticket evidence for runtime counterfactuals and their limits.'])
    save(ROOT/'FIX_VERIFICATION.json',report)
    print(json.dumps(dict(status=report['status'],patch=patch_pin,verification=pin(ROOT/'FIX_VERIFICATION.json'))))


if __name__=='__main__':main()
