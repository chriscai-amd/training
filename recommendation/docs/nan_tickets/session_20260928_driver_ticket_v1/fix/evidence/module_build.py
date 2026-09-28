#!/usr/bin/env python3
"""Build a CWSR-only candidate against the reinstalled source, in tmpfs."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
B = ROOT.parent
SOURCE = Path('/usr/src/amdgpu-7.1.1-2397345.24.04')
WORK = Path('/dev/shm/mi450_current_cwsr_build_20260923')
SCRATCH = WORK / 'source'
MODULE = Path('/lib/modules/6.14.0-37-generic/updates/dkms/amdgpu.ko.zst')
HANDLER = B / 'session_20260921_CWSR_llvm_roundtrip_v2/corrected.bin'
PATCH = B / 'session_20260921_CWSR_bank_neutral_fix_feasibility_v1/proposed_non_wave_start_source_change_v2.diff'

def pin(p):
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def save(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def inventory():
    return {str(p.relative_to(SOURCE)): pin(p) for p in sorted(SOURCE.rglob('*'))
            if p.is_file() and not p.is_symlink()}

def execute(label, argv):
    save(label + '.start.json', {'argv': argv, 'cwd':str(SCRATCH), 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    env = dict(os.environ, AMDGPU_BTF='0')
    with (ROOT / (label + '.log')).open('x') as f:
        proc = subprocess.run(argv, cwd=SCRATCH, env=env, stdout=f, stderr=subprocess.STDOUT)
    save(label + '.terminal.json', {'returncode':proc.returncode, 'log':pin(ROOT / (label + '.log'))})
    if proc.returncode:
        raise RuntimeError(label + ' failed')

def retain_module(name):
    source = SCRATCH / 'amd/amdgpu/amdgpu.ko'
    # Keep unstripped intermediate in tmpfs; durable artifact retains all alloc sections.
    full = WORK / (name + '_amdgpu.ko')
    shutil.copyfile(source, full)
    target = ROOT / (name + '_amdgpu.stripped.ko')
    subprocess.run(['strip', '--strip-debug', '-o', str(target), str(full)], check=True)
    return {'unstripped_tmpfs':pin(full), 'stripped':pin(target),
            'metadata':{k:subprocess.check_output(['modinfo','-F',k,str(target)],text=True).strip()
                        for k in ['version','srcversion','vermagic','depends']}}

def main():
    assert os.getuid() != 0
    assert os.uname().release == '6.14.0-37-generic'
    assert not WORK.exists()
    assert shutil.disk_usage('/dev/shm').free > 10 * 1024**3
    installed = pin(MODULE)
    assert installed['sha256'] == 'dae133624fad0ae7bb7d95dd563edbc5218bcd0ba5ea1f5b44473093a0cc93d2'
    original = inventory()
    save('source_inventory_before.json', original)
    WORK.mkdir()
    shutil.copytree(SOURCE, SCRATCH, symlinks=True)
    with tarfile.open(ROOT / 'installed_source.tar.gz', 'w:gz', compresslevel=1) as archive:
        archive.add(SOURCE, arcname='source', recursive=True)
    save('inputs.json', {'installed':installed, 'source_archive':pin(ROOT / 'installed_source.tar.gz'),
                        'source_inventory':pin(ROOT / 'source_inventory_before.json'),
                        'handler':pin(HANDLER), 'patch':pin(PATCH), 'builder':pin(Path(__file__))})
    print('Building current baseline in tmpfs', flush=True)
    execute('baseline_build', ['make', 'KERNELVER=6.14.0-37-generic', 'num_cpu_cores=32',
                              'module_build_dir=' + str(WORK / 'build_link')])
    baseline = retain_module('baseline')
    save('baseline.json', baseline)
    header = SCRATCH / 'amd/amdkfd/cwsr_trap_handler.h'
    before = header.read_text()
    pattern = re.compile(r'(static const uint32_t cwsr_trap_gfx12_1_0_hex\[\] = \{)(.*?)(\};)', re.S)
    match = pattern.search(before)
    assert match
    old = b''.join(int(w,16).to_bytes(4,'little') for w in re.findall(r'0x[0-9a-fA-F]+',match[2]))
    assert hashlib.sha256(old).hexdigest() == '0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290'
    corrected = HANDLER.read_bytes()
    assert len(corrected) == 5672 and hashlib.sha256(corrected).hexdigest() == '68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80'
    replacement = '\n' + ''.join(f'\t0x{int.from_bytes(corrected[i:i+4],"little"):08x},\n' for i in range(0,len(corrected),4))
    header.write_text(before[:match.start(2)] + replacement + before[match.end(2):])
    execute('candidate_asm_patch', ['patch','--batch','--forward','-p1','-d',str(SCRATCH/'amd/amdkfd'),'-i',str(PATCH)])
    for name in ['cwsr_trap_handler.h','cwsr_trap_handler_gfx12.asm']:
        shutil.copyfile(SCRATCH/'amd/amdkfd'/name,ROOT/name)
    print('Building current candidate with only CWSR handler change', flush=True)
    execute('candidate_build', ['make','-j32','TTM_NAME=amdttm','SCHED_NAME=amd-sched','-C',
                               '/lib/modules/6.14.0-37-generic/build','M='+str(SCRATCH)])
    candidate = retain_module('candidate')
    assert installed == pin(MODULE)
    assert original == inventory(), 'Installed source changed during build'
    save('BUILD.json', {'status':'BUILT_NOT_LOADED_AWAITING_BYTE_AND_RELOCATION_AUDIT',
                       'baseline':baseline,'candidate':candidate,'installed_unchanged':True,
                       'installed_sources_unchanged':True,'GPU_actions':False,'driver_actions':False})
    print(json.dumps({'status':'BUILT_NOT_LOADED_AWAITING_AUDIT','candidate':candidate}),flush=True)

if __name__ == '__main__':
    main()
