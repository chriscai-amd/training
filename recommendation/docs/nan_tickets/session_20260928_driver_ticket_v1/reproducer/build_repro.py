#!/usr/bin/env python3
"""CPU-only optional shader rebuild; the exact executed HSACO is already supplied."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.dont_write_bytecode=True
import run_repro as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llvm-bin',type=Path,default=Path('/opt/rocm/llvm/bin'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg,original,layout=r.verify_exact()
    r.require(not os.path.lexists(args.output),'Build output must be fresh')
    args.output.mkdir();source=r.ROOT/'exact/sentinel_shader_root_v2.s'
    commands=[['llvm-mc','--triple=amdgcn-amd-amdhsa','--mcpu=gfx1250','--filetype=obj',str(source),'-o',str(args.output/'sentinel.o')],
              ['ld.lld','-shared',str(args.output/'sentinel.o'),'-o',str(args.output/'sentinel.hsaco')],
              ['llvm-objdump','-d','--mcpu=gfx1250',str(args.output/'sentinel.hsaco')]]
    runs=[]
    for index,command in enumerate(commands):
        tool=args.llvm_bin/command[0];r.require(tool.is_file(),'Missing LLVM tool: '+str(tool))
        command[0]=str(tool.absolute())
        run=subprocess.run(command,capture_output=True,timeout=120)
        (args.output/f'{index}.stdout').write_bytes(run.stdout);(args.output/f'{index}.stderr').write_bytes(run.stderr)
        runs.append(dict(command=command,returncode=run.returncode))
        r.require(run.returncode==0,'CPU shader build command failed: '+command[0])
    code=(args.output/'sentinel.hsaco').read_bytes();rebuilt=r.elf_layout(code)
    r.require(rebuilt==layout,'Rebuilt native functions/return sites/sections differ from executed image')
    result=dict(status='PASS_CPU_SHADER_REBUILD_NATIVE_BYTES_MATCH',commands=runs,
        LLVM_version=subprocess.check_output([str(args.llvm_bin/'llvm-mc'),'--version'],text=True),
        shader_source=r.record(source),executed_image=r.record(r.ROOT/'exact/sentinel_shader_root_v2.hsaco'),
        rebuilt_image=r.record(args.output/'sentinel.hsaco'),whole_ELF_byte_identical=code==original,
        all_14_kernel_bytes_and_return_sites_equal=True,all_text_rodata_metadata_sections_equal=True,
        GPU_actions=False,driver_actions=False)
    with (args.output/'BUILD_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
