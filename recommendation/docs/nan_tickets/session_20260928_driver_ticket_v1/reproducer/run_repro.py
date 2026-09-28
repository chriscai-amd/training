#!/usr/bin/env python3
"""Portable adapter around the exact executed MI450 CWSR sentinel host/native image.

--check-only is CPU-only. A normal run performs 14 small GPU kernel launches.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import struct
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
MODES=(0x00,0x41,0x45,0x82,0x05,0x81,0x85)
EXACT={
 'run_sentinel_host_v1.py':(21231,'be1582e96b7dad21991a562a6d4aec2664d331bd30ae64e1117cbd2d045ace89'),
 'sentinel_shader_root_v2.s':(100058,'5172641877ade249d232cd5c721aed1e404b98921afe367043ad905784417a00'),
 'sentinel_shader_root_v2.hsaco':(32152,'4f840df38a5d815abe50824f37dedbde4bf48fffdec6ca84530cd631f69e5484'),
 'sentinel_host_configuration_v2.json':(4898,'4a7fd0b589959eec2f3c1f73fd86afacca18affb732d122df624bf0f7e4f2554'),
}


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def record(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=digest(raw))


def expected_cases():
    return [dict(mode_low_byte=m,trap=t,symbol=f'sentinel_{m:02x}_{int(t)}') for m in MODES for t in (False,True)]


def elf_layout(raw):
    """Read ELF64 sections and function symbols; no LLVM installation required."""
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    require(h[0][:6]==b'\x7fELF\x02\x01' and h[1]==3 and h[2]==224,'Expected AMDGPU ELF64LE shared object')
    require(h[11]==64 and h[12]>0,'Unexpected ELF section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,h[6]+i*h[11]) for i in range(h[12])]
    strings=sections[h[13]];names=raw[strings[4]:strings[4]+strings[5]]
    named={names[s[0]:].split(b'\0',1)[0].decode():s for s in sections}
    symbols={}
    for s in sections:
        if s[1] not in (2,11):continue
        strings=sections[s[6]];names=raw[strings[4]:strings[4]+strings[5]]
        require(s[9]==24,'Unexpected symbol record size')
        for offset in range(s[4],s[4]+s[5],s[9]):
            name,info,other,index,address,size=struct.unpack_from('<IBBHQQ',raw,offset)
            name=names[name:].split(b'\0',1)[0].decode()
            if name and info&15==2 and index!=0:
                value=dict(address=address,bytes=size,section=index)
                require(name not in symbols or symbols[name]==value,'Conflicting function symbol')
                symbols[name]=value
    functions={};ret=struct.pack('<I',0xb88df801);trap=struct.pack('<I',0xbf900003)
    for case in expected_cases():
        name=case['symbol'];symbol=symbols[name];section=sections[symbol['section']]
        start=section[4]+symbol['address']-section[3];code=raw[start:start+symbol['bytes']]
        returns=[i for i in range(0,len(code),4) if code[i:i+4]==ret]
        require(len(returns)==1,'Expected unique post-trap MODE read')
        target=returns[0]
        require(code[target-4:target]==(trap if case['trap'] else struct.pack('<I',0xbf800000)),
                'Exact S_TRAP 3 / NOP return site differs')
        require(sum(code[i:i+4]==trap for i in range(0,len(code),4))==int(case['trap']),'Wrong explicit trap count')
        functions[name]=dict(address=symbol['address'],bytes=symbol['bytes'],kernel_sha256=digest(code),
                             return_PC_offset=symbol['address']+target,explicit_S_TRAP_3=case['trap'])
    require(set(functions)==set(symbols),'Unexpected extra native functions')
    return {'functions':functions,'sections':{name:dict(bytes=s[5],sha256=digest(raw[s[4]:s[4]+s[5]]))
                                             for name,s in named.items() if name in ('.text','.rodata','.note')}}


def verify_exact():
    for name,(size,sha) in EXACT.items():
        raw=(ROOT/'exact'/name).read_bytes()
        require(len(raw)==size and digest(raw)==sha,'Exact executed asset changed: '+name)
    cfg=json.loads((ROOT/'exact/sentinel_host_configuration_v2.json').read_text())
    require(cfg['cases']==expected_cases() and cfg['grid']==[1,1,1] and cfg['block']==[32,1,1]
            and cfg['packet_words']==40 and cfg['kernarg_bytes']==8 and cfg['shared_bytes']==0,'Frozen ABI/cases differ')
    code=(ROOT/'exact/sentinel_shader_root_v2.hsaco').read_bytes()
    return cfg,code,elf_layout(code)


def check_packet(raw,case,expect):
    require(expect in ('original','fixed'),'Choose original or fixed expectation')
    require(len(raw)==160,'Expected exactly 160 packet bytes')
    w=list(struct.unpack('<40I',raw));m=case['mode_low_byte']
    tags=[0x11000000*(bank+1)+lane for bank in range(3) for lane in (0,1,16)]
    after=list(tags);source=m&3;destination=(m>>6)&3
    copy_expected=expect=='original' and case['trap'] and source!=destination
    if copy_expected:after[3*destination]=tags[3*source]
    mode=((m>>6)&3)|((m&3)<<2)|(((m>>2)&3)<<4)|(((m>>4)&3)<<6)
    checks={
      'header_and_case':w[:4]==[0x53545250,1,m,int(case['trap'])],
      'completion_and_reserved':w[32:]==[0x434f4d50]+[0xa5a5a5a5]*7,
      'all_nine_initial_tags':w[8:17]==tags,
      'all_nine_final_tags':w[17:26]==after,
      'requested_MODE_bank_fields':((w[4]>>12)&255)==mode,
      'full_MODE_preserved':w[4]==w[5],
      'full_wave_EXEC_preserved':w[30]==w[31]==0xffffffff,
      'trap_enabled_before_and_after':bool(w[6]&64) and bool(w[7]&64),
      'full64_trap_return_PC':not case['trap'] or w[26:28]==w[28:30],
    }
    return dict(pass_all=all(checks.values()),checks=checks,case=case,
        expected_classification='PREDICTED_LANE0_CROSS_BANK_COPY' if copy_expected else 'UNCHANGED',
        changed_tags=[dict(vgpr=1+256*(i//3),lane=(0,1,16)[i%3],before=f'0x{a:08x}',after=f'0x{b:08x}')
                      for i,(a,b) in enumerate(zip(w[8:17],w[17:26])) if a!=b],
        intended_return_PC=(w[27]<<32)|w[26],returned_TTMP=(w[29]<<32)|w[28])


def consume_packets(rows,expect,layout):
    require([r['case'] for r in rows]==expected_cases(),'Expected exact ordered 14 cases')
    observations=[];bias=None
    for row in rows:
        value=check_packet(row['raw'],row['case'],expect)
        current=value['intended_return_PC']-layout['functions'][row['case']['symbol']]['return_PC_offset']
        if bias is None:bias=current
        value['checks']['native_return_PC_target']=current==bias and current>0 and current%4096==0
        value['pass_all']=all(value['checks'].values());observations.append(value)
    return dict(status='PASS_EXPECTED_'+expect.upper()+'_SENTINEL' if all(r['pass_all'] for r in observations) else 'FAIL_SENTINEL_ORACLE',
                expectation=expect,cases=observations,all_14_complete=len(observations)==14,
                changed_cases=sum(bool(r['changed_tags']) for r in observations),module_load_bias=bias)


def check_witnesses(layout):
    witnesses=json.loads((ROOT/'evidence/witnesses.json').read_text());results={}
    for role,recording in witnesses['recordings'].items():
        rows=[]
        for row in recording['cases']:
            raw=(ROOT/row['packet']['package_path']).read_bytes()
            require(len(raw)==row['packet']['bytes'] and digest(raw)==row['packet']['sha256'],'Saved witness packet changed')
            rows.append(dict(case=row['case'],raw=raw))
        result=consume_packets(rows,role,layout)
        require(result['status']=='PASS_EXPECTED_'+role.upper()+'_SENTINEL','Saved witness oracle failed')
        results[role]=dict(status=result['status'],cases=14,changed_cases=result['changed_cases'])
    require(set(results)=={'original','fixed'},'Both observed driver witnesses required')
    return results


def load_host():
    spec=importlib.util.spec_from_file_location('exact_executed_sentinel_host',ROOT/'exact/run_sentinel_host_v1.py')
    host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
    return host


def run(args,cfg,code,layout):
    require(args.expect is not None and args.output is not None,'Run requires --expect and --output')
    require(args.hip_library is not None and args.hsa_library is not None,'Run requires explicit HIP and HSA library paths')
    require(not os.path.lexists(args.output),'Output must be a fresh directory')
    require(args.output.parent.is_dir() and args.output.parent.resolve()==args.output.parent,'Canonical existing output parent required')
    require(0<=args.device<64,'Invalid device')
    host=load_host()
    libraries={key:host.record(path.resolve(strict=True)) for key,path in [('hip',args.hip_library),('hsa',args.hsa_library)]}
    cfg=copy.deepcopy(cfg);cfg.update(libraries=libraries,module=host.record(ROOT/'exact/sentinel_shader_root_v2.hsaco'),device=args.device)
    # The original host and native image stay byte-identical. Only its recorded
    # runtime paths and irrelevant training environment contract are supplied here.
    host.LIBRARY_PINS=libraries;host.REQUIRED_ENV={}
    args.output.mkdir()
    identity=host.write_json(args.output/'portable_configuration.json',cfg)
    cfg,identity,checked_code=host.check_configuration(identity['path'],identity['sha256'])
    require(checked_code==code,'Module changed before launch')
    host.write_json(args.output/'adapter.json',dict(status='PORTABLE_WRAPPER_WITH_EXACT_EXECUTED_HOST_AND_SHADER',
        wrapper=host.record(__file__),original_host=host.record(ROOT/'exact/run_sentinel_host_v1.py'),
        native=cfg['module'],libraries=libraries,expectation=args.expect,
        portability_changes=['Host library pins supplied from explicit local runtime paths.',
                             'Unused training-specific REQUIRED_ENV changed to empty.',
                             'Configuration module path and selected device rebound; all cases/ABI/native bytes unchanged.'],
        kernel_release=os.uname().release,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    transport=args.output/'transport';rc=host.run(cfg,identity,code,transport)
    if rc:return rc
    summary=json.loads((transport/'result.json').read_text())
    require(summary['transport_status']=='COMPLETE_ALL_14_CASES' and summary['host_exit_code']==0
            and summary['completed_transport_cases']==14,'Incomplete HIP transport')
    rows=[]
    for ordinal,row in enumerate(summary['cases']):
        case=row['case'];base=transport/f'case_{ordinal:02d}_{case["mode_low_byte"]:02x}_{int(case["trap"])}'
        require(row['transport_status']=='PASS','Failed case transport')
        for kind in ('output','prefill_actual','prefill_expected'):
            require(Path(row[kind]['path'])==base/(kind+'.bin'),'Unexpected packet path')
            host.pin(row[kind])
        require((base/'prefill_actual.bin').read_bytes()==(base/'prefill_expected.bin').read_bytes()==b'\xa5'*160,'Prefill differs')
        rows.append(dict(case=case,raw=(base/'output.bin').read_bytes()))
    verdict=consume_packets(rows,args.expect,layout)
    verdict['transport_result']=host.record(transport/'result.json')
    verdict['process_scope']='Caller must retain this process exit and reap it; no timeout/recovery is implemented in this adapter.'
    host.write_json(args.output/'verdict.json',verdict)
    print(json.dumps({k:verdict[k] for k in ('status','expectation','all_14_complete','changed_cases')}),flush=True)
    return int(verdict['status']=='FAIL_SENTINEL_ORACLE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--suite',choices=['sentinel'],default='sentinel')
    parser.add_argument('--expect',choices=['original','fixed']);parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--hip-library',type=Path);parser.add_argument('--hsa-library',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg,code,layout=verify_exact();witnesses=check_witnesses(layout)
    if args.check_only:
        print(json.dumps(dict(status='PASS_CPU_ONLY_PORTABLE_SENTINEL_CHECK',exact_native_sha256=digest(code),
            native_functions=len(layout['functions']),explicit_traps=7,cases=14,packet_bytes=160,
            observed_witnesses=witnesses,CDLL_called=False,HIP_initialized=False)),flush=True);return 0
    if args.output is not None:args.output=args.output.absolute()
    return run(args,cfg,code,layout)


if __name__=='__main__':raise SystemExit(main())
