#!/usr/bin/env python3
"""One root-owned portable sentinel run using current original-driver live gates."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import traceback

ROOT=Path(__file__).resolve().parent
B=ROOT.parent
RUNTIME=Path('/home/chcai/dlrm_data/root_cause_20260919/session_20260928_driver_ticket_runtime_v1')
STAGE=RUNTIME/'reproducer'
SOURCE=B/'session_20260928_driver_ticket_v1/reproducer'
EXECUTION=ROOT/'execution'
PINS={
 'controller':dict(path=str(B/'session_20260927_original_DW_root_v1/controller_v2.py'),bytes=12375,sha256='2b68906cbee23d8dcc0be3492cd36de74f0940df4d5ecad8a79ef9d80e1957f7'),
 'binding':dict(path=str(B/'session_20260927_original_DW_root_v1/BINDING_v2.json'),bytes=7717,sha256='7ee26fa55b342f08bb86691cab18d415cfb6b60c1dcbdb2f2fae572b1fbdc8c8'),
 'package':dict(path=str(SOURCE/'SOURCE.json'),bytes=13663,sha256='04265d023cc6d924e5724c40ab9ae4f78abf50fb4c7eeda29a17a4ce35314dd1'),
 'package_peer':dict(path=str(B/'session_20260928_driver_ticket_peer_v1/REPRODUCER_REVIEW.json'),bytes=15732,sha256='302754824556fa1269de12b2a3146f187324fde2520664c11b3b39a8ed86aba6'),
 'archive':dict(path=str(B/'session_20260928_original_DW_archive_root_v1/execution/terminal.json'),bytes=5632,sha256='0b1fda52bf97025e24f8a27584c0f597457161fe2f95d8d735f1d73d541d9384'),
 'recorder':dict(path='/home/chcai/dlrm_data/root_cause_20260919/session_20260927_original_DW_runtime_v1/record_then_exec.py',bytes=1101,sha256='f15a6420ffe817dfef8a0c46c361fbcf2210e5f1f2d680bb5aea01cc48365056'),
}
SUCCESS='PASS_ROOT_PORTABLE_ORIGINAL_SENTINEL_REPRODUCTION'


def require(ok,message):
    if not ok:raise ValueError(message)


def pin(path):
    path=Path(path);before=path.stat();require(before.st_size<4<<20,'Only small source/result reads permitted')
    raw=path.read_bytes();after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed while read')
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def check(row):
    require(pin(row['path'])==row,'Pinned source/result changed: '+row['path']);return Path(row['path'])


def read(row):
    raw=check(row).read_bytes()
    require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Changed JSON reread')
    return json.loads(raw)


def save(path,value):
    raw=value if isinstance(value,bytes) else (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return pin(path)


def module(name,row):
    spec=importlib.util.spec_from_file_location(name,check(row));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def container_path(path):
    path=str(path);prefix='/home/chcai/dlrm_data'
    require(path.startswith(prefix+'/'),'Unexpected host runtime path')
    return '/data/mlperf_dlrm_v4'+path[len(prefix):]


def host_pin(row):
    prefix='/data/mlperf_dlrm_v4';path=row['path']
    require(path.startswith(prefix+'/'),'Unexpected container result path')
    result=dict(row,path='/home/chcai/dlrm_data'+path[len(prefix):]);check(result);return result


def package_files():
    package=read(PINS['package']);peer=read(PINS['package_peer'])
    require(peer['status']=='PASS_INDEPENDENT_PORTABLE_SENTINEL_CPU_SOURCE_AND_ORACLE_REVIEW'
            and peer['unresolved_findings']==[],'Portable source peer failed')
    rows=package['files'];require(len(rows)==41 and len({r['package_path'] for r in rows})==41,'Wrong source inventory')
    sources=[]
    for row in rows:
        relative=Path(row['package_path']);expected=SOURCE/relative
        require(not relative.is_absolute() and '..' not in relative.parts and relative!=Path('SOURCE.json')
                and row['path']==str(expected),'Invalid package source path')
        item={k:row[k] for k in ('path','bytes','sha256')};check(item);sources.append((relative,item))
    require((Path('run_repro.py'),peer['source']) in sources and (Path('build_repro.py'),peer['builder']) in sources,
            'Reviewed portable program/builder absent')
    return sources+[(Path('SOURCE.json'),PINS['package'])]


def verify_release(release_pin):
    release=read(release_pin)
    require(release['status']=='ROOT_PORTABLE_SENTINEL_READY' and release['controller']==pin(__file__),'Wrong root release')
    source=read(release['source']);peer=read(release['peer'])
    require(release['source']['path']==str(ROOT/'SOURCE.json') and release['controller'] in source['files'],'Wrong controller source')
    require(peer['status']=='PASS_INDEPENDENT_PORTABLE_SENTINEL_ROOT_CONTROLLER_REVIEW'
            and peer['controller']==release['controller'] and peer['source']==release['source']
            and peer['findings']==[],'Wrong root controller peer')
    for row in source['files']:check(row)
    for row in PINS.values():check(row)
    package_files();return release


def verify_stage(copies):
    require(len(copies)==42,'Expected exact stage count')
    for row in copies:
        check(row['source']);check(row['destination'])
        require((row['source']['bytes'],row['source']['sha256'])==
                (row['destination']['bytes'],row['destination']['sha256']),'Copied source differs')
    require({str(p) for p in STAGE.rglob('*') if p.is_file()}=={r['destination']['path'] for r in copies}
            and all(not p.is_symlink() for p in STAGE.rglob('*')),'Staged file inventory differs')


def commands(plan):
    prefix=plan['serial_arms'][0]['command'];prefix=prefix[:prefix.index('python3')]
    require(prefix[:2]==['docker','exec'] and 'LD_PRELOAD' in prefix and 'HIP_VISIBLE_DEVICES' in prefix,'Wrong preserved environment prefix')
    runner=container_path(STAGE/'run_repro.py')
    cpu=prefix+['python3','-B',runner,'--check-only']
    gpu=['python3','-B',runner,'--suite','sentinel','--expect','original','--hip-library',
         '/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7','--hsa-library',
         '/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1',
         '--device','0','--output',container_path(RUNTIME/'result')]
    producer=prefix+['python3','-B',container_path(PINS['recorder']['path']),'--expected-boot-id',plan['expected_boot'],
                     '--receipt',container_path(RUNTIME/'producer_process.json'),'--']+gpu
    return cpu,producer


def consume(receipt,copies):
    verify_stage(copies)
    runner=module('staged_portable_sentinel',pin(STAGE/'run_repro.py'));cfg,code,layout=runner.verify_exact()
    vp=pin(RUNTIME/'result/verdict.json');verdict=read(vp)
    sp=pin(RUNTIME/'result/transport/result.json');summary=read(sp)
    require(host_pin(verdict['transport_result'])==sp,'Verdict/transport identity mismatch')
    require(summary['transport_status']=='COMPLETE_ALL_14_CASES' and summary['host_exit_code']==0
            and summary['completed_transport_cases']==14 and summary['pid']==receipt['pid'],'Incomplete/mismatched producer')
    require(host_pin(summary['host_source'])==pin(STAGE/'exact/run_sentinel_host_v1.py'),'Wrong original host')
    configuration_pin=host_pin(summary['configuration'])
    require(configuration_pin['path']==str(RUNTIME/'result/portable_configuration.json'),'Wrong portable configuration path')
    config=read(configuration_pin)
    require(all(config[key]==cfg[key] for key in ('schema','packet_words','kernarg_bytes','shared_bytes','prefill_byte','layout'))
            and summary['required_environment']=={},'Portable ABI/environment differs')
    adapter_pin=pin(RUNTIME/'result/adapter.json');adapter=read(adapter_pin)
    require(host_pin(adapter['wrapper'])==pin(STAGE/'run_repro.py')
            and host_pin(adapter['original_host'])==pin(STAGE/'exact/run_sentinel_host_v1.py')
            and adapter['native']==config['module'] and adapter['libraries']==config['libraries']
            and adapter['expectation']=='original','Wrong portable adapter/source identity')
    require(config['cases']==cfg['cases'] and config['grid']==[1,1,1] and config['block']==[32,1,1]
            and config['device']==0 and host_pin(config['module'])==pin(STAGE/'exact/sentinel_shader_root_v2.hsaco'),'Wrong native/launch/config')
    for phase in ('maps_after_init','maps_at_terminal'):
        maps=summary[phase];host_pin(maps['maps'])
        for key,expected in cfg['libraries'].items():
            actual=maps['libraries'][key]['pinned_file']
            require((actual['bytes'],actual['sha256'])==(expected['bytes'],expected['sha256'])
                    and actual==config['libraries'][key],'Mapped runtime changed')
    expected_functions=['hipInit','hipGetDeviceCount','hipSetDevice','hipRuntimeGetVersion','hipDriverGetVersion',
                        'hipDeviceGetName','hipModuleLoadData','hipMalloc']
    expected_functions+=['hipModuleGetFunction','hipMemcpy','hipMemcpy','hipModuleLaunchKernel','hipDeviceSynchronize','hipMemcpy']*14
    expected_functions+=['hipFree','hipModuleUnload']
    calls=summary['hip_calls']
    require(len(calls)==94 and [r['function'] for r in calls]==expected_functions
            and all(r['result_code']==0 and r['index']==i and r['phase']=='end' for i,r in enumerate(calls)),'94 HIP calls not complete/successful')
    journal=pin(RUNTIME/'result/transport/hip_calls.jsonl')
    events=[json.loads(line) for line in check(journal).read_text().splitlines()]
    require(len(events)==188 and all(events[2*i+1]==call and events[2*i]['phase']=='begin'
            and events[2*i]['index']==i and events[2*i]['function']==call['function']
            and events[2*i]['context']==call['context'] and events[2*i]['started_utc']==call['started_utc']
            for i,call in enumerate(calls)),
            'HIP begin/end journal differs')
    require([r['context'] for r in calls[:8]]==['initialization']*6+['module_load','output_allocation']
            and [r['context'] for r in calls[-2:]]==['cleanup']*2,'HIP initialization/cleanup contexts differ')
    rows=[];packets=[];case_receipts=[]
    pointer=int(summary['output_device_pointer'],16);require(0<pointer<1<<64,'Invalid output device pointer')
    require([r['case'] for r in summary['cases']]==runner.expected_cases(),'Wrong case inventory')
    for i,row in enumerate(summary['cases']):
        require(row['transport_status']=='PASS' and row['ordinal']==i,'Incomplete case')
        base=RUNTIME/'result/transport'/f'case_{i:02d}_{row["case"]["mode_low_byte"]:02x}_{int(row["case"]["trap"])}'
        case_pin=pin(base/'result.json');require(read(case_pin)==row,'Case result/summary differs');case_receipts.append(case_pin)
        begin=8+6*i;end=begin+6
        require(row['call_index_begin']==begin and row['call_index_end_exclusive']==end
                and row['calls']==calls[begin:end] and all(call['context']==i for call in calls[begin:end])
                and row['output_device_pointer']==summary['output_device_pointer']
                and row['kernarg_output_pointer_bytes']==pointer.to_bytes(8,'little').hex(),
                'Case process/argument/call identity differs')
        for name in ('output','prefill_actual','prefill_expected'):
            item=host_pin(row[name]);require(item['path']==str(base/(name+'.bin')),'Wrong retained packet path')
            packets.append(item)
        require((base/'prefill_actual.bin').read_bytes()==(base/'prefill_expected.bin').read_bytes()==b'\xa5'*160,'Prefill mismatch')
        rows.append(dict(case=row['case'],raw=(base/'output.bin').read_bytes()))
    expected=runner.consume_packets(rows,'original',layout)
    require(expected['status']=='PASS_EXPECTED_ORIGINAL_SENTINEL' and expected['changed_cases']==3
            and expected['all_14_complete'] is True and all(r['pass_all'] for r in expected['cases']), 'Portable numerical oracle failed')
    require(all(verdict[key]==value for key,value in expected.items()),'Saved/fresh verdict differs')
    return dict(verdict=vp,transport=sp,journal=journal,adapter=adapter_pin,configuration=configuration_pin,
                case_receipts=case_receipts,packets=packets,complete_HIP_calls=94,
                complete_packet_bytes=2240,original_predicted_copy_cases=3,all_other_cases_unchanged=True)


def execute(ctx,controller,release_pin):
    plan=ctx[2];base=ctx[3];helper=ctx[7];archive=read(PINS['archive'])
    require(archive['status']=='PASS_ROOT_SUPERVISED_ORIGINAL_DW_ARCHIVE_RAW_RETAINED'
            and archive['archive_process_may_be_alive'] is False,'Archive incomplete')
    require(archive['process']['status']=='EXITED' and archive['process']['returncode']==0
            and archive['process']['process_wait_completed'] is True,'Archive process incomplete')
    previous=archive['last_kernel_state'];require(read(archive['final_gate'])['kernel_state']==previous,'Wrong archive predecessor')
    with base.lock():
        require(not os.path.lexists(EXECUTION) and not os.path.lexists(RUNTIME),'Fresh execution/runtime required')
        require(RUNTIME.parent.resolve()==RUNTIME.parent,'Linked runtime parent')
        EXECUTION.mkdir()
        result=dict(status='FAIL_PORTABLE_REPRO_EVIDENCE_PRESERVED',release=release_pin,controller=pin(__file__),
                    previous_kernel_state=previous,GPU_actions=False,driver_actions=False,producer_may_be_alive=False,
                    process_signal_sent=False,raw_files_removed=False)
        save(EXECUTION/'START.json',result)
        try:
            gate,_=controller.common_gate(ctx,previous,EXECUTION/'initial_gate','before portable sentinel stage')
            result['initial_gate']=gate;previous=read(gate)['kernel_state']
            verify_release(release_pin);RUNTIME.mkdir();STAGE.mkdir();copies=[]
            for relative,source in package_files():
                destination=STAGE/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                copies.append(dict(source=source,destination=save(destination,check(source).read_bytes())))
            verify_stage(copies);result['stage']=save(RUNTIME/'STAGE.json',dict(status='PASS_EXACT_42_FILE_STAGE',package=PINS['package'],files=copies))
            cpu,producer=commands(plan)
            checked=helper.run_step(cpu,EXECUTION/'CPU_preflight',120);result['CPU_preflight']=checked
            require(checked['status']=='EXITED' and checked['returncode']==0 and checked['process_wait_completed'] is True,'CPU preflight incomplete')
            pre=json.loads((EXECUTION/'CPU_preflight/stdout.log').read_text())
            require(pre['status']=='PASS_CPU_ONLY_PORTABLE_SENTINEL_CHECK' and pre['CDLL_called'] is False
                    and pre['HIP_initialized'] is False and pre['cases']==14,'Wrong CPU preflight receipt')
            library_dir=EXECUTION/'libraries';library_dir.mkdir()
            result['container']=base.container(ctx[4],False,library_dir,helper)
            gate,_=controller.common_gate(ctx,previous,EXECUTION/'release_gate','immediately before portable sentinel')
            result['release_gate']=gate;previous=read(gate)['kernel_state']
            verify_release(release_pin);verify_stage(copies);check(PINS['recorder'])
            result.update(producer_may_be_alive=True,GPU_actions=True)
            produced=helper.run_step(producer,EXECUTION/'producer',900);result['producer']=produced
            require(produced['status']=='EXITED' and produced['process_wait_completed'] is True,'Producer timeout/unreaped: no signal/retry')
            c=dict(boot_id=plan['expected_boot'],container_name=plan['container_name'],
                   repro_command=producer,repro_process_receipt=str(RUNTIME/'producer_process.json'))
            live,receipt,alive=base.producer_absent(c,'repro',EXECUTION/'producer_liveness.json')
            result.update(producer_liveness=live,producer_may_be_alive=False)
            gate,_=controller.common_gate(ctx,previous,EXECUTION/'after_producer_gate','after reaped portable sentinel')
            result['after_producer_gate']=gate;previous=read(gate)['kernel_state']
            require(produced['returncode']==0,'Portable producer failed')
            result['numerical_evidence']=consume(receipt,copies)
            verify_release(release_pin);verify_stage(copies)
            gate,_=controller.common_gate(ctx,previous,EXECUTION/'final_gate','after complete portable sentinel consumption')
            result['final_gate']=gate;previous=read(gate)['kernel_state'];result['status']=SUCCESS
        except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc())
        finally:
            result['last_kernel_state']=previous;terminal=save(EXECUTION/'terminal.json',result)
    print(json.dumps(dict(status=result['status'],terminal=terminal)),flush=True)
    return int(result['status']!=SUCCESS)


def main():
    require(__debug__ and os.geteuid()==0 and 'PYTHONOPTIMIZE' not in os.environ,'Root nonoptimized Python required')
    p=argparse.ArgumentParser();p.add_argument('--release-sha256',required=True);args=p.parse_args()
    release_pin=pin(ROOT/'RELEASE.json');require(release_pin['sha256']==args.release_sha256,'Wrong root release SHA')
    verify_release(release_pin);controller=module('pinned_DW_controller',PINS['controller'])
    ctx=controller.load(PINS['binding']['sha256']);require(ctx[0]==PINS['binding'],'Wrong DW context')
    return execute(ctx,controller,release_pin)


if __name__=='__main__':raise SystemExit(main())
