#!/usr/bin/env python3
"""Constant-nonzero control for the original separated-RNG HSTU forward.

Logical X=1,U=.5,W=1,B=0,mask=7 and dropout=.5 require exact concatenated
Y=[1,2,0] and Mean=1 when D is a power of two. Rstd uses a separate tolerance.
CPU source/geometry inspection is the default; GPU execution is explicit.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import repro_ln_mul_dropout_forward_zero as zero

common, support, replay = zero.common, zero.support, zero.replay
DEFAULT_CONFIG = {**zero.DEFAULT_CONFIG, 'dropout_ratio':.5}
REVIEWED_ZERO_SHA256 = '33467906c06287963819b0e710d08bdb903b217e44933c6a9075d6e57843f177'
INPUT_RECIPE = {'X':1., 'U':.5, 'W':1., 'B':0., 'RANDOM_MASK':7}
DESIGN = {
    'variant':'constant_nonzero_exact_1_2_0_v1', 'logical_inputs':INPUT_RECIPE,
    'expected_y_sections':[1.,2.,0.], 'expected_mean':1.,
    'dropout_intervention':{'source_default':.3, 'control':.5, 'historical_failed_dispatch_value':'not captured'},
    'proof':'Power-of-two D gives exact sum D and reciprocal1/D, so Mean1, centeredX0 and variance0. Kept U=.5 and X1 scaled by2 give sections1 and2; normalized multiplied section remains0.',
    'scope':'Nonzero constant control; identical-value stale reads and errors masked by zero variance remain possible.'}
LIMITS = [line.replace('Exact zero is required for Y and Mean',
                      'Exact constants1/2/0 are required for the three Y sections and1 for Mean')
          for line in zero.LIMITS] + [
    'This explicit dropout0.5 intervention differs from the source default0.3; compiled instruction identity is compared separately.',
    'Constant X has zero variance and does not exercise general nonzero normalization arithmetic. A pass does not clear the historical abort.',
]


def require_reviewed_helper():
    actual=support.file_hash(Path(zero.__file__))
    if actual!=REVIEWED_ZERO_SHA256:
        raise ValueError('Reviewed frozen zero helper identity differs')
    return actual


def geometry_plan(config,chunk_elements=1<<24):
    plan=zero.geometry_plan(config,chunk_elements)
    d=config['cols']
    if d & (d-1):
        raise ValueError('Exact constant Mean requires power-of-two feature width')
    if config['dropout_ratio']!=.5:
        raise ValueError('This exact1/2/0 control requires dropout_ratio0.5')
    return {**plan,'control_design':copy.deepcopy(DESIGN),
            'exact_oracle':{'Y_sections':[1.,2.,0.],'Mean':1.}}


def make_buffers(config,device='cpu'):
    geometry_plan(config)
    state=zero.make_buffers(config,device)
    for name,tensor in state['inputs'].items():
        tensor.fill_(INPUT_RECIPE[name])
    return state


def check_inputs(inputs,chunk_elements=1<<24):
    torch=common._torch()
    if type(chunk_elements) is not int or chunk_elements<1 or set(inputs)!=set(INPUT_RECIPE):
        raise ValueError('Exact input mapping and positive chunk limit required')
    rows,names=[],[]
    for name,tensor in inputs.items():
        raw=zero.flat_storage(tensor)
        offset=tensor.storage_offset()
        span=1+sum((size-1)*step for size,step in zip(tensor.shape,tensor.stride()))
        exterior=[raw[:offset],raw[offset+span:]]
        if tensor.ndim==2 and tensor.shape[0]>1 and tensor.stride(0)>tensor.shape[1]:
            exterior.append(raw.as_strided((tensor.shape[0]-1,tensor.stride(0)-tensor.shape[1]),
                                            (tensor.stride(0),1),offset+tensor.shape[1]))
        def differences(regions,number):
            bits=zero.expected_bits(tensor.dtype,number)
            partial=[(chunk.view(zero.integer_dtype(chunk))!=bits).sum() for region in regions
                     for chunk in common._logical_chunks(region,chunk_elements) if chunk.numel()]
            return torch.stack(partial).sum() if partial else torch.zeros((),dtype=torch.int64,device=tensor.device)
        canary=zero.INPUT_CANARY_MASK if name=='RANDOM_MASK' else zero.INPUT_CANARY_FLOAT
        rows.append(torch.stack((differences([tensor],INPUT_RECIPE[name]),differences(exterior,canary))));names.append(name)
    values=torch.stack(rows).cpu().tolist()
    return {'failed':any(any(v) for v in values),
            'tensors':{name:{'different_backing_elements':int(sum(v)),'different_logical_elements':int(v[0]),
                            'different_guard_or_padding_elements':int(v[1])} for name,v in zip(names,values)},
            'scope':'Every complete input backing element, with exact constant logical bits and distinct17/85 guard/padding canaries'}


def check_outputs(outputs,config,chunk_elements=1<<24):
    torch=common._torch();plan=geometry_plan(config,chunk_elements)
    if set(outputs)!={'Y','Mean','Rstd'}:
        raise ValueError('Exactly Y,Mean,Rstd outputs required')
    results,names,section_results=[],[],[]
    for name,tensor in outputs.items():
        spec=plan['buffers'][name]
        dtype=torch.bfloat16 if name=='Y' else torch.float32
        if (tensor.dtype!=dtype or list(tensor.shape)!=spec['shape'] or list(tensor.stride())!=spec['stride']
                or tensor.storage_offset()!=spec['storage_offset'] or tensor.untyped_storage().nbytes()!=spec['bytes']):
            raise ValueError('Output geometry/dtype differs: '+name)
        if name=='Y':
            d=config['cols'];regions=[(tensor[:,i*d:(i+1)*d],expected) for i,expected in enumerate((1.,2.,0.))]
        else:
            regions=[(tensor,1. if name=='Mean' else None)]
        per_region=[]
        for region,expected in regions:
            chunks=[]
            for value in common._logical_chunks(region,chunk_elements):
                finite=torch.isfinite(value)
                if name=='Rstd':
                    error=(value.double()-plan['expected_rstd']).abs()
                    mismatch=(~finite)|(error>plan['expected_rstd']*plan['rstd_relative_tolerance'])
                    maximum=torch.where(finite,error,0.).max()
                else:
                    integer=value.view(zero.integer_dtype(value))
                    mismatch=(integer & ((1<<(8*value.element_size()-1))-1))!=0 if expected==0 else integer!=zero.expected_bits(value.dtype,expected)
                    maximum=torch.zeros((),dtype=torch.float64,device=value.device)
                chunks.append(torch.stack((mismatch.sum().double(),(~finite).sum().double(),maximum)))
            matrix=torch.stack(chunks)
            per_region.append(torch.stack((matrix[:,0].sum(),matrix[:,1].sum(),matrix[:,2].max())))
        regions_matrix=torch.stack(per_region)
        raw=zero.flat_storage(tensor).view(zero.integer_dtype(tensor));g=config['guard_elements'];tail=g+spec['span_elements']
        canary=zero.expected_bits(tensor.dtype,zero.OUTPUT_CANARY)
        guards=(raw[:g]!=canary).sum()+(raw[tail:]!=canary).sum()
        results.append(torch.stack((regions_matrix[:,0].sum(),regions_matrix[:,1].sum(),regions_matrix[:,2].max(),guards.double())))
        names.append(name)
        if name=='Y':section_results=per_region
    values=torch.stack(results).cpu().tolist()
    sections=torch.stack(section_results).cpu().tolist()
    return {'failed':any(v[0] or v[3] for v in values),
            'tensors':{name:{'mismatched_elements':int(v[0]),'nonfinite_elements':int(v[1]),
                            'max_abs_rstd_error':v[2],'changed_guard_elements':int(v[3])} for name,v in zip(names,values)},
            'y_sections':[{'expected':expected,'mismatched_elements':int(v[0]),'nonfinite_elements':int(v[1])}
                          for expected,v in zip((1.,2.,0.),sections)],
            'expected_mean':1.,'rstd_reference':plan['expected_rstd'],'rstd_relative_tolerance':plan['rstd_relative_tolerance']}


class OriginalLauncher(zero.OriginalLauncher):
    def __init__(self,state,config,code_directory):
        geometry_plan(config)
        super().__init__(state,config,code_directory)
        body=Path(__file__).read_bytes()
        self.evidence={**self.evidence,'constant_control_design':copy.deepcopy(DESIGN),
                       'constant_driver_source_sha256':support.file_hash(Path(__file__))}
        directory=Path(code_directory)
        support.atomic_new_save(directory/Path(__file__).name,lambda stream:stream.write(body))
        support.atomic_new_save(directory/'constant_control.json',lambda stream:stream.write((json.dumps(self.evidence,indent=2)+'\n').encode()))


def source_inventory():
    return {**zero.source_inventory(),str(Path(__file__).resolve()):support.file_hash(Path(__file__))}


def run_loop(config,*,failure_path,repeats=1000,device='cpu',launch=None,launch_factory=None,
             max_resident_bytes=64<<30,max_capture_bytes=64<<30,chunk_elements=1<<24,
             chunk_bytes=64<<20,configuration=None,progress=None,before_call=None):
    require_reviewed_helper();torch=common._torch();support.new_path(failure_path)
    if type(repeats) is not int or not 1<=repeats<=1000 or type(chunk_bytes) is not int or chunk_bytes<1:
        raise ValueError('Positive chunks and repeats1..1000 required')
    if (launch is None)==(launch_factory is None):
        raise ValueError('Exactly one explicit launch or factory required; no implicit CPU kernel')
    plan=geometry_plan(config,chunk_elements);resource=plan['resource_plan']
    if resource['resident_peak_before_backend_workspace_bytes']>max_resident_bytes or resource['complete_failure_storage_bytes']>max_capture_bytes:
        raise ValueError('Resource budget insufficient before allocation/dispatch')
    state=make_buffers(config,device)
    if check_inputs(state['inputs'],chunk_elements)['failed']:
        raise RuntimeError('Synthetic initialization differs from exact constant recipe')
    pristine=replay.digest_tree(state['inputs'],chunk_bytes)
    runtime={'configuration':copy.deepcopy(configuration),'geometry_plan':plan,'control_design':copy.deepcopy(DESIGN),
             'versions':replay.runtime_versions(),'source_sha256_before':source_inventory(),'device':str(device),
             'repeats':repeats,'synthetic_recipe':dict(INPUT_RECIPE),
             'input_guard_and_padding_recipe':{'BF16':zero.INPUT_CANARY_FLOAT,'int8':zero.INPUT_CANARY_MASK},
             'output_initialization_recipe':{'logical':'NaN poison every iteration; no payload assumed','guard_and_padding':zero.OUTPUT_CANARY},
             'reviewed_zero_helper_sha256':REVIEWED_ZERO_SHA256,'limits':LIMITS}
    if launch_factory is not None:launch=launch_factory(state,config)
    runtime['kernel']=copy.deepcopy(getattr(launch,'evidence',{'injected_callable':True}))
    records=[]
    for iteration in range(1,repeats+1):
        zero.poison_outputs(state['outputs'])
        if before_call:before_call({'status':'DISPATCHING','iteration':iteration,'runtime':runtime})
        started=time.monotonic();launch(state,config)
        outputs=check_outputs(state['outputs'],config,chunk_elements);inputs=check_inputs(state['inputs'],chunk_elements)
        record={'iteration':iteration,'failed':outputs['failed'] or inputs['failed'],'outputs':outputs,'inputs':inputs,
                'seconds_including_checks':time.monotonic()-started}
        if record['failed'] or iteration==repeats:
            current_inputs=replay.digest_tree(state['inputs'],chunk_bytes)
            record['input_hashes_match_initial']=current_inputs==pristine
            record['failed']|=not record['input_hashes_match_initial']
        records.append(record)
        if record['failed']:
            from nan_backward_boundaries import _cpu_copy_tree
            observed=replay.digest_tree(state,chunk_bytes)
            copied,stored=_cpu_copy_tree(state,chunk_bytes,max_capture_bytes)
            if (replay.digest_tree(copied,chunk_bytes)!=observed
                    or check_outputs(copied['outputs'],config,chunk_elements)!=outputs
                    or check_inputs(copied['inputs'],chunk_elements)!=inputs):
                raise RuntimeError('First-failure full copy differs in bytes or classification')
            artifact={'format':'ln_mul_dropout_forward_constant_failure_v1','iteration':iteration,
                      'configuration':dict(config),'current_at_failure':copied,'observed_record':record,
                      'initial_input_digest':pristine,'observed_digest':observed,'storage_bytes':stored,
                      'runtime':runtime,'source_sha256_at_failure':source_inventory(),
                      'capture_timing':'Existing first failed completed call; no rerun; original synthetic recipe and initial hashes retained'}
            support.atomic_new_save(failure_path,lambda stream:torch.save(artifact,stream))
            if progress:progress(record)
            return {'status':'FAIL','iterations_completed':iteration,'iterations':records,'runtime':runtime,
                    'failure_capture':{'path':str(failure_path),'sha256':support.file_hash(failure_path),
                                       'bytes':Path(failure_path).stat().st_size,'storage_bytes':stored}}
        if progress:progress(record)
    return {'status':'PASS','iterations_completed':repeats,'iterations':records,'runtime':runtime,'final_input_hashes_match_initial':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name,value in DEFAULT_CONFIG.items():parser.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--context',type=Path)
    parser.add_argument('--failure-dump',type=Path)
    parser.add_argument('--code-directory',type=Path)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--repeats',type=int,default=1000)
    parser.add_argument('--chunk-elements',type=int,default=1<<24)
    parser.add_argument('--copy-chunk-mib',type=int,default=64)
    parser.add_argument('--max-resident-gib',type=int,default=64)
    parser.add_argument('--max-capture-gib',type=int,default=64)
    parser.add_argument('--cpu-threads',type=int,default=2)
    args=parser.parse_args(argv);config={key:getattr(args,key) for key in DEFAULT_CONFIG}
    plan=geometry_plan(config,args.chunk_elements)
    if min(args.copy_chunk_mib,args.max_resident_gib,args.max_capture_gib,args.cpu_threads)<1 or not 1<=args.repeats<=1000:
        parser.error('Positive resources and repeats1..1000 required')
    if args.gpu and any(v is None for v in (args.context,args.failure_dump,args.code_directory)):
        parser.error('GPU requires --context, --failure-dump, --code-directory')
    files=[p for p in (args.report,args.failure_dump,args.context) if p is not None]
    paths=files+[p.with_name(p.name+'.tmp') for p in (args.report,args.failure_dump) if p is not None]
    if len({p.resolve() for p in paths})!=len(paths):parser.error('Source/output/temp paths must differ')
    for p in (args.report,args.failure_dump):
        if p is not None:support.new_path(p)
    if args.code_directory and (args.code_directory.exists() or any(p.resolve().is_relative_to(args.code_directory.resolve()) for p in files)):
        parser.error('Code directory must be new and separate from source/report/failure paths')
    root=Path(__file__).resolve().parents[1]
    report={'status':'PREPARING','geometry_plan':plan,'control_design':copy.deepcopy(DESIGN),'limits':LIMITS,
            'historical_candidate':zero.HISTORICAL_CANDIDATE,
            'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}}
    support.atomic_new_save(args.report,lambda stream:stream.write((json.dumps(report,indent=2)+'\n').encode()))
    def save():
        temporary=args.report.with_name(args.report.name+'.tmp');owned=False
        try:
            with temporary.open('xb') as stream:
                owned=True;stream.write((json.dumps(report,indent=2,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,args.report)
            descriptor=os.open(args.report.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        except BaseException:
            if owned:temporary.unlink(missing_ok=True)
            raise
    try:
        report['reviewed_zero_helper_sha256']=require_reviewed_helper()
        report['source_sha256_before']=source_inventory()
        report['address_source_audit']=zero.audit_address_source((root/zero.MODEL_SOURCE).read_text())
        if args.gpu:
            context=json.loads(args.context.read_text());report['context']={'path':str(args.context),'sha256':support.file_hash(args.context),'captured':context}
            report['restored_environment']=support.restore_environment(context)
            torch=common._torch();torch.set_num_threads(args.cpu_threads);replay.validate_runtime(context,replay.runtime_versions())
            for name in (zero.MODEL_SOURCE,'generative_recommenders/common.py','generative_recommenders/ops/utils.py','generative_recommenders/ops/triton/triton_addmm.py'):
                if context.get('source_sha256',{}).get(name)!=support.file_hash(root/name):raise ValueError('Original numerical source differs from context: '+name)
            sys.path.insert(0,str(root));report['device']={'name':torch.cuda.get_device_name(0),'properties':str(torch.cuda.get_device_properties(0))}
            def before_call(row):report.update(status=row['status'],dispatching_iteration=row['iteration'],runtime=row['runtime']);save()
            def progress(row):
                report.update(status='FAIL' if row['failed'] else 'RUNNING',iterations_completed=row['iteration'],last_completed_call=row);save()
                print(json.dumps({'iteration':row['iteration'],'failed':row['failed'],'seconds':row['seconds_including_checks']}),flush=True)
            configuration={key:copy.deepcopy(value) for key,value in report.items() if key!='status'}
            report.update(run_loop(config,failure_path=args.failure_dump,repeats=args.repeats,device='cuda:0',
                          launch_factory=lambda state,c:OriginalLauncher(state,c,args.code_directory),
                          max_resident_bytes=args.max_resident_gib<<30,max_capture_bytes=args.max_capture_gib<<30,
                          chunk_elements=args.chunk_elements,chunk_bytes=args.copy_chunk_mib<<20,
                          configuration=configuration,progress=progress,before_call=before_call))
        else:report['status']='CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH'
        report['source_sha256_after']=source_inventory();report['source_files_unchanged_during_run']=report['source_sha256_before']==report['source_sha256_after']
        if not report['source_files_unchanged_during_run'] and report['status'] in ('PASS','CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH'):report['status']='SOURCE_CHANGED_DURING_RUN'
        save()
    except BaseException as error:
        report.update(status='ERROR',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});save();raise
    print(json.dumps({'status':report['status'],'report':str(args.report)}),flush=True)
    return 0 if report['status'] in ('PASS','CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH') else 1


if __name__=='__main__':raise SystemExit(main())
