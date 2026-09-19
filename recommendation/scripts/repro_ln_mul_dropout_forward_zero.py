#!/usr/bin/env python3
"""Synthetic exact-zero control for the original separated-RNG HSTU forward.

Default is CPU plan/source inspection only. Explicit --gpu compiles the original
inner JIT, archives its code object before dispatch, and repeatedly launches a
selected configuration. This is not a replay of the uncaptured aborted inputs.
"""
from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import traceback

import replay_additional_linear_capture as replay

support = replay.support
common = replay.common
KERNEL = '_ln_mul_dropout_fwd_rng'
MODEL_SOURCE = 'generative_recommenders/ops/triton/triton_hstu_linear.py'
DEFAULT_CONFIG = dict(rows=1423285, cols=512, stride_x=512, stride_u=512,
                      block_n=16, num_warps=1, eps=1e-6, dropout_ratio=.3, guard_elements=256)
HISTORICAL_CANDIDATE = {
    'cache_directory':'5XYGNJ6XHD4JKTST2W5SLSBGHYXLP2GTN77UIWKWGQ7535YLFIMQ',
    'kernel_hash':'edf066a7d738f8954e53d5bb25c8263e2eb7e8d36fff445956343fddf70b2a19',
    'hsaco_sha256':'94bb8c35798f99dbedefbca85e3b3dcc57d7fa7bce2032ec86b0111d8b3b45fc',
    'amdgcn_sha256':'d135bd8e0f7d8a949983ec450f6088863682319775a0b9697fed2bdbb090f96a',
    'hsaco_text_sha256':'cfcb01329ae8a731f2508f99fd9ca8396f27204f855c6d365b835bab37ffdce7',
    'private_segment_bytes':696,'shared_segment_bytes':64,'vgpr_count':1024,'sgpr_count':107,
    'scope':'Unique observed cache candidate matching reported private696/shared64/one-wave dispatch among the30 retained variants. No historical dispatch code object was directly captured.'}
INPUT_CANARY_FLOAT = 17
INPUT_CANARY_MASK = 85
OUTPUT_CANARY = 23
ARGUMENTS = ('X', 'U', 'Y', 'W', 'B', 'Mean', 'Rstd', 'RANDOM_MASK', 'N', 'D', 'eps',
             'dropout_ratio', 'stride_x', 'stride_u', 'stride_y', 'stride_mask',
             'SILU_U', 'BLOCK_D', 'BLOCK_N', 'TRAINING', 'CONCAT_U', 'CONCAT_X', 'MUL_U_ACTIVATION_TYPE')
LIMITS = [
    'Synthetic selected-configuration control; original aborted input values, allocator history and preceding workload are unavailable.',
    'The abort grid bounds N to1423281..1423296 for BN16. Default N1423285 is inferred from the successful same-start smoke Linear rows plus8192 contextual rows, not directly captured from the failed dispatch.',
    'Dense X/U strides512 and eps/dropout are source-derived controls; cached scalar ABI proves divisibility specializations, not runtime scalar values.',
    'Exact zero is required for Y and Mean; Rstd has a separately stated FP32 sqrt/reciprocal tolerance.',
    'Input backing bytes are checked at every completed call. Transient modifications that revert before checking remain unobservable.',
    'Synthetic inputs, poisoned outputs, canaries and full checks perturb timing; a pass does not clear the production fault.',
    'A hard GPU abort can prevent any output copy. The pre-dispatch report and archived compiled code identify this experiment, not the historical failed dispatch.',
]


def fp32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error):
        return math.copysign(math.inf, value)


def geometry_plan(config, chunk_elements=1 << 24):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError('Complete known synthetic configuration required')
    c = dict(config)
    for name in ('rows', 'cols', 'stride_x', 'stride_u', 'block_n', 'num_warps', 'guard_elements'):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError('Invalid positive integer geometry: ' + name)
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError('Positive chunk_elements required')
    if c['block_n'] not in (1, 2, 4, 8, 16) or c['num_warps'] not in (1, 2, 4):
        raise ValueError('Configuration must belong to original forward autotune family')
    n, d, g = c['rows'], c['cols'], c['guard_elements']
    if c['stride_x'] < d or c['stride_u'] < d or g % 16:
        raise ValueError('Nonoverlapping rows and guards aligned to16 elements required')
    block_d = 1 << (d - 1).bit_length()
    blocks = (n + c['block_n'] - 1) // c['block_n']
    if block_d > 32768 or max(n, d, c['stride_x'], c['stride_u'], 3*d, blocks*c['block_n'] - 1) >= 1 << 31:
        raise ValueError('Geometry exceeds original fused-width or signed32 row/scalar indexing range')
    for name in ('eps', 'dropout_ratio'):
        if type(c[name]) not in (int, float) or not math.isfinite(c[name]):
            raise ValueError('Finite real scalar required: ' + name)
    eps, drop = fp32(c['eps']), fp32(c['dropout_ratio'])
    if not math.isfinite(eps) or eps < 2.0**-126 or not 0 <= drop < 1:
        raise ValueError('Effective FP32 eps must be positive normal and dropout in[0,1)')
    shapes = {'X': ((n, d), (c['stride_x'], 1), 2), 'U': ((n, d), (c['stride_u'], 1), 2),
              'W': ((d,), (1,), 2), 'B': ((d,), (1,), 2), 'RANDOM_MASK': ((n, d), (d, 1), 1),
              'Y': ((n, 3*d), (3*d, 1), 2), 'Mean': ((n,), (1,), 4), 'Rstd': ((n,), (1,), 4)}
    buffers = {}
    for name, (shape, stride, element) in shapes.items():
        span = 1 + sum((size - 1)*step for size, step in zip(shape, stride))
        count = span + 2*g
        if count*element >= 1 << 63:
            raise ValueError('Buffer byte address exceeds signed64 range: ' + name)
        buffers[name] = dict(shape=list(shape), stride=list(stride), storage_offset=g,
                             span_elements=span, storage_elements=count, bytes=count*element,
                             last_logical_element_index=g+span-1)
    inputs = sum(buffers[k]['bytes'] for k in ('X','U','W','B','RANDOM_MASK'))
    outputs = sum(buffers[k]['bytes'] for k in ('Y','Mean','Rstd'))
    scratch = min(chunk_elements, max(v['storage_elements'] for v in buffers.values()))*64
    return dict(configuration=c, effective_fp32_eps=eps, effective_fp32_dropout_ratio=drop,
                block_d=block_d, grid=[blocks,1,1], buffers=buffers,
                expected_rstd=1/math.sqrt(eps), rstd_relative_tolerance=2.0**-20,
                rstd_absolute_tolerance=0., exact_oracle=['Y','Mean'],
                rstd_tolerance_scope='16 times FP32 unit roundoff relative acceptance tolerance; not a formal bound for the generated sqrt/reciprocal instructions.',
                address_guard=dict(last_padded_row=blocks*c['block_n']-1,
                                   largest_y_linear_index=n*3*d-1,
                                   crosses_old_signed32_y_product=(n-1)*3*d > (1<<31)-1,
                                   required_pointer_row_dtype='int64'),
                resource_plan=dict(input_storage_bytes=inputs, output_storage_bytes=outputs,
                                   resident_storage_bytes=inputs+outputs,
                                   bounded_checker_scratch_bytes=scratch,
                                   resident_peak_before_backend_workspace_bytes=inputs+outputs+scratch,
                                   complete_failure_storage_bytes=inputs+outputs,
                                   scope='Tensor and checker allowances only; compiler, driver scratch and allocator reservation excluded.'))


def flat_storage(tensor):
    torch = common._torch()
    return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes()//tensor.element_size(),), (1,))


def integer_dtype(tensor):
    torch = common._torch()
    return {1:torch.int8, 2:torch.int16, 4:torch.int32}[tensor.element_size()]


def expected_bits(dtype, value):
    torch = common._torch()
    t = torch.tensor([value], dtype=dtype)
    return int(t.view(integer_dtype(t))[0])


def make_buffers(config, device='cpu'):
    torch = common._torch()
    plan = geometry_plan(config)
    inputs, outputs = {}, {}
    for name, spec in plan['buffers'].items():
        dtype = torch.int8 if name == 'RANDOM_MASK' else torch.float32 if name in ('Mean','Rstd') else torch.bfloat16
        initial = 7 if name == 'RANDOM_MASK' else 1 if name == 'W' else float('nan') if name in ('Y','Mean','Rstd') else 0
        backing_fill = OUTPUT_CANARY if name in ('Y','Mean','Rstd') else INPUT_CANARY_MASK if name=='RANDOM_MASK' else INPUT_CANARY_FLOAT
        backing = torch.full((spec['storage_elements'],), backing_fill, dtype=dtype, device=device)
        view = backing.as_strided(spec['shape'], spec['stride'], spec['storage_offset'])
        view.fill_(initial)
        (outputs if name in ('Y','Mean','Rstd') else inputs)[name] = view
    return {'inputs':inputs, 'outputs':outputs}


def check_inputs(inputs, chunk_elements=1 << 24):
    torch = common._torch()
    if type(chunk_elements) is not int or chunk_elements<1:
        raise ValueError('Positive chunk_elements required')
    if set(inputs) != {'X','U','W','B','RANDOM_MASK'}:
        raise ValueError('Exact synthetic input mapping required')
    counts, names = [], []
    for name, tensor in inputs.items():
        raw = flat_storage(tensor)
        bits = expected_bits(tensor.dtype, 7 if name == 'RANDOM_MASK' else 1 if name == 'W' else 0)
        canary = expected_bits(tensor.dtype,INPUT_CANARY_MASK if name=='RANDOM_MASK' else INPUT_CANARY_FLOAT)
        offset=tensor.storage_offset()
        span=1+sum((size-1)*step for size,step in zip(tensor.shape,tensor.stride()))
        exterior=[raw[:offset],raw[offset+span:]]
        if tensor.ndim==2 and tensor.shape[0]>1 and tensor.stride(0)>tensor.shape[1]:
            exterior.append(raw.as_strided((tensor.shape[0]-1,tensor.stride(0)-tensor.shape[1]),
                                            (tensor.stride(0),1),offset+tensor.shape[1]))
        def differences(regions,expected):
            partial=[(chunk.view(integer_dtype(chunk))!=expected).sum() for region in regions
                     for chunk in common._logical_chunks(region,chunk_elements) if chunk.numel()]
            return torch.stack(partial).sum() if partial else torch.zeros((),dtype=torch.int64,device=tensor.device)
        counts.append(torch.stack((differences([tensor],bits),differences(exterior,canary))));names.append(name)
    values = torch.stack(counts).cpu().tolist()
    result = {name: {'different_backing_elements':int(sum(value)), 'different_logical_elements':int(value[0]),
                     'different_guard_or_padding_elements':int(value[1])} for name,value in zip(names,values)}
    return {'failed':any(any(v) for v in values), 'tensors':result,
            'scope':'Integer bits of every complete input backing element; logical constants and distinct nonzero guard/padding canaries checked separately'}


def poison_outputs(outputs):
    for tensor in outputs.values():
        tensor.fill_(float('nan'))


def check_outputs(outputs, config, chunk_elements=1 << 24):
    torch = common._torch()
    plan = geometry_plan(config, chunk_elements)
    if set(outputs) != {'Y','Mean','Rstd'}:
        raise ValueError('Exactly Y, Mean and Rstd required')
    rows, names = [], []
    for name,tensor in outputs.items():
        spec = plan['buffers'][name]
        expected_dtype = torch.bfloat16 if name=='Y' else torch.float32
        if (tensor.dtype!=expected_dtype or list(tensor.shape) != spec['shape']
                or list(tensor.stride()) != spec['stride'] or tensor.storage_offset() != spec['storage_offset']
                or tensor.untyped_storage().nbytes()!=spec['bytes']):
            raise ValueError('Output geometry differs: ' + name)
        flat = tensor.reshape(-1)
        chunks = []
        for start in range(0,flat.numel(),chunk_elements):
            value = flat[start:start+chunk_elements]
            finite = torch.isfinite(value)
            if name == 'Rstd':
                error = (value.double()-plan['expected_rstd']).abs()
                mismatch = (~finite) | (error > plan['expected_rstd']*plan['rstd_relative_tolerance'])
                maximum = torch.where(finite,error,0.).max()
            else:
                mask = (1 << (8*value.element_size()-1))-1
                mismatch = (value.view(integer_dtype(value)) & mask) != 0
                maximum = torch.zeros((),dtype=torch.float64,device=value.device)
            chunks.append(torch.stack((mismatch.sum().double(), (~finite).sum().double(),maximum)))
        matrix = torch.stack(chunks)
        raw = flat_storage(tensor).view(integer_dtype(tensor))
        g = config['guard_elements']; tail = g+spec['span_elements']
        canary = expected_bits(tensor.dtype,OUTPUT_CANARY)
        guard_bad = (raw[:g]!=canary).sum()+(raw[tail:]!=canary).sum()
        rows.append(torch.stack((matrix[:,0].sum(),matrix[:,1].sum(),matrix[:,2].max(),guard_bad.double())))
        names.append(name)
    values = torch.stack(rows).cpu().tolist()
    result = {name:dict(mismatched_elements=int(v[0]),nonfinite_elements=int(v[1]),
                       max_abs_rstd_error=v[2],changed_guard_elements=int(v[3])) for name,v in zip(names,values)}
    return {'failed':any(v[0] or v[3] for v in values),'tensors':result,
            'rstd_reference':plan['expected_rstd'],'rstd_relative_tolerance':plan['rstd_relative_tolerance']}


def audit_address_source(source):
    tree = ast.parse(source)
    node = next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==KERNEL),None)
    if node is None:
        raise ValueError('Original forward kernel body missing')
    assignments = {target.id:statement.value for statement in ast.walk(node) if isinstance(statement,ast.Assign)
                   for target in statement.targets if isinstance(target,ast.Name)}
    wide = assignments.get('rows_i64')
    if wide is None or ast.unparse(wide) != 'rows.to(tl.int64)':
        raise ValueError('Original row widening guard is absent or changed')
    alias = assignments.get('row_offsets_i64')
    if alias is None or ast.unparse(alias) != 'rows_i64':
        raise ValueError('Mask pointer widening alias is absent')
    products = []
    for expr in ast.walk(node):
        if not isinstance(expr,ast.BinOp) or not isinstance(expr.op,ast.Mult):
            continue
        names = {n.id for n in ast.walk(expr) if isinstance(n,ast.Name)}
        if names & {'stride_x','stride_u','stride_y','stride_mask'}:
            if not names & {'rows_i64','row_offsets_i64'}:
                raise ValueError('Narrow row-stride pointer product: '+ast.unparse(expr))
            products.append(ast.unparse(expr))
    if len(products)<4:
        raise ValueError('Expected all widened pointer products')
    return {'verified_widened_pointer_products':products,
            'kernel_ast_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
            'scope':'Source arithmetic guard; generated IR/assembly and executed hardware remain separate evidence.'}


def elf_text_hash(binary):
    if binary[:6] != b'\x7fELF\x02\x01' or len(binary)<64:
        return None
    offset = struct.unpack_from('<Q',binary,40)[0]
    size,count,names_index = struct.unpack_from('<HHH',binary,58)
    if size<64 or names_index>=count or offset+size*count>len(binary):
        return None
    def section(index): return struct.unpack_from('<IIQQQQIIQQ',binary,offset+size*index)
    names = section(names_index)
    strings = binary[names[4]:names[4]+names[5]]
    for index in range(count):
        values = section(index)
        start = values[0]; end = strings.find(b'\0',start)
        if end>=start and strings[start:end]==b'.text' and values[4]+values[5]<=len(binary):
            return hashlib.sha256(binary[values[4]:values[4]+values[5]]).hexdigest()
    return None


def kernel_evidence(compiled):
    asm = compiled.asm
    blobs = {name:(value if isinstance(value,bytes) else value.encode()) for name,value in asm.items() if isinstance(value,(str,bytes))}
    def serializable(value):
        if dataclasses.is_dataclass(value): return serializable(dataclasses.asdict(value))
        if hasattr(value,'_asdict'): return serializable(value._asdict())
        if isinstance(value,dict): return {key:serializable(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)): return [serializable(item) for item in value]
        return value
    metadata = serializable(compiled.metadata)
    text_hash=elf_text_hash(blobs.get('hsaco',b''))
    assembly=blobs.get('amdgcn',b'').decode(errors='replace')
    directives={}
    for label,directive in (('private_segment_bytes','private_segment_fixed_size'),
                            ('shared_segment_bytes','group_segment_fixed_size'),
                            ('vgpr_count','next_free_vgpr'),('sgpr_count','next_free_sgpr')):
        match=re.search(r'\.amdhsa_'+directive+r'\s+(\d+)',assembly)
        directives[label]=int(match.group(1)) if match else None
    return {'name':compiled.name,'hash':compiled.hash,'metadata':metadata,
            'n_regs':getattr(compiled,'n_regs',None),'n_spills':getattr(compiled,'n_spills',None),
            'function_handle':str(getattr(compiled,'function',None)),
            'loaded_kernel_sha256':hashlib.sha256(compiled.kernel).hexdigest() if isinstance(getattr(compiled,'kernel',None),bytes) else None,
            'artifacts':{name:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for name,value in blobs.items()},
            'hsaco_text_sha256':text_hash,'assembly_directives':directives,
            'historical_candidate':HISTORICAL_CANDIDATE,
            'matches_historical_candidate_hsaco':hashlib.sha256(blobs.get('hsaco',b'')).hexdigest()==HISTORICAL_CANDIDATE['hsaco_sha256'],
            'matches_historical_candidate_text':text_hash==HISTORICAL_CANDIDATE['hsaco_text_sha256'],
            'scope':'Exact CompiledKernel object used by the direct runner; archived before first experimental dispatch. Not identification of the historical abort code object.'}


def kernel_arguments(state,config):
    plan = geometry_plan(config)
    values = {**state['inputs'],**state['outputs'], 'N':config['rows'],'D':config['cols'],
              'eps':config['eps'],'dropout_ratio':config['dropout_ratio'],
              'stride_x':config['stride_x'],'stride_u':config['stride_u'],
              'stride_y':3*config['cols'],'stride_mask':config['cols'],
              'SILU_U':False,'BLOCK_D':plan['block_d'],'BLOCK_N':config['block_n'],
              'TRAINING':True,'CONCAT_U':True,'CONCAT_X':True,'MUL_U_ACTIVATION_TYPE':'none'}
    return [values[name] for name in ARGUMENTS]


def original_inner_jit(wrapper):
    from triton.runtime.jit import JITFunction
    seen=set()
    while not isinstance(wrapper,JITFunction):
        if id(wrapper) in seen or not hasattr(wrapper,'fn'):
            raise ValueError('Cannot reach original inner JITFunction')
        seen.add(id(wrapper));wrapper=wrapper.fn
    return wrapper


class OriginalLauncher:
    def __init__(self,state,config,code_directory):
        module = importlib.import_module('generative_recommenders.ops.triton.triton_hstu_linear')
        jit = original_inner_jit(getattr(module,KERNEL))
        if tuple(jit.arg_names)!=ARGUMENTS:
            raise ValueError('Original JIT argument mapping changed')
        args = kernel_arguments(state,config); grid = tuple(geometry_plan(config)['grid'])
        compiled = jit.warmup(*args,grid=grid,num_warps=config['num_warps'],num_stages=1)
        self.runner = compiled[grid]
        self.state = state; self.config = copy.deepcopy(config)
        self.evidence = kernel_evidence(compiled)
        directory = Path(code_directory)
        directory.mkdir(parents=True,exist_ok=False)
        for name,value in compiled.asm.items():
            if not isinstance(value,(bytes,str)) or not name.replace('_','').isalnum():
                continue
            binary = value if isinstance(value,bytes) else value.encode()
            support.atomic_new_save(directory/(KERNEL+'.'+name),lambda stream,b=binary:stream.write(b))
        support.atomic_new_save(directory/'metadata.json',lambda stream:stream.write((json.dumps(self.evidence,indent=2)+'\n').encode()))
        root=Path(__file__).resolve().parents[1]
        for source in (root/MODEL_SOURCE,Path(__file__).resolve()):
            body=source.read_bytes()
            support.atomic_new_save(directory/source.name,lambda stream,b=body:stream.write(b))

    def __call__(self,state,config):
        if state is not self.state or config!=self.config:
            raise ValueError('Compiled runner buffers/configuration changed')
        self.runner(*kernel_arguments(state,config))


def source_inventory():
    root = Path(__file__).resolve().parents[1]
    sources = replay.source_inventory()
    for name in ('scripts/repro_ln_mul_dropout_forward_zero.py',MODEL_SOURCE,
                 'generative_recommenders/common.py','generative_recommenders/ops/utils.py',
                 'generative_recommenders/ops/triton/triton_addmm.py'):
        sources[str(root/name)] = support.file_hash(root/name)
    return sources


def run_loop(config, *, failure_path, repeats=1000,device='cpu',launch=None,
             max_resident_bytes=64<<30,max_capture_bytes=64<<30,chunk_elements=1<<24,
             chunk_bytes=64<<20,configuration=None,progress=None,before_call=None,
             launch_factory=None):
    torch = common._torch()
    support.new_path(failure_path)
    if type(repeats) is not int or not 1<=repeats<=1000 or type(chunk_bytes) is not int or chunk_bytes<1:
        raise ValueError('Positive chunks and repeats1..1000 required')
    if (launch is None)==(launch_factory is None):
        raise ValueError('Exactly one explicit launch or factory required; no implicit CPU kernel')
    plan = geometry_plan(config,chunk_elements)
    resource = plan['resource_plan']
    if resource['resident_peak_before_backend_workspace_bytes']>max_resident_bytes or resource['complete_failure_storage_bytes']>max_capture_bytes:
        raise ValueError('Resource budget insufficient before allocation/dispatch')
    state = make_buffers(config,device)
    initial_checks = check_inputs(state['inputs'],chunk_elements)
    if initial_checks['failed']:
        raise RuntimeError('Synthetic input initialization differs from construction recipe')
    pristine = replay.digest_tree(state['inputs'],chunk_bytes)
    runtime = {'configuration':copy.deepcopy(configuration),'geometry_plan':plan,'versions':replay.runtime_versions(),
               'source_sha256_before':source_inventory(),'device':str(device),'repeats':repeats,
               'synthetic_recipe':{'X':0,'U':0,'W':1,'B':0,'RANDOM_MASK':7},
               'input_guard_and_padding_recipe':{'BF16':INPUT_CANARY_FLOAT,'int8':INPUT_CANARY_MASK},
               'output_initialization_recipe':{'logical':'NaN poison every iteration; payload is not assumed',
                                               'guard_and_padding':OUTPUT_CANARY},
               'limits':LIMITS}
    if launch_factory is not None:
        launch = launch_factory(state,config)
    runtime['kernel'] = copy.deepcopy(getattr(launch,'evidence',{'injected_callable':True}))
    records = []
    for iteration in range(1,repeats+1):
        poison_outputs(state['outputs'])
        if before_call:
            before_call({'status':'DISPATCHING','iteration':iteration,'runtime':runtime})
        started = time.monotonic()
        launch(state,config)
        outputs = check_outputs(state['outputs'],config,chunk_elements)
        inputs = check_inputs(state['inputs'],chunk_elements)
        record = {'iteration':iteration,'failed':outputs['failed'] or inputs['failed'],
                  'outputs':outputs,'inputs':inputs,'seconds_including_checks':time.monotonic()-started}
        if record['failed'] or iteration==repeats:
            current_inputs = replay.digest_tree(state['inputs'],chunk_bytes)
            record['input_hashes_match_initial'] = current_inputs==pristine
            record['failed'] |= not record['input_hashes_match_initial']
        records.append(record)
        if record['failed']:
            from nan_backward_boundaries import _cpu_copy_tree
            observed = replay.digest_tree(state,chunk_bytes)
            copied,stored = _cpu_copy_tree(state,chunk_bytes,max_capture_bytes)
            if replay.digest_tree(copied,chunk_bytes)!=observed:
                raise RuntimeError('First-failure copy differs from observed complete storage bytes')
            saved_outputs = check_outputs(copied['outputs'],config,chunk_elements)
            saved_inputs = check_inputs(copied['inputs'],chunk_elements)
            if saved_outputs!=outputs or saved_inputs!=inputs:
                raise RuntimeError('First-failure CPU classification differs')
            artifact = {'format':'ln_mul_dropout_forward_zero_failure_v1','iteration':iteration,
                        'configuration':dict(config),'current_at_failure':copied,'observed_record':record,
                        'initial_input_digest':pristine,'observed_digest':observed,'storage_bytes':stored,
                        'runtime':runtime,'source_sha256_at_failure':source_inventory(),
                        'capture_timing':'Existing first failed completed call; no rerun; pristine synthetic values described by recipe and initial hashes'}
            support.atomic_new_save(failure_path,lambda stream:torch.save(artifact,stream))
            if progress: progress(record)
            return {'status':'FAIL','iterations_completed':iteration,'iterations':records,'runtime':runtime,
                    'failure_capture':{'path':str(failure_path),'sha256':support.file_hash(failure_path),
                                       'bytes':Path(failure_path).stat().st_size,'storage_bytes':stored}}
        if progress: progress(record)
    return {'status':'PASS','iterations_completed':repeats,'iterations':records,'runtime':runtime,
            'final_input_hashes_match_initial':True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--failure-dump',type=Path)
    parser.add_argument('--code-directory',type=Path)
    parser.add_argument('--context',type=Path)
    parser.add_argument('--gpu',action='store_true')
    for name,value in DEFAULT_CONFIG.items():
        parser.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    parser.add_argument('--repeats',type=int,default=1000)
    parser.add_argument('--chunk-elements',type=int,default=1<<24)
    parser.add_argument('--copy-chunk-mib',type=int,default=64)
    parser.add_argument('--max-resident-gib',type=int,default=64)
    parser.add_argument('--max-capture-gib',type=int,default=64)
    parser.add_argument('--cpu-threads',type=int,default=2)
    args = parser.parse_args(argv)
    config = {key:getattr(args,key) for key in DEFAULT_CONFIG}
    plan = geometry_plan(config,args.chunk_elements)
    if min(args.copy_chunk_mib,args.max_resident_gib,args.max_capture_gib,args.cpu_threads)<1 or not 1<=args.repeats<=1000:
        parser.error('Positive resources and repeats1..1000 required')
    if args.gpu and any(v is None for v in (args.failure_dump,args.code_directory,args.context)):
        parser.error('GPU execution requires --failure-dump, --code-directory and --context')
    files = [p for p in (args.report,args.failure_dump,args.context) if p is not None]
    extended = files+[p.with_name(p.name+'.tmp') for p in (args.report,args.failure_dump) if p is not None]
    if len({p.resolve() for p in extended})!=len(extended):
        parser.error('Source/output/temp paths must differ')
    for p in (args.report,args.failure_dump):
        if p is not None: support.new_path(p)
    if args.code_directory and (args.code_directory.exists() or any(p.resolve().is_relative_to(args.code_directory.resolve()) for p in files)):
        parser.error('Code directory must be new and separate from source/report/failure paths')
    root = Path(__file__).resolve().parents[1]
    report = {'status':'PREPARING','geometry_plan':plan,'limits':LIMITS,
              'historical_candidate':HISTORICAL_CANDIDATE,
              'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}}
    support.atomic_new_save(args.report,lambda stream:stream.write((json.dumps(report,indent=2)+'\n').encode()))
    def save():
        temporary = args.report.with_name(args.report.name+'.tmp'); owned=False
        try:
            with temporary.open('xb') as stream:
                owned=True; stream.write((json.dumps(report,indent=2,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,args.report)
            descriptor=os.open(args.report.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        except BaseException:
            if owned: temporary.unlink(missing_ok=True)
            raise
    try:
        report['source_sha256_before']=source_inventory()
        report['address_source_audit']=audit_address_source((root/MODEL_SOURCE).read_text())
        if args.gpu:
            context=json.loads(args.context.read_text())
            report['context']={'path':str(args.context),'sha256':support.file_hash(args.context),'captured':context}
            report['restored_environment']=support.restore_environment(context)
            torch=common._torch();torch.set_num_threads(args.cpu_threads)
            replay.validate_runtime(context,replay.runtime_versions())
            for name in (MODEL_SOURCE,'generative_recommenders/common.py','generative_recommenders/ops/utils.py','generative_recommenders/ops/triton/triton_addmm.py'):
                if context.get('source_sha256',{}).get(name)!=support.file_hash(root/name):
                    raise ValueError('Original numerical source differs from context: '+name)
            sys.path.insert(0,str(root))
            report['device']={'name':torch.cuda.get_device_name(0),'properties':str(torch.cuda.get_device_properties(0))}
            def before_call(row):
                report.update(status=row['status'],dispatching_iteration=row['iteration'],runtime=row['runtime']);save()
            def progress(row):
                report.update(status='FAIL' if row['failed'] else 'RUNNING',iterations_completed=row['iteration'],last_completed_call=row);save()
                print(json.dumps({'iteration':row['iteration'],'failed':row['failed'],'seconds':row['seconds_including_checks']}),flush=True)
            configuration={key:copy.deepcopy(value) for key,value in report.items() if key!='status'}
            report.update(run_loop(config,failure_path=args.failure_dump,repeats=args.repeats,device='cuda:0',
                          launch_factory=lambda state,c:OriginalLauncher(state,c,args.code_directory),
                          max_resident_bytes=args.max_resident_gib<<30,max_capture_bytes=args.max_capture_gib<<30,
                          chunk_elements=args.chunk_elements,chunk_bytes=args.copy_chunk_mib<<20,
                          configuration=configuration,progress=progress,before_call=before_call))
        else:
            report['status']='CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH'
        report['source_sha256_after']=source_inventory()
        report['source_files_unchanged_during_run']=report['source_sha256_before']==report['source_sha256_after']
        if not report['source_files_unchanged_during_run'] and report['status'] in ('PASS','CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH'):
            report['status']='SOURCE_CHANGED_DURING_RUN'
        save()
    except BaseException as error:
        report.update(status='ERROR',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});save();raise
    print(json.dumps({'status':report['status'],'report':str(args.report)}),flush=True)
    return 0 if report['status'] in ('PASS','CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH') else 1


if __name__=='__main__':
    raise SystemExit(main())
