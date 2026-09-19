#!/usr/bin/env python3
"""Standalone exact-zero-gradient reproducer for weighted input-LN backward.

Run one stage/BLOCK_N arm per process, only on an idle GPU. No capture or dataset
is needed. --stage dx invokes the original production inner JIT directly and
checks DX plus both FP32 partial-gradient arrays without running reduction.
--stage helper calls the public production backward and checks its three final
outputs. Both stages poison DX and both partial arrays before every DX launch;
the helper stage also poisons final parameter gradients before reduction.
With --check-before-reduction, the helper additionally probes DX and both
partials immediately after DX returns, retaining only compact probe results.
--check-partials-before-reduction probes only the two partial arrays at that
boundary; DX still receives the normal final helper check. The two options are
mutually exclusive. With 2,048 partial tiles, partial-only checks scan 8 MiB,
avoiding an extra 2.35 GiB DX scan at the default full size, but still change
execution timing.
The inputs are synthetic:
bounded random BF16 X, unit weights, zero bias, actual production forward
statistics, and exactly zero DY. All three backward outputs must equal zero.
Output checks use bounded reductions every iteration; all input bytes are
compared with independent snapshots at checkpoints. Checks alter timing and
memory traffic. An asynchronous GPU error does not identify its originating
kernel from the reporting Python/checker frame.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time


CHUNK_ELEMENTS = 16 * 1024**2
MAX_ARTIFACT_BYTES = 100 * 1024**2
EXPECTED_BN8_HASH = 'e34c0e52265268b1d8e1dbd1b24149b417544e8fd4b45af4fd79b52e1cf64d40'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('dx', 'helper'), required=True)
    parser.add_argument('--block-n', type=int, choices=(1, 8), required=True)
    parser.add_argument('--max-vgpr', type=int)
    parser.add_argument('--rows', type=int, default=2468813)
    parser.add_argument('--repeat', type=int, default=1000)
    parser.add_argument('--check-every', type=int, default=10)
    before_reduction = parser.add_mutually_exclusive_group()
    before_reduction.add_argument('--check-before-reduction', action='store_true',
                                  help='helper only: also probe DX and partial gradients before reduction')
    before_reduction.add_argument('--check-partials-before-reduction', action='store_true',
                                  help='helper only: probe partial gradients before reduction without an extra DX scan')
    parser.add_argument('--seed', type=int, default=947)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.rows, args.repeat, args.check_every) < 1:
        parser.error('rows, repeat, and check-every must be positive')
    if args.max_vgpr is not None and args.max_vgpr < 1:
        parser.error('max-vgpr must be positive')
    if (args.check_before_reduction or args.check_partials_before_reduction) and args.stage != 'helper':
        parser.error('before-reduction checks require --stage helper')
    for protected in (Path(__file__),):
        if args.report.resolve() == protected.resolve() or (
            args.report.exists() and protected.exists() and args.report.samefile(protected)
        ):
            parser.error('report must not overwrite script')
    return args


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    return value


def scan_bad(torch, numel, device, mask_at, chunk_elements=CHUNK_ELEMENTS):
    """Count failures and retain the first coordinate, with bounded temporaries."""
    counts, firsts = [], []
    for start in range(0, numel, chunk_elements):
        mask = mask_at(start, min(start + chunk_elements, numel))
        count = mask.sum(dtype=torch.int64)
        # A bool and uint8 have the same byte representation. argmax returns
        # the first true position without an element-sized int64 index array.
        index = mask.view(torch.uint8).argmax().to(torch.int64) + start
        counts.append(count)
        firsts.append(torch.where(count != 0, index, torch.full_like(index, numel)))
    if not counts:
        zero = torch.zeros((), dtype=torch.int64, device=device)
        return zero, zero
    return torch.stack(counts).sum(), torch.stack(firsts).amin()


def zero_probe(torch, tensor, chunk_elements=CHUNK_ELEMENTS):
    if not tensor.is_contiguous():
        raise ValueError('zero control requires contiguous production outputs')
    flat = tensor.view(-1)
    count, index = scan_bad(
        torch, flat.numel(), flat.device,
        lambda start, stop: flat[start:stop] != 0,
        chunk_elements,
    )
    safe = index.clamp(max=max(0, flat.numel() - 1))
    value = flat.index_select(0, safe.reshape(1))[0]
    sample = (tensor.index_select(0, (safe // tensor.shape[1]).reshape(1))
              if tensor.ndim == 2 else tensor.detach().clone())
    return {'count': count, 'index': index, 'value': value, 'sample': sample,
            'shape': list(tensor.shape)}


def scalar_rows(torch, pending):
    flattened = [(item, name, probe)
                 for item in pending for name, probe in item['outputs'].items()]
    packed = torch.stack([
        torch.stack((probe['count'].double(), probe['index'].double(), probe['value'].double()))
        for _, _, probe in flattened
    ]).cpu().tolist()
    failures = []
    for (item, name, probe), (count, index, value) in zip(flattened, packed):
        if count:
            index = int(index)
            coordinate = ([index // probe['shape'][1], index % probe['shape'][1]]
                          if len(probe['shape']) == 2 else [index])
            failure = {'iteration': item['iteration'], 'output': name, 'nonzero_elements': int(count),
                       'first_coordinate': coordinate, 'first_value': finite_json(value)}
            if 'observation_point' in item:
                failure['observation_point'] = item['observation_point']
            failures.append(failure)
    return failures


def compact_failure(torch, args, report, failures, pending, inputs, snapshots, byte_checks):
    """Persist only compact output evidence and selected input rows, never full X/DY."""
    rows = {failure['first_coordinate'][0] for failure in failures if failure['output'] == 'd_x'}
    # A partial-gradient coordinate identifies a tile, not a unique input row.
    # Include a representative row from that tile and label the mapping below.
    rows.update(failure['first_coordinate'][0] * args.block_n
                for failure in failures if failure['output'].startswith('partial_')
                and failure['first_coordinate'][0] * args.block_n < args.rows)
    mutations = {}
    for name, unchanged in byte_checks.items():
        if unchanged:
            continue
        left, right = inputs[name].view(torch.uint8).view(-1), snapshots[name].view(torch.uint8).view(-1)
        count, index = scan_bad(torch, left.numel(), left.device,
                                lambda start, stop: left[start:stop] != right[start:stop])
        count, index = int(count.cpu()), int(index.cpu())
        element = index // inputs[name].element_size()
        mutations[name] = {'different_bytes': count, 'first_byte_offset': index,
                           'first_element_offset': element}
        if name in ('x', 'dy'):
            rows.add(element // 512)
        elif name in ('mean', 'rstd'):
            rows.add(element)
    rows = sorted({r + delta for r in rows for delta in (-1, 0, 1)
                   if 0 <= r + delta < args.rows})[:32] or [0]
    indices = torch.tensor(rows, dtype=torch.int64, device=inputs['x'].device)
    sampled = {}
    for name, value in inputs.items():
        current, before = ((value.index_select(0, indices), snapshots[name].index_select(0, indices))
                           if name in ('x', 'dy', 'mean', 'rstd') else (value, snapshots[name]))
        sampled[name] = {'at_checkpoint': current.detach().cpu(), 'initial_snapshot': before.detach().cpu()}
    evidence = []
    for failure in failures[:32]:
        item = next(item for item in pending if item['iteration'] == failure['iteration']
                    and item.get('observation_point') == failure.get('observation_point'))
        sample = item['outputs'][failure['output']]['sample'].detach().cpu()
        record = {**failure, 'sample': sample,
                  'sample_meaning': 'first failing output row' if failure['output'] == 'd_x'
                  else 'first failing partial-gradient tile' if failure['output'].startswith('partial_')
                  else 'complete parameter-gradient vector'}
        if failure['output'].startswith('partial_'):
            tile = failure['first_coordinate'][0]
            first_row = tile * args.block_n
            record['partial_tile_provenance'] = {
                'tile': tile, 'has_input_rows': first_row < args.rows,
                'representative_input_row': first_row if first_row < args.rows else None,
            }
        evidence.append(record)
    payload = {'format_version': 1, 'diagnostic': 'weighted_ln_zero_dy', 'seed': args.seed,
               'stage': args.stage, 'tile_num': report['tile_num'],
               'check_before_reduction': report.get('check_before_reduction', False),
               'check_partials_before_reduction': report.get('check_partials_before_reduction', False),
               'partial_tile_input_mapping': 'row=(tile + k*tile_num)*BLOCK_N + lane for 0<=row<N; tiles with tile*BLOCK_N>=N have no input rows',
               'shape': [args.rows, 512], 'block_n': args.block_n, 'input_sample_rows': rows,
               'outputs': evidence, 'inputs': sampled, 'input_mutations': mutations,
               'selected_norm_configs': report['selected_norm_configs'],
               'actual_norm_launches': report.get('actual_norm_launches', []),
               'input_snapshot_timing': 'after setup, before the first backward',
               'current_input_timing': 'at the failing checkpoint; not necessarily first failing iteration'}

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(v) for v in value)
        return 0

    if tensor_bytes(payload) > MAX_ARTIFACT_BYTES // 2:
        raise RuntimeError('compact evidence unexpectedly exceeds tensor byte budget')
    path = args.report.with_name(args.report.stem + f'.failure.iter{report["iteration"]:06d}.pid{os.getpid()}.pt')
    with path.open('xb') as handle:
        torch.save(payload, handle)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        path.unlink()
        raise RuntimeError('compact evidence exceeded 100 MiB and was removed')
    return {'path': str(path), 'bytes': path.stat().st_size, 'input_sample_rows': rows,
            'input_mutations': mutations, 'saved_output_samples': len(evidence)}



def tile_count(rows, multiprocessors):
    return max(1, min(multiprocessors * 8, rows // 4))


def launch_configs(block_n, max_vgpr=None, *, hip=True):
    dx = {'BLOCK_N': block_n, 'num_warps': 1, 'num_stages': 1, 'num_ctas': 1}
    reduction = {'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 1, 'num_ctas': 1}
    if hip:
        dx.update(llvm_fn_attrs=f'amdgpu-num-vgpr={max_vgpr}' if max_vgpr else [],
                  waves_per_eu=0, enable_fp_fusion=True)
        reduction.update(llvm_fn_attrs=[], waves_per_eu=0, enable_fp_fusion=True)
    elif max_vgpr is not None:
        raise ValueError('--max-vgpr requires the ROCm Triton backend')
    return {'dx': dx, 'reduction': reduction}


def inner_jit(kernel):
    while hasattr(getattr(kernel, 'fn', None), 'run'):
        kernel = kernel.fn
    return kernel


def record_launch(report, role, compiled, kwargs):
    key = (role, compiled.hash)
    records = report.setdefault('actual_norm_launches', [])
    if any((record['role'], record['compiled_hash']) == key for record in records):
        return
    metadata = compiled.metadata
    metadata = metadata._asdict() if hasattr(metadata, '_asdict') else metadata
    records.append({
        'role': role, 'compiled_hash': compiled.hash, 'kernel_name': compiled.name,
        'registers': getattr(compiled, 'n_regs', None), 'spills': getattr(compiled, 'n_spills', None),
        'compiled_metadata': metadata,
        'launch_config': {k: v for k, v in kwargs.items() if k in (
            'BLOCK_N', 'BLOCK_D', 'num_warps', 'num_stages', 'num_ctas', 'llvm_fn_attrs',
            'waves_per_eu', 'enable_fp_fusion', 'IS_SWISH')},
        'artifact_sha256': {k: hashlib.sha256(v if isinstance(v, bytes) else v.encode()).hexdigest()
                            for k, v in compiled.asm.items() if k in ('llir', 'amdgcn', 'hsaco')},
    })
    if role == 'dx' and report['block_n'] == 8 and report['max_vgpr'] is None:
        report['dx_matches_known_bn8_hash'] = compiled.hash == EXPECTED_BN8_HASH


def direct_dx(torch, kernel, inputs, config, tiles, report):
    x, dy = inputs['x'], inputs['dy']
    dx = torch.empty_like(x)
    partial_dw = torch.empty((tiles, x.shape[1]), dtype=torch.float32, device=x.device)
    partial_db = torch.empty_like(partial_dw)
    for output in (dx, partial_dw, partial_db):
        output.fill_(float('nan'))
    kwargs = {**config, 'BLOCK_D': 512, 'IS_SWISH': False, 'N': x.shape[0],
              'grid': (tiles,), 'warmup': False}
    compiled = inner_jit(kernel).run(
        dx, dy, partial_dw, partial_db, x, inputs['weight'], inputs['bias'], inputs['mean'], inputs['rstd'],
        dx.stride(0), dy.stride(0), x.stride(0), x.shape[1], 1e-6, **kwargs,
    )
    record_launch(report, 'dx', compiled, kwargs)
    return dx, partial_dw, partial_db


def append_before_reduction_probes(torch, pending, report, dx, partial_dw, partial_db, *, partials_only=False):
    """Read the live DX results without retaining any full production allocation."""
    names = ('partial_d_norm_weight', 'partial_d_norm_bias')
    values = (partial_dw, partial_db)
    if not partials_only:
        names, values = ('d_x', *names), (dx, *values)
    report['phase'] = ('bounded_before_reduction_partial_zero_checks' if partials_only
                       else 'bounded_before_reduction_zero_checks')
    pending.append({
        'iteration': report['iteration'], 'observation_point': 'before_reduction',
        'outputs': {name: zero_probe(torch, value) for name, value in zip(names, values)},
    })
    report['phase'] = 'input_norm_helper_reduction'


@contextmanager
def helper_launch_scope(triton, module, configs, report, *, after_dx=None):
    """Use the public helper, pin its two kernels, and poison its own allocations."""
    handles = []
    try:
        for role, kernel in (('dx', module._weighted_layer_norm_bwd_dx),
                             ('reduction', module._layer_norm_bwd_dwdb)):
            config = configs[role]
            fields = set(inspect.signature(triton.Config).parameters) - {'kwargs'}
            tuning = {k: v for k, v in config.items() if k in fields}
            kwargs = {k: v for k, v in config.items() if k not in fields}
            pinned = triton.Config(kwargs, **tuning)
            handles.extend(((kernel, 'configs', kernel.configs), (kernel, 'cache', dict(kernel.cache))))
            kernel.configs = [pinned]
            kernel.cache.clear()
            jit = inner_jit(kernel)
            original = jit.run
            handles.append((jit, 'run', original))

            def launch(*positional, _original=original, _role=role, **kwargs):
                if not kwargs.get('warmup', False):
                    # Production callers use positional tensor arguments. DX,
                    # partial DW/DB are indices 0/2/3; final DW/DB are 2/3.
                    for index in ((0, 2, 3) if _role == 'dx' else (2, 3)):
                        positional[index].fill_(float('nan'))
                compiled = _original(*positional, **kwargs)
                if _role == 'dx' and after_dx is not None and not kwargs.get('warmup', False):
                    after_dx(positional[0], positional[2], positional[3])
                if compiled is not None and not kwargs.get('warmup', False):
                    record_launch(report, _role, compiled, kwargs)
                return compiled

            jit.run = launch
        yield
    finally:
        for obj, name, original in reversed(handles):
            setattr(obj, name, original)


def all_input_bytes_equal(torch, value, snapshot):
    left, right = value.view(torch.uint8).view(-1), snapshot.view(torch.uint8).view(-1)
    flags = [(left[start:start + CHUNK_ELEMENTS] == right[start:start + CHUNK_ELEMENTS]).all()
             for start in range(0, left.numel(), CHUNK_ELEMENTS)]
    return torch.stack(flags).all()


def run(args, report, publish):
    for name, value in {'AMDGCN_USE_BUFFER_OPS': '0', 'TRITON_FULL_AUTOTUNE': '0',
                        'TRITON_ALLOW_PIPELINING': '0', 'PYTORCH_CUDA_ALLOC_CONF': '',
                        'PYTORCH_ALLOC_CONF': '', 'HSA_ENABLE_COREDUMP': '0'}.items():
        os.environ.setdefault(name, value)
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    import torch
    import triton
    from generative_recommenders.dlrm_v4.train.nan_tripwire import _all_finite, _all_within_bound
    from generative_recommenders.ops.triton import triton_layer_norm as layer_norm

    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    tiles = tile_count(args.rows, properties.multi_processor_count)
    configs = launch_configs(args.block_n, args.max_vgpr, hip=bool(torch.version.hip))
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    report.update(tile_num=tiles, multiprocessors=properties.multi_processor_count,
                  tile_num_formula='max(1, min(multiprocessors*8, rows//4))', selected_norm_configs=configs)
    report['source'] = {'root': str(source),
                        'layer_norm_sha256': hashlib.sha256(Path(layer_norm.__file__).read_bytes()).hexdigest(),
                        'torch': str(torch.__version__), 'triton': triton.__version__, 'hip': torch.version.hip,
                        'arch': getattr(properties, 'gcnArchName', properties.name)}
    report['environment'] = {name: os.environ.get(name) for name in (
        'AMDGCN_USE_BUFFER_OPS', 'TRITON_FULL_AUTOTUNE', 'TRITON_ALLOW_PIPELINING',
        'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'AMD_SERIALIZE_KERNEL',
        'HSTU_BWD_MAX_VGPR', 'WEIGHTED_LN_BWD_BLOCK_N', 'HSA_ENABLE_COREDUMP',
        'TRITON_HIP_USE_EXPERT_SCHEDULING', 'TRITON_HIP_USE_COEXEC_SCHEDULER',
        'AMDGCN_SCALARIZE_PACKED_FOPS', 'TRITON_HIP_USE_IN_THREAD_TRANSPOSE')}
    report['phase'] = 'setup_forward'
    publish('setup_start')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        x = torch.empty((args.rows, 512), device='cuda', dtype=torch.bfloat16).uniform_(-1, 1, generator=generator)
        weight = torch.ones(512, device='cuda', dtype=torch.bfloat16)
        bias = torch.zeros_like(weight)
        y, mean, rstd = layer_norm.triton_weighted_layer_norm_fwd(x, weight, bias, 1e-6)
        dy = torch.zeros_like(x)
        inputs = {'dy': dy, 'x': x, 'weight': weight, 'bias': bias, 'mean': mean, 'rstd': rstd}
        setup_flags = {name + ':finite': _all_finite(value, CHUNK_ELEMENTS)
                       for name, value in {**inputs, 'forward_y': y}.items()}
        setup_flags.update(x_bounded=_all_within_bound(x, 1.0, CHUNK_ELEMENTS),
                           dy_exact_zero=_all_within_bound(dy, 0.0, CHUNK_ELEMENTS),
                           weight_ones=(weight == 1).all(), bias_zero=(bias == 0).all(),
                           positive_rstd=(rstd > 0).all(), rstd_bounded=(rstd <= 1000.0).all(),
                           mean_bounded=(mean.abs() <= 1.0).all())
        report['setup_checks'] = dict(zip(setup_flags, torch.stack(list(setup_flags.values())).cpu().tolist()))
        del y, setup_flags
        if not all(report['setup_checks'].values()):
            report['status'] = 'SETUP_FAILURE'
            publish('setup_failure')
            return 1
        fixed = {**inputs, 'learnable': True, 'eps': 1e-6, 'BLOCK_D': 512}
        snapshots = {name: value.clone() for name, value in inputs.items()}
        versions = {name: value._version for name, value in inputs.items()}
        report['input_bytes'] = sum(value.numel() * value.element_size() for value in inputs.values())
        report['snapshot_bytes'] = report['input_bytes']
        names = ('d_x', 'partial_d_norm_weight', 'partial_d_norm_bias') if args.stage == 'dx' else (
            'd_x', 'd_norm_weight', 'd_norm_bias')
        shapes = ((args.rows, 512), (tiles, 512), (tiles, 512)) if args.stage == 'dx' else (
            (args.rows, 512), (512,), (512,))
        dtypes = (torch.bfloat16, torch.float32, torch.float32) if args.stage == 'dx' else (torch.bfloat16,) * 3
        report['checked_outputs'] = list(names)
        pending = []
        after_dx = None
        before_reduction = args.check_before_reduction or args.check_partials_before_reduction
        if before_reduction:
            report['checked_before_reduction_outputs'] = ['partial_d_norm_weight', 'partial_d_norm_bias']
            report['checked_before_reduction_shapes'] = [[tiles, 512], [tiles, 512]]
            if args.check_before_reduction:
                report['checked_before_reduction_outputs'].insert(0, 'd_x')
                report['checked_before_reduction_shapes'].insert(0, [args.rows, 512])

            def after_dx(*values):
                append_before_reduction_probes(torch, pending, report, *values,
                                              partials_only=args.check_partials_before_reduction)

        scope = (helper_launch_scope(triton, layer_norm, configs, report, after_dx=after_dx)
                 if args.stage == 'helper' else nullcontext())
        started = time.monotonic()
        with scope:
            for iteration in range(1, args.repeat + 1):
                report.update(iteration=iteration, phase='input_norm_' + args.stage)
                outputs = (direct_dx(torch, layer_norm._weighted_layer_norm_bwd_dx, inputs, configs['dx'], tiles, report)
                           if args.stage == 'dx' else layer_norm.triton_weighted_layer_norm_bwd(**fixed))
                report['phase'] = 'bounded_output_zero_checks'
                if len(outputs) != 3 or any(not isinstance(value, torch.Tensor) for value in outputs):
                    raise ValueError('production weighted LN returned unexpected output structure')
                for name, value, shape, dtype in zip(names, outputs, shapes, dtypes):
                    if tuple(value.shape) != shape or value.dtype != dtype or value.device != x.device:
                        raise ValueError('production output shape/dtype/device changed: ' + name)
                probes = {name: zero_probe(torch, value) for name, value in zip(names, outputs)}
                item = {'iteration': iteration, 'outputs': probes}
                if before_reduction:
                    item['observation_point'] = 'after_helper'
                    observed = sum(record['iteration'] == iteration and record.get('observation_point') == 'before_reduction'
                                   for record in pending)
                    if observed != 1:
                        raise RuntimeError(f'expected one before-reduction observation, observed {observed}')
                pending.append(item)
                del item
                del outputs, probes
                if iteration % args.check_every and iteration != args.repeat:
                    continue
                report['phase'] = 'checkpoint_input_byte_checks'
                tracked = {name: {'before': versions[name], 'after': value._version}
                           for name, value in inputs.items() if value._version != versions[name]}
                byte_flags = {name: all_input_bytes_equal(torch, value, snapshots[name])
                              for name, value in inputs.items()}
                byte_checks = dict(zip(byte_flags, torch.stack(list(byte_flags.values())).cpu().tolist()))
                failures = scalar_rows(torch, pending)
                failed = bool(failures or tracked or not all(byte_checks.values()))
                check = {'through': iteration, 'seconds': time.monotonic() - started,
                         'output_failures': [failure for failure in failures
                                             if failure.get('observation_point') != 'before_reduction'],
                         'input_bytes_unchanged': byte_checks, 'tracked_input_changes': tracked}
                if before_reduction:
                    check['before_reduction_failures'] = [failure for failure in failures
                                                          if failure.get('observation_point') == 'before_reduction']
                    check['before_reduction_checked_iterations'] = [record['iteration'] for record in pending
                                                                    if record.get('observation_point') == 'before_reduction']
                report['checks'].append(check)
                report.update(iterations_completed=iteration, status='FAIL' if failed else (
                    'PASS' if iteration == args.repeat else 'RUNNING'), phase='checkpoint')
                if failed:
                    report['positive_reproducer'] = bool(failures)
                    if before_reduction:
                        report['positive_before_reduction'] = bool(check['before_reduction_failures'])
                    report['failure_scope'] = ('nonzero gradient observed; input mutation also observed'
                                               if failures and (tracked or not all(byte_checks.values()))
                                               else 'nonzero gradient observed' if failures else 'input mutation observed')
                    publish('failure_detected')
                    try:
                        report['failure_artifact'] = compact_failure(
                            torch, args, report, failures, pending, inputs, snapshots, byte_checks,
                        )
                    except Exception as error:
                        report['failure_artifact_error'] = repr(error)
                    publish('failure')
                    return 1
                pending.clear()
                publish('checkpoint')
        report['phase'] = 'complete'
        report['peak_device_allocated_bytes'] = torch.cuda.max_memory_allocated()
        publish('complete')
    return 0


def main():
    args = arguments()
    report = {'diagnostic': 'weighted_ln_zero_dy', 'oracle': 'all checked gradients equal numerical zero, including signed zero',
              'positive_reproducer': False, 'status': 'RUNNING', 'stage': args.stage, 'phase': 'initialization', 'iteration': 0,
              'iterations_completed': 0, 'rows': args.rows, 'features': 512, 'dtype': 'torch.bfloat16',
              'block_n': args.block_n, 'max_vgpr': args.max_vgpr, 'repeat': args.repeat,
              'check_every': args.check_every, 'seed': args.seed, 'chunk_elements': CHUNK_ELEMENTS, 'checks': [],
              'check_before_reduction': args.check_before_reduction,
              'check_partials_before_reduction': args.check_partials_before_reduction,
              'poison': 'NaN before every DX/partial write; helper final gradients also poisoned before reduction',
              'expected_uncapped_bn8_dx_hash': EXPECTED_BN8_HASH,
              'limitations': ['synthetic zero-gradient input; passing does not validate nonzero-gradient training',
                              'input bytes checked at checkpoints; intermediate changes that revert are not excluded',
                              'poisoning, checks, and snapshots alter timing, allocation reuse, and memory traffic',
                              'an asynchronous error reported by a checker does not attribute its source kernel']}

    def publish(event):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(finite_json(report), indent=2, default=str, allow_nan=False) + '\n')
        print(json.dumps({'event': event, 'status': report['status'], 'stage': report['stage'], 'phase': report['phase'],
                          'block_n': args.block_n, 'iteration': report['iteration'],
                          'iterations_completed': report['iterations_completed'], 'report': str(args.report)},
                         allow_nan=False), flush=True)

    try:
        return run(args, report, publish)
    except Exception as error:
        report.update(status='ERROR', error=repr(error),
                      error_attribution='reporting phase is not proof of the originating GPU kernel')
        publish('error')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
