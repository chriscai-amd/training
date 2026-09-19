#!/usr/bin/env python3
"""Independent zero-input oracle for recomputed weighted layer-norm forward.

Calls the production public forward helper with zero BF16 X, unit weight,
zero bias, eps=1e-6, and no supplied statistics. Expected Y and mean are exactly
zero (either sign); float32 rstd must be finite and within 1e-3 of 1000, with
no relative tolerance. The tolerance permits about 16 float32 ULPs at 1000.
Omitting --block-n and --num-warps preserves production autotuning; specifying
either filters only that field in the existing production configurations.
NaN poisoning is included in autotuner benchmark calls and can affect their
timings and selected configuration. Checks observe the final helper launch,
not outputs overwritten by earlier autotune candidates. Run on an idle GPU.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


FEATURES = 512
EPS = 1e-6
RSTD_EXPECTED = 1000.0
RSTD_ATOL = 1e-3
CHUNK_ELEMENTS = 16 * 1024**2
MAX_ARTIFACT_BYTES = 100 * 1024**2
OUTPUTS = ('y', 'mean', 'rstd')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='production recommendation/ directory')
    parser.add_argument('--block-n', type=int, choices=(1, 8))
    parser.add_argument('--num-warps', type=int, choices=(1, 2, 4, 8))
    parser.add_argument('--rows', type=int, default=2468813)
    parser.add_argument('--repeat', type=int, default=1000)
    parser.add_argument('--check-every', type=int, default=10)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.rows, args.repeat, args.check_every) < 1:
        parser.error('rows, repeat, and check-every must be positive')
    for protected in (Path(__file__), args.source_root / 'generative_recommenders/ops/triton/triton_layer_norm.py'):
        if args.report.resolve() == protected.resolve() or (
            args.report.exists() and protected.exists() and args.report.samefile(protected)
        ):
            parser.error('report must not overwrite diagnostic or production source')
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
    counts, firsts = [], []
    for start in range(0, numel, chunk_elements):
        mask = mask_at(start, min(start + chunk_elements, numel))
        count = mask.sum(dtype=torch.int64)
        index = mask.view(torch.uint8).argmax().to(torch.int64) + start
        counts.append(count)
        firsts.append(torch.where(count != 0, index, torch.full_like(index, numel)))
    return torch.stack(counts).sum(), torch.stack(firsts).amin()


def oracle_probe(torch, tensor, expected, atol, chunk_elements=CHUNK_ELEMENTS):
    if not tensor.is_contiguous() or not tensor.numel():
        raise ValueError('oracle requires nonempty contiguous production outputs')
    flat = tensor.view(-1)

    def mask_at(start, stop):
        part = flat[start:stop]
        return part != 0 if expected == 0 and atol == 0 else (
            ~torch.isfinite(part) | ((part - expected).abs() > atol))

    count, index = scan_bad(torch, flat.numel(), flat.device, mask_at, chunk_elements)
    safe = index.clamp(max=flat.numel() - 1)
    value = flat.index_select(0, safe.reshape(1))
    # index_select copies storage: never retain the full Y/mean/rstd allocation.
    sample = (tensor.index_select(0, (safe // tensor.shape[1]).reshape(1))
              if tensor.ndim == 2 else value)
    return {'count': count, 'index': index, 'value': value[0], 'sample': sample,
            'shape': list(tensor.shape), 'expected': expected, 'atol': atol}


def scalar_failures(torch, pending):
    flattened = [(item, name, probe) for item in pending for name, probe in item['outputs'].items()]
    packed = torch.stack([torch.stack((probe['count'].double(), probe['index'].double(), probe['value'].double()))
                          for _, _, probe in flattened]).cpu().tolist()
    failures = []
    for (item, name, probe), (count, index, value) in zip(flattened, packed):
        if count:
            index = int(index)
            coordinate = ([index // FEATURES, index % FEATURES] if name == 'y' else [index])
            failures.append({'iteration': item['iteration'], 'output': name, 'bad_elements': int(count),
                             'first_coordinate': coordinate, 'first_value': finite_json(value),
                             'expected': probe['expected'], 'absolute_tolerance': probe['atol'],
                             'compiled_hash': item['compiled_hash']})
    return failures


def input_bytes_equal(torch, value, snapshot):
    left, right = value.view(torch.uint8).view(-1), snapshot.view(torch.uint8).view(-1)
    return torch.stack([(left[start:start + CHUNK_ELEMENTS] == right[start:start + CHUNK_ELEMENTS]).all()
                        for start in range(0, left.numel(), CHUNK_ELEMENTS)]).all()


def config_record(config):
    fields = ('num_warps', 'num_stages', 'num_ctas', 'llvm_fn_attrs', 'waves_per_eu',
              'enable_fp_fusion', 'maxnreg', 'ir_override')
    return {**config.kwargs, **{name: getattr(config, name) for name in fields if hasattr(config, name)}}


def record_launch(report, compiled, kwargs):
    records = report['actual_compiled_launches']
    found = next((item for item in records if item['compiled_hash'] == compiled.hash), None)
    if found is None:
        metadata = compiled.metadata
        metadata = metadata._asdict() if hasattr(metadata, '_asdict') else metadata
        found = {'compiled_hash': compiled.hash, 'kernel_name': compiled.name,
                 'registers': getattr(compiled, 'n_regs', None), 'spills': getattr(compiled, 'n_spills', None),
                 'compiled_metadata': metadata,
                 'launch_config': {name: value for name, value in kwargs.items() if name in (
                     'BLOCK_N', 'BLOCK_D', 'num_warps', 'num_stages', 'num_ctas', 'llvm_fn_attrs',
                     'waves_per_eu', 'enable_fp_fusion', 'IS_SWISH', 'TRAINING', 'COMPUTE_MEAN_AND_RSTD')},
                 'artifact_sha256': {name: hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()
                                    for name, value in compiled.asm.items() if name in ('llir', 'amdgcn', 'hsaco')},
                 'first_iteration': report['iteration'], 'nonwarmup_launch_count': 0}
        records.append(found)
    found['nonwarmup_launch_count'] += 1
    found['last_iteration'] = report['iteration']
    report['last_compiled_hash'] = compiled.hash


@contextmanager
def forward_launch_scope(module, block_n, report, *, num_warps=None):
    kernel = module._weighted_layer_norm_fwd
    report['production_configs'] = [config_record(config) for config in kernel.configs]
    saved_configs, saved_cache = kernel.configs, dict(kernel.cache)
    filtered = block_n is not None or num_warps is not None
    jit = kernel
    while hasattr(getattr(jit, 'fn', None), 'run'):
        jit = jit.fn
    original_run = jit.run
    try:
        if filtered:
            selected = [config for config in kernel.configs
                        if (block_n is None or config.kwargs.get('BLOCK_N') == block_n)
                        and (num_warps is None or config.num_warps == num_warps)]
            if not selected:
                raise ValueError(f'production forward has no configuration matching BLOCK_N={block_n}, num_warps={num_warps}')
            kernel.configs = selected
            kernel.cache.clear()
        report['eligible_configs'] = [config_record(config) for config in kernel.configs]

        def launch(*positional, **kwargs):
            warmup = kwargs.get('warmup', False)
            if not warmup:
                if not kwargs.get('TRAINING') or not kwargs.get('COMPUTE_MEAN_AND_RSTD') or kwargs.get('IS_SWISH'):
                    raise ValueError('expected ordinary weighted forward with recomputed statistics')
                for index in (1, 4, 5):
                    positional[index].fill_(float('nan'))
            compiled = original_run(*positional, **kwargs)
            if not warmup:
                report['observed_launch_count'] += 1
                report['last_compiled_hash'] = None
                if compiled is not None:
                    record_launch(report, compiled, kwargs)
            return compiled

        jit.run = launch
        yield
    finally:
        jit.run = original_run
        if filtered:
            kernel.configs = saved_configs
            kernel.cache = saved_cache


def compact_failure(torch, args, report, failures, pending, inputs, snapshots, byte_checks):
    rows = {failure['first_coordinate'][0] for failure in failures}
    mutations = {}
    for name, unchanged in byte_checks.items():
        if unchanged:
            continue
        left, right = inputs[name].view(torch.uint8).view(-1), snapshots[name].view(torch.uint8).view(-1)
        count, index = scan_bad(torch, left.numel(), left.device,
                               lambda start, stop: left[start:stop] != right[start:stop])
        count, index = int(count.cpu()), int(index.cpu())
        mutations[name] = {'different_bytes': count, 'first_byte_offset': index}
        if name == 'x':
            rows.add(index // inputs[name].element_size() // FEATURES)
    rows = sorted({row + delta for row in rows for delta in (-1, 0, 1)
                   if 0 <= row + delta < args.rows})[:32] or [0]
    indices = torch.tensor(rows, dtype=torch.int64, device=inputs['x'].device)
    sampled = {name: {'at_checkpoint': (value.index_select(0, indices) if name == 'x' else value).detach().cpu(),
                      'initial_snapshot': (snapshots[name].index_select(0, indices) if name == 'x' else snapshots[name]).detach().cpu()}
               for name, value in inputs.items()}
    evidence = []
    for failure in failures[:32]:
        item = next(item for item in pending if item['iteration'] == failure['iteration'])
        evidence.append({**failure, 'sample': item['outputs'][failure['output']]['sample'].detach().cpu(),
                         'sample_meaning': 'first failing Y row' if failure['output'] == 'y' else 'first failing scalar row statistic'})
    payload = {'format_version': 1, 'diagnostic': report['diagnostic'], 'oracle': report['oracle'],
               'shape': [args.rows, FEATURES], 'input_sample_rows': rows, 'inputs': sampled,
               'input_mutations': mutations, 'outputs': evidence, 'source': report['source'],
               'actual_compiled_launches': report['actual_compiled_launches'],
               'input_snapshot_timing': 'before first forward; current samples are from failure checkpoint'}

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(item) for item in value)
        return 0

    if tensor_bytes(payload) > MAX_ARTIFACT_BYTES // 2:
        raise RuntimeError('compact evidence exceeded tensor budget')
    path = args.report.with_name(args.report.stem + f'.failure.iter{report["iteration"]:06d}.pid{os.getpid()}.pt')
    with path.open('xb') as handle:
        torch.save(payload, handle)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        path.unlink()
        raise RuntimeError('compact evidence exceeded 100 MiB and was removed')
    return {'path': str(path), 'bytes': path.stat().st_size, 'input_sample_rows': rows, 'input_mutations': mutations}


def run(args, report, publish):
    expected_source = args.source_root.resolve() / 'generative_recommenders/ops/triton/triton_layer_norm.py'
    if not expected_source.is_file():
        raise FileNotFoundError(f'production layer-norm source is missing: {expected_source}')
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    import triton
    from generative_recommenders.ops.triton import triton_layer_norm as layer_norm

    if Path(layer_norm.__file__).resolve() != expected_source:
        raise RuntimeError(f'imported layer norm does not match --source-root: {layer_norm.__file__}')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    report['source'] = {'root': str(args.source_root.resolve()), 'layer_norm_path': layer_norm.__file__,
                        'layer_norm_sha256': hashlib.sha256(Path(layer_norm.__file__).read_bytes()).hexdigest(),
                        'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'torch': str(torch.__version__), 'triton': triton.__version__, 'hip': torch.version.hip,
                        'arch': getattr(properties, 'gcnArchName', properties.name)}
    report['environment'] = {name: os.environ.get(name) for name in (
        'AMDGCN_USE_BUFFER_OPS', 'TRITON_FULL_AUTOTUNE', 'TRITON_ALLOW_PIPELINING',
        'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'AMD_SERIALIZE_KERNEL',
        'HSTU_BWD_MAX_VGPR', 'WEIGHTED_LN_BWD_BLOCK_N', 'HSA_ENABLE_COREDUMP',
        'TRITON_HIP_USE_EXPERT_SCHEDULING', 'TRITON_HIP_USE_COEXEC_SCHEDULER',
        'AMDGCN_SCALARIZE_PACKED_FOPS', 'TRITON_HIP_USE_IN_THREAD_TRANSPOSE')}
    report['phase'] = 'setup_inputs'
    publish('setup_start')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        inputs = {'x': torch.zeros((args.rows, FEATURES), device='cuda', dtype=torch.bfloat16),
                  'weight': torch.ones(FEATURES, device='cuda', dtype=torch.bfloat16),
                  'bias': torch.zeros(FEATURES, device='cuda', dtype=torch.bfloat16)}
        snapshots = {name: value.clone() for name, value in inputs.items()}
        versions = {name: value._version for name, value in inputs.items()}
        report['input_bytes'] = sum(value.numel() * value.element_size() for value in inputs.values())
        report['snapshot_bytes'] = report['input_bytes']
        setup = {name: oracle_probe(torch, value, 1 if name == 'weight' else 0, 0)
                 for name, value in inputs.items()}
        setup_counts = torch.stack([probe['count'] for probe in setup.values()]).cpu().tolist()
        report['setup_bad_elements'] = dict(zip(setup, setup_counts))
        del setup
        if any(setup_counts):
            report.update(status='SETUP_FAILURE', phase='setup_inputs')
            publish('setup_failure')
            return 1
        pending = []
        started = time.monotonic()
        with forward_launch_scope(layer_norm, args.block_n, report, num_warps=args.num_warps):
            for iteration in range(1, args.repeat + 1):
                report.update(iteration=iteration, phase='weighted_forward', last_compiled_hash=None)
                prior_launches = report['observed_launch_count']
                outputs = layer_norm.triton_weighted_layer_norm_fwd(**inputs, eps=EPS)
                if report['observed_launch_count'] == prior_launches or report['last_compiled_hash'] is None:
                    raise RuntimeError('public helper returned without a recorded forward JIT launch')
                report['phase'] = 'bounded_output_oracle_checks'
                expected_shapes = ((args.rows, FEATURES), (args.rows,), (args.rows,))
                expected_dtypes = (torch.bfloat16, torch.float32, torch.float32)
                if len(outputs) != 3 or any(not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                                           or value.dtype != dtype or value.device != inputs['x'].device
                                           for value, shape, dtype in zip(outputs, expected_shapes, expected_dtypes)):
                    raise ValueError('production forward output structure/shape/dtype/device changed')
                pending.append({'iteration': iteration, 'compiled_hash': report['last_compiled_hash'],
                                'outputs': {name: oracle_probe(torch, value, RSTD_EXPECTED if name == 'rstd' else 0,
                                                                RSTD_ATOL if name == 'rstd' else 0)
                                            for name, value in zip(OUTPUTS, outputs)}})
                del outputs
                if iteration % args.check_every and iteration != args.repeat:
                    continue
                report['phase'] = 'checkpoint_input_byte_checks'
                byte_flags = {name: input_bytes_equal(torch, value, snapshots[name]) for name, value in inputs.items()}
                byte_checks = dict(zip(byte_flags, torch.stack(list(byte_flags.values())).cpu().tolist()))
                tracked = {name: {'before': versions[name], 'after': value._version}
                           for name, value in inputs.items() if value._version != versions[name]}
                failures = scalar_failures(torch, pending)
                failed = bool(failures or tracked or not all(byte_checks.values()))
                report['checks'].append({'through': iteration, 'seconds': time.monotonic() - started,
                                         'output_failures': failures, 'input_bytes_unchanged': byte_checks,
                                         'tracked_input_changes': tracked,
                                         'final_helper_launches': [{key: item[key] for key in ('iteration', 'compiled_hash')}
                                                                   for item in pending]})
                report.update(iterations_completed=iteration, status='FAIL' if failed else (
                    'PASS' if iteration == args.repeat else 'RUNNING'), phase='checkpoint')
                if failed:
                    report['positive_reproducer'] = bool(failures)
                    publish('failure_detected')
                    try:
                        report['failure_artifact'] = compact_failure(torch, args, report, failures, pending,
                                                                    inputs, snapshots, byte_checks)
                    except Exception as error:
                        report['failure_artifact_error'] = repr(error)
                    publish('failure')
                    return 1
                pending.clear()
                publish('checkpoint')
        report.update(phase='complete', peak_device_allocated_bytes=torch.cuda.max_memory_allocated())
        publish('complete')
    return 0


def main():
    args = arguments()
    report = {'diagnostic': 'weighted_ln_zero_x_forward', 'status': 'RUNNING', 'positive_reproducer': False,
              'phase': 'initialization', 'iteration': 0, 'iterations_completed': 0,
              'rows': args.rows, 'features': FEATURES, 'dtype': 'torch.bfloat16', 'eps': EPS,
              'block_n_filter': args.block_n, 'num_warps_filter': args.num_warps,
              'repeat': args.repeat, 'check_every': args.check_every,
              'oracle': {'y': {'expected': 0, 'atol': 0}, 'mean': {'expected': 0, 'atol': 0},
                         'rstd': {'expected': RSTD_EXPECTED, 'atol': RSTD_ATOL, 'rtol': 0, 'finite_required': True}},
              'poison': 'Y/Mean/Rstd filled with NaN before every nonwarmup recomputing weighted forward launch',
              'observed_launch_count': 0, 'actual_compiled_launches': [], 'checks': [],
              'chunk_elements': CHUNK_ELEMENTS,
              'limitations': ['zero-input control does not validate nonzero-input training',
                              'poisoning affects autotune timing and may change selected configuration',
                              'only final helper outputs are checked; autotune candidates can overwrite earlier results',
                              'input bytes checked at checkpoints; intermediate changes that revert are not excluded',
                              'poisoning, probes, and snapshots alter allocation reuse, timing, and memory traffic',
                              'checker phase does not identify the originating kernel of an asynchronous GPU error']}

    def publish(event):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(finite_json(report), indent=2, default=str, allow_nan=False) + '\n')
        print(json.dumps({'event': event, 'status': report['status'], 'phase': report['phase'],
                          'iteration': report['iteration'], 'block_n_filter': args.block_n,
                          'num_warps_filter': args.num_warps,
                          'report': str(args.report)}, allow_nan=False), flush=True)

    try:
        return run(args, report, publish)
    except Exception as error:
        report.update(status='ERROR', error=repr(error),
                      error_attribution='reporting phase is not proof of the originating GPU kernel')
        publish('error')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
