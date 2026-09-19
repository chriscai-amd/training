#!/usr/bin/env python3
"""Exact constant-input controls for the production projection backward.

These are generated inputs at an explicit shape, not captured training operands.
Both controls have exact BF16 answers even with FP32 reduction. The nonzero
control uses one nonzero DZ column and X=0 to test every DX store independently
of an approximate GEMM reference. Existing first-failure tensors are retained.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import time

import stress_addmm_backward_boundary as support


def prepare_constant(rows, width, projected, case, *, chunk_bytes=64 << 20,
                     max_bytes=32 << 30):
    import torch
    if (any(type(v) is not int or v < 1 for v in (rows, width, projected))
            or rows > (1 << 24) or case not in ('zero', 'one_column')):
        raise ValueError('Positive dimensions, rows <=2^24, and a known case required')
    needed = 2 * (rows * width * 2 + width * projected * 2
                  + rows * projected + projected)
    if needed > max_bytes:
        raise ValueError('Generated inputs and references exceed CPU budget')
    x = torch.full((rows, width), 0.125 if case == 'zero' else 0., dtype=torch.bfloat16)
    w = torch.full((width, projected), 0.03125, dtype=torch.bfloat16)
    dz = torch.zeros((rows, projected), dtype=torch.bfloat16)
    dx = torch.zeros_like(x)
    dw = torch.zeros_like(w)
    db = torch.zeros(projected, dtype=torch.bfloat16)
    if case == 'one_column':
        dz[:, 0] = 2. ** -20
        dx.fill_(2. ** -25)
        db[0] = rows * 2. ** -20
    values = {'x': x, 'w': w, 'dz': dz, 'is_y_1d': True}
    original = {'inputs': values, 'references': (dx, dw, db)}
    controls = {
        'autocast': {'cpu': {'enabled': False, 'dtype': 'torch.bfloat16'},
                    'cuda': {'enabled': False, 'dtype': 'torch.float16'}},
        'float32_matmul_precision': 'highest', 'allow_tf32': False,
        'allow_fp16_reduced_precision_reduction': True,
        'allow_bf16_reduced_precision_reduction': True,
    }
    inputs = support.digests(values, chunk_bytes=chunk_bytes, threshold=1e6)
    refs = support.digests(support.reference_map(original['references']),
                           chunk_bytes=chunk_bytes, threshold=1e6)
    return {'original': original, 'prepared': original, 'controls': controls,
            'original_input_digests': inputs, 'prepared_input_digests': inputs,
            'original_reference_digests': refs, 'prepared_reference_digests': refs,
            'transformation': {
                'mode': 'generated_projection_' + case + '_v1',
                'zero_oracle': case == 'zero',
                'provenance': 'Synthetic constants; no claim of captured training bytes',
                'oracle': ('DZ=0, finite X=1/8,W=1/32: DX,DW,DB are numerical zero.'
                           if case == 'zero' else
                           'X=0,W=1/32,DZ[:,0]=2^-20 and other DZ=0: '
                           'DX=2^-25 everywhere,DW=0,DB[0]=BF16(M*2^-20),others0. '
                           'M<=2^24 makes the positive FP32 partial sums exact. '
                           'Actual reduced-precision implementation is part of the tested component.')}}


def write_json(path, value):
    temporary = path.with_name(path.name + '.update')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--projected', type=int, default=2048)
    parser.add_argument('--case', choices=('zero', 'one_column'), required=True)
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--failure-dump', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 1000 or args.cpu_threads < 1:
        parser.error('Repeats1..1000 and positive CPU threads required')
    if args.report.resolve() == args.failure_dump.resolve():
        parser.error('Distinct report and failure paths required')
    support.new_path(args.report)
    support.new_path(args.failure_dump)
    for key in ('AMDGCN_USE_BUFFER_OPS', 'TRITON_FULL_AUTOTUNE', 'TRITON_ALLOW_PIPELINING'):
        if os.environ.get(key) != '0':
            parser.error(key + '=0 required before imports')
    import torch
    torch.set_num_threads(args.cpu_threads)
    root = Path(__file__).resolve().parents[1]
    sources = (*support.SOURCE_FILES, 'scripts/repro_projection_backward_constant.py')
    inventory = {name: support.file_hash(root / name) for name in sources}
    report = {'format': 'projection_backward_constant_report_v1', 'status': 'PREPARING',
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_sha256_before': inventory,
              'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'hip': torch.version.hip},
              'limits': ['Generated constants at the stated shape; no training-state equivalence.',
                         'BLAS preference/library hashes do not identify an executed kernel.',
                         *support.LIMITS]}
    support.atomic_new_save(args.report, lambda stream: stream.write((json.dumps(report, indent=2)+'\n').encode()))
    started = time.monotonic()
    try:
        prepared = prepare_constant(args.rows, args.width, args.projected, args.case)
        report.update(status='READY', transformation=prepared['transformation'],
                      resource_plan=support.resource_plan(prepared),
                      prepared_input_digests=prepared['prepared_input_digests'],
                      prepared_reference_digests=prepared['prepared_reference_digests'])
        write_json(args.report, report)
        if args.gpu:
            # Import and execute the checked-in public function, in original order.
            module = importlib.import_module('generative_recommenders.ops.triton.triton_addmm')
            report.update(device=torch.cuda.get_device_name(0), blas_preference=support.blas_preference())
            def progress(record):
                print(json.dumps({'iteration': record['iteration'], 'failed': record['failed'],
                                  'classifications': record['classifications']}), flush=True)
            result = support.run_stress(
                {'format': 'synthetic_projection_constants_v1', 'provenance': prepared['transformation']},
                prepared, function=module.triton_addmm_bwd, failure_path=args.failure_dump,
                configuration=report.copy(), repeats=args.repeats,
                strict_exact=True, rtol=0, atol=0, progress=progress)
            report.update(result, loaded_blas_libraries=support.loaded_blas_libraries())
        else:
            report['status'] = 'CPU_PREPARED'
        report['source_sha256_after'] = {name: support.file_hash(root / name) for name in sources}
        report['source_files_unchanged'] = report['source_sha256_after'] == inventory
        if not report['source_files_unchanged']:
            report['status'] = 'SOURCE_CHANGED'
    except BaseException as error:
        report.update(status='ERROR', error={'type': type(error).__name__, 'message': str(error)})
        write_json(args.report, report)
        raise
    report['seconds'] = time.monotonic() - started
    write_json(args.report, report)
    print(json.dumps({k: report.get(k) for k in ('status', 'iterations_completed', 'seconds')}), flush=True)
    return 0 if report['status'] in ('PASS', 'CPU_PREPARED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
