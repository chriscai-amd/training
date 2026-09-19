#!/usr/bin/env python3
"""Isolate the captured column-major Addmm weight-gradient multiplication.

Runs only torch.mm(DY.T, X).T, with the actual saved X and zero-transformed
DY. Reuses the frozen replay's full input checks, exact-zero oracle, first
failure retention and execution controls. No forward, DX or DB is computed.
The native kernel must be established separately from the launch trace.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import replay_additional_linear_capture as frozen

FROZEN_SHA256 = 'd7f84fb69cf15489f4480ab22c5f519cb8b0a21759b9952ce876302b7ad24c6f'
OPERATION = 'torch.mm(DY.T, X).T'
MANDATORY = ('AMDGCN_USE_BUFFER_OPS', 'TRITON_FULL_AUTOTUNE', 'TRITON_ALLOW_PIPELINING')


def validate_saved_orientation(inputs):
    x, mat2, dy = (inputs[k] for k in ('x', 'mat2', 'dy'))
    if (x.ndim != 2 or mat2.ndim != 2 or dy.ndim != 2
            or tuple(mat2.stride()) != (1, mat2.shape[0])
            or tuple(x.stride()) != (x.shape[1], 1)
            or tuple(dy.stride()) != (dy.shape[1], 1)
            or x.dtype != dy.dtype or x.dtype != mat2.dtype
            or tuple(dy.shape) != (x.shape[0], mat2.shape[1])
            or x.shape[1] != mat2.shape[0]):
        raise ValueError('Requires captured contiguous X/DY and column-major saved mat2')


def execute_saved_orientation(resident, metadata, route):
    if route != 'raw_dw':
        raise ValueError('Only the frozen raw_dw checking harness is used')
    values = resident['pristine_inputs']
    validate_saved_orientation(values)
    torch = frozen.common._torch()
    with frozen.restored_blas_preference(metadata), frozen.helpers.restored_execution_controls(
            metadata['execution_controls_backward']), torch.no_grad():
        result = torch.mm(values['dy'].t(), values['x']).t()
    if result.stride() != values['mat2'].stride():
        raise ValueError('Raw output layout differs from saved Addmm weight-gradient orientation')
    return {'raw_dw': result}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--failure-dump', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--repeats', type=int, default=1000)
    p.add_argument('--cpu-threads', type=int, default=8)
    a = p.parse_args(argv)
    if not 1 <= a.repeats <= 1000 or a.cpu_threads < 1:
        p.error('Repeats 1..1000 and positive CPU thread count required')
    paths = [a.capture, a.context, a.report, a.failure_dump,
             a.report.with_name(a.report.name + '.update'),
             a.failure_dump.with_name(a.failure_dump.name + '.tmp')]
    if len({x.resolve() for x in paths}) != len(paths):
        p.error('Inputs, outputs and temporary paths must be distinct')
    for path in paths[2:]:
        if os.path.lexists(path):
            raise FileExistsError(str(path))
    support = frozen.support
    sources = {**frozen.source_inventory(),
               'repro_linear_dw_saved_orientation.py': support.file_hash(__file__)}
    if support.file_hash(frozen.__file__) != FROZEN_SHA256:
        raise ValueError('Frozen replay implementation differs from reviewed source')
    if a.gpu and any(os.environ.get(k) != '0' for k in MANDATORY):
        p.error('All three mandatory GPU environment controls must be zero')
    context = json.loads(a.context.read_text())
    restored = support.restore_environment(context)
    torch = frozen.common._torch()
    torch.set_num_threads(a.cpu_threads)
    versions = frozen.runtime_versions()
    if a.gpu:
        frozen.validate_runtime(context, versions)
    start = time.monotonic()
    payload = torch.load(a.capture, map_location='cpu', mmap=True, weights_only=False)
    frozen.validate_capture(payload)
    validate_saved_orientation(payload['pristine_inputs'])
    if payload['metadata'].get('preferred_blas_library') is None:
        raise ValueError('Captured BLAS preference required')
    prepared = frozen.prepare_capture(payload, zero_dy=True)
    report = {
        'format': 'linear_saved_orientation_zero_DW_report_v1', 'status': 'CPU_PREPARED',
        'operation': OPERATION, 'frozen_harness_route': 'raw_dw',
        'configuration': {'operation': OPERATION, 'injected_callable': 'execute_saved_orientation',
                          'source_sha256_before': sources,
                          'capture': {'path': str(a.capture), 'sha256': support.file_hash(a.capture)},
                          'context': {'path': str(a.context), 'sha256': support.file_hash(a.context)},
                          'restored_environment': restored, 'versions': versions,
                          'transformation': prepared['transformation'],
                          'prepared_digest': prepared['prepared_digest']},
        'limits': ['Only the saved-orientation DW multiplication is executed per call.',
                   'Original X/mat2 are retained; only DY is zeroed. Every output must be numerical zero.',
                   'Failure onset is intermittent; isolated allocator/history differs from training.',
                   'Named native kernel and argument identity require separate trace verification.',
                   'Original captured anomalous DW is retained as evidence, not an expected answer.'],
    }
    temporary = paths[4]
    def save():
        with temporary.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(a.report)
    save()
    try:
        if a.gpu:
            result = frozen.run_replay(payload, prepared, route='raw_dw', repeats=a.repeats,
                device='cuda:0', failure_path=a.failure_dump, function=execute_saved_orientation,
                max_resident_bytes=32 << 30, max_capture_bytes=32 << 30,
                configuration=report['configuration'],
                progress=lambda row: print(json.dumps({'iteration': row['iteration'],
                    'failed': row['failed'], 'outputs': row['outputs']}), flush=True))
            report.update(result)
    except BaseException as error:
        report.update(status='ERROR', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        after = {**frozen.source_inventory(),
                 'repro_linear_dw_saved_orientation.py': support.file_hash(__file__)}
        report.update(source_sha256_after=after, source_files_unchanged=after == sources,
                      seconds=time.monotonic() - start)
        if after != sources:
            report['status'] = 'SOURCE_CHANGED'
        save()
    print(json.dumps({'status': report['status'], 'iterations_completed': report.get('iterations_completed')}))
    return 0 if report['status'] in ('PASS', 'CPU_PREPARED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
