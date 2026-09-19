#!/usr/bin/env python3
"""Replay the pinned W low-half sign-bit counterfactual with the frozen DX path.

The derived bundle changes only W[:,18::32] BF16 sign bits. The frozen driver
generates the original full zero-DZ layout and runs one DX=torch.mm(DZ,W.T)
per iteration. The exact-zero oracle is unchanged. Failure onset, tile and
coordinates need not repeat the observations made with the original W.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
import repro_projection_dx_zero as frozen

COUNTERFACTUAL_BUNDLE_SHA256 = 'af1a536c4490daec2d5e79911c110c9141f896b9fccb83a3de09ddb1a10ad9f2'
COUNTERFACTUAL_W_SHA256 = '9dd8b0f4883a899dbefbf60d845beea1acbf11efbf904813b178cfb0dfe95832'
SOURCE_BUNDLE_SHA256 = '8d77dd0db4330ed098389b57804cbb55b8f82c4840ae591e04a60491abbac112'
EXPORTER_SHA256 = '191449030c848adbea482cf8b426b9e397ea3d54ea8222ec78ccf961bf5054eb'
FROZEN_DRIVER_SHA256 = '49d0e58e32fd3583f500655525b03eaaad8579bb72b842c98f695eab0aaf9614'
INTERVENTION_FORMAT = 'projection_packed_W_low_half_sign_counterfactual_v1'
INTERVENTION_OPERATION = 'W[:,18::32].raw_BF16_bits XOR 0x8000'
LIMITS = [
    'One direct DX matrix multiplication per call, with derived counterfactual W and zero-generated DZ at the captured full layout.',
    'Only W[:,18::32] raw BF16 sign bits were changed; paired high halves, magnitude bits, remaining W bytes, DZ layout and execution controls are preserved.',
    'No DB/DW, surrounding training workload, or claim of original allocator equivalence.',
    'Failure call, tile, coordinate and frequency are not predicted. A new failure must be analyzed against its current input bytes, not an old coordinate mask.',
    'The unchanged exact-zero oracle tests the direct DX boundary. A packed-word representation match alone does not identify a faulty instruction or hardware cause.',
    'BLAS preference and loaded libraries do not identify the executed kernel or establish byte-identical loaded code.',
]


def validate_counterfactual_bundle(bundle):
    """Check explicit derived labels before frozen.prepare allocates full DZ."""
    if (not isinstance(bundle, dict) or bundle.get('format') != frozen.BUNDLE_FORMAT
            or bundle.get('w_storage_sha256') != COUNTERFACTUAL_W_SHA256):
        raise ValueError('Requires the pinned derived counterfactual W bundle')
    provenance = bundle.get('provenance') or {}
    intervention = provenance.get('counterfactual_intervention') or {}
    if (intervention.get('format') != INTERVENTION_FORMAT
            or intervention.get('operation') != INTERVENTION_OPERATION
            or (provenance.get('source_bundle') or {}).get('sha256') != SOURCE_BUNDLE_SHA256
            or (provenance.get('exporter') or {}).get('sha256') != EXPORTER_SHA256
            or not isinstance(provenance.get('source_bundle_provenance'), dict)
            or not provenance['source_bundle_provenance']):
        raise ValueError('Counterfactual intervention or source provenance mismatch')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--failure-dump', type=Path, required=True)
    parser.add_argument('--backend', choices=('captured', 'default', 'hipblas', 'hipblaslt'), default='captured')
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 1000 or args.cpu_threads < 1:
        parser.error('Repeats 1..1000 and positive CPU threads required')
    if len({p.resolve() for p in (args.bundle, args.report, args.failure_dump)}) != 3:
        parser.error('Bundle, report and failure paths must be distinct')
    support = frozen.support
    support.new_path(args.report)
    support.new_path(args.failure_dump)
    temporary = args.report.with_name(args.report.name + '.update')
    if os.path.lexists(temporary):
        raise FileExistsError(str(temporary))
    if args.gpu:
        for name in frozen.MANDATORY_GPU_ENVIRONMENT:
            if os.environ.get(name) != '0':
                parser.error(name + '=0 required before input load and GPU initialization')
    root = Path(__file__).resolve().parents[1]
    sources = tuple(dict.fromkeys((*support.SOURCE_FILES, 'scripts/repro_projection_dx_zero.py',
                                  'scripts/repro_projection_packed_w_counterfactual.py')))
    inventory = {name: support.file_hash(root / name) for name in sources}
    bundle_sha256 = support.file_hash(args.bundle)
    if inventory['scripts/repro_projection_dx_zero.py'] != FROZEN_DRIVER_SHA256:
        raise ValueError('Frozen DX driver identity mismatch')
    if bundle_sha256 != COUNTERFACTUAL_BUNDLE_SHA256:
        raise ValueError('Counterfactual bundle identity mismatch')
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    bundle = torch.load(args.bundle, weights_only=False, map_location='cpu')
    validate_counterfactual_bundle(bundle)
    prepared = frozen.prepare(bundle)
    backend = frozen.BACKENDS.get(prepared['blas_preference']) if args.backend == 'captured' else args.backend
    if backend is None:
        parser.error('Bundle has no supported captured backend preference; choose an explicit backend')
    report = {
        'format': 'projection_packed_W_counterfactual_zero_report_v1', 'status': 'CPU_PREPARED',
        'bundle': {'path': str(args.bundle), 'sha256': bundle_sha256},
        'bundle_provenance': prepared['bundle_provenance'],
        'counterfactual_intervention': bundle['provenance']['counterfactual_intervention'],
        'mathematical_DX_oracle': 'All numerical zeros; finite derived W multiplied by exact-zero DZ',
        'requested_backend': backend,
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'source_sha256_before': inventory,
        'frozen_driver_sha256_required': FROZEN_DRIVER_SHA256,
        'versions': {'torch': str(torch.__version__), 'hip': torch.version.hip},
        'environment': {name: os.environ.get(name) for name in (
            'HIPBLASLT_TENSILE_LIBPATH', 'ROCBLAS_TENSILE_LIBPATH', 'HIPBLASLT_WORKSPACE_SIZE',
            'CUBLASLT_WORKSPACE_SIZE', 'TENSILE_STREAMK_DATA_PARALLEL', 'TENSILE_STREAMK_FIXED_GRID',
            'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
            'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL',
            *frozen.MANDATORY_GPU_ENVIRONMENT)},
        'input_digests': prepared['input_digests'], 'execution_controls': prepared['execution_controls'],
        'limits': list(LIMITS),
    }
    support.atomic_new_save(args.report, lambda stream: stream.write(
        (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()))
    try:
        if args.gpu:
            report.update(device=torch.cuda.get_device_name(0), blas_preference_before=support.blas_preference())
            report.update(frozen.run(
                prepared, failure_path=args.failure_dump, repeats=args.repeats, backend=backend,
                configuration=report.copy(),
                progress=lambda rec: print(json.dumps({'iteration': rec['iteration'], 'failed': rec['failed']}), flush=True)))
            report['loaded_blas_libraries'] = support.loaded_blas_libraries()
    except BaseException as error:
        report.update(status='ERROR', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        report['source_sha256_after'] = {name: support.file_hash(root / name) for name in sources}
        report['source_files_unchanged'] = report['source_sha256_after'] == inventory
        if not report['source_files_unchanged']:
            report['status'] = 'SOURCE_CHANGED'
        report['seconds'] = time.monotonic() - started
        with temporary.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.report)
    print(json.dumps({'status': report['status'], 'iterations_completed': report.get('iterations_completed')}), flush=True)
    return 0 if report['status'] in ('PASS', 'CPU_PREPARED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
