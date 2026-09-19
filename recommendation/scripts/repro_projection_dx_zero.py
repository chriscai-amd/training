#!/usr/bin/env python3
"""Run only DX=torch.mm(DZ,W.T), with retained finite W and exact-zero DZ.

A small trusted CPU bundle supplies the actual W bytes and original DZ layout.
DZ is generated as zero at that layout. There is one matrix multiplication per
iteration, no warmup, and no DB or DW operation. The first failing existing DX
and both current inputs are retained without rerunning the multiplication.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import torch
import stress_addmm_backward_boundary as support
from nan_backward_boundaries import _cpu_copy_tree

BUNDLE_FORMAT = 'retained_projection_W_and_zero_DZ_layout_v1'
BACKENDS = {'_BlasBackend.Default': 'default', '_BlasBackend.Cublas': 'hipblas',
            '_BlasBackend.Cublaslt': 'hipblaslt'}
MANDATORY_GPU_ENVIRONMENT = ('AMDGCN_USE_BUFFER_OPS', 'TRITON_FULL_AUTOTUNE', 'TRITON_ALLOW_PIPELINING')


def prepare(bundle, *, max_bytes=32 << 30, chunk_bytes=64 << 20):
    if bundle.get('format') != BUNDLE_FORMAT:
        raise ValueError('Requires a retained projection W / zero-DZ layout bundle')
    w, layout = bundle['w'], bundle['dz_layout']
    if (not isinstance(w, torch.Tensor) or w.device.type != 'cpu' or w.dtype != torch.bfloat16
            or w.ndim != 2 or min(w.shape) < 1 or not w.is_contiguous()
            or w.storage_offset() != 0 or w.untyped_storage().nbytes() != w.numel() * 2):
        raise ValueError('Original W must be full-storage contiguous CPU BF16 matrix')
    shape, stride = layout.get('shape'), layout.get('stride')
    if (not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
            or shape[1] != w.shape[1] or stride != [shape[1], 1]
            or layout.get('storage_offset') != 0 or layout.get('dtype') != 'torch.bfloat16'
            or layout.get('storage_bytes') != shape[0] * shape[1] * 2):
        raise ValueError('Requires the original full contiguous BF16 DZ layout')
    required = w.untyped_storage().nbytes() + layout['storage_bytes']
    if required > max_bytes:
        raise ValueError('Pristine inputs exceed CPU storage budget')
    raw = memoryview(w.view(torch.uint8).numpy()).cast('B')
    if hashlib.sha256(raw).hexdigest() != bundle['w_storage_sha256']:
        raise ValueError('Retained W backing checksum mismatch')
    if not bool(torch.isfinite(w).all()):
        raise ValueError('The exact-zero oracle requires finite retained W')
    controls = bundle['execution_controls']
    support.validate_controls(controls)
    if any(value['enabled'] for value in controls['autocast'].values()):
        raise ValueError('Captured saved-operand call must have autocast disabled')
    inputs = {'w': w, 'dz': torch.zeros(shape, dtype=torch.bfloat16)}
    return {'inputs': inputs, 'execution_controls': controls, 'bundle_provenance': bundle.get('provenance'),
            'blas_preference': bundle.get('blas_preference'),
            'input_digests': support.digests(inputs, chunk_bytes=chunk_bytes, threshold=1e6)}


def check_zero(output, *, shape, dtype, device, chunk_elements=1 << 24):
    if (not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(shape)
            or output.dtype != dtype or output.device != device):
        raise ValueError('DX shape, dtype or device differs from the actual mm contract')
    total = torch.zeros(4, dtype=torch.int64, device=output.device)
    for chunk in support.common._logical_chunks(output.detach(), chunk_elements):
        bits = chunk.contiguous().view(torch.int16)
        magnitude = bits & 0x7fff
        total[:3] += torch.stack(((magnitude != 0).sum(), (magnitude == 0x7f80).sum(),
                                 (magnitude > 0x7f80).sum()))
        total[3] = torch.maximum(total[3], magnitude.max().to(torch.int64))
    nonzero, infinity, nan, maximum_bits = total.cpu().tolist()
    return {'nonzero_magnitude_bits_count': nonzero, 'infinity_count': infinity, 'nan_count': nan,
            'maximum_magnitude_bits': maximum_bits,
            'maximum_abs_if_finite': None if maximum_bits >= 0x7f80 else struct.unpack('>f', struct.pack('>I', maximum_bits << 16))[0],
            'failed': bool(nonzero), 'oracle': 'All numerical zeros; signed zeros accepted and subnormal/nonfinite bits rejected'}


@contextmanager
def selected_backend(backend):
    previous = str(torch.backends.cuda.preferred_blas_library())
    if backend is None:
        yield {'requested': None, 'observed': previous}
        return
    if backend not in ('default', 'hipblas', 'hipblaslt') or previous not in BACKENDS:
        raise ValueError('Supported/restorable BLAS preference required')
    expected = {value: key for key, value in BACKENDS.items()}[backend]
    try:
        torch.backends.cuda.preferred_blas_library(backend)
        effective = str(torch.backends.cuda.preferred_blas_library())
        # ROCm resolves the symbolic default to a concrete preference getter.
        if effective not in BACKENDS or (backend != 'default' and effective != expected):
            raise ValueError('BLAS preference did not restore exactly')
        yield {'requested': backend, 'observed': effective,
               'scope': 'PyTorch preference, not proof of the dispatched library/kernel'}
    finally:
        torch.backends.cuda.preferred_blas_library(BACKENDS[previous])


def run(prepared, *, failure_path, repeats=1000, device='cuda:0', backend=None,
        function=None, chunk_bytes=64 << 20, chunk_elements=1 << 24,
        max_resident_bytes=32 << 30, max_capture_bytes=64 << 30, progress=None, configuration=None):
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError('Repeats must be1..1000')
    support.new_path(failure_path)
    before = prepared['inputs']
    expected_shape = (before['dz'].shape[0], before['w'].shape[0])
    before_bytes = support.storage_bytes(before)
    output_bytes = expected_shape[0] * expected_shape[1] * before['w'].element_size()
    if before_bytes + output_bytes > max_resident_bytes or 2 * before_bytes + output_bytes > max_capture_bytes:
        raise ValueError('Resident inputs/output or complete failure exceed budget before dispatch')
    resident, copied = support.common.restore_raw_tree(before, device=device, chunk_bytes=chunk_bytes,
                                                      max_bytes=max_resident_bytes)
    def integrity(current):
        return support.digests(current, chunk_bytes=chunk_bytes, threshold=1e6) == prepared['input_digests']
    if not integrity(resident):
        raise ValueError('Initial resident inputs differ from pristine full backing bytes')
    versions = {name: tensor._version for name, tensor in resident.items()}
    records = []
    function = torch.mm if function is None else function
    with selected_backend(backend) as selection:
        for iteration in range(1, repeats + 1):
            with support.helpers.restored_execution_controls(prepared['execution_controls']) as controls, torch.no_grad():
                output = function(resident['dz'], resident['w'].t())
            checks = check_zero(output, shape=expected_shape, dtype=resident['w'].dtype,
                                device=resident['w'].device, chunk_elements=chunk_elements)
            changed = [name for name, tensor in resident.items() if tensor._version != versions[name]]
            record = {'iteration': iteration, 'checks': checks, 'changed_input_versions': changed,
                      'execution_controls': controls, 'blas_preference': selection,
                      'failed': checks['failed'] or bool(changed)}
            if not record['failed'] and iteration == repeats:
                record['final_input_bytes_unchanged'] = integrity(resident)
                record['failed'] |= not record['final_input_bytes_unchanged']
            records.append(record)
            if record['failed']:
                # Copy the existing first failed call, with no following mm.
                current, size = _cpu_copy_tree({'inputs': resident, 'dx': output}, chunk_bytes,
                                              max_capture_bytes - before_bytes)
                current_digests = support.digests(current['inputs'], chunk_bytes=chunk_bytes, threshold=1e6)
                saved_checks = check_zero(current['dx'], shape=expected_shape, dtype=before['w'].dtype,
                                         device=before['w'].device, chunk_elements=chunk_elements)
                observation_matches = saved_checks == checks
                status = 'FAIL' if observation_matches else 'CAPTURE_INTEGRITY_FAILURE'
                source_root = Path(__file__).resolve().parents[1]
                sources = tuple(dict.fromkeys((*support.SOURCE_FILES, 'scripts/repro_projection_dx_zero.py')))
                artifact = {'format': 'projection_dx_only_zero_failure_v1', 'format_version': 1,
                    'status': status,
                    'operation': 'torch.mm(DZ,W.T)', 'iteration': iteration, 'record': record,
                    'pristine_inputs': before, 'current_at_failure': current,
                    'pristine_input_digests': prepared['input_digests'], 'current_input_digests': current_digests,
                    'input_backing_bytes_unchanged': current_digests == prepared['input_digests'],
                    'saved_DX_zero_check': saved_checks, 'bundle_provenance': prepared['bundle_provenance'],
                    'saved_DX_check_matches_observed': observation_matches,
                    'capture_integrity_failure': not observation_matches,
                    'configuration': copy.deepcopy(configuration),
                    'runtime_versions': {'torch': str(torch.__version__), 'hip': torch.version.hip},
                    'source_sha256_at_failure': {name: support.file_hash(source_root / name) for name in sources},
                    'storage_bytes': before_bytes + size,
                    'scope': 'Only the single DX torch.mm was invoked per iteration; no DB/DW calls or warmup. Existing first failed output retained without rerun. Full before/failure equality does not exclude transient internal mutation.'}
                support.atomic_new_save(failure_path, lambda stream: torch.save(artifact, stream))
                if progress:
                    progress(record)
                return {'status': status, 'iterations_completed': iteration, 'iterations': records,
                        'failure_capture': {'path': str(failure_path), 'bytes': Path(failure_path).stat().st_size,
                            'sha256': support.file_hash(failure_path), 'input_backing_bytes_unchanged': artifact['input_backing_bytes_unchanged'],
                            'saved_DX_check_matches_observed': observation_matches,
                            'saved_DX_zero_check': saved_checks}, 'resident_input_bytes': copied}
            del output
            if progress:
                progress(record)
    return {'status': 'PASS', 'iterations_completed': repeats, 'iterations': records,
            'final_input_bytes_unchanged': records[-1]['final_input_bytes_unchanged'], 'resident_input_bytes': copied}


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
        parser.error('Repeats1..1000 and positive CPU threads required')
    if len({p.resolve() for p in (args.bundle, args.report, args.failure_dump)}) != 3:
        parser.error('Bundle, report and failure paths must be distinct')
    support.new_path(args.report)
    support.new_path(args.failure_dump)
    if args.gpu:
        for name in MANDATORY_GPU_ENVIRONMENT:
            if os.environ.get(name) != '0':
                parser.error(name + '=0 required before GPU initialization')
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    bundle = torch.load(args.bundle, weights_only=False, map_location='cpu')
    prepared = prepare(bundle)
    backend = BACKENDS.get(prepared['blas_preference']) if args.backend == 'captured' else args.backend
    if backend is None:
        parser.error('Bundle has no supported captured backend preference; choose an explicit backend')
    root = Path(__file__).resolve().parents[1]
    sources = tuple(dict.fromkeys((*support.SOURCE_FILES, 'scripts/repro_projection_dx_zero.py')))
    inventory = {name: support.file_hash(root / name) for name in sources}
    report = {'format': 'projection_dx_only_zero_report_v1', 'status': 'CPU_PREPARED',
        'bundle': {'path': str(args.bundle), 'sha256': support.file_hash(args.bundle)},
        'bundle_provenance': prepared['bundle_provenance'], 'requested_backend': backend,
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'source_sha256_before': inventory,
        'versions': {'torch': str(torch.__version__), 'hip': torch.version.hip},
        'environment': {name: os.environ.get(name) for name in ('HIPBLASLT_TENSILE_LIBPATH', 'ROCBLAS_TENSILE_LIBPATH',
            'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'LD_LIBRARY_PATH', *MANDATORY_GPU_ENVIRONMENT)},
        'input_digests': prepared['input_digests'], 'execution_controls': prepared['execution_controls'],
        'limits': ['One directDX matrix multiplication per call, with original retained W and zero-generated DZ at the captured layout.',
                   'No DB/DW, surrounding training workload, or claim of original allocator equivalence.',
                   'BLAS preference and loaded libraries do not identify the executed kernel.']}
    support.atomic_new_save(args.report, lambda stream: stream.write((json.dumps(report, indent=2) + '\n').encode()))
    try:
        if args.gpu:
            report.update(device=torch.cuda.get_device_name(0), blas_preference_before=support.blas_preference())
            report.update(run(prepared, failure_path=args.failure_dump, repeats=args.repeats, backend=backend,
                              configuration=report.copy(),
                              progress=lambda rec: print(json.dumps({'iteration': rec['iteration'], 'failed': rec['failed']}), flush=True)))
            report['loaded_blas_libraries'] = support.loaded_blas_libraries()
        report['source_sha256_after'] = {name: support.file_hash(root / name) for name in sources}
        report['source_files_unchanged'] = report['source_sha256_after'] == inventory
        if not report['source_files_unchanged']:
            report['status'] = 'SOURCE_CHANGED'
    except BaseException as error:
        report.update(status='ERROR', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        temporary = args.report.with_name(args.report.name + '.update')
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
