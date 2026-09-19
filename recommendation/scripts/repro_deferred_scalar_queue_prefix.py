#!/usr/bin/env python3
"""Prefix the retained final-dense queue with labeled synthetic finite pairs.

This tests batch length and final-dense row offset. It does not recreate the
original preceding training scans, their tensor sizes, streams or allocation
history. The original retained bundle and native provenance remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch
import repro_deferred_scalar_queue as base

FORMAT = 'retained_final_dense_synthetic_prefix_control_v1'
LIMITS = [base.LIMITS[0], *base.LIMITS[2:],
    'The synthetic prefix is an intervention: separate FP64 two-element inputs, with unique exact finite endpoints.',
    'It reproduces the selected row offset and total endpoint count only; original prefix producers, reductions, values and allocation history are absent.',
    'There are two standalone enqueue scans, synthetic prefix then actual final_dense; original native enqueue67 remains provenance.',
    'Per-scan oracle checks use local row indices; reports include global batch offsets and retain the unmodified native replay scans.',
]


def inventory():
    return dict(base.inventory(), **{'scripts/repro_deferred_scalar_queue_prefix.py': base.support.file_hash(__file__)})


def prepare_control(bundle, prefix_rows, *, max_bytes=256 << 20):
    bundle = base.prepare(bundle, max_bytes=max_bytes)
    if type(prefix_rows) is not int or not 1 <= prefix_rows <= 1000:
        raise ValueError('Synthetic prefix rows must be an integer1..1000')
    # Separate full stores and exactly representable values, without RNG or GPU work.
    prefix = {'synthetic_prefix': {f'row_{i:04d}': torch.tensor([-(i + 1) - 0.25, (i + 1) + 0.5],
                                          dtype=torch.float64) for i in range(prefix_rows)}}
    names, reference = base.cpu_reference(prefix, bundle['chunk_bytes'])
    specs = {n: {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
                 'storage_offset': t.storage_offset(), 'numel': t.numel(), 'requires_grad': False}
             for n, t in base._tensors(prefix)}
    prefix_contract = {'row_names': names, 'reference_endpoints': reference, 'source_specs': specs,
                       'abs_threshold': bundle['abs_threshold'],
                       'enqueue_metadata': {'kind': 'synthetic_endpoint_prefix', 'stage': 'finite_pair_prefix_control'}}
    inputs = {'prefix': prefix, 'final_dense': bundle['inputs']}
    manifest = base.support.manifest(inputs)
    if manifest['storage_bytes'] > max_bytes:
        raise ValueError('Original inputs plus synthetic prefix exceed byte budget')
    return {'inputs': inputs, 'input_manifest': manifest, 'contracts': [prefix_contract, bundle],
            'intervention': {'label': FORMAT, 'synthetic_prefix_rows': prefix_rows,
                'synthetic_prefix_dtype': 'torch.float64', 'synthetic_prefix_shape': [2],
                'synthetic_prefix_endpoint_formula': 'row i = [-(i+1)-0.25, (i+1)+0.5]',
                'synthetic_prefix_full_stores': prefix_rows, 'final_dense_row_start': prefix_rows,
                'batch_rows': prefix_rows + len(bundle['row_names']),
                'matches_original_final_dense_row_start': prefix_rows == bundle['native_batch_position']['row_start'],
                'matches_original_batch_rows': prefix_rows + len(bundle['row_names']) == bundle['native_batch_position']['batch_rows']}}


def replay_once(inputs, control):
    contracts = control['contracts']
    bundle = contracts[1]
    for name, tensor in base._tensors(inputs['final_dense']):
        tensor.requires_grad_(bundle['source_specs'][name]['requires_grad'])
    queue = base.frozen.DeferredScalarQueue(chunk_bytes=bundle['chunk_bytes'], abs_threshold=bundle['abs_threshold'])
    queued = [queue.enqueue(inputs[key], contract['enqueue_metadata'])
              for key, contract in zip(('prefix', 'final_dense'), contracts)]
    pending = list(queue.pending.values())
    if len(pending) != 1:
        raise ValueError('One replay device required')
    endpoints = [endpoint for _, endpoint in pending[0]]
    names = [entry['name'] for entry, _ in pending[0]]
    expected_names = [name for contract in contracts for name in contract['row_names']]
    if names != expected_names:
        raise ValueError('Prefix/final-dense pending endpoint order mismatch')
    reads = []
    original = base.frozen._read_endpoint_batch
    def reader(batch):
        rows = original(batch)
        reads.append({'batch': batch, 'rows': copy.deepcopy(rows)})
        return rows
    with patch.object(base.frozen, '_read_endpoint_batch', reader):
        resolved = queue.flush()
    if len(reads) != 1:
        raise ValueError('Expected one combined prefix/final-dense batch readback')
    devices = {name: str(tensor.device) for key in ('prefix', 'final_dense')
               for name, tensor in base._tensors(inputs[key])}
    return {'endpoints': endpoints, 'batch': reads[0]['batch']}, {
        'row_names': names, 'rows': reads[0]['rows'], 'queued_scans': queued, 'resolved_scans': resolved,
        'replay_source_devices': devices, 'reader_calls': len(reads), 'frozen_pending_cleared': not bool(queue.pending)}


def analyze_observation(output, metadata, control):
    expected_names = [name for contract in control['contracts'] for name in contract['row_names']]
    count = len(expected_names)
    if (metadata['row_names'] != expected_names or len(output['endpoints']) != count
            or list(output['batch'].shape) != [count, 2] or len(metadata['rows']) != count
            or len(metadata['queued_scans']) != 2 or len(metadata['resolved_scans']) != 2):
        raise ValueError('Combined prefix/final-dense observation structure mismatch')
    checks, errors, cursor = [], {}, 0
    for scan_index, contract in enumerate(control['contracts']):
        size = len(contract['row_names'])
        queued = copy.deepcopy(metadata['queued_scans'][scan_index])
        resolved = copy.deepcopy(metadata['resolved_scans'][scan_index])
        index_ok = queued['enqueue_index'] == resolved['enqueue_index'] == scan_index
        # The reusable single-scan oracle expects local index0. Raw metadata is
        # kept intact in the observation and artifact; validate actual indices above.
        queued['enqueue_index'] = resolved['enqueue_index'] = 0
        local_metadata = dict(metadata, row_names=contract['row_names'], rows=metadata['rows'][cursor:cursor + size],
                              queued_scan=queued, resolved_scans=[resolved])
        local_output = {'endpoints': output['endpoints'][cursor:cursor + size],
                        'batch': output['batch'][cursor:cursor + size]}
        check = base.analyze_observation(local_output, local_metadata, contract)
        if not index_ok:
            check['errors']['metadata_disagreement'].append('actual_combined_enqueue_index')
            check['failed'] = True
        checks.append({'scan_index': scan_index, 'global_row_start': cursor, 'check': check})
        for kind, items in check['errors'].items():
            errors.setdefault(kind, []).extend((cursor + item) if isinstance(item, int)
                                               else f'scan{scan_index}:{item}' for item in items)
        cursor += size
    return {'failed': any(errors.values()), 'row_count': count, 'errors': errors, 'scan_checks': checks}


def run(control, *, failure_path, repeats=100, device='cuda:0', max_bytes=256 << 20,
        copy_chunk_bytes=64 << 20, function=None, progress=None):
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError('Repeats must be an integer1..1000')
    base.support.new_path(failure_path)
    if base.support.manifest(control['inputs']) != control['input_manifest']:
        raise ValueError('Control input bytes/layout/aliases changed before restore')
    inputs, _ = base.restore_raw_tree(control['inputs'], device=device, chunk_bytes=copy_chunk_bytes, max_bytes=max_bytes)
    restored, _ = base._cpu_copy_tree(inputs, copy_chunk_bytes, max_bytes)
    if base.support.manifest(restored) != control['input_manifest']:
        raise ValueError('Initial combined input restore changed bytes/layout/aliases')
    del restored
    versions = {name: tensor._version for name, tensor in base._tensors(inputs)}
    function = replay_once if function is None else function
    records = []
    for iteration in range(1, repeats + 1):
        output, metadata = function(inputs, control)
        observed, _ = base._cpu_copy_tree(output, copy_chunk_bytes, max_bytes)
        check = analyze_observation(observed, metadata, control)
        changed = [name for name, tensor in base._tensors(inputs) if tensor._version != versions[name]]
        record = {'iteration': iteration, 'check': check, 'changed_input_versions': changed}
        failed = check['failed'] or bool(changed)
        if failed or iteration == repeats:
            current, _ = base._cpu_copy_tree({'inputs': inputs, 'outputs': output}, copy_chunk_bytes, max_bytes)
            record['full_input_bytes_unchanged'] = base.support.manifest(current['inputs']) == control['input_manifest']
            record['output_bytes_stable_during_capture'] = base.support.manifest(observed) == base.support.manifest(current['outputs'])
            record['saved_output_check'] = analyze_observation(current['outputs'], metadata, control)
            failed |= not record['full_input_bytes_unchanged'] or not record['output_bytes_stable_during_capture']
        record['failed'] = failed
        records.append(record)
        if failed:
            status = 'FAIL' if record['output_bytes_stable_during_capture'] else 'CAPTURE_INTEGRITY_FAILURE'
            bundle = control['contracts'][1]
            artifact = {'format': FORMAT, 'status': status, 'iteration': iteration, 'record': record,
                'pristine_inputs': control['inputs'], 'input_manifest': control['input_manifest'],
                'intervention': control['intervention'], 'first_cpu_outputs': observed,
                'native_metadata_and_rows': metadata, 'current_at_failure': current,
                'reference_endpoints_by_scan': [c['reference_endpoints'] for c in control['contracts']],
                'original_source_scan': bundle['source_scan'], 'original_native_batch_position': bundle['native_batch_position'],
                'bundle_provenance': bundle['provenance'], 'source_sha256': inventory(), 'limits': LIMITS}
            base.support.atomic_save(failure_path, lambda f: torch.save(artifact, f))
            if progress:
                progress(record)
            return {'status': status, 'iterations_completed': iteration, 'records': records,
                    'failure_capture': {'path': str(failure_path), 'sha256': base.support.file_hash(failure_path),
                                        'bytes': Path(failure_path).stat().st_size}}
        if progress:
            progress(record)
        del output, observed
    return {'status': 'PASS', 'iterations_completed': repeats, 'records': records,
            'final_input_bytes_unchanged': records[-1]['full_input_bytes_unchanged']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--failure-dump', type=Path)
    p.add_argument('--prefix-endpoint-rows', type=int, default=229)
    p.add_argument('--max-bytes', type=int, default=256 << 20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args(argv)
    if a.max_bytes <= 0 or a.cpu_threads < 1 or not 1 <= a.repeats <= 1000 or not 1 <= a.prefix_endpoint_rows <= 1000:
        p.error('Positive byte/thread budgets; repeats and prefix rows1..1000 required')
    if a.gpu:
        if not a.failure_dump:
            p.error('--failure-dump required for GPU execution')
        for key in base.support.MANDATORY_GPU_ENVIRONMENT:
            if os.environ.get(key) != '0':
                p.error(key + '=0 required before GPU initialization')
    paths = [v.resolve() for v in (a.bundle, a.report, a.failure_dump) if v]
    if len(paths) != len(set(paths)):
        p.error('Input and output paths must be distinct')
    base.support.new_path(a.report)
    if a.failure_dump:
        base.support.new_path(a.failure_dump)
    torch.set_num_threads(a.cpu_threads)
    start = time.monotonic()
    bundle = torch.load(a.bundle, map_location='cpu', weights_only=False)
    control = prepare_control(bundle, a.prefix_endpoint_rows, max_bytes=a.max_bytes)
    report = {'format': FORMAT, 'status': 'CPU_PREPARED',
              'bundle': {'path': str(a.bundle), 'sha256': base.support.file_hash(a.bundle)},
              'intervention': control['intervention'], 'input_manifest': control['input_manifest'],
              'original_source_scan': bundle['source_scan'], 'original_native_batch_position': bundle['native_batch_position'],
              'bundle_provenance': bundle['provenance'], 'source_sha256': inventory(), 'limits': LIMITS}
    if a.gpu:
        report.update(run(control, failure_path=a.failure_dump, repeats=a.repeats, max_bytes=a.max_bytes,
                          progress=lambda r: print(json.dumps(base.clean_json(r)), flush=True)))
    report.update(seconds=time.monotonic() - start,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  runtime_versions={'torch': str(torch.__version__), 'hip': torch.version.hip},
                  gpu_initialized=torch.cuda.is_initialized())
    base.support.save_json(a.report, base.clean_json(report))
    print(json.dumps({'status': report['status'], 'report': str(a.report)}))
    return int(report['status'] in ('FAIL', 'CAPTURE_INTEGRITY_FAILURE'))


if __name__ == '__main__':
    raise SystemExit(main())
