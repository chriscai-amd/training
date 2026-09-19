#!/usr/bin/env python3
"""Replay the frozen DeferredScalarQueue over retained final dense views.

CPU extraction preserves every backing byte, view layout, alias and row order.
GPU execution requires --gpu. The frozen enqueue/flush methods run unchanged;
a scoped reader wrapper retains the actual stacked batch and returned rows.
Existing endpoints, batch, rows and inputs are saved at the first disagreement.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch
import nan_backward_training_deferred as frozen
import repro_gradient_comparison as support
from nan_backward_boundaries import _cpu_copy_tree, _logical_chunks, _tensors
from replay_backward_boundary import restore_raw_tree

BUNDLE_FORMAT = 'retained_final_dense_scalar_queue_bundle_v1'
FAILURE_FORMAT = 'retained_final_dense_scalar_queue_failure_v1'
STAGE = 'after_backward_return_all_dense'
RESOLVED_SCAN_KEYS = ('enqueue_index', 'host_enqueue_unix_time', 'observation',
                      'values_resolved_on_host', 'summaries', 'flagged_names')
SOURCE_FILES = tuple(dict.fromkeys((*support.SOURCE_FILES,
    'scripts/repro_deferred_scalar_queue.py', 'scripts/nan_backward_training_deferred.py',
    'scripts/nan_backward_training_batched.py', 'scripts/nan_backward_training_all_layers.py',
    'scripts/nan_backward_training_extended.py', 'scripts/nan_backward_training_probe.py',
    'scripts/nan_backward_training_upstream.py')))
LIMITS = [
    'One retained final-dense scan is replayed in its original tensor order; earlier training scans and producers are absent.',
    'The standalone queue uses local enqueue index0; source enqueue index and original native summaries remain provenance.',
    'Endpoints and the stacked batch stay alive through CPU capture, extending their original lifetimes.',
    'A bad retained endpoint identifies the per-tensor endpoint path, including reduction, endpoint stack and FP64 cast; its substeps are not separately retained.',
    'Full input bytes are checked after restore, at failure and at successful termination. Other attempts use version checks.',
    'Post-flush CPU copies do not prove intermediate bytes were unchanged earlier; transfer or later mutation remains possible.',
]


def inventory():
    root = Path(__file__).resolve().parents[1]
    return {p: support.file_hash(root / p) for p in SOURCE_FILES}


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def cpu_reference(tree, chunk_bytes):
    """Independent CPU extrema, using separate amin/amax and no GPU queue."""
    names, endpoints = [], []
    for name, tensor in _tensors(tree):
        if (tensor.device.type != 'cpu' or tensor.layout != torch.strided or tensor.is_quantized
                or not tensor.is_floating_point() or tensor.is_complex() or not tensor.numel()
                or tensor.is_conj() or tensor.is_neg()):
            raise ValueError('Requires nonempty, plain strided CPU floating views')
        minimum, maximum = None, None
        for chunk, _ in _logical_chunks(tensor.detach(), max(1, chunk_bytes // tensor.element_size())):
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError('Every selected logical input must be finite')
            low, high = float(torch.amin(chunk)), float(torch.amax(chunk))
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
        names.append(name)
        endpoints.append((minimum, maximum))
    if not names:
        raise ValueError('No scalar queue inputs')
    return names, torch.tensor(endpoints, dtype=torch.float64)


def scan_metadata(scan):
    return {k: copy.deepcopy(v) for k, v in scan.items() if k not in RESOLVED_SCAN_KEYS}


def source_layout(tree, scan):
    if not isinstance(tree, dict) or list(tree) != ['parameters', 'gradients']:
        raise ValueError('Exact final_dense parameters/gradients insertion order required')
    named = list(_tensors(tree))
    if [name for name, _ in named] != [e['name'] for e in scan['summaries']]:
        raise ValueError('Retained tensor order differs from original native endpoint rows')
    specs = {}
    for (name, tensor), entry in zip(named, scan['summaries']):
        actual = {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
                  'storage_offset': tensor.storage_offset(), 'numel': tensor.numel()}
        if (any(entry.get(k) != v for k, v in actual.items()) or entry.get('scanned') is not True
                or type(entry.get('requires_grad')) is not bool):
            raise ValueError('Retained layout differs from native scan metadata: ' + name)
        specs[name] = dict(actual, requires_grad=entry['requires_grad'])
    return specs


def native_batch_position(scans, source_scan):
    devices = {e['source_device'] for e in source_scan['summaries']}
    if len(devices) != 1:
        raise ValueError('One original final-dense source device required')
    device = next(iter(devices))
    ranges, cursor, selected = [], 0, None
    for ordinal, scan in enumerate(scans):
        if scan['enqueue_index'] != ordinal:
            raise ValueError('Original enqueue order must be complete and consecutive')
        count = sum(e.get('scanned') is True and e['numel'] > 0 and e['source_device'] == device
                    for e in scan['summaries'])
        row_range = {'enqueue_index': ordinal, 'stage': scan.get('stage'), 'kind': scan.get('kind'),
                     'row_start': cursor, 'row_count': count}
        ranges.append(row_range)
        if scan is source_scan:
            selected = row_range
        cursor += count
    if selected is None or selected['row_count'] != len(source_scan['summaries']):
        raise ValueError('Original final-dense batch placement is incomplete')
    return {'source_device': device, 'row_start': selected['row_start'],
            'row_count': selected['row_count'], 'batch_rows': cursor, 'scan_row_ranges': ranges,
            'standalone_row_start': 0, 'standalone_enqueue_index': 0}


def extract(payload, *, provenance, chunk_bytes=64 << 20, max_bytes=256 << 20):
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError('Positive chunk bytes required')
    tree = payload['final_dense']
    original_scans = payload['metadata']['deferred_scans']
    scans = [s for s in original_scans if s.get('stage') == STAGE]
    if len(scans) != 1:
        raise ValueError('Exactly one original final-dense scalar scan required')
    source_scan = scans[0]
    names, reference = cpu_reference(tree, chunk_bytes)
    entries = source_scan['summaries']
    specs = source_layout(tree, source_scan)
    batch_position = native_batch_position(original_scans, source_scan)
    thresholds = {e['abs_threshold'] for e in entries}
    if len(thresholds) != 1 or not math.isfinite(next(iter(thresholds))) or next(iter(thresholds)) <= 0:
        raise ValueError('One positive native threshold required')
    original = support.manifest(tree)
    copied, size = _cpu_copy_tree(tree, chunk_bytes, max_bytes)
    if support.manifest(copied) != original:
        raise ValueError('Extraction changed full bytes, layouts or aliases')
    metadata = scan_metadata(source_scan)
    return {'format': BUNDLE_FORMAT, 'inputs': copied, 'input_manifest': original,
            'row_names': names, 'reference_endpoints': reference, 'source_specs': specs,
            'chunk_bytes': chunk_bytes, 'abs_threshold': next(iter(thresholds)),
            'enqueue_metadata': metadata, 'source_scan': copy.deepcopy(source_scan),
            'native_batch_position': batch_position,
            'storage_bytes': size, 'provenance': dict(provenance, session=payload.get('session'),
                attempt=payload.get('attempt'), source_enqueue_index=source_scan['enqueue_index'],
                capture_source_sha256=payload['metadata'].get('source_sha256'))}


def prepare(bundle, *, max_bytes=256 << 20):
    if bundle.get('format') != BUNDLE_FORMAT:
        raise ValueError('Unexpected scalar-queue bundle format')
    if type(bundle['chunk_bytes']) is not int or bundle['chunk_bytes'] <= 0:
        raise ValueError('Positive chunk bytes required')
    if not math.isfinite(bundle['abs_threshold']) or bundle['abs_threshold'] <= 0:
        raise ValueError('Positive finite threshold required')
    names, refs = cpu_reference(bundle['inputs'], bundle['chunk_bytes'])
    actual = support.manifest(bundle['inputs'])
    if (actual != bundle['input_manifest'] or actual['storage_bytes'] > max_bytes
            or names != bundle['row_names'] or not support._same_raw(refs, bundle['reference_endpoints'])):
        raise ValueError('Full input bytes/layout/order or CPU reference mismatch')
    if (source_layout(bundle['inputs'], bundle['source_scan']) != bundle['source_specs']
            or list(bundle['source_specs']) != names
            or scan_metadata(bundle['source_scan']) != bundle['enqueue_metadata']
            or any(e['abs_threshold'] != bundle['abs_threshold'] for e in bundle['source_scan']['summaries'])):
        raise ValueError('Source row/layout/enqueue metadata mismatch')
    position = bundle['native_batch_position']
    cursor = 0
    for ordinal, item in enumerate(position['scan_row_ranges']):
        if item['enqueue_index'] != ordinal or item['row_start'] != cursor or item['row_count'] < 0:
            raise ValueError('Original batch row range mismatch')
        cursor += item['row_count']
    selected = position['scan_row_ranges'][bundle['source_scan']['enqueue_index']]
    if (position['row_count'] != len(names) or position['row_start'] != selected['row_start']
            or selected['row_count'] != len(names) or position['batch_rows'] != cursor
            or position['standalone_row_start'] != 0 or position['standalone_enqueue_index'] != 0
            or any(e['source_device'] != position['source_device'] for e in bundle['source_scan']['summaries'])):
        raise ValueError('Original/standalone batch row placement mismatch')
    return bundle


def replay_once(inputs, bundle):
    """Use frozen methods; retain only references in the scoped reader hook."""
    for name, tensor in _tensors(inputs):
        tensor.requires_grad_(bundle['source_specs'][name]['requires_grad'])
    queue = frozen.DeferredScalarQueue(chunk_bytes=bundle['chunk_bytes'], abs_threshold=bundle['abs_threshold'])
    queued = queue.enqueue(inputs, bundle['enqueue_metadata'])
    endpoints, entry_names, devices = [], [], []
    for device, group in queue.pending.items():
        devices.append(device)
        for entry, endpoint in group:
            entry_names.append(entry['name'])
            endpoints.append(endpoint)
    if entry_names != bundle['row_names'] or len(devices) != 1:
        raise ValueError('Frozen pending endpoint order/device differs from bundle')
    reads = []
    original_reader = frozen._read_endpoint_batch
    def reader(batch):
        rows = original_reader(batch)
        reads.append({'batch': batch, 'rows': copy.deepcopy(rows)})
        return rows
    with patch.object(frozen, '_read_endpoint_batch', reader):
        scans = queue.flush()
    if len(reads) != 1:
        raise ValueError('Expected exactly one frozen endpoint batch readback')
    return {'endpoints': endpoints, 'batch': reads[0]['batch']}, {
        'row_names': entry_names, 'rows': reads[0]['rows'], 'queued_scan': queued, 'resolved_scans': scans,
        'replay_source_devices': {name: str(tensor.device) for name, tensor in _tensors(inputs)},
        'reader_calls': len(reads), 'frozen_pending_cleared': not bool(queue.pending)}


def _row_equal(left, right):
    return len(left) == len(right) == 2 and all(float(a) == float(b) or
        (math.isnan(float(a)) and math.isnan(float(b))) for a, b in zip(left, right))


def analyze_observation(tensors, metadata, bundle):
    names, references = bundle['row_names'], bundle['reference_endpoints']
    endpoints, batch = tensors['endpoints'], tensors['batch']
    if metadata['row_names'] != names or len(endpoints) != len(names):
        raise ValueError('Retained endpoint row mapping mismatch')
    if batch.device.type != 'cpu' or batch.dtype != torch.float64 or list(batch.shape) != [len(names), 2]:
        raise ValueError('Expected full CPU FP64 [rows,2] batch')
    if len(metadata['rows']) != len(names) or len(metadata['resolved_scans']) != 1:
        raise ValueError('Incomplete native CPU rows/scans')
    resolved = metadata['resolved_scans'][0]
    queued = metadata['queued_scan']
    summaries = resolved['summaries']
    if [e['name'] for e in summaries] != names:
        raise ValueError('Resolved native summaries changed row order')
    metadata_errors = []
    for label, scan, is_resolved in [('queued', queued, False), ('resolved', resolved, True)]:
        if (scan_metadata(scan) != bundle['enqueue_metadata'] or scan['enqueue_index'] != 0
                or scan['values_resolved_on_host'] is not is_resolved
                or scan['observation'] != 'deferred_device_reduction'):
            metadata_errors.append(label + ':scan')
        if [e['name'] for e in scan['summaries']] != names:
            metadata_errors.append(label + ':row_order')
        else:
            for entry in scan['summaries']:
                expected = bundle['source_specs'][entry['name']]
                if (any(entry.get(k) != v for k, v in expected.items()) or entry.get('scanned') is not True
                        or entry.get('abs_threshold') != bundle['abs_threshold']
                        or entry.get('source_device') != metadata['replay_source_devices'][entry['name']]):
                    metadata_errors.append(label + ':' + entry['name'])
    if metadata['reader_calls'] != 1 or metadata['frozen_pending_cleared'] is not True:
        metadata_errors.append('reader_count_or_pending_cleanup')
    bad_endpoints, bad_stack, bad_readback, bad_summary, rows, expected_flagged = [], [], [], [], [], []
    for i, name in enumerate(names):
        endpoint = endpoints[i]
        if endpoint.device.type != 'cpu' or endpoint.dtype != torch.float64 or list(endpoint.shape) != [2]:
            raise ValueError('Expected retained CPU FP64 endpoint pair')
        endpoint_row, batch_row = endpoint.tolist(), batch[i].tolist()
        read_row, reference = metadata['rows'][i], references[i].tolist()
        if not _row_equal(endpoint_row, reference):
            bad_endpoints.append(i)
        if not support._same_raw(endpoint, batch[i]):
            bad_stack.append(i)
        if not _row_equal(read_row, batch_row):
            bad_readback.append(i)
        summary = summaries[i]
        summary_row = [summary['min'], summary['max']]
        # Nonfinite native summaries are strings; compare them to raw returned rows explicitly.
        normalized_rows = [str(v) if isinstance(v, float) and not math.isfinite(v) else v for v in read_row]
        finite = all(math.isfinite(v) for v in read_row)
        magnitude = max(abs(v) for v in read_row) if finite else None
        extreme = finite and magnitude > bundle['abs_threshold']
        if not finite or extreme:
            expected_flagged.append(name)
        if (summary_row != normalized_rows or summary.get('finite') is not finite
                or summary.get('extreme') is not extreme or summary.get('max_abs_finite') != magnitude):
            bad_summary.append(i)
        if i in bad_endpoints or i in bad_stack or i in bad_readback or i in bad_summary:
            rows.append({'index': i, 'name': name, 'reference': reference, 'endpoint': endpoint_row,
                         'batch': batch_row, 'readback': read_row, 'summary': summary_row})
    if resolved.get('flagged_names') != expected_flagged:
        metadata_errors.append('resolved:flagged_names')
    errors = {'per_tensor_endpoint_disagreement': bad_endpoints, 'stack_or_row_mapping_disagreement': bad_stack,
              'readback_disagreement': bad_readback, 'summary_row_mapping_disagreement': bad_summary,
              'metadata_disagreement': metadata_errors}
    return {'failed': any(errors.values()), 'row_count': len(names), 'errors': errors,
            'disagreeing_rows': clean_json(rows), 'reference': 'Exact CPU numerical extrema; signed zeros accepted.'}


def run(bundle, *, failure_path, repeats=100, device='cuda:0', max_bytes=256 << 20,
        copy_chunk_bytes=64 << 20, function=None, progress=None):
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError('Repeats1..1000 required')
    support.new_path(failure_path)
    bundle = prepare(bundle, max_bytes=max_bytes)
    inputs, _ = restore_raw_tree(bundle['inputs'], device=device, chunk_bytes=copy_chunk_bytes, max_bytes=max_bytes)
    restored, _ = _cpu_copy_tree(inputs, copy_chunk_bytes, max_bytes)
    if support.manifest(restored) != bundle['input_manifest']:
        raise ValueError('Initial restored full bytes/layout/aliases differ')
    del restored
    versions = {n: t._version for n, t in _tensors(inputs)}
    records = []
    function = replay_once if function is None else function
    for iteration in range(1, repeats + 1):
        output, metadata = function(inputs, bundle)
        observed, _ = _cpu_copy_tree(output, copy_chunk_bytes, max_bytes)
        check = analyze_observation(observed, metadata, bundle)
        changed = [n for n, t in _tensors(inputs) if t._version != versions[n]]
        record = {'iteration': iteration, 'check': check, 'changed_input_versions': changed}
        failed = check['failed'] or bool(changed)
        if failed or iteration == repeats:
            current, _ = _cpu_copy_tree({'inputs': inputs, 'outputs': output}, copy_chunk_bytes, max_bytes)
            record['full_input_bytes_unchanged'] = support.manifest(current['inputs']) == bundle['input_manifest']
            record['output_bytes_stable_during_capture'] = support.manifest(observed) == support.manifest(current['outputs'])
            record['saved_output_check'] = analyze_observation(current['outputs'], metadata, bundle)
            failed |= not record['full_input_bytes_unchanged'] or not record['output_bytes_stable_during_capture']
        record['failed'] = failed
        records.append(record)
        if failed:
            status = 'FAIL' if record['output_bytes_stable_during_capture'] else 'CAPTURE_INTEGRITY_FAILURE'
            artifact = {'format': FAILURE_FORMAT, 'status': status, 'iteration': iteration, 'record': record,
                        'pristine_inputs': bundle['inputs'], 'reference_endpoints': bundle['reference_endpoints'],
                        'first_cpu_outputs': observed, 'native_metadata_and_rows': metadata,
                        'current_at_failure': current, 'input_manifest': bundle['input_manifest'],
                        'source_scan': bundle['source_scan'], 'bundle_provenance': bundle['provenance'],
                        'native_batch_position': bundle['native_batch_position'],
                        'source_sha256': inventory(), 'limits': LIMITS}
            support.atomic_save(failure_path, lambda f: torch.save(artifact, f))
            if progress:
                progress(record)
            return {'status': status, 'iterations_completed': iteration, 'records': records,
                    'failure_capture': {'path': str(failure_path), 'sha256': support.file_hash(failure_path),
                                        'bytes': Path(failure_path).stat().st_size}}
        if progress:
            progress(record)
        del output, observed
    return {'status': 'PASS', 'iterations_completed': repeats, 'records': records,
            'final_input_bytes_unchanged': records[-1]['full_input_bytes_unchanged']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--extract-payload', type=Path)
    inputs.add_argument('--bundle', type=Path)
    p.add_argument('--bundle-output', type=Path)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--failure-dump', type=Path)
    p.add_argument('--chunk-bytes', type=int, default=64 << 20)
    p.add_argument('--max-bytes', type=int, default=256 << 20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args(argv)
    if a.chunk_bytes <= 0 or a.max_bytes <= 0 or a.cpu_threads < 1 or not 1 <= a.repeats <= 1000:
        p.error('Positive budgets/threads and repeats1..1000 required')
    if a.extract_payload and (not a.bundle_output or a.gpu):
        p.error('Extraction requires --bundle-output and is CPU-only')
    if a.gpu:
        if not a.failure_dump:
            p.error('--failure-dump required for GPU execution')
        for key in support.MANDATORY_GPU_ENVIRONMENT:
            if os.environ.get(key) != '0':
                p.error(key + '=0 required before GPU initialization')
    paths = [v.resolve() for v in (a.extract_payload, a.bundle, a.bundle_output, a.report, a.failure_dump) if v]
    if len(paths) != len(set(paths)):
        p.error('Input and output paths must be distinct')
    support.new_path(a.report)
    if a.failure_dump:
        support.new_path(a.failure_dump)
    torch.set_num_threads(a.cpu_threads)
    start = time.monotonic()
    if a.extract_payload:
        support.new_path(a.bundle_output)
        before = a.extract_payload.stat()
        payload = torch.load(a.extract_payload, mmap=True, map_location='cpu', weights_only=False)
        bundle = extract(payload, provenance={'payload_path': str(a.extract_payload), 'payload_bytes': before.st_size,
                         'payload_mtime_ns': before.st_mtime_ns, 'whole_payload_sha256': 'not_computed'},
                         chunk_bytes=a.chunk_bytes, max_bytes=a.max_bytes)
        after = a.extract_payload.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source payload changed during extraction')
        support.atomic_save(a.bundle_output, lambda f: torch.save(bundle, f))
        prepare(torch.load(a.bundle_output, map_location='cpu', weights_only=False), max_bytes=a.max_bytes)
        report = {'status': 'CPU_BUNDLE_VERIFIED', 'bundle': {'path': str(a.bundle_output),
                  'sha256': support.file_hash(a.bundle_output), 'bytes': a.bundle_output.stat().st_size}}
    else:
        bundle = prepare(torch.load(a.bundle, map_location='cpu', weights_only=False), max_bytes=a.max_bytes)
        report = {'status': 'CPU_PREPARED', 'bundle': {'path': str(a.bundle), 'sha256': support.file_hash(a.bundle)}}
        if a.gpu:
            report.update(run(bundle, failure_path=a.failure_dump, repeats=a.repeats, max_bytes=a.max_bytes,
                              progress=lambda r: print(json.dumps(clean_json(r)), flush=True)))
    report.update(format='deferred_scalar_queue_replay_report_v1', row_names=bundle['row_names'],
                  input_manifest=bundle['input_manifest'], reference_endpoints=bundle['reference_endpoints'].tolist(),
                  source_scan=bundle['source_scan'], bundle_provenance=bundle['provenance'],
                  native_batch_position=bundle['native_batch_position'],
                  source_sha256=inventory(), seconds=time.monotonic() - start, limits=LIMITS,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  runtime_versions={'torch': str(torch.__version__), 'hip': torch.version.hip},
                  gpu_initialized=torch.cuda.is_initialized())
    support.save_json(a.report, clean_json(report))
    print(json.dumps({'status': report['status'], 'report': str(a.report)}))
    return int(report['status'] in ('FAIL', 'CAPTURE_INTEGRITY_FAILURE'))


if __name__ == '__main__':
    raise SystemExit(main())
