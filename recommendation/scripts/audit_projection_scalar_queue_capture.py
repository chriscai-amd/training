#!/usr/bin/env python3
"""Audit saved scalar-queue stages and bounded final-dense CPU extrema.

Uses a CPU mmap load and reads only the tiny retained endpoint/batch tensors
and final_dense logical views. No producer, queue, native object or GPU kernel
is executed. A supplied full-payload certificate links whole-file identity;
this auditor does not rescan the remaining multi-GiB backing stores.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time
from unittest import mock

import torch

FORMAT = 'actual_deferred_scalar_queue_stages_v1'
FINAL_STAGE = 'after_backward_return_all_dense'


def file_record(path):
    data = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: clean_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(child) for child in value]
    return value


def tensors(value, path=''):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f'{path}/{key}' if path else str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensors(child, f'{path}/{index}' if path else str(index))


def fp64_bits(value):
    return struct.pack('>d', float(value)).hex()


def tensor_pair(tensor):
    if (tensor.device.type != 'cpu' or tensor.dtype != torch.float64
            or tuple(tensor.shape) != (2,) or tensor.is_conj() or tensor.is_neg()):
        raise ValueError('Expected a plain CPU FP64 endpoint pair')
    # Read the stored bits before conversion to Python floats can quiet a NaN.
    raw = [f'{value & ((1 << 64) - 1):016x}' for value in tensor.view(torch.int64).tolist()]
    return tensor.tolist(), raw


def python_pair(value, *, summary=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Expected exactly two scalar values')
    if any(not isinstance(x, (int, float)) and not (summary and x in ('nan', 'inf', '-inf')) for x in value):
        raise ValueError('Invalid scalar representation')
    numbers = list(map(float, value))
    # A textual NaN summary no longer encodes a sign/payload to compare.
    raw = [None if summary and isinstance(x, str) and math.isnan(n) else fp64_bits(n)
           for x, n in zip(value, numbers)]
    return numbers, raw


def comparison(left, right):
    lv, lb = left
    rv, rb = right
    numerical = all(a == b or math.isnan(a) and math.isnan(b) for a, b in zip(lv, rv))
    comparable = all(x is not None for x in (*lb, *rb))
    return {'left': lv, 'right': rv, 'left_bits': lb, 'right_bits': rb,
            'nan_aware_equal': numerical,
            'raw_FP64_equal': lb == rb if comparable else None}


def audit_payload(payload, *, max_dense_bytes=256 << 20, max_rows=4096):
    diagnostic = payload['scalar_queue_diagnostics']
    if diagnostic.get('format') != FORMAT or diagnostic.get('flush_complete') is not True:
        raise ValueError('Complete actual scalar-queue stages required')
    native = diagnostic['native_resolved_scans']
    final = payload['metadata']['deferred_scans']
    if not native or len(native) != len(final) or len(native) > max_rows:
        raise ValueError('Invalid native/final scan coverage')
    for i, (before, after) in enumerate(zip(native, final)):
        if (before.get('enqueue_index') != i or after.get('enqueue_index') != i
                or len(before['summaries']) != len(after['summaries'])
                or before.get('values_resolved_on_host') is not True
                or after.get('values_resolved_on_host') is not True):
            raise ValueError('Invalid scan order or summary coverage')
    groups = diagnostic['groups']
    if diagnostic['reader_calls'] != len(groups) or len({g['source_device'] for g in groups}) != len(groups):
        raise ValueError('Invalid device/readback coverage')
    expected_by_device = {}
    for i, scan in enumerate(native):
        for j, entry in enumerate(scan['summaries']):
            if entry['scanned'] and entry['numel'] > 0:
                expected_by_device.setdefault(entry['source_device'], []).append((i, j, entry))
    if list(expected_by_device) != [g['source_device'] for g in groups]:
        raise ValueError('Missing or reordered device groups')
    if sum(map(len, expected_by_device.values())) > max_rows:
        raise ValueError('Scalar row budget exceeded')
    stages = ('endpoint_to_batch', 'batch_to_returned', 'returned_to_native_summary',
              'returned_to_final_summary', 'final_dense_CPU_to_endpoint',
              'final_dense_CPU_to_batch', 'final_dense_CPU_to_returned',
              'final_dense_CPU_to_native_summary')
    checks = {name: [] for name in stages}
    violations, metadata_errors, all_rows, lookup = [], [], [], {}

    def add_check(stage, left, right, mapping):
        checks[stage].append({'mapping': mapping, **comparison(left, right)})

    def invariant(stage, pair, mapping):
        low, high = pair[0]
        if not (math.isnan(low) or math.isnan(high)) and low > high:
            violations.append({'stage': stage, 'mapping': mapping, 'pair': pair[0]})

    for group in groups:
        device = group['source_device']
        expected = expected_by_device[device]
        size = len(expected)
        if any(len(group[key]) != size for key in ('row_mapping', 'endpoint_tensors', 'returned_rows', 'queued_entries')) or group['row_count'] != size:
            raise ValueError('Incomplete device row coverage')
        batch = group['batch_tensor']
        if batch.device.type != 'cpu' or batch.dtype != torch.float64 or tuple(batch.shape) != (size, 2):
            raise ValueError('Invalid retained batch')
        for row, (i, j, entry) in enumerate(expected):
            mapping = {'batch_row_index': row, 'enqueue_index': i, 'summary_index': j, 'name': entry['name']}
            if group['row_mapping'][row] != mapping:
                raise ValueError('Retained row mapping differs from native scan order')
            mapping = {'source_device': device, **mapping}
            queued = group['queued_entries'][row]
            for key, value in queued.items():
                if key not in entry or entry[key] != value:
                    metadata_errors.append({'mapping': mapping, 'field': 'queued/' + key, 'actual': entry.get(key), 'expected': value})
            fe = final[i]['summaries'][j]
            for key in ('name', 'shape', 'stride', 'dtype', 'source_device', 'storage_offset', 'requires_grad', 'numel', 'scanned', 'stream_id', 'abs_threshold'):
                if entry.get(key) != fe.get(key):
                    metadata_errors.append({'mapping': mapping, 'field': 'final/' + key, 'actual': fe.get(key), 'expected': entry.get(key)})
            pairs = {'endpoint': tensor_pair(group['endpoint_tensors'][row]),
                     'batch': tensor_pair(batch[row]),
                     'returned': python_pair(group['returned_rows'][row]),
                     'native_summary': python_pair([entry['min'], entry['max']], summary=True),
                     'final_summary': python_pair([fe['min'], fe['max']], summary=True)}
            lookup[i, j] = (mapping, pairs)
            all_rows.append({'mapping': mapping, 'stages': {k: {'values': v[0], 'bits': v[1]} for k, v in pairs.items()}})
            for left, right in (('endpoint', 'batch'), ('batch', 'returned'), ('returned', 'native_summary'), ('returned', 'final_summary')):
                add_check(left + '_to_' + right, pairs[left], pairs[right], mapping)
            for stage, pair in pairs.items():
                invariant(stage, pair, mapping)
            values = pairs['returned'][0]
            finite = all(math.isfinite(x) for x in values)
            magnitude = max(map(abs, values)) if finite else None
            derived = {'finite': finite, 'extreme': finite and magnitude > entry['abs_threshold'], 'max_abs_finite': magnitude}
            for label, summary in (('native', entry), ('final', fe)):
                for key, expected_value in derived.items():
                    if summary.get(key) != expected_value:
                        metadata_errors.append({'mapping': mapping, 'field': label + '/' + key, 'actual': summary.get(key), 'expected': expected_value})
    for i, (nscan, fscan) in enumerate(zip(native, final)):
        for label, scan in (('native', nscan), ('final', fscan)):
            if label == 'final' and scan.get('expected_zero'):
                wanted = [e['name'] for e in scan['summaries'] if not e['finite'] or e['max'] != 0]
            else:
                wanted = [e['name'] for e in scan['summaries'] if not e['finite'] or e['extreme']]
            if scan['flagged_names'] != wanted:
                metadata_errors.append({'enqueue_index': i, 'field': label + '/flagged_names', 'actual': scan['flagged_names'], 'expected': wanted})
    selected = [(i, s) for i, s in enumerate(native) if s.get('stage') == FINAL_STAGE]
    if len(selected) != 1:
        raise ValueError('Exactly one final-dense scan required')
    i, scan = selected[0]
    named = list(tensors(payload['final_dense']))
    if not named or [name for name, _ in named] != [e['name'] for e in scan['summaries']]:
        raise ValueError('Final-dense tensor names/order differ from source scan')
    backing = {t.untyped_storage()._cdata: t.untyped_storage().nbytes() for _, t in named}
    if sum(backing.values()) > max_dense_bytes:
        raise ValueError('Final-dense backing byte budget exceeded')
    references = []
    for j, ((name, tensor), entry) in enumerate(zip(named, scan['summaries'])):
        if (tensor.device.type != 'cpu' or tensor.layout != torch.strided or not tensor.is_floating_point()
                or not tensor.numel() or tensor.is_conj() or tensor.is_neg()):
            raise ValueError('Final-dense reference requires plain nonempty CPU floating views')
        actual = dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                      storage_offset=tensor.storage_offset(), numel=tensor.numel())
        if any(entry.get(key) != value for key, value in actual.items()):
            raise ValueError('Final-dense layout differs from retained native metadata')
        # Separate full-view CPU reductions: independent of the frozen queue's
        # chunked aminmax, endpoint stack, transfer and summary implementation.
        ref = python_pair([float(torch.amin(tensor)), float(torch.amax(tensor))])
        mapping, pairs = lookup[i, j]
        references.append({'mapping': mapping, 'values': ref[0], 'bits': ref[1]})
        invariant('final_dense_CPU_reference', ref, mapping)
        for stage in ('endpoint', 'batch', 'returned', 'native_summary'):
            add_check('final_dense_CPU_to_' + stage, ref, pairs[stage], mapping)
    counts = {stage: {'rows': len(entries),
                     'nan_aware_mismatches': sum(not e['nan_aware_equal'] for e in entries),
                     'raw_FP64_comparable_rows': sum(e['raw_FP64_equal'] is not None for e in entries),
                     'raw_FP64_mismatches': sum(e['raw_FP64_equal'] is False for e in entries)}
              for stage, entries in checks.items()}
    mismatches = {stage: [e for e in entries if not e['nan_aware_equal'] or e['raw_FP64_equal'] is False]
                  for stage, entries in checks.items()}
    return {'format': 'actual_scalar_queue_capture_CPU_audit_v1',
            'status': 'PASS' if not any(mismatches.values()) and not violations and not metadata_errors else 'MISMATCH',
            'attempt': payload.get('attempt'), 'capture_variant': payload.get('capture_variant'),
            'scan_count': len(native), 'device_groups': list(expected_by_device),
            'counts': counts, 'mismatches': mismatches, 'min_above_max': violations,
            'metadata_errors': metadata_errors, 'final_dense_backings': len(backing),
            'final_dense_storage_bytes': sum(backing.values()), 'final_dense_CPU_references': references,
            'all_stage_rows': all_rows,
            'limits': ['Only retained scalar stages and final_dense logical views are read; remaining raw payload tensors are not rescanned.',
                       'Textual NaN summaries do not preserve raw NaN payload/sign; those raw comparisons are explicitly unavailable.',
                       'CPU reductions may select different signed-zero or NaN payload representatives; raw differences are reported separately from NaN-aware numerical equality.',
                       'The saved endpoint/batch tensors are observations after the original flush, not runtime instruction/register traces.',
                       'Agreement in this capture does not certify other steps or establish a training fix.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('payload', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--full-payload-certificate', type=Path)
    parser.add_argument('--max-dense-mib', type=int, default=256)
    parser.add_argument('--max-rows', type=int, default=4096)
    args = parser.parse_args(argv)
    if args.report.exists() or args.max_dense_mib <= 0 or args.max_rows <= 0:
        parser.error('Require a fresh report path and positive budgets')
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'LD_PRELOAD'):
        os.environ[key] = ''
    if torch.cuda.is_initialized():
        raise RuntimeError('GPU already initialized')
    torch.set_num_threads(4)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    started = time.monotonic()
    before = args.payload.stat()
    with ExitStack() as stack:
        forbidden = AssertionError('CPU scalar audit forbids GPU/compile/native loading')
        for obj, name in ((torch.cuda, '_lazy_init'), (torch, 'compile'), (torch.jit, 'script'),
                          (torch.jit, 'trace'), (torch.ops, 'load_library'), (torch.classes, 'load_library')):
            stack.enter_context(mock.patch.object(obj, name, side_effect=forbidden))
        payload = torch.load(args.payload, map_location='cpu', weights_only=False, mmap=True)
        result = audit_payload(payload, max_dense_bytes=args.max_dense_mib << 20, max_rows=args.max_rows)
    after = args.payload.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert random.getstate() == python_rng and torch.equal(torch_rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    result.update(auditor=file_record(Path(__file__).resolve()),
                  payload={'path': str(args.payload), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns},
                  seconds=time.monotonic() - started,
                  guards={'GPU_uninitialized': True, 'Torch_RNG_unchanged': True, 'Python_RNG_unchanged': True,
                          'payload_stat_unchanged': True, 'CPU_mmap_bounded_reads': True})
    if args.full_payload_certificate:
        certificate = json.loads(args.full_payload_certificate.read_text())
        if Path(certificate['payload']).name != args.payload.name or certificate['bytes'] != before.st_size:
            raise ValueError('Full-payload certificate path/size differs')
        result['full_payload_certificate'] = file_record(args.full_payload_certificate)
        result['payload']['sha256_from_full_payload_certificate'] = certificate['sha256']
        result['payload']['whole_hash_recomputed_in_this_audit'] = False
    args.report.write_text(json.dumps(clean_json(result), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': file_record(args.report), 'status': result['status'],
                      'counts': result['counts'], 'min_above_max': len(result['min_above_max']),
                      'metadata_errors': len(result['metadata_errors'])}, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
