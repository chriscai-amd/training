#!/usr/bin/env python3
"""CPU audit of a pristine input-stage output-gradient GEMM capture.

Reads the complete trusted local artifact and all BF16 logical/backing bytes.
Does not run the captured operation or initialize a GPU.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(64 << 20):
            digest.update(block)
    return {'path': str(Path(path).resolve()), 'bytes': Path(path).stat().st_size,
            'sha256': digest.hexdigest()}


def bf16_value(bits):
    return struct.unpack('<f', struct.pack('<I', int(bits) << 16))[0]


def row_summary(mask, max_ranges=100):
    import numpy as np
    rows = np.flatnonzero(mask)
    if not rows.size:
        return {'count': 0, 'range': None, 'contiguous_ranges': [], 'ranges_truncated': False}
    starts = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1]
    ends = np.r_[starts[1:] - 1, rows.size - 1]
    ranges = [[int(rows[a]), int(rows[b])] for a, b in zip(starts[:max_ranges], ends[:max_ranges])]
    return {'count': int(rows.size), 'range': [int(rows[0]), int(rows[-1])],
            'contiguous_range_count': int(starts.size), 'contiguous_ranges': ranges,
            'ranges_truncated': bool(starts.size > max_ranges)}


def scan_bf16(tensor, threshold=1e6, chunk_rows=8192):
    import numpy as np
    import torch
    assert tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16
    assert tensor.ndim == 2 and tensor.is_contiguous() and tensor.storage_offset() == 0
    assert tensor.untyped_storage().nbytes() == tensor.numel() * 2
    assert chunk_rows > 0 and threshold > 0
    array = tensor.view(torch.uint16).numpy()
    histogram = np.zeros(65536, dtype=np.int64)
    digest = hashlib.sha256()
    row_masks = {name: np.zeros(tensor.shape[0], dtype=bool)
                 for name in ('nonfinite', 'finite_extreme', 'finite_gt_1e20', 'finite_gt_1e30', 'any_anomaly')}
    features = {name: np.zeros(tensor.shape[1], dtype=np.int64) for name in row_masks}
    counts = Counter()
    for start in range(0, tensor.shape[0], chunk_rows):
        block = array[start:start + chunk_rows]
        digest.update(memoryview(block).cast('B'))
        histogram += np.bincount(block.reshape(-1), minlength=65536)
        absbits = block & 0x7fff
        finite = absbits < 0x7f80
        values = (absbits.astype(np.uint32) << 16).view(np.float32)
        masks = {'nonfinite': ~finite,
                 'finite_extreme': finite & (values > threshold),
                 'finite_gt_1e20': finite & (values > 1e20),
                 'finite_gt_1e30': finite & (values > 1e30)}
        masks['any_anomaly'] = masks['nonfinite'] | masks['finite_extreme']
        for name, mask in masks.items():
            counts[name] += int(np.count_nonzero(mask))
            row_masks[name][start:start + block.shape[0]] = mask.any(axis=1)
            features[name] += mask.sum(axis=0)
    finite_indices = np.r_[np.arange(0x7f80), np.arange(0x8000, 0xff80)]
    present_finite = finite_indices[histogram[finite_indices] > 0]
    max_bits = int(np.max(present_finite & 0x7fff))
    max_abs = bf16_value(max_bits)
    classes = {
        'positive_zero': int(histogram[0]), 'negative_zero': int(histogram[0x8000]),
        'positive_infinity': int(histogram[0x7f80]), 'negative_infinity': int(histogram[0xff80]),
        'positive_nan': int(histogram[0x7f81:0x8000].sum()),
        'negative_nan': int(histogram[0xff81:0x10000].sum()),
        'quiet_nan': int(histogram[0x7fc0:0x8000].sum() + histogram[0xffc0:0x10000].sum()),
        'signaling_nan': int(histogram[0x7f81:0x7fc0].sum() + histogram[0xff81:0xffc0].sum()),
        'subnormal': int(histogram[1:0x80].sum() + histogram[0x8001:0x8080].sum()),
        'normal': int(histogram[0x80:0x7f80].sum() + histogram[0x8080:0xff80].sum()),
    }
    assert int(histogram.sum()) == tensor.numel()
    assert counts['nonfinite'] == sum(classes[name] for name in
        ('positive_infinity', 'negative_infinity', 'positive_nan', 'negative_nan'))
    finite_values = [bf16_value(bits) for bits in present_finite]
    top = sorted((int(bits) for bits in present_finite), key=lambda bits: bits & 0x7fff, reverse=True)[:32]
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'storage_offset': tensor.storage_offset(),
        'dtype': str(tensor.dtype), 'numel': tensor.numel(), 'complete_backing_bytes': tensor.untyped_storage().nbytes(),
        'logical_and_complete_backing_sha256': digest.hexdigest(), 'all_backing_bytes_are_logical': True,
        'full_BF16_histogram_sha256': hashlib.sha256(histogram.astype('<i8').tobytes()).hexdigest(),
        'full_BF16_histogram_total': int(histogram.sum()), 'bit_classes': classes, 'counts': dict(counts),
        'max_abs_finite': max_abs, 'min_finite': min(finite_values), 'max_finite': max(finite_values),
        'rows': {name: row_summary(mask) for name, mask in row_masks.items()},
        'feature_counts': {name: value.tolist() for name, value in features.items()},
        'nonfinite_bit_patterns': [{'bits': f'0x{bits:04x}', 'count': int(histogram[bits])}
            for bits in [*range(0x7f80, 0x8000), *range(0xff80, 0x10000)] if histogram[bits]],
        'largest_finite_bit_patterns': [{'bits': f'0x{bits:04x}', 'value': bf16_value(bits),
                                        'count': int(histogram[bits])} for bits in top],
        'abs_threshold': threshold, 'chunk_rows': chunk_rows}


def audit_capture(run_directory, *, chunk_rows=8192):
    import torch
    torch.set_num_threads(1)
    run_directory = Path(run_directory)
    outcome_path = run_directory / 'outcome.json'
    event_path = run_directory / 'boundaries/boundaries.jsonl'
    outcome = json.loads(outcome_path.read_text())
    raw_path = run_directory / 'boundaries' / Path(outcome['dump_path']).name
    payload = torch.load(raw_path, map_location='cpu', weights_only=False, mmap=True)
    assert payload['format'] == 'training_backward_monitor_v1'
    assert payload['operation'] == 'hstu_output_grad_mm' and payload['stage'] == 'input'
    assert payload['input_snapshot_pristine'] is True and payload['outputs'] is None
    assert payload['args'] == () and set(payload['kwargs']) == {'dout', 'output_weight'}
    event = payload['event']
    assert event['operation_executed'] is False and event['outputs'] is None
    observation = event['input_snapshot_observation']
    assert observation['source'] == 'pristine_pre_call_cpu_snapshot'
    assert observation['operation_started'] is False and observation['input_mutation_during_operation_excluded'] is True
    assert observation['source_devices_synchronized_before_copy'] is True
    assert event['input_snapshot_pristine'] is True and event['device_trigger_persists_in_cpu_snapshot'] is True
    assert outcome['status'] == 'anomaly' and outcome['last_attempted_step'] == payload['attempt']['step']
    assert outcome['completed_steps'] == payload['attempt']['step'] - 1
    matched = {'before': [], 'after': [], 'dump_complete': []}
    with event_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            saved = json.loads(line)
            if (saved.get('session') == payload['session'] and saved.get('attempt') == payload['attempt']
                    and saved.get('call') == event['call'] and saved.get('event') in matched):
                assert saved['layer'] == payload['layer'] and saved['operation'] == payload['operation']
                matched[saved['event']].append((line_number, saved))
    assert len(matched['before']) == len(matched['dump_complete']) == 1 and not matched['after']
    before_line, before = matched['before'][0]
    dump_line, dumped = matched['dump_complete'][0]
    assert before_line < dump_line and before['inputs'] == event['input_device_monitor']
    assert before['input_observation']['operation_started'] is False
    assert all(dumped[key] == value for key, value in event.items())
    assert Path(dumped['dump']).name == raw_path.name and dumped['stage'] == 'input'
    tensors = {}
    for name, tensor in payload['kwargs'].items():
        record = next(item for item in event['inputs'] if item['name'] == name)
        result = scan_bf16(tensor, record['abs_threshold'], chunk_rows)
        for field in ('shape', 'stride', 'storage_offset', 'dtype', 'numel', 'max_abs_finite'):
            assert result[field] == record[field], (name, field)
        assert result['counts']['nonfinite'] == record['nonfinite_count']
        assert result['counts']['finite_extreme'] == record['extreme_count']
        assert result['rows']['any_anomaly']['range'] == record['affected_row_range']
        result['full_scan_matches_saved_CPU_summary'] = True
        tensors[name] = result
    dout, weight = payload['kwargs']['dout'], payload['kwargs']['output_weight']
    assert dout.shape[1] == weight.shape[1]
    assert dout.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr()
    assert sum(t['complete_backing_bytes'] for t in tensors.values()) == event['capture_storage_bytes'] == event['input_bytes']
    assert event['output_bytes'] == 0 and not torch.cuda.is_initialized()
    return {'format': 'pristine_training_output_MM_input_CPU_audit_v1',
        'artifacts': {'capture': file_identity(raw_path), 'events': file_identity(event_path),
                      'outcome': file_identity(outcome_path), 'context': file_identity(run_directory / 'context.json')},
        'identity': {key: payload[key] for key in ('session', 'attempt', 'operation', 'layer', 'stage', 'probe_mode')},
        'call': event['call'], 'before_event_line': before_line, 'dump_complete_event_line': dump_line,
        'matching_before_and_dump_complete': True, 'matching_after_events': 0,
        'input_snapshot_pristine': True, 'operation_executed': False, 'outputs_saved': False,
        'input_snapshot_observation': observation, 'execution_controls': payload['execution_controls'],
        'tensors': tensors, 'gpu_initialized': torch.cuda.is_initialized(),
        'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': [
            'Every logical and backing BF16 input byte is scanned; each contiguous tensor covers its complete independent storage.',
            'Payload, terminal outcome, matching before event and dump_complete event establish this input-stage capture identity.',
            'The input capture precedes this selected operation; no output or after event exists for the captured call.',
            'Bad incoming dout establishes propagation into this unexecuted layer-1 GEMM, not the upstream producing component.',
            'This does not identify the earlier arithmetic step, validate all upstream input bytes, or establish instruction/hardware provenance.',
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--chunk-rows', type=int, default=8192)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('report must be a new file')
    result = audit_capture(args.run_directory, chunk_rows=args.chunk_rows)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'report': str(args.report), 'capture': result['artifacts']['capture'],
        'identity': result['identity'], 'input_snapshot_pristine': True, 'operation_executed': False,
        'tensor_summary': {name: {key: record[key] for key in
            ('shape', 'numel', 'bit_classes', 'counts', 'max_abs_finite', 'rows', 'nonfinite_bit_patterns')}
            for name, record in result['tensors'].items()}}, indent=2))


if __name__ == '__main__':
    main()
