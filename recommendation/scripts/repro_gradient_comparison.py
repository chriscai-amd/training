#!/usr/bin/env python3
"""Isolate the retained post-accumulate/final gradient comparison.

CPU extraction preserves full backing stores, view offsets, strides and aliases.
GPU replay is opt-in and runs the frozen comparison operators once per attempt,
retaining == masks, all() results and the scalar chain before any host read.
The first disagreement is saved without rerunning those operators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from nan_backward_boundaries import _cpu_copy_tree, _logical_chunks, _tensors
from replay_backward_boundary import restore_raw_tree

BUNDLE_FORMAT = 'retained_gradient_comparison_bundle_v1'
FAILURE_FORMAT = 'retained_gradient_comparison_failure_v1'
MANDATORY_GPU_ENVIRONMENT = ('AMDGCN_USE_BUFFER_OPS', 'TRITON_FULL_AUTOTUNE', 'TRITON_ALLOW_PIPELINING')
SOURCE_FILES = ('scripts/repro_gradient_comparison.py', 'scripts/nan_backward_boundaries.py',
                'scripts/replay_backward_boundary.py', 'scripts/capture_additional_linear_deferred.py')
LIMITS = [
    'This replays retained comparison operands; original producer ordering, surrounding training and allocator history are absent.',
    'Retaining intermediates and reading masks on the CPU changes allocation lifetimes and timing.',
    'Full input backing bytes are checked after restore, at failure and at successful termination; intervening calls have version checks.',
    'Matching saved inputs does not exclude transient mutation during execution or a capture/transfer fault.',
    'The deferred scalar-summary queue is not replayed by this comparison-only script.',
]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def new_path(path):
    p = Path(path)
    if os.path.lexists(p) or os.path.lexists(str(p) + '.tmp'):
        raise FileExistsError(str(p))
    return p


def atomic_save(path, writer):
    p = new_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + '.tmp')
    owned = False
    try:
        with tmp.open('xb') as f:
            owned = True
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, p)
    finally:
        if owned:
            tmp.unlink(missing_ok=True)


def save_json(path, value):
    atomic_save(path, lambda f: f.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode()))


def raw_bytes(t):
    s = t.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(s, 0, (s.nbytes(),), (1,))


def manifest(tree):
    """CPU byte hashes plus canonical alias groups; no GPU reductions."""
    groups, stores, views = {}, [], {}
    for name, t in _tensors(tree):
        if t.device.type != 'cpu' or t.layout != torch.strided or t.is_quantized or t.is_conj() or t.is_neg():
            raise ValueError('Manifest requires plain strided CPU tensors')
        sid = t.untyped_storage()._cdata
        if sid not in groups:
            groups[sid] = len(stores)
            b = raw_bytes(t)
            stores.append({'bytes': b.numel(), 'sha256': hashlib.sha256(memoryview(b.numpy())).hexdigest()})
        logical = t.detach().contiguous().reshape(-1).view(torch.uint8)
        views[name] = {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
                       'storage_offset': t.storage_offset(), 'storage_group': groups[sid],
                       'logical_sha256': hashlib.sha256(memoryview(logical.numpy())).hexdigest()}
    return {'stores': stores, 'views': views, 'storage_bytes': sum(s['bytes'] for s in stores)}


def validate_inputs(inputs):
    if not isinstance(inputs, dict) or set(inputs) != {'weight', 'bias'}:
        raise ValueError('Bundle must retain both weight and bias operands')
    for name, pair in inputs.items():
        if not isinstance(pair, dict) or set(pair) != {'expected', 'actual'}:
            raise ValueError('Each pair needs expected and actual views')
        a, b = pair['expected'], pair['actual']
        for t in (a, b):
            if (not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or t.dtype != torch.float32
                    or t.layout != torch.strided or t.is_conj() or t.is_neg() or t.numel() == 0):
                raise ValueError('Selected views must be nonempty plain CPU FP32 tensors')
            if not bool(torch.isfinite(t).all()):
                raise ValueError('Selected views must be finite')
        if a.shape != b.shape or not torch.equal(a, b):
            raise ValueError('Selected logical operands must be exactly numerically equal')
        if not torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)):
            raise ValueError('Selected logical operands must also be bitwise identical')


def extract(payload, *, provenance, chunk_bytes=64 << 20, max_bytes=256 << 20):
    stages = payload['stages']
    inputs = {n: {'expected': stages['parameter_post_' + n], 'actual': stages['after_backward_return'][n]}
              for n in ('weight', 'bias')}
    validate_inputs(inputs)
    original = manifest(inputs)
    copied, total = _cpu_copy_tree(inputs, chunk_bytes, max_bytes)
    if manifest(copied) != original:
        raise ValueError('Extraction changed a full backing, layout or alias relationship')
    return {'format': BUNDLE_FORMAT, 'inputs': copied, 'input_manifest': original,
            'comparison_chunk_bytes': chunk_bytes, 'storage_bytes': total,
            'provenance': dict(provenance, capture_format=payload.get('format'),
                               capture_variant=payload.get('capture_variant'), session=payload.get('session'),
                               attempt=payload.get('attempt'), trigger=payload.get('trigger'),
                               capture_source_sha256=payload.get('metadata', {}).get('source_sha256'))}


def prepare(bundle, *, max_bytes=256 << 20):
    if bundle.get('format') != BUNDLE_FORMAT:
        raise ValueError('Unexpected bundle format')
    validate_inputs(bundle['inputs'])
    actual = manifest(bundle['inputs'])
    if actual != bundle['input_manifest'] or actual['storage_bytes'] > max_bytes:
        raise ValueError('Bundle full bytes/layout/aliases or storage budget mismatch')
    if type(bundle['comparison_chunk_bytes']) is not int or bundle['comparison_chunk_bytes'] <= 0:
        raise ValueError('Positive captured comparison chunk_bytes required')
    return bundle


def layout_control(bundle, layout):
    if layout == 'original':
        return bundle
    if layout != 'compact':
        raise ValueError('Layout must be original or compact')
    inputs = {n: {role: t.clone(memory_format=torch.contiguous_format) for role, t in pair.items()}
              for n, pair in bundle['inputs'].items()}
    validate_inputs(inputs)
    compact = manifest(inputs)
    if any(compact['views'][n]['logical_sha256'] != v['logical_sha256']
           for n, v in bundle['input_manifest']['views'].items()):
        raise ValueError('Compact-layout intervention changed logical bytes')
    return dict(bundle, inputs=inputs, input_manifest=compact, storage_bytes=compact['storage_bytes'],
                provenance=dict(bundle['provenance'], layout_intervention='compact_contiguous_offset0_independent_storages',
                                original_input_manifest=bundle['input_manifest']))


def compare_once(expected, actual, *, chunk_bytes):
    """Frozen operators, with Python references retaining each existing result."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.device != actual.device:
        raise ValueError('Comparison shape/dtype/device mismatch')
    chunks = []
    with torch.no_grad(), torch.autocast(device_type=actual.device.type, enabled=False):
        equal = torch.ones((), dtype=torch.bool, device=actual.device)
        initial = equal
        limit = max(1, chunk_bytes // actual.element_size())
        for left, right in zip(_logical_chunks(expected, limit), _logical_chunks(actual, limit)):
            mask = left[0] == right[0]
            reduced = torch.all(mask)
            equal = torch.logical_and(equal, reduced)
            chunks.append({'origin': list(left[1]), 'mask': mask, 'all': reduced, 'running_equal': equal})
        mismatch = torch.logical_not(equal).to(torch.float32)
    return {'initial_equal': initial, 'chunks': chunks, 'mismatch': mismatch}


def _same_raw(a, b):
    return (a.device.type == b.device.type == 'cpu' and a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def analyze_observation(observed, pair, *, chunk_bytes):
    """CPU-only oracle distinguishes equality, reduction and scalar-chain errors."""
    for _, t in _tensors(observed):
        if t.device.type != 'cpu':
            raise ValueError('Observation must be copied to CPU before analysis')
    a, b = pair['expected'], pair['actual']
    if a.device.type != 'cpu' or b.device.type != 'cpu':
        raise ValueError('Oracle operands must be CPU')
    problems = []
    if not _same_raw(observed['initial_equal'], torch.tensor(True)):
        problems.append('initial_scalar_disagreement')
    limit = max(1, chunk_bytes // b.element_size())
    paired = list(zip(_logical_chunks(a, limit), _logical_chunks(b, limit)))
    if len(paired) != len(observed['chunks']):
        raise ValueError('Captured chunk count differs from frozen traversal')
    expected_running = bool(observed['initial_equal'])
    mask_false = 0
    for i, ((left, right), chunk) in enumerate(zip(paired, observed['chunks'])):
        oracle = left[0] == right[0]
        mask = chunk['mask']
        if chunk['origin'] != list(left[1]) or mask.shape != oracle.shape or mask.dtype != torch.bool:
            raise ValueError('Captured equality mask layout differs from comparison traversal')
        if not _same_raw(mask, oracle):
            problems.append(f'elementwise_equality_disagreement:{i}')
        mask_false += int((~mask).sum())
        reduced = bool(mask.all())
        if not _same_raw(chunk['all'], torch.tensor(reduced)):
            problems.append(f'all_reduction_disagreement:{i}')
        # Use the recorded reduction as the input to the recorded scalar chain.
        expected_running = expected_running and bool(chunk['all'])
        if not _same_raw(chunk['running_equal'], torch.tensor(expected_running)):
            problems.append(f'logical_and_disagreement:{i}')
        expected_running = bool(chunk['running_equal'])
    expected_mismatch = torch.tensor(float(not expected_running), dtype=torch.float32)
    if not _same_raw(observed['mismatch'], expected_mismatch):
        problems.append('logical_not_or_float_conversion_disagreement')
    if not _same_raw(observed['mismatch'], torch.tensor(0.0)):
        problems.append('final_exact_equality_oracle_disagreement')
    return {'failed': bool(problems), 'classifications': problems, 'false_mask_elements': mask_false,
            'chunk_count': len(paired), 'mismatch_raw_hex': bytes(observed['mismatch'].reshape(-1).view(torch.uint8).tolist()).hex()}


def source_inventory():
    root = Path(__file__).resolve().parents[1]
    return {p: file_hash(root / p) for p in SOURCE_FILES}


def run(prepared, *, failure_path, repeats=1000, pairs=('weight',), device='cuda:0',
        max_bytes=256 << 20, copy_chunk_bytes=64 << 20, compare_function=None, progress=None):
    if type(repeats) is not int or not 1 <= repeats <= 1000 or not pairs or any(n not in ('weight', 'bias') for n in pairs):
        raise ValueError('Repeats1..1000 and weight/bias pairs required')
    new_path(failure_path)
    prepared = prepare(prepared, max_bytes=max_bytes)
    pristine = prepared['inputs']
    resident, _ = restore_raw_tree(pristine, device=device, chunk_bytes=copy_chunk_bytes, max_bytes=max_bytes)
    restored, _ = _cpu_copy_tree(resident, copy_chunk_bytes, max_bytes)
    if manifest(restored) != prepared['input_manifest']:
        raise ValueError('Initial full restored bytes/layout/aliases disagree')
    del restored
    versions = {n: t._version for n, t in _tensors(resident)}
    function = compare_once if compare_function is None else compare_function
    records = []
    for iteration in range(1, repeats + 1):
        for name in pairs:
            pair = resident[name]
            output = function(pair['expected'], pair['actual'], chunk_bytes=prepared['comparison_chunk_bytes'])
            observed, _ = _cpu_copy_tree(output, copy_chunk_bytes, max_bytes)
            check = analyze_observation(observed, pristine[name], chunk_bytes=prepared['comparison_chunk_bytes'])
            changed = [n for n, t in _tensors(resident) if versions[n] != t._version]
            record = {'iteration': iteration, 'pair': name, 'check': check, 'changed_input_versions': changed}
            failed = check['failed'] or bool(changed)
            final = iteration == repeats and name == pairs[-1]
            if failed or final:
                # Existing outputs and inputs copied jointly; no comparison rerun.
                current, _ = _cpu_copy_tree({'inputs': resident, 'observation': output}, copy_chunk_bytes, max_bytes)
                current_manifest = manifest(current['inputs'])
                record['full_input_bytes_unchanged'] = current_manifest == prepared['input_manifest']
                record['observation_bytes_stable_during_capture'] = manifest(observed) == manifest(current['observation'])
                failed |= not record['full_input_bytes_unchanged'] or not record['observation_bytes_stable_during_capture']
                saved_check = analyze_observation(current['observation'], current['inputs'][name],
                                                  chunk_bytes=prepared['comparison_chunk_bytes'])
                record['saved_current_input_check'] = saved_check
            record['failed'] = failed
            records.append(record)
            if failed:
                status = 'FAIL' if record['observation_bytes_stable_during_capture'] else 'CAPTURE_INTEGRITY_FAILURE'
                artifact = {'format': FAILURE_FORMAT, 'status': status, 'iteration': iteration, 'pair': name,
                            'record': record, 'pristine_inputs': pristine, 'first_cpu_observation': observed,
                            'current_at_failure': current, 'pristine_input_manifest': prepared['input_manifest'],
                            'current_input_manifest': current_manifest, 'bundle_provenance': prepared['provenance'],
                            'source_sha256': source_inventory(), 'limits': LIMITS}
                atomic_save(failure_path, lambda f: torch.save(artifact, f))
                if progress:
                    progress(record)
                return {'status': status, 'iterations_completed': iteration, 'records': records,
                        'failure_capture': {'path': str(failure_path), 'bytes': Path(failure_path).stat().st_size,
                                            'sha256': file_hash(failure_path)}, 'limits': LIMITS}
            if progress:
                progress(record)
            del output, observed
    return {'status': 'PASS', 'iterations_completed': repeats, 'records': records,
            'final_input_bytes_unchanged': records[-1]['full_input_bytes_unchanged'], 'limits': LIMITS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--extract-payload', type=Path)
    group.add_argument('--bundle', type=Path)
    parser.add_argument('--bundle-output', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--failure-dump', type=Path)
    parser.add_argument('--comparison-chunk-bytes', type=int, default=64 << 20)
    parser.add_argument('--max-bytes', type=int, default=256 << 20)
    parser.add_argument('--pairs', choices=('weight', 'bias', 'both'), default='weight')
    parser.add_argument('--layout', choices=('original', 'compact'), default='original')
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.max_bytes <= 0 or args.comparison_chunk_bytes <= 0 or args.cpu_threads < 1 or not 1 <= args.repeats <= 1000:
        parser.error('Positive budgets/threads/chunk bytes and repeats1..1000 required')
    if args.extract_payload and (not args.bundle_output or args.gpu):
        parser.error('Extraction requires --bundle-output and is CPU-only')
    if args.extract_payload and args.layout != 'original':
        parser.error('Extraction preserves original layout; compact is a replay intervention')
    if args.gpu:
        if not args.failure_dump:
            parser.error('--failure-dump required for GPU replay')
        for key in MANDATORY_GPU_ENVIRONMENT:
            if os.environ.get(key) != '0':
                parser.error(key + '=0 required before GPU initialization')
    paths = [p.resolve() for p in (args.extract_payload, args.bundle, args.bundle_output, args.report, args.failure_dump) if p]
    if len(paths) != len(set(paths)):
        parser.error('Input and output paths must be distinct')
    new_path(args.report)
    if args.failure_dump:
        new_path(args.failure_dump)
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    if args.extract_payload:
        new_path(args.bundle_output)
        before = args.extract_payload.stat()
        payload = torch.load(args.extract_payload, map_location='cpu', mmap=True, weights_only=False)
        provenance = {'payload_path': str(args.extract_payload), 'payload_bytes': before.st_size,
                      'payload_mtime_ns': before.st_mtime_ns, 'whole_payload_sha256': 'not_computed',
                      'scope': 'Complete selected backing stores hashed; unrelated large payload stores not read.'}
        bundle = extract(payload, provenance=provenance, chunk_bytes=args.comparison_chunk_bytes, max_bytes=args.max_bytes)
        after = args.extract_payload.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Payload changed during extraction')
        atomic_save(args.bundle_output, lambda f: torch.save(bundle, f))
        loaded = torch.load(args.bundle_output, map_location='cpu', weights_only=False)
        prepare(loaded, max_bytes=args.max_bytes)
        report = {'status': 'CPU_BUNDLE_VERIFIED', 'bundle': {'path': str(args.bundle_output),
                    'sha256': file_hash(args.bundle_output), 'bytes': args.bundle_output.stat().st_size},
                  'input_manifest': bundle['input_manifest'], 'provenance': bundle['provenance']}
    else:
        bundle = prepare(torch.load(args.bundle, map_location='cpu', weights_only=False), max_bytes=args.max_bytes)
        bundle = layout_control(bundle, args.layout)
        report = {'status': 'CPU_PREPARED', 'bundle': {'path': str(args.bundle), 'sha256': file_hash(args.bundle)},
                  'input_manifest': bundle['input_manifest'], 'provenance': bundle['provenance']}
        if args.gpu:
            pairs = ('weight', 'bias') if args.pairs == 'both' else (args.pairs,)
            report.update(run(bundle, failure_path=args.failure_dump, repeats=args.repeats, pairs=pairs,
                              max_bytes=args.max_bytes, progress=lambda r: print(json.dumps(r), flush=True)))
    report.update(format='gradient_comparison_report_v1', seconds=time.monotonic() - started,
                  source_sha256=source_inventory(), limits=LIMITS,
                  runtime_versions={'torch': str(torch.__version__), 'hip': torch.version.hip},
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  gpu_initialized=torch.cuda.is_initialized())
    save_json(args.report, report)
    print(json.dumps({'status': report['status'], 'report': str(args.report)}))
    return 1 if report['status'] in ('FAIL', 'CAPTURE_INTEGRITY_FAILURE') else 0


if __name__ == '__main__':
    raise SystemExit(main())
