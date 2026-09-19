#!/usr/bin/env python3
"""Replay trusted focused Linear captures with isolated GEMM/autograd routes.

CPU inspection is the default. GPU execution is explicit and requires context.
Original anomalous inputs remain propagation experiments; zero-DY is a distinct
transformation with an exact numerical-zero oracle. No failure is rerun.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import replay_backward_boundary as common
import replay_hstu_output_boundary as helpers
import stress_addmm_backward_boundary as support
import audit_training_arithmetic_bounds as arithmetic

FORMAT = 'additional_linear_capture_v1'
ROUTES = ('raw_dw', 'addmm', 'autocast_linear')
SOURCE_FILES = (
    'replay_additional_linear_capture.py', 'replay_backward_boundary.py',
    'replay_hstu_output_boundary.py', 'stress_addmm_backward_boundary.py',
    'stress_hstu_output_boundary.py', 'nan_backward_boundaries.py',
    'audit_training_arithmetic_bounds.py', 'audit_training_gradient_flow.py',
)
LIMITS = [
    'Isolated replay does not recreate training allocator history, DDP buckets or surrounding kernels.',
    'Explicit X.T @ DY and AddmmBackward may use different BLAS orientations and output layouts.',
    'Original saved gradients are observations, not independent mathematical references.',
    'Complete input bytes are checked initially and at failure/final return; intermediate version checks do not exclude transient or untracked writes.',
    'An operation exception before outputs return may leave outputs unavailable; complete failure preservation applies to completed calls.',
]


def named_tensors(tree, prefix=''):
    torch = common._torch()
    if isinstance(tree, torch.Tensor):
        yield prefix, tree
    elif isinstance(tree, dict):
        for name, value in tree.items():
            yield from named_tensors(value, prefix + '/' + str(name))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from named_tensors(value, prefix + '/' + str(index))


def digest_tree(tree, chunk_bytes=64 << 20, threshold=1e6):
    """Full backing hashes and logical/layout identity, including singleton views."""
    torch = common._torch()
    storages, raw, tensors = {}, [], {}
    for name, value in named_tensors(tree):
        storage = value.untyped_storage()
        key = (str(value.device), storage._cdata)
        if key not in storages:
            storages[key] = len(raw)
            byte_view = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
            digest = hashlib.sha256()
            for start in range(0, storage.nbytes(), chunk_bytes):
                digest.update(byte_view[start:start + chunk_bytes].cpu().numpy().tobytes())
            raw.append({'bytes': storage.nbytes(), 'sha256': digest.hexdigest()})
        logical = hashlib.sha256()
        nonfinite = extreme = 0
        maximum = 0.
        for chunk in common._logical_chunks(value, max(1, chunk_bytes // 32)):
            cpu = torch.empty(chunk.shape, dtype=chunk.dtype, device='cpu').copy_(chunk.detach())
            logical.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
            numbers = cpu.double()
            finite = torch.isfinite(numbers)
            nonfinite += int((~finite).sum())
            extreme += int((finite & (numbers.abs() > threshold)).sum())
            if finite.any():
                maximum = max(maximum, float(numbers[finite].abs().max()))
        tensors[name] = {'shape': list(value.shape), 'stride': list(value.stride()),
                         'dtype': str(value.dtype), 'storage_offset': value.storage_offset(),
                         'storage_index': storages[key], 'logical_sha256': logical.hexdigest(),
                         'nonfinite_count': nonfinite, 'extreme_count': extreme, 'max_abs_finite': maximum}
    return {'tensors': tensors, 'raw_storages': raw}


def validate_capture(payload):
    torch = common._torch()
    if (not isinstance(payload, dict) or payload.get('format') != FORMAT
            or type(payload.get('format_version')) is not int or payload['format_version'] != 1):
        raise ValueError('Requires additional_linear_capture_v1')
    metadata = payload.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Capture metadata is required')
    provenance = metadata.get('input_provenance', {})
    if (not isinstance(provenance, dict) or provenance.get('pristine') is not True
            or provenance.get('operation_started') is not False):
        raise ValueError('Explicit pristine pre-node input provenance is required')
    if metadata.get('node_kind') != 'AddmmBackward0':
        raise ValueError('Requires actual captured AddmmBackward0 node')
    if any(type(metadata.get(key)) not in (int, float) or metadata[key] != 1 for key in ('alpha', 'beta')):
        raise ValueError('Only ordinary Linear alpha=beta=1 is supported')
    preference = metadata.get('preferred_blas_library')
    if preference is not None and preference not in BLAS_PREFERENCES:
        raise ValueError('Unsupported captured BLAS preference: ' + str(preference))
    for phase in ('forward', 'backward'):
        support.validate_controls(metadata.get('execution_controls_' + phase))
    if any(v['enabled'] for v in metadata['execution_controls_backward']['autocast'].values()):
        raise ValueError('Saved-operand backward replay requires captured autocast disabled')
    inputs = payload.get('pristine_inputs')
    if not isinstance(inputs, dict) or set(inputs) != {'x', 'mat2', 'dy'}:
        raise ValueError('Exactly x, mat2 and dy saved operands are required')
    for name, value in named_tensors(payload):
        if (value.device.type != 'cpu' or value.layout != torch.strided or value.is_quantized
                or value.is_conj() or value.is_neg() or not value.is_floating_point()):
            raise ValueError('Every captured tensor must be real CPU strided storage: ' + name)
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) < 1:
            raise ValueError('Saved operands must be nonempty matrices: ' + name)
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('Saved operands must be BF16 or FP32')
    x, mat2, dy = (inputs[k] for k in ('x', 'mat2', 'dy'))
    if x.dtype != mat2.dtype or x.dtype != dy.dtype or mat2.shape[0] != x.shape[1] or tuple(dy.shape) != (x.shape[0], mat2.shape[1]):
        raise ValueError('Saved Addmm operand shape/dtype mismatch')
    forward = payload.get('forward_inputs')
    if forward is not None:
        if not isinstance(forward, dict) or set(forward) != {'x', 'weight', 'bias'}:
            raise ValueError('Forward inputs require x, weight and bias')
        shapes = {'x': x.shape, 'weight': (mat2.shape[1], mat2.shape[0]), 'bias': (mat2.shape[1],)}
        for name, value in forward.items():
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shapes[name]) or value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError('Forward input shape/dtype mismatch: ' + name)
    stages = payload.get('stages', {})
    if not isinstance(stages, dict) or not isinstance(stages.get('raw_small', {}), dict):
        raise ValueError('Stages and raw_small must be mappings')
    raw = stages.get('raw_small', {}).get('dmat2')
    if raw is not None and (not isinstance(raw, torch.Tensor) or raw.shape != mat2.shape or raw.dtype != mat2.dtype):
        raise ValueError('Captured raw dmat2 has incompatible shape/dtype')
    return {'inputs': inputs, 'forward_inputs': forward, 'metadata': copy.deepcopy(metadata),
            'stages': payload.get('stages', {}), 'outputs': payload.get('outputs', {})}


def prepare_capture(payload, *, zero_dy=False, chunk_bytes=64 << 20, max_bytes=64 << 30):
    if type(zero_dy) is not bool or type(chunk_bytes) is not int or chunk_bytes <= 0 or max_bytes <= 0:
        raise ValueError('Boolean zero_dy and positive copy/storage limits required')
    bound = validate_capture(payload)
    operands = {'pristine_inputs': bound['inputs']}
    if bound['forward_inputs'] is not None:
        operands['forward_inputs'] = bound['forward_inputs']
    original_digest = digest_tree(operands, chunk_bytes)
    transformed = operands
    if zero_dy:
        dy_storage = bound['inputs']['dy'].untyped_storage()._cdata
        protected = {'x': bound['inputs']['x'], 'mat2': bound['inputs']['mat2'],
                     **{('forward_' + k): v for k, v in (bound['forward_inputs'] or {}).items()}}
        if any(t.untyped_storage()._cdata == dy_storage for t in protected.values()):
            raise ValueError('Zero DY must not share backing storage with protected operands')
        if any(v['nonfinite_count'] for v in digest_tree(protected, chunk_bytes)['tensors'].values()):
            raise ValueError('Zero oracle requires finite protected x/mat2/forward inputs')
        transformed, _ = common.restore_raw_tree(operands, device='cpu', chunk_bytes=chunk_bytes, max_bytes=max_bytes)
        transformed['pristine_inputs']['dy'].zero_()
    prepared_digest = digest_tree(transformed, chunk_bytes)
    bridge = None
    if bound['forward_inputs'] is not None:
        forward = bound['forward_inputs']
        bridge = {name: equal_values(value, bound['inputs'][name]) for name, value in {
            'x': forward['x'].to(bound['inputs']['x'].dtype),
            'mat2': forward['weight'].to(bound['inputs']['mat2'].dtype).t()}.items()}
    return {'original': payload, 'prepared': transformed, 'metadata': bound['metadata'],
            'forward_to_backward_cast_value_bridge': bridge,
            'transformation': {'zero_dy': zero_dy, 'mode': 'zero_dy_exact_oracle' if zero_dy else 'original_saved_operands',
                               'original_inputs_nonfinite': any(v['nonfinite_count'] for v in original_digest['tensors'].values()),
                               'scope': 'Only logical DY is zeroed; complete CPU storage clone preserves aliases and unused bytes.' if zero_dy else 'Original finite/nonfinite input evidence is preserved; anomalous inputs make this a propagation experiment.'},
            'original_digest': original_digest, 'prepared_digest': prepared_digest}


def equal_values(left, right, chunk_elements=1 << 24):
    torch = common._torch()
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    # Compare bytes so NaN payloads and signed zeros remain evidence.
    for a, b in zip(common._logical_chunks(left, chunk_elements), common._logical_chunks(right, chunk_elements)):
        aa = torch.empty(a.shape, dtype=a.dtype, device=a.device).copy_(a.detach())
        bb = torch.empty(b.shape, dtype=b.dtype, device=b.device).copy_(b.detach())
        if not bool((aa.reshape(-1).view(torch.uint8) == bb.reshape(-1).view(torch.uint8)).all()):
            return False
    return True


BLAS_PREFERENCES = {'_BlasBackend.Default': 'default', '_BlasBackend.Cublas': 'hipblas',
                    '_BlasBackend.Cublaslt': 'hipblaslt', '_BlasBackend.Ck': 'ck'}


@contextmanager
def restored_blas_preference(metadata):
    torch = common._torch()
    function = getattr(torch.backends.cuda, 'preferred_blas_library', None)
    captured = metadata.get('preferred_blas_library')
    if captured is None:
        yield {'restored': False, 'reason': 'Capture lacks backend preference'}
        return
    if function is None or captured not in BLAS_PREFERENCES:
        raise ValueError('Cannot restore captured BLAS preference')
    previous = function()
    try:
        function(BLAS_PREFERENCES[captured])
        if str(function()) != captured:
            raise ValueError('BLAS preference did not restore exactly')
        yield {'restored': True, 'captured': captured, 'effective': str(function())}
    finally:
        function(previous)


def execute_route(resident, metadata, route):
    with restored_blas_preference(metadata):
        return _execute_route(resident, metadata, route)


def _execute_route(resident, metadata, route):
    torch = common._torch()
    if route not in ROUTES:
        raise ValueError('Unknown replay route')
    x, mat2, dy = (resident['pristine_inputs'][k] for k in ('x', 'mat2', 'dy'))
    backward = metadata['execution_controls_backward']
    if route == 'raw_dw':
        with helpers.restored_execution_controls(backward), torch.no_grad():
            return {'raw_dw': torch.mm(x.t(), dy)}
    if route == 'addmm':
        with helpers.restored_execution_controls(backward), torch.enable_grad():
            a, b = x.detach().requires_grad_(), mat2.detach().requires_grad_()
            bias = torch.zeros(mat2.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
            y = torch.addmm(bias, a, b, alpha=1, beta=1)
            if type(y.grad_fn).__name__ != 'AddmmBackward0':
                raise ValueError('Unexpected actual Addmm backward node')
            dx, dw, db = torch.autograd.grad(y, (a, b, bias), grad_outputs=dy)
            return {'raw_dw': dw, 'dx': dx, 'db': db}
    if 'forward_inputs' not in resident:
        raise ValueError('autocast_linear requires original forward inputs')
    forward = resident['forward_inputs']
    with torch.enable_grad():
        fx, weight, bias = (forward[k].detach().requires_grad_() for k in ('x', 'weight', 'bias'))
        with helpers.restored_execution_controls(metadata['execution_controls_forward']):
            y = torch.nn.functional.linear(fx, weight, bias)
        node = y.grad_fn
        if type(node).__name__ != 'AddmmBackward0' or not hasattr(node, '_saved_mat1'):
            raise ValueError('Unexpected autocast Linear Addmm node')
        if not equal_values(node._saved_mat1, x) or not equal_values(node._saved_mat2, mat2):
            raise ValueError('Replayed forward saved operands differ from captured backward-time operands; possible original forward-to-backward mutation')
        if y.dtype != dy.dtype or y.shape != dy.shape:
            raise ValueError('Replayed forward output cannot accept captured DY')
        raw = {}
        def capture(grad_inputs, grad_outputs):
            if len(grad_inputs) != 3 or any(v is None for v in grad_inputs):
                raise ValueError('Unexpected raw Linear gradient tuple')
            raw.update(db=grad_inputs[0], dx=grad_inputs[1], raw_dw=grad_inputs[2])
        handle = node.register_hook(capture)
        try:
            with helpers.restored_execution_controls(backward):
                _, dw, db = torch.autograd.grad(y, (fx, weight, bias), grad_outputs=dy)
        finally:
            handle.remove()
        if set(raw) != {'db', 'dx', 'raw_dw'}:
            raise ValueError('Raw Addmm backward hook did not run exactly once')
        return {**raw, 'dw': dw, 'leaf_db': db}


def scan_outputs(outputs, *, zero_dy=False, threshold=1e6, chunk_elements=1 << 24):
    torch = common._torch()
    if not isinstance(outputs, dict) or 'raw_dw' not in outputs or not outputs:
        raise ValueError('Replay must return named outputs including raw_dw')
    result = {}
    for name, tensor in outputs.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError('Replay output must be a floating tensor')
        rows = []
        for chunk in common._logical_chunks(tensor, chunk_elements):
            value = torch.empty(chunk.shape, dtype=chunk.dtype, device=chunk.device).copy_(chunk.detach())
            numbers = value.double()
            finite = torch.isfinite(numbers)
            integer = {2: torch.int16, 4: torch.int32, 8: torch.int64}[value.element_size()]
            magnitude_mask = (1 << (value.element_size() * 8 - 1)) - 1
            nonzero = ((value.reshape(-1).view(integer) & magnitude_mask) != 0).sum()
            maximum = torch.where(finite, numbers.abs(), 0.).amax() if numbers.numel() else torch.zeros((), device=tensor.device)
            rows.append(torch.stack(((~finite).sum().double(), (finite & (numbers.abs() > threshold)).sum().double(), nonzero.double(), maximum)))
        if rows:
            matrix = torch.stack(rows)
            counts = torch.cat((matrix[:, :3].sum(0), matrix[:, 3].amax().reshape(1))).cpu().tolist()
        else:
            counts = (0, 0, 0, 0.)
        result[name] = dict(zip(('nonfinite_count', 'extreme_count', 'nonzero_including_nonfinite_count'), map(int, counts[:3])))
        result[name]['max_abs_finite'] = counts[3]
    failed = any(v['nonzero_including_nonfinite_count'] if zero_dy else v['nonfinite_count'] or v['extreme_count'] for v in result.values())
    return {'failed': bool(failed), 'zero_dy': zero_dy, 'outputs': result}


def raw_to_leaf_conversion(outputs, route):
    if route != 'autocast_linear':
        return {'applicable': False}
    torch = common._torch()
    if any(not isinstance(outputs.get(key), torch.Tensor) for key in ('raw_dw', 'db', 'dw', 'leaf_db')):
        raise ValueError('Autocast replay must retain raw and leaf weight/bias gradients')
    with torch.no_grad():
        weight = equal_values(outputs['raw_dw'].t().to(outputs['dw'].dtype), outputs['dw'])
        bias = equal_values(outputs['db'].to(outputs['leaf_db'].dtype), outputs['leaf_db'])
    return {'applicable': True, 'weight_matches_raw_transpose_cast': weight,
            'bias_matches_raw_cast': bias, 'failed': not (weight and bias),
            'scope': 'Exact transformation of this call\'s raw gradient, independent of its mathematical correctness.'}


def resource_plan(prepared, route):
    """Conservative live tensor bound; backend workspaces remain unbounded here."""
    if route not in ROUTES:
        raise ValueError('Unknown replay route')
    x, mat2, dy = (prepared['prepared']['pristine_inputs'][k] for k in ('x', 'mat2', 'dy'))
    m, k, n, element = x.shape[0], x.shape[1], mat2.shape[1], x.element_size()
    raw_dw = k * n * element
    if route == 'raw_dw':
        temporary = raw_dw
        largest_checked = k * n
    elif route == 'addmm':
        temporary = element * (m * n + m * k + k * n + 2 * n)
        largest_checked = max(m * k, k * n)
    else:
        forward = prepared['prepared'].get('forward_inputs')
        if forward is None:
            raise ValueError('autocast_linear requires original forward inputs')
        # Optional saved casts, forward output, raw node outputs, returned
        # original-dtype gradients, and an additional largest-matrix temporary.
        casts = element * (m * k + k * n + n)
        forward_output = m * n * element
        raw_outputs = element * (m * k + k * n + n)
        leaf_outputs = sum(forward[name].numel() * forward[name].element_size() for name in ('x', 'weight', 'bias'))
        temporary = casts + forward_output + raw_outputs + leaf_outputs + max(forward_output, m * k * 4)
        largest_checked = max(m * k, k * n)
    scratch = min(largest_checked, 1 << 24) * 64
    resident = support.storage_bytes(prepared['prepared'])
    original = support.storage_bytes({'original': prepared['original'], 'prepared': prepared['prepared']})
    return {'resident_storage_bytes': resident, 'route_peak_tensor_allowance_bytes': temporary,
            'bounded_check_scratch_allowance_bytes': scratch,
            'resident_peak_before_backend_workspace_bytes': resident + temporary + scratch,
            'conservative_complete_failure_storage_bytes': original + resident + temporary,
            'scope': 'Conservative route tensor/checker allowance; backend workspace and allocator reservation excluded.'}


def conditional_dw_bound(prepared):
    x = prepared['prepared']['pristine_inputs']['x']
    entries = prepared['prepared_digest']['tensors']
    a, b = entries['/pristine_inputs/x'], entries['/pristine_inputs/dy']
    if x.dtype != common._torch().bfloat16:
        return {'applicable': False, 'reason': 'BF16 FP32-accumulation model only'}
    if a['nonfinite_count'] or b['nonfinite_count']:
        return {'applicable': False, 'reason': 'Nonfinite saved input; propagation experiment'}
    bound, reason = arithmetic.gemm_bound(a['max_abs_finite'], b['max_abs_finite'], x.shape[0])
    return {'applicable': bound is not None, 'reason': reason, 'bound': bound,
            'scope': 'Conditional on FP32 accumulation and one BF16 rounding; actual BLAS reduction precision is not proved by flags. No unconditional finite-input impossibility claim.'}


def original_reference_comparisons(outputs, payload):
    stages = payload.get('stages', {})
    raw = stages.get('raw_small', {})
    references = {'raw_dw': raw.get('dmat2'), 'db': raw.get('bias'),
                  'dw': stages.get('parameter_pre_weight'), 'leaf_db': stages.get('parameter_pre_bias')}
    results = {}
    torch = common._torch()
    for name, value in outputs.items():
        reference = references.get(name)
        if not isinstance(reference, torch.Tensor) or reference.shape != value.shape or reference.dtype != value.dtype:
            continue
        compared = support.checks_helper.compare_device_outputs((value,), (reference.to(value.device),), (name,),
                                                                 threshold=1e6, rtol=1e-2, atol=1e-12)
        results[name] = compared[name]
    return {'comparisons': results,
            'scope': 'Original observed raw/leaf gradients; may be nonfinite or incorrect. Differences are evidence and do not define the correctness oracle.'}


def source_inventory():
    root = Path(__file__).resolve().parent
    return {str(root / name): support.file_hash(root / name) for name in SOURCE_FILES}


def runtime_versions():
    torch = common._torch()
    versions = {'python': platform.python_version(), 'torch': str(torch.__version__), 'hip': torch.version.hip}
    for package in ('torch', 'triton', 'numpy'):
        try:
            versions[package + '_distribution'] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package + '_distribution'] = None
    try:
        versions['triton'] = importlib.import_module('triton').__version__
    except ImportError:
        versions['triton'] = None
    return versions


def validate_runtime(context, versions):
    if not isinstance(context, dict):
        raise ValueError('Runtime context must be a mapping')
    for key in ('torch', 'hip'):
        if key not in context:
            raise ValueError('Context runtime identity missing: ' + key)
    for key in ('torch', 'hip', 'triton', 'torch_distribution', 'triton_distribution', 'numpy_distribution'):
        if key in context and context[key] != versions.get(key):
            raise ValueError('Captured/current runtime differs: ' + key)


def run_replay(payload, prepared, *, route='raw_dw', repeats=1000, device='cpu', failure_path,
               max_resident_bytes=64 << 30, max_capture_bytes=128 << 30,
               chunk_bytes=64 << 20, threshold=1e6, function=None, progress=None, configuration=None):
    torch = common._torch()
    support.new_path(failure_path)
    if route not in ROUTES or type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError('Valid route and repeats1..1000 required')
    if (not math.isfinite(threshold) or threshold <= 0 or type(chunk_bytes) is not int or chunk_bytes <= 0
            or min(max_resident_bytes, max_capture_bytes) <= 0):
        raise ValueError('Positive finite scan limits required')
    x, mat2 = (prepared['prepared']['pristine_inputs'][k] for k in ('x', 'mat2'))
    plan = resource_plan(prepared, route)
    original_bytes = support.storage_bytes({'original': payload, 'prepared': prepared['prepared']})
    if plan['resident_peak_before_backend_workspace_bytes'] > max_resident_bytes:
        raise ValueError('Resident budget is insufficient before dispatch')
    if plan['conservative_complete_failure_storage_bytes'] > max_capture_bytes:
        raise ValueError('Complete failure capture budget is insufficient before dispatch')
    resident, _ = common.restore_raw_tree(prepared['prepared'], device=device, chunk_bytes=chunk_bytes, max_bytes=max_resident_bytes)
    if digest_tree(resident, chunk_bytes) != prepared['prepared_digest']:
        raise RuntimeError('Restored operands differ in logical/layout/raw bytes')
    versions = {name: t._version for name, t in named_tensors(resident)}
    dispatch = function or execute_route
    records = []
    bound = conditional_dw_bound(prepared)
    runtime = {'versions': runtime_versions(), 'device': str(device),
               'route': route, 'metadata': prepared['metadata'], 'resource_plan': plan,
               'source_sha256_before': source_inventory(), 'configuration': copy.deepcopy(configuration),
               'arguments': {'repeats': repeats, 'threshold': threshold, 'chunk_bytes': chunk_bytes,
                             'max_resident_bytes': max_resident_bytes, 'max_capture_bytes': max_capture_bytes},
               'captured_blas_preference': prepared['metadata'].get('preferred_blas_library'),
               'blas_preference_before': support.blas_preference(),
               'injected_function': function is not None,
               'conditional_raw_dw_bound': bound,
               'forward_to_backward_cast_value_bridge': prepared['forward_to_backward_cast_value_bridge']}
    for iteration in range(1, repeats + 1):
        started = time.monotonic()
        outputs = dispatch(resident, prepared['metadata'], route)
        if (not isinstance(outputs, dict) or not isinstance(outputs.get('raw_dw'), torch.Tensor)
                or outputs['raw_dw'].shape != mat2.shape or outputs['raw_dw'].dtype != mat2.dtype):
            raise ValueError('Raw DW shape/dtype differs from saved mat2')
        checks = scan_outputs(outputs, zero_dy=prepared['transformation']['zero_dy'], threshold=threshold)
        conversion = raw_to_leaf_conversion(outputs, route)
        changes = [name for name, t in named_tensors(resident) if t._version != versions[name]]
        record = {'iteration': iteration, 'route': route, **checks, 'changed_input_versions': changes,
                  'raw_to_leaf_conversion': conversion}
        record['seconds_including_checks'] = time.monotonic() - started
        record['failed'] |= bool(changes) or conversion.get('failed', False)
        if record['failed'] or iteration == repeats:
            current_digest = digest_tree(resident, chunk_bytes)
            record['input_bytes_match_prepared'] = current_digest == prepared['prepared_digest']
            record['failed'] |= not record['input_bytes_match_prepared']
            if not prepared['transformation']['zero_dy']:
                record['original_reference_comparisons'] = original_reference_comparisons(outputs, payload)
        records.append(record)
        if record['failed']:
            from nan_backward_boundaries import _cpu_copy_tree
            union = {'inputs': resident, 'outputs': outputs}
            needed = original_bytes + support.storage_bytes(union)
            if needed > max_capture_bytes:
                raise ValueError('Actual first-failure storage exceeds budget; original outputs remain live')
            observed = digest_tree(outputs, chunk_bytes, threshold)
            current, _ = _cpu_copy_tree(union, chunk_bytes, max_capture_bytes - original_bytes)
            saved = scan_outputs(current['outputs'], zero_dy=prepared['transformation']['zero_dy'], threshold=threshold)
            if (digest_tree(current['outputs'], chunk_bytes, threshold) != observed or saved != checks
                    or digest_tree(current['inputs'], chunk_bytes) != current_digest):
                raise RuntimeError('Failure copy does not match observed output bytes/classification')
            artifact = {'format': 'additional_linear_replay_failure_v1', 'iteration': iteration,
                        'route': route, 'original_capture': payload, 'prepared_pre_call': prepared['prepared'],
                        'current_at_failure': current, 'observed_record': record,
                        'transformation': prepared['transformation'], 'original_digest': prepared['original_digest'],
                        'prepared_digest': prepared['prepared_digest'], 'current_input_digest': current_digest,
                        'observed_output_digest': observed, 'saved_output_checks': saved,
                        'storage_bytes': needed, 'metadata': prepared['metadata'],
                        'runtime': {**runtime, 'blas_preference': support.blas_preference(),
                                    'source_sha256_at_failure': source_inventory(),
                                    'loaded_blas_libraries': support.loaded_blas_libraries(),
                                    'backend_identity_limit': 'Loaded library hashes/preferences do not identify the executed GEMM code object or instruction.'},
                        'capture_timing': 'Existing first failing completed call; no rerun', 'limits': LIMITS}
            support.atomic_new_save(failure_path, lambda stream: torch.save(artifact, stream))
            if progress:
                progress(record)
            return {'status': 'FAIL', 'iterations_completed': iteration, 'iterations': records, 'runtime': runtime,
                    'failure_capture': {'path': str(failure_path), 'sha256': support.file_hash(failure_path),
                                        'bytes': Path(failure_path).stat().st_size, 'storage_bytes': needed}}
        if progress:
            progress(record)
        del outputs
    return {'status': 'PASS', 'iterations_completed': repeats, 'iterations': records, 'runtime': runtime,
            'final_input_bytes_match_prepared': True,
            'interpretation': 'No configured replay trigger; original nonfinite inputs remain propagation-only evidence.' if prepared['transformation']['original_inputs_nonfinite'] else 'No configured replay trigger; passing magnitude checks do not establish exact arithmetic correctness.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--context', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--failure-dump', type=Path)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--route', choices=ROUTES, default='raw_dw')
    parser.add_argument('--zero-dy', action='store_true')
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--max-resident-gib', type=int, default=64)
    parser.add_argument('--max-capture-gib', type=int, default=128)
    parser.add_argument('--copy-chunk-mib', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=1e6)
    args = parser.parse_args(argv)
    paths = [p for p in (args.capture, args.context, args.report, args.failure_dump) if p is not None]
    extended = paths + [p.with_name(p.name + '.tmp') for p in (args.report, args.failure_dump) if p is not None]
    if len({p.resolve() for p in extended}) != len(extended):
        parser.error('Source/output/temp paths must be distinct')
    support.new_path(args.report)
    if args.failure_dump:
        support.new_path(args.failure_dump)
    if (min(args.max_resident_gib, args.max_capture_gib, args.copy_chunk_mib, args.cpu_threads) < 1
            or not 1 <= args.repeats <= 1000 or not math.isfinite(args.threshold) or args.threshold <= 0):
        parser.error('Positive limits and repeats1..1000 required')
    if args.gpu and (args.context is None or args.failure_dump is None):
        parser.error('GPU execution requires --context and --failure-dump')
    configuration = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     'route': args.route, 'limits': LIMITS}
    report = {'status': 'PREPARING', 'configuration': configuration, 'iterations': []}
    support.atomic_new_save(args.report, lambda stream: stream.write((json.dumps(report, indent=2, allow_nan=False) + '\n').encode()))

    def save_report():
        temporary = args.report.with_name(args.report.name + '.tmp')
        owned = False
        try:
            with temporary.open('xb') as stream:
                owned = True
                stream.write((json.dumps(report, indent=2, allow_nan=False) + '\n').encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.report)
            descriptor = os.open(args.report.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            if owned:
                temporary.unlink(missing_ok=True)
            raise

    try:
        configuration['source_sha256_before'] = source_inventory()
        configuration['capture'] = {'path': str(args.capture), 'sha256': support.file_hash(args.capture),
                                    'bytes': args.capture.stat().st_size}
        context = json.loads(args.context.read_text()) if args.context else None
        configuration['context'] = {'path': str(args.context), 'sha256': support.file_hash(args.context)} if args.context else None
        configuration['captured_context'] = context
        environment = support.restore_environment(context) if context else None
        configuration['restored_environment'] = environment
        torch = common._torch()
        torch.set_num_threads(args.cpu_threads)
        versions = runtime_versions()
        configuration['versions'] = versions
        if args.gpu:
            validate_runtime(context, versions)
        payload = torch.load(args.capture, map_location='cpu', mmap=True, weights_only=False)
        prepared = prepare_capture(payload, zero_dy=args.zero_dy, chunk_bytes=args.copy_chunk_mib << 20,
                                   max_bytes=args.max_capture_gib << 30)
        configuration.update(transformation=prepared['transformation'], original_digest=prepared['original_digest'],
                             prepared_digest=prepared['prepared_digest'], captured_execution_metadata=prepared['metadata'])
        report.update(status='READY' if args.gpu else 'CPU_INSPECTED',
                      resource_plan=resource_plan(prepared, args.route), conditional_raw_dw_bound=conditional_dw_bound(prepared),
                      forward_to_backward_cast_value_bridge=prepared['forward_to_backward_cast_value_bridge'])
        save_report()
        if args.gpu:
            if prepared['metadata'].get('preferred_blas_library') is None:
                raise ValueError('GPU replay requires captured BLAS selection preference')
            configuration['device'] = {'name': torch.cuda.get_device_name(0), 'properties': str(torch.cuda.get_device_properties(0))}

            def progress(row):
                report['iterations'].append(row)
                report.update(status='FAIL' if row['failed'] else 'RUNNING', iterations_completed=row['iteration'])
                save_report()
                print(json.dumps(row, allow_nan=False), flush=True)

            report.update(run_replay(payload, prepared, route=args.route, repeats=args.repeats, device='cuda:0',
                          failure_path=args.failure_dump, max_resident_bytes=args.max_resident_gib << 30,
                          max_capture_bytes=args.max_capture_gib << 30, chunk_bytes=args.copy_chunk_mib << 20,
                          threshold=args.threshold, progress=progress, configuration=configuration))
            report['backend'] = {'preference': support.blas_preference(), 'loaded_libraries': support.loaded_blas_libraries()}
        report['source_sha256_after'] = source_inventory()
        report['cuda_initialized'] = torch.cuda.is_initialized()
        report['source_files_unchanged_during_run'] = report['source_sha256_after'] == configuration['source_sha256_before']
        if report['status'] in ('PASS', 'CPU_INSPECTED') and not report['source_files_unchanged_during_run']:
            report['status'] = 'SOURCE_CHANGED_DURING_RUN'
        save_report()
    except BaseException as error:
        report.update(status='ERROR', error={'type': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc()},
                      failure_artifact_exists=bool(args.failure_dump and args.failure_dump.is_file()))
        save_report()
        raise
    print(json.dumps({'report': str(args.report), 'status': report['status']}), flush=True)
    return 0 if report['status'] in ('PASS', 'CPU_INSPECTED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
