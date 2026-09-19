#!/usr/bin/env python3
"""Inspect or replay trusted training attention/SiLU read/write captures.

Default inspection is CPU-only. --gpu requires the capture context. Output
faults from monitor mode contain post-call inputs and require the explicit
--allow-postcall-inputs flag; these runs are conditional experiments, not a
reproduction of the original call's pristine inputs or runtime history.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import replay_backward_boundary as common
import replay_hstu_output_boundary as helpers


FORMAT = 'training_backward_readwrite_v1'
ATTENTION = 'hstu_attention_bwd'
SILU = 'hstu_silu_bwd'
ARGUMENTS = {
    ATTENTION: ('dout', 'q', 'k', 'v', 'dq', 'dk', 'dv', 'seq_offsets', 'num_targets',
                'N', 'alpha', 'max_attn_len', 'contextual_seq_len', 'sort_by_length_indices',
                'enable_tma', 'num_softmax_heads'),
    SILU: ('grad_output', 'self', 'grad_input'),
}
WRITES = {ATTENTION: ('dq', 'dk', 'dv'), SILU: ('grad_input',)}
OUTPUTS = {ATTENTION: ('dq', 'dk', 'dv'), SILU: ('du',)}
SCALARS = ('N', 'alpha', 'max_attn_len', 'contextual_seq_len', 'enable_tma', 'num_softmax_heads')
LIMITS = [
    'Full argument storage bytes/layouts/aliases are restored, but original addresses, allocator history, compiled objects, autotuner history and driver/hardware state are not reconstructed.',
    'Calls use the current production public API, including its pre-hooks and contiguous-copy behavior; source and stack checks do not establish original compiled-kernel identity.',
    'An output-fault monitor capture has post-call read inputs and destinations; explicit conditional replay does not establish the original call inputs or initial destination bytes.',
    'Read-logical hashes are compared after each call. Full backing hashes can change because destinations are writable and may share storage with other arguments; padding outside destinations is not separately certified.',
    'Output comparison uses all logical bytes and anomaly counts, not an independent numerical attention oracle. No strict history-DQ-zero assumption is made.',
]


def _signature(operation):
    return inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY if name == 'grad_input'
                          else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ARGUMENTS[operation]
    ])


def output_digest(tensor, chunk_bytes, threshold):
    """Bounded logical hash, including singleton slices with non-unit stride.

    The frozen common digest uses contiguous(), which may retain a stride>1 for
    a singleton chunk; a dtype-to-byte view then fails. Explicit flat storage
    keeps this new adapter correct without changing frozen helper sources.
    """
    torch = common._torch()
    digest, nonfinite, extreme, nonzero, largest = hashlib.sha256(), 0, 0, 0, 0.0
    for chunk in common._logical_chunks(tensor, max(1, chunk_bytes // 32)):
        flat = torch.empty(chunk.numel(), dtype=chunk.dtype, device='cpu')
        flat.copy_(chunk.detach().to('cpu').resolve_conj().resolve_neg().reshape(-1))
        digest.update(flat.view(torch.uint8).numpy().tobytes())
        values = flat.to(torch.float64)
        finite = torch.isfinite(values)
        nonfinite += int((~finite).sum())
        extreme += int((finite & (values.abs() > threshold)).sum())
        nonzero += int((values != 0).sum())
        if bool(finite.any()):
            largest = max(largest, float(values[finite].abs().max()))
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
            'logical_sha256': digest.hexdigest(), 'nonfinite_count': nonfinite,
            'extreme_count': extreme, 'nonzero_count_including_nonfinite': nonzero,
            'max_abs_finite': largest}


def argument_digests(values, *, chunk_bytes, threshold):
    """Hash all raw backing bytes and logical arguments with the safe digest."""
    torch = common._torch()
    logical, identities, raw = {}, {}, []
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            logical[name] = {'scalar': value}
            continue
        storage = value.untyped_storage()
        identity = storage._cdata
        if identity not in identities:
            identities[identity] = len(raw)
            view = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
            digest = hashlib.sha256()
            for start in range(0, storage.nbytes(), chunk_bytes):
                digest.update(view[start:start + chunk_bytes].to('cpu').numpy().tobytes())
            raw.append({'bytes': storage.nbytes(), 'sha256': digest.hexdigest()})
        logical[name] = {**output_digest(value, chunk_bytes, threshold),
                         'storage_offset': value.storage_offset(), 'storage_index': identities[identity],
                         'is_conj': value.is_conj(), 'is_neg': value.is_neg()}
    return {'inputs': logical, 'raw_storages': raw}


def validate_capture(payload, *, allow_postcall_inputs=False):
    """Validate evidence and bind arguments without importing a GPU operation."""
    if payload.get('format') != FORMAT or payload.get('format_version') != 1:
        raise ValueError('Requires training_backward_readwrite_v1 format version 1')
    operation, stage, mode = (payload.get(key) for key in ('operation', 'stage', 'probe_mode'))
    if operation not in ARGUMENTS or stage not in ('input', 'output', 'finite'):
        raise ValueError('Unsupported read/write operation or stage')
    if mode not in ('pristine', 'monitor_on_anomaly'):
        raise ValueError('Unsupported capture probe_mode')
    event = payload.get('event', {})
    if event.get('operation') != operation or ('layer' in payload and event.get('layer') != payload['layer']):
        raise ValueError('Capture payload and event operation/layer identities disagree')
    pristine = payload.get('input_snapshot_pristine')
    expected_pristine = mode == 'pristine' or stage == 'input'
    for record in (payload, event):
        if (record.get('input_snapshot_pristine') is not expected_pristine
                or record.get('initial_destination_bytes_retained') is not expected_pristine):
            raise ValueError('Inconsistent input snapshot or destination provenance markers')
    if event.get('operation_executed') is not (stage != 'input'):
        raise ValueError('Inconsistent operation-execution provenance')
    if not pristine and not allow_postcall_inputs:
        raise ValueError('Post-call input capture requires --allow-postcall-inputs; replay is conditional')
    values = dict(_signature(operation).bind(*payload.get('args', ()), **payload.get('kwargs', {})).arguments)
    reads = tuple(name for name in values if name not in WRITES[operation])
    roles = event.get('argument_roles', {})
    if (roles.get('write_only') != list(WRITES[operation]) or roles.get('read') != list(reads)
            or event.get('destination_initial_values_scanned') is not False):
        raise ValueError('Capture argument roles do not match the public operation')
    controls = payload.get('execution_controls')
    if not isinstance(controls, dict) or controls != event.get('execution_controls'):
        raise ValueError('Missing or inconsistent operation execution controls')
    torch = common._torch()
    layouts = event.get('argument_layouts', {})
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    if set(layouts) != set(tensors):
        raise ValueError('Capture argument-layout names differ from tensor arguments')
    groups = {}
    reverse_groups = {}
    for name, value in tensors.items():
        if (value.device.type != 'cpu' or value.layout != torch.strided
                or value.is_quantized or value.is_complex()):
            raise ValueError(f'{name} requires a CPU real strided nonquantized tensor')
        spec = layouts[name]
        actual = {'shape': list(value.shape), 'stride': list(value.stride()), 'dtype': str(value.dtype),
                  'storage_offset': value.storage_offset(), 'storage_bytes': value.untyped_storage().nbytes()}
        if any(spec.get(key) != item for key, item in actual.items()):
            raise ValueError(f'{name} tensor layout differs from capture metadata')
        for key, actual_flag in (('is_conj', value.is_conj()), ('is_neg', value.is_neg())):
            if key in spec and spec[key] != actual_flag:
                raise ValueError(f'{name} tensor {key} differs from capture metadata')
        group, identity = spec['storage_group'], value.untyped_storage()._cdata
        if groups.setdefault(group, identity) != identity or reverse_groups.setdefault(identity, group) != group:
            raise ValueError('Argument storage alias groups differ from capture metadata')
    if operation == SILU:
        reference = values['self']
        if not isinstance(reference, torch.Tensor) or not all(isinstance(values[name], torch.Tensor) and values[name].is_floating_point()
                   and values[name].shape == reference.shape and values[name].dtype == reference.dtype
                   for name in ARGUMENTS[SILU]):
            raise ValueError('SiLU gradient, input and destination shapes/dtypes must agree')
    else:
        q, k, v = (values[name] for name in ('q', 'k', 'v'))
        if (not all(isinstance(item, torch.Tensor) and item.ndim == 3 for item in (q, k, v))
                or k.shape != q.shape or v.shape[:2] != q.shape[:2] or min(q.shape[1:]) < 1
                or v.shape[-1] < 1):
            raise ValueError('Attention requires compatible [rows, heads, features] Q/K/V shapes')
        for name, reference in (('q', q), ('k', q), ('v', v), ('dq', q), ('dk', q), ('dv', v), ('dout', v)):
            value = values[name]
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                    or value.shape != reference.shape or value.dtype != q.dtype):
                raise ValueError(f'Attention {name} has an incompatible shape/dtype')
        for name in ('N', 'max_attn_len', 'contextual_seq_len', 'num_softmax_heads'):
            if type(values[name]) is not int or values[name] < (1 if name == 'N' else 0):
                raise ValueError(f'Invalid attention integer scalar {name}')
        if values['num_softmax_heads'] > q.shape[1] or type(values['enable_tma']) is not bool:
            raise ValueError('Invalid attention head count or TMA flag')
        if type(values['alpha']) not in (int, float) or not math.isfinite(values['alpha']):
            raise ValueError('Attention alpha must be a finite ordinary scalar')
        offsets = values['seq_offsets']
        if (not isinstance(offsets, torch.Tensor) or offsets.dtype not in (torch.int32, torch.int64)
                or offsets.ndim != 1 or offsets.numel() < 2):
            raise ValueError('Attention requires a one-dimensional integer sequence-offset tensor')
        lengths = offsets[1:] - offsets[:-1]
        if (int(offsets[0]) != 0 or int(offsets[-1]) != q.shape[0]
                or bool((lengths < 0).any()) or bool((lengths > values['N']).any())):
            raise ValueError('Attention sequence offsets disagree with rows or N')
        count = lengths.numel()
        for name in ('num_targets', 'sort_by_length_indices'):
            value = values[name]
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64) or value.shape != (count,):
                    raise ValueError(f'Invalid attention {name}')
                if name == 'num_targets' and (bool((value < 0).any()) or bool((value > lengths).any())):
                    raise ValueError('Attention target counts exceed sequence lengths')
                if name == 'sort_by_length_indices' and not torch.equal(value.sort().values.to(torch.int64), torch.arange(count)):
                    raise ValueError('Attention sort indices must form a permutation')
        if 'scalar_arguments' in event and event['scalar_arguments'] != {name: values[name] for name in SCALARS}:
            raise ValueError('Attention scalar metadata disagrees with saved call arguments')
    captured = payload.get('outputs')
    if stage == 'input':
        if captured is not None:
            raise ValueError('An unexecuted input capture must not contain operation outputs')
    else:
        if not isinstance(captured, (tuple, list)) or len(captured) != len(WRITES[operation]):
            raise ValueError('Wrong number of captured operation outputs')
        for tensor, name in zip(captured, WRITES[operation]):
            destination = values[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or tensor.layout != torch.strided or tensor.shape != destination.shape
                    or tensor.dtype != destination.dtype):
                raise ValueError('Captured output shape/dtype differs from destination')
            if not pristine and (tensor.untyped_storage()._cdata != destination.untyped_storage()._cdata
                                 or tensor.storage_offset() != destination.storage_offset()
                                 or tensor.stride() != destination.stride()):
                raise ValueError('Post-call output must alias its captured destination')
    provenance = {'input_snapshot_pristine': pristine, 'initial_destination_bytes_retained': pristine,
                  'original_operation_executed': stage != 'input',
                  'replay_interpretation': ('operation replay from captured pre-call arguments' if pristine else
                                            'conditional replay from post-call arguments; original inputs and initial destinations unavailable'),
                  'allow_postcall_inputs': allow_postcall_inputs}
    return values, reads, provenance


def prepare_arguments(payload, *, allow_postcall_inputs=False, attention_zero_oracle=False,
                      chunk_bytes=64 << 20, max_bytes=64 << 30, threshold=1e6):
    """Prepare an explicitly transformed zero oracle in cloned CPU backings."""
    values, reads, provenance = validate_capture(payload, allow_postcall_inputs=allow_postcall_inputs)
    original = argument_digests(values, chunk_bytes=chunk_bytes, threshold=threshold)
    transformation = None
    if attention_zero_oracle:
        if payload['operation'] != ATTENTION or values['num_softmax_heads'] != 0:
            raise ValueError('Attention zero oracle requires attention and num_softmax_heads=0')
        torch = common._torch()
        k = original['inputs']['k']
        if (k['nonfinite_count'] or k['max_abs_finite'] > torch.finfo(torch.float32).max
                or abs(values['alpha']) > torch.finfo(torch.float32).max):
            raise ValueError('Zero oracle requires finite K and alpha representable in FP32')
        fresh, copied = common.restore_raw_tree(values, device='cpu', chunk_bytes=chunk_bytes, max_bytes=max_bytes)
        zeroed = ('q', 'v', 'dout', 'dq', 'dk', 'dv')
        # CPU zero_ touches only each strided logical view. Raw restore preserves
        # every unused packed byte before this explicit transformation.
        for name in zeroed:
            fresh[name].zero_()
        transformed = argument_digests(fresh, chunk_bytes=chunk_bytes, threshold=threshold)
        unchanged = [name for name in values if name not in zeroed]
        if any(original['inputs'][name] != transformed['inputs'][name] for name in unchanged):
            raise ValueError('Zero transformation changed protected K/metadata, possibly through overlapping aliases')
        if any(transformed['inputs'][name]['nonzero_count_including_nonfinite'] for name in zeroed):
            raise ValueError('Zero transformation did not produce numerical zero in every selected logical element')
        transformation = {
            'mode': 'attention_zero_oracle_v1', 'cpu_cloned_storage_bytes': copied,
            'zeroed_logical_arguments': list(zeroed), 'protected_unchanged_arguments': unchanged,
            'original_argument_digests': original, 'transformed_argument_digests': transformed,
            'unused_packed_bytes': 'retained by complete raw clone followed only by CPU zero_ on selected logical views',
            'oracle': 'All DQ/DK/DV values must be numerical zero; either signed zero is accepted. Any nonzero or nonfinite value is a violation.',
            'derivation': 'Q=0 gives scores=0 and SiLU(scores)=0. V=0 and dOut=0 give dScores=0. Hence DQ=dScores*K=0, DK=dScores*Q=0, and DV=SiLU(scores)*dOut=0, with finite K, alpha and N>0.',
            'limits': ['The original attention inputs are deliberately changed; this is a mathematical oracle at the captured shape/layout, not original-call numerical replay.',
                       'Initial destination zeros prevent opaque initialization from explaining a nonzero output, but a kernel that never writes some destinations may pass this oracle.',
                       'Finite K/alpha must remain representable in FP32. No assertion is made about arbitrary unsupported datatype conversions or faulty memory/index/compiler behavior.'],
        }
        provenance = {**provenance, 'original_capture_interpretation': provenance['replay_interpretation'],
                      'arguments_transformed': True,
                      'replay_interpretation': 'attention zero-oracle experiment on explicitly transformed arguments; original capture provenance remains ' + ('pristine' if provenance['input_snapshot_pristine'] else 'conditional post-call')}
        values = fresh
    else:
        provenance = {**provenance, 'arguments_transformed': False}
    return {'values': values, 'reads': reads, 'provenance': provenance,
            'transformation': transformation, 'original_argument_digests': original,
            'prepared_argument_digests': transformed if attention_zero_oracle else original}


def _validate_failure_path(path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if os.path.lexists(path) or os.path.lexists(temporary):
        raise ValueError('Failure dump and temporary file must be new paths')
    return path, temporary


def save_replay_failure(path, *, payload, prepared, fresh, attempt, reasons,
                        chunk_bytes, max_capture_bytes, replay_context=None):
    """Atomically retain prepared pre-call bytes and the faulting post-call union.

    Prepared arguments already live on the CPU and are never passed directly to
    the operation. Only fresh independently restored storages are passed. The
    before-call full hashes bridge these CPU bytes to the actual replay inputs.
    Their provenance is separate from the original training snapshot markers.
    """
    from nan_backward_boundaries import _cpu_copy_tree
    from nan_backward_training_probe import _storage_bytes

    torch = common._torch()
    target, temporary = _validate_failure_path(path)
    values = prepared['values']
    names = OUTPUTS[payload['operation']]
    destinations = tuple(fresh[name] for name in WRITES[payload['operation']])
    post_value = {'arguments': fresh, 'outputs': dict(zip(names, destinations))}
    pre_bytes = _storage_bytes(values)
    post_bytes = _storage_bytes(post_value)
    total = pre_bytes + post_bytes
    if total > max_capture_bytes:
        raise ValueError(f'Failure capture needs {total} bytes (prepared {pre_bytes} + post-call union {post_bytes}); limit {max_capture_bytes}')
    started = time.time()
    post, copied_bytes = _cpu_copy_tree(post_value, chunk_bytes, max_capture_bytes - pre_bytes)
    if copied_bytes != post_bytes:
        raise RuntimeError('Post-call storage copy accounting changed during failure capture')
    pre_digests = argument_digests(values, chunk_bytes=chunk_bytes, threshold=attempt['threshold'])
    post_digests = argument_digests(post['arguments'], chunk_bytes=chunk_bytes, threshold=attempt['threshold'])
    output_digests = {name: output_digest(post['outputs'][name], chunk_bytes, attempt['threshold']) for name in names}
    pre_matches = pre_digests == attempt['argument_digests_before'] == prepared['prepared_argument_digests']
    post_matches = post_digests == attempt['argument_digests_after'] and output_digests == attempt['outputs']
    copied_output_fault = any(item['nonzero_count_including_nonfinite'] for item in output_digests.values()) if attempt['zero_oracle_violation'] is not None else any(item['nonfinite_count'] or item['extreme_count'] for item in output_digests.values())
    copied_read_mutation = any(pre_digests['inputs'][name] != post_digests['inputs'][name] for name in prepared['reads'])
    original_keys = ('format', 'format_version', 'session', 'attempt', 'operation', 'layer', 'stage',
                     'probe_mode', 'input_snapshot_pristine', 'initial_destination_bytes_retained', 'note', 'event')
    capture = {
        'format': 'training_readwrite_replay_failure_v1', 'format_version': 1,
        'operation': payload['operation'], 'layer': payload.get('layer'), 'replay_repeat': attempt['repeat'],
        'failure_reasons': list(reasons), 'replay_context': copy.deepcopy(replay_context),
        'original_training_capture': {key: copy.deepcopy(payload.get(key)) for key in original_keys},
        'replay_provenance': copy.deepcopy(prepared['provenance']),
        'transformation': copy.deepcopy(prepared['transformation']),
        'prepared_pre_call': {
            'args': (), 'kwargs': values,
            'input_snapshot_pristine_for_this_prepared_replay': pre_matches,
            'initial_destination_bytes_retained_for_this_prepared_replay': pre_matches,
            'verification': 'Every logical value, scalar, full backing byte, layout and alias group matched the fresh replay arguments before dispatch; these CPU arguments were not passed to the operation.',
            'matches_actual_replay_pre_call_hashes': pre_matches,
            'matches_original_training_capture_bytes': pre_digests == prepared['original_argument_digests'],
            'argument_digests': pre_digests,
        },
        'post_call': {**post, 'argument_digests': post_digests, 'output_digests': output_digests,
                      'matches_observed_post_call_hashes': post_matches},
        'execution_controls': copy.deepcopy(attempt['execution_controls']),
        'observed_attempt': copy.deepcopy(attempt),
        'capture_budget': {'prepared_pre_call_storage_bytes': pre_bytes,
                           'post_call_argument_output_union_storage_bytes': post_bytes,
                           'total_unique_snapshot_storage_bytes': total,
                           'configured_max_capture_bytes': max_capture_bytes,
                           'accounting': 'Separate pre-call and post-call snapshots; output/argument aliases within post-call union counted once. Prepared CPU storage is serialized directly; post-call storage is copied once.'},
        'trigger_persists_in_cpu_snapshot': (not attempt['output_anomaly'] or copied_output_fault) and (attempt['read_logical_values_unchanged'] or copied_read_mutation),
        'snapshot_started_unix_time': started, 'snapshot_completed_unix_time': time.time(),
        'limits': [*LIMITS,
                   'Pristine prepared replay bytes do not upgrade the original training capture provenance. Zero-oracle inputs are deliberate transformations.',
                   'Pre-call and post-call snapshots retain separate storage identities; aliases within each snapshot are preserved.',
                   'Snapshot verification flags must be checked if copied values differ from device observations.'],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    owned_temporary = False
    try:
        with temporary.open('xb') as stream:
            owned_temporary = True
            torch.save(capture, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if owned_temporary:
            temporary.unlink(missing_ok=True)
        raise
    return {'path': str(target), 'bytes': target.stat().st_size, 'sha256': _file_hash(target),
            'failure_reasons': list(reasons), 'capture_storage_bytes': total,
            'prepared_pre_call_bytes_verified': pre_matches,
            'post_call_snapshot_matches_observed_hashes': post_matches,
            'trigger_persists_in_cpu_snapshot': capture['trigger_persists_in_cpu_snapshot']}


def replay_capture(payload, *, allow_postcall_inputs=False, repeats=3, device='cuda:0',
                   chunk_bytes=64 << 20, max_bytes=64 << 30, threshold=1e6,
                   function=None, progress=None, attention_zero_oracle=False, _prepared=None,
                   failure_dump=None, max_capture_bytes=128 << 30, replay_context=None):
    """Replay fresh whole backing storages; injection/device support CPU tests."""
    if repeats < 1 or chunk_bytes < 1 or max_bytes < 1 or max_capture_bytes < 1 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Replay limits and finite threshold must be positive')
    prepared = _prepared or prepare_arguments(payload, allow_postcall_inputs=allow_postcall_inputs,
                                              attention_zero_oracle=attention_zero_oracle, chunk_bytes=chunk_bytes,
                                              max_bytes=max_bytes, threshold=threshold)
    values, reads, provenance = (prepared[key] for key in ('values', 'reads', 'provenance'))
    if failure_dump is not None:
        _validate_failure_path(failure_dump)
        required_capture_bytes = 2 * sum(item['bytes'] for item in prepared['prepared_argument_digests']['raw_storages'])
        if required_capture_bytes > max_capture_bytes:
            raise ValueError(f'Failure capture needs {required_capture_bytes} bytes for prepared and post-call snapshots; limit {max_capture_bytes}')
    torch = common._torch()
    operation = payload['operation']
    injected_function = function is not None
    if function is None:
        if operation == ATTENTION:
            function = importlib.import_module('generative_recommenders.ops.triton.triton_hstu_attention').triton_hstu_attention_bwd
        else:
            function = torch.ops.aten.silu_backward
    synchronize = torch.cuda.synchronize if str(device).startswith('cuda') else lambda: None
    saved = prepared['prepared_argument_digests']
    captured = payload.get('outputs')
    expected = [output_digest(item, chunk_bytes, threshold) for item in captured] if captured is not None else None
    attempts, first = [], None

    def notify(phase, repeat):
        if progress:
            progress({'phase': phase, 'repeat': repeat, 'attempts': list(attempts), 'provenance': provenance,
                      'transformation': prepared['transformation']})

    for repeat in range(repeats):
        notify('restore_arguments', repeat)
        fresh, size = common.restore_raw_tree(values, device=device, chunk_bytes=chunk_bytes, max_bytes=max_bytes)
        synchronize()
        before = argument_digests(fresh, chunk_bytes=chunk_bytes, threshold=threshold)
        if before != saved:
            raise RuntimeError('Restored argument logical/backing hashes or layouts differ from capture')
        destination_layouts = {name: (fresh[name].data_ptr(), fresh[name].untyped_storage()._cdata,
                                      tuple(fresh[name].shape), tuple(fresh[name].stride()), fresh[name].storage_offset())
                               for name in WRITES[operation]}
        notify('execute_operation', repeat)
        started = time.monotonic()
        with helpers.restored_execution_controls(payload['execution_controls']) as controls, torch.no_grad():
            result = function(**fresh)
        synchronize()
        seconds = time.monotonic() - started
        if operation == ATTENTION and result is not None:
            raise RuntimeError('Attention public API unexpectedly returned values')
        if operation == SILU and (not isinstance(result, torch.Tensor) or result.data_ptr() != fresh['grad_input'].data_ptr()):
            raise RuntimeError('SiLU public API did not return its original destination')
        for name, expected_layout in destination_layouts.items():
            value = fresh[name]
            actual = (value.data_ptr(), value.untyped_storage()._cdata, tuple(value.shape), tuple(value.stride()), value.storage_offset())
            if actual != expected_layout:
                raise RuntimeError('Operation changed a destination allocation or layout')
        notify('inspect_results', repeat)
        after = argument_digests(fresh, chunk_bytes=chunk_bytes, threshold=threshold)
        observed = [output_digest(fresh[name], chunk_bytes, threshold) for name in WRITES[operation]]
        read_unchanged = all(before['inputs'][name] == after['inputs'][name] for name in reads)
        oracle_violation = attention_zero_oracle and any(item['nonzero_count_including_nonfinite'] for item in observed)
        attempts.append({'repeat': repeat, **provenance, 'fresh_argument_storage_bytes': size,
                         'threshold': threshold, 'device': str(device), 'function_injected': injected_function,
                         'matches_prepared_argument_bytes_and_layouts': True,
                         'matches_original_capture_argument_bytes_and_layouts': before == prepared['original_argument_digests'],
                         'argument_digests_before': before, 'argument_digests_after': after,
                         'read_logical_values_unchanged': read_unchanged,
                         'all_argument_backing_bytes_unchanged': before['raw_storages'] == after['raw_storages'],
                         'execution_controls': controls, 'operation_seconds': seconds,
                         'outputs': dict(zip(OUTPUTS[operation], observed)), 'captured_output_digests': expected,
                         'matches_captured_logical_bytes': [helpers.same_digest(a, b) for a, b in zip(observed, expected)] if expected is not None and not attention_zero_oracle else None,
                         'matches_first_repeat_logical_bytes': [helpers.same_digest(a, b) for a, b in zip(observed, first)] if first is not None else None,
                         'output_anomaly': bool(oracle_violation) if attention_zero_oracle else any(item['nonfinite_count'] or item['extreme_count'] for item in observed),
                         'zero_oracle_violation': bool(oracle_violation) if attention_zero_oracle else None})
        fault = attempts[-1]['output_anomaly'] or not read_unchanged
        if failure_dump is not None and fault:
            reasons = (['zero_oracle_violation'] if oracle_violation else
                       ['numerical_output_fault'] if attempts[-1]['output_anomaly'] else [])
            if not read_unchanged:
                reasons.append('read_input_mutation')
            notify('save_failure_dump', repeat)
            attempts[-1]['failure_dump'] = save_replay_failure(
                failure_dump, payload=payload, prepared=prepared, fresh=fresh, attempt=attempts[-1],
                reasons=reasons, chunk_bytes=chunk_bytes, max_capture_bytes=max_capture_bytes,
                replay_context=replay_context)
            notify('failure_dump_complete', repeat)
        first = observed if first is None else first
        del fresh, result
        notify('complete_repeat', repeat)
        if not read_unchanged:
            raise RuntimeError('Operation changed captured read-input logical values; result retained in progress report')
        if oracle_violation:
            notify('zero_oracle_violation', repeat)
            break
        if failure_dump is not None and fault:
            break
    return attempts


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--context', type=Path)
    parser.add_argument('--allow-postcall-inputs', action='store_true')
    parser.add_argument('--allow-source-change', action='store_true')
    parser.add_argument('--attention-vgpr-cap', type=int)
    parser.add_argument('--attention-zero-oracle', action='store_true',
                        help='Explicitly zero cloned Q/V/dOut and initial destinations; require every output to be numerical zero')
    parser.add_argument('--failure-dump', type=Path,
                        help='New path for complete prepared inputs and faulting outputs; stops on first output fault or read mutation')
    parser.add_argument('--max-capture-gib', type=int, default=128,
                        help='Combined unique storage budget for prepared pre-call and post-call snapshots (default 128 GiB)')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chunk-mib', type=int, default=64)
    parser.add_argument('--max-input-gib', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=1e6)
    args = parser.parse_args(argv)
    if min(args.repeats, args.chunk_mib, args.max_input_gib, args.max_capture_gib, args.cpu_threads) < 1 or not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error('limits and finite threshold must be positive')
    if args.report.exists() or args.report.resolve() == args.dump.resolve():
        parser.error('--report must be a new path distinct from the capture')
    report_temporary = args.report.with_name(args.report.name + '.tmp')
    if os.path.lexists(args.report) or os.path.lexists(report_temporary):
        parser.error('Report and its temporary file must be new paths')
    if args.gpu and args.context is None:
        parser.error('--gpu requires the original capture --context')
    if args.failure_dump is not None:
        if not args.gpu:
            parser.error('--failure-dump requires --gpu; CPU inspection does not execute an operation')
        if args.failure_dump.resolve() in {path.resolve() for path in (args.dump, args.report, args.context) if path is not None}:
            parser.error('--failure-dump must be distinct from capture, context and report')
        try:
            _validate_failure_path(args.failure_dump)
        except ValueError as error:
            parser.error(str(error))
    output_paths = [args.report, report_temporary]
    if args.failure_dump is not None:
        output_paths += [args.failure_dump, args.failure_dump.with_name(args.failure_dump.name + '.tmp')]
    resolved_outputs = [path.resolve() for path in output_paths]
    protected_inputs = {path.resolve() for path in (args.dump, args.context) if path is not None}
    if len(set(resolved_outputs)) != len(resolved_outputs) or protected_inputs.intersection(resolved_outputs):
        parser.error('Report/failure targets and temporary paths must be distinct from each other and all inputs')
    if (args.attention_vgpr_cap is not None and args.attention_vgpr_cap < 0
            or not args.gpu and (args.attention_vgpr_cap is not None or args.allow_source_change)):
        parser.error('source/cap comparisons require --gpu and a nonnegative cap')
    context = json.loads(args.context.read_text()) if args.context else None
    root, changed, overrides = Path(__file__).resolve().parents[1], [], {}
    if args.gpu:
        if context.get('world_size') != 1:
            parser.error('requires single-rank capture context')
        helpers.restore_replay_environment(context['environment'])
        if args.attention_vgpr_cap is not None:
            overrides['HSTU_BWD_MAX_VGPR'] = str(args.attention_vgpr_cap)
            os.environ.update(overrides)
        for name, expected in context['source_sha256'].items():
            if name.startswith('generative_recommenders/'):
                path = root / name
                if not path.is_file() or _file_hash(path) != expected:
                    changed.append(name)
        if changed and not args.allow_source_change:
            parser.error(f'captured model source differs: {changed}; comparison requires --allow-source-change')
        sys.path.insert(0, str(root))
    torch = common._torch()
    torch.set_num_threads(args.cpu_threads)
    versions = {'torch': torch.__version__, 'hip': torch.version.hip}
    if args.gpu:
        triton = importlib.import_module('triton')
        versions.update(triton=triton.__version__, triton_distribution=importlib.metadata.version('triton'))
        for key, value in versions.items():
            if context.get(key) != value:
                parser.error(f'stack mismatch for {key}: capture={context.get(key)!r}, replay={value!r}')
    payload = torch.load(args.dump, map_location='cpu', weights_only=False, mmap=True)
    prepared = prepare_arguments(payload, allow_postcall_inputs=args.allow_postcall_inputs,
                                 attention_zero_oracle=args.attention_zero_oracle, chunk_bytes=args.chunk_mib << 20,
                                 max_bytes=args.max_input_gib << 30, threshold=args.threshold)
    values, reads, provenance = (prepared[key] for key in ('values', 'reads', 'provenance'))
    if args.attention_vgpr_cap is not None and payload['operation'] != ATTENTION:
        parser.error('--attention-vgpr-cap is only meaningful for attention captures')
    report = {'format': 'training_readwrite_boundary_replay_v1', 'dump': str(args.dump),
              'dump_sha256': _file_hash(args.dump), 'context': str(args.context) if args.context else None,
              'context_sha256': _file_hash(args.context) if args.context else None,
              'operation': payload['operation'], 'layer': payload.get('layer'), 'attempt': payload.get('attempt'),
              'provenance': provenance, 'limits': LIMITS, 'versions': versions,
              'runner_source_sha256': {path.name: _file_hash(path) for path in
                                       (Path(__file__), Path(common.__file__), Path(helpers.__file__))},
              'transformation': prepared['transformation'],
              'changed_model_source': changed, 'environment_overrides': overrides,
              'captured_environment': context['environment'] if args.gpu else None,
              'failure_dump_requested': str(args.failure_dump) if args.failure_dump else None,
              'failure_capture_budget_bytes': args.max_capture_gib << 30,
              'replay_status': 'CPU inspection; GPU replay not requested',
              'original_argument_digests': prepared['original_argument_digests'],
              'argument_digests': prepared['prepared_argument_digests'],
              'captured_outputs': [output_digest(tensor, args.chunk_mib << 20, args.threshold) for tensor in payload['outputs']] if payload['outputs'] is not None else None}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.report.with_name(args.report.name + '.tmp')
        with temporary.open('w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.report)

    save()
    if args.gpu:
        report['replay_status'] = 'running'

        def progress(state):
            report['progress'] = {key: value for key, value in state.items() if key != 'attempts'}
            report['attempts'] = state['attempts']
            save()

        try:
            report['attempts'] = replay_capture(payload, allow_postcall_inputs=args.allow_postcall_inputs,
                                                repeats=args.repeats, chunk_bytes=args.chunk_mib << 20,
                                                max_bytes=args.max_input_gib << 30, threshold=args.threshold,
                                                progress=progress, attention_zero_oracle=args.attention_zero_oracle,
                                                _prepared=prepared, failure_dump=args.failure_dump,
                                                max_capture_bytes=args.max_capture_gib << 30,
                                                replay_context={key: report[key] for key in ('dump', 'dump_sha256', 'context', 'context_sha256', 'versions', 'runner_source_sha256', 'changed_model_source', 'environment_overrides', 'captured_environment')})
            report['replay_status'] = ('zero_oracle_violation; stopped at first nonzero/nonfinite output'
                                       if any(item['zero_oracle_violation'] for item in report['attempts']) else
                                       'numerical_output_fault; complete failure dump saved and replay stopped'
                                       if any(item.get('failure_dump') for item in report['attempts']) else
                                       'completed; interpret output hashes/anomalies under stated provenance limits')
        except BaseException as error:
            report['replay_status'] = 'failed'
            report['error'] = {'type': type(error).__name__, 'message': str(error)}
            save()
            raise
        save()
    print(json.dumps({'report': str(args.report), 'status': report['replay_status'], 'provenance': provenance}, indent=2))


if __name__ == '__main__':
    main()
