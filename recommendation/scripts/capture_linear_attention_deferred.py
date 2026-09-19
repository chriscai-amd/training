#!/usr/bin/env python3
"""Retain real layer-1 attention operands/outputs alongside deferred Linear.

The attention trigger is a generous scalar lead, not an arithmetic certificate.
All selected snapshots are taken on device before host flag resolution, and
the original public attention call executes exactly once.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import capture_additional_linear_deferred as base
from nan_backward_boundaries import BoundaryProbeError, _execution_controls, _logical_chunks, _tensors
from nan_backward_training_extended import _argument_layouts

BASE_SHA256 = 'f1d6fbffe1b7f1bb1df66ab6347a8cefb64fabd892dbb5d63ebaaeaea584d81e'
VARIANT = 'linear_and_layer1_attention_deferred_device_capture_v1'
ATTENTION_FORMAT = 'actual_attention_readwrite_device_snapshots_v1'
TARGET_LAYER = '._stu_layers.1'
TRIGGER_FACTOR = 10.0
LIMITS = [
    *(item for item in base.LIMITS if not item.startswith('All three STU layers')),
    'All three STU layers retain the established42 scalar endpoints. Only selected layer1 attention raw tensors are captured; recomputed forward operations and residual addition remain uncaptured.',
    'Layer1 attention adds independent full-storage pre-call reads and initial destinations, plus immediate post-call reads and DQ/DK/DV outputs; packed U/DU padding is preserved but not treated as a written attention output. Initial destination bytes are never fault-scanned.',
    'The DQ trigger is ten times a universal real-arithmetic SiLU derivative bound using actual N, alpha, shapes and input scalar extrema. It is a producer lead only; CPU selected-query references or exact replay oracles must certify the error.',
    'Attention read-input byte comparisons cover complete backing storage, including integer offsets and shared packed UVQK. This does not certify unobserved forward statistics or the actual residual sum.',
    'Use split_capture before the existing Linear replay validator; the attention boundary contains integer sequence metadata that is outside the original Linear format.',
]


def verify_sources():
    inventory = base.verify_sources()
    path = Path(base.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
        raise BoundaryProbeError('Frozen deferred Linear source changed')
    inventory['scripts/' + path.name] = BASE_SHA256
    inventory['scripts/' + Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return inventory


def unique_storage_bytes(value):
    return sum({base._storage_key(tensor): tensor.untyped_storage().nbytes()
                for _, tensor in _tensors(value)}.values())


def storage_aliases(reads, outputs):
    groups = {}
    for role, values in (('read', reads), ('write_only', outputs)):
        for name, tensor in _tensors(values):
            group = groups.setdefault(base._storage_key(tensor), {
                'storage_group': len(groups), 'device': str(tensor.device),
                'storage_nbytes': tensor.untyped_storage().nbytes(),
                'read': [], 'write_only': []})
            group[role].append(name)
    all_groups = list(groups.values())
    return all_groups, [group for group in all_groups if group['read'] and group['write_only']]


def pending_endpoint(queue, scan_index, name):
    scan = queue.scans[scan_index]
    entries = [entry for entry in scan['summaries'] if entry['name'] == name]
    if len(entries) != 1 or not entries[0]['scanned']:
        raise BoundaryProbeError(f'Missing scalar endpoint {scan_index}/{name}')
    entry = entries[0]
    found = [endpoint for pending in queue.pending.values() for candidate, endpoint in pending if candidate is entry]
    if len(found) != 1:
        raise BoundaryProbeError('Scalar endpoint identity not retained in deferred queue')
    endpoint = found[0]
    if endpoint.device.type == 'cuda':
        consumer = torch.cuda.current_stream(endpoint.device)
        producer = queue.streams[str(endpoint.device)][entry['stream_id']]
        if consumer.cuda_stream != producer.cuda_stream:
            event = torch.cuda.Event()
            event.record(producer)
            consumer.wait_event(event)
            queue.events.append(event)
        endpoint.record_stream(consumer)
    return endpoint


def generous_dq_trigger(endpoints, *, rows, value_dim, N, alpha):
    if (any(type(n) is not int or n < 1 for n in (rows, value_dim, N))
            or type(alpha) not in (int, float) or not math.isfinite(alpha)):
        raise BoundaryProbeError('Positive actual shapes/N and finite alpha required')
    if set(endpoints) != {'q', 'k', 'v', 'dout', 'dq'}:
        raise BoundaryProbeError('Unexpected DQ trigger endpoint names')
    device = endpoints['dq'].device
    if any(value.device != device or tuple(value.shape) != (2,) for value in endpoints.values()):
        raise BoundaryProbeError('DQ trigger endpoints must be same-device min/max pairs')
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        magnitudes = {name: value.abs().amax() for name, value in endpoints.items()}
        finite_inputs = torch.stack([torch.isfinite(endpoints[name]).all()
                                     for name in ('q', 'k', 'v', 'dout')]).all()
        # The factor10 makes this a deliberately loose lead. It is not a
        # substitute for FP64 selected-query or valid exact-zero certification.
        real_bound = abs(float(alpha)) * (min(rows, N) / N) * 1.1 * value_dim
        real_bound = real_bound * magnitudes['v'] * magnitudes['dout'] * magnitudes['k']
        threshold = TRIGGER_FACTOR * real_bound
        eligible = finite_inputs & torch.isfinite(threshold)
        exceeds = (~torch.isfinite(magnitudes['dq'])) | (magnitudes['dq'] > threshold)
        mismatch = (eligible & exceeds).to(torch.float32)
    return mismatch, {'input_finite': finite_inputs, 'eligible': eligible,
                      'real_bound': real_bound, 'generous_threshold': threshold,
                      'observed_dq_max_abs': magnitudes['dq']}


def queue_input_integrity(probe, boundary):
    before, after = boundary['inputs_before'], boundary['inputs_after']
    seen, names = set(), []
    for name, tensor in _tensors(before):
        key = base._storage_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        other = after[name]
        probe.snapshots.wait_reads((tensor, other))
        left, right = base._storage_bytes(tensor), base._storage_bytes(other)
        if left.shape != right.shape:
            raise BoundaryProbeError('Attention input backing size changed')
        with torch.no_grad():
            equal = torch.ones((), dtype=torch.bool, device=tensor.device)
            for a, b in zip(_logical_chunks(left, probe.chunk_bytes), _logical_chunks(right, probe.chunk_bytes)):
                equal = equal & torch.all(a[0] == b[0])
            flag = (~equal).to(torch.float32)
        stage = 'attention_layer1_input_integrity_' + name
        scan = probe.scalar_queue.enqueue({'mismatch': flag}, {
            'kind': 'attention_input_integrity', 'stage': stage, 'expected_zero': True,
            'comparison_scope': 'complete backing bytes; one comparison per independent read-input storage',
            'attention_call': boundary['call'], 'attention_layer': boundary['layer']})
        probe.emit('comparison_queued', **scan)
        names.append(name)
    boundary['input_storage_integrity_comparisons'] = names
    return len(names)


class CapturingHSTUProbe(base.ScalarHSTUProbe):
    def __init__(self, model, directory, *, linear_probe, **options):
        self.linear_probe = linear_probe
        super().__init__(model, directory, **options)
        matches = [layer for layer in self.resolved_layer_operations if layer.endswith(TARGET_LAYER)]
        if len(matches) != 1:
            self.close()
            raise BoundaryProbeError('Expected exactly one middle STU layer')
        self.capture_layer = matches[0]

    def _batched_readwrite_monitor(self, operation, function, signature, args, kwargs, writes, owner):
        if operation != 'hstu_attention_bwd' or owner['layer'] != self.capture_layer:
            return super()._batched_readwrite_monitor(operation, function, signature, args, kwargs, writes, owner)
        probe = self.linear_probe
        if 'attention_boundary' in probe.frame:
            raise BoundaryProbeError('Repeated selected attention call in one attempt')
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        reads = {name: value for name, value in bound.arguments.items() if name not in writes}
        if tuple(writes) != ('dq', 'dk', 'dv'):
            raise BoundaryProbeError('Unexpected selected attention destination names')
        outputs = {name: bound.arguments[name] for name in writes}
        q, k, v, g = (reads[name] for name in ('q', 'k', 'v', 'dout'))
        if (q.ndim != 3 or k.shape != q.shape or v.ndim != 3 or v.shape[:2] != q.shape[:2]
                or g.shape != v.shape or min(q.shape) < 1
                or any(value.dtype != torch.bfloat16 for value in (q, k, v, g))):
            raise BoundaryProbeError('Selected attention capture requires compatible BF16 rank3 operands')
        scalars = {name: reads[name] for name in ('N', 'alpha', 'max_attn_len', 'contextual_seq_len', 'enable_tma', 'num_softmax_heads')}
        if scalars['enable_tma'] is not False or scalars['num_softmax_heads'] != 0:
            raise BoundaryProbeError('Selected capture supports ordinary non-TMA SiLU attention')
        # Joint observations preserve unexpected aliases between role groups.
        # Pre/post observations remain independent from each other and live data.
        required = 2 * unique_storage_bytes((reads, outputs))
        if probe.snapshots.bytes + required > probe.max_bytes:
            raise BoundaryProbeError(f'Attention pre/post complete snapshots need {required} bytes; snapshot budget would be exceeded before the original call')
        aliases, cross_role_aliases = storage_aliases(reads, outputs)
        before = probe.copy({'inputs_before': reads, 'initial_destinations': outputs})
        boundary = {'format': ATTENTION_FORMAT, 'format_version': 1, 'layer': owner['layer'],
                    'operation': operation, 'attempt': dict(probe.attempt), 'scalar_arguments': scalars,
                    'argument_roles': {'read': list(reads), 'write_only': list(writes)},
                    'argument_layouts': _argument_layouts(bound.arguments),
                    'input_layouts_before': _argument_layouts(reads),
                    'initial_destination_layouts': _argument_layouts(outputs),
                    'output_layouts': _argument_layouts(outputs),
                    'execution_controls_before': _execution_controls(),
                    'input_snapshot_pristine': True, 'initial_destination_bytes_retained': True,
                    'destination_initial_values_scanned': False,
                    'argument_storage_alias_groups': aliases,
                    'read_write_storage_aliases': cross_role_aliases,
                    'read_write_storages_disjoint': not cross_role_aliases,
                    'snapshot_ordering': 'Joint pre-call read/initial-destination copy and joint immediate post-call read/output copy on actual current autograd stream; aliases preserved within each observation and source dependencies joined before final resolution',
                    'snapshot_budget_bytes': required, 'operation_executed': False,
                    **before}
        probe.frame['attention_boundary'] = boundary
        before_index = len(probe.scalar_queue.scans)
        actual_calls = 0

        def execute(*actual_args, **actual_kwargs):
            nonlocal actual_calls
            actual_calls += 1
            if actual_calls != 1:
                raise BoundaryProbeError('Selected original attention producer was rerun')
            result = function(*actual_args, **actual_kwargs)
            boundary['operation_executed'] = True
            boundary.update(probe.copy({'inputs_after': reads, 'outputs': outputs}))
            boundary['input_layouts_after'] = _argument_layouts(reads)
            boundary['execution_controls_after'] = _execution_controls()
            boundary['post_call_copies_enqueued_time'] = time.time()
            return result

        result = super()._batched_readwrite_monitor(operation, execute, signature, args, kwargs, writes, owner)
        after_index = len(probe.scalar_queue.scans) - 1
        before_scan, after_scan = (probe.scalar_queue.scans[index] for index in (before_index, after_index))
        if (actual_calls != 1 or after_index != before_index + 1 or before_scan['phase'] != 'before'
                or after_scan['phase'] != 'after' or before_scan['call'] != after_scan['call']):
            raise BoundaryProbeError('Selected attention did not retain one actual before/after pair')
        boundary.update(call=before_scan['call'], before_scan_index=before_index,
                        after_scan_index=after_index, actual_call_count=actual_calls)
        count = queue_input_integrity(probe, boundary)
        if cross_role_aliases:
            scan = probe.scalar_queue.enqueue({'mismatch': torch.ones((), dtype=torch.float32, device=q.device)}, {
                'kind': 'attention_alias', 'stage': 'attention_layer1_read_write_storage_alias', 'expected_zero': True,
                'attention_call': boundary['call'], 'attention_layer': boundary['layer'],
                'interpretation': 'Unexpected shared backing storage between read and write-only arguments; retained joint observations preserve actual aliases'})
            probe.emit('comparison_queued', **scan)
            count += 1
        endpoints = {name: pending_endpoint(probe.scalar_queue, before_index, name) for name in ('q', 'k', 'v', 'dout')}
        endpoints['dq'] = pending_endpoint(probe.scalar_queue, after_index, 'dq')
        flag, metrics = generous_dq_trigger(endpoints, rows=q.shape[0], value_dim=v.shape[2], N=scalars['N'], alpha=scalars['alpha'])
        boundary['trigger_metrics'] = metrics
        boundary['trigger_policy'] = {'factor': TRIGGER_FACTOR, 'silu_derivative_envelope': 1.1,
            'formula': 'factor * abs(alpha) * min(total_rows,N)/N * 1.1 * value_dim * Vmax * DOutmax * Kmax',
            'interpretation': 'Generous scalar producer lead only; selected-query CPU reference or exact replay oracle required',
            'ineligible': 'Nonfinite read inputs or nonfinite threshold do not establish a finite-input bound; ordinary endpoint nonfinite flags still apply'}
        scan = probe.scalar_queue.enqueue({'mismatch': flag}, {
            'kind': 'attention_trigger', 'stage': 'attention_layer1_DQ_generous_bound', 'expected_zero': True,
            'attention_call': boundary['call'], 'attention_layer': boundary['layer'], 'factor': TRIGGER_FACTOR})
        probe.emit('comparison_queued', **scan)
        boundary['extra_deferred_scan_count'] = count + 1
        return result


class LinearAttentionProbe(base.DeferredAdditionalLinearProbe):
    def __init__(self, model, directory, **options):
        if options.get('monitor_hstu', True) is not True:
            raise ValueError('Attention capture requires the all-layer monitor')
        def factory(model, directory, **hstu_options):
            return CapturingHSTUProbe(model, directory, linear_probe=self, **hstu_options)
        base._scoped(base.DeferredAdditionalLinearProbe.__init__, verify_sources=verify_sources,
                     ScalarHSTUProbe=factory, VARIANT=VARIANT, LIMITS=LIMITS)(self, model, directory, **options)

    def after_backward_return(self):
        boundary = (self.frame or {}).get('attention_boundary')
        if not boundary or boundary.get('actual_call_count') != 1:
            raise BoundaryProbeError('Missing selected original attention capture at backward return')
        expected = 65 + boundary['extra_deferred_scan_count']
        # The frozen base adds four final scans after this entry point.
        if len(self.scalar_queue.scans) != expected - 4:
            raise BoundaryProbeError('Unexpected deferred capture scan coverage')
        self.frame['metadata']['expected_deferred_scans'] = expected
        return super().after_backward_return()

    def save_frame(self, trigger):
        return base._scoped(base.DeferredAdditionalLinearProbe.save_frame, VARIANT=VARIANT, LIMITS=LIMITS)(self, trigger)


def split_capture(payload):
    if payload.get('capture_variant') != VARIANT or payload.get('attention_boundary', {}).get('format') != ATTENTION_FORMAT:
        raise ValueError('Requires a deferred Linear/attention capture')
    return {key: value for key, value in payload.items() if key != 'attention_boundary'}, payload['attention_boundary']


def _archive_sources(directory, context):
    context['source_sha256'].update(verify_sources())
    return base.base._archive_sources(directory, context)


def configure_environment(options, environment):
    verify_sources()
    base.configure_environment(options, environment)
    recorded = json.loads(environment['NAN_TRAINING_OPTIONS'])
    recorded.update(monitor_variant=VARIANT, attention_capture_layer=1,
                    attention_DQ_trigger_factor=TRIGGER_FACTOR,
                    attention_capture_format=ATTENTION_FORMAT)
    environment['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)


def diagnostic_worker(*args, **kwargs):
    options = json.loads(os.environ['NAN_TRAINING_OPTIONS'])
    if (options.get('attention_capture_layer') != 1 or options.get('attention_DQ_trigger_factor') != TRIGGER_FACTOR
            or options.get('attention_capture_format') != ATTENTION_FORMAT):
        raise ValueError('Worker lost selected attention capture policy')
    return base._scoped(base.diagnostic_worker, verify_sources=verify_sources, VARIANT=VARIANT,
                        LIMITS=LIMITS, DeferredAdditionalLinearProbe=LinearAttentionProbe,
                        _archive_sources=_archive_sources)(*args, **kwargs)


def main(argv=None):
    return base._scoped(base.frozen.main, __doc__=__doc__, configure_environment=configure_environment,
                        diagnostic_worker=diagnostic_worker)(argv)


if __name__ == '__main__':
    main()
