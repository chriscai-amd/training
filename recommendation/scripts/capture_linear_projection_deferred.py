#!/usr/bin/env python3
"""Retain the actual layer-2 projection backward beside deferred Linear.

The selected public function runs once. Complete pre-call input backing stores
and immediate post-call input/output stores survive until the existing deferred
flags are resolved after the real backward return. No producer is replayed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch
import capture_additional_linear_deferred as base
from nan_backward_boundaries import BoundaryProbeError, _execution_controls, _logical_chunks, _pointer, _tensors
from nan_backward_training_extended import _argument_layouts

BASE_SHA256 = 'f1d6fbffe1b7f1bb1df66ab6347a8cefb64fabd892dbb5d63ebaaeaea584d81e'
PROJECTION_SOURCE = 'generative_recommenders/ops/triton/triton_addmm.py'
PROJECTION_SHA256 = '21d1c0d12e44f7e22d2255366538ef389a8af72d9af6ed640783a0b1b76ec71a'
VARIANT = 'linear_and_layer2_projection_deferred_device_capture_v1'
PROJECTION_FORMAT = 'actual_projection_backward_device_snapshots_v1'
TARGET_LAYER = '._stu_layers.2'
LIMITS = [
    *(item for item in base.LIMITS if not item.startswith('All three STU layers')),
    'All three STU layers retain the established 42 scalar endpoints. Layer2 projection backward additionally retains its actual pristine x/w/dz, post-call x/w/dz and raw DX/DW/DB; recomputed forward operations and residual additions remain uncaptured.',
    'The public triton_addmm_bwd backward currently calls native torch.sum and torch.mm; this capture does not assert which BLAS implementation ran or certify actual internal operand reads.',
    'Three input byte-integrity flags compare complete backing stores. Full pre-call input and joint post-call input/output observations preserve aliases within an observation and remain independent across observations.',
    'A first nonfinite/extreme or input-mutation flag saves the already-retained call after the entire backward returns. Output fault location and input continuity remain evidence, not a backend root-cause certificate.',
    'Use split_capture to pass the Linear-only payload to its existing replay validator; the projection frame has its own provenance and schema.',
]


def verify_sources():
    inventory = base.verify_sources()
    path = Path(base.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
        raise BoundaryProbeError('Frozen deferred Linear source changed')
    source = path.parents[1] / PROJECTION_SOURCE
    if hashlib.sha256(source.read_bytes()).hexdigest() != PROJECTION_SHA256:
        raise BoundaryProbeError('Projection implementation changed')
    inventory['scripts/' + path.name] = BASE_SHA256
    inventory[PROJECTION_SOURCE] = PROJECTION_SHA256
    inventory['scripts/' + Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return inventory


def unique_storage_bytes(value):
    return sum({base._storage_key(tensor): tensor.untyped_storage().nbytes()
                for _, tensor in _tensors(value)}.values())


def queue_input_integrity(probe, boundary):
    # Compare each public tensor, including aliases, so all three input names
    # have explicit coverage. All comparisons cover complete backing bytes.
    for name in ('x', 'w', 'dz'):
        left = boundary['inputs_before'][name]
        right = boundary['inputs_after'][name]
        probe.snapshots.wait_reads((left, right))
        left, right = base._storage_bytes(left), base._storage_bytes(right)
        if left.shape != right.shape:
            raise BoundaryProbeError('Projection input backing size changed')
        with torch.no_grad():
            equal = torch.ones((), dtype=torch.bool, device=left.device)
            for a, b in zip(_logical_chunks(left, probe.chunk_bytes), _logical_chunks(right, probe.chunk_bytes)):
                equal = equal & torch.all(a[0] == b[0])
            flag = (~equal).to(torch.float32)
        scan = probe.scalar_queue.enqueue({'mismatch': flag}, {
            'kind': 'projection_input_integrity', 'stage': 'projection_layer2_input_integrity_' + name,
            'expected_zero': True, 'comparison_scope': 'complete backing bytes',
            'projection_call': boundary['call'], 'projection_layer': boundary['layer']})
        probe.emit('comparison_queued', **scan)
    boundary['input_storage_integrity_comparisons'] = ['x', 'w', 'dz']


class CapturingHSTUProbe(base.ScalarHSTUProbe):
    def __init__(self, model, directory, *, linear_probe, **options):
        self.linear_probe = linear_probe
        super().__init__(model, directory, **options)
        matches = [layer for layer in self.resolved_layer_operations if layer.endswith(TARGET_LAYER)]
        if len(matches) != 1:
            self.close()
            raise BoundaryProbeError('Expected exactly one top STU layer')
        self.capture_layer = matches[0]

    def _run_monitor(self, operation, function, signature, args, kwargs):
        if operation != 'triton_addmm_bwd':
            return super()._run_monitor(operation, function, signature, args, kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        owner = self._pointers.get(_pointer(bound.arguments['w']))
        if owner is None or owner['layer'] != self.capture_layer:
            return super()._run_monitor(operation, function, signature, args, kwargs)
        probe = self.linear_probe
        if 'projection_boundary' in probe.frame:
            raise BoundaryProbeError('Repeated selected projection call in one attempt')
        reads = dict(bound.arguments)
        x, w, dz = (reads[name] for name in ('x', 'w', 'dz'))
        if (set(reads) != {'x', 'w', 'dz', 'is_y_1d'} or reads['is_y_1d'] is not True
                or any(t.ndim != 2 or min(t.shape) < 1 or t.dtype != torch.bfloat16 for t in (x, w, dz))
                or x.shape[1] != w.shape[0] or tuple(dz.shape) != (x.shape[0], w.shape[1])
                or any(t.device != x.device for t in (w, dz))):
            raise BoundaryProbeError('Selected projection requires compatible BF16 matrices and a 1D bias')
        # Pinned production source returns newly allocated dense native-mm
        # matrices and a sum vector. A changed/oversized output still fails the
        # snapshot ledger; these are the expected public output allocation sizes.
        expected_output_bytes = (x.numel() + w.numel() + w.shape[1]) * x.element_size()
        required = 2 * unique_storage_bytes(reads) + expected_output_bytes
        if probe.snapshots.bytes + required > probe.max_bytes:
            raise BoundaryProbeError(f'Projection pre/post/output snapshots need {required} bytes; snapshot budget would be exceeded before the original call')
        boundary = {
            'format': PROJECTION_FORMAT, 'format_version': 1,
            'layer': owner['layer'], 'operation': operation, 'attempt': dict(probe.attempt),
            'scalar_arguments': {'is_y_1d': True},
            'argument_roles': {'read': list(reads), 'write_only': []},
            'output_names': ['dx', 'dw', 'db'],
            'monitor_output_names': ['d_normed_x', 'd_uvqk_weight', 'd_uvqk_bias'],
            'argument_layouts': _argument_layouts(reads),
            'execution_controls_before': _execution_controls(),
            'preferred_blas_library_before': str(torch.backends.cuda.preferred_blas_library()),
            'blas_library_environment': {name: os.environ.get(name) for name in
                ('HIPBLASLT_TENSILE_LIBPATH', 'ROCBLAS_TENSILE_LIBPATH')},
            'input_snapshot_pristine': True, 'operation_executed': False,
            'snapshot_budget_bytes': required, 'expected_output_allocation_bytes': expected_output_bytes,
            'snapshot_ordering': 'Pre-call input copies and joint immediate post-call input/output copies on the actual current autograd stream; source dependencies joined before final resolution',
            'inputs_before': probe.copy(reads),
        }
        probe.frame['projection_boundary'] = boundary
        before_index = len(probe.scalar_queue.scans)
        actual_calls = 0

        def execute(*actual_args, **actual_kwargs):
            nonlocal actual_calls
            actual_calls += 1
            if actual_calls != 1:
                raise BoundaryProbeError('Selected original projection producer was rerun')
            result = function(*actual_args, **actual_kwargs)
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                raise BoundaryProbeError('Unexpected projection output structure')
            outputs = dict(zip(('dx', 'dw', 'db'), result))
            expected_shapes = (x.shape, w.shape, (w.shape[1],))
            if any(not isinstance(t, torch.Tensor) or t.shape != shape or t.dtype != x.dtype or t.device != x.device
                   for t, shape in zip(result, expected_shapes)):
                raise BoundaryProbeError('Unexpected projection output shape/dtype/device')
            boundary['operation_executed'] = True
            boundary.update(probe.copy({'inputs_after': reads, 'outputs': outputs}))
            boundary['post_call_argument_output_layouts'] = _argument_layouts({'inputs': reads, 'outputs': outputs})
            boundary['execution_controls_after'] = _execution_controls()
            boundary['preferred_blas_library_after'] = str(torch.backends.cuda.preferred_blas_library())
            boundary['post_call_copies_enqueued_time'] = time.time()
            return result

        result = super()._run_monitor(operation, execute, signature, args, kwargs)
        after_index = len(probe.scalar_queue.scans) - 1
        before_scan, after_scan = (probe.scalar_queue.scans[index] for index in (before_index, after_index))
        if (actual_calls != 1 or after_index != before_index + 1 or before_scan['phase'] != 'before'
                or after_scan['phase'] != 'after' or before_scan['call'] != after_scan['call']):
            raise BoundaryProbeError('Selected projection lost its actual before/after pair')
        boundary.update(call=before_scan['call'], before_scan_index=before_index,
                        after_scan_index=after_index, actual_call_count=actual_calls)
        queue_input_integrity(probe, boundary)
        return result


class LinearProjectionProbe(base.DeferredAdditionalLinearProbe):
    def __init__(self, model, directory, **options):
        if options.get('monitor_hstu', True) is not True:
            raise ValueError('Projection capture requires the all-layer monitor')
        def factory(model, directory, **hstu_options):
            return CapturingHSTUProbe(model, directory, linear_probe=self, **hstu_options)
        base._scoped(base.DeferredAdditionalLinearProbe.__init__, verify_sources=verify_sources,
                     ScalarHSTUProbe=factory, VARIANT=VARIANT, LIMITS=LIMITS)(self, model, directory, **options)

    def after_backward_return(self):
        boundary = (self.frame or {}).get('projection_boundary')
        if not boundary or boundary.get('actual_call_count') != 1:
            raise BoundaryProbeError('Missing selected original projection capture at backward return')
        # Frozen base adds four final scans after this entry point.
        if len(self.scalar_queue.scans) != 68 - 4:
            raise BoundaryProbeError('Unexpected deferred projection scan coverage')
        self.frame['metadata']['expected_deferred_scans'] = 68
        return super().after_backward_return()

    def save_frame(self, trigger):
        return base._scoped(base.DeferredAdditionalLinearProbe.save_frame, VARIANT=VARIANT, LIMITS=LIMITS)(self, trigger)


def split_capture(payload):
    if payload.get('capture_variant') != VARIANT or payload.get('projection_boundary', {}).get('format') != PROJECTION_FORMAT:
        raise ValueError('Requires a deferred Linear/projection capture')
    return {key: value for key, value in payload.items() if key != 'projection_boundary'}, payload['projection_boundary']


def _archive_sources(directory, context):
    context['source_sha256'].update(verify_sources())
    return base.base._archive_sources(directory, context)


def configure_environment(options, environment):
    verify_sources()
    base.configure_environment(options, environment)
    recorded = json.loads(environment['NAN_TRAINING_OPTIONS'])
    recorded.update(monitor_variant=VARIANT, projection_capture_layer=2,
                    projection_capture_format=PROJECTION_FORMAT)
    environment['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)


def diagnostic_worker(*args, **kwargs):
    options = json.loads(os.environ['NAN_TRAINING_OPTIONS'])
    if options.get('projection_capture_layer') != 2 or options.get('projection_capture_format') != PROJECTION_FORMAT:
        raise ValueError('Worker lost selected projection capture policy')
    return base._scoped(base.diagnostic_worker, verify_sources=verify_sources, VARIANT=VARIANT,
                        LIMITS=LIMITS, DeferredAdditionalLinearProbe=LinearProjectionProbe,
                        _archive_sources=_archive_sources)(*args, **kwargs)


def main(argv=None):
    return base._scoped(base.frozen.main, __doc__=__doc__, configure_environment=configure_environment,
                        diagnostic_worker=diagnostic_worker)(argv)


if __name__ == '__main__':
    main()
