#!/usr/bin/env python3
"""Retain the actual deferred scalar stages beside mitigated projection capture.

The frozen queue still produces each endpoint and flushes once. At that flush,
retain references to its endpoints, its actual stack result and the exact rows
returned by its existing reader. The established save path copies these only on
its first flagged scan or an explicitly selected finite step.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import capture_linear_projection_mitigated as mitigated
import nan_backward_training_deferred as scalar

base = mitigated.base
VARIANT = mitigated.VARIANT
DIAGNOSTICS_FORMAT = 'actual_deferred_scalar_queue_stages_v1'
POLICY = {
    'format': DIAGNOSTICS_FORMAT,
    'endpoint_capture': 'References to the actual frozen pending FP64 endpoints before flush',
    'batch_capture': 'Reference to the actual stack passed to the existing reader',
    'rows_capture': 'Exact Python rows returned by that reader before summary conversion',
    'mapping': 'Source device, device batch row, enqueue index and summary index',
    'enqueue': 'Inherited frozen enqueue; no additional reductions',
    'flush': 'One call to the frozen flush; one existing reader call per nonempty device group',
    'save': 'Existing first flagged scan or selected finite step after actual backward return',
    'boundary_readbacks_added': 0,
    'stream_joins_added': 0,
}
LIMITS = [*mitigated.LIMITS,
    'Scalar endpoints and batches are retained references, not new boundary snapshots. Their device bytes are observed by the existing save path after the original flush; exact CPU rows preserve what that flush returned.',
    'Retaining tiny scalar storages and batches extends their lifetime. No extra reductions, stacks, stream joins or boundary host readbacks are introduced.',
    'The original scan order, flag rules and first flagged trigger are unchanged. This capture alone does not detect every finite scalar mismatch or identify a faulty reduction, stack, transfer or writer.',
]


def verify_sources():
    inventory = mitigated.verify_sources()
    path = Path(__file__).resolve()
    inventory['scripts/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inventory


class DiagnosticScalarQueue(scalar.DeferredScalarQueue):
    """Keep original enqueue and scope the reader override to this flush only."""
    def __init__(self, **options):
        super().__init__(**options)
        self.diagnostics = None

    def flush(self):
        if self.flushed:
            # Let the original enforce its one-flush lifecycle.
            return super().flush()
        positions = {id(entry): (scan['enqueue_index'], index)
                     for scan in self.scans
                     for index, entry in enumerate(scan['summaries'])}
        groups = []
        for device, pending in self.pending.items():
            mapping = []
            for row, (entry, endpoint) in enumerate(pending):
                enqueue, summary = positions[id(entry)]
                mapping.append({'batch_row_index': row, 'enqueue_index': enqueue,
                                'summary_index': summary, 'name': entry['name']})
            groups.append({'source_device': device, 'row_count': len(pending),
                           'row_mapping': mapping,
                           'queued_entries': copy.deepcopy([entry for entry, _ in pending]),
                           'endpoint_tensors': [endpoint for _, endpoint in pending]})
        self.diagnostics = {'format': DIAGNOSTICS_FORMAT, 'policy': copy.deepcopy(POLICY),
                            'groups': groups, 'reader_calls': 0, 'flush_complete': False}
        original_reader = scalar._read_endpoint_batch

        def reader(batch):
            index = self.diagnostics['reader_calls']
            self.diagnostics['reader_calls'] += 1
            if index >= len(groups):
                raise base.BoundaryProbeError('Unexpected extra deferred reader call')
            group = groups[index]
            if str(batch.device) != group['source_device'] or tuple(batch.shape) != (group['row_count'], 2):
                raise base.BoundaryProbeError('Deferred batch lost its pending row mapping')
            group['batch_tensor'] = batch
            rows = original_reader(batch)
            group['returned_rows'] = rows
            return rows

        result = base._scoped(scalar.DeferredScalarQueue.flush,
                              _read_endpoint_batch=reader)(self)
        if self.diagnostics['reader_calls'] != len(groups):
            raise base.BoundaryProbeError('Incomplete deferred reader coverage')
        self.diagnostics['native_resolved_scans'] = copy.deepcopy(result)
        self.diagnostics['flush_complete'] = True
        return result

    def release_diagnostics(self):
        self.diagnostics = None


class DiagnosticLinearProjectionProbe(mitigated.ProjectionProbe):
    def __init__(self, *args, **kwargs):
        base._scoped(mitigated.ProjectionProbe.__init__, verify_sources=verify_sources,
                     VARIANT=VARIANT, LIMITS=LIMITS)(self, *args, **kwargs)

    def set_attempt(self, mode, repeat, step):
        return base._scoped(base.DeferredAdditionalLinearProbe.set_attempt,
                            DeferredScalarQueue=DiagnosticScalarQueue)(self, mode, repeat, step)

    def after_backward_return(self):
        try:
            return super().after_backward_return()
        finally:
            if self.frame is None and self.scalar_queue is not None:
                self.scalar_queue.release_diagnostics()

    def save_frame(self, trigger):
        diagnostics = self.scalar_queue.diagnostics
        if diagnostics is None or not diagnostics['flush_complete']:
            raise base.BoundaryProbeError('Cannot save incomplete deferred scalar stages')
        self.frame['scalar_queue_diagnostics'] = diagnostics
        self.frame['metadata']['scalar_queue_diagnostics_policy'] = copy.deepcopy(POLICY)
        return base._scoped(mitigated.ProjectionProbe.save_frame,
                            VARIANT=VARIANT, LIMITS=LIMITS)(self, trigger)

    def close(self):
        queue = getattr(self, 'scalar_queue', None)
        if isinstance(queue, DiagnosticScalarQueue):
            queue.release_diagnostics()
        return super().close()


# The same public factory name as the mitigated launcher, for scoped composition.
ProjectionProbe = DiagnosticLinearProjectionProbe


def split_capture(payload):
    if (payload.get('capture_variant') != VARIANT
            or payload.get('scalar_queue_diagnostics', {}).get('format') != DIAGNOSTICS_FORMAT
            or payload.get('metadata', {}).get('scalar_queue_diagnostics_policy') != POLICY):
        raise ValueError('Requires the actual deferred scalar stages capture')
    normalized = dict(payload, capture_variant=mitigated.VARIANT)
    return mitigated.split_capture(normalized)


def configure_environment(options, environment):
    base._scoped(mitigated.configure_environment, verify_sources=verify_sources,
                 VARIANT=VARIANT)(options, environment)
    recorded = json.loads(environment['NAN_TRAINING_OPTIONS'])
    recorded['scalar_queue_diagnostics_policy'] = POLICY
    environment['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)


def validate_worker(options, environment):
    base._scoped(mitigated.validate_worker, verify_sources=verify_sources,
                 VARIANT=VARIANT)(options, environment)
    if options.get('scalar_queue_diagnostics_policy') != POLICY:
        raise ValueError('Worker lost deferred scalar stages policy')


def _archive_sources(directory, context):
    context['source_sha256'].update(verify_sources())
    return mitigated.training._archive_sources(directory, context)


def diagnostic_worker(*args, **kwargs):
    return base._scoped(mitigated.diagnostic_worker, ProjectionProbe=ProjectionProbe,
                        validate_worker=validate_worker, _archive_sources=_archive_sources,
                        VARIANT=VARIANT, LIMITS=LIMITS)(*args, **kwargs)


def main(argv=None):
    return base._scoped(base.frozen.main, __doc__=__doc__,
                        configure_environment=configure_environment,
                        diagnostic_worker=diagnostic_worker)(argv)


if __name__ == '__main__':
    main()
