#!/usr/bin/env python3
"""Capture real Linear backward stages on device; resolve once after backward.

The unchanged all-layer STU scalar wrappers share the same deferred queue. No
engine completion callback resolves or stops this diagnostic. Complete backing
copies retain the selected Linear's original operands and every gradient stage,
including raw DX. CPU copying happens only on a fault or an explicit finite step.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types

import torch
import capture_additional_linear as frozen
import capture_training_backward as base
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError, _cpu_copy_tree, _logical_chunks, _source_specs, _summaries, _tensors
from nan_backward_training_all_layers import AllLayersBatchedTrainingBackwardBoundaryProbe
from nan_backward_training_deferred import DeferredAllLayersTrainingBackwardProbe, DeferredScalarQueue
from nan_backward_training_upstream import ALL_OPERATIONS

VARIANT = 'additional_linear_deferred_device_stages_v1'
FROZEN_SOURCES = {
    'capture_additional_linear.py': 'ef0a270e556fa520cf9195042270aaba71fa83752a061e7e070591af264838f9',
    'capture_training_backward.py': '3aadf4c87380d6e1613c46bc50279a0961cff7430843a4435f5cdb550fe75d8d',
    'nan_backward_boundaries.py': '68947ef9b59ae4aa7c0776bf77825242678ed363a2ce4100b235c84c47009e44',
    'nan_backward_training_probe.py': '8c1f8767e8fd7c202317cab83fbc839c5a21c9b599038c56cc435aac68e219a6',
    'nan_backward_training_batched.py': 'ed494a67f1ba940371b70545fbb710dc86c56c4e0690d71c2b0892a4f53e4020',
    'nan_backward_training_extended.py': '766008f2df633f8f7a4359d5ebb419db274e30b44b21b5d351f157484ddec2e3',
    'nan_backward_training_upstream.py': '0c985d740578085d4f64fad2d05bcf5c25e94ad492aad49599803ba24ca81ea4',
    'nan_backward_training_all_layers.py': 'f25bd8bba4d687b40e2b6f6f11423f8bb526e4c98adca33d43faca24c323d066',
    'nan_backward_training_deferred.py': '5af39b2673b2468994f8d8e9130a9037a71711a0aedb11eb04aef7734e6c83ce',
    'nan_replay_capture.py': '5d7f07f37d4b255ac190f8f030e51ba09ae267c219a00aed04aa64962ab829f8',
    'nan_replay_state.py': 'e5a482b13bc6d53c00d90318de9d52d149b35ae12f77d8f476d3e28ff6ec5817',
}
LIMITS = [
    'Selected Linear has original forward and pre-node inputs, immediate post-node inputs, raw DX/DW/DB, converted, accumulated and backward-return gradients in independent device snapshots.',
    'All three STU layers retain the established 42 scalar endpoints; their raw tensors, recomputed forward operations and residual addition remain uncaptured.',
    'Host enqueue order and per-stream FIFO do not establish total causality across streams. Copies, reductions and longer storage lifetimes perturb scheduling.',
    'Entire backing stores preserve offsets, padding and aliases, but early bucket snapshots certify the selected logical view only; neighboring slots can have concurrent writers.',
    'Flags are resolved after actual autograd.backward return, before clipping and the explicit optimizer. Sparse fused updates may already occur inside backward.',
    'A fault or explicitly requested finite step saves the retained frame without rerunning any producer; this is not a complete model/optimizer/allocator replay.',
]


def verify_sources():
    root = Path(__file__).resolve().parent
    inventory = {}
    for name, expected in FROZEN_SOURCES.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise BoundaryProbeError(f'Frozen diagnostic source changed: {name}')
        inventory[f'scripts/{name}'] = actual
    inventory[f'scripts/{Path(__file__).name}'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return inventory


def _storage_key(tensor):
    return str(tensor.device), tensor.untyped_storage()._cdata


def _storage_bytes(tensor):
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),), (1,))


class DeviceSnapshots:
    """Independent byte copies per observation; alias preservation within it."""
    def __init__(self, max_bytes):
        if max_bytes <= 0:
            raise ValueError('Snapshot byte budget must be positive')
        self.max_bytes, self.bytes = max_bytes, 0
        self.producers = {}
        self.streams = defaultdict(dict)
        self.events = []
        self.records = []

    def wait_reads(self, value):
        seen = set()
        for _, tensor in _tensors(value):
            key = _storage_key(tensor)
            if key in seen or key not in self.producers:
                continue
            seen.add(key)
            producer, event = self.producers[key]
            consumer = torch.cuda.current_stream(tensor.device)
            if consumer.cuda_stream != producer.cuda_stream:
                consumer.wait_event(event)
            tensor.record_stream(consumer)

    def copy(self, value):
        originals = {}
        for _, tensor in _tensors(value):
            if (tensor.layout != torch.strided or tensor.is_quantized or tensor.is_complex()
                    or tensor.device.type not in ('cpu', 'cuda')):
                raise BoundaryProbeError('Device snapshots require real strided CPU/CUDA tensors')
            originals.setdefault(_storage_key(tensor), tensor)
        needed = sum(t.untyped_storage().nbytes() for t in originals.values())
        if self.bytes + needed > self.max_bytes:
            raise BoundaryProbeError(f'Device snapshot needs {needed} bytes; {self.max_bytes - self.bytes} remain')
        self.wait_reads(value)
        copies = {}
        with torch.no_grad():
            for key, tensor in originals.items():
                source = _storage_bytes(tensor)
                target = source.clone()
                copies[key] = target
                stream_id = None
                if tensor.device.type == 'cuda':
                    stream = torch.cuda.current_stream(tensor.device)
                    stream_id = int(stream.cuda_stream)
                    source.record_stream(stream)
                    target.record_stream(stream)
                    event = torch.cuda.Event()
                    event.record(stream)
                    self.events.append(event)
                    self.producers[_storage_key(target)] = stream, event
                    self.streams[str(tensor.device)][stream_id] = stream
                self.records.append({'observation': len(self.records), 'device': str(tensor.device),
                                     'stream_id': stream_id, 'bytes': source.numel(),
                                     'source_storage_pointer': source.data_ptr(),
                                     'snapshot_storage_pointer': target.data_ptr()})
        self.bytes += needed

        def rebuild(item):
            if isinstance(item, torch.Tensor):
                backing = copies[_storage_key(item)]
                result = torch.empty(0, dtype=item.dtype, device=item.device).set_(
                    backing.untyped_storage(), item.storage_offset(), item.shape, item.stride())
                return torch._neg_view(result) if item.is_neg() else result
            if isinstance(item, dict):
                return {key: rebuild(child) for key, child in item.items()}
            if isinstance(item, list):
                return [rebuild(child) for child in item]
            if isinstance(item, tuple):
                return tuple(rebuild(child) for child in item)
            return item
        return rebuild(value)

    def join(self, queue):
        """Enqueue collector dependencies after backward; never read host values."""
        groups = defaultdict(dict)
        for collection in (self.streams, queue.streams):
            for device, streams in collection.items():
                groups[device].update(streams)
        records = []
        for device, streams in groups.items():
            collector = torch.cuda.current_stream(torch.device(device))
            for stream_id, producer in streams.items():
                event = torch.cuda.Event()
                event.record(producer)
                collector.wait_event(event)
                self.events.append(event)
                records.append({'device': device, 'producer_stream_id': stream_id,
                                'collector_stream_id': int(collector.cuda_stream)})
        return records


class ScalarHSTUProbe(DeferredAllLayersTrainingBackwardProbe):
    """Use established wrappers with no dense sentinel or completion callback."""
    def __init__(self, model, directory, **options):
        self.scalar_queue = None
        AllLayersBatchedTrainingBackwardBoundaryProbe.__init__(
            self, model, directory, optimizer=None, **options)


class DeferredOptimizerProxy(frozen.OptimizerProxy):
    def step(self, *args, **kwargs):
        p = self.probe
        if args or kwargs or p.failed or p.closed or not p.backward_complete or p.optimizer_complete:
            raise BoundaryProbeError('Expected one optimizer step after a successful deferred resolution')
        result = self.original.step()
        p.optimizer_complete = True
        return result


class DeferredAdditionalLinearProbe(frozen.AdditionalLinearProbe):
    def __init__(self, model, directory, *, monitor_hstu=True, hstu_modules=None, **options):
        self.hstu = None
        self.snapshots = None
        self.scalar_queue = None
        self.observed = {}
        self.source_inventory = verify_sources()
        # Reject explicit zero budgets instead of the frozen helper's `or` defaults.
        for key in ('max_bytes', 'chunk_bytes', 'abs_threshold'):
            if options.get(key) is not None and (not math.isfinite(options[key]) or options[key] <= 0):
                raise ValueError(f'{key} must be positive and finite')
        super().__init__(model, directory, **options)
        try:
            if self.optimizer_proxy is not None:
                self.optimizer_proxy = DeferredOptimizerProxy(self.optimizer_proxy.original, self)
            if monitor_hstu:
                self.hstu = ScalarHSTUProbe(model, self.directory / 'hstu',
                    mode='monitor_on_anomaly', selected_operations=ALL_OPERATIONS, **(hstu_modules or {}))
            self.emit('deferred_installed', variant=VARIANT, hstu_scalar_monitor=monitor_hstu,
                      source_sha256=self.source_inventory, limits=LIMITS)
        except BaseException:
            self.close()
            raise

    def set_attempt(self, mode, repeat, step):
        if self.closed or self.failed:
            raise BoundaryProbeError('Cannot reuse a closed or failed deferred probe')
        if self.attempt is not None and (not self.backward_complete or
                (self.optimizer_proxy is not None and not self.optimizer_complete)):
            raise BoundaryProbeError('Previous deferred attempt has not completed its optimizer step')
        if self.attempt is not None and step != self.attempt['step'] + 1:
            raise BoundaryProbeError('Nonconsecutive deferred attempt')
        if self.scalar_queue is not None and not self.scalar_queue.flushed:
            raise BoundaryProbeError('Previous deferred attempt was not resolved')
        self.scalar_queue = DeferredScalarQueue(chunk_bytes=self.chunk_bytes, abs_threshold=self.abs_threshold)
        self.snapshots = DeviceSnapshots(self.max_bytes)
        self.observed = {}
        super().set_attempt(mode, repeat, step)
        if self.hstu is not None:
            self.hstu.set_attempt(mode, repeat, step)
            self.hstu.scalar_queue = self.scalar_queue

    def copy(self, value):
        copied = self.snapshots.copy(value)
        self.copied_bytes = self.snapshots.bytes
        return copied

    def scan(self, stage, value, *, batched=False, fault_extra=None):
        self.check_live()
        if stage == 'forward_output':
            value = self.frame['forward_output'] = self.copy(value)
        self.snapshots.wait_reads(value)
        scan = self.scalar_queue.enqueue(value, {'kind': 'linear_stage', 'stage': stage})
        self.emit('scan_queued', **scan)
        # Keep only immutable stage snapshots or final, completed dense values.
        if stage != 'before_forward':
            self.observed[stage] = value
        return scan

    def compare(self, stage, expected, actual, *, full_storage=False):
        if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.device != actual.device:
            raise BoundaryProbeError(f'Comparison layout/dtype mismatch at {stage}')
        self.snapshots.wait_reads((expected, actual))
        if full_storage:
            expected, actual = _storage_bytes(expected), _storage_bytes(actual)
            if expected.shape != actual.shape:
                raise BoundaryProbeError(f'Comparison backing size mismatch at {stage}')
        with torch.no_grad(), torch.autocast(device_type=actual.device.type, enabled=False):
            equal = torch.ones((), dtype=torch.bool, device=actual.device)
            limit = max(1, self.chunk_bytes // actual.element_size())
            for left, right in zip(_logical_chunks(expected, limit), _logical_chunks(actual, limit)):
                equal = torch.logical_and(equal, torch.all(left[0] == right[0]))
            mismatch = torch.logical_not(equal).to(torch.float32)
        scan = self.scalar_queue.enqueue({'mismatch': mismatch},
            {'kind': 'linear_comparison', 'stage': stage, 'expected_zero': True,
             'comparison_scope': 'complete backing bytes' if full_storage else 'all logical values; signed zero accepted'})
        self.emit('comparison_queued', **scan)

    def node_pre(self, node, grad_outputs):
        super().node_pre(node, grad_outputs)
        self.frame['metadata'].update(pre_node_copy_completed_time=None,
            pre_node_copy_enqueued_time=time.time(),
            input_snapshot_ordering='Device copies enqueued before the actual Addmm node on its current autograd stream; completion joined after backward return')

    def node_post(self, grad_inputs, grad_outputs):
        self.check_live()
        if not self.pre_called or self.post_called or len(grad_inputs) != 3:
            raise BoundaryProbeError('Unexpected Addmm raw output lifecycle')
        self.post_called = True
        db, dx, dmat2 = grad_inputs
        x, mat2 = (self.frame['pristine_inputs'][key] for key in ('x', 'mat2'))
        if (not all(isinstance(t, torch.Tensor) for t in grad_inputs)
                or tuple(db.shape) != (self.module.out_features,) or dx.shape != x.shape
                or dmat2.shape != mat2.shape or any(t.dtype != x.dtype for t in grad_inputs)):
            raise BoundaryProbeError('Unexpected raw Addmm gradient mapping')
        raw = self.copy({'bias': db, 'dx': dx, 'dmat2': dmat2})
        self.frame['metadata']['raw_output_specs'] = frozen.tensor_metadata({'bias': db, 'dx': dx, 'dmat2': dmat2})
        self.frame['stages']['raw_small'] = {'bias': raw['bias'], 'dmat2': raw['dmat2']}
        self.frame['stages']['raw_dx'] = raw['dx']
        self.scan('raw_addmm_outputs', raw)
        current = self.frame['post_node_inputs'] = self.copy(self.live_inputs)
        self.scan('post_node_inputs', current)
        for key in ('x', 'mat2', 'dy'):
            self.compare(f'input_integrity_{key}', self.frame['pristine_inputs'][key], current[key], full_storage=True)
        self.live_inputs = None

    def parameter_pre(self, name, gradient):
        self.check_live()
        if not self.post_called or name in self.pre_leaves:
            raise BoundaryProbeError('Unexpected leaf incoming gradient lifecycle')
        self.pre_leaves.add(name)
        parameter = getattr(self.module, name)
        pair = self.copy({'prior': parameter.grad, 'incoming': gradient})
        stages = self.frame['stages']
        stages[f'parameter_prior_{name}'], stages[f'parameter_pre_{name}'] = pair['prior'], pair['incoming']
        self.frame['metadata'][f'parameter_prior_{name}_specs'] = frozen.tensor_metadata(parameter.grad)
        self.frame['metadata'][f'parameter_pre_{name}_specs'] = frozen.tensor_metadata(gradient)
        self.scan(f'parameter_prior_{name}', {'preexisting_gradient': pair['prior']})
        self.scan(f'parameter_pre_{name}', {'gradient': pair['incoming']})
        raw = stages['raw_small']['dmat2'].T if name == 'weight' else stages['raw_small']['bias']
        self.snapshots.wait_reads(raw)
        self.compare(f'raw_to_parameter_pre_{name}', raw.to(pair['incoming'].dtype), pair['incoming'])
        return gradient

    def parameter_post(self, name, parameter):
        self.check_live()
        if name not in self.pre_leaves or name in self.post_leaves or parameter.grad is None:
            raise BoundaryProbeError('Unexpected post-accumulate leaf lifecycle')
        self.post_leaves.add(name)
        stages = self.frame['stages']
        current = stages[f'parameter_post_{name}'] = self.copy(parameter.grad)
        self.frame['metadata'][f'parameter_post_{name}_specs'] = frozen.tensor_metadata(parameter.grad)
        self.scan(f'parameter_post_{name}', {'gradient': current})
        before, incoming = stages[f'parameter_prior_{name}'], stages[f'parameter_pre_{name}']
        self.snapshots.wait_reads((before, incoming))
        expected = incoming if before is None else before + incoming
        self.compare(f'leaf_accumulation_{name}', expected, current)

    def after_backward_return(self):
        self.check_live()
        if (self.in_backward or not self.pre_called or not self.post_called
                or self.pre_leaves != {'weight', 'bias'} or self.post_leaves != {'weight', 'bias'}
                or self.backward_complete):
            raise BoundaryProbeError('Actual backward returned without complete Linear lifecycle')
        if self.hstu is not None:
            self.hstu.assert_complete_operation_scans()
        self.frame['metadata']['producer_stream_joins_after_backward_return'] = self.snapshots.join(self.scalar_queue)
        final = {'weight': self.module.weight.grad, 'bias': self.module.bias.grad}
        if any(t is None for t in final.values()):
            raise BoundaryProbeError('Selected gradient missing after backward return')
        stages = self.frame['stages']
        stages['after_backward_return'] = self.copy(final)
        self.frame['metadata']['final_specs'] = frozen.tensor_metadata(final)
        self.scan('after_backward_return_selected', stages['after_backward_return'])
        for name in ('weight', 'bias'):
            self.compare(f'post_accumulate_to_final_{name}', stages[f'parameter_post_{name}'], stages['after_backward_return'][name])
        self.dense_scan('after_backward_return_all_dense', gradients=True)
        scans = self.scalar_queue.flush()
        for scan in scans:
            if scan.get('expected_zero'):
                scan['flagged_names'] = [entry['name'] for entry in scan['summaries']
                                         if not entry['finite'] or entry['max'] != 0]
        flagged = [scan for scan in scans if scan['flagged_names']]
        self.frame['metadata'].update(actual_backward_return_observed=True,
            snapshot_storage_copies=self.snapshots.records, deferred_scans=scans,
            source_sha256=self.source_inventory, snapshot_device_bytes=self.copied_bytes,
            first_flagged_enqueue_index=flagged[0]['enqueue_index'] if flagged else None)
        self.emit('deferred_flush', scan_count=len(scans), scans=scans,
                  actual_backward_return_observed=True,
                  first_flagged_enqueue_index=flagged[0]['enqueue_index'] if flagged else None)
        if flagged:
            self.failed = True
            self.save_frame(flagged[0])
        self.backward_complete = True
        if self.attempt['step'] == self.save_finite_step:
            self.save_frame(None)
        self.emit('attempt_backward_complete', retained_storage_bytes=self.copied_bytes)
        self.clear_call()
        # Release all per-attempt snapshots after resolution, before the update.
        self.frame = None
        self.observed.clear()
        self.snapshots = None

    def save_frame(self, trigger):
        stage = trigger.get('stage', f"{trigger.get('operation', 'dense')}_{trigger.get('phase', 'unknown')}") if trigger else 'after_backward_return'
        observed = self.observed.get(stage)
        full = {'frame': self.frame, 'fault_snapshot': observed,
                'final_dense': {'parameters': self.parameters,
                                'gradients': {name: p.grad for name, p in self.parameters.items() if p.grad is not None}}}
        copied, size = _cpu_copy_tree(full, self.chunk_bytes, self.max_bytes)
        summaries = _summaries(copied['fault_snapshot'], self.chunk_bytes,
                              _source_specs(observed), self.abs_threshold) if observed is not None else []
        persists = None
        if trigger is not None and observed is not None:
            bad = {item['name'] for item in summaries if not item['finite'] or item['extreme_count']}
            persists = set(trigger['flagged_names']) <= bad
        payload = {'format': frozen.FORMAT, 'format_version': 1, 'capture_variant': VARIANT,
                   'session': self.session, 'attempt': dict(self.attempt), 'target': self.name,
                   'stage': stage, **copied['frame'], 'fault_snapshot': copied['fault_snapshot'],
                   'final_dense': copied['final_dense'], 'fault_snapshot_summaries': summaries,
                   'trigger': trigger, 'trigger_persists_in_cpu_snapshot': persists,
                   'capture_storage_bytes': size, 'limits': LIMITS}
        suffix = stage if trigger else 'finite'
        target = self.directory / f'linear-deferred-{self.session}-s{self.attempt["step"]}-{suffix}.pt'
        frozen.atomic_save(target, payload)
        self.emit('dump_complete' if trigger else 'finite_dump_complete', stage=stage, dump=str(target),
                  capture_storage_bytes=size, trigger_persists_in_cpu_snapshot=persists)
        os.fsync(self.events.fileno())
        if trigger:
            raise BoundaryAnomalyError(target, 'additional_linear_deferred', self.name, stage)

    def close(self):
        if self.hstu is not None:
            self.hstu.close()
            self.hstu = None
        super().close()
        self.observed.clear()
        self.snapshots = None
        self.scalar_queue = None


def _scoped(function, **overrides):
    result = types.FunctionType(function.__code__, {**function.__globals__, **overrides},
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _archive_sources(directory, context):
    context['source_sha256'].update(verify_sources())
    return base._archive_sources(directory, context)


def configure_environment(options, environment):
    verify_sources()
    _scoped(frozen.configure_environment, VARIANT=VARIANT)(options, environment)
    recorded = json.loads(environment['NAN_TRAINING_OPTIONS'])
    recorded.update(hstu_scalar_monitor=True, hstu_layer_indices=[0, 1, 2],
                    hstu_operations=list(ALL_OPERATIONS), hstu_expected_endpoints=42,
                    deferred_resolution='after actual autograd.backward return')
    environment['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)
    environment['NAN_BACKWARD_TARGET'] = '*'
    environment['NAN_BACKWARD_SAVE_ALL'] = '0'


def diagnostic_worker(*args, **kwargs):
    verify_sources()
    options = json.loads(os.environ['NAN_TRAINING_OPTIONS'])
    if (options.get('hstu_scalar_monitor') is not True or options.get('hstu_layer_indices') != [0, 1, 2]
            or options.get('hstu_operations') != list(ALL_OPERATIONS) or options.get('hstu_expected_endpoints') != 42
            or os.environ.get('NAN_BACKWARD_TARGET') != '*' or os.environ.get('NAN_BACKWARD_SAVE_ALL') != '0'):
        raise ValueError('Deferred worker lost all-layer scalar scope')
    scoped_base = types.SimpleNamespace(**vars(base))
    scoped_base.run_instrumented_loop = _scoped(base.run_instrumented_loop, LIMITS=LIMITS, _archive_sources=_archive_sources)
    return _scoped(frozen.diagnostic_worker, VARIANT=VARIANT, LIMITS=LIMITS,
                   AdditionalLinearProbe=DeferredAdditionalLinearProbe, base=scoped_base)(*args, **kwargs)


def main(argv=None):
    return _scoped(frozen.main, __doc__=__doc__, configure_environment=configure_environment,
                   diagnostic_worker=diagnostic_worker)(argv)


if __name__ == '__main__':
    main()
