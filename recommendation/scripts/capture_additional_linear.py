#!/usr/bin/env python3
"""Capture the additional embedding final Linear, before clipping, on a cold run.

Full pristine forward and Addmm-node input storages are retained for one attempt.
Raw Addmm outputs, FP32 leaf incoming gradients, post-accumulation gradients and
actual post-backward-return gradients are distinct observations. Sparse fused
updates may already occur inside backward. Synchronization/copies perturb timing.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import torch
import capture_training_backward as base
from nan_backward_boundaries import (BoundaryAnomalyError, BoundaryProbeError,
    _cpu_copy_tree, _execution_controls, _source_specs, _summaries, _tensors)
from nan_backward_training_probe import monitor_summaries, _monitor_faults
from nan_backward_training_batched import batched_monitor_summaries

FORMAT = 'additional_linear_capture_v1'
TARGET = '_additional_embedding_mlp.2'
VARIANT = 'additional_linear_pristine_autograd_stages_v1'
LIMITS = [
    'Only the additional embedding final Linear is captured at individual autograd stages; other explicit optimizer parameters/gradients receive dense phase scans.',
    'Original forward and saved backward operands are independent complete CPU storage snapshots; raw DX is retained only on a raw-output fault.',
    'Actual backward return precedes final gradient scans and trainer clipping; leaf post-accumulate hooks alone do not establish final DDP state.',
    'No full model, optimizer, data or allocator replay is captured. Sparse fused optimizer updates may already have occurred inside backward.',
    'CPU copies, reductions and synchronization alter timing; a finite diagnostic run does not clear intermittent failures.',
]


def tensor_metadata(value):
    specs = _source_specs(value)
    for name, tensor in _tensors(value):
        specs[name].update(storage_bytes=tensor.untyped_storage().nbytes(),
                           storage_pointer=tensor.untyped_storage().data_ptr(),
                           data_pointer=tensor.data_ptr())
    return specs


def atomic_save(target, payload):
    """Publish without replacing an existing artifact; leave no partial final file."""
    target = Path(target)
    temporary = target.with_suffix('.pt.tmp')
    owns_temporary = False
    try:
        with temporary.open('xb') as stream:
            owns_temporary = True
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        temporary.unlink()
        descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if owns_temporary:
            temporary.unlink(missing_ok=True)
        raise


class OptimizerProxy:
    def __init__(self, original, probe):
        self.original, self.probe = original, probe

    def __getattr__(self, name):
        return getattr(self.original, name)

    def step(self, *args, **kwargs):
        p = self.probe
        if args or kwargs or not p.backward_complete or p.optimizer_complete:
            raise BoundaryProbeError('Expected exactly one no-argument step after observed backward return')
        p.dense_scan('before_optimizer_step', gradients=True)
        result = self.original.step()
        p.dense_scan('after_optimizer_step', gradients=True)
        p.optimizer_complete = True
        return result


class AdditionalLinearProbe:
    def __init__(self, model, directory, *, optimizer=None, mode='pristine',
                 selected_operations=None, target=TARGET, expected_dims=(256, 512),
                 abs_threshold=None, chunk_bytes=None, max_bytes=None, save_finite_step=0):
        if mode != 'pristine':
            raise ValueError('Focused Linear capture requires pristine inputs')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.chunk_bytes = chunk_bytes or int(os.getenv('NAN_BACKWARD_CHUNK_MIB', '64')) * 2**20
        self.max_bytes = max_bytes or int(os.getenv('NAN_BACKWARD_MAX_CAPTURE_GIB', '64')) * 2**30
        self.abs_threshold = float(abs_threshold or os.getenv('NAN_BACKWARD_ABS_THRESHOLD', '1e6'))
        if self.chunk_bytes <= 0 or self.max_bytes <= 0 or not math.isfinite(self.abs_threshold) or self.abs_threshold <= 0:
            raise ValueError('Positive finite budgets and threshold required')
        matches = [(n, m) for n, m in model.named_modules() if n.endswith(target)]
        if len(matches) != 1:
            raise BoundaryProbeError(f'Expected exactly one target Linear: {matches}')
        self.name, self.module = matches[0]
        if not isinstance(self.module, torch.nn.Linear) or self.module.bias is None:
            raise BoundaryProbeError('Target must be nn.Linear with bias')
        if (self.module.in_features, self.module.out_features) != tuple(expected_dims):
            raise BoundaryProbeError('Unexpected target Linear dimensions')
        self.model = model
        self.save_finite_step = save_finite_step
        self.parameters = {}
        if optimizer is not None:
            ids = {id(p) for g in optimizer.param_groups for p in g['params']}
            seen = set()
            for name, parameter in model.named_parameters():
                if id(parameter) not in ids:
                    continue
                seen.add(id(parameter))
                if (hasattr(parameter, 'local_shards') or hasattr(parameter, 'to_local')
                        or getattr(parameter, '_in_backward_optimizers', None) or not parameter.requires_grad):
                    continue
                if parameter.layout != torch.strided or not parameter.is_floating_point():
                    raise BoundaryProbeError(f'Unsupported optimizer parameter {name}')
                self.parameters[name] = parameter
            if seen != ids or not self.parameters:
                raise BoundaryProbeError('Explicit optimizer parameter identity mismatch')
        else:
            self.parameters = dict(model.named_parameters())  # CPU fixtures only
        if not {id(self.module.weight), id(self.module.bias)} <= {id(p) for p in self.parameters.values()}:
            raise BoundaryProbeError('Target leaves must be ordinary explicit-optimizer parameters')
        self.optimizer_proxy = OptimizerProxy(optimizer, self) if optimizer is not None else None
        self.session = f'{os.getpid()}-{time.time_ns()}'
        self.handles, self.call_handles = [], []
        self.closed = self.failed = False
        self.attempt = None
        self.backward_complete = self.optimizer_complete = False
        self.in_backward = False
        self.frame = None
        self.copied_bytes = 0
        self.events = (self.directory / 'events.jsonl').open('x')
        self.original_backward = torch.autograd.backward
        self.wrapped_backward = None
        try:
            self.handles.append(self.module.register_forward_pre_hook(self.forward_pre))
            self.handles.append(self.module.register_forward_hook(self.forward_post))
            for name, parameter in (('weight', self.module.weight), ('bias', self.module.bias)):
                self.handles.append(parameter.register_hook(functools.partial(self.parameter_pre, name)))
                self.handles.append(parameter.register_post_accumulate_grad_hook(
                    functools.partial(self.parameter_post, name)))

            @functools.wraps(self.original_backward)
            def backward(*args, **kwargs):
                if self.attempt is None or self.in_backward or self.backward_complete:
                    raise BoundaryProbeError('Unexpected/reentrant/multiple backward in focused diagnostic')
                self.in_backward = True
                try:
                    result = self.original_backward(*args, **kwargs)
                finally:
                    self.in_backward = False
                self.after_backward_return()
                return result

            self.wrapped_backward = backward
            torch.autograd.backward = backward
            self.emit('installed', target=self.name, parameters=list(self.parameters),
                      policy='Final stage runs after original torch.autograd.backward returned; before trainer clips. No queue_callback timing assumption.')
        except BaseException:
            self.close()
            raise

    def emit(self, event, **values):
        self.events.write(json.dumps({'event': event, 'session': self.session, 'attempt': self.attempt,
                                     'time': time.time(), **values}, allow_nan=False) + '\n')
        self.events.flush()

    def copy(self, value):
        copied, size = _cpu_copy_tree(value, self.chunk_bytes, self.max_bytes - self.copied_bytes)
        self.copied_bytes += size
        return copied

    def check_live(self):
        if self.closed or self.failed or self.attempt is None:
            raise BoundaryProbeError('Focused hook requires live nonfailed attempt')

    def scan(self, stage, value, *, batched=False, fault_extra=None):
        scanner = batched_monitor_summaries if batched else monitor_summaries
        summaries = scanner(value, chunk_bytes=self.chunk_bytes, abs_threshold=self.abs_threshold)
        bad, extreme = _monitor_faults(summaries)
        self.emit('scan', stage=stage, summaries=summaries, bad=bad, extreme=extreme)
        if bad or extreme:
            observed = {**value, **fault_extra} if fault_extra else value
            self.fault(stage, observed, {'bad': bad, 'extreme': extreme, 'summaries': summaries})
        return summaries

    def fault(self, stage, observed, trigger):
        self.failed = True
        # This union includes current saved operands, but only pristine_inputs
        # establish pre-node bytes. Full backing storage of bucket views is kept.
        copied = self.copy(observed)
        snapshot = _summaries(copied, self.chunk_bytes, _source_specs(observed), self.abs_threshold)
        bad = {s['name'] for s in snapshot if not s['finite']}
        extreme = {s['name'] for s in snapshot if s['extreme_count']}
        persists = (set(trigger.get('bad', ())) <= bad and set(trigger.get('extreme', ())) <= extreme
                    if 'bad' in trigger else None)
        target = self.directory / f'linear-{self.session}-s{self.attempt["step"]}-{stage}.pt'
        payload = {'format': FORMAT, 'format_version': 1, 'session': self.session, 'attempt': dict(self.attempt),
                   'stage': stage, 'target': self.name, **(self.frame or {}),
                   'fault_snapshot': copied, 'fault_specs': tensor_metadata(observed),
                   'fault_snapshot_summaries': snapshot, 'trigger': trigger,
                   'trigger_persists_in_cpu_snapshot': persists,
                   'capture_storage_bytes': self.copied_bytes,
                   'limits': ['Selected Linear only; other gradients receive scalar final scans.',
                              'No complete model/optimizer/allocator state; copies and scans perturb timing.',
                              'Sparse fused updates may already have occurred inside backward.']}
        atomic_save(target, payload)
        self.emit('dump_complete', stage=stage, dump=str(target), trigger_persists_in_cpu_snapshot=persists)
        os.fsync(self.events.fileno())
        raise BoundaryAnomalyError(target, 'additional_linear', self.name, stage)

    def set_attempt(self, mode, repeat, step):
        if self.closed or self.failed:
            raise BoundaryProbeError('Cannot reuse closed/failed focused probe')
        if self.attempt is not None and (not self.backward_complete or (self.optimizer_proxy and not self.optimizer_complete)):
            raise BoundaryProbeError('Previous focused attempt incomplete')
        if self.attempt is not None and step != self.attempt['step'] + 1:
            raise BoundaryProbeError('Nonconsecutive focused attempt')
        self.clear_call()
        self.attempt = {'mode': mode, 'repeat': repeat, 'step': step}
        self.frame = {'metadata': {}, 'stages': {}}
        self.copied_bytes = 0
        self.forward_calls = 0
        self.pre_called = self.post_called = False
        self.pre_leaves, self.post_leaves = set(), set()
        self.backward_complete = self.optimizer_complete = False
        self.dense_scan('before_forward', gradients=False)

    def forward_pre(self, module, args):
        self.check_live()
        if len(args) != 1 or self.forward_calls or not isinstance(args[0], torch.Tensor) or args[0].ndim != 2:
            raise BoundaryProbeError('Expected one two-dimensional Linear call per attempt')
        self.forward_calls += 1
        values = {'x': args[0], 'weight': module.weight, 'bias': module.bias}
        device = args[0].device.type
        self.frame['metadata'].update(execution_controls_forward=_execution_controls(),
            autocast_enabled=torch.is_autocast_enabled(device),
            autocast_dtype=str(torch.get_autocast_dtype(device)), device_type=device,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
            hipblaslt_tensile_libpath=os.getenv('HIPBLASLT_TENSILE_LIBPATH'),
            rocblas_tensile_libpath=os.getenv('ROCBLAS_TENSILE_LIBPATH'),
            forward_specs=tensor_metadata(values))
        self.frame['forward_inputs'] = self.copy(values)
        initial = {'weight': module.weight.grad, 'bias': module.bias.grad}
        self.frame['stages']['preexisting_grad'] = self.copy(initial)
        self.scan('forward_inputs', self.frame['forward_inputs'])

    def forward_post(self, module, args, output):
        self.check_live()
        node = output.grad_fn
        if (node is None or type(node).__name__ != 'AddmmBackward0'
                or not all(hasattr(node, key) for key in ('_saved_mat1', '_saved_mat2', '_saved_alpha', '_saved_beta'))):
            raise BoundaryProbeError(f'Unsupported actual Linear autograd node: {node}')
        self.frame['metadata'].update(node_kind=type(node).__name__, alpha=node._saved_alpha,
            beta=node._saved_beta, output_specs=tensor_metadata(output),
            saved_forward_specs=tensor_metadata({'x': node._saved_mat1, 'mat2': node._saved_mat2}))
        if node._saved_alpha != 1 or node._saved_beta != 1:
            raise BoundaryProbeError('Expected ordinary nn.Linear Addmm alpha=beta=1')
        self.scan('forward_output', output)
        # Node prehook gets the actual incoming gradient after output hooks.
        self.call_handles.append(node.register_prehook(functools.partial(self.node_pre, node)))
        self.call_handles.append(node.register_hook(self.node_post))

    def node_pre(self, node, grad_outputs):
        self.check_live()
        if not self.in_backward or self.pre_called or len(grad_outputs) != 1:
            raise BoundaryProbeError('Unexpected Addmm prehook lifecycle')
        self.pre_called = True
        x, mat2, dy = node._saved_mat1, node._saved_mat2, grad_outputs[0]
        if (not isinstance(dy, torch.Tensor) or x.ndim != 2 or mat2.ndim != 2
                or x.shape[1] != self.module.in_features
                or tuple(mat2.shape) != (self.module.in_features, self.module.out_features)
                or tuple(dy.shape) != (x.shape[0], self.module.out_features)
                or x.dtype != mat2.dtype or x.dtype != dy.dtype):
            raise BoundaryProbeError('Actual saved Addmm operand mapping mismatch')
        values = {'x': x, 'mat2': mat2, 'dy': dy}
        self.frame['metadata'].update(pristine_specs=tensor_metadata(values),
            execution_controls_backward=_execution_controls(),
            input_provenance={'pristine': True, 'operation_started': False},
            input_snapshot_pristine=True, pre_node_copy_completed_time=None)
        self.frame['pristine_inputs'] = self.copy(values)
        self.live_inputs = values
        self.frame['metadata']['pre_node_copy_completed_time'] = time.time()
        self.scan('node_inputs', self.frame['pristine_inputs'])

    def node_post(self, grad_inputs, grad_outputs):
        self.check_live()
        if not self.pre_called or self.post_called or len(grad_inputs) != 3:
            raise BoundaryProbeError('Unexpected Addmm raw output mapping/lifecycle')
        self.post_called = True
        db, dx, dmat2 = grad_inputs
        x, mat2, dy = (self.frame['pristine_inputs'][k] for k in ('x', 'mat2', 'dy'))
        if (not all(isinstance(t, torch.Tensor) for t in (db, dx, dmat2))
                or tuple(db.shape) != (self.module.out_features,)
                or dx.shape != x.shape or dmat2.shape != mat2.shape
                or any(t.dtype != x.dtype for t in (db, dx, dmat2))):
            raise BoundaryProbeError('Unexpected actual raw Addmm gradient tuple')
        raw = {'bias': db, 'dx': dx, 'dmat2': dmat2}
        self.frame['metadata']['raw_output_specs'] = tensor_metadata(raw)
        self.frame['stages']['raw_small'] = self.copy({'bias': db, 'dmat2': dmat2})
        # Raw DW/bias are already independently frozen. DX is copied only on a
        # trigger because it is large; raw output scans still cover every element.
        self.scan('raw_addmm_outputs', raw, fault_extra={'current_inputs': self.live_inputs})
        self.live_inputs = None

    def parameter_pre(self, name, gradient):
        self.check_live()
        if not self.post_called or name in self.pre_leaves:
            raise BoundaryProbeError('Leaf gradient arrived without one raw Addmm result')
        self.pre_leaves.add(name)
        parameter = getattr(self.module, name)
        pair = self.copy({'prior': parameter.grad, 'incoming': gradient})
        self.frame['stages'][f'parameter_prior_{name}'] = pair['prior']
        copied = pair['incoming']
        self.frame['stages'][f'parameter_pre_{name}'] = copied
        self.frame['metadata'][f'parameter_prior_{name}_specs'] = tensor_metadata(parameter.grad)
        self.frame['metadata'][f'parameter_pre_{name}_specs'] = tensor_metadata(gradient)
        self.scan(f'parameter_prior_{name}', {'preexisting_gradient': parameter.grad})
        self.scan(f'parameter_pre_{name}', {'gradient': gradient})
        raw = self.frame['stages']['raw_small']['dmat2'].T if name == 'weight' else self.frame['stages']['raw_small']['bias']
        expected = raw.to(copied.dtype)
        equal = torch.equal(expected, copied)
        self.emit('comparison', stage=f'raw_to_parameter_pre_{name}', equal=equal,
                  operation='Exact transpose (weight only) and dtype conversion, compared on CPU')
        if not equal:
            self.fault(f'parameter_pre_{name}_mismatch', {'gradient': gradient},
                       {'comparison': 'raw_gradient_transpose_cast', 'equal': False})
        return gradient

    def parameter_post(self, name, parameter):
        self.check_live()
        if name not in self.pre_leaves or name in self.post_leaves or parameter.grad is None:
            raise BoundaryProbeError('Unexpected post-accumulate leaf lifecycle')
        self.post_leaves.add(name)
        self.frame['stages'][f'parameter_post_{name}'] = self.copy(parameter.grad)
        self.frame['metadata'][f'parameter_post_{name}_specs'] = tensor_metadata(parameter.grad)
        self.scan(f'parameter_post_{name}', {'gradient': parameter.grad})
        before = self.frame['stages'][f'parameter_prior_{name}']
        incoming = self.frame['stages'][f'parameter_pre_{name}']
        expected = incoming if before is None else before + incoming
        equal = torch.equal(expected, self.frame['stages'][f'parameter_post_{name}'])
        self.emit('comparison', stage=f'leaf_accumulation_{name}', equal=equal,
                  note='Observed leaf post-accumulate hook; not a final-DDP timing guarantee')
        if not equal:
            self.fault(f'parameter_post_{name}_mismatch', {'gradient': parameter.grad},
                       {'comparison': 'preexisting_plus_incoming', 'equal': False})

    def dense_scan(self, stage, *, gradients):
        value = {'parameters': self.parameters}
        if gradients:
            value['gradients'] = {n: p.grad for n, p in self.parameters.items() if p.grad is not None}
            if not value['gradients']:
                raise BoundaryProbeError('No dense gradients at final backward observation')
        self.scan(stage, value, batched=True)

    def after_backward_return(self):
        self.check_live()
        if (not self.pre_called or not self.post_called or self.pre_leaves != {'weight', 'bias'}
                or self.post_leaves != {'weight', 'bias'} or self.backward_complete):
            raise BoundaryProbeError('Actual backward returned without complete selected Linear lifecycle')
        final = {'weight': self.module.weight.grad, 'bias': self.module.bias.grad}
        if any(v is None for v in final.values()):
            raise BoundaryProbeError('Selected gradient missing after backward return')
        self.frame['stages']['after_backward_return'] = self.copy(final)
        self.frame['metadata']['final_specs'] = tensor_metadata(final)
        self.scan('after_backward_return_selected', final)
        for name in ('weight', 'bias'):
            equal = torch.equal(self.frame['stages'][f'parameter_post_{name}'],
                                self.frame['stages']['after_backward_return'][name])
            self.emit('comparison', stage=f'post_accumulate_to_final_{name}', equal=equal,
                      note='World size one; hook observed value versus actual backward-return value')
            if not equal:
                self.fault(f'after_backward_return_{name}_mismatch',
                           {'selected': final, 'dense_gradients': {n: p.grad for n, p in self.parameters.items()}},
                           {'comparison': 'post_accumulate_to_final', 'equal': False})
        self.dense_scan('after_backward_return_all_dense', gradients=True)
        self.backward_complete = True
        if self.attempt['step'] == self.save_finite_step:
            target = self.directory / f'linear-{self.session}-s{self.attempt["step"]}-finite.pt'
            atomic_save(target, {'format': FORMAT, 'format_version': 1, 'session': self.session, 'attempt': dict(self.attempt),
                                'target': self.name, 'stage': 'after_backward_return',
                                'observation': 'selected_and_dense_scans_below_threshold',
                                **self.frame, 'capture_storage_bytes': self.copied_bytes,
                                'limits': LIMITS})
            self.emit('finite_dump_complete', dump=str(target))
        self.emit('attempt_backward_complete', retained_storage_bytes=self.copied_bytes)
        self.clear_call()
        # Pristine state retained until optimizer sentinels finish; discarded at
        # next set_attempt. No live tensors/graphs are kept by the frame.

    def clear_call(self):
        for handle in self.call_handles:
            handle.remove()
        self.call_handles.clear()
        self.live_inputs = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.clear_call()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.wrapped_backward is not None and torch.autograd.backward is self.wrapped_backward:
            torch.autograd.backward = self.original_backward
        self.frame = None
        self.events.close()


def configure_environment(options, environment):
    if environment.get('HSTU_BWD_MAX_VGPR', '0') != '0':
        raise ValueError('Focused baseline requires uncapped attention HSTU_BWD_MAX_VGPR=0')
    if environment.get('WEIGHTED_LN_BWD_BLOCK_N', '0') != '0':
        raise ValueError('Focused baseline requires unchanged LN selection WEIGHTED_LN_BWD_BLOCK_N=0')
    candidate = dict(environment)
    candidate.pop('HSTU_BWD_MAX_VGPR', None)
    recorded = {**options, 'capture_mode': 'pristine', 'target': TARGET,
                'operations': ['hstu_output_grad_mm'], 'monitor_variant': VARIANT,
                'attention_vgpr_cap': 0, 'weighted_ln_block_n': 0}
    base.configure_environment(recorded, candidate)
    recorded['operations'] = ['additional_linear_autograd_stages']
    candidate['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)
    candidate['HSTU_BWD_MAX_VGPR'] = '0'
    candidate['WEIGHTED_LN_BWD_BLOCK_N'] = '0'
    environment.update(candidate)


def diagnostic_worker(local_rank, world_size, node_rank, gpus_per_node,
                      master_addr, master_port, gin_file, mode):
    if (local_rank, world_size, node_rank, gpus_per_node, mode) != (0, 1, 0, 1, 'streaming-train-eval'):
        raise ValueError('Unsupported focused diagnostic topology')
    options = json.loads(os.environ['NAN_TRAINING_OPTIONS'])
    if (options.get('monitor_variant') != VARIANT or options.get('capture_mode') != 'pristine'
            or options.get('target') != TARGET or os.environ.get('HSTU_BWD_MAX_VGPR') != '0'
            or os.environ.get('WEIGHTED_LN_BWD_BLOCK_N') != '0'):
        raise ValueError('Focused worker lost baseline provenance')
    import gin
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()
    utils = importlib.import_module('generative_recommenders.dlrm_v4.train.utils')
    trainer = importlib.import_module('generative_recommenders.dlrm_v4.train.train_ranker')
    original = utils.streaming_train_eval_loop

    @functools.wraps(original)
    def instrumented(*args, **kwargs):
        holder = {}
        def factory(model, directory, **probe_options):
            holder['probe'] = AdditionalLinearProbe(model, directory, optimizer=kwargs['optimizer'],
                save_finite_step=options.get('save_finite_step', 0), **probe_options)
            return holder['probe']
        def with_optimizer(**loop_options):
            loop_options['optimizer'] = holder['probe'].optimizer_proxy
            return original(**loop_options)
        # Preserve the frozen shared wrapper while recording this diagnostic's
        # actual scope in context/outcome instead of the HSTU wrapper limits.
        scoped = types.FunctionType(base.run_instrumented_loop.__code__,
                                    {**base.run_instrumented_loop.__globals__, 'LIMITS': LIMITS},
                                    base.run_instrumented_loop.__name__, base.run_instrumented_loop.__defaults__)
        scoped.__kwdefaults__ = base.run_instrumented_loop.__kwdefaults__
        return scoped(with_optimizer, options, args, kwargs, probe_factory=factory)

    utils.streaming_train_eval_loop = instrumented
    try:
        trainer._main_func(local_rank, world_size, node_rank, gpus_per_node,
                           master_addr, master_port, gin_file, mode)
    except SystemExit as error:
        path = Path(options['directory']) / 'outcome.json'
        if error.code == 42 and path.is_file() and json.loads(path.read_text()).get('status') == 'bounded_complete':
            print(f'[focused-linear] completed {options["steps"]} steps: {path}', flush=True)
            return
        raise
    finally:
        utils.streaming_train_eval_loop = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--abs-threshold', type=float, default=1e6)
    parser.add_argument('--save-finite-step', type=int, default=0,
                        help='Retain one complete finite attempt for standalone replay; 0 disables')
    parser.add_argument('--dataset', default='yambda-5b')
    args = parser.parse_args(argv)
    options = vars(args)
    options['directory'] = str(Path(args.directory).resolve())
    if Path(options['directory']).exists():
        parser.error('--directory must be new')
    if args.save_finite_step < 0 or args.save_finite_step > args.steps:
        parser.error('--save-finite-step must be zero or within the requested step bound')
    try:
        configure_environment(options, os.environ)
    except ValueError as error:
        parser.error(str(error))
    trainer = importlib.import_module('generative_recommenders.dlrm_v4.train.train_ranker')
    original_worker, original_argv = trainer._main_func, sys.argv
    trainer._main_func = diagnostic_worker
    sys.argv = [sys.argv[0], '--dataset', args.dataset, '--mode', 'streaming-train-eval']
    try:
        return trainer.main()
    finally:
        trainer._main_func, sys.argv = original_worker, original_argv


if __name__ == '__main__':
    main()
