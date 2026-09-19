#!/usr/bin/env python3
"""Capture original uncapped projection/Linear boundaries and actual scalar stages.

The actual worker uses attention cap0, unchanged weighted-LN selection and
default hipBLASLt workspace/grid controls. The reviewed diagnostic queue keeps
its original producers, one flush and existing readback. No flag is bypassed.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path

import capture_linear_projection_deferred as capture
import capture_linear_projection_diagnostics as stages

base = capture.base
training = base.base
VARIANT = 'linear_projection_uncapped_default_workspace_scalar_stages_v1'
DIAGNOSTICS_FORMAT = stages.DIAGNOSTICS_FORMAT
POLICY = copy.deepcopy(stages.POLICY)
DiagnosticScalarQueue = stages.DiagnosticScalarQueue
FROZEN_SOURCES = {
    'capture_linear_projection_deferred.py': '2cf870a06afbe0ba04163583574c1bc48112cc9fbd752395bb76889163356ef5',
    'capture_linear_projection_mitigated.py': 'd3f1850a1f75a26645399e881f083b4cde067fb2325a0f69b798a36d295e25b6',
    'capture_linear_projection_diagnostics.py': '6a53a0c684144e60a16d716563ed5308304b35b9ee5113e5d95cb9a36fef6c57',
}
REQUIRED = {
    'HSTU_BWD_MAX_VGPR': '0',
    'WEIGHTED_LN_BWD_BLOCK_N': '0',
    'AMDGCN_USE_BUFFER_OPS': '0',
    'TRITON_FULL_AUTOTUNE': '0',
    'TRITON_ALLOW_PIPELINING': '0',
}
ABSENT = ('HIPBLASLT_WORKSPACE_SIZE', 'CUBLASLT_WORKSPACE_SIZE',
          'TENSILE_STREAMK_DATA_PARALLEL', 'TENSILE_STREAMK_FIXED_GRID')
CONTROL = {
    'kind': 'original_uncapped_default_workspace_baseline',
    'attention_vgpr_cap': 0,
    'weighted_ln_block_n': 0,
    'workspace_policy': 'No hipBLASLt or CUBLAS workspace override',
    'required_environment': REQUIRED,
    'absent_environment': list(ABSENT),
}
LIMITS = [*capture.LIMITS,
    'The actual worker uses uncapped attention, unchanged weighted-LN selection and no workspace, fixed-grid or data-parallel override. This records an environment policy, not a runtime kernel identity.',
    'The frozen DiagnosticScalarQueue is reused directly. Its module imports the mitigated wrapper as a dependency, but this worker and probe use the original baseline capture and never execute the mitigated worker.',
    'Actual endpoint and batch references are observed later by the existing save path; exact returned Python rows preserve the original flush-return observation.',
    'Retained scalar storages extend allocation lifetimes and can perturb scheduling. No additional reductions, stacks, stream joins or boundary readbacks are introduced.',
    'Original 68-scan coverage, first flagged stop and selected finite save are unchanged. Finite wrong summaries without an original flag require a selected finite save or another original trigger.',
]


def verify_sources():
    # Include the imported queue module and its full transitive source chain.
    inventory = stages.verify_sources()
    directory = Path(__file__).resolve().parent
    for name, expected in FROZEN_SOURCES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise base.BoundaryProbeError('Frozen baseline diagnostic dependency changed: ' + name)
        inventory['scripts/' + name] = expected
    path = Path(__file__).resolve()
    inventory['scripts/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inventory


def validate_environment(environment):
    for name, expected in REQUIRED.items():
        if environment.get(name) != expected:
            raise ValueError(f'Baseline diagnostics requires {name}={expected}')
    for name in ABSENT:
        if name in environment:
            raise ValueError(f'Baseline diagnostics requires {name} unset')


def configure_environment(options, environment):
    verify_sources()
    validate_environment(environment)
    capture.configure_environment(options, environment)
    recorded = json.loads(environment['NAN_TRAINING_OPTIONS'])
    recorded.update(monitor_variant=VARIANT, attention_vgpr_cap=0,
                    weighted_ln_block_n=0, baseline_control=CONTROL,
                    scalar_queue_diagnostics_policy=POLICY)
    environment['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)
    validate_environment(environment)


class BaselineDiagnosticProjectionProbe(capture.LinearProjectionProbe):
    def __init__(self, *args, **kwargs):
        base._scoped(capture.LinearProjectionProbe.__init__, verify_sources=verify_sources,
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
            raise base.BoundaryProbeError('Cannot save incomplete baseline scalar stages')
        self.frame['scalar_queue_diagnostics'] = diagnostics
        self.frame['metadata']['scalar_queue_diagnostics_policy'] = copy.deepcopy(POLICY)
        self.frame['metadata']['baseline_control'] = copy.deepcopy(CONTROL)
        return base._scoped(base.DeferredAdditionalLinearProbe.save_frame,
                            VARIANT=VARIANT, LIMITS=LIMITS)(self, trigger)

    def close(self):
        queue = getattr(self, 'scalar_queue', None)
        if isinstance(queue, DiagnosticScalarQueue):
            queue.release_diagnostics()
        return super().close()


ProjectionProbe = BaselineDiagnosticProjectionProbe
LinearProjectionProbe = BaselineDiagnosticProjectionProbe


def split_capture(payload):
    metadata = payload.get('metadata', {})
    if (payload.get('capture_variant') != VARIANT
            or metadata.get('baseline_control') != CONTROL
            or metadata.get('scalar_queue_diagnostics_policy') != POLICY
            or payload.get('scalar_queue_diagnostics', {}).get('format') != DIAGNOSTICS_FORMAT):
        raise ValueError('Requires the uncapped default-workspace scalar-stage capture')
    return capture.split_capture(dict(payload, capture_variant=capture.VARIANT))


def _archive_sources(directory, context):
    context['source_sha256'].update(verify_sources())
    return training._archive_sources(directory, context)


def validate_worker(options, environment):
    verify_sources()
    validate_environment(environment)
    if json.loads(environment.get('NAN_TRAINING_OPTIONS', '{}')) != options:
        raise ValueError('Baseline diagnostic worker lost serialized options')
    if (options.get('monitor_variant') != VARIANT
            or options.get('capture_mode') != 'pristine'
            or options.get('target') != base.frozen.TARGET
            or options.get('attention_vgpr_cap') != 0
            or options.get('weighted_ln_block_n') != 0
            or options.get('baseline_control') != CONTROL
            or options.get('scalar_queue_diagnostics_policy') != POLICY
            or options.get('projection_capture_layer') != 2
            or options.get('projection_capture_format') != capture.PROJECTION_FORMAT
            or options.get('hstu_scalar_monitor') is not True
            or options.get('hstu_layer_indices') != [0, 1, 2]
            or options.get('hstu_operations') != list(base.ALL_OPERATIONS)
            or options.get('hstu_expected_endpoints') != 42
            or environment.get('NAN_BACKWARD_TARGET') != '*'
            or environment.get('NAN_BACKWARD_SAVE_ALL') != '0'):
        raise ValueError('Baseline diagnostic worker lost capture provenance or scalar policy')


def diagnostic_worker(local_rank, world_size, node_rank, gpus_per_node,
                      master_addr, master_port, gin_file, mode):
    if (local_rank, world_size, node_rank, gpus_per_node, mode) != (0, 1, 0, 1, 'streaming-train-eval'):
        raise ValueError('Unsupported baseline diagnostic topology')
    options = json.loads(os.environ['NAN_TRAINING_OPTIONS'])
    validate_worker(options, os.environ)
    import gin
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()
    validate_worker(options, os.environ)
    utils = importlib.import_module('generative_recommenders.dlrm_v4.train.utils')
    trainer = importlib.import_module('generative_recommenders.dlrm_v4.train.train_ranker')
    original = utils.streaming_train_eval_loop

    @functools.wraps(original)
    def instrumented(*args, **kwargs):
        holder = {}
        def factory(model, directory, **probe_options):
            holder['probe'] = ProjectionProbe(model, directory, optimizer=kwargs['optimizer'],
                save_finite_step=options.get('save_finite_step', 0), **probe_options)
            return holder['probe']
        def with_optimizer(**loop_options):
            loop_options['optimizer'] = holder['probe'].optimizer_proxy
            return original(**loop_options)
        loop = base._scoped(training.run_instrumented_loop, LIMITS=LIMITS,
                            _archive_sources=_archive_sources)
        return loop(with_optimizer, options, args, kwargs, probe_factory=factory)

    # Use the established worker-local loop handoff and always restore it.
    # Frozen capture/queue globals and the real baseline environment stay intact.
    utils.streaming_train_eval_loop = instrumented
    try:
        trainer._main_func(local_rank, world_size, node_rank, gpus_per_node,
                           master_addr, master_port, gin_file, mode)
    except SystemExit as error:
        path = Path(options['directory']) / 'outcome.json'
        if error.code == 42 and path.is_file() and json.loads(path.read_text()).get('status') == 'bounded_complete':
            print(f'[baseline-projection-diagnostics] completed {options["steps"]} steps: {path}', flush=True)
            return
        raise
    finally:
        utils.streaming_train_eval_loop = original


def main(argv=None):
    return base._scoped(base.frozen.main, __doc__=__doc__,
                        configure_environment=configure_environment,
                        diagnostic_worker=diagnostic_worker)(argv)


if __name__ == '__main__':
    main()
