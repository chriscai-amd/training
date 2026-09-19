#!/usr/bin/env python3
"""Run the retained Linear/projection capture with three explicit mitigations.

This is a combined experiment, not isolated attribution or a validated NaN fix.
The original baseline capture and numerical implementations remain frozen.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
from pathlib import Path

import capture_linear_projection_deferred as capture

base = capture.base
training = base.base
VARIANT = 'linear_projection_cap256_bn1_workspace1_capture_v1'
CAPTURE_SHA256 = '2cf870a06afbe0ba04163583574c1bc48112cc9fbd752395bb76889163356ef5'
REQUIRED = {
    'HSTU_BWD_MAX_VGPR': '256',
    'WEIGHTED_LN_BWD_BLOCK_N': '1',
    'HIPBLASLT_WORKSPACE_SIZE': '1',
    'AMDGCN_USE_BUFFER_OPS': '0',
    'TRITON_FULL_AUTOTUNE': '0',
    'TRITON_ALLOW_PIPELINING': '0',
}
ABSENT = ('CUBLASLT_WORKSPACE_SIZE', 'TENSILE_STREAMK_DATA_PARALLEL',
          'TENSILE_STREAMK_FIXED_GRID')
CONTROL = {
    'kind': 'combined_mitigation_experiment',
    'attention_vgpr_cap': 256, 'weighted_ln_block_n': 1,
    'hipblaslt_workspace_kib': 1,
    'required_environment': REQUIRED, 'absent_environment': list(ABSENT),
    'scope': 'Combined finite controls do not establish isolated causality or a complete training fix.',
}
LIMITS = [*capture.LIMITS, CONTROL['scope'],
          'The actual process uses cap256, weighted-LN BN1 and 1 KiB hipBLASLt workspace; kernel selection varies with shape.']


def verify_sources():
    inventory = capture.verify_sources()
    path = Path(capture.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != CAPTURE_SHA256:
        raise capture.BoundaryProbeError('Frozen projection capture changed')
    inventory['scripts/' + path.name] = CAPTURE_SHA256
    inventory['scripts/' + Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return inventory


def validate_environment(environment):
    for name, expected in REQUIRED.items():
        if environment.get(name) != expected:
            raise ValueError(f'Combined mitigation requires {name}={expected}')
    for name in ABSENT:
        if name in environment:
            raise ValueError(f'Combined mitigation requires {name} unset')


def configure_environment(options, environment):
    verify_sources()
    validate_environment(environment)
    # Reuse baseline topology/path validation on a private configuration copy.
    # This copy is never installed as the process environment or used to launch.
    candidate = dict(environment, HSTU_BWD_MAX_VGPR='0', WEIGHTED_LN_BWD_BLOCK_N='0')
    capture.configure_environment(options, candidate)
    candidate.update(REQUIRED)
    recorded = json.loads(candidate['NAN_TRAINING_OPTIONS'])
    recorded.update(monitor_variant=VARIANT, attention_vgpr_cap=256,
                    weighted_ln_block_n=1, mitigation_control=CONTROL)
    candidate['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)
    validate_environment(candidate)
    environment.update(candidate)


class ProjectionProbe(capture.LinearProjectionProbe):
    def __init__(self, *args, **kwargs):
        base._scoped(capture.LinearProjectionProbe.__init__, verify_sources=verify_sources,
                     VARIANT=VARIANT, LIMITS=LIMITS)(self, *args, **kwargs)

    def save_frame(self, trigger):
        self.frame['metadata']['mitigation_control'] = CONTROL
        return base._scoped(base.DeferredAdditionalLinearProbe.save_frame,
                            VARIANT=VARIANT, LIMITS=LIMITS)(self, trigger)


def split_capture(payload):
    if payload.get('capture_variant') != VARIANT or payload.get('metadata', {}).get('mitigation_control') != CONTROL:
        raise ValueError('Requires the explicit combined mitigation capture')
    # Existing tensor validator accepts the same frame schema. Keep actual
    # configuration metadata while adapting only its variant discriminator.
    normalized = dict(payload, capture_variant=capture.VARIANT)
    return capture.split_capture(normalized)


def _archive_sources(directory, context):
    context['source_sha256'].update(verify_sources())
    return training._archive_sources(directory, context)


def validate_worker(options, environment):
    verify_sources()
    validate_environment(environment)
    if (options.get('monitor_variant') != VARIANT
            or options.get('capture_mode') != 'pristine'
            or options.get('target') != base.frozen.TARGET
            or options.get('attention_vgpr_cap') != 256
            or options.get('weighted_ln_block_n') != 1
            or options.get('mitigation_control') != CONTROL
            or options.get('projection_capture_layer') != 2
            or options.get('projection_capture_format') != capture.PROJECTION_FORMAT
            or options.get('hstu_scalar_monitor') is not True
            or options.get('hstu_layer_indices') != [0, 1, 2]
            or options.get('hstu_operations') != list(base.ALL_OPERATIONS)
            or options.get('hstu_expected_endpoints') != 42
            or environment.get('NAN_BACKWARD_TARGET') != '*'
            or environment.get('NAN_BACKWARD_SAVE_ALL') != '0'):
        raise ValueError('Combined mitigation worker lost capture provenance')


def diagnostic_worker(local_rank, world_size, node_rank, gpus_per_node,
                      master_addr, master_port, gin_file, mode):
    if (local_rank, world_size, node_rank, gpus_per_node, mode) != (0, 1, 0, 1, 'streaming-train-eval'):
        raise ValueError('Unsupported combined mitigation topology')
    options = json.loads(os.environ['NAN_TRAINING_OPTIONS'])
    validate_worker(options, os.environ)
    import gin
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()
    # Check again after bootstrap, before importing trainer components.
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

    utils.streaming_train_eval_loop = instrumented
    try:
        trainer._main_func(local_rank, world_size, node_rank, gpus_per_node,
                           master_addr, master_port, gin_file, mode)
    except SystemExit as error:
        path = Path(options['directory']) / 'outcome.json'
        if error.code == 42 and path.is_file() and json.loads(path.read_text()).get('status') == 'bounded_complete':
            print(f'[mitigated-projection] completed {options["steps"]} steps: {path}', flush=True)
            return
        raise
    finally:
        utils.streaming_train_eval_loop = original


def main(argv=None):
    return base._scoped(base.frozen.main, __doc__=__doc__, configure_environment=configure_environment,
                        diagnostic_worker=diagnostic_worker)(argv)


if __name__ == '__main__':
    main()
