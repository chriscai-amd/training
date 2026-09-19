"""Guarded CPU tests for baseline provenance and actual retained scalar stages."""
from contextlib import contextmanager
import copy
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

import torch
import capture_linear_projection_baseline_diagnostics as target
import capture_linear_projection_deferred as original
import test_capture_linear_projection_deferred as fixture


class BaselineDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ProjectionCaptureTests.setUpClass()
        cls.sources = target.verify_sources()

    @classmethod
    def tearDownClass(cls):
        assert cls.sources == target.verify_sources()
        fixture.ProjectionCaptureTests.tearDownClass()

    def setUp(self):
        guard = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def environment(self, directory='/tmp/baseline-projection-diagnostics'):
        env = dict(target.REQUIRED)
        target.configure_environment({'directory': str(directory), 'steps': 2,
            'save_finite_step': 1, 'abs_threshold': 1e6, 'dataset': 'yambda-5b'}, env)
        return env

    @contextmanager
    def probe(self, **options):
        helper = fixture.ProjectionCaptureTests()
        helper.setUp()
        proxy = types.SimpleNamespace(**vars(original))
        proxy.LinearProjectionProbe = target.ProjectionProbe
        try:
            with mock.patch.object(fixture, 'capture', proxy), helper.probe(**options) as values:
                yield (*values, helper)
        finally:
            helper.doCleanups()

    @contextmanager
    def worker_modules(self, bootstrap, trainer_main=None, loop=None):
        gin = types.ModuleType('gin')
        gin.parse_config_file = lambda *args, **kwargs: target.validate_environment(os.environ)
        boot = types.ModuleType('generative_recommenders.dlrm_v4.train._env_bootstrap')
        boot.apply_env_bootstrap = bootstrap
        utils = types.SimpleNamespace(streaming_train_eval_loop=loop or (lambda **kwargs: kwargs))
        trainer = types.SimpleNamespace(_main_func=trainer_main or (lambda *args: None))
        modules = {'generative_recommenders.dlrm_v4.train.utils': utils,
                   'generative_recommenders.dlrm_v4.train.train_ranker': trainer}
        with mock.patch.dict(sys.modules, {'gin': gin, boot.__name__: boot}), \
                mock.patch.object(target.importlib, 'import_module', side_effect=modules.__getitem__) as imports:
            yield utils, trainer, imports

    def test_configured_policy_has_exact_baseline_controls_and_frozen_queue(self):
        before = dict(target.capture.configure_environment.__globals__)
        env = self.environment()
        options = json.loads(env['NAN_TRAINING_OPTIONS'])
        target.validate_worker(options, env)
        self.assertEqual({name: env[name] for name in target.REQUIRED}, target.REQUIRED)
        self.assertTrue(all(name not in env for name in target.ABSENT))
        self.assertEqual(options['monitor_variant'], target.VARIANT)
        self.assertEqual(options['baseline_control'], target.CONTROL)
        self.assertEqual(options['scalar_queue_diagnostics_policy'], target.POLICY)
        self.assertNotIn('mitigation_control', options)
        self.assertIs(target.DiagnosticScalarQueue, target.stages.DiagnosticScalarQueue)
        self.assertIs(target.DiagnosticScalarQueue.enqueue, target.stages.scalar.DeferredScalarQueue.enqueue)
        self.assertEqual(before, target.capture.configure_environment.__globals__)

    def test_all_environment_drift_rejected_without_mutating_caller(self):
        for name in target.REQUIRED:
            for value in (None, '', '1', '256'):
                env = dict(target.REQUIRED)
                if value is None:
                    env.pop(name)
                else:
                    env[name] = value
                before = copy.deepcopy(env)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'requires'):
                    target.configure_environment({}, env)
                self.assertEqual(env, before)
        for name in target.ABSENT:
            for value in ('', '0', '1'):
                env = dict(target.REQUIRED, **{name: value})
                before = copy.deepcopy(env)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'unset'):
                    target.configure_environment({}, env)
                self.assertEqual(env, before)

    def test_worker_policy_and_topology_drift_rejected_before_import(self):
        env = self.environment()
        options = json.loads(env['NAN_TRAINING_OPTIONS'])
        for name in ('monitor_variant', 'capture_mode', 'target', 'attention_vgpr_cap',
                     'weighted_ln_block_n', 'baseline_control', 'scalar_queue_diagnostics_policy',
                     'projection_capture_layer', 'projection_capture_format', 'hstu_scalar_monitor',
                     'hstu_layer_indices', 'hstu_operations', 'hstu_expected_endpoints'):
            bad = dict(options, **{name: None})
            bad_env = dict(env, NAN_TRAINING_OPTIONS=json.dumps(bad))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'provenance or scalar policy'):
                target.validate_worker(bad, bad_env)
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(target.importlib, 'import_module', side_effect=AssertionError('Early trainer import')):
            with self.assertRaisesRegex(ValueError, 'topology'):
                target.diagnostic_worker(1, 1, 0, 1, 'localhost', '29888', 'none', 'streaming-train-eval')
            bad = copy.deepcopy(options)
            bad['scalar_queue_diagnostics_policy']['boundary_readbacks_added'] = 1
            os.environ['NAN_TRAINING_OPTIONS'] = json.dumps(bad)
            with self.assertRaisesRegex(ValueError, 'scalar policy'):
                target.diagnostic_worker(0, 1, 0, 1, 'localhost', '29888', 'none', 'streaming-train-eval')

    def test_worker_bootstrap_drift_rejected_before_trainer_import(self):
        for name in (*target.REQUIRED, *target.ABSENT, 'NAN_TRAINING_OPTIONS'):
            env = self.environment()
            def bootstrap():
                target.validate_environment(os.environ)
                os.environ[name] = '{}' if name == 'NAN_TRAINING_OPTIONS' else '1'
            with self.subTest(name=name), mock.patch.dict(os.environ, env, clear=True), \
                    self.worker_modules(bootstrap) as (_, _, imports), self.assertRaises(ValueError):
                target.diagnostic_worker(0, 1, 0, 1, 'localhost', '29888', 'none', 'streaming-train-eval')
            imports.assert_not_called()

    def test_worker_executes_baseline_policy_and_restores_loop_on_return_and_error(self):
        env = self.environment()
        proxy, optimizer = object(), object()
        observations = []
        original_import = builtins.__import__
        def original_loop(**kwargs):
            target.validate_environment(os.environ)
            self.assertIs(kwargs['optimizer'], proxy)
            self.assertIs(builtins.__import__, original_import)
            observations.append('original_loop')
        def fake_probe(model, directory, **kwargs):
            target.validate_environment(os.environ)
            self.assertIs(kwargs['optimizer'], optimizer)
            self.assertEqual(kwargs['save_finite_step'], 1)
            observations.append('probe')
            return types.SimpleNamespace(optimizer_proxy=proxy)
        def instrumented_loop(function, options, args, kwargs, *, probe_factory):
            self.assertEqual(globals()['LIMITS'], target.LIMITS)
            self.assertIs(globals()['_archive_sources'], target._archive_sources)
            self.assertEqual(options['baseline_control'], target.CONTROL)
            probe_factory('model', 'directory')
            return function(**kwargs)
        for fail in (False, True):
            with mock.patch.dict(os.environ, env, clear=True), \
                    self.worker_modules(lambda: target.validate_environment(os.environ), loop=original_loop) as (utils, trainer, imports), \
                    mock.patch.object(target.training, 'run_instrumented_loop', instrumented_loop), \
                    mock.patch.object(target, 'ProjectionProbe', side_effect=fake_probe):
                def main(*args):
                    target.validate_environment(os.environ)
                    utils.streaming_train_eval_loop(optimizer=optimizer)
                    if fail:
                        raise RuntimeError('trainer fixture error')
                trainer._main_func = main
                before_globals = dict(main.__globals__)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'trainer fixture error'):
                        target.diagnostic_worker(0, 1, 0, 1, 'localhost', '29888', 'none', 'streaming-train-eval')
                else:
                    target.diagnostic_worker(0, 1, 0, 1, 'localhost', '29888', 'none', 'streaming-train-eval')
                self.assertIs(utils.streaming_train_eval_loop, original_loop)
                self.assertIs(builtins.__import__, original_import)
                self.assertEqual(main.__globals__, before_globals)
                self.assertEqual(dict(os.environ), env)
                self.assertEqual(imports.call_count, 2)
        self.assertEqual(observations, ['probe', 'original_loop', 'probe', 'original_loop'])

    def test_actual_backward_retains_baseline_metadata_68_scans_and_scalar_stages(self):
        with self.probe(save_finite_step=1) as (model, probe, state, helper):
            reader = target.stages.scalar._read_endpoint_batch
            with mock.patch.object(target.stages.scalar, '_read_endpoint_batch', wraps=reader) as read:
                helper.forward(model).sum().backward()
            self.assertEqual(read.call_count, 1)
            payload = helper.payload(probe)
            self.assertEqual(payload['capture_variant'], target.VARIANT)
            self.assertEqual(payload['metadata']['baseline_control'], target.CONTROL)
            self.assertNotIn('mitigation_control', payload['metadata'])
            self.assertEqual(payload['metadata']['scalar_queue_diagnostics_policy'], target.POLICY)
            self.assertEqual(len(payload['metadata']['deferred_scans']), 68)
            diag = payload['scalar_queue_diagnostics']
            self.assertEqual((diag['reader_calls'], diag['flush_complete']), (1, True))
            group = diag['groups'][0]
            for mapping, endpoint, batchrow, returned in zip(group['row_mapping'], group['endpoint_tensors'],
                                                           group['batch_tensor'], group['returned_rows']):
                self.assertTrue(torch.equal(endpoint, batchrow))
                self.assertEqual(endpoint.tolist(), returned)
                entry = diag['native_resolved_scans'][mapping['enqueue_index']]['summaries'][mapping['summary_index']]
                self.assertEqual(entry['name'], mapping['name'])
                self.assertEqual([entry['min'], entry['max']], returned)
            self.assertEqual((state.projection_backward_calls, state.attention_calls, state.silu_calls), (3, 3, 3))
            self.assertIsNone(probe.scalar_queue.diagnostics)
            linear, projection = target.split_capture(payload)
            import replay_additional_linear_capture as replay
            replay.validate_capture(linear)
            self.assertEqual(projection['actual_call_count'], 1)
            for name in ('capture_variant',):
                with self.assertRaises(ValueError):
                    target.split_capture(dict(payload, **{name: original.VARIANT}))
            changed = dict(payload, metadata=dict(payload['metadata'], baseline_control={}))
            with self.assertRaises(ValueError):
                target.split_capture(changed)

    def test_first_returned_comparison_fault_retained_without_producer_rerun(self):
        with self.probe() as (model, probe, state, helper):
            original_reader = target.stages.scalar._read_endpoint_batch
            selected = []
            def reader(batch):
                self.assertFalse(probe.in_backward)
                rows = original_reader(batch)
                mapping = probe.scalar_queue.diagnostics['groups'][0]['row_mapping']
                choices = [r for r in mapping if probe.scalar_queue.scans[r['enqueue_index']].get('expected_zero')]
                selected.append(choices[0])
                for item in choices[:2]:
                    rows[item['batch_row_index']] = [1., 1.]
                return rows
            with mock.patch.object(target.stages.scalar, '_read_endpoint_batch', reader), \
                    self.assertRaises(target.base.BoundaryAnomalyError):
                helper.forward(model).sum().backward()
            payload = helper.payload(probe)
            item = selected[0]
            self.assertEqual(payload['trigger']['enqueue_index'], item['enqueue_index'])
            group = payload['scalar_queue_diagnostics']['groups'][0]
            self.assertEqual(group['returned_rows'][item['batch_row_index']], [1., 1.])
            self.assertEqual(group['batch_tensor'][item['batch_row_index']].tolist(), [0., 0.])
            self.assertEqual(group['endpoint_tensors'][item['batch_row_index']].tolist(), [0., 0.])
            self.assertEqual(state.projection_backward_calls, 3)
            self.assertTrue(probe.failed)
            with self.assertRaisesRegex(target.base.BoundaryProbeError, 'failed'):
                probe.set_attempt('training', 0, 2)

    def test_actual_nan_projection_fault_preserves_existing_first_trigger(self):
        def corrupt(x, w, dz, result):
            result[0][1, 2] = float('nan')
            return result
        with self.probe(alter=corrupt) as (model, probe, state, helper):
            with self.assertRaises(target.base.BoundaryAnomalyError):
                helper.forward(model).sum().backward()
            payload = helper.payload(probe)
            self.assertEqual(payload['stage'], 'triton_addmm_bwd_after')
            self.assertEqual(payload['trigger']['flagged_names'], ['d_normed_x'])
            self.assertTrue(torch.isnan(payload['projection_boundary']['outputs']['dx'][1, 2]))
            self.assertEqual(state.projection_backward_calls, 3)
            self.assertEqual(payload['scalar_queue_diagnostics']['reader_calls'], 1)

    def test_healthy_attempts_release_diagnostics_without_copy_or_early_read(self):
        with self.probe() as (model, probe, state, helper):
            original_reader = target.stages.scalar._read_endpoint_batch
            queues = []
            def reader(batch):
                self.assertFalse(probe.in_backward)
                self.assertEqual(probe.post_leaves, {'weight', 'bias'})
                return original_reader(batch)
            for step in (1, 2):
                if step == 2:
                    model.zero_grad(set_to_none=True)
                    probe.set_attempt('training', 0, step)
                queues.append(probe.scalar_queue)
                with mock.patch.object(target.stages.scalar, '_read_endpoint_batch', side_effect=reader) as read, \
                        mock.patch.object(target.base, '_cpu_copy_tree', side_effect=AssertionError('Healthy frame copy')):
                    helper.forward(model).sum().backward()
                self.assertEqual(read.call_count, 1)
                self.assertIsNone(probe.scalar_queue.diagnostics)
                self.assertIsNone(probe.frame)
            self.assertIsNot(queues[0], queues[1])
            self.assertEqual(state.projection_backward_calls, 6)
            self.assertFalse(list(probe.directory.glob('*.pt')))

    def test_source_archive_contains_exact_baseline_and_queue_dependencies(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            context = {'source_sha256': {}}
            result = target._archive_sources(directory, context)
            expected = target.verify_sources()
            self.assertEqual(context['source_sha256'], expected)
            with tarfile.open(directory/result['path'], 'r:gz') as archive:
                for relative, digest in expected.items():
                    self.assertEqual(hashlib.sha256(archive.extractfile(relative).read()).hexdigest(), digest)
            self.assertIn('scripts/capture_linear_projection_baseline_diagnostics.py', result['entries'])
            self.assertIn('scripts/capture_linear_projection_diagnostics.py', result['entries'])
            self.assertIn('scripts/capture_linear_projection_mitigated.py', result['entries'])


if __name__ == '__main__':
    unittest.main()
