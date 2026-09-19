"""CPU provenance and actual retained-frame tests for the combined experiment."""
import copy
import json
import os
import unittest
from unittest import mock

import torch
import capture_linear_projection_deferred as original
import capture_linear_projection_mitigated as target
import test_capture_linear_projection_deferred as fixture


class MitigationTests(unittest.TestCase):
    def setUp(self):
        guard = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def environment(self):
        env = dict(target.REQUIRED)
        target.configure_environment({'directory': '/tmp/combined-projection', 'steps': 2,
            'save_finite_step': 1, 'abs_threshold': 1e6, 'dataset': 'yambda-5b'}, env)
        return env

    def test_actual_controls_recorded_and_original_baseline_unchanged(self):
        before = target.verify_sources()
        env = self.environment()
        target.validate_worker(json.loads(env['NAN_TRAINING_OPTIONS']), env)
        self.assertEqual({k: env[k] for k in target.REQUIRED}, target.REQUIRED)
        self.assertEqual(before, target.verify_sources())
        with self.assertRaisesRegex(ValueError, 'uncapped attention'):
            original.configure_environment({}, dict(target.REQUIRED))

    def test_each_control_drift_rejected_before_worker_imports(self):
        env = self.environment()
        options = json.loads(env['NAN_TRAINING_OPTIONS'])
        for key in target.REQUIRED:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'requires'):
                target.validate_worker(options, dict(env, **{key: 'bad'}))
        for key in target.ABSENT:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unset'):
                target.validate_worker(options, dict(env, **{key: ''}))
        for key in ['monitor_variant', 'mitigation_control', 'projection_capture_layer',
                    'attention_vgpr_cap', 'weighted_ln_block_n', 'hstu_expected_endpoints']:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'provenance'):
                target.validate_worker(dict(options, **{key: None}), env)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(target.importlib, 'import_module',
                side_effect=AssertionError('Import before topology validation')), self.assertRaisesRegex(ValueError, 'topology'):
            target.diagnostic_worker(1, 1, 0, 1, 'localhost', '29888', 'none', 'streaming-train-eval')

    def test_failed_configuration_does_not_mutate_caller_environment(self):
        env = dict(target.REQUIRED, CUBLASLT_WORKSPACE_SIZE='1')
        before = copy.deepcopy(env)
        with self.assertRaisesRegex(ValueError, 'unset'):
            target.configure_environment({}, env)
        self.assertEqual(env, before)

    def test_actual_cpu_backward_retains_mitigation_metadata_and_replay_schema(self):
        # Use the established three-layer CPU model and real autograd hooks.
        helper = fixture.ProjectionCaptureTests()
        fixture.ProjectionCaptureTests.setUpClass()
        try:
            helper.setUp()
            # The context helper resolves its capture module dynamically. Replace
            # only that reference, not the original class used by the subclass.
            import types
            proxy = types.SimpleNamespace(**vars(original))
            proxy.LinearProjectionProbe = target.ProjectionProbe
            with mock.patch.object(fixture, 'capture', proxy), helper.probe(save_finite_step=1) as (model, probe, state):
                helper.forward(model).sum().backward()
                payload = helper.payload(probe)
                self.assertEqual(payload['capture_variant'], target.VARIANT)
                self.assertEqual(payload['metadata']['mitigation_control'], target.CONTROL)
                self.assertEqual(state.projection_backward_calls, 3)
                self.assertEqual(len(payload['metadata']['deferred_scans']), 68)
                linear, projection = target.split_capture(payload)
                import replay_additional_linear_capture as replay
                replay.validate_capture(linear)
                self.assertEqual(projection['actual_call_count'], 1)
        finally:
            helper.doCleanups()
            fixture.ProjectionCaptureTests.tearDownClass()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
