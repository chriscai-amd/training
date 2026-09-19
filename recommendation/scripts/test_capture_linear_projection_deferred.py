"""CPU-only actual-autograd capture, fault, alias and lifecycle regressions."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import capture_additional_linear as frozen
import capture_additional_linear_deferred as base
import capture_linear_projection_deferred as capture
import nan_backward_training_deferred as scalar
from test_capture_additional_linear import TinyModel, flat_storage
from test_nan_backward_training_extended import make_full_modules
from test_nan_backward_training_upstream import Stack


class ProjectionCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original_backward = torch.autograd.backward
        cls.rng = torch.get_rng_state().clone()
        cls.sources = capture.verify_sources()

    @classmethod
    def tearDownClass(cls):
        assert torch.autograd.backward is cls.original_backward
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()
        assert cls.sources == capture.verify_sources()

    def setUp(self):
        guard = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU initialization forbidden'))
        guard.start()
        self.addCleanup(guard.stop)
        env = mock.patch.dict(os.environ, {'NAN_BACKWARD_TARGET': '*', 'NAN_BACKWARD_SAVE_ALL': '0'})
        env.start()
        self.addCleanup(env.stop)

    @contextmanager
    def probe(self, *, alter=None, save_finite_step=0, **options):
        compute, preprocess, output, state = make_full_modules()
        preprocess.nullcontext = nullcontext
        original = preprocess.triton_addmm_bwd
        state.projection_backward_calls = 0
        @wraps(original)
        def backward(x, w, dz, is_y_1d):
            state.projection_backward_calls += 1
            result = original(x, w, dz, is_y_1d)
            if state.projection_backward_calls == 1 and alter is not None:
                result = alter(x, w, dz, result)
            return result
        preprocess.triton_addmm_bwd = backward
        model = TinyModel((4, 4))
        model.stack = Stack(compute)
        model.forward = lambda value: model.stack(model._additional_embedding_mlp(value))
        original_preprocess_backward = preprocess._HSTUPreprocessAndAttentionFunction.backward
        with tempfile.TemporaryDirectory() as directory:
            probe = capture.LinearProjectionProbe(model, Path(directory) / 'capture',
                expected_dims=(4, 4), chunk_bytes=4096, max_bytes=64 << 20,
                save_finite_step=save_finite_step,
                hstu_modules={'compute_module': compute, 'preprocess_module': preprocess, 'output_module': output}, **options)
            try:
                probe.set_attempt('training', 0, 1)
                yield model, probe, state
            finally:
                probe.close()
                self.assertIs(preprocess._HSTUPreprocessAndAttentionFunction.backward, original_preprocess_backward)
                self.assertIs(torch.autograd.backward, type(self).original_backward)
                self.assertFalse(model.linear._forward_hooks)
                self.assertFalse(model.linear.weight._backward_hooks)
                self.assertIsNone(probe.snapshots)

    def forward(self, model):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).div(64).requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            return model(x)

    def payload(self, probe):
        paths = list(probe.directory.glob('*.pt'))
        self.assertEqual(len(paths), 1)
        return torch.load(paths[0], weights_only=False, map_location='cpu')

    def test_actual_selected_call_full_inputs_outputs_and_all68_scans(self):
        with self.probe(save_finite_step=1) as (model, probe, state):
            self.forward(model).sum().backward()
            payload = self.payload(probe)
            boundary = payload['projection_boundary']
            self.assertEqual(state.projection_backward_calls, 3)
            self.assertEqual((state.attention_calls, state.silu_calls), (3, 3))
            self.assertTrue(boundary['layer'].endswith('._stu_layers.2'))
            self.assertEqual(boundary['actual_call_count'], 1)
            self.assertEqual(boundary['call'], 6)
            self.assertEqual(boundary['scalar_arguments'], {'is_y_1d': True})
            self.assertEqual(set(boundary['inputs_before']), {'x', 'w', 'dz', 'is_y_1d'})
            self.assertEqual(set(boundary['outputs']), {'dx', 'dw', 'db'})
            self.assertEqual(boundary['input_storage_integrity_comparisons'], ['x', 'w', 'dz'])
            self.assertTrue(boundary['input_snapshot_pristine'])
            scans = payload['metadata']['deferred_scans']
            self.assertEqual(len(scans), 68)
            self.assertEqual(sum(s['kind'] == 'operation' for s in scans), 42)
            self.assertTrue(payload['metadata']['actual_backward_return_observed'])
            self.assertTrue(all(not s['flagged_names'] for s in scans))
            self.assertIsNone(probe.hstu.sentinel)
            self.assertIsNone(probe.frame)
            self.assertTrue(all(t.grad_fn is None and not t.requires_grad for _, t in base._tensors(payload)))

    def test_complete_backing_pre_post_independent_and_split_replay_compatible(self):
        import replay_additional_linear_capture as replay
        with self.probe(save_finite_step=1) as (model, probe, _):
            self.forward(model).sum().backward()
            payload = self.payload(probe)
            linear, boundary = capture.split_capture(payload)
            replay.validate_capture(linear)
            self.assertNotIn('projection_boundary', linear)
            self.assertIs(boundary, payload['projection_boundary'])
            for name in ('x', 'w', 'dz'):
                before, after = boundary['inputs_before'][name], boundary['inputs_after'][name]
                self.assertTrue(torch.equal(flat_storage(before), flat_storage(after)))
                self.assertNotEqual(before.untyped_storage()._cdata, after.untyped_storage()._cdata)
            dz = boundary['inputs_before']['dz']
            self.assertEqual(dz.stride(), tuple(boundary['argument_layouts']['dz']['stride']))
            self.assertEqual(dz.untyped_storage().nbytes(), boundary['argument_layouts']['dz']['storage_bytes'])
            self.assertTrue(torch.equal(boundary['outputs']['dx'], dz @ boundary['inputs_before']['w'].t()))

    def test_no_hook_host_values_cpu_copy_or_early_resolution(self):
        with self.probe() as (model, probe, state):
            original_reader = scalar._read_endpoint_batch
            calls = []
            def read(batch):
                self.assertFalse(probe.in_backward)
                self.assertEqual(state.projection_backward_calls, 3)
                self.assertEqual(probe.post_leaves, {'weight', 'bias'})
                calls.append(1)
                return original_reader(batch)
            with mock.patch.object(scalar, '_read_endpoint_batch', side_effect=read), \
                    mock.patch.object(base, '_cpu_copy_tree', side_effect=AssertionError('Healthy CPU frame copy')), \
                    mock.patch.object(frozen, '_cpu_copy_tree', side_effect=AssertionError('Frozen copy')), \
                    mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host item')), \
                    mock.patch.object(torch, 'equal', side_effect=AssertionError('Host equality')):
                self.forward(model).sum().backward()
            self.assertEqual(calls, [1])
            self.assertFalse(list(probe.directory.glob('*.pt')))

    def test_finite_input_nan_dx_retained_after_entire_backward_without_rerun(self):
        def corrupt(x, w, dz, result):
            result[0][1, 2] = float('nan')
            return result
        with self.probe(alter=corrupt) as (model, probe, state):
            with self.assertRaises(base.BoundaryAnomalyError):
                self.forward(model).sum().backward()
            payload = self.payload(probe)
            boundary = payload['projection_boundary']
            self.assertEqual(state.projection_backward_calls, 3)
            self.assertEqual(payload['stage'], 'triton_addmm_bwd_after')
            self.assertEqual(payload['trigger']['call'], boundary['call'])
            self.assertEqual(payload['trigger']['flagged_names'], ['d_normed_x'])
            self.assertTrue(torch.isnan(boundary['outputs']['dx'][1, 2]))
            self.assertTrue(torch.isfinite(boundary['outputs']['dw']).all())
            self.assertTrue(torch.isfinite(boundary['outputs']['db']).all())
            for name in ('x', 'w', 'dz'):
                self.assertTrue(torch.isfinite(boundary['inputs_before'][name]).all())
                self.assertTrue(torch.equal(flat_storage(boundary['inputs_before'][name]), flat_storage(boundary['inputs_after'][name])))
            self.assertTrue(torch.isnan(payload['pristine_inputs']['dy']).any())
            self.assertIn('after_backward_return', payload['stages'])
            self.assertTrue(payload['metadata']['actual_backward_return_observed'])
            self.assertTrue(probe.failed)

    def test_postcall_input_mutation_is_retained_and_first_fault_below_threshold(self):
        def mutate(x, w, dz, result):
            dz.fill_(42)
            return result
        with self.probe(alter=mutate) as (model, probe, state):
            with self.assertRaises(base.BoundaryAnomalyError):
                self.forward(model).sum().backward()
            payload = self.payload(probe)
            boundary = payload['projection_boundary']
            self.assertEqual(payload['stage'], 'projection_layer2_input_integrity_dz')
            self.assertTrue(torch.all(boundary['inputs_after']['dz'] == 42))
            self.assertTrue(torch.all(boundary['inputs_before']['dz'] != 42))
            self.assertEqual(state.projection_backward_calls, 3)

    def test_joint_postcall_snapshot_preserves_actual_output_input_alias(self):
        def alias(x, w, dz, result):
            return x, result[1], result[2]
        with self.probe(alter=alias, save_finite_step=1) as (model, probe, _):
            self.forward(model).sum().backward()
            b = self.payload(probe)['projection_boundary']
            self.assertEqual(b['inputs_after']['x'].untyped_storage()._cdata, b['outputs']['dx'].untyped_storage()._cdata)
            self.assertNotEqual(b['inputs_before']['x'].untyped_storage()._cdata, b['outputs']['dx'].untyped_storage()._cdata)
            layouts = b['post_call_argument_output_layouts']
            self.assertEqual(layouts['inputs/x']['storage_group'], layouts['outputs/dx']['storage_group'])

    def test_budget_rejects_before_selected_public_call_and_partial_flush(self):
        with self.probe() as (model, probe, state):
            result = self.forward(model)
            probe.max_bytes = probe.snapshots.bytes + 1
            prior_bytes = probe.snapshots.bytes
            with mock.patch.object(scalar, '_read_endpoint_batch', side_effect=AssertionError('Partial flush')), \
                    self.assertRaisesRegex(capture.BoundaryProbeError, 'before the original call'):
                result.sum().backward()
            self.assertEqual(state.projection_backward_calls, 0)
            self.assertEqual(probe.snapshots.bytes, prior_bytes)
            self.assertFalse(probe.scalar_queue.flushed)

    def test_two_attempts_release_snapshots_and_preserve_selection(self):
        with self.probe(save_finite_step=2) as (model, probe, state):
            self.forward(model).sum().backward()
            self.assertIsNone(probe.frame)
            model.zero_grad(set_to_none=True)
            probe.set_attempt('training', 0, 2)
            self.forward(model).sum().backward()
            p = self.payload(probe)
            self.assertEqual(p['attempt']['step'], 2)
            self.assertEqual(p['projection_boundary']['attempt']['step'], 2)
            self.assertEqual(state.projection_backward_calls, 6)

    def test_launcher_and_worker_preserve_scope_without_global_mutation(self):
        before = dict(base.configure_environment.__globals__)
        environment = {}
        capture.configure_environment({'directory': '/tmp/new-projection', 'steps': 1000,
            'abs_threshold': 1e6, 'dataset': 'yambda-5b', 'save_finite_step': 0}, environment)
        recorded = json.loads(environment['NAN_TRAINING_OPTIONS'])
        self.assertEqual(recorded['monitor_variant'], capture.VARIANT)
        self.assertEqual(recorded['projection_capture_layer'], 2)
        self.assertEqual(recorded['projection_capture_format'], capture.PROJECTION_FORMAT)
        self.assertEqual(recorded['hstu_expected_endpoints'], 42)
        self.assertEqual(before, base.configure_environment.__globals__)
        recorded['projection_capture_layer'] = 1
        with mock.patch.dict(os.environ, {'NAN_TRAINING_OPTIONS': json.dumps(recorded)}), \
                self.assertRaisesRegex(ValueError, 'projection capture policy'):
            capture.diagnostic_worker()
        with mock.patch.object(capture, 'PROJECTION_SHA256', 'bad'), self.assertRaises(capture.BoundaryProbeError):
            capture.verify_sources()

    def test_missing_selected_capture_rejected_before_any_resolution(self):
        with self.probe() as (model, probe, _):
            result = self.forward(model)
            original = probe.after_backward_return
            def missing():
                probe.frame.pop('projection_boundary')
                return original()
            with mock.patch.object(probe, 'after_backward_return', side_effect=missing), \
                    mock.patch.object(scalar, '_read_endpoint_batch', side_effect=AssertionError('Incomplete resolution')), \
                    self.assertRaisesRegex(capture.BoundaryProbeError, 'Missing selected'):
                result.sum().backward()


if __name__ == '__main__':
    unittest.main()
