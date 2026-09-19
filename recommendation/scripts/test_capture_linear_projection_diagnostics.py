"""CPU tests of actual queue stages, first-fault capture and frozen lifecycle."""
from contextlib import contextmanager
import copy
import json
import math
import os
import sys
import types
import unittest
from unittest import mock

import torch
import capture_linear_projection_diagnostics as target
import capture_linear_projection_deferred as original
import test_capture_linear_projection_deferred as fixture


class DiagnosticsTests(unittest.TestCase):
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

    def queue(self):
        queue = target.DiagnosticScalarQueue(chunk_bytes=4096, abs_threshold=1e6)
        queue.enqueue({'x': torch.tensor([-3., 5.]), 'skip': torch.tensor([1, 2]),
                       'empty': torch.empty(0)}, {'kind': 'first'})
        queue.enqueue({'a': torch.tensor([-2., 7.]), 'b': torch.tensor([4., 9.])}, {'kind': 'second'})
        return queue

    def test_original_enqueue_and_flush_once_actual_object_identity_and_row_mapping(self):
        self.assertIs(target.DiagnosticScalarQueue.enqueue, target.scalar.DeferredScalarQueue.enqueue)
        queue = self.queue()
        endpoints = [tensor for _, tensor in queue.pending['cpu']]
        calls = []
        original_reader = target.scalar._read_endpoint_batch
        reader_globals = target.scalar.DeferredScalarQueue.flush.__globals__
        before = dict(reader_globals)
        def reader(batch):
            rows = original_reader(batch)
            calls.append((batch, rows))
            return rows
        original_profile = sys.getprofile()
        flush_code = target.scalar.DeferredScalarQueue.flush.__code__
        entered = []
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code is flush_code:
                entered.append(1)
        try:
            sys.setprofile(profile)
            with mock.patch.object(target.scalar, '_read_endpoint_batch', reader), \
                    mock.patch.object(torch, 'stack', wraps=torch.stack) as stack:
                result = queue.flush()
                self.assertEqual(stack.call_count, 1)
        finally:
            sys.setprofile(original_profile)
        self.assertEqual(entered, [1])
        self.assertEqual(before, reader_globals)
        group = queue.diagnostics['groups'][0]
        self.assertEqual(group['row_mapping'], [
            {'batch_row_index': 0, 'enqueue_index': 0, 'summary_index': 0, 'name': 'x'},
            {'batch_row_index': 1, 'enqueue_index': 1, 'summary_index': 0, 'name': 'a'},
            {'batch_row_index': 2, 'enqueue_index': 1, 'summary_index': 1, 'name': 'b'}])
        self.assertTrue(all(a is b for a, b in zip(group['endpoint_tensors'], endpoints)))
        self.assertIs(group['batch_tensor'], calls[0][0])
        self.assertIs(group['returned_rows'], calls[0][1])
        self.assertEqual(result, queue.diagnostics['native_resolved_scans'])
        self.assertTrue(queue.diagnostics['flush_complete'])
        self.assertFalse(queue.pending or queue.streams or queue.events)
        self.assertNotIn('finite', group['queued_entries'][0])
        with self.assertRaisesRegex(target.base.BoundaryProbeError, 'only once'):
            queue.flush()
        with self.assertRaisesRegex(target.base.BoundaryProbeError, 'flushed scalar'):
            queue.enqueue({}, {})

    def test_distinct_endpoint_stack_and_returned_row_faults_remain_distinct(self):
        for stage in ('endpoint', 'stack', 'rows'):
            with self.subTest(stage=stage):
                queue = self.queue()
                if stage == 'endpoint':
                    queue.pending['cpu'][1][1][1] = 42
                original_reader, original_stack = target.scalar._read_endpoint_batch, torch.stack
                def stack(*args, **kwargs):
                    batch = original_stack(*args, **kwargs)
                    if stage == 'stack':
                        batch[1, 1] = 42
                    return batch
                def reader(batch):
                    rows = original_reader(batch)
                    if stage == 'rows':
                        rows[1][1] = 42
                    return rows
                with mock.patch.object(torch, 'stack', stack), \
                        mock.patch.object(target.scalar, '_read_endpoint_batch', reader):
                    scans = queue.flush()
                group = queue.diagnostics['groups'][0]
                self.assertEqual(group['endpoint_tensors'][1][1], 42 if stage == 'endpoint' else 7)
                self.assertEqual(group['batch_tensor'][1, 1], 7 if stage == 'rows' else 42)
                self.assertEqual(group['returned_rows'][1][1], 42)
                self.assertEqual(scans[1]['summaries'][0]['max'], 42)

    def test_returned_nan_inf_signed_zero_preserved_before_summary_conversion(self):
        queue = self.queue()
        rows = [[float('nan'), float('inf')], [-0.0, 0.0], [-2.0, -1.0]]
        with mock.patch.object(target.scalar, '_read_endpoint_batch', return_value=rows):
            queue.flush()
        retained = queue.diagnostics['groups'][0]['returned_rows']
        self.assertIs(retained, rows)
        self.assertTrue(math.isnan(retained[0][0]))
        self.assertEqual(retained[0][1], float('inf'))
        self.assertEqual(math.copysign(1, retained[1][0]), -1)

    def test_reader_exception_preserves_partial_evidence_and_never_retries(self):
        queue = self.queue()
        original_reader = target.scalar._read_endpoint_batch
        with mock.patch.object(target.scalar, '_read_endpoint_batch', side_effect=RuntimeError('injected')) as reader:
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                queue.flush()
            self.assertEqual(reader.call_count, 1)
        self.assertIs(target.scalar._read_endpoint_batch, original_reader)
        self.assertTrue(queue.flushed)
        self.assertFalse(queue.diagnostics['flush_complete'])
        group = queue.diagnostics['groups'][0]
        self.assertIn('batch_tensor', group)
        self.assertNotIn('returned_rows', group)
        with self.assertRaisesRegex(target.base.BoundaryProbeError, 'only once'):
            queue.flush()
        queue.release_diagnostics()
        self.assertIsNone(queue.diagnostics)

    def test_empty_queue_flush_preserves_zero_read_lifecycle(self):
        queue = target.DiagnosticScalarQueue(chunk_bytes=4096, abs_threshold=1e6)
        queue.enqueue({'integer': torch.tensor([1]), 'empty': torch.empty(0)}, {'kind': 'empty'})
        with mock.patch.object(target.scalar, '_read_endpoint_batch', side_effect=AssertionError('extra reader')):
            scans = queue.flush()
        self.assertEqual(queue.diagnostics['reader_calls'], 0)
        self.assertEqual(queue.diagnostics['groups'], [])
        self.assertEqual(scans[0]['flagged_names'], [])

    def test_actual_backward_saved_stages_all68_scans_and_split_compatibility(self):
        with self.probe(save_finite_step=1) as (model, probe, state, helper):
            helper.forward(model).sum().backward()
            payload = helper.payload(probe)
            self.assertEqual(payload['capture_variant'], target.mitigated.VARIANT)
            self.assertEqual(payload['metadata']['mitigation_control'], target.mitigated.CONTROL)
            self.assertEqual(payload['metadata']['scalar_queue_diagnostics_policy'], target.POLICY)
            diag = payload['scalar_queue_diagnostics']
            self.assertTrue(diag['flush_complete'])
            self.assertEqual(diag['reader_calls'], 1)
            self.assertEqual(len(diag['native_resolved_scans']), 68)
            self.assertEqual(len(payload['metadata']['deferred_scans']), 68)
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
            import replay_additional_linear_capture as replay
            linear, projection = target.split_capture(payload)
            replay.validate_capture(linear)
            self.assertEqual(projection['actual_call_count'], 1)
            self.assertFalse(any(t.requires_grad or t.grad_fn for _, t in target.base._tensors(payload)))

    def test_first_returned_comparison_fault_saved_without_producer_rerun(self):
        with self.probe() as (model, probe, state, helper):
            original_reader = target.scalar._read_endpoint_batch
            recorded = []
            def reader(batch):
                self.assertFalse(probe.in_backward)
                self.assertEqual(state.projection_backward_calls, 3)
                rows = original_reader(batch)
                diag = probe.scalar_queue.diagnostics
                mapping = diag['groups'][0]['row_mapping']
                choices = [item for item in mapping if probe.scalar_queue.scans[item['enqueue_index']].get('expected_zero')]
                for item in choices[:2]:
                    rows[item['batch_row_index']] = [1.0, 1.0]
                recorded.append(choices[0])
                return rows
            with mock.patch.object(target.scalar, '_read_endpoint_batch', reader), \
                    self.assertRaises(target.base.BoundaryAnomalyError):
                helper.forward(model).sum().backward()
            payload = helper.payload(probe)
            first = recorded[0]
            self.assertEqual(payload['trigger']['enqueue_index'], first['enqueue_index'])
            self.assertEqual(payload['metadata']['first_flagged_enqueue_index'], first['enqueue_index'])
            diag = payload['scalar_queue_diagnostics']
            self.assertEqual(diag['reader_calls'], 1)
            group = diag['groups'][0]
            index = first['batch_row_index']
            self.assertEqual(group['returned_rows'][index], [1., 1.])
            self.assertEqual(group['batch_tensor'][index].tolist(), [0., 0.])
            self.assertEqual(group['endpoint_tensors'][index].tolist(), [0., 0.])
            self.assertEqual(diag['native_resolved_scans'][first['enqueue_index']]['flagged_names'], [])
            self.assertEqual(payload['trigger']['flagged_names'], ['mismatch'])
            self.assertTrue(probe.failed)
            self.assertEqual(state.projection_backward_calls, 3)
            with self.assertRaisesRegex(target.base.BoundaryProbeError, 'failed'):
                probe.set_attempt('training', 0, 2)

    def test_actual_nan_producer_fault_captures_first_existing_trigger(self):
        def corrupt(x, w, dz, result):
            result[0][1, 2] = float('nan')
            return result
        with self.probe(alter=corrupt) as (model, probe, state, helper):
            with self.assertRaises(target.base.BoundaryAnomalyError):
                helper.forward(model).sum().backward()
            payload = helper.payload(probe)
            self.assertEqual(payload['stage'], 'triton_addmm_bwd_after')
            self.assertEqual(payload['trigger']['flagged_names'], ['d_normed_x'])
            self.assertEqual(state.projection_backward_calls, 3)
            self.assertEqual(payload['scalar_queue_diagnostics']['reader_calls'], 1)
            self.assertTrue(payload['metadata']['actual_backward_return_observed'])

    def test_healthy_no_copy_or_boundary_readback_and_two_attempt_release(self):
        with self.probe() as (model, probe, state, helper):
            original_reader = target.scalar._read_endpoint_batch
            queues = []
            def reader(batch):
                self.assertFalse(probe.in_backward)
                self.assertEqual(probe.post_leaves, {'weight', 'bias'})
                return original_reader(batch)
            for step in (1, 2):
                if step == 2:
                    model.zero_grad(set_to_none=True)
                    probe.set_attempt('training', 0, 2)
                queues.append(probe.scalar_queue)
                with mock.patch.object(target.scalar, '_read_endpoint_batch', side_effect=reader) as read, \
                        mock.patch.object(target.base, '_cpu_copy_tree', side_effect=AssertionError('healthy CPU copy')):
                    helper.forward(model).sum().backward()
                self.assertEqual(read.call_count, 1)
                self.assertIsNone(probe.scalar_queue.diagnostics)
                self.assertIsNone(probe.frame)
            self.assertIsNot(queues[0], queues[1])
            self.assertEqual(state.projection_backward_calls, 6)
            self.assertFalse(list(probe.directory.glob('*.pt')))

    def test_launcher_provenance_and_worker_policy_drift_rejected(self):
        env = dict(target.mitigated.REQUIRED)
        original_globals = dict(target.mitigated.configure_environment.__globals__)
        target.configure_environment({'directory': '/tmp/diagnostic-projection', 'steps': 2,
            'save_finite_step': 1, 'abs_threshold': 1e6, 'dataset': 'yambda-5b'}, env)
        options = json.loads(env['NAN_TRAINING_OPTIONS'])
        target.validate_worker(options, env)
        self.assertEqual(options['monitor_variant'], target.mitigated.VARIANT)
        self.assertEqual(options['scalar_queue_diagnostics_policy'], target.POLICY)
        self.assertEqual(original_globals, target.mitigated.configure_environment.__globals__)
        self.assertIn('scripts/capture_linear_projection_diagnostics.py', target.verify_sources())
        bad = copy.deepcopy(options)
        bad['scalar_queue_diagnostics_policy']['boundary_readbacks_added'] = 1
        with self.assertRaisesRegex(ValueError, 'scalar stages policy'):
            target.validate_worker(bad, env)
        with mock.patch.dict(os.environ, {'NAN_TRAINING_OPTIONS': json.dumps(bad)}, clear=True), \
                self.assertRaises(ValueError):
            target.diagnostic_worker(0, 1, 0, 1, 'localhost', '29888', 'none', 'streaming-train-eval')


if __name__ == '__main__':
    unittest.main()
