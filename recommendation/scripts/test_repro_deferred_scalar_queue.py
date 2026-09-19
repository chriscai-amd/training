"""CPU-only byte provenance and injected-fault tests for the real scalar queue."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import repro_deferred_scalar_queue as target


def payload():
    parameter_store = torch.arange(48, dtype=torch.float32).sub(20).div(32)
    bias_store = torch.zeros(8, dtype=torch.float32)
    bucket = torch.arange(211, dtype=torch.float32).sub(50).div(1024)
    tree = {'parameters': {'weight': parameter_store[7:19].reshape(3, 4).t().requires_grad_(),
                           'bias': bias_store[2:5].requires_grad_()},
            'gradients': {'weight': bucket[37:49].reshape(4, 3), 'bias': bucket[57:60]}}
    queue = target.frozen.DeferredScalarQueue(chunk_bytes=64 << 20, abs_threshold=1e6)
    queue.enqueue({'first': torch.tensor([-3., 5.]), 'second': torch.tensor([2., 7.])},
                  {'kind': 'earlier_scan', 'stage': 'before_final'})
    queue.enqueue(tree, {'kind': 'final_dense', 'stage': target.STAGE})
    scans = queue.flush()
    return {'final_dense': tree, 'metadata': {'deferred_scans': scans, 'source_sha256': {}},
            'session': 'fixture', 'attempt': {'step': 1}}


def bundle(chunk_bytes=64 << 20):
    return target.extract(payload(), provenance={'test': True}, chunk_bytes=chunk_bytes)


class ScalarQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()

    def setUp(self):
        trap = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        trap.start()
        self.addCleanup(trap.stop)

    def test_full_backings_offsets_aliases_native_positions_and_roundtrip(self):
        source = payload()
        b = target.extract(source, provenance={})
        weight, bias = b['inputs']['gradients'].values()
        self.assertEqual((weight.storage_offset(), weight.stride(), weight.untyped_storage().nbytes()),
                         (37, (3, 1), 844))
        self.assertEqual(weight.untyped_storage()._cdata, bias.untyped_storage()._cdata)
        self.assertNotEqual(weight.untyped_storage()._cdata,
                            source['final_dense']['gradients']['weight'].untyped_storage()._cdata)
        self.assertEqual(b['inputs']['parameters']['weight'].stride(), (1, 4))
        self.assertEqual(b['storage_bytes'], 844 + 192 + 32)
        position = b['native_batch_position']
        self.assertEqual((position['row_start'], position['row_count'], position['batch_rows']), (2, 4, 6))
        self.assertEqual(b['row_names'], ['parameters/weight', 'parameters/bias', 'gradients/weight', 'gradients/bias'])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bundle.pt'
            torch.save(b, p)
            loaded = target.prepare(torch.load(p, map_location='cpu', weights_only=False))
            self.assertEqual(loaded['input_manifest'], b['input_manifest'])

    def test_tampered_backing_padding_and_reference_rejected(self):
        b = bundle()
        target.support.raw_bytes(b['inputs']['gradients']['weight'])[0] ^= 1
        with self.assertRaisesRegex(ValueError, 'Full input bytes'):
            target.prepare(b)
        b = bundle()
        b['reference_endpoints'][0, 0] = 42
        with self.assertRaisesRegex(ValueError, 'CPU reference'):
            target.prepare(b)

    def test_native_order_layout_and_metadata_mismatches_rejected(self):
        for mutation in ('order', 'layout', 'requires_grad', 'metadata', 'position'):
            b = bundle()
            if mutation == 'order':
                b['source_scan']['summaries'].reverse()
            elif mutation == 'layout':
                b['source_scan']['summaries'][0]['storage_offset'] += 1
            elif mutation == 'requires_grad':
                b['source_specs']['parameters/weight']['requires_grad'] = False
            elif mutation == 'metadata':
                b['enqueue_metadata']['kind'] = 'different'
            else:
                b['native_batch_position']['row_start'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                target.prepare(b)

    def test_empty_integer_nonfinite_views_rejected(self):
        for replacement in (torch.empty(0), torch.ones(3, dtype=torch.int64),
                            torch.tensor([float('nan')]), torch.tensor([float('inf')])):
            p = payload()
            p['final_dense']['gradients']['bias'] = replacement
            with self.assertRaises(ValueError):
                target.extract(p, provenance={})

    def test_actual_frozen_methods_read_once_retain_endpoints_and_restore_reader(self):
        b = bundle(chunk_bytes=16)
        original = target.frozen._read_endpoint_batch
        with mock.patch.object(target.frozen, '_read_endpoint_batch', wraps=original) as reader:
            output, metadata = target.replay_once(b['inputs'], b)
            self.assertIs(target.frozen._read_endpoint_batch, reader)
            self.assertEqual(reader.call_count, 1)
        self.assertIs(target.frozen._read_endpoint_batch, original)
        self.assertTrue(metadata['frozen_pending_cleared'])
        self.assertEqual(len(output['endpoints']), 4)
        self.assertFalse(target.analyze_observation(output, metadata, b)['failed'])
        with mock.patch.object(target.frozen, '_read_endpoint_batch', side_effect=RuntimeError('injected')) as reader:
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                target.replay_once(b['inputs'], b)
            self.assertIs(target.frozen._read_endpoint_batch, reader)
        self.assertIs(target.frozen._read_endpoint_batch, original)

    def test_endpoint_faults_propagate_without_false_stack_or_readback_blame(self):
        b = bundle()
        original = target.frozen.DeferredScalarQueue.enqueue
        for fault in ('min_gt_max', 'lag_two_max', 'nan', 'inf', 'huge'):
            def injected(queue, value, metadata):
                result = original(queue, value, metadata)
                endpoints = [endpoint for group in queue.pending.values() for _, endpoint in group]
                if fault == 'min_gt_max':
                    endpoints[0].copy_(torch.tensor([3., 2.], dtype=torch.float64))
                elif fault == 'lag_two_max':
                    endpoints[2][1] = endpoints[0][1]
                else:
                    endpoints[0][1] = {'nan': float('nan'), 'inf': float('inf'), 'huge': 1e30}[fault]
                return result
            with self.subTest(fault=fault), mock.patch.object(target.frozen.DeferredScalarQueue, 'enqueue', injected):
                output, metadata = target.replay_once(b['inputs'], b)
                check = target.analyze_observation(output, metadata, b)
                self.assertTrue(check['errors']['per_tensor_endpoint_disagreement'])
                for key in ('stack_or_row_mapping_disagreement', 'readback_disagreement',
                            'summary_row_mapping_disagreement', 'metadata_disagreement'):
                    self.assertEqual(check['errors'][key], [])
                target.clean_json(check)

    def test_stacked_batch_row_shift_distinct_from_good_endpoints(self):
        b = bundle()
        original = torch.stack
        def shifted(values, *args, **kwargs):
            result = original(values, *args, **kwargs)
            if result.shape == (4, 2):
                result = result.roll(1, 0)
            return result
        with mock.patch.object(torch, 'stack', side_effect=shifted):
            output, metadata = target.replay_once(b['inputs'], b)
        errors = target.analyze_observation(output, metadata, b)['errors']
        self.assertEqual(errors['per_tensor_endpoint_disagreement'], [])
        self.assertEqual(errors['stack_or_row_mapping_disagreement'], [0, 1, 2, 3])
        self.assertEqual(errors['readback_disagreement'], [])
        self.assertEqual(errors['summary_row_mapping_disagreement'], [])

    def test_cpu_readback_row_shift_distinct_from_good_batch(self):
        b = bundle()
        original = target.frozen._read_endpoint_batch
        def shifted(batch):
            rows = original(batch)
            return rows[1:] + rows[:1]
        with mock.patch.object(target.frozen, '_read_endpoint_batch', side_effect=shifted):
            output, metadata = target.replay_once(b['inputs'], b)
        errors = target.analyze_observation(output, metadata, b)['errors']
        self.assertEqual(errors['per_tensor_endpoint_disagreement'], [])
        self.assertEqual(errors['stack_or_row_mapping_disagreement'], [])
        self.assertEqual(errors['readback_disagreement'], [0, 1, 2, 3])
        self.assertEqual(errors['summary_row_mapping_disagreement'], [])

    def test_native_derived_summary_and_metadata_faults_detected(self):
        b = bundle()
        output, metadata = target.replay_once(b['inputs'], b)
        for field, value in [('min', 8), ('finite', False), ('extreme', True), ('max_abs_finite', 42)]:
            bad = copy.deepcopy(metadata)
            bad['resolved_scans'][0]['summaries'][0][field] = value
            self.assertEqual(target.analyze_observation(output, bad, b)['errors']['summary_row_mapping_disagreement'], [0])
        for mutation in ('kind', 'stage', 'layout', 'requires_grad', 'flagged', 'lifecycle'):
            bad = copy.deepcopy(metadata)
            if mutation in ('kind', 'stage'):
                bad['queued_scan'][mutation] = 'wrong'
            elif mutation == 'layout':
                bad['resolved_scans'][0]['summaries'][0]['stride'] = [3, 1]
            elif mutation == 'requires_grad':
                bad['queued_scan']['summaries'][0]['requires_grad'] = False
            elif mutation == 'flagged':
                bad['resolved_scans'][0]['flagged_names'] = ['parameters/weight']
            else:
                bad['resolved_scans'][0]['values_resolved_on_host'] = False
            with self.subTest(mutation=mutation):
                self.assertTrue(target.analyze_observation(output, bad, b)['errors']['metadata_disagreement'])

    def test_success_full_input_bytes_and_no_fault_artifact(self):
        b = bundle()
        calls = []
        def once(inputs, bundle):
            calls.append(1)
            return target.replay_once(inputs, bundle)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            report = target.run(b, failure_path=p, repeats=3, device='cpu', function=once)
            self.assertEqual((report['status'], len(calls)), ('PASS', 3))
            self.assertTrue(report['final_input_bytes_unchanged'])
            self.assertFalse(p.exists())

    def test_first_failure_retained_without_another_enqueue_or_flush(self):
        b = bundle()
        calls = []
        def once(inputs, bundle):
            calls.append(1)
            output, metadata = target.replay_once(inputs, bundle)
            if len(calls) == 2:
                output['batch'][0, 1] = 1e30
            return output, metadata
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            report = target.run(b, failure_path=p, repeats=5, device='cpu', function=once)
            self.assertEqual((report['status'], report['iterations_completed'], len(calls)), ('FAIL', 2, 2))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertEqual(saved['first_cpu_outputs']['batch'][0, 1], 1e30)
            self.assertTrue(saved['record']['full_input_bytes_unchanged'])
            self.assertTrue(saved['record']['output_bytes_stable_during_capture'])
            self.assertEqual(saved['current_at_failure']['inputs']['gradients']['weight'].storage_offset(), 37)
            self.assertEqual(saved['native_batch_position']['row_start'], 2)

    def test_input_mutation_saved_and_pristine_preserved(self):
        b = bundle()
        def mutate(inputs, bundle):
            output = target.replay_once(inputs, bundle)
            inputs['gradients']['weight'][0, 0] = 3
            return output
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            report = target.run(b, failure_path=p, repeats=3, device='cpu', function=mutate)
            self.assertEqual(report['status'], 'FAIL')
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertFalse(saved['record']['full_input_bytes_unchanged'])
            self.assertIn('gradients/weight', saved['record']['changed_input_versions'])
            self.assertNotEqual(saved['pristine_inputs']['gradients']['weight'][0, 0], 3)
            self.assertEqual(saved['current_at_failure']['inputs']['gradients']['weight'][0, 0], 3)

    def test_changed_capture_preserves_both_observations(self):
        b = bundle()
        calls = []
        def once(inputs, bundle):
            calls.append(1)
            output, metadata = target.replay_once(inputs, bundle)
            output['batch'][0, 1] = 1e30
            return output, metadata
        original = target._cpu_copy_tree
        def changed(tree, *args):
            result, size = original(tree, *args)
            if isinstance(tree, dict) and 'outputs' in tree:
                result['outputs']['batch'][0, 1] = 0
            return result, size
        with tempfile.TemporaryDirectory() as d, mock.patch.object(target, '_cpu_copy_tree', side_effect=changed):
            p = Path(d) / 'fault.pt'
            report = target.run(b, failure_path=p, repeats=3, device='cpu', function=once)
            self.assertEqual((report['status'], len(calls)), ('CAPTURE_INTEGRITY_FAILURE', 1))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertEqual(saved['first_cpu_outputs']['batch'][0, 1], 1e30)
            self.assertEqual(saved['current_at_failure']['outputs']['batch'][0, 1], 0)

    def test_bad_initial_backing_restore_rejected_before_queue(self):
        b = bundle()
        original = target.restore_raw_tree
        def bad_restore(*args, **kwargs):
            tree, size = original(*args, **kwargs)
            target.support.raw_bytes(tree['gradients']['weight'])[0] ^= 1
            return tree, size
        with tempfile.TemporaryDirectory() as d, mock.patch.object(target, 'restore_raw_tree', side_effect=bad_restore):
            with self.assertRaisesRegex(ValueError, 'Initial restored full bytes'):
                target.run(b, failure_path=Path(d) / 'fault.pt', repeats=1, device='cpu',
                           function=lambda *a: self.fail('Queue must not execute'))

    def test_gpu_environment_gate_before_load_or_initialization(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict('os.environ', {}, clear=True):
            with mock.patch.object(torch, 'load', side_effect=AssertionError('load must not run')):
                with self.assertRaises(SystemExit):
                    target.main(['--bundle', str(Path(d) / 'in.pt'), '--report', str(Path(d) / 'out.json'),
                                 '--failure-dump', str(Path(d) / 'fault.pt'), '--gpu'])


if __name__ == '__main__':
    unittest.main()
