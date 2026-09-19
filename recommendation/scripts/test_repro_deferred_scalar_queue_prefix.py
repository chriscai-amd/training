"""CPU controls for the labeled prefix intervention and combined first capture."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import repro_deferred_scalar_queue_prefix as target
from test_repro_deferred_scalar_queue import bundle


class PrefixTests(unittest.TestCase):
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

    def test_prefix_exact_finite_pairs_separate_stores_original_bundle_unmodified(self):
        b = bundle()
        original_metadata = copy.deepcopy(b['source_scan'])
        original_manifest = copy.deepcopy(b['input_manifest'])
        c = target.prepare_control(b, 2)
        self.assertEqual(c['contracts'][0]['reference_endpoints'].tolist(), [[-1.25, 1.5], [-2.25, 2.5]])
        self.assertEqual(len(c['input_manifest']['stores']), len(original_manifest['stores']) + 2)
        self.assertEqual(c['input_manifest']['storage_bytes'], original_manifest['storage_bytes'] + 32)
        self.assertEqual(b['source_scan'], original_metadata)
        self.assertEqual(b['input_manifest'], original_manifest)
        self.assertTrue(c['intervention']['matches_original_final_dense_row_start'])
        self.assertTrue(c['intervention']['matches_original_batch_rows'])
        c = target.prepare_control(b, 229)
        self.assertEqual(c['intervention']['batch_rows'], 233)
        self.assertFalse(c['intervention']['matches_original_batch_rows'])
        for bad in (0, -1, 1001, 1.5):
            with self.assertRaises(ValueError):
                target.prepare_control(b, bad)

    def test_one_actual_combined_readback_two_scans_and_all_retained_endpoints(self):
        c = target.prepare_control(bundle(), 3)
        original = target.base.frozen._read_endpoint_batch
        with mock.patch.object(target.base.frozen, '_read_endpoint_batch', wraps=original) as reader:
            output, metadata = target.replay_once(c['inputs'], c)
            self.assertEqual(reader.call_count, 1)
            self.assertIs(target.base.frozen._read_endpoint_batch, reader)
        self.assertIs(target.base.frozen._read_endpoint_batch, original)
        self.assertEqual(output['batch'].shape, (7, 2))
        self.assertEqual(len(output['endpoints']), 7)
        self.assertEqual([s['enqueue_index'] for s in metadata['resolved_scans']], [0, 1])
        self.assertTrue(metadata['frozen_pending_cleared'])
        self.assertFalse(target.analyze_observation(output, metadata, c)['failed'])

    def test_prefix_endpoint_nan_propagation_not_misattributed_to_readback(self):
        c = target.prepare_control(bundle(), 3)
        original = target.base.frozen.DeferredScalarQueue.enqueue
        def injected(queue, inputs, metadata):
            result = original(queue, inputs, metadata)
            if metadata['kind'] == 'synthetic_endpoint_prefix':
                next(iter(queue.pending.values()))[1][1][1] = float('nan')
            return result
        with mock.patch.object(target.base.frozen.DeferredScalarQueue, 'enqueue', injected):
            output, metadata = target.replay_once(c['inputs'], c)
        check = target.analyze_observation(output, metadata, c)
        self.assertEqual(check['errors']['per_tensor_endpoint_disagreement'], [1])
        self.assertEqual(check['errors']['stack_or_row_mapping_disagreement'], [])
        self.assertEqual(check['errors']['readback_disagreement'], [])
        self.assertFalse(check['scan_checks'][1]['check']['failed'])

    def test_outer_batch_shift_across_prefix_final_dense_boundary(self):
        c = target.prepare_control(bundle(), 3)
        original = torch.stack
        def shifted(values, *args, **kwargs):
            result = original(values, *args, **kwargs)
            return result.roll(1, 0) if result.shape == (7, 2) else result
        with mock.patch.object(torch, 'stack', side_effect=shifted):
            output, metadata = target.replay_once(c['inputs'], c)
        errors = target.analyze_observation(output, metadata, c)['errors']
        self.assertEqual(errors['per_tensor_endpoint_disagreement'], [])
        self.assertEqual(errors['stack_or_row_mapping_disagreement'], list(range(7)))
        self.assertEqual(errors['readback_disagreement'], [])

    def test_combined_readback_shift_and_original_actual_enqueue_indices_checked(self):
        c = target.prepare_control(bundle(), 3)
        original = target.base.frozen._read_endpoint_batch
        def shifted(batch):
            rows = original(batch)
            return rows[1:] + rows[:1]
        with mock.patch.object(target.base.frozen, '_read_endpoint_batch', side_effect=shifted):
            output, metadata = target.replay_once(c['inputs'], c)
        errors = target.analyze_observation(output, metadata, c)['errors']
        self.assertEqual(errors['stack_or_row_mapping_disagreement'], [])
        self.assertEqual(errors['readback_disagreement'], list(range(7)))
        metadata['resolved_scans'][1]['enqueue_index'] = 9
        self.assertIn('scan1:actual_combined_enqueue_index',
                      target.analyze_observation(output, metadata, c)['errors']['metadata_disagreement'])
        self.assertEqual(metadata['resolved_scans'][1]['enqueue_index'], 9)

    def test_full_combined_bytes_checked_and_first_failure_without_rerun(self):
        c = target.prepare_control(bundle(), 3)
        calls = []
        def injected(inputs, control):
            calls.append(1)
            output, metadata = target.replay_once(inputs, control)
            if len(calls) == 2:
                output['batch'][3, 1] = 1e30
            return output, metadata
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            result = target.run(c, failure_path=p, repeats=4, device='cpu', function=injected)
            self.assertEqual((result['status'], result['iterations_completed'], len(calls)), ('FAIL', 2, 2))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertEqual(saved['first_cpu_outputs']['batch'].shape, (7, 2))
            self.assertEqual(saved['first_cpu_outputs']['batch'][3, 1], 1e30)
            self.assertTrue(saved['record']['full_input_bytes_unchanged'])
            self.assertEqual(saved['original_native_batch_position']['row_start'], 2)
            self.assertEqual(saved['intervention']['final_dense_row_start'], 3)
            self.assertEqual([s['enqueue_index'] for s in saved['native_metadata_and_rows']['resolved_scans']], [0, 1])

    def test_prefix_input_mutation_saved_with_pristine_prefix_bytes(self):
        c = target.prepare_control(bundle(), 3)
        def mutated(inputs, control):
            result = target.replay_once(inputs, control)
            inputs['prefix']['synthetic_prefix']['row_0000'][0] = -99
            return result
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            result = target.run(c, failure_path=p, repeats=3, device='cpu', function=mutated)
            self.assertEqual((result['status'], result['iterations_completed']), ('FAIL', 1))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertFalse(saved['record']['full_input_bytes_unchanged'])
            self.assertEqual(saved['pristine_inputs']['prefix']['synthetic_prefix']['row_0000'][0], -1.25)
            self.assertEqual(saved['current_at_failure']['inputs']['prefix']['synthetic_prefix']['row_0000'][0], -99)

    def test_changed_capture_preserves_first_and_later_combined_batch(self):
        c = target.prepare_control(bundle(), 3)
        def injected(inputs, control):
            output, metadata = target.replay_once(inputs, control)
            output['batch'][3, 1] = 1e30
            return output, metadata
        original = target.base._cpu_copy_tree
        def changed(tree, *args):
            result, size = original(tree, *args)
            if isinstance(tree, dict) and 'outputs' in tree:
                result['outputs']['batch'][3, 1] = 0
            return result, size
        with tempfile.TemporaryDirectory() as d, mock.patch.object(target.base, '_cpu_copy_tree', side_effect=changed):
            p = Path(d) / 'fault.pt'
            result = target.run(c, failure_path=p, repeats=3, device='cpu', function=injected)
            self.assertEqual((result['status'], result['iterations_completed']), ('CAPTURE_INTEGRITY_FAILURE', 1))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertEqual(saved['first_cpu_outputs']['batch'][3, 1], 1e30)
            self.assertEqual(saved['current_at_failure']['outputs']['batch'][3, 1], 0)

    def test_success_checks_prefix_and_final_dense_full_bytes(self):
        c = target.prepare_control(bundle(), 3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            result = target.run(c, failure_path=p, repeats=2, device='cpu')
            self.assertEqual(result['status'], 'PASS')
            self.assertTrue(result['final_input_bytes_unchanged'])
            self.assertFalse(p.exists())

    def test_gpu_gate_before_loading_or_initialization(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict('os.environ', {}, clear=True):
            with mock.patch.object(torch, 'load', side_effect=AssertionError('load must not run')):
                with self.assertRaises(SystemExit):
                    target.main(['--bundle', str(Path(d) / 'in.pt'), '--report', str(Path(d) / 'out.json'),
                                 '--failure-dump', str(Path(d) / 'fault.pt'), '--gpu'])


if __name__ == '__main__':
    unittest.main()
