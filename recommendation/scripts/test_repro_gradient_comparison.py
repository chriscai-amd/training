"""CPU-only tests for preserved layouts, stage attribution and first failure."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import repro_gradient_comparison as target


def payload():
    bucket = torch.arange(211, dtype=torch.float32).div(1024)
    weight = bucket[37:49].reshape(3, 4)
    bias = bucket[57:60]
    return {'format': 'test_fixture', 'stages': {'parameter_post_weight': weight.clone(),
             'parameter_post_bias': bias.clone(), 'after_backward_return': {'weight': weight, 'bias': bias}},
            'metadata': {'source_sha256': {}}, 'session': 'fixture', 'attempt': {'step': 1}}


def bundle(chunk_bytes=64 << 20):
    return target.extract(payload(), provenance={'test': True}, chunk_bytes=chunk_bytes)


class ComparisonTests(unittest.TestCase):
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

    def test_extraction_full_backings_offsets_aliases_and_roundtrip(self):
        source = payload()
        b = target.extract(source, provenance={'test': True})
        actual = b['inputs']['weight']['actual']
        bias = b['inputs']['bias']['actual']
        self.assertEqual((actual.storage_offset(), actual.stride(), actual.untyped_storage().nbytes()), (37, (4, 1), 844))
        self.assertEqual(actual.untyped_storage()._cdata, bias.untyped_storage()._cdata)
        self.assertNotEqual(actual.untyped_storage()._cdata, source['stages']['after_backward_return']['weight'].untyped_storage()._cdata)
        self.assertEqual(b['storage_bytes'], 844 + 48 + 12)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bundle.pt'
            torch.save(b, p)
            loaded = target.prepare(torch.load(p, map_location='cpu', weights_only=False))
            self.assertEqual(loaded['input_manifest'], b['input_manifest'])

    def test_tampered_unselected_backing_padding_rejected(self):
        b = bundle()
        target.raw_bytes(b['inputs']['weight']['actual'])[0] ^= 1
        with self.assertRaisesRegex(ValueError, 'full bytes'):
            target.prepare(b)

    def test_nonfinite_or_unequal_selected_inputs_rejected(self):
        for value in (float('nan'), float('inf'), 8.0):
            p = payload()
            p['stages']['parameter_post_weight'][0, 0] = value
            with self.assertRaises(ValueError):
                target.extract(p, provenance={})

    def test_compact_control_preserves_logical_bytes_and_labels_intervention(self):
        b = bundle()
        c = target.layout_control(b, 'compact')
        self.assertEqual(c['inputs']['weight']['actual'].storage_offset(), 0)
        self.assertEqual(c['storage_bytes'], 120)
        self.assertNotEqual(c['inputs']['weight']['actual'].untyped_storage()._cdata,
                            c['inputs']['bias']['actual'].untyped_storage()._cdata)
        self.assertIn('layout_intervention', c['provenance'])
        self.assertEqual(b['inputs']['weight']['actual'].storage_offset(), 37)

    def test_frozen_sequence_retains_every_chunk_and_one_call_per_attempt(self):
        b = bundle(chunk_bytes=16)
        calls = []
        def compare(a, z, **kw):
            calls.append(1)
            out = target.compare_once(a, z, **kw)
            self.assertEqual(len(out['chunks']), 3)
            return out
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            r = target.run(b, failure_path=p, repeats=3, device='cpu', compare_function=compare)
            self.assertEqual((r['status'], len(calls)), ('PASS', 3))
            self.assertTrue(r['final_input_bytes_unchanged'])
            self.assertFalse(p.exists())

    def test_elementwise_vs_reduction_vs_scalar_attribution(self):
        b = bundle()
        pair = b['inputs']['weight']
        for stage, expected in [('mask', 'elementwise_equality_disagreement:0'),
                                ('all', 'all_reduction_disagreement:0'),
                                ('running_equal', 'logical_and_disagreement:0'),
                                ('mismatch', 'logical_not_or_float_conversion_disagreement')]:
            out = target.compare_once(pair['expected'], pair['actual'], chunk_bytes=b['comparison_chunk_bytes'])
            if stage == 'mask':
                out['chunks'][0]['mask'][0, 0] = False
            elif stage == 'mismatch':
                out['mismatch'].fill_(1)
            else:
                out['chunks'][0][stage].fill_(False)
            result = target.analyze_observation(out, pair, chunk_bytes=b['comparison_chunk_bytes'])
            self.assertIn(expected, result['classifications'])

    def test_bad_mask_detected_even_if_final_gpu_scalar_claims_equal(self):
        b = bundle()
        calls = []
        def compare(a, z, **kw):
            calls.append(1)
            out = target.compare_once(a, z, **kw)
            out['chunks'][0]['mask'][1, 2] = False
            return out
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            r = target.run(b, failure_path=p, repeats=4, device='cpu', compare_function=compare)
            self.assertEqual((r['status'], len(calls)), ('FAIL', 1))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertEqual(saved['first_cpu_observation']['mismatch'].item(), 0)
            self.assertFalse(saved['current_at_failure']['observation']['chunks'][0]['mask'][1, 2])
            self.assertTrue(saved['record']['full_input_bytes_unchanged'])
            self.assertEqual(saved['current_at_failure']['inputs']['weight']['actual'].storage_offset(), 37)

    def test_first_bad_reduction_saved_without_rerun(self):
        b = bundle()
        calls = []
        def compare(a, z, **kw):
            calls.append(1)
            out = target.compare_once(a, z, **kw)
            if len(calls) == 2:
                out['chunks'][0]['all'].fill_(False)
                out['chunks'][0]['running_equal'].fill_(False)
                out['mismatch'].fill_(1)
            return out
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            r = target.run(b, failure_path=p, repeats=4, device='cpu', compare_function=compare)
            self.assertEqual((r['status'], r['iterations_completed'], len(calls)), ('FAIL', 2, 2))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertIn('all_reduction_disagreement:0', saved['record']['check']['classifications'])
            self.assertTrue(bool(saved['first_cpu_observation']['chunks'][0]['mask'].all()))

    def test_input_mutation_and_capture_change_remain_visible(self):
        b = bundle()
        def mutate(a, z, **kw):
            out = target.compare_once(a, z, **kw)
            a[0, 0] = 3
            return out
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            r = target.run(b, failure_path=p, repeats=3, device='cpu', compare_function=mutate)
            self.assertEqual(r['status'], 'FAIL')
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertFalse(saved['record']['full_input_bytes_unchanged'])
            self.assertNotEqual(saved['pristine_inputs']['weight']['expected'][0, 0], 3)
            self.assertEqual(saved['current_at_failure']['inputs']['weight']['expected'][0, 0], 3)

    def test_changed_failure_observation_is_not_reported_as_stable_capture(self):
        b = bundle()
        calls = []
        def compare(a, z, **kw):
            calls.append(1)
            out = target.compare_once(a, z, **kw)
            out['mismatch'].fill_(1)
            return out
        original = target._cpu_copy_tree
        def altered_copy(tree, *args):
            result, size = original(tree, *args)
            if isinstance(tree, dict) and 'observation' in tree:
                result['observation']['mismatch'].zero_()
            return result, size
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fault.pt'
            with mock.patch.object(target, '_cpu_copy_tree', side_effect=altered_copy):
                r = target.run(b, failure_path=p, repeats=3, device='cpu', compare_function=compare)
            self.assertEqual((r['status'], len(calls)), ('CAPTURE_INTEGRITY_FAILURE', 1))
            saved = torch.load(p, map_location='cpu', weights_only=False)
            self.assertEqual(saved['first_cpu_observation']['mismatch'].item(), 1)
            self.assertEqual(saved['current_at_failure']['observation']['mismatch'].item(), 0)
            self.assertFalse(saved['record']['observation_bytes_stable_during_capture'])

    def test_bad_initial_full_restore_is_rejected_before_comparison(self):
        b = bundle()
        original = target.restore_raw_tree
        def bad_restore(*args, **kwargs):
            tree, size = original(*args, **kwargs)
            target.raw_bytes(tree['weight']['actual'])[0] ^= 1
            return tree, size
        with tempfile.TemporaryDirectory() as d, mock.patch.object(target, 'restore_raw_tree', side_effect=bad_restore):
            with self.assertRaisesRegex(ValueError, 'Initial full restored'):
                target.run(b, failure_path=Path(d) / 'fault.pt', repeats=1, device='cpu',
                           compare_function=lambda *a, **k: self.fail('Comparison must not execute'))

    def test_gpu_environment_gate_before_load_or_initialization(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict('os.environ', {}, clear=True):
            with mock.patch.object(torch, 'load', side_effect=AssertionError('load must not run')):
                with self.assertRaises(SystemExit):
                    target.main(['--bundle', str(Path(d) / 'in.pt'), '--report', str(Path(d) / 'out.json'),
                                 '--failure-dump', str(Path(d) / 'fault.pt'), '--gpu'])


if __name__ == '__main__':
    unittest.main()
