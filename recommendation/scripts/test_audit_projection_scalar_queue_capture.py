"""CPU injected-stage and metadata regressions for the bounded saved audit."""
import copy
import json
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest import mock

import torch
import audit_projection_scalar_queue_capture as target
from capture_linear_projection_diagnostics import DiagnosticScalarQueue


def fixture():
    storage = torch.arange(18, dtype=torch.float32).sub(5).div(8)
    final_dense = {'parameters': {'weight': storage[2:6].reshape(2, 2).t(), 'bias': storage[7:9]},
                   'gradients': {'weight': torch.tensor([[0., 0.], [0., 0.]]), 'bias': torch.tensor([0., 0.])}}
    queue = DiagnosticScalarQueue(chunk_bytes=4096, abs_threshold=1e6)
    queue.enqueue({'prefix': torch.tensor([0., 4.]), 'integer': torch.tensor([1])}, {'stage': 'prefix'})
    queue.enqueue({'mismatch': torch.tensor(0.)}, {'stage': 'comparison', 'expected_zero': True})
    queue.enqueue(final_dense, {'stage': target.FINAL_STAGE})
    final = queue.flush()
    return {'attempt': {'step': 1}, 'capture_variant': 'fixture', 'scalar_queue_diagnostics': queue.diagnostics,
            'metadata': {'deferred_scans': final}, 'final_dense': final_dense}


class SavedScalarAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.torch_rng = torch.get_rng_state().clone()
        cls.python_rng = random.getstate()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.torch_rng, torch.get_rng_state())
        assert cls.python_rng == random.getstate()
        assert not torch.cuda.is_initialized()

    def setUp(self):
        guard = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_healthy_padding_noncontiguous_and_complete_mapping(self):
        result = target.audit_payload(fixture())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['scan_count'], 3)
        self.assertEqual(result['counts']['endpoint_to_batch']['rows'], 6)
        self.assertEqual(result['counts']['final_dense_CPU_to_endpoint']['rows'], 4)
        self.assertEqual(result['final_dense_backings'], 3)
        self.assertEqual(result['metadata_errors'], [])

    def test_faults_at_each_retained_stage_are_separately_reported(self):
        cases = {'endpoint': ('endpoint_to_batch',),
                 'batch': ('endpoint_to_batch', 'batch_to_returned'),
                 'returned': ('batch_to_returned', 'returned_to_native_summary', 'returned_to_final_summary'),
                 'native_summary': ('returned_to_native_summary',),
                 'final_summary': ('returned_to_final_summary',)}
        for stage, expected in cases.items():
            with self.subTest(stage=stage):
                payload = fixture()
                group = payload['scalar_queue_diagnostics']['groups'][0]
                if stage == 'endpoint': group['endpoint_tensors'][0][1] = 42
                elif stage == 'batch': group['batch_tensor'][0, 1] = 42
                elif stage == 'returned': group['returned_rows'][0][1] = 42
                elif stage == 'native_summary': payload['scalar_queue_diagnostics']['native_resolved_scans'][0]['summaries'][0]['max'] = 42
                else: payload['metadata']['deferred_scans'][0]['summaries'][0]['max'] = 42
                result = target.audit_payload(payload)
                self.assertEqual(result['status'], 'MISMATCH')
                actual = tuple(name for name, rows in result['mismatches'].items() if rows)
                self.assertEqual(actual, expected)

    def test_signed_zero_difference_is_raw_only(self):
        payload = fixture();group = payload['scalar_queue_diagnostics']['groups'][0]
        group['batch_tensor'][0, 0] = -0.0
        result = target.audit_payload(payload)
        for stage in ('endpoint_to_batch', 'batch_to_returned'):
            self.assertEqual(result['counts'][stage]['nan_aware_mismatches'], 0)
            self.assertEqual(result['counts'][stage]['raw_FP64_mismatches'], 1)
        self.assertEqual(result['min_above_max'], [])

    def test_raw_nan_payloads_and_textual_nan_comparability(self):
        raw_a, raw_b = 0x7ff8000000000001, 0x7ff8000000000002
        a = torch.tensor([raw_a, raw_a], dtype=torch.int64).view(torch.float64)
        b = torch.tensor([raw_b, raw_a], dtype=torch.int64).view(torch.float64)
        result = target.comparison(target.tensor_pair(a), target.tensor_pair(b))
        self.assertTrue(result['nan_aware_equal'])
        self.assertFalse(result['raw_FP64_equal'])
        self.assertEqual(result['left_bits'][0], '7ff8000000000001')
        text = target.comparison(target.tensor_pair(a), target.python_pair(['nan', 'nan'], summary=True))
        self.assertTrue(text['nan_aware_equal'])
        self.assertIsNone(text['raw_FP64_equal'])

    def test_inverted_finite_and_infinite_intervals_reported(self):
        for pair in ([5., -2.], [float('inf'), -float('inf')]):
            payload = fixture();payload['scalar_queue_diagnostics']['groups'][0]['endpoint_tensors'][0][:] = torch.tensor(pair)
            result = target.audit_payload(payload)
            self.assertEqual(len(result['min_above_max']), 1)
            self.assertEqual(result['min_above_max'][0]['stage'], 'endpoint')

    def test_independent_final_dense_values_distinguish_later_input_change(self):
        payload = fixture();payload['final_dense']['parameters']['weight'][0, 0] = -77
        result = target.audit_payload(payload)
        for stage in ('endpoint_to_batch', 'batch_to_returned', 'returned_to_native_summary'):
            self.assertEqual(result['mismatches'][stage], [])
        for stage in ('endpoint', 'batch', 'returned', 'native_summary'):
            self.assertEqual(len(result['mismatches']['final_dense_CPU_to_' + stage]), 1)

    def test_derived_metadata_and_flags_checked(self):
        for field in ('finite', 'extreme', 'max_abs_finite', 'flagged_names'):
            payload = fixture();scan = payload['scalar_queue_diagnostics']['native_resolved_scans'][0]
            if field == 'flagged_names': scan[field] = ['prefix']
            else: scan['summaries'][0][field] = {'finite': False, 'extreme': True, 'max_abs_finite': 123}[field]
            result = target.audit_payload(payload)
            self.assertTrue(result['metadata_errors'])
            self.assertEqual(result['status'], 'MISMATCH')

    def test_native_expected_zero_flags_remain_distinct_from_final_flags(self):
        payload = fixture();group = payload['scalar_queue_diagnostics']['groups'][0]
        group['endpoint_tensors'][1].fill_(1);group['batch_tensor'][1].fill_(1);group['returned_rows'][1] = [1., 1.]
        for scan in (payload['scalar_queue_diagnostics']['native_resolved_scans'][1], payload['metadata']['deferred_scans'][1]):
            scan['summaries'][0].update(min=1., max=1., max_abs_finite=1.)
        payload['metadata']['deferred_scans'][1]['flagged_names'] = ['mismatch']
        result = target.audit_payload(payload)
        self.assertEqual(result['status'], 'PASS')

    def test_missing_duplicate_reordered_rows_and_layout_rejected(self):
        for kind in ('mapping', 'missing', 'devices', 'order', 'layout'):
            payload = fixture();d = payload['scalar_queue_diagnostics'];group = d['groups'][0]
            if kind == 'mapping': group['row_mapping'][0]['enqueue_index'] = 2
            elif kind == 'missing': group['returned_rows'].pop()
            elif kind == 'devices': d['groups'].append(copy.deepcopy(group));d['reader_calls'] = 2
            elif kind == 'order': d['native_resolved_scans'][0]['enqueue_index'] = 7
            else: payload['final_dense']['parameters']['weight'] = payload['final_dense']['parameters']['weight'].contiguous()
            with self.subTest(kind=kind), self.assertRaises(ValueError): target.audit_payload(payload)

    def test_bounds_reject_before_reference_reductions(self):
        payload = fixture()
        with mock.patch.object(torch, 'amin', side_effect=AssertionError('Read before budget guard')):
            with self.assertRaisesRegex(ValueError, 'byte budget'): target.audit_payload(payload, max_dense_bytes=1)
            with self.assertRaisesRegex(ValueError, 'row budget'): target.audit_payload(payload, max_rows=3)

    def test_cli_mmap_certificate_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory);path = directory/'capture.pt';report = directory/'audit.json';certificate=directory/'full.json'
            torch.save(fixture(), path)
            certificate.write_text(json.dumps({'payload': str(path), 'bytes': path.stat().st_size, 'sha256': target.file_record(path)['sha256']}))
            self.assertEqual(target.main([str(path), '--report', str(report), '--full-payload-certificate', str(certificate)]), 0)
            result = json.loads(report.read_text())
            self.assertEqual(result['status'], 'PASS')
            self.assertFalse(result['payload']['whole_hash_recomputed_in_this_audit'])
            with self.assertRaises(SystemExit): target.main([str(path), '--report', str(report)])


if __name__ == '__main__':
    unittest.main()
