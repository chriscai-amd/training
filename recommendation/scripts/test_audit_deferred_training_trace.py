"""CPU checks using actual frozen deferred-monitor event records."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from audit_deferred_training_trace import DeferredEventAudit, main
from nan_backward_boundaries import BoundaryAnomalyError
from nan_backward_training_deferred import DeferredAllLayersTrainingBackwardProbe
from test_nan_backward_training_extended import make_full_modules
from test_nan_backward_training_upstream import Stack


class DeferredAuditTest(unittest.TestCase):
    def events(self, fault=False):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "NAN_BACKWARD_TARGET": "*", "NAN_BACKWARD_SAVE_ALL": "0",
            "NAN_BACKWARD_ABS_THRESHOLD": "1e20", "NAN_BACKWARD_CHUNK_MIB": "1",
            "NAN_BACKWARD_MAX_CAPTURE_GIB": "32"}):
            compute, preprocess, output, state = make_full_modules()
            model = Stack(compute)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
            probe = DeferredAllLayersTrainingBackwardProbe(
                model, directory, optimizer=optimizer, compute_module=compute,
                preprocess_module=preprocess, output_module=output)
            try:
                probe.set_attempt("training", 0, 1)
                state.extreme_attention = fault
                x = (torch.arange(12, dtype=torch.float64).reshape(3, 4) / 64).requires_grad_()
                if fault:
                    with self.assertRaises(BoundaryAnomalyError):
                        model(x).sum().backward()
                else:
                    model(x).sum().backward()
                    probe.sentinel.optimizer_proxy.step()
            finally:
                probe.close()
            return [json.loads(line) for line in (Path(directory) / 'boundaries.jsonl').read_text().splitlines()]

    @staticmethod
    def audited(events):
        audit = DeferredEventAudit()
        for event in events:
            audit.consume(event)
        return audit

    def test_real_queue_and_resolved_trace_have_exact_coverage_and_distinct_state(self):
        audit = self.audited(self.events())
        report = audit.report({"status": "bounded_complete", "completed_steps": 1})
        attempt, = report['resolved_attempts']
        self.assertEqual(attempt['resolved_scans'], 44)
        self.assertEqual(attempt['queued_operation_endpoints'], 42)
        self.assertEqual(attempt['queued_dense_scans'], 2)
        self.assertEqual(attempt['unique_paired_calls'], 21)
        self.assertEqual(attempt['missing_gradient_names'], [])
        self.assertEqual(report['observed_reduction_streams'], [('cpu', None)])
        self.assertIsNone(report['first_flagged_observation_in_host_enqueue_order'])
        self.assertIn('No explicit optimizer', report['placement_evidence']['direct_event_observation'])

    def test_first_flagged_enqueue_index_identifies_attention_output(self):
        report = self.audited(self.events(fault=True)).report()
        first = report['first_flagged_observation_in_host_enqueue_order']
        self.assertEqual((first['operation'], first['phase']), ('hstu_attention_bwd', 'after'))
        self.assertEqual(first['flagged_names'], ['dq'])
        self.assertGreater(report['resolved_attempts'][0]['flagged_scan_count'], 1)

    def test_pairing_error_detected_even_when_queued_and_resolved_metadata_agree(self):
        events = self.events()
        queued = next(event for event in events if event['event'] == 'after')
        index = queued['enqueue_index']
        queued['call'] = 999
        flush = next(event for event in events if event['event'] == 'deferred_flush')
        flush['scans'][index]['call'] = 999
        with self.assertRaisesRegex(ValueError, 'call IDs'):
            self.audited(events)

    def test_stream_identity_and_resolved_metadata_checks(self):
        events = self.events()
        invalid = copy.deepcopy(events)
        queued = next(event for event in invalid if event['event'] == 'before')
        queued['summaries'][0]['stream_id'] = 7
        flush = next(event for event in invalid if event['event'] == 'deferred_flush')
        flush['scans'][queued['enqueue_index']]['summaries'][0]['stream_id'] = 7
        with self.assertRaisesRegex(ValueError, 'stream identity'):
            self.audited(invalid)
        # CPU-created records exercise multi-stream metadata validation only;
        # they are not represented as an actual multi-stream GPU run.
        for event in events:
            if event['event'] in ('before', 'after', 'dense_phase_queued'):
                for item in event['summaries']:
                    item['source_device'], item['stream_id'] = 'cuda:0', event['enqueue_index'] % 2 + 100
            if event['event'] == 'deferred_flush':
                for scan in event['scans']:
                    for item in scan['summaries']:
                        item['source_device'], item['stream_id'] = 'cuda:0', scan['enqueue_index'] % 2 + 100
        report = self.audited(events).report()
        self.assertEqual(report['observed_reduction_stream_count'], 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_auditor_import_uses_no_torch(self):
        code = "import sys; import audit_deferred_training_trace; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_live_report_hash_covers_only_complete_parsed_records(self):
        events = self.events()
        complete = b''.join(json.dumps(event).encode() + b'\n' for event in events)
        trailing = b'{"event":"still_writing'
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'boundaries').mkdir()
            (directory / 'boundaries/boundaries.jsonl').write_bytes(complete + trailing)
            (directory / 'outcome.json').write_text(json.dumps({'status': 'running'}))
            target = directory / 'review.json'
            arguments = ['--directory', str(directory), '--report', str(target)]
            main(arguments)
            report = json.loads(target.read_text())
            self.assertEqual(report['parsed_events_prefix_sha256'], hashlib.sha256(complete).hexdigest())
            self.assertEqual(report['parsed_events_prefix_bytes'], len(complete))
            self.assertEqual(report['unparsed_event_bytes_at_report'], len(trailing))
            self.assertEqual(report['resolved_attempt_count'], 1)
            with self.assertRaises(FileExistsError):
                main(arguments)
            target.unlink()
            (directory / 'outcome.json').write_text(json.dumps({'status': 'bounded_complete'}))
            with self.assertRaisesRegex(ValueError, 'incomplete record'):
                main(arguments)


if __name__ == '__main__':
    unittest.main()
