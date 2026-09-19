"""CPU mapping/provenance tests built from frozen-monitor-generated events."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from audit_deferred_attention_bounds import audit_file, main
from audit_deferred_training_trace import SUMMARY_RESULTS
from test_audit_deferred_training_trace import DeferredAuditTest


class DeferredAttentionBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Actual frozen monitor generates the structural event schema on CPU.
        # Only attention tensor summaries/scalars are replaced by controlled
        # BF16 rank-three values below; these are not claimed GPU observations.
        cls.original = DeferredAuditTest().events()

    def events(self, spike=True):
        events = copy.deepcopy(self.original)
        flush = next(event for event in events if event['event'] == 'deferred_flush')
        for scan in flush['scans']:
            if scan.get('operation') != 'hstu_attention_bwd':
                continue
            scan['scalar_arguments'] = {'N': 4, 'alpha': 0.5, 'num_softmax_heads': 0,
                                        'enable_tma': False, 'max_attn_len': 0,
                                        'contextual_seq_len': 0}
            names = ('q', 'k', 'v', 'dout') if scan['phase'] == 'before' else ('dq', 'dk', 'dv')
            template = scan['summaries'][0]
            summaries = []
            for name in names:
                value = 1.0 if name in ('q', 'k', 'v') else 0.0
                if spike and name == 'dq' and scan['layer'].endswith('.2'):
                    value = 0.5
                summaries.append({**template, 'name': name, 'dtype': 'torch.bfloat16',
                                  'shape': [4, 2, 4], 'numel': 32, 'scanned': True,
                                  'min': value, 'max': value, 'max_abs_finite': value,
                                  'finite': True, 'extreme': False})
            scan['summaries'] = summaries
        return self.synchronize(events)

    @staticmethod
    def synchronize(events):
        flush = next(event for event in events if event['event'] == 'deferred_flush')
        queued = [event for event in events if event['event'] in ('before', 'after', 'dense_phase_queued')]
        first = None
        for index, (scan, event) in enumerate(zip(flush['scans'], queued)):
            scan['enqueue_index'] = index
            scan['flagged_names'] = [item['name'] for item in scan['summaries']
                                     if not item['finite'] or item['extreme']]
            if scan['flagged_names'] and first is None:
                first = index
            event.update(copy.deepcopy(scan))
            event['event'] = scan['phase'] if scan['kind'] == 'operation' else 'dense_phase_queued'
            event['values_resolved_on_host'] = False
            event.pop('flagged_names', None)
            event['summaries'] = [{key: value for key, value in item.items() if key not in SUMMARY_RESULTS}
                                  for item in scan['summaries']]
        flush['first_flagged_enqueue_index'] = first
        return events

    @staticmethod
    def encode(events):
        return b''.join(json.dumps(event, allow_nan=False).encode() + b'\n' for event in events)

    def audit(self, events, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(self.encode(events))
            return audit_file(path, **kwargs)

    def test_zero_oracle_spike_uses_resolved_flush_and_preserves_native_locations(self):
        events = self.events()
        raw = self.encode(events)
        lines = raw.splitlines(keepends=True)
        report = self.audit(events)
        self.assertEqual(report['complete_attention_pairs_checked'], 3)
        self.assertEqual(report['real_bound_exceeding_calls'], 1)
        violation, = report['violations']
        self.assertEqual(violation['outputs']['dq']['bound'], 0.0)
        self.assertEqual(violation['outputs']['dq']['observed_max_abs'], 0.5)
        self.assertEqual(violation['outputs']['dq']['ratio_to_bound'], 'infinite')
        flush_line = next(i for i, event in enumerate(events, 1) if event['event'] == 'deferred_flush')
        prov = violation['provenance']
        location = prov['resolved_record']
        self.assertEqual(location['line'], flush_line)
        self.assertEqual(location['byte_offset'], sum(map(len, lines[:flush_line - 1])))
        self.assertEqual(location['sha256'], hashlib.sha256(lines[flush_line - 1]).hexdigest())
        flush = events[flush_line - 1]
        for phase in ('before', 'after'):
            scan = flush['scans'][prov[phase + '_scan_index']]
            queued_line = prov[phase + '_queued_record']['line']
            self.assertEqual((scan['layer'], scan['call'], scan['phase']),
                             (violation['layer'], violation['call'], phase))
            self.assertEqual(events[queued_line - 1]['enqueue_index'], prov[phase + '_enqueue_index'])
            self.assertLess(queued_line, flush_line)
        self.assertEqual(len(report['flow_for_violating_attempts']), 42)
        self.assertTrue(all(row['provenance']['resolved_record'] == location
                            for row in report['flow_for_violating_attempts']))

    def test_complete_queued_pairs_without_flush_are_never_observed_numerically(self):
        events = self.events()
        events = events[:next(i for i, event in enumerate(events) if event['event'] == 'deferred_flush')]
        with patch('audit_deferred_attention_bounds.real_bounds.bounds_for_pair',
                   side_effect=AssertionError('Unresolved values reached arithmetic')):
            report = self.audit(events)
        self.assertEqual(report['complete_attention_pairs_checked'], 0)
        self.assertEqual(report['queued_records_ignored_numerically'], 44)
        self.assertFalse(report['event_structure_audit']['latest_attempt_has_flush'])

    def test_interleaved_attention_calls_pair_by_owner_and_call(self):
        events = self.events()
        scans = next(event['scans'] for event in events if event['event'] == 'deferred_flush')
        indices = [i for i, scan in enumerate(scans) if scan.get('operation') == 'hstu_attention_bwd']
        selected = [scans[i] for i in indices]
        # Reorder six attention observations as all three before, then after.
        selected = [scan for scan in selected if scan['phase'] == 'before'] + [
                    scan for scan in selected if scan['phase'] == 'after']
        for index, scan in zip(indices, selected):
            scans[index] = scan
        report = self.audit(self.synchronize(events))
        self.assertEqual(report['complete_attention_pairs_checked'], 3)
        violation, = report['violations']
        self.assertTrue(violation['layer'].endswith('.2'))
        self.assertGreater(violation['provenance']['after_enqueue_index'] -
                           violation['provenance']['before_enqueue_index'], 1)

    def test_queued_and_resolved_scalar_disagreement_fails_before_arithmetic(self):
        events = self.events()
        queued = next(event for event in events if event.get('operation') == 'hstu_attention_bwd')
        queued['scalar_arguments']['N'] = 99
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            self.audit(events)

    def test_call_id_reuse_and_changed_full_attempt_identity_are_rejected(self):
        events = self.events()
        scans = next(event['scans'] for event in events if event['event'] == 'deferred_flush')
        before = [scan for scan in scans if scan.get('operation') == 'hstu_attention_bwd' and scan['phase'] == 'before']
        wrong_call = before[1]['call']
        for scan in scans:
            if scan.get('operation') == 'hstu_attention_bwd' and scan['layer'] == before[0]['layer']:
                scan['call'] = wrong_call
        with self.assertRaisesRegex(ValueError, 'shared by different'):
            self.audit(self.synchronize(events))
        for key in ('session', 'repeat'):
            events = self.events()
            flush = next(event for event in events if event['event'] == 'deferred_flush')
            if key == 'session':
                flush[key] = 'different-session'
            else:
                flush['attempt'][key] += 1
            with self.assertRaisesRegex(ValueError, 'full identity'):
                self.audit(events)

    def test_nonfinite_input_skips_and_nonfinite_output_is_conditional_violation(self):
        events = self.events(False)
        scans = next(event['scans'] for event in events if event['event'] == 'deferred_flush')
        before = next(scan for scan in scans if scan.get('operation') == 'hstu_attention_bwd' and scan['phase'] == 'before')
        before['summaries'][0].update(finite=False, min=None, max=None)
        report = self.audit(self.synchronize(events))
        self.assertEqual(report['complete_attention_pairs_checked'], 2)
        skip, = report['skips']
        self.assertEqual(skip['reason'], 'nonfinite_tensor')
        self.assertIn('resolved_record', skip['provenance'])
        events = self.events(False)
        scans = next(event['scans'] for event in events if event['event'] == 'deferred_flush')
        after = next(scan for scan in scans if scan.get('operation') == 'hstu_attention_bwd' and scan['phase'] == 'after')
        after['summaries'][0].update(finite=False, min=None, max=None)
        report = self.audit(self.synchronize(events))
        violation, = report['violations']
        self.assertFalse(violation['outputs']['dq']['output_finite'])

    def test_missing_attempt_cannot_bypass_queued_or_flush_identity_checks(self):
        for kind in ('before', 'after', 'dense_phase_queued', 'deferred_flush'):
            with self.subTest(kind=kind):
                events = self.events()
                event = next(event for event in events if event['event'] == kind)
                del event['attempt']
                event['session'] = 'unrelated-session'
                with self.assertRaisesRegex(ValueError, 'Missing attempt identity'):
                    self.audit(events)

    def test_missing_actual_scalar_is_explicit_skip_without_defaults(self):
        events = self.events(False)
        scans = next(event['scans'] for event in events if event['event'] == 'deferred_flush')
        for scan in scans:
            if scan.get('operation') == 'hstu_attention_bwd' and scan['layer'].endswith('.2'):
                del scan['scalar_arguments']['N']
        report = self.audit(self.synchronize(events))
        self.assertEqual(report['complete_attention_pairs_checked'], 2)
        self.assertEqual(report['skips'][0]['reason'], 'requires_positive_actual_N')

    def test_pinned_prefix_hash_includes_partial_line_but_parses_only_complete_records(self):
        complete = self.encode(self.events())
        partial = b'{"event":"unfinished'
        suffix = b' appended outside requested prefix'
        prefix = complete + partial
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(prefix + suffix)
            report = audit_file(path, prefix_bytes=len(prefix),
                                expected_prefix_sha256=hashlib.sha256(prefix).hexdigest())
            source = report['source']
            self.assertEqual(source['complete_record_sha256'], hashlib.sha256(complete).hexdigest())
            self.assertEqual(source['trailing_partial_line_bytes'], len(partial))
            self.assertEqual(source['appended_or_outside_prefix_bytes_not_audited'], len(suffix))
            self.assertEqual(report['complete_attention_pairs_checked'], 3)
            with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                audit_file(path, expected_prefix_sha256='0' * 64)
            for count in (-1, True, len(prefix + suffix) + 1):
                with self.assertRaisesRegex(ValueError, 'prefix_bytes'):
                    audit_file(path, prefix_bytes=count)

    def test_malformed_complete_records_and_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            for raw in (b'{"event":\n', b'{"event":"x","event":"y"}\n',
                        b'{"event":"x","value":NaN}\n'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    audit_file(path)

    def test_cli_refuses_overwrite_and_import_is_host_only(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, target = directory / 'events.jsonl', directory / 'report.json'
            source.write_bytes(self.encode(self.events(False)))
            args = [str(source), '--report', str(target)]
            self.assertEqual(main(args), 0)
            with self.assertRaises(FileExistsError):
                main(args)
        result = subprocess.run([sys.executable, '-B', '-c',
                                 "import sys; import audit_deferred_attention_bounds; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"],
                                cwd=Path(__file__).parent, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
