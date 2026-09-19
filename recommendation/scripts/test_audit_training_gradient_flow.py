"""CPU-only fixtures for range propagation, pairing and live-prefix handling."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_training_gradient_flow as audit


OPERATIONS = [audit.OUTPUT, audit.NORM, audit.WEIGHT, audit.ATTENTION,
              audit.SILU, audit.ADDMM, audit.INPUT_NORM]
LAYERS = ['model._stu_layers.' + str(index) for index in range(3)]


def tensor(name, low=-1.0, high=1.0, numel=8, shape=None):
    return {'name': name, 'min': low, 'max': high, 'numel': numel,
            'shape': shape if shape is not None else [numel], 'dtype': 'torch.bfloat16',
            'finite': True, 'scanned': True, 'extreme': False}


def fixture(step=1, session='fixture', selected=None):
    selected = OPERATIONS if selected is None else selected
    attempt = {'mode': 'training', 'repeat': 0, 'step': step}
    events = [{'event': 'installed', 'session': session, 'layers': LAYERS,
               'selected_operations': selected},
              {'event': 'attempt_start', 'session': session, 'attempt': attempt}]
    call = 0
    for layer in reversed(LAYERS):
        calls = {
            audit.OUTPUT: ([tensor('dout')], [tensor('dy', numel=24)]),
            audit.NORM: ([tensor('dy', numel=24)],
                         [tensor('d_attn', -3, 3, shape=[2, 4]), tensor('d_u', -2, 2),
                          tensor('y', -5, 5, numel=24)]),
            audit.WEIGHT: ([tensor('y', -5, 5, numel=24), tensor('dout')],
                          [tensor('d_output_weight', -.01, .01)]),
            audit.ATTENTION: ([tensor('dout', -3, 3, shape=[2, 2, 2])],
                             [tensor('dq'), tensor('dk', -.1, .1), tensor('dv', -.2, .2)]),
            audit.SILU: ([tensor('grad_output', -2, 2)], [tensor('du', -.5, .5)]),
            audit.ADDMM: ([tensor('dz', numel=32)], [tensor('d_normed_x', -.25, .25)]),
            audit.INPUT_NORM: ([tensor('dy', -.25, .25)], [tensor('d_x', -.125, .125)]),
        }
        for operation in selected:
            call += 1
            inputs, outputs = calls[operation]
            common = {'session': session, 'attempt': attempt, 'call': call,
                      'operation': operation, 'layer': layer}
            events.append({'event': 'before', **common, 'inputs': copy.deepcopy(inputs)})
            events.append({'event': 'after', **common, 'inputs': copy.deepcopy(inputs),
                           'outputs': copy.deepcopy(outputs)})
    return events


def find(events, operation, direction, name, layer=2):
    event = next(event for event in events if event.get('operation') == operation
                 and event.get('layer') == LAYERS[layer]
                 and event['event'] == ('before' if direction == 'inputs' else 'after'))
    return next(item for item in event[direction] if item['name'] == name)


def encode(events):
    return b''.join((json.dumps(event) + '\n').encode() for event in events)


class GradientFlowAuditTests(unittest.TestCase):
    def run_fixture(self, events, *, suffix=b'', **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(encode(events) + suffix)
            return audit.audit_file(path, **kwargs)

    def test_healthy_seven_operations_have_23_checks_with_reshape(self):
        report = self.run_fixture(fixture())
        self.assertEqual(report['status'], 'checked_prefix_no_violations')
        self.assertEqual(report['selected_boundary_steps_complete'], 1)
        self.assertEqual(sum(row['checked'] for row in report['edge_counts'].values()), 23)
        self.assertEqual(report['edge_counts'][audit.PACKED_EDGE]['passed'], 3)
        self.assertEqual(report['edge_counts'][audit.RESIDUAL_EDGE]['passed'], 2)
        self.assertAlmostEqual(report['max_residual_ratio']['ratio'], 1 / 1.125)

    def test_uses_before_inputs_not_repeated_after_inputs(self):
        events = fixture()
        for event in events:
            if event['event'] == 'after':
                for item in event['inputs']:
                    item.update(min='nan', max=1e37, finite=False, extreme=True)
        report = self.run_fixture(events)
        self.assertEqual(report['violation_count'], 0)
        self.assertEqual(report['anomalies']['tensor_observation_count'], 0)
        self.assertEqual(sum(row['passed'] for row in report['edge_counts'].values()), 23)

    def test_direct_minmax_dtype_and_numel_differences_are_detected(self):
        for field, value in [('min', -2), ('max', 2), ('dtype', 'torch.float32'), ('numel', 16)]:
            with self.subTest(field=field):
                events = fixture()
                item = find(events, audit.NORM, 'inputs', 'dy')
                item[field] = value
                if field == 'numel':
                    item['shape'] = [value]
                report = self.run_fixture(events)
                self.assertEqual(report['violation_count'], 1)
                self.assertIn(field, report['violations'][0]['differences'])

    def test_packed_union_checks_all_four_parts_and_total_numel(self):
        for change in ('missing_dk_extreme', 'wrong_count'):
            with self.subTest(change=change):
                events = fixture()
                if change == 'missing_dk_extreme':
                    find(events, audit.ATTENTION, 'outputs', 'dk')['min'] = -4
                else:
                    item = find(events, audit.ADDMM, 'inputs', 'dz')
                    item.update(numel=40, shape=[40])
                report = self.run_fixture(events)
                self.assertEqual(report['violation_count'], 1)
                self.assertEqual(report['violations'][0]['edge'], audit.PACKED_EDGE)

    def test_consistent_huge_propagation_is_separate_from_anomaly_events(self):
        events = fixture()
        for operation, direction, name in [(audit.ATTENTION, 'outputs', 'dq'),
                                           (audit.ADDMM, 'inputs', 'dz')]:
            find(events, operation, direction, name).update(max=1e34, extreme=True)
        events.append({'event': 'dump_complete', 'session': 'fixture',
                       'attempt': {'mode': 'training', 'repeat': 0, 'step': 1},
                       'bad_outputs': [], 'extreme_outputs': ['dq']})
        report = self.run_fixture(events)
        self.assertEqual(report['violation_count'], 0)
        self.assertEqual(report['status'], 'anomaly_observed')
        self.assertEqual(report['anomalies']['tensor_observation_count'], 2)
        self.assertEqual(report['anomalies']['explicit_event_count'], 1)
        self.assertEqual(report['anomalies']['capture_event_count'], 1)
        self.assertEqual(report['endpoint_magnitude_maxima'][audit.ATTENTION + '.outputs.dq']['max_abs'], 1e34)

    def test_residual_amplification_violation_and_strict_zero_rule(self):
        events = fixture()
        for operation in (audit.OUTPUT, audit.WEIGHT):
            find(events, operation, 'inputs', 'dout', layer=1).update(min=-100, max=100)
        report = self.run_fixture(events)
        self.assertEqual(report['violation_count'], 1)
        self.assertIn('residual_bound', report['violations'][0]['differences'])
        self.assertGreater(report['max_residual_ratio']['ratio'], 2)
        zero = fixture()
        for operation in (audit.OUTPUT, audit.WEIGHT):
            find(zero, operation, 'inputs', 'dout').update(min=0.0, max=0.0)
        find(zero, audit.INPUT_NORM, 'outputs', 'd_x').update(min=0.0, max=0.0)
        report = self.run_fixture(zero)
        self.assertEqual(report['max_residual_ratio']['ratio'], 'inf')
        self.assertEqual(report['edge_counts'][audit.RESIDUAL_EDGE]['violations'], 1)
        json.dumps(report, allow_nan=False)
        for layer in (0, 1):
            for operation in (audit.OUTPUT, audit.WEIGHT):
                find(zero, operation, 'inputs', 'dout', layer).update(min=0.0, max=0.0)
            find(zero, audit.INPUT_NORM, 'outputs', 'd_x', layer).update(min=0.0, max=0.0)
        report = self.run_fixture(zero)
        self.assertEqual(report['violation_count'], 0)
        self.assertEqual(report['max_residual_ratio']['ratio'], 0.0)

    def test_live_prefix_hash_skips_partial_step_and_incomplete_line(self):
        first = fixture()
        second = fixture(step=2)[1:3]  # attempt_start and the first before, without after.
        prefix = encode(first + second) + b'{"event":'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(prefix)
            report = audit.audit_file(path)
            self.assertEqual(report['selected_boundary_steps_complete'], 1)
            self.assertEqual(report['selected_boundary_steps_skipped'], 1)
            self.assertEqual(report['source']['trailing_partial_line_bytes'], len(b'{"event":'))
            self.assertEqual(report['source']['prefix_sha256'], hashlib.sha256(prefix).hexdigest())
            self.assertEqual(sum(row['checked'] for row in report['edge_counts'].values()), 23)
            self.assertEqual(sum(row['skipped'] for row in report['edge_counts'].values()), 23)
            with path.open('ab') as stream:
                stream.write(b'"irrelevant_later_event"}\n')
            repeated = audit.audit_file(path, prefix_bytes=len(prefix))
            self.assertEqual(repeated['source']['prefix_sha256'], report['source']['prefix_sha256'])
            self.assertEqual(repeated['edge_counts'], report['edge_counts'])
            self.assertGreater(repeated['source']['appended_bytes_not_audited'], 0)

    def test_wrong_call_ids_do_not_pair_by_layer_and_operation(self):
        events = fixture()
        next(event for event in events if event['event'] == 'after')['call'] = 100
        report = self.run_fixture(events)
        self.assertEqual(report['selected_boundary_steps_complete'], 0)
        self.assertEqual(sum(row['checked'] for row in report['edge_counts'].values()), 0)
        self.assertEqual(len(report['skipped_steps'][0]['unmatched_calls']), 2)

    def test_mismatched_identity_duplicate_event_and_reversed_call_order(self):
        for mutation in ('identity', 'duplicate', 'order'):
            with self.subTest(mutation=mutation):
                events = fixture()
                if mutation == 'identity':
                    events[3]['layer'] = LAYERS[1]
                elif mutation == 'duplicate':
                    events.insert(3, copy.deepcopy(events[2]))
                else:
                    events[2], events[3] = events[3], events[2]
                report = self.run_fixture(events)
                self.assertGreater(report['violation_count'], 0)
                self.assertEqual(report['selected_boundary_steps_complete'], 0)

    def test_producer_consumer_order_is_checked_beyond_matching_call_pairs(self):
        events = fixture()
        events[2:6] = events[4:6] + events[2:4]
        report = self.run_fixture(events)
        self.assertEqual(report['selected_boundary_steps_complete'], 1)
        self.assertEqual(report['violation_count'], 1)
        self.assertIn('observation_order', report['violations'][0]['differences'])

    def test_missing_unselected_operations_and_optional_y_are_explicit_skips(self):
        selected = [audit.OUTPUT, audit.NORM, audit.ADDMM, audit.INPUT_NORM]
        report = self.run_fixture(fixture(selected=selected))
        self.assertEqual(report['selected_boundary_steps_complete'], 1)
        self.assertEqual(sum(row['checked'] for row in report['edge_counts'].values()), 8)
        self.assertEqual(sum(row['skipped'] for row in report['edge_counts'].values()), 15)
        events = fixture()
        after = next(event for event in events if event.get('operation') == audit.NORM
                     and event['event'] == 'after')
        after['outputs'] = [item for item in after['outputs'] if item['name'] != 'y']
        report = self.run_fixture(events)
        self.assertEqual(report['violation_count'], 0)
        self.assertEqual(sum(row['skipped'] for row in report['edge_counts'].values()), 1)

    def test_nonfinite_and_unscanned_summaries_are_not_finite_range_passes(self):
        for update, reason in [({'finite': False, 'min': 'nan'}, 'nonfinite_or_unknown_tensor'),
                               ({'scanned': False}, 'unscanned_tensor')]:
            with self.subTest(reason=reason):
                events = fixture()
                find(events, audit.ATTENTION, 'inputs', 'dout').update(update)
                report = self.run_fixture(events)
                edge = report['edge_counts']['output_norm.d_attn_to_attention.dout']
                self.assertEqual(edge['passed'], 2)
                self.assertEqual(edge['skip_reasons'][reason], 1)

    def test_bad_shape_or_summary_is_a_violation_not_a_range_pass(self):
        for update in ({'shape': [9]}, {'min': 10, 'max': 1}):
            events = fixture()
            find(events, audit.ATTENTION, 'inputs', 'dout').update(update)
            report = self.run_fixture(events)
            self.assertEqual(report['violation_count'], 1)
            self.assertEqual(report['violations'][0]['differences'], ['invalid_tensor_summary'])

    def test_session_boundaries_do_not_merge_and_repeated_steps_are_rejected(self):
        report = self.run_fixture(fixture() + fixture(session='second'))
        self.assertEqual(report['selected_boundary_steps_complete'], 2)
        self.assertEqual(report['violation_count'], 0)
        report = self.run_fixture(fixture() + fixture()[1:])
        self.assertEqual(report['selected_boundary_steps_complete'], 1)
        self.assertEqual(report['selected_boundary_steps_skipped'], 1)
        self.assertTrue(any(item['kind'] == 'non_increasing_attempt_step' for item in report['violations']))

    def test_malformed_complete_line_is_not_silently_treated_as_live_tail(self):
        events = fixture()
        data = encode(events[:3]) + b'{broken}\n' + encode(events[3:])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(data)
            report = audit.audit_file(path, max_details=0)
        self.assertEqual(report['violation_count'], 1)
        self.assertTrue(report['violation_details_truncated'])
        self.assertEqual(report['selected_boundary_steps_complete'], 0)
        self.assertEqual(report['source']['prefix_sha256'], hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    unittest.main()
