"""CPU fixtures for dimension-derived arithmetic bounds and paired provenance."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import audit_training_arithmetic_bounds as audit


OUTPUT, PROJECTION = audit.SPECS
LAYER = 'model._stu_layers.1'


def tensor(name, shape, magnitude=1.0, dtype='torch.bfloat16'):
    return {'name': name, 'shape': shape, 'numel': math.prod(shape),
            'min': -magnitude, 'max': magnitude, 'finite': True,
            'scanned': True, 'dtype': dtype}


def fixture(step=1, operations=None):
    operations = operations or [audit.pairing.OUTPUT, audit.pairing.ADDMM]
    attempt = {'mode': 'training', 'repeat': 0, 'step': step}
    events = [{'event': 'installed', 'session': 'test', 'layers': [LAYER],
               'selected_operations': operations},
              {'event': 'attempt_start', 'session': 'test', 'attempt': attempt}]
    controls = {'allow_bf16_reduced_precision_reduction': True}
    for call, op in enumerate(operations, 1):
        if op == audit.pairing.OUTPUT:
            a, b, c = 'dout', 'output_weight', 'dy'
            shapes = ([3, 4], [5, 4], [3, 5])
        else:
            a, b, c = 'dz', 'w', 'd_normed_x'
            shapes = ([3, 8], [2, 8], [3, 2])
        inputs = [tensor(a, shapes[0], .125), tensor(b, shapes[1], .25)]
        output = [tensor(c, shapes[2], .0625)]
        common = {'session': 'test', 'attempt': attempt, 'layer': LAYER,
                  'operation': op, 'call': call, 'execution_controls': controls,
                  'input_observation': {'operation_started': False, 'snapshot_present': False}}
        events.append({'event': 'before', **copy.deepcopy(common), 'inputs': copy.deepcopy(inputs)})
        events.append({'event': 'after', **copy.deepcopy(common), 'inputs': copy.deepcopy(inputs), 'outputs': output})
    return events


def item(events, operation, direction, name):
    event = next(e for e in events if e.get('operation') == operation
                 and e['event'] == ('before' if direction == 'inputs' else 'after'))
    return next(t for t in event[direction] if t['name'] == name)


def encode(events):
    return b''.join((json.dumps(e, allow_nan=False) + '\n').encode() for e in events)


class ArithmeticBoundsTests(unittest.TestCase):
    def run_fixture(self, events, suffix=b'', **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(encode(events) + suffix)
            return audit.audit_file(path, **kwargs)

    def test_dimensions_and_two_transposed_weight_contracts(self):
        report = self.run_fixture(fixture())
        self.assertEqual(report['status'], 'checked_prefix_no_violations')
        self.assertEqual([r['checked'] for r in report['bound_counts'].values()], [1, 1])
        self.assertEqual(report['maximum_output_to_bound_ratio'][OUTPUT]['reduction_dimension'], 4)
        self.assertEqual(report['maximum_output_to_bound_ratio'][PROJECTION]['reduction_dimension'], 8)

    def test_impossible_output_far_below_absolute_monitor_threshold(self):
        events = fixture()
        t = item(events, audit.pairing.OUTPUT, 'outputs', 'dy')
        t.update(min=-2.0, max=2.0)
        report = self.run_fixture(events)
        self.assertEqual(report['arithmetic_bound_violation_count'], 1)
        self.assertEqual(report['violations'][0]['kind'], 'arithmetic_bound_exceeded')
        self.assertEqual(report['violations'][0]['model'], audit.MODEL)

    def test_true_reduced_precision_flag_retained_as_conditional(self):
        report = self.run_fixture(fixture())
        self.assertTrue(next(iter(report['execution_control_profiles'].values()))['before']['allow_bf16_reduced_precision_reduction'])
        self.assertEqual(report['bound_counts'][OUTPUT]['conditional_FP32_checks'], 1)
        self.assertEqual(report['maximum_output_to_bound_ratio'][OUTPUT]['model'], audit.MODEL)

    def test_recorded_whole_layer_target_restricts_expected_coverage(self):
        events = fixture(operations=[audit.pairing.OUTPUT])
        events[0]['layers'] = ['model._stu_layers.0', LAYER, 'model._stu_layers.2']
        events[0]['target'] = '_stu_layers.1.'
        report = self.run_fixture(events)
        self.assertEqual(report['selected_boundary_steps_complete'], 1)
        self.assertEqual(report['bound_counts'][OUTPUT]['passed'], 1)
        self.assertEqual(report['installed_target_filters']['test']['effective_layers'], [LAYER])

    def test_parameter_specific_target_is_not_misread_as_whole_layer(self):
        events = fixture(operations=[audit.pairing.OUTPUT])
        events[0]['target'] = '_output_weight'
        report = self.run_fixture(events)
        self.assertGreater(report['violation_count'], 0)
        self.assertEqual(report['bound_counts'][OUTPUT]['checked'], 0)

    def test_exact_zero_does_not_allow_even_one_subnormal(self):
        events = fixture()
        item(events, audit.pairing.OUTPUT, 'inputs', 'dout').update(min=0.0, max=0.0)
        item(events, audit.pairing.OUTPUT, 'outputs', 'dy').update(min=0.0, max=2.0 ** -133)
        report = self.run_fixture(events)
        self.assertEqual(report['arithmetic_bound_violation_count'], 1)
        self.assertEqual(report['violations'][0]['model'], 'exact_zero_product_oracle')
        self.assertEqual(report['violations'][0]['bound'], 0)
        self.assertEqual(report['violations'][0]['ratio_to_bound'], 'inf')

    def test_nonzero_tiny_products_receive_absolute_underflow_allowance(self):
        bound, reason = audit.gemm_bound(2.0 ** -133, 2.0 ** -133, 4)
        self.assertIsNone(reason)
        self.assertGreater(bound['underflow_allowance'], 0)
        self.assertGreater(bound['bound'], 2.0 ** -133)

    def test_k_limit_and_overflow_applicability_are_explicit(self):
        b, reason = audit.gemm_bound(.125, .25, audit.MAX_K)
        self.assertIsNone(reason)
        self.assertAlmostEqual(b['gamma_K'], 1 / 3)
        self.assertEqual(audit.gemm_bound(.125, .25, audit.MAX_K + 1)[1], 'reduction_dimension_outside_model')
        self.assertEqual(audit.gemm_bound(audit.BF16_MAX, audit.BF16_MAX, 4)[1], 'model_does_not_exclude_intermediate_or_output_overflow')

    def test_shape_contract_violation_is_metadata_problem(self):
        events = fixture()
        t = item(events, audit.pairing.OUTPUT, 'inputs', 'output_weight')
        t.update(shape=[4, 5], numel=20)
        report = self.run_fixture(events)
        self.assertEqual(report['violations'][0]['kind'], 'GEMM_shape_contract_mismatch')
        self.assertEqual(report['arithmetic_bound_violation_count'], 0)

    def test_unsupported_dtype_or_missing_shape_is_skipped(self):
        for alteration, reason in [('dtype', 'requires_BF16_inputs_and_output'),
                                   ('shape', 'requires_explicit_matrix_shapes')]:
            with self.subTest(alteration=alteration):
                events = fixture(operations=[audit.pairing.OUTPUT])
                t = item(events, audit.pairing.OUTPUT, 'inputs', 'dout')
                if alteration == 'dtype':
                    t['dtype'] = 'torch.float32'
                else:
                    t.pop('shape')
                report = self.run_fixture(events)
                self.assertEqual(report['bound_counts'][OUTPUT]['skip_reasons'][reason], 1)
                self.assertEqual(report['status'], 'insufficient_applicable_checks')

    def test_after_inputs_never_replace_before_observations(self):
        events = fixture()
        for e in events:
            if e.get('event') == 'after':
                for t in e['inputs']:
                    t.update(min=-1e34, max=1e34, finite=False)
        report = self.run_fixture(events)
        self.assertEqual(report['violation_count'], 0)
        self.assertEqual(report['maximum_output_to_bound_ratio'][OUTPUT]['input_magnitudes'], [.125, .25])

    def test_incomplete_step_and_duplicate_pair_never_checked(self):
        events = fixture()
        report = self.run_fixture(events[:-1])
        self.assertEqual(report['selected_boundary_steps_skipped'], 1)
        self.assertEqual(sum(r['checked'] for r in report['bound_counts'].values()), 0)
        events.append(copy.deepcopy(events[-1]))
        report = self.run_fixture(events)
        self.assertGreater(report['violation_count'], 0)
        self.assertEqual(sum(r['checked'] for r in report['bound_counts'].values()), 0)

    def test_changed_pair_identity_rejected(self):
        events = fixture()
        events[-1]['layer'] = 'model._stu_layers.2'
        report = self.run_fixture(events)
        self.assertEqual(sum(r['checked'] for r in report['bound_counts'].values()), 0)
        self.assertTrue(any(v['kind'] == 'call_pair_identity_mismatch' for v in report['violations']))

    def test_nonfinite_output_remains_anomaly_with_explicit_skip(self):
        events = fixture()
        item(events, audit.pairing.OUTPUT, 'outputs', 'dy').update(finite=False, min='nan', max='nan')
        report = self.run_fixture(events)
        self.assertEqual(report['status'], 'anomaly_observed')
        self.assertEqual(report['bound_counts'][OUTPUT]['skip_reasons']['nonfinite_or_unknown_tensor'], 1)
        self.assertEqual(report['arithmetic_bound_violation_count'], 0)

    def test_fixed_prefix_hash_excludes_appended_complete_attempt(self):
        first = fixture()
        second = fixture(2)[1:]
        prefix = encode(first)
        report = self.run_fixture(first + second, prefix_bytes=len(prefix))
        self.assertEqual(report['source']['prefix_sha256'], hashlib.sha256(prefix).hexdigest())
        self.assertEqual(report['selected_boundary_steps_complete'], 1)
        self.assertGreater(report['source']['appended_bytes_not_audited'], 0)

    def test_partial_line_hashed_but_not_parsed(self):
        events = fixture()
        suffix = b'{"event":"before"'
        report = self.run_fixture(events, suffix=suffix)
        self.assertEqual(report['source']['prefix_sha256'], hashlib.sha256(encode(events) + suffix).hexdigest())
        self.assertEqual(report['source']['trailing_partial_line_bytes'], len(suffix))
        self.assertEqual(report['violation_count'], 0)

    def test_malformed_complete_line_is_retained_as_structural_violation(self):
        report = self.run_fixture(fixture(), suffix=b'{"event":NaN}\n')
        self.assertTrue(any(v['kind'] == 'malformed_complete_line' for v in report['violations']))


if __name__ == '__main__':
    unittest.main()
