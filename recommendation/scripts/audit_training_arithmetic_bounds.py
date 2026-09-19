#!/usr/bin/env python3
"""Check conditional GEMM magnitude bounds from paired training scalar events.

No tensors, GPU framework, monitor mutation or runtime arithmetic proof.
Only complete selected-operation attempts in a hashed byte prefix are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import audit_training_gradient_flow as pairing


SPECS = {
    'output_gradient_mm.dy': (pairing.OUTPUT, 'dout', 'output_weight', 'dy'),
    'projection_input_gradient_mm.d_normed_x': (pairing.ADDMM, 'dz', 'w', 'd_normed_x'),
}
FP32_U = 2.0 ** -24
BF16_U = 2.0 ** -8
MIN_NORMAL = 2.0 ** -126
MAX_K = 1 << 22
BF16_MAX = (2.0 - 2.0 ** -7) * 2.0 ** 127
FP32_MAX = (2.0 - 2.0 ** -23) * 2.0 ** 127
MODEL = 'conditional_on_FP32_accumulation_and_one_final_BF16_rounding'


def gemm_bound(a_abs, b_abs, reduction):
    """Conservative bound for BF16 A @ B.T, conditional on the stated model.

    A BF16 product is exactly representable in FP32 except range underflow.
    Allow an absolute error eta=2^-126 for each product/addition, covering
    gradual underflow or flushing tiny results to zero. At most K FP32 adds
    along each term's path gives gamma_K=K*u/(1-K*u). A conservative bound
    before the final cast is (K*Amax*Bmax + 2*K*eta)/(1-K*u).
    For K<=2^22 the final BF16 rounding coefficient is below 2; the absolute
    underflow coefficient is below 4*(K+1). Intermediate/output overflow is
    separately excluded using the tighter expression. No reduced-precision
    intermediate accumulation is covered. Exact zero products need no eta.
    """
    if (not isinstance(reduction, int) or isinstance(reduction, bool)
            or reduction < 1 or reduction > MAX_K):
        return None, 'reduction_dimension_outside_model'
    if any(not isinstance(x, (float, int)) or isinstance(x, bool)
           or not math.isfinite(x) or x < 0 or x > BF16_MAX for x in (a_abs, b_abs)):
        return None, 'input_magnitude_outside_finite_BF16_range'
    if a_abs == 0 or b_abs == 0:
        return {'bound': 0.0, 'exact_zero_oracle': True, 'reduction_dimension': reduction,
                'ideal_absolute_sum_bound': 0.0, 'underflow_allowance': 0.0,
                'gamma_K': reduction * FP32_U / (1 - reduction * FP32_U)}, None
    ideal = reduction * a_abs * b_abs
    gamma = reduction * FP32_U / (1 - reduction * FP32_U)
    fp32_upper = (ideal + 2 * reduction * MIN_NORMAL) / (1 - reduction * FP32_U)
    rounded_upper = (1 + BF16_U) * fp32_upper + MIN_NORMAL
    host_margin = 1 + 64 * sys.float_info.epsilon
    if fp32_upper >= FP32_MAX / host_margin or rounded_upper >= BF16_MAX / host_margin:
        return None, 'model_does_not_exclude_intermediate_or_output_overflow'
    allowance = 4 * (reduction + 1) * MIN_NORMAL
    # Scalar observations are exact BF16 values in the real monitor; one host
    # nextafter protects the final double calculation from rounding downward.
    bound = math.nextafter(2 * ideal + allowance, math.inf)
    return {'bound': bound, 'exact_zero_oracle': False, 'reduction_dimension': reduction,
            'ideal_absolute_sum_bound': ideal, 'underflow_allowance': allowance,
            'gamma_K': gamma, 'FP32_intermediate_upper_bound': fp32_upper,
            'final_BF16_upper_bound_before_loose_factor_two': rounded_upper}, None


class ArithmeticAudit(pairing.FlowAudit):
    def __init__(self, max_details=100):
        super().__init__(max_details)
        # Keep the parent counters only for its complete-step bookkeeping;
        # its propagation checks are never executed by this subclass.
        self.edges.update({name: {'checked': 0, 'passed': 0, 'violations': 0,
                                  'skipped': 0, 'skip_reasons': Counter(),
                                  'exact_zero_checks': 0, 'conditional_FP32_checks': 0}
                           for name in SPECS})
        self.bound_violations = 0
        self.max_ratios = {}
        self.execution_control_profiles = {}
        self.installed_target_filters = {}

    def consume(self, event, line):
        if event.get('event') == 'installed':
            target = event.get('target', '*')
            original_layers = event.get('layers', [])
            selected_layers = original_layers
            if target not in ('*', 'all'):
                if isinstance(target, str) and target.endswith('.'):
                    selected_layers = [layer for layer in original_layers
                                       if isinstance(layer, str) and target in layer + '.']
                else:
                    selected_layers = []
                if not selected_layers:
                    self.problem('unsupported_or_empty_layer_target_filter', line=line, target=target)
            self.installed_target_filters[event.get('session')] = {
                'target': target, 'original_layers': original_layers,
                'effective_layers': selected_layers,
                'scope': 'Only whole-layer substring filters ending with a dot are supported; parameter-specific filters are rejected.'}
            event = {**event, 'layers': selected_layers}
        super().consume(event, line)
        if self.current and event.get('event') in ('before', 'after'):
            compact = self.current['pairs'].get(event.get('call'), {}).get(event['event'])
            if compact is not None:
                compact['execution_controls'] = event.get('execution_controls')
                compact['input_observation'] = event.get('input_observation')

    def finish_step(self, closed_by):
        state = self.current
        prior_skipped = self.incomplete_steps
        config = self.configs.get(state['key'][0]) if state else None
        super().finish_step(closed_by)
        if self.incomplete_steps > prior_skipped and config:
            for _layer in config['layers']:
                for name, (operation, *_) in SPECS.items():
                    if operation in config['operations']:
                        self.skip(name, 'incomplete_selected_step')

    def audit_layer(self, pairs, layer):
        for name, (operation, a_name, b_name, c_name) in SPECS.items():
            if (layer, operation) not in pairs:
                self.skip(name, 'operation_not_installed')
                continue
            requested = [(layer, (operation, 'inputs', a_name)),
                         (layer, (operation, 'inputs', b_name)),
                         (layer, (operation, 'outputs', c_name))]
            values = self.endpoints(pairs, name, layer, requested)
            if values is None:
                continue
            a, b, c = values
            if any(t['dtype'] != 'torch.bfloat16' for t in values):
                self.skip(name, 'requires_BF16_inputs_and_output')
                continue
            if any(not isinstance(t.get('shape'), list) or len(t['shape']) != 2 for t in values):
                self.skip(name, 'requires_explicit_matrix_shapes')
                continue
            m, k = a['shape']
            n, bk = b['shape']
            if k != bk or c['shape'] != [m, n]:
                self.problem('GEMM_shape_contract_mismatch', bound_name=name,
                             shapes=[t['shape'] for t in values], **self._context(layer))
                self.skip(name, 'GEMM_shape_contract_mismatch')
                continue
            magnitudes = [max(abs(t['min']), abs(t['max'])) for t in values]
            if magnitudes[2] > BF16_MAX:
                self.problem('output_magnitude_outside_finite_BF16_range', bound_name=name,
                             **self._context(layer))
                self.skip(name, 'output_magnitude_outside_finite_BF16_range')
                continue
            bound, reason = gemm_bound(magnitudes[0], magnitudes[1], k)
            if reason:
                self.skip(name, reason)
                continue
            pair = pairs[layer, operation]
            before_controls = pair['before']['execution_controls']
            after_controls = pair['after']['execution_controls']
            profile = {'before': before_controls, 'after': after_controls,
                       'controls_equal': before_controls == after_controls}
            profile_key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
            saved_profile = self.execution_control_profiles.setdefault(profile_key, {**profile, 'calls_checked': 0})
            saved_profile['calls_checked'] += 1
            observed = magnitudes[2]
            ratio = observed / bound['bound'] if bound['bound'] else (0.0 if observed == 0 else math.inf)
            within = observed <= bound['bound']
            context = {'bound_name': name, **self._context(layer), 'call': pair['before']['call'],
                       'before_line': pair['before']['line'], 'after_line': pair['after']['line'],
                       'shapes': [t['shape'] for t in values], 'input_magnitudes': magnitudes[:2],
                       'output_magnitude': observed, **bound,
                       'ratio_to_bound': pairing._json_number(ratio),
                       'model': 'exact_zero_product_oracle' if bound['exact_zero_oracle'] else MODEL,
                       'execution_control_profile': profile_key,
                       'input_observation': pair['before']['input_observation']}
            if name not in self.max_ratios or ratio > self.max_ratios[name][0]:
                self.max_ratios[name] = ratio, context
            count = self.edges[name]
            count['checked'] += 1
            count['exact_zero_checks' if bound['exact_zero_oracle'] else 'conditional_FP32_checks'] += 1
            if within:
                count['passed'] += 1
            else:
                count['violations'] += 1
                self.bound_violations += 1
                self.problem('arithmetic_bound_exceeded', **context)

    def result(self):
        report = super().result()
        report['format'] = 'training_conditional_arithmetic_bound_audit_v1'
        report['bound_counts'] = {name: self.edges[name] for name in SPECS}
        if not report['violation_count'] and report['status'] != 'anomaly_observed' and not any(
                row['checked'] for row in report['bound_counts'].values()):
            report['status'] = 'insufficient_applicable_checks'
        report.pop('edge_counts')
        report.pop('max_residual_ratio')
        report['arithmetic_bound_violation_count'] = self.bound_violations
        report['maximum_output_to_bound_ratio'] = {name: item[1] for name, item in self.max_ratios.items()}
        report['execution_control_profiles'] = self.execution_control_profiles
        report['installed_target_filters'] = self.installed_target_filters
        report['arithmetic_model'] = {
            'name': MODEL, 'formula': 'Cmax <= 2*K*Amax*Bmax + 4*(K+1)*2^-126',
            'FP32_unit_roundoff': FP32_U, 'BF16_unit_roundoff': BF16_U,
            'maximum_reduction_dimension': MAX_K,
            'underflow_policy': 'Absolute allowance covers gradual underflow and flush-to-zero; exact zero inputs retain an exact zero oracle.',
            'overflow_policy': 'Skip unless the tighter FP32/BF16 model bounds exclude intermediate and final-cast overflow.',
            'derivation': '(1+u_BF16)*(K*Amax*Bmax + 2*K*2^-126)/(1-K*u_FP32) + 2^-126 is below the loose bound for K<=2^22.',
        }
        report['scope'] = [
            'Only output-gradient GEMM and projection input-gradient GEMM; both are modeled as A @ B.T with dimensions from the observed shapes.',
            'Only before.inputs and matching after.outputs from complete installed selected-operation attempts; no after.inputs substitution.',
            'The nonzero-input bound is conditional on FP32 accumulation with at most K rounding operations along each term path and one final BF16 rounding.',
            'Recorded torch controls are retained. allow_bf16_reduced_precision_reduction=true does not establish FP32-only accumulation; even a false flag is not runtime proof.',
            'No tensor bytes, mathematical equality, intermediate values, exact loaded backend implementation or whole-training correctness are verified.',
            'A violation is a producer-triage lead under the stated model, not sole proof of a faulty hardware instruction. A passing bound does not establish correct GEMM output.',
            'No terminal trainer result is inferred from a live prefix. Maxima may occur at different steps/layers.',
        ]
        return report


def audit_file(path, *, prefix_bytes=None, max_details=100):
    path = Path(path)
    if max_details < 0:
        raise ValueError('max_details must be nonnegative')
    auditor, digest = ArithmeticAudit(max_details), hashlib.sha256()
    consumed = line_number = partial_bytes = 0
    with path.open('rb') as stream:
        initial = os.fstat(stream.fileno())
        limit = initial.st_size if prefix_bytes is None else prefix_bytes
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= initial.st_size:
            raise ValueError('prefix_bytes must lie within the file size observed at open')
        while consumed < limit:
            raw = stream.readline(min(limit - consumed, (1 << 20) + 1))
            if not raw:
                raise ValueError('File was truncated while reading the pinned prefix')
            if len(raw) > 1 << 20:
                raise ValueError('A JSONL line exceeds the 1 MiB audit limit')
            digest.update(raw)
            consumed += len(raw)
            line_number += 1
            if not raw.endswith(b'\n'):
                if consumed != limit:
                    raise ValueError('File was truncated inside the pinned prefix')
                partial_bytes = len(raw)
                break
            try:
                def reject_constant(value):
                    raise ValueError('Nonstandard JSON numeric constant: ' + value)
                event = json.loads(raw, parse_constant=reject_constant)
                if not isinstance(event, dict):
                    raise ValueError('JSONL event must be an object')
            except (ValueError, UnicodeError) as error:
                auditor.problem('malformed_complete_line', line=line_number, error=str(error))
                if auditor.current:
                    auditor.current['invalid'] = True
                continue
            auditor.consume(event, line_number)
        final = os.fstat(stream.fileno())
    report = auditor.result()
    report['source'] = {'path': str(path.resolve()), 'prefix_bytes': limit,
        'prefix_sha256': digest.hexdigest(), 'bytes_read': consumed,
        'file_size_at_open': initial.st_size, 'file_size_after_read': final.st_size,
        'device': initial.st_dev, 'inode': initial.st_ino,
        'complete_lines': line_number - bool(partial_bytes), 'trailing_partial_line_bytes': partial_bytes,
        'appended_bytes_not_audited': max(0, final.st_size - limit),
        'prefix_policy': 'Hash exactly the size pinned at open or requested; hash but do not parse an unfinished final line.'}
    report['analysis_sources'] = [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                  for p in (Path(__file__), Path(pairing.__file__))]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    parser.add_argument('--prefix-bytes', type=int)
    parser.add_argument('--max-details', type=int, default=100)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.resolve() == args.events.resolve():
        parser.error('report must name a new file distinct from events')
    try:
        report = audit_file(args.events, prefix_bytes=args.prefix_bytes, max_details=args.max_details)
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({'report': str(args.report), 'status': report['status'],
                      'bound_counts': report['bound_counts'], 'source': report['source']}, indent=2))
    return 1 if report['violation_count'] or report['status'] == 'anomaly_observed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
