#!/usr/bin/env python3
"""Audit recorded training-gradient ranges without loading tensors or using GPUs.

The input is an append-only monitor JSONL file. A fixed byte prefix is hashed
while it is streamed; an unfinished final line is retained in the hash but is
not parsed. Only attempts with every installed selected before/after call pair
present exactly once are checked. Missing operations/tensors and nonfinite or
unscanned summaries are explicit skips. Equality here means scalar min/max,
dtype and element-count agreement, never byte identity or mathematical
correctness. No full-training coverage or terminal trainer outcome is inferred.
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
import re
import sys


OUTPUT = 'hstu_output_grad_mm'
NORM = 'triton_layer_norm_mul_dropout_bwd'
WEIGHT = 'hstu_output_weight_mm'
ATTENTION = 'hstu_attention_bwd'
SILU = 'hstu_silu_bwd'
ADDMM = 'triton_addmm_bwd'
INPUT_NORM = 'triton_weighted_layer_norm_bwd'
DIRECT_EDGES = (
    ('output_grad.dy_to_output_norm.dy', (OUTPUT, 'outputs', 'dy'), (NORM, 'inputs', 'dy')),
    ('output_norm.d_attn_to_attention.dout', (NORM, 'outputs', 'd_attn'), (ATTENTION, 'inputs', 'dout')),
    ('output_norm.d_u_to_silu.grad_output', (NORM, 'outputs', 'd_u'), (SILU, 'inputs', 'grad_output')),
    ('output_norm.y_to_output_weight.y', (NORM, 'outputs', 'y'), (WEIGHT, 'inputs', 'y')),
    ('output_grad.dout_to_output_weight.dout', (OUTPUT, 'inputs', 'dout'), (WEIGHT, 'inputs', 'dout')),
    ('addmm.d_normed_x_to_input_norm.dy', (ADDMM, 'outputs', 'd_normed_x'), (INPUT_NORM, 'inputs', 'dy')),
)
PACKED_EDGE = 'packed_du_dv_dq_dk_to_addmm.dz'
RESIDUAL_EDGE = 'residual_to_lower_layer.dout'
FIELDS = ('name', 'dtype', 'numel', 'min', 'max', 'finite', 'scanned', 'shape',
          'extreme', 'extreme_count')
ANOMALY_FIELDS = ('bad_inputs', 'bad_outputs', 'extreme_inputs', 'extreme_outputs')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_number(value):
    return value if math.isfinite(value) else str(value)


def _layer_below(layer):
    match = re.fullmatch(r'(.*\.)([0-9]+)', layer)
    if not match:
        return None, 'unrecognized_layer_index'
    index = int(match[2])
    return (match[1] + str(index - 1), None) if index else (None, None)


def _attempt_key(event):
    attempt = event.get('attempt')
    if (not isinstance(attempt, dict) or not isinstance(attempt.get('step'), int)
            or isinstance(attempt['step'], bool)):
        raise ValueError('Event lacks an integer attempt.step')
    session = event.get('session')
    if not isinstance(session, str):
        raise ValueError('Event lacks a session string')
    return session, attempt.get('mode'), attempt.get('repeat'), attempt['step']


class FlowAudit:
    def __init__(self, max_details=100):
        self.max_details = max_details
        self.configs, self.last_steps = {}, {}
        self.current = None
        self.event_counts = Counter()
        self.edges = {name: {'checked': 0, 'passed': 0, 'violations': 0,
                             'skipped': 0, 'skip_reasons': Counter()}
                      for name in [*(edge[0] for edge in DIRECT_EDGES), PACKED_EDGE, RESIDUAL_EDGE]}
        self.violation_count, self.violations = 0, []
        self.attempt_count, self.complete_steps, self.incomplete_steps = 0, 0, 0
        self.first_checked_attempt = self.last_checked_attempt = None
        self.skipped_steps, self.endpoint_maxima = [], {}
        self.max_residual_ratio = None
        self.anomaly_events, self.capture_events, self.tensor_anomalies = 0, 0, 0
        self.anomaly_samples = []
        self.last_event = None

    def problem(self, kind, **details):
        self.violation_count += 1
        if len(self.violations) < self.max_details:
            self.violations.append({'kind': kind, **details})

    def _context(self, layer=None):
        return {'session': self.current['key'][0], 'attempt': self.current['attempt'],
                **({'layer': layer} if layer is not None else {})}

    def _anomaly_sample(self, value):
        if len(self.anomaly_samples) < self.max_details:
            self.anomaly_samples.append(value)

    def consume(self, event, line):
        name = event.get('event')
        self.event_counts[str(name)] += 1
        self.last_event = {key: event.get(key) for key in
                           ('event', 'session', 'attempt', 'call', 'layer', 'operation', 'unix_time')}
        self.last_event['line'] = line
        if name == 'dump_complete':
            self.capture_events += 1
        if 'anomaly' in str(name).lower() or any(event.get(key) for key in ANOMALY_FIELDS):
            self.anomaly_events += 1
            self._anomaly_sample({'kind': 'event', **self.last_event,
                                  **{key: event.get(key) for key in ANOMALY_FIELDS if event.get(key)}})
        direction = 'inputs' if name == 'before' else 'outputs' if name == 'after' else None
        if direction:
            for item in event.get(direction, []) or []:
                if (item.get('finite') is False or item.get('extreme') is True
                        or item.get('extreme_count', 0)):
                    self.tensor_anomalies += 1
                    self._anomaly_sample({'kind': 'tensor', **self.last_event, 'direction': direction,
                                          'tensor': {key: item.get(key) for key in FIELDS}})
        if name == 'installed':
            session, layers, operations = (event.get(key) for key in ('session', 'layers', 'selected_operations'))
            if (not isinstance(session, str) or not isinstance(layers, list) or not layers
                    or not all(isinstance(layer, str) for layer in layers)
                    or len(set(layers)) != len(layers) or not isinstance(operations, list)
                    or not operations or not all(isinstance(op, str) for op in operations)
                    or len(set(operations)) != len(operations)):
                self.problem('invalid_installed_expectations', line=line)
                return
            if session in self.configs:
                self.problem('duplicate_installed_session', line=line, session=session)
            self.configs[session] = {'layers': layers, 'operations': operations}
            return
        if name not in ('attempt_start', 'before', 'after'):
            return
        try:
            key = _attempt_key(event)
        except ValueError as error:
            self.problem('invalid_attempt_identity', line=line, error=str(error))
            if self.current:
                self.current['invalid'] = True
            return
        if name == 'attempt_start' or self.current is None or self.current['key'] != key:
            self.finish_step('next_attempt')
            self.attempt_count += 1
            self.current = {'key': key, 'attempt': event['attempt'], 'pairs': {},
                            'before_order': [], 'after_order': [], 'invalid': False,
                            'first_line': line, 'started': name == 'attempt_start'}
            previous = self.last_steps.get(key[:3])
            if previous is not None and key[3] <= previous:
                self.problem('non_increasing_attempt_step', line=line, previous=previous, **self._context())
                self.current['invalid'] = True
            self.last_steps[key[:3]] = key[3]
            if name != 'attempt_start':
                self.problem('missing_attempt_start', line=line, **self._context())
                self.current['invalid'] = True
        if name == 'attempt_start':
            return
        call = event.get('call')
        if not isinstance(call, int) or isinstance(call, bool) or call < 1:
            self.problem('invalid_call_id', line=line, **self._context())
            self.current['invalid'] = True
            return
        if not all(isinstance(event.get(key), str) for key in ('layer', 'operation')):
            self.problem('invalid_call_identity', line=line, call=call, **self._context())
            self.current['invalid'] = True
            return
        pair = self.current['pairs'].setdefault(call, {})
        if name in pair:
            self.problem('duplicate_call_event', line=line, event=name, call=call, **self._context())
            self.current['invalid'] = True
            return
        compact = {key: event.get(key) for key in ('layer', 'operation', 'call')}
        compact.update(line=line, tensors=[{key: item.get(key) for key in FIELDS if key in item}
                                          for item in event.get(direction, []) or []])
        pair[name] = compact
        self.current[name + '_order'].append(call)

    def skip(self, edge, reason):
        self.edges[edge]['skipped'] += 1
        self.edges[edge]['skip_reasons'][reason] += 1

    def checked(self, edge, layer, differences, **details):
        counter = self.edges[edge]
        counter['checked'] += 1
        if differences:
            counter['violations'] += 1
            self.problem('edge_violation', edge=edge, differences=differences,
                         **self._context(layer), **details)
        else:
            counter['passed'] += 1

    def tensor(self, pairs, layer, endpoint):
        operation, direction, name = endpoint
        pair = pairs.get((layer, operation))
        if pair is None:
            return None, 'missing_operation'
        event = pair['before' if direction == 'inputs' else 'after']
        values = [item for item in event['tensors'] if item.get('name') == name]
        if not values:
            return None, 'missing_tensor'
        if len(values) != 1:
            return None, 'duplicate_tensor_name'
        item = values[0]
        if item.get('scanned') is not True:
            return None, 'unscanned_tensor'
        if item.get('finite') is not True:
            return None, 'nonfinite_or_unknown_tensor'
        if item.get('numel') == 0:
            return None, 'empty_tensor'
        if (not isinstance(item.get('numel'), int) or isinstance(item.get('numel'), bool)
                or item['numel'] < 0 or not isinstance(item.get('dtype'), str)
                or not _number(item.get('min')) or not _number(item.get('max'))
                or item['min'] > item['max']):
            return None, 'invalid_tensor_summary'
        shape = item.get('shape')
        if shape is not None and (not isinstance(shape, list)
                or not all(isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in shape)
                or math.prod(shape) != item['numel']):
            return None, 'invalid_tensor_summary'
        result = {**item, 'line': event['line'], 'call': event['call']}
        magnitude = max(abs(item['min']), abs(item['max']))
        key = '.'.join(endpoint)
        if key not in self.endpoint_maxima or magnitude > self.endpoint_maxima[key]['max_abs']:
            self.endpoint_maxima[key] = {'max_abs': magnitude, 'min': item['min'], 'max': item['max'],
                                         'line': event['line'], **self._context(layer)}
        return result, None

    def endpoints(self, pairs, edge, layer, requested):
        values = []
        for owner, endpoint in requested:
            value, reason = self.tensor(pairs, owner, endpoint)
            if reason:
                if reason in ('invalid_tensor_summary', 'duplicate_tensor_name'):
                    self.checked(edge, layer, [reason], endpoint=list(endpoint), endpoint_layer=owner)
                else:
                    self.skip(edge, reason)
                return None
            values.append(value)
        return values

    def audit_layer(self, pairs, layer):
        for edge, source, destination in DIRECT_EDGES:
            values = self.endpoints(pairs, edge, layer, [(layer, source), (layer, destination)])
            if values is None:
                continue
            left, right = values
            differences = [field for field in ('dtype', 'numel', 'min', 'max') if left[field] != right[field]]
            if left['line'] >= right['line']:
                differences.append('observation_order')
            self.checked(edge, layer, differences, source=left, destination=right)
        requested = [(layer, (SILU, 'outputs', 'du'))]
        requested += [(layer, (ATTENTION, 'outputs', name)) for name in ('dv', 'dq', 'dk')]
        requested.append((layer, (ADDMM, 'inputs', 'dz')))
        values = self.endpoints(pairs, PACKED_EDGE, layer, requested)
        if values is not None:
            parts, destination = values[:-1], values[-1]
            expected = {'numel': sum(part['numel'] for part in parts),
                        'min': min(part['min'] for part in parts),
                        'max': max(part['max'] for part in parts)}
            differences = [field for field, value in expected.items() if destination[field] != value]
            if any(part['dtype'] != destination['dtype'] for part in parts):
                differences.append('dtype')
            if any(part['line'] >= destination['line'] for part in parts):
                differences.append('observation_order')
            self.checked(PACKED_EDGE, layer, differences, expected_union=expected,
                         parts=parts, destination=destination)
        lower, reason = _layer_below(layer)
        if lower is None:
            if reason:
                self.skip(RESIDUAL_EDGE, reason)
            return
        values = self.endpoints(pairs, RESIDUAL_EDGE, layer,
                                [(layer, (INPUT_NORM, 'outputs', 'd_x')),
                                 (layer, (OUTPUT, 'inputs', 'dout')),
                                 (lower, (OUTPUT, 'inputs', 'dout'))])
        if values is None:
            return
        dx, bypass, destination = values
        differences = [field for field in ('dtype', 'numel')
                       if any(value[field] != destination[field] for value in (dx, bypass))]
        if max(dx['line'], bypass['line']) >= destination['line']:
            differences.append('observation_order')
        if differences:
            self.checked(RESIDUAL_EDGE, layer, differences, sources=[dx, bypass], destination=destination)
            return
        magnitudes = [max(abs(value['min']), abs(value['max'])) for value in values]
        scale = max(magnitudes[:2])
        if scale == 0:
            ratio = 0.0 if magnitudes[2] == 0 else math.inf
            within_bound = magnitudes[2] == 0  # No absolute epsilon can excuse zero -> nonzero.
        else:
            scaled_sum = magnitudes[0] / scale + magnitudes[1] / scale
            scaled_lower = magnitudes[2] / scale
            ratio = scaled_lower / scaled_sum
            # Factor two is the deliberately generous arithmetic allowance;
            # 64 double epsilons cover only host scalar comparison rounding.
            within_bound = scaled_lower <= 2 * scaled_sum * (1 + 64 * sys.float_info.epsilon)
        if not within_bound:
            differences.append('residual_bound')
        location = {'ratio': _json_number(ratio), 'd_x_abs': magnitudes[0],
                    'upper_dout_abs': magnitudes[1], 'lower_dout_abs': magnitudes[2],
                    'lower_layer': lower, **self._context(layer)}
        if self.max_residual_ratio is None or ratio > self.max_residual_ratio[0]:
            self.max_residual_ratio = ratio, location
        self.checked(RESIDUAL_EDGE, layer, differences, bound_observation=location)

    def finish_step(self, closed_by):
        if self.current is None:
            return
        state = self.current
        config = self.configs.get(state['key'][0])
        pairs, unmatched = {}, []
        for call, pair in state['pairs'].items():
            if set(pair) != {'before', 'after'}:
                unmatched.append({'call': call, 'events': sorted(pair)})
                continue
            before, after = pair['before'], pair['after']
            if (before['layer'], before['operation']) != (after['layer'], after['operation']):
                self.problem('call_pair_identity_mismatch', call=call, **self._context())
                state['invalid'] = True
                continue
            identity = before['layer'], before['operation']
            if before['line'] >= after['line'] or identity in pairs:
                self.problem('invalid_call_pair_order_or_duplicate_operation', call=call, **self._context())
                state['invalid'] = True
            pairs[identity] = pair
        if (not unmatched and state['before_order'] != state['after_order']):
            self.problem('before_after_call_order_mismatch', **self._context())
            state['invalid'] = True
        expected = {(layer, op) for layer in config['layers'] for op in config['operations']} if config else set()
        missing = sorted(expected - set(pairs))
        unexpected = sorted(set(pairs) - expected) if config else []
        complete = bool(config) and not (state['invalid'] or unmatched or missing or unexpected)
        layers = config['layers'] if config else sorted({layer for layer, _ in pairs if isinstance(layer, str)})
        if complete:
            self.complete_steps += 1
            self.last_checked_attempt = self._context()
            if self.first_checked_attempt is None:
                self.first_checked_attempt = self.last_checked_attempt
            for layer in layers:
                self.audit_layer(pairs, layer)
        else:
            self.incomplete_steps += 1
            for layer in layers:
                for edge, *_ in DIRECT_EDGES:
                    self.skip(edge, 'incomplete_selected_step')
                self.skip(PACKED_EDGE, 'incomplete_selected_step')
                lower, reason = _layer_below(layer)
                if lower is not None or reason:
                    self.skip(RESIDUAL_EDGE, 'incomplete_selected_step')
            if len(self.skipped_steps) < self.max_details:
                self.skipped_steps.append({**self._context(), 'closed_by': closed_by,
                    'missing_pairs': [list(pair) for pair in missing],
                    'unexpected_pairs': [list(pair) for pair in unexpected],
                    'unmatched_calls': unmatched, 'invalid_pairing': state['invalid'],
                    'installed_expectations_available': config is not None})
        self.current = None

    def result(self):
        self.finish_step('prefix_end')
        anomaly = bool(self.anomaly_events or self.tensor_anomalies)
        status = ('violations_found' if self.violation_count else 'anomaly_observed' if anomaly else
                  'checked_prefix_no_violations' if self.complete_steps else 'insufficient_complete_steps')
        return {'format': 'training_gradient_flow_scalar_audit_v1', 'status': status,
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'attempts_seen': self.attempt_count, 'selected_boundary_steps_complete': self.complete_steps,
            'first_checked_attempt': self.first_checked_attempt, 'last_checked_attempt': self.last_checked_attempt,
            'selected_boundary_steps_skipped': self.incomplete_steps, 'skipped_steps': self.skipped_steps,
            'skipped_step_details_truncated': self.incomplete_steps > len(self.skipped_steps),
            'installed_sessions': self.configs, 'event_counts': dict(self.event_counts),
            'last_parsed_event': self.last_event, 'edge_counts': self.edges,
            'violation_count': self.violation_count, 'violations': self.violations,
            'violation_details_truncated': self.violation_count > len(self.violations),
            'anomalies': {'explicit_event_count': self.anomaly_events, 'capture_event_count': self.capture_events,
                          'tensor_observation_count': self.tensor_anomalies, 'samples': self.anomaly_samples,
                          'samples_truncated': self.anomaly_events + self.tensor_anomalies > len(self.anomaly_samples)},
            'max_residual_ratio': self.max_residual_ratio[1] if self.max_residual_ratio else None,
            'endpoint_magnitude_maxima': self.endpoint_maxima,
            'scope': ['Only recorded scalar min/max, dtype and numel consistency; not byte identity or mathematical correctness.',
                      'Before.inputs and after.outputs are used; repeated after.inputs are never substituted for pre-call inputs.',
                      'Only fully paired installed selected-operation steps are checked; no whole-training coverage or trainer completion is inferred.',
                      'Reshapes preserve numel and extrema; tensor shapes need not match between endpoints.',
                      'Packed union checks do not prove storage aliasing, partition coverage or absence of intervening mutations.',
                      'Residual bound is lower_abs <= 2 * (upper_input_norm_dx_abs + upper_output_dout_abs), with finite summaries and matching dtype/numel.',
                      'Endpoint maxima can come from different steps and layers; they are not one causal failing-value chain.']}


def audit_file(path, *, prefix_bytes=None, max_details=100):
    path = Path(path)
    if max_details < 0:
        raise ValueError('max_details must be nonnegative')
    audit, digest = FlowAudit(max_details), hashlib.sha256()
    consumed, line_number, partial_bytes = 0, 0, 0
    with path.open('rb') as stream:
        initial = os.fstat(stream.fileno())
        limit = initial.st_size if prefix_bytes is None else prefix_bytes
        if not isinstance(limit, int) or not 0 <= limit <= initial.st_size:
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
                audit.problem('malformed_complete_line', line=line_number, error=str(error))
                if audit.current:
                    audit.current['invalid'] = True
                continue
            audit.consume(event, line_number)
        final = os.fstat(stream.fileno())
    report = audit.result()
    report['source'] = {'path': str(path.resolve()), 'prefix_bytes': limit,
        'prefix_sha256': digest.hexdigest(), 'bytes_read': consumed,
        'file_size_at_open': initial.st_size, 'file_size_after_read': final.st_size,
        'device': initial.st_dev, 'inode': initial.st_ino,
        'complete_lines': line_number - bool(partial_bytes), 'trailing_partial_line_bytes': partial_bytes,
        'appended_bytes_not_audited': max(0, final.st_size - limit),
        'prefix_policy': 'Read only the size pinned at open (or requested prefix); include an unfinished final line in the SHA but skip parsing it.'}
    report['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    parser.add_argument('--prefix-bytes', type=int)
    parser.add_argument('--max-details', type=int, default=100)
    parser.add_argument('--report', type=Path, help='new report path; default is stdout')
    args = parser.parse_args(argv)
    if args.max_details < 0 or (args.report and (args.report.exists() or args.report.resolve() == args.events.resolve())):
        parser.error('max-details must be nonnegative and report must name a new file distinct from events')
    try:
        report = audit_file(args.events, prefix_bytes=args.prefix_bytes, max_details=args.max_details)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.report:
        with args.report.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')
    return 1 if report['violation_count'] or report['status'] == 'anomaly_observed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
