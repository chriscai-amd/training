#!/usr/bin/env python3
"""CPU-only conditional attention bounds from resolved deferred-flush scans.

The full deferred event auditor checks queued/resolved structure first. Only
resolved scan summaries are passed to the unchanged exact-real bound helper.
Original JSONL locations and scan/enqueue indices remain attached to results.
No Torch import, GPU work, tensor replay, or training-success claim is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import audit_all_layers_real_attention_bounds as real_bounds
import audit_deferred_training_trace as deferred_events

MAX_RECORD_BYTES = 16 << 20


def identity(event):
    attempt = event.get('attempt')
    if not isinstance(attempt, dict):
        raise ValueError('Missing attempt identity')
    session, mode = event.get('session'), attempt.get('mode')
    repeat, step = attempt.get('repeat'), attempt.get('step')
    if (not isinstance(session, str) or not session or not isinstance(mode, str)
            or not mode or type(repeat) is not int or repeat < 0
            or type(step) is not int or step < 1):
        raise ValueError('Invalid session/mode/repeat/step identity')
    return session, mode, repeat, step


def source_identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


class DeferredAttentionAudit:
    def __init__(self):
        self.structure = deferred_events.DeferredEventAudit()
        self.current = None
        self.queued_locations = {}
        self.rows = []
        self.skips = []
        self.flows = []
        self.flushes = 0
        self.queued_records = 0

    def consume(self, event, location):
        kind = event.get('event')
        if kind == 'attempt_start':
            current = identity(event)
            if self.current is not None and current[:3] != self.current[:3]:
                raise ValueError('Session/mode/repeat changed within one trace')
            self.current = current
            self.queued_locations = {}
        elif (kind in ('before', 'after', 'dense_phase_queued', 'deferred_flush',
                       'deferred_trace_fault_complete') or event.get('attempt') is not None):
            if identity(event) != self.current:
                raise ValueError('Event full identity differs from current attempt')
        # This validates the 42+2 coverage, call ownership, stream metadata,
        # queued/resolved metadata equality, and explicit unresolved state.
        self.structure.consume(event)
        if kind in ('before', 'after', 'dense_phase_queued'):
            self.queued_records += 1
            self.queued_locations[event['enqueue_index']] = location
        elif kind == 'deferred_flush':
            if identity(event) != self.current:
                raise ValueError('Flush full identity differs from current attempt')
            self._flush(event, location)

    def _flush(self, event, location):
        self.flushes += 1
        session, mode, repeat, step = self.current
        base = {'session': session, 'mode': mode, 'repeat': repeat, 'step': step}
        pairs = {}
        for index, scan in enumerate(event['scans']):
            if scan['kind'] != 'operation' or scan['operation'] != real_bounds.ATTENTION:
                continue
            if scan['values_resolved_on_host'] is not True:
                raise ValueError('Attention scan is unresolved')
            if type(scan['call']) is not int or scan['call'] < 1:
                raise ValueError('Invalid attention call identity')
            pair = pairs.setdefault((scan['layer'], scan['call']), {})
            pair[scan['phase']] = (index, scan)
        violation = False
        for (layer, call), pair in pairs.items():
            bi, before = pair['before']
            ai, after = pair['after']
            # The shared structure auditor has already required both phases
            # in order, without assuming adjacent enqueues for a call.
            context = {**base, 'layer': layer, 'call': call,
                       'operation': real_bounds.ATTENTION,
                       'provenance': {'resolved_record': location,
                                      'before_scan_index': bi, 'after_scan_index': ai,
                                      'before_enqueue_index': before['enqueue_index'],
                                      'after_enqueue_index': after['enqueue_index'],
                                      'before_queued_record': self.queued_locations[before['enqueue_index']],
                                      'after_queued_record': self.queued_locations[after['enqueue_index']]}}
            # These temporary dictionaries only adapt the helper's interface;
            # they are never emitted as fictitious original JSONL events.
            inputs = {'inputs': before['summaries'],
                      'scalar_arguments': before.get('scalar_arguments')}
            outputs = {'outputs': after['summaries'],
                       'scalar_arguments': after.get('scalar_arguments')}
            try:
                result = real_bounds.bounds_for_pair(inputs, outputs)
            except (ValueError, KeyError, TypeError) as error:
                self.skips.append({**context, 'reason': str(error)})
                continue
            row = {**context, **result}
            self.rows.append(row)
            violation |= any(item['exceeds_real_bound'] for item in result['outputs'].values())
        if violation:
            for index, scan in enumerate(event['scans']):
                if scan['kind'] != 'operation':
                    continue
                self.flows.append({**base, 'layer': scan['layer'], 'call': scan['call'],
                                   'operation': scan['operation'], 'phase': scan['phase'],
                                   'provenance': {'resolved_record': location,
                                                  'scan_index': index,
                                                  'enqueue_index': scan['enqueue_index'],
                                                  'queued_record': self.queued_locations[scan['enqueue_index']]},
                                   'summaries': [dict(item) for item in scan['summaries'] if item['scanned']]})

    def report(self):
        violations = [row for row in self.rows
                      if any(item['exceeds_real_bound'] for item in row['outputs'].values())]
        status = ('checked_resolved_scans_with_bound_exceedances' if violations else
                  'checked_resolved_scans_no_bound_exceedances' if self.rows else
                  'no_supported_resolved_attention_pairs')
        if self.skips:
            status += '_with_skips'
        return {
            'format': 'deferred_resolved_attention_bounds_v1', 'status': status,
            'resolved_flushes': self.flushes, 'complete_attention_pairs_checked': len(self.rows),
            'checked_pairs_by_layer': dict(Counter(row['layer'] for row in self.rows)),
            'queued_records_ignored_numerically': self.queued_records,
            'real_bound_exceeding_calls': len(violations),
            'real_bound_exceeding_output_counts': dict(Counter(
                name for row in violations for name, item in row['outputs'].items()
                if item['exceeds_real_bound'])),
            'violations': violations, 'skips': self.skips,
            'all_checked_calls': self.rows, 'flow_for_violating_attempts': self.flows,
            'event_structure_audit': self.structure.report(),
            'arithmetic': 'Unchanged bounds_for_pair: exact Fraction real arithmetic; outward-rounded displayed bounds.',
            'scope': [
                'Only values_resolved_on_host=true summaries inside deferred_flush.scans are numerical observations. Standalone queued records supply provenance and structural checks only.',
                'Each result retains session, mode, repeat, step, layer, call, original flush JSONL line/byte offset/hash, scan index, enqueue index, and queued source line/offset/hash. Indices are zero-based; JSONL line numbers are one-based.',
                'Actual N/alpha and matching scalar arguments are required. Valid disjoint sequence offsets and L<=N remain assumptions; integer offset values are not scanned.',
                'Mathematical zero initialization and exact exp/reciprocal/arithmetic are assumed. Initial destination bytes, actual FP32/BF16 arithmetic, overflow/underflow and compiled scheduling are not certified.',
                'Full queued/resolved structure is checked by the separate event auditor. Resolved attempts do not independently prove optimizer-step completion or training success.',
                'Recorded enqueue order does not establish total physical causality across streams. Deferred observations resolve after backward, so a flagged input can already have propagated.',
                'Flow entries are successive scalar summaries from the same resolved flush, not elementwise identities or proof of a huge-value mechanism. No raw tensors or pristine replay inputs are retained.',
                'Finite bound passes do not establish correct output. Nonfinite read-input calls are skipped explicitly; an output nonfinite with supported finite inputs is a conditional bound violation.',
            ],
        }


def audit_file(path, *, prefix_bytes=None, expected_prefix_sha256=None):
    path = Path(path)
    audit = DeferredAttentionAudit()
    digest, complete_digest = hashlib.sha256(), hashlib.sha256()
    consumed = complete_bytes = complete_lines = partial_bytes = 0
    with path.open('rb') as stream:
        initial = os.fstat(stream.fileno())
        limit = initial.st_size if prefix_bytes is None else prefix_bytes
        if type(limit) is not int or not 0 <= limit <= initial.st_size:
            raise ValueError('prefix_bytes must lie within file size at open')
        while consumed < limit:
            offset = consumed
            raw = stream.readline(min(limit - consumed, MAX_RECORD_BYTES + 1))
            if not raw:
                raise ValueError('File truncated while reading pinned prefix')
            if len(raw) > MAX_RECORD_BYTES:
                raise ValueError('JSONL record exceeds 16 MiB limit')
            digest.update(raw)
            consumed += len(raw)
            if not raw.endswith(b'\n'):
                if consumed != limit:
                    raise ValueError('File truncated inside pinned prefix')
                partial_bytes = len(raw)
                break

            def reject_constant(value):
                raise ValueError('Nonstandard JSON numeric constant: ' + value)

            def unique_keys(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('Duplicate JSON key: ' + key)
                    result[key] = value
                return result

            event = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
            if not isinstance(event, dict):
                raise ValueError('JSONL record must be an object')
            complete_lines += 1
            location = {'line': complete_lines, 'byte_offset': offset, 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest()}
            try:
                audit.consume(event, location)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'Invalid deferred event at JSONL line {complete_lines}: {error}') from error
            complete_digest.update(raw)
            complete_bytes += len(raw)
        final = os.fstat(stream.fileno())
    if expected_prefix_sha256 is not None and digest.hexdigest() != expected_prefix_sha256.lower():
        raise ValueError('Pinned prefix SHA256 differs from expected identity')
    report = audit.report()
    report['source'] = {
        'path': str(path.resolve()), 'prefix_bytes': limit, 'prefix_sha256': digest.hexdigest(),
        'complete_record_bytes': complete_bytes, 'complete_record_sha256': complete_digest.hexdigest(),
        'complete_lines': complete_lines, 'trailing_partial_line_bytes': partial_bytes,
        'file_size_at_open': initial.st_size, 'file_size_after_read': final.st_size,
        'appended_or_outside_prefix_bytes_not_audited': max(0, final.st_size - limit),
        'device': initial.st_dev, 'inode': initial.st_ino,
        'expected_prefix_sha256': expected_prefix_sha256,
        'policy': 'Pin size at open or requested prefix. Hash every requested byte including a partial final record; parse complete newline-terminated records only. Also hash parsed complete records separately. Hashes identify observed bytes, not historical immutability.',
    }
    report['analysis_sources'] = [source_identity(path) for path in (
        __file__, real_bounds.__file__, deferred_events.__file__)]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--prefix-bytes', type=int)
    parser.add_argument('--expected-prefix-sha256')
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError(args.report)
    report = audit_file(args.events, prefix_bytes=args.prefix_bytes,
                        expected_prefix_sha256=args.expected_prefix_sha256)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in (
        'status', 'resolved_flushes', 'complete_attention_pairs_checked',
        'real_bound_exceeding_calls', 'real_bound_exceeding_output_counts', 'source')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
