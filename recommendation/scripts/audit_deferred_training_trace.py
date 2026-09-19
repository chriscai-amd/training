#!/usr/bin/env python3
"""Read-only deferred trace audit/watcher; imports no Torch and launches no GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

SUMMARY_RESULTS = {"finite", "extreme", "min", "max", "max_abs_finite"}
SCAN_RESULTS = {"summaries", "values_resolved_on_host", "flagged_names"}


class DeferredEventAudit:
    def __init__(self):
        self.expected = None
        self.parameter_names = None
        self.excluded = None
        self.current_step = None
        self.queued = {}
        self.flushed = False
        self.attempts = []
        self.first_flagged = None
        self.streams = set()
        self.events = 0

    def consume(self, event):
        self.events += 1
        event_type = event["event"]
        if event_type == "installed":
            if self.expected is not None:
                raise ValueError("Repeated installation record")
            selected = event["resolved_layer_operations"]
            self.expected = {(layer, operation, phase) for layer, operations in selected.items()
                             for operation in operations for phase in ("before", "after")}
            if len(selected) != 3 or len(self.expected) != 42:
                raise ValueError("Installed trace does not cover exactly 42 endpoints")
            return
        if event_type == "dense_sentinel_installed":
            self.parameter_names = set(event["dense_parameter_names"])
            self.excluded = event["excluded_optimizer_parameters"]
            if not self.parameter_names or len(self.parameter_names) != len(event["dense_parameter_names"]):
                raise ValueError("Invalid dense parameter selection")
            return
        if event_type == "attempt_start":
            step = event["attempt"]["step"]
            if step != (self.current_step or 0) + 1 or self.current_step is not None and not self.flushed:
                raise ValueError("Attempt starts without a preceding resolved trace")
            self.current_step, self.queued, self.flushed = step, {}, False
            return
        if event.get("attempt") is not None and event["attempt"]["step"] != self.current_step:
            raise ValueError("Event attempt differs from current attempt")
        if event_type in ("before", "after", "dense_phase_queued"):
            if self.current_step is None or self.flushed:
                raise ValueError("Queued scan is outside an open attempt")
            index = event["enqueue_index"]
            if index != len(self.queued) or event["values_resolved_on_host"] is not False:
                raise ValueError("Invalid queue index or prematurely resolved queued event")
            if any(SUMMARY_RESULTS.intersection(item) for item in event["summaries"]):
                raise ValueError("Queued event contains resolved numeric results")
            self.queued[index] = event
            return
        if event_type == "deferred_flush":
            return self._flush(event)
        if event_type == "deferred_trace_fault_complete":
            if self.first_flagged is None or event["first_flagged_enqueue_index"] != self.first_flagged["enqueue_index"]:
                raise ValueError("Fault artifact event does not match the first flagged observation")

    def _flush(self, event):
        if self.flushed or self.expected is None or self.parameter_names is None:
            raise ValueError("Unexpected deferred flush")
        scans = event["scans"]
        if event["scan_count"] != 44 or len(scans) != 44 or set(self.queued) != set(range(44)):
            raise ValueError("Deferred flush must resolve exactly 42 operation plus two dense scans")
        endpoints = Counter()
        pairs = {}
        dense = []
        observed_streams = set()
        producer_streams = set()
        flagged = []
        missing_gradients = None
        for index, scan in enumerate(scans):
            queued = self.queued[index]
            if scan["enqueue_index"] != index or scan["values_resolved_on_host"] is not True:
                raise ValueError("Invalid resolved scan ordering/state")
            if any(queued.get(key) != value for key, value in scan.items() if key not in SCAN_RESULTS):
                raise ValueError("Resolved scan metadata differs from its queued observation")
            if len(scan["summaries"]) != len(queued["summaries"]):
                raise ValueError("Resolved scan changes tensor coverage")
            for before, after in zip(queued["summaries"], scan["summaries"]):
                if before != {key: value for key, value in after.items() if key not in SUMMARY_RESULTS}:
                    raise ValueError("Resolved tensor metadata differs from queued metadata")
                if not SUMMARY_RESULTS.issubset(after):
                    raise ValueError("Resolved tensor is missing numeric result fields")
                device, stream = after["source_device"], after["stream_id"]
                if (device.startswith("cuda") and type(stream) is not int
                        or device == "cpu" and stream is not None):
                    raise ValueError("Invalid per-summary stream identity")
                observed_streams.add((device, stream))
                if after["scanned"] and after["numel"]:
                    producer_streams.add((device, stream))
            names = [item["name"] for item in scan["summaries"] if not item["finite"] or item["extreme"]]
            if scan["flagged_names"] != names:
                raise ValueError("Flagged names disagree with resolved scalar results")
            if names:
                flagged.append(scan)
            if scan["kind"] == "operation":
                identity = (scan["layer"], scan["operation"], scan["phase"])
                endpoints[identity] += 1
                call = scan["call"]
                pair = pairs.setdefault(call, {"owner": identity[:2], "phases": []})
                if pair["owner"] != identity[:2]:
                    raise ValueError("Call ID is shared by different operation owners")
                pair["phases"].append(scan["phase"])
            elif scan["kind"] == "dense_phase":
                dense.append((index, scan["phase"]))
                params = {item["name"].removeprefix("parameters/") for item in scan["summaries"]
                          if item["name"].startswith("parameters/")}
                grads = {item["name"].removeprefix("gradients/") for item in scan["summaries"]
                         if item["name"].startswith("gradients/")}
                if params != self.parameter_names:
                    raise ValueError("Dense parameter coverage differs from installed selection")
                if scan["phase"] == "before_forward" and grads:
                    raise ValueError("Before-forward phase unexpectedly scans gradients")
                if scan["phase"] == "after_backward_before_clip":
                    missing_gradients = scan["missing_gradient_names"]
                    if not grads or grads != self.parameter_names - set(missing_gradients):
                        raise ValueError("Accumulated gradient coverage/missing list disagree")
            else:
                raise ValueError("Unexpected scan kind")
        if set(endpoints) != self.expected or set(endpoints.values()) != {1}:
            raise ValueError("Missing, duplicate or unexpected operation endpoints")
        if len(pairs) != 21 or any(pair["phases"] != ["before", "after"] for pair in pairs.values()):
            raise ValueError("Operation call IDs do not form 21 unique before/after pairs")
        if dense != [(0, "before_forward"), (43, "after_backward_before_clip")]:
            raise ValueError("Dense scans do not surround all 42 operation observations")
        first = flagged[0]["enqueue_index"] if flagged else None
        if event["first_flagged_enqueue_index"] != first:
            raise ValueError("Flush first-flagged index is inconsistent")
        if event["phase"] != "after_backward_before_clip" or event["raw_snapshot_present"] is not False:
            raise ValueError("Unexpected flush phase or raw-snapshot claim")
        summary = {"step": self.current_step, "queued_operation_endpoints": 42, "queued_dense_scans": 2,
                   "resolved_scans": 44, "unique_paired_calls": 21, "missing_gradient_names": missing_gradients,
                   "recorded_tensor_streams": sorted(observed_streams),
                   "recorded_reduction_producer_streams": sorted(producer_streams),
                   "recorded_reduction_stream_count": len(producer_streams),
                   "expected_readback_devices_from_scanned_tensors": sorted({device for device, _ in producer_streams}),
                   "first_flagged_enqueue_index": first, "flagged_scan_count": len(flagged)}
        self.streams.update(producer_streams)
        if flagged and self.first_flagged is None:
            self.first_flagged = {"step": self.current_step, **copy.deepcopy(flagged[0])}
        self.attempts.append(summary)
        self.flushed = True
        return summary

    def report(self, outcome=None):
        return {"format": "deferred_training_event_audit_v1", "event_records": self.events,
                "resolved_attempts": self.attempts, "resolved_attempt_count": len(self.attempts),
                "latest_observed_attempt": self.current_step,
                "latest_attempt_has_flush": self.flushed,
                "latest_attempt_queued_scan_count": len(self.queued),
                "dense_parameter_count": len(self.parameter_names or ()),
                "excluded_optimizer_parameters": self.excluded,
                "observed_reduction_streams": sorted(self.streams),
                "observed_reduction_stream_count": len(self.streams),
                "first_flagged_observation_in_host_enqueue_order": self.first_flagged,
                "outcome": outcome,
                "placement_evidence": {
                    "direct_event_observation": "Two queued dense phases surround 42 operation scans; one resolved flush labels after_backward_before_clip. No explicit optimizer-entry/return events are emitted.",
                    "source_enforced_order": "Reviewed frozen callback sets backward_complete only after flush; optimizer proxy requires that flag before delegating step; next attempt requires optimizer_complete.",
                    "limit": "This is a source-backed ordering guarantee, not an independent GPU timestamp measurement of the optimizer call."},
                "limits": ["Observed stream IDs describe diagnostic current streams, not every internal PyTorch/TorchRec/GPU stream.",
                           "First flagged host enqueue index is not total cross-stream physical causality.",
                           "Flags resolve only after backward, so bad before-call inputs can propagate through downstream work.",
                           "Resolved attempts are not automatically completed optimizer/training steps; terminal outcome or the next attempt supplies additional progress evidence.",
                           "No raw input/output snapshot or pristine replay evidence is present."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--watch', action='store_true')
    options = parser.parse_args(argv)
    path = options.directory / 'boundaries/boundaries.jsonl'
    audit, first_notified, last_progress = DeferredEventAudit(), False, 0
    parsed_digest, parsed_bytes = hashlib.sha256(), 0

    def consume_line(line):
        nonlocal first_notified, last_progress, parsed_bytes
        if not line.endswith(b'\n'):
            raise ValueError('Terminal event file ends in an incomplete record')
        summary = audit.consume(json.loads(line))
        parsed_digest.update(line)
        parsed_bytes += len(line)
        if audit.first_flagged is not None and not first_notified:
            print(json.dumps({'FIRST_FLAGGED_ENQUEUE_OBSERVATION': audit.first_flagged}), flush=True)
            first_notified = True
        if summary and len(audit.attempts) // 50 > last_progress:
            last_progress = len(audit.attempts) // 50
            print(json.dumps({'resolved_attempts': len(audit.attempts),
                              'observed_reduction_stream_count': len(audit.streams)}), flush=True)

    with path.open('rb') as stream:
        while True:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    stream.seek(position)
                    break
                consume_line(line)
            outcome = json.loads((options.directory / 'outcome.json').read_text())
            terminal = outcome['status'] not in ('installing', 'running')
            if not options.watch or terminal:
                # The terminal outcome is written after the probe closes its
                # event file; drain a possible last append before finalizing.
                if terminal:
                    for line in stream:
                        consume_line(line)
                break
            time.sleep(3)
    report = audit.report(outcome)
    report['run_directory'] = str(options.directory)
    report['parsed_events_prefix_sha256'] = parsed_digest.hexdigest()
    report['parsed_events_prefix_bytes'] = parsed_bytes
    report['events_bytes_at_report'] = path.stat().st_size
    report['unparsed_event_bytes_at_report'] = report['events_bytes_at_report'] - parsed_bytes
    options.report.parent.mkdir(parents=True, exist_ok=True)
    with options.report.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(options.report), 'outcome': outcome,
                      'resolved_attempts': len(audit.attempts), 'observed_reduction_stream_count': len(audit.streams)}), flush=True)


if __name__ == '__main__':
    main()
