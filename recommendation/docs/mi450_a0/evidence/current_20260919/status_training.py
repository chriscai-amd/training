#!/usr/bin/env python3
"""Read-only handoff status. Does not import Torch or touch a GPU queue."""
import argparse
import collections
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import time

p = argparse.ArgumentParser()
p.add_argument('--run', default='train_linear_projection_diagnostics_cap256_bn1_workspace1_privateDW_start0_1000_v1')
p.add_argument('--output', type=Path)
a = p.parse_args()
D = Path('/home/chcai/dlrm_data/root_cause_20260919')
L = Path('/home/chcai/mi450_logs/root_cause_20260919')
directory = D / a.run
outcome = json.loads((directory / 'outcome.json').read_text()) if (directory / 'outcome.json').exists() else None
counts = collections.Counter()
last_complete = last_flush = last_attempt = 0
flagged = []
last_session = None
events_path = directory / 'boundaries/events.jsonl'
if events_path.exists():
    with events_path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                continue
            e = json.loads(line)
            counts[e['event']] += 1
            step = (e.get('attempt') or {}).get('step', 0)
            last_attempt = max(last_attempt, step)
            last_session = e.get('session', last_session)
            if e['event'] == 'attempt_backward_complete':
                last_complete = max(last_complete, step)
            if e['event'] == 'deferred_flush':
                last_flush = max(last_flush, step)
                for scan in e['scans']:
                    if scan['flagged_names']:
                        flagged.append({'step': step, 'stage': scan.get('stage'),
                                        'enqueue_index': scan['enqueue_index'],
                                        'names': scan['flagged_names']})
owners, owner_errors = [], []
if os.geteuid() == 0:
    device = os.stat('/dev/kfd')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            with os.scandir(proc / 'fd') as fds:
                for fd in fds:
                    try:
                        info = fd.stat(follow_symlinks=True)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISCHR(info.st_mode) and info.st_rdev == device.st_rdev:
                        status = (proc / 'status').read_text().splitlines()
                        nspid = next(x.split(':', 1)[1].split() for x in status if x.startswith('NSpid:'))
                        ticks = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
                        owners.append({'host_pid': int(proc.name), 'namespace_pids': list(map(int, nspid)),
                                       'command': (proc / 'comm').read_text().strip(), 'start_ticks': ticks})
                        break
        except FileNotFoundError:
            continue
        except PermissionError as error:
            owner_errors.append(str(error))
else:
    owner_errors.append('Run with sudo -n for complete cross-namespace KFD owner inventory')
files = {}
for path in [events_path, L / (a.run + '.log'), D / (a.run + '_hip.jsonl')]:
    if path.exists():
        s = path.stat()
        files[str(path)] = {'bytes': s.st_size, 'seconds_since_write': time.time() - s.st_mtime}
record = {
    'format': 'private_DW_training_status_snapshot_v1',
    'snapshot_utc': datetime.now(timezone.utc).isoformat(),
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'run': a.run, 'directory': str(directory), 'capture_session': last_session,
    'outcome': {k: v for k, v in outcome.items() if k != 'limits'} if outcome else None,
    'last_observed_attempt': last_attempt, 'last_resolved_backward_scan': last_flush,
    'last_backward_complete_event': last_complete,
    'flagged_scans': flagged, 'event_counts': dict(counts), 'files': files,
    'kfd_owners': owners, 'kfd_owner_errors': owner_errors,
    'retained_frames': [str(x) for x in (directory / 'boundaries').glob('*.pt')],
    'limits': ['A live outcome.json keeps its initial counters until terminal update; do not use its zero counters as progress.',
               'Backward completion is not an independently observed completed optimizer step.',
               'This is a timestamped file/process snapshot; it does not establish a terminal result or numerical correctness of every tensor.',
               'KFD-owning dataloader children can inherit descriptors from the same single training workload.']}
data = json.dumps(record, indent=2) + '\n'
if a.output:
    with a.output.open('x') as stream:
        stream.write(data)
print(data, end='')
