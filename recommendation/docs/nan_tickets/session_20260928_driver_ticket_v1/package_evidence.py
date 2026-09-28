#!/usr/bin/env python3
"""Copy authenticated compact evidence into the driver ticket attachment."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
B = ROOT.parent


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(row):
    assert pin(row['path']) == row, row['path']
    return json.loads(Path(row['path']).read_bytes())


def main():
    out = ROOT / 'evidence'
    out.mkdir()
    mapping = []

    def copy(name, row):
        assert pin(row['path']) == row, row['path']
        target = out / name
        with target.open('xb') as stream:
            stream.write(Path(row['path']).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        actual = pin(target)
        assert (actual['bytes'], actual['sha256']) == (row['bytes'], row['sha256'])
        mapping.append(dict(relative='evidence/' + name, source=row,
                            bytes=actual['bytes'], sha256=actual['sha256']))

    comparison = dict(path=str(B / 'session_20260928_original_DW_result_peer_v1/CORRECTED_COMPARISON.json'),
                      bytes=31923, sha256='7764e9a84abaa5a89a6fdbf92574cbe78d99f87acb56092fb4fa5fe3fda5613e')
    comp = read(comparison)
    copy('current_vs_corrected_DW.json', comparison)
    for name, key in [('native_causal_audit.json', 'native_baseline_and_handler_correction_audit'),
                      ('saved_DW_full_output_review.json', 'saved_full_corrected_output_review'),
                      ('saved_corrected_DW_complete.json', 'saved_corrected_complete_consumer'),
                      ('sentinel_original_vs_corrected_review.json', 'firmware_attestation')]:
        copy(name, comp[key])
    full = read(comp['saved_full_corrected_output_review'])
    for name, key in [('DW_all_1030_differences.json', 'all_differences'),
                      ('DW_healthy_reference.bf16.bin', 'baseline'),
                      ('DW_historical_corrupt_reference.bf16.bin', 'historical')]:
        copy(name, full[key])
    rows = [
        ('current_DW_complete.json', 'session_20260927_original_DW_execution_v1/complete_v1/complete_consumer.json',
         49423, 'fab72bcc4b87e52d08da5f22fdb540c0be842c37283520d005f65dff45ccf093'),
        ('current_DW_matrix.json', 'session_20260927_original_DW_execution_v1/complete_v1/quartet_matrix.json',
         35872, '9021be10fdcfbc7ea8ea3654a96d3260fcb1fcca70d0e379fca6ca1499b3454d'),
        ('current_DW_independent_review.json', 'session_20260928_original_DW_result_peer_v1/COMPLETE_RESULT_REVIEW.json',
         40552, '6a855bc631d08b61be0f4b1534425bea5c28771c6e7d789c0eeeca1839bb4fea'),
        ('current_DW_archive_review.json', 'session_20260928_original_DW_result_peer_v1/ARCHIVE_RESULT_REVIEW.json',
         20732, 'fe62963c0c48ee87c4c0961f930e97d17276681a3d02f3278113412df826b64f'),
    ]
    for name, path, size, digest in rows:
        copy(name, dict(path=str(B / path), bytes=size, sha256=digest))
    causal = read(comp['native_baseline_and_handler_correction_audit'])
    inline_path = '/home/chcai/training/recommendation/docs/mi450_a0/evidence/current_20260919/CWSR_inline_correction_20260921_evening.json'
    inline = next(row for row in causal['selected_evidence'] if row['path'] == inline_path)
    copy('inline_prologue_6144_calls_review.json', inline)
    manifest = dict(status='PASS_COMPACT_EVIDENCE_COPIES', files=mapping,
                    GPU_actions=False, training_operands_included=False,
                    notes=['Original audit bytes are preserved, including their local provenance paths.',
                           'Large training inputs and module binaries remain local evidence dependencies.',
                           'The standalone register reproducer does not need those training inputs.'])
    with (ROOT / 'EVIDENCE_COPIES.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(files=len(mapping), bytes=sum(row['bytes'] for row in mapping))))


if __name__ == '__main__':
    main()
