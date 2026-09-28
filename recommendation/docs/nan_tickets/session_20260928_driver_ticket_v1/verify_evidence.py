#!/usr/bin/env python3
"""Verify the attached evidence bytes and independently count DW differences.

CPU-only. Recorded run receipts are checked as records; this does not rerun HIP
or the omitted large training-input audit.
"""
import hashlib
import json
from pathlib import Path
import struct

if not __debug__:
    raise RuntimeError('Run verification without Python optimization (-O or PYTHONOPTIMIZE).')

ROOT = Path(__file__).resolve().parent
ARMS = ('sink_control_nop', 'sink_candidate_nop', 'sink_control_trap', 'sink_candidate_trap')
EVIDENCE_NAMES = {
    'current_vs_corrected_DW.json', 'native_causal_audit.json',
    'saved_DW_full_output_review.json', 'saved_corrected_DW_complete.json',
    'sentinel_original_vs_corrected_review.json', 'DW_all_1030_differences.json',
    'DW_healthy_reference.bf16.bin', 'DW_historical_corrupt_reference.bf16.bin',
    'current_DW_complete.json', 'current_DW_matrix.json', 'current_DW_independent_review.json',
    'current_DW_archive_review.json', 'inline_prologue_6144_calls_review.json',
}


def main():
    inventory = json.loads((ROOT / 'EVIDENCE_COPIES.json').read_bytes())
    assert inventory['status'] == 'PASS_COMPACT_EVIDENCE_COPIES'
    relative_paths = [row['relative'] for row in inventory['files']]
    assert len(relative_paths) == len(EVIDENCE_NAMES)
    assert set(relative_paths) == {'evidence/' + name for name in EVIDENCE_NAMES}
    for row in inventory['files']:
        path = ROOT / row['relative']
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == row['bytes'] == row['source']['bytes']
        assert hashlib.sha256(raw).hexdigest() == row['sha256'] == row['source']['sha256']
    folder = ROOT / 'evidence'
    healthy = (folder / 'DW_healthy_reference.bf16.bin').read_bytes()
    corrupt = (folder / 'DW_historical_corrupt_reference.bf16.bin').read_bytes()
    assert len(healthy) == len(corrupt) == 262144
    healthy_words, corrupt_words = [struct.unpack('<131072H', raw) for raw in (healthy, corrupt)]
    differences = json.loads((folder / 'DW_all_1030_differences.json').read_bytes())
    indexes = [i for i, pair in enumerate(zip(healthy_words, corrupt_words)) if pair[0] != pair[1]]
    assert indexes == [row['native_word_index'] for row in differences]
    counts = {'nan': 0, 'finite_above20': 0, 'finite_at_most20': 0}
    for index, row in zip(indexes, differences):
        before, after = healthy_words[index], corrupt_words[index]
        assert (row['M'], row['N']) == (index % 256, index // 256)
        assert int(row['baseline_hex'], 16) == before and int(row['historical_hex'], 16) == after
        magnitude = after & 0x7fff
        assert magnitude != 0x7f80
        if magnitude > 0x7f80:
            category = 'nan'
        else:
            value = struct.unpack('<f', struct.pack('<I', after << 16))[0]
            category = 'finite_above20' if abs(value) > 20 else 'finite_at_most20'
        assert row['class_'] == category
        counts[category] += 1
    assert len(indexes) == 1030 and counts == {'nan': 640, 'finite_above20': 384, 'finite_at_most20': 6}
    baseline_sha, corrupt_sha = [hashlib.sha256(raw).hexdigest() for raw in (healthy, corrupt)]
    matrix = json.loads((folder / 'current_DW_matrix.json').read_bytes())
    assert matrix['status'] == 'PASS_SINK_QUARTET_CONTROL_REPRODUCTION_AND_CANDIDATE_PREVENTION'
    assert matrix['issues'] == [] and len(matrix['arms']) == 4
    assert tuple(arm['arm'] for arm in matrix['arms']) == ARMS
    observed = []
    for arm in matrix['arms']:
        assert [call['iteration'] for call in arm['calls']] == [1, 2, 3, 4]
        expected = corrupt_sha if arm['arm'] == 'sink_control_trap' else baseline_sha
        for call in arm['calls']:
            assert call['output']['bytes'] == 262144 and call['output']['sha256'] == expected
            assert call['comparison_to_per_arm_reference']['complete_bytes_match']
            observed.append(dict(arm=arm['arm'], iteration=call['iteration'], expected_sha256=expected))
    saved = json.loads((folder / 'saved_DW_full_output_review.json').read_bytes())
    assert saved['status'] == 'PASS_INDEPENDENT_COMPLETE_ORIGINAL_AND_ARCHIVED_CORRECTED_DW_OUTPUT_BYTE_REAUDIT'
    assert saved['all_16_corrected_outputs_exact_baseline'] and len(saved['outputs']) == 16
    assert [(call['arm'], call['iteration']) for call in saved['outputs']] == [
        (arm, iteration) for arm in ARMS for iteration in range(1, 5)]
    for call in saved['outputs']:
        assert call['corrected_archive_mapping']['arm'] == call['arm']
        assert call['corrected_archive_mapping']['kind'] == 'output'
        assert call['corrected_archive_mapping']['relative'] == f"results_{call['arm']}/call_{call['iteration']:02d}_output.bf16.bin"
        assert call['corrected_archive_mapping']['raw']['bytes'] == 262144
        assert call['corrected_archive_mapping']['raw']['sha256'] == baseline_sha
        assert call['corrected_scan']['complete_words_scanned'] == 131072
        assert call['corrected_scan']['differences_from_baseline'] == 0
    inline = json.loads((folder / 'inline_prologue_6144_calls_review.json').read_bytes())
    assert inline['status'] == 'PASS_INDEPENDENT_COMPLETED_FILE_AUDIT' and inline['suite'] == 'full'
    assert inline['calls'] == 6144
    assert {name: (row['calls'], row['copied_lane0']) for name, row in inline['matrix'].items()} == {
        'none': (2048, 0), 'original': (2048, 1536), 'corrected': (2048, 0)}
    print(json.dumps(dict(status='PASS_ATTACHED_EVIDENCE_BYTES_AND_DW_DIFFERENCES',
        authenticated_evidence_files=len(inventory['files']), attached_reference_words_compared=131072,
        differing_words=len(indexes), difference_classes=counts,
        current_recorded_output_mappings_checked=len(observed), corrected_recorded_output_mappings_checked=16,
        GPU_actions=False, large_training_inputs_read=False)))


if __name__ == '__main__':
    main()
