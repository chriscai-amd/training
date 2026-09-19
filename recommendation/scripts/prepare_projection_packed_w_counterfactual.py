#!/usr/bin/env python3
"""CPU-export a labeled W low-half sign-bit control for the frozen DX replay.

Only W[:,18::32] BF16 sign bits change. DZ layout, execution controls, all
unselected W bytes and the paired high BF16 halves remain unchanged. The
mathematical DX oracle is still exact numerical zero; onset is not predicted.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
from unittest.mock import patch
import zipfile

SOURCE_BUNDLE_SHA256 = '8d77dd0db4330ed098389b57804cbb55b8f82c4840ae591e04a60491abbac112'
SOURCE_W_SHA256 = 'c54f83af34b940a0d34c5571c0c0f7f29b5565ddd2c1b7ab60995223ad2a67a7'
SOURCE_MODEL_SHA256 = '78e8e18b283e6d74f7d2113a22cf8c298b66b303b307ed330d09ccc94cf4b485'
FROZEN_DRIVER_SHA256 = '49d0e58e32fd3583f500655525b03eaaad8579bb72b842c98f695eab0aaf9614'
FORMAT = 'projection_packed_W_low_half_sign_counterfactual_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path.read_bytes())}


def new_path(path):
    if os.path.lexists(path) or os.path.lexists(str(path) + '.tmp'):
        raise FileExistsError(str(path))


def atomic_save(path, data):
    path = Path(path)
    new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    owned = False
    try:
        with temporary.open('xb') as stream:
            owned = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def rounded_bf16(word):
    if ((word >> 23) & 255) == 255:
        raise ValueError('Packed FP32 word must be finite')
    # Integer nearest-even rounding, independent of native conversion kernels.
    high, low = word >> 16, word & 65535
    return (high + int(low > 32768 or (low == 32768 and high & 1))) & 65535


def transform(raw):
    if len(raw) != 2097152 or digest(raw) != SOURCE_W_SHA256:
        raise ValueError('Requires exact retained 512x2048 BF16 W backing')
    changed = bytearray(raw)
    offsets = [2 * (row * 2048 + k) + 1 for row in range(512) for k in range(18, 2048, 32)]
    for offset in offsets:
        changed[offset] ^= 0x80
    actual_changed = [i for i, (a, b) in enumerate(zip(raw, changed)) if a != b]
    if actual_changed != offsets or any(raw[i] ^ changed[i] != 0x80 for i in offsets):
        raise ValueError('Changed bytes differ from the selected sign-bit mask')
    before_words = struct.unpack('<1048576H', raw)
    after_words = struct.unpack('<1048576H', changed)
    if any((a & 0x7fff) != (b & 0x7fff) or (b & 0x7f80) == 0x7f80
           for a, b in zip(before_words, after_words)):
        raise ValueError('Counterfactual changed magnitude or finiteness')
    selected = {offset // 2 for offset in offsets}
    if any(before_words[i + 1] != after_words[i + 1] for i in selected):
        raise ValueError('Paired high BF16 half changed')
    before_packed = struct.unpack('<524288I', raw)
    after_packed = struct.unpack('<524288I', changed)
    # Check all packed words, including the unchanged complement.
    before_rounded = [rounded_bf16(word) for word in before_packed]
    after_rounded = [rounded_bf16(word) for word in after_packed]
    deltas = collections.Counter(after_rounded[i // 2] - before_rounded[i // 2] for i in selected)
    if deltas != {-1: 16334, 1: 16434}:
        raise ValueError('Unexpected exhaustive packed-word rounding predictions')
    selected_bytes = set(offsets)
    complement_before = bytes(v for i, v in enumerate(raw) if i not in selected_bytes)
    complement_after = bytes(v for i, v in enumerate(changed) if i not in selected_bytes)
    if complement_before != complement_after:
        raise ValueError('Unchanged-complement bytes differ')
    report = {'source_W_sha256': digest(raw), 'counterfactual_W_sha256': digest(changed),
              'total_bytes': len(raw), 'changed_bytes': len(offsets), 'changed_bits': len(offsets),
              'changed_byte_offsets_sha256': digest(struct.pack('<' + 'I' * len(offsets), *offsets)),
              'unchanged_complement_bytes': len(complement_before),
              'unchanged_complement_sha256': digest(complement_before),
              'full_unchanged_complement_equal': True, 'all_W_magnitude_bits_unchanged': True,
              'all_W_finite': True, 'all_paired_high_halves_unchanged': True,
              'all_packed_FP32_words_finite': True, 'selected_pair_RNE_bits_delta_counts': dict(deltas)}
    return bytes(changed), before_words, after_words, report


def validate_candidates(model, before, after):
    arms = []
    for arm in model['arms']:
        elements, groups = [], []
        for element in arm['elements']:
            predictions = []
            for candidate in element['restricted_candidates']:
                row, k = candidate['W_row'], candidate['K_even']
                if (row != element['coordinate'][1] + candidate['row_delta']
                        or candidate['row_delta'] not in (-2, -1) or k % 32 != 18):
                    raise ValueError('Original restricted candidate geometry mismatch')
                index = row * 2048 + k
                lo, hi = before[index:index + 2]
                new_lo, new_hi = after[index:index + 2]
                old_bits, new_bits = rounded_bf16(lo | (hi << 16)), rounded_bf16(new_lo | (new_hi << 16))
                if (old_bits != int(element['bits'], 16) or hi != new_hi or (lo ^ new_lo) != 0x8000
                        or abs(new_bits - old_bits) != 1):
                    raise ValueError('Original/counterfactual candidate rounding mismatch')
                predictions.append({'W_row': row, 'row_delta': candidate['row_delta'], 'K_even': k,
                    'original_observed_bits': hex(old_bits), 'counterfactual_packed_RNE_bits': hex(new_bits),
                    'literal_high_half_before_and_after': hex(hi), 'unchanged_high_half': True,
                    'packed_RNE_bits_delta': new_bits - old_bits})
            if not predictions:
                raise ValueError('Every original observed element must retain its candidate predictions')
            elements.append({'coordinate': element['coordinate'], 'predictions': predictions})
        by_coordinate = {tuple(e['coordinate']): e for e in elements}
        for group in arm['groups']:
            predictions = []
            for candidate in group['restricted_common_candidates']:
                values = []
                for column in group['DX_columns']:
                    matches = [p for p in by_coordinate[(group['DX_row'], column)]['predictions']
                               if p['row_delta'] == candidate['row_delta'] and p['K_even'] == candidate['K_even']]
                    if len(matches) != 1:
                        raise ValueError('Original paired/group candidate is absent from element predictions')
                    values.append({'DX_column': column, **matches[0]})
                predictions.append({'candidate': candidate, 'elements': values})
            groups.append({'DX_row': group['DX_row'], 'column_block128': group['column_block128'],
                           'element_count': group['element_count'], 'predictions': predictions})
        if len(elements) != arm['counts']['elements']:
            raise ValueError('Original model element count mismatch')
        arms.append({'source_CPU_certificate': arm['CPU_certificate'], 'elements': elements, 'groups': groups,
                     'counts': {'elements': len(elements), 'candidate_predictions': sum(len(e['predictions']) for e in elements),
                                'groups': len(groups), 'group_candidate_predictions': sum(len(g['predictions']) for g in groups)}})
    if [a['counts']['elements'] for a in arms] != [136, 137, 75]:
        raise ValueError('Requires the three original call181/99/73 model arms')
    return arms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--model-report', type=Path, required=True)
    parser.add_argument('--bundle-output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in vars(args).values()]
    if len(set(paths)) != len(paths):
        parser.error('Inputs and outputs must be distinct')
    new_path(args.bundle_output)
    new_path(args.report)
    source_record, model_record = file_record(args.source_bundle), file_record(args.model_report)
    driver = Path(__file__).with_name('repro_projection_dx_zero.py')
    if (source_record['sha256'] != SOURCE_BUNDLE_SHA256 or model_record['sha256'] != SOURCE_MODEL_SHA256
            or file_record(driver)['sha256'] != FROZEN_DRIVER_SHA256):
        raise ValueError('Source bundle/model or frozen driver identity mismatch')
    with zipfile.ZipFile(args.source_bundle) as archive:
        if archive.read('archive/byteorder') != b'little':
            raise ValueError('Original source must be little endian')
        raw = archive.read('archive/data/0')
    transformed, before, after, byte_checks = transform(raw)
    model = json.loads(args.model_report.read_text())
    candidate_checks = validate_candidates(model, before, after)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('Exporter must start without GPU initialization')
    rng = torch.get_rng_state().clone()
    with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('Exporter is CPU-only')):
        source = torch.load(args.source_bundle, map_location='cpu', weights_only=False)
        if (source['format'] != 'retained_projection_W_and_zero_DZ_layout_v1'
                or set(source) != {'format', 'w', 'w_storage_sha256', 'dz_layout', 'execution_controls', 'blas_preference', 'provenance'}
                or source['w_storage_sha256'] != SOURCE_W_SHA256):
            raise ValueError('Original bundle schema/hash mismatch')
        weight = source['w']
        if (weight.device.type != 'cpu' or weight.dtype != torch.bfloat16 or list(weight.shape) != [512, 2048]
                or list(weight.stride()) != [2048, 1] or weight.storage_offset() != 0
                or weight.untyped_storage().nbytes() != len(raw) or weight.requires_grad
                or bytes(memoryview(weight.view(torch.uint8).numpy()).cast('B')) != raw):
            raise ValueError('Deserialized source does not match raw full W storage')
        intervention = {'format': FORMAT, 'operation': 'W[:,18::32].raw_BF16_bits XOR 0x8000',
                        'changes_only': 'one sign bit in the low BF16 half of each selected packed word',
                        'mathematical_DX_oracle': 'Exact numerical zero with finite W and generated zero DZ',
                        'onset': 'No prediction of failing call, tile, coordinate or failure frequency'}
        derived = {k: copy.deepcopy(v) for k, v in source.items() if k != 'w'}
        derived['w'] = torch.frombuffer(bytearray(transformed), dtype=torch.uint8).clone().view(torch.bfloat16).reshape(weight.shape)
        derived['w_storage_sha256'] = digest(transformed)
        derived['provenance'].update(source_bundle=source_record, source_bundle_provenance=copy.deepcopy(source['provenance']),
            counterfactual_intervention=intervention, exporter=file_record(__file__),
            W='Derived counterfactual W: only the selected BF16 sign bits differ from the retained actual W.',
            scope='Input-only representation test for frozen direct DX replay; not an original retained input or a causal instruction certificate.')
        buffer = io.BytesIO()
        torch.save(derived, buffer)
        serialized = buffer.getvalue()
        loaded = torch.load(io.BytesIO(serialized), map_location='cpu', weights_only=False)
        loaded_raw = bytes(memoryview(loaded['w'].view(torch.uint8).numpy()).cast('B'))
        if loaded_raw != transformed or loaded['w_storage_sha256'] != digest(loaded_raw):
            raise ValueError('Counterfactual serialization changed full W bytes')
        unchanged_fields = [k for k in source if k not in ('w', 'w_storage_sha256', 'provenance')]
        if any(loaded[k] != source[k] for k in unchanged_fields):
            raise ValueError('Counterfactual changed layout/control fields')
        if (loaded['w'].shape != weight.shape or loaded['w'].stride() != weight.stride()
                or loaded['w'].storage_offset() != weight.storage_offset()
                or loaded['w'].untyped_storage().nbytes() != weight.untyped_storage().nbytes()
                or loaded['w'].requires_grad != weight.requires_grad):
            raise ValueError('Counterfactual changed W layout metadata')
        if not torch.equal(rng, torch.get_rng_state()) or torch.cuda.is_initialized():
            raise ValueError('CPU exporter changed RNG or initialized GPU')
    if (file_record(args.source_bundle) != source_record or file_record(args.model_report) != model_record
            or file_record(driver)['sha256'] != FROZEN_DRIVER_SHA256):
        raise ValueError('Inputs or frozen driver changed during export')
    atomic_save(args.bundle_output, serialized)
    report = {'format': FORMAT, 'status': 'CPU_EXPORTED_AND_ROUNDTRIP_VERIFIED', 'exporter': file_record(__file__),
              'source_bundle': source_record, 'source_model': model_record, 'frozen_driver': file_record(driver),
              'counterfactual_bundle': file_record(args.bundle_output), 'intervention': intervention,
              'byte_checks': byte_checks, 'preserved_layout_control_fields': unchanged_fields,
              'all_original_metadata_preserved_except_explicit_W_hash_and_derived_provenance': True,
              'original_candidate_predictions': candidate_checks,
              'CPU_RNG_unchanged': True, 'GPU_initialized': False,
              'limits': ['Candidate predictions describe the original retained locations; later intermittent failure locations need not repeat.',
                         'A newly retained failure must be reanalyzed against current input bytes; original coordinates are not an expected-failure mask.',
                         'No GPU execution, mathematical reference GEMM, emitted-kernel identity or instruction attribution was performed.',
                         'Frozen replay driver will generate the original full zero-DZ allocation; this exporter does not allocate DZ.']}
    atomic_save(args.report, (json.dumps(report, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'status': report['status'], 'bundle': report['counterfactual_bundle'], 'report': file_record(args.report)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
