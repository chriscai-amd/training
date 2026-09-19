#!/usr/bin/env python3
"""Prepare/verify a one-leaf private hipBLASLt catalog using CPU file operations.

Run inside the recorded container; original library paths are container paths.
This module never imports Torch or loads a native library. Preparation creates
new files exclusively. Verification is read-only and checks every library entry.
"""
from pathlib import Path
import collections
import hashlib
import json
import os
import struct
import sys
import zlib

ROOT = Path(__file__).absolute().parent
INSTALLED = Path('/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250')
PRIVATE = ROOT / 'gfx1250'
STEM = 'TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250'
TARGET = STEM + '.dat.zlib'
MASTER = 'TensileLibrary_lazy_gfx1250.dat.zlib'
ORIGINAL_SHA = '8ac6594e0e67b351b853ee679d815acbf79f3fb138c789ce7d225ce9f67e5d59'
ORIGINAL_PACKED_SHA = '3a8322d4dc00eb727def7ae98a39e14b45df8e719a10b0af1ff72948307e8684'
MASTER_SHA = '908c9fb056975e44f27aa675238e7a5751943f50f1bb0312c05f33337f5fa402'
CHANGE_PATH = ('library', 'rows', 0, 'library', 'table', 32, 'index')


def guard(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {'torch', 'triton', 'ctypes', 'subprocess'}:
        raise RuntimeError(('forbidden import', args[0]))
    if event.startswith(('ctypes.', 'subprocess.')) or event in {'os.system', 'os.fork', 'os.posix_spawn'}:
        raise RuntimeError(('forbidden event', event))


sys.addaudithook(guard)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': sha(data)}


def write_new(name, data):
    with (ROOT / name).open('xb') as f:
        f.write(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()


def inflate(data):
    decoder = zlib.decompressobj()
    result = decoder.decompress(data) + decoder.flush()
    assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    return result


def unpack(data):
    """Strict subset of MessagePack used by these catalogs, recording value spans."""
    spans = {}

    def parse(pos, path):
        start = pos
        assert pos < len(data)
        code = data[pos]
        pos += 1
        if code < 128:
            value = code
        elif code >= 224:
            value = code - 256
        elif code == 192:
            value = None
        elif code in (194, 195):
            value = code == 195
        elif code in (202, 203, 204, 205, 206, 207, 208, 209, 210, 211):
            fmt = {202: '>f', 203: '>d', 204: '>B', 205: '>H', 206: '>I', 207: '>Q',
                   208: '>b', 209: '>h', 210: '>i', 211: '>q'}[code]
            value = struct.unpack_from(fmt, data, pos)[0]
            pos += struct.calcsize(fmt)
        else:
            if 128 <= code <= 143:
                kind, count = 'map', code & 15
            elif 144 <= code <= 159:
                kind, count = 'array', code & 15
            elif 160 <= code <= 191:
                kind, count = 'str', code & 31
            else:
                specs = {196: ('bin', '>B'), 197: ('bin', '>H'), 198: ('bin', '>I'),
                         217: ('str', '>B'), 218: ('str', '>H'), 219: ('str', '>I'),
                         220: ('array', '>H'), 221: ('array', '>I'),
                         222: ('map', '>H'), 223: ('map', '>I')}
                assert code in specs, hex(code)
                kind, fmt = specs[code]
                count = struct.unpack_from(fmt, data, pos)[0]
                pos += struct.calcsize(fmt)
            if kind in ('str', 'bin'):
                value = data[pos:pos + count]
                assert len(value) == count
                if kind == 'str':
                    value = value.decode('utf-8')
                pos += count
            elif kind == 'array':
                value = []
                for i in range(count):
                    item, pos = parse(pos, path + (i,))
                    value.append(item)
            else:
                value = {}
                for i in range(count):
                    key, pos = parse(pos, path + ('__map_key__', i))
                    assert isinstance(key, str) and key not in value
                    item, pos = parse(pos, path + (key,))
                    value[key] = item
        assert pos <= len(data)
        assert path not in spans
        spans[path] = (start, pos)
        return value, pos

    result, end = parse(0, ())
    assert end == len(data)
    return result, spans


def pointer(path):
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in path)


def diff(before, after, path=()):
    assert type(before) is type(after), path
    if isinstance(before, dict):
        assert list(before) == list(after), path
        return [d for key in before for d in diff(before[key], after[key], path + (key,))]
    if isinstance(before, list):
        assert len(before) == len(after), path
        return [d for i, (a, b) in enumerate(zip(before, after)) for d in diff(a, b, path + (i,))]
    return [] if before == after else [{'path': pointer(path), 'before': before, 'after': after}]


def string_matches(value, expected, path=()):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in string_matches(v, expected, path + (k,))]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in string_matches(v, expected, path + (i,))]
    return [pointer(path)] if value == expected else []


def verify(expected_manifest_sha=None):
    manifest_data = (ROOT / 'manifest.json').read_bytes()
    if expected_manifest_sha is not None:
        assert sha(manifest_data) == expected_manifest_sha
    manifest = json.loads(manifest_data)
    assert manifest['schema_version'] == 1
    assert manifest['private_library_path'] == str(PRIVATE)
    assert manifest['installed_library_path'] == str(INSTALLED)
    assert manifest['target_catalog_filename'] == TARGET
    assert identity(Path(__file__).absolute()) == manifest['preparation_source']
    inventory = manifest['inventory']
    assert [x.name for x in sorted(INSTALLED.iterdir())] == [r['name'] for r in inventory]
    assert [x.name for x in sorted(PRIVATE.iterdir())] == [r['name'] for r in inventory]
    for row in inventory:
        original, private = INSTALLED / row['name'], PRIVATE / row['name']
        assert original.is_file() and not original.is_symlink()
        assert identity(original) == row['original'], row['name']
        assert identity(private) == row['private'], row['name']
        if row['name'] == TARGET:
            assert row['kind'] == 'derived_regular_file' and not private.is_symlink()
        else:
            assert row['kind'] == 'original_symlink' and private.is_symlink()
            assert os.readlink(private) == str(original) == row['symlink_target']
            a, b = original.stat(), private.stat()
            assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    before = inflate((INSTALLED / TARGET).read_bytes())
    after = inflate((PRIVATE / TARGET).read_bytes())
    original, spans = unpack(before)
    derived, derived_spans = unpack(after)
    expected_diff = [{'path': pointer(CHANGE_PATH), 'before': 103, 'after': 102}]
    assert diff(original, derived) == expected_diff == manifest['catalog']['structured_diff']
    assert spans == derived_spans
    assert len(before) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert changed == [spans[CHANGE_PATH][0]] == manifest['catalog']['changed_decompressed_byte_offsets']
    assert before[changed[0]] == 103 and after[changed[0]] == 102
    assert original['solutions'] == derived['solutions']
    table = derived['library']['rows'][0]['library']['table']
    assert len(table) == 35 and all(r['index'] == 102 for r in table)
    return {'status': 'PASS_CPU_FILE_VALIDATION', 'manifest': identity(ROOT / 'manifest.json'),
            'private_library_path': str(PRIVATE), 'entries': len(inventory),
            'original_symlinks': len(inventory) - 1, 'derived_regular_files': 1,
            'derived_catalog': identity(PRIVATE / TARGET), 'structured_diff': expected_diff}


def prepare():
    assert str(ROOT) == '/data/mlperf_dlrm_v4/root_cause_20260919/dw_solution102_private_catalog_v1'
    assert not PRIVATE.exists() and not (ROOT / 'manifest.json').exists()
    sources = sorted(INSTALLED.iterdir())
    assert len(sources) == 402 and all(p.is_file() and not p.is_symlink() for p in sources)
    source_pins = {p.name: identity(p) for p in sources}
    assert source_pins[TARGET]['sha256'] == ORIGINAL_SHA
    assert source_pins[MASTER]['sha256'] == MASTER_SHA
    compressed = (INSTALLED / TARGET).read_bytes()
    packed = inflate(compressed)
    assert len(packed) == 11797 and sha(packed) == ORIGINAL_PACKED_SHA
    original, spans = unpack(packed)
    assert list(original) == ['solutions', 'library']
    assert [(s['index'], s['libraryLogicIndex']) for s in original['solutions']] == [(102, 0), (103, 1)]
    table = original['library']['rows'][0]['library']['table']
    assert len(table) == 35
    assert dict(collections.Counter(r['index'] for r in table)) == {102: 34, 103: 1}
    assert table[32] == {'key': [256, 256, 1, 512], 'index': 103}
    start, end = spans[CHANGE_PATH]
    assert end == start + 1 and packed[start:end] == bytes([103])
    modified = packed[:start] + bytes([102]) + packed[end:]
    derived, new_spans = unpack(modified)
    expected_diff = [{'path': pointer(CHANGE_PATH), 'before': 103, 'after': 102}]
    assert diff(original, derived) == expected_diff and spans == new_spans
    assert original['solutions'] == derived['solutions']
    sol_start, sol_end = spans[('solutions',)]
    assert packed[sol_start:sol_end] == modified[sol_start:sol_end]
    output = zlib.compress(modified, level=9)
    assert inflate(output) == modified
    master_data = (INSTALLED / MASTER).read_bytes()
    master_decoded, _ = unpack(inflate(master_data))
    master_refs = string_matches(master_decoded, STEM)
    assert master_refs
    PRIVATE.mkdir()
    inventory = []
    for source in sources:
        dest = PRIVATE / source.name
        row = {'name': source.name, 'original': source_pins[source.name]}
        if source.name == TARGET:
            with dest.open('xb') as f:
                f.write(output)
            row['kind'] = 'derived_regular_file'
        else:
            dest.symlink_to(source)
            row.update(kind='original_symlink', symlink_target=str(source))
        row['private'] = identity(dest)
        inventory.append(row)
    write_new('catalog.source.dat.zlib', compressed)
    write_new('catalog.source.decoded.json', json_bytes(original))
    write_new('catalog.derived.decoded.json', json_bytes(derived))
    write_new('master.source.dat.zlib', master_data)
    write_new('master.source.decoded.json', json_bytes(master_decoded))
    for source in sources:
        assert identity(source) == source_pins[source.name]
    manifest = {
        'schema_version': 1,
        'status': 'PREPARED_AND_CPU_FILE_VALIDATED',
        'scope': 'Alternative DW kernel-selection catalog; CPU file operations only, no native loading or GPU execution.',
        'private_library_path': str(PRIVATE),
        'installed_library_path': str(INSTALLED),
        'target_catalog_filename': TARGET,
        'preparation_source': identity(Path(__file__).absolute()),
        'recommended_environment': {'HIPBLASLT_TENSILE_LIBPATH': str(PRIVATE)},
        'catalog': {
            'original': source_pins[TARGET], 'derived': identity(PRIVATE / TARGET),
            'original_decompressed': {'bytes': len(packed), 'sha256': sha(packed)},
            'derived_decompressed': {'bytes': len(modified), 'sha256': sha(modified)},
            'structured_diff': expected_diff,
            'changed_decompressed_byte_offsets': [start],
            'encoding': 'The selected positive-fixint leaf is re-encoded as 0x66 instead of 0x67. Every other decompressed MessagePack byte is identical. The full valid stream is recompressed with zlib level 9.',
            'matching_type': 'GridBased', 'matching_entries': 35,
            'table_indices_before': {'102': 34, '103': 1}, 'table_indices_after': {'102': 35},
            'selected_row_key_unchanged': [256, 256, 1, 512],
            'solutions_preserved': [{'index': s['index'], 'libraryLogicIndex': s['libraryLogicIndex'], 'kernelName': s['kernelName']} for s in original['solutions']],
            'solutions_messagepack_span': [sol_start, sol_end],
            'solutions_messagepack_sha256_unchanged': sha(packed[sol_start:sol_end]),
            'decoded_original': identity(ROOT / 'catalog.source.decoded.json'),
            'decoded_derived': identity(ROOT / 'catalog.derived.decoded.json')},
        'master': {'original': source_pins[MASTER], 'private': identity(PRIVATE / MASTER),
                   'lazy_reference_value': STEM, 'lazy_reference_paths': master_refs,
                   'decoded': identity(ROOT / 'master.source.decoded.json')},
        'native_dw_code_object': {'original': source_pins[STEM + '.co'], 'private': identity(PRIVATE / (STEM + '.co'))},
        'preservation': {'installed_file_count': len(sources), 'installed_total_bytes': sum(p['bytes'] for p in source_pins.values()),
                         'all_installed_hashes_rechecked_unchanged': True, 'private_original_symlink_count': len(sources) - 1,
                         'private_derived_regular_file_count': 1,
                         'all_native_objects_and_other_catalogs': 'Absolute symlinks to corresponding installed originals; exact names, bytes, SHA256 and targets recorded below.'},
        'limitations': [
            'Redirects every entry of this one matching table to solution 102. It does not bypass the unchanged solution compatibility predicates or alter other library branches.',
            'Successful static preparation does not prove runtime selection, numerical correctness for arbitrary sizes, or elimination of an intermittent fault.',
            'Both solution definitions and their native objects remain available. Explicit index selection can still request solution 103; clear tuning overrides for the catalog-only control.',
            'This is an alternative-kernel control, not an isolated VGPR-count intervention.',
            'Set the private path before the first hipBLASLt use in a fresh process and verify the launched full kernel name.',
            'The frozen replay restores its captured environment. Root must use a separately labeled and pinned derivative context whose only change is HIPBLASLT_TENSILE_LIBPATH.',
            'Private symlink targets are container-absolute paths and may be unresolved from the host filesystem.'
        ],
        'inventory': inventory,
    }
    write_new('manifest.json', json_bytes(manifest))
    report = verify()
    write_new('validation.json', json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        prepare()
    elif len(sys.argv) == 3 and sys.argv[1] == '--verify':
        print(json.dumps(verify(sys.argv[2]), indent=2))
    else:
        raise SystemExit('Usage: prepare_private_catalog.py --prepare | --verify EXPECTED_MANIFEST_SHA256')
