#!/usr/bin/env python3
"""Standalone serial CWSR sentinel host; --check-only never loads HIP."""
from pathlib import Path
import argparse
import ctypes as C
import datetime
import hashlib
import json
import os
import struct
import sys
import traceback

MODES = (0x00, 0x41, 0x45, 0x82, 0x05, 0x81, 0x85)
REQUIRED_ENV = {
    'AMDGCN_USE_BUFFER_OPS': '0', 'TRITON_FULL_AUTOTUNE': '0',
    'TRITON_ALLOW_PIPELINING': '0', 'HIPBLASLT_WORKSPACE_SIZE': '1',
    'HSTU_BWD_MAX_VGPR': '256', 'WEIGHTED_LN_BWD_BLOCK_N': '1',
    'HIPBLASLT_TENSILE_LIBPATH': '/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250',
}
LIBRARY_PINS = {
    'hip': dict(path='/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7',
        bytes=28048929, sha256='df1330b0dcd594904beebf991b98dd8121be38a3ed975c8ce74986df8e330394'),
    'hsa': dict(path='/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1',
        bytes=4887073, sha256='4abe49f88ef391128610ab0f6e699d44cadd001df5937a00adb9c5655c9cbf86'),
}
PACKET_LAYOUT = dict(mode_before=4, mode_after=5, status_before=6, status_after=7,
    case_mode=2, case_trap=3, intended_return_pc=[26, 27], returned_ttmp=[28, 29],
    exec_before=30, exec_after=31,
    tags=[dict(vgpr=vgpr, lane=lane, before_word=8 + i, after_word=17 + i,
        expected_bits=(0x11000000 * (1 + i // 3)) + lane)
        for i, (vgpr, lane) in enumerate((v, lane) for v in (1, 257, 513) for lane in (0, 1, 16))],
    constant_words=[dict(word=0, role='magic', expected_bits=0x53545250),
        dict(word=1, role='version', expected_bits=1),
        dict(word=32, role='completion', expected_bits=0x434f4d50)]
        + [dict(word=i, role='reserved', expected_bits=0xa5a5a5a5) for i in range(33, 40)])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(path):
    path = Path(path)
    before = path.stat()
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed while hashing: {path}')
    return dict(path=str(path), bytes=before.st_size, sha256=digest)


def pin(row):
    actual = record(row['path'])
    if actual != row:
        raise ValueError(f'File pin mismatch: {row["path"]}')
    return Path(row['path'])


def write_bytes(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return record(path)


def write_json(path, value):
    return write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def mode_byte(low):
    return ((low >> 6) & 3) | ((low & 3) << 2) | (((low >> 2) & 3) << 4) | (((low >> 4) & 3) << 6)


def check_configuration(path, digest):
    identity = record(path)
    if identity['sha256'] != digest:
        raise ValueError('Configuration SHA256 mismatch')
    cfg = json.loads(Path(path).read_text())
    if cfg['schema'] != 'MI450-CWSR-SENTINEL-HOST-v1':
        raise ValueError('Unsupported configuration schema')
    if cfg['libraries'] != LIBRARY_PINS:
        raise ValueError('Unexpected HIP/HSA runtime pins')
    for item in cfg['libraries'].values():
        pin(item)
    code = pin(cfg['module']).read_bytes()
    if not code.startswith(b'\x7fELF'):
        raise ValueError('Module is not ELF')
    if hashlib.sha256(code).hexdigest() != cfg['module']['sha256']:
        raise ValueError('Module changed after pin check')
    expected_cases = [(m, trap) for m in MODES for trap in (False, True)]
    if [(r['mode_low_byte'], r['trap']) for r in cfg['cases']] != expected_cases:
        raise ValueError('Cases must be the exact ordered 7 modes x no-trap/trap sweep')
    if any(r['symbol'] != f'sentinel_{r["mode_low_byte"]:02x}_{int(r["trap"])}' for r in cfg['cases']):
        raise ValueError('Unexpected kernel symbol')
    if cfg['grid'] != [1, 1, 1] or cfg['block'] != [32, 1, 1] or cfg['shared_bytes'] != 0:
        raise ValueError('Unexpected launch dimensions')
    if cfg['kernarg_bytes'] != 8 or C.sizeof(C.c_void_p) != 8 or sys.byteorder != 'little':
        raise ValueError('Only the reviewed 64-bit little-endian output-pointer ABI is supported')
    if not isinstance(cfg['device'], int) or not 0 <= cfg['device'] < 64:
        raise ValueError('Invalid device ordinal')
    if cfg['packet_words'] != 40:
        raise ValueError('Expected reviewed 40-word packet')
    if cfg['prefill_byte'] != 0xa5:
        raise ValueError('Expected 0xa5 output prefill')
    layout = cfg['layout']
    if layout != PACKET_LAYOUT:
        raise ValueError('Packet layout differs from reviewed interface')
    indices = [layout[k] for k in ('mode_before', 'mode_after', 'status_before', 'status_after',
        'case_mode', 'case_trap', 'exec_before', 'exec_after')]
    indices += layout['intended_return_pc'] + layout['returned_ttmp']
    constants = layout['constant_words']
    if not constants or not any(r['role'] == 'completion' for r in constants):
        raise ValueError('A completion marker is required')
    if any(r['role'] not in ('magic', 'version', 'completion', 'reserved') or not 0 <= r['expected_bits'] <= 0xffffffff for r in constants):
        raise ValueError('Invalid packet constants')
    if any(r['role'] == 'completion' and r['expected_bits'] == 0xa5a5a5a5 for r in constants):
        raise ValueError('Completion marker cannot equal prefill')
    indices.extend(r['word'] for r in constants)
    tags = layout['tags']
    if [(r['vgpr'], r['lane']) for r in tags] != [(v, lane) for v in (1, 257, 513) for lane in (0, 1, 16)]:
        raise ValueError('Expected v1/v257/v513 tags at lanes0/1/16')
    for tag in tags:
        indices.extend([tag['before_word'], tag['after_word']])
        if not 0 <= tag['expected_bits'] <= 0xffffffff:
            raise ValueError('Invalid tag bits')
    if len(set(indices)) != len(indices) or any(not isinstance(i, int) or not 0 <= i < cfg['packet_words'] for i in indices):
        raise ValueError('Packet field overlap or out-of-bounds field')
    if set(indices) != set(range(cfg['packet_words'])):
        raise ValueError('Every packet word must have a declared field or constant')
    if len({r['expected_bits'] for r in tags}) != len(tags):
        raise ValueError('Sentinel tags must be distinct')
    return cfg, identity, code


def classify(raw, cfg, case):
    words = struct.unpack('<' + 'I' * cfg['packet_words'], raw)
    layout = cfg['layout']
    tags = layout['tags']
    before = [words[t['before_word']] for t in tags]
    after = [words[t['after_word']] for t in tags]
    expected = [t['expected_bits'] for t in tags]
    copied = list(expected)
    source = 1 + 256 * (case['mode_low_byte'] & 3)
    destination = 1 + 256 * ((case['mode_low_byte'] >> 6) & 3)
    source_index = next(i for i, t in enumerate(tags) if (t['vgpr'], t['lane']) == (source, 0))
    destination_index = next(i for i, t in enumerate(tags) if (t['vgpr'], t['lane']) == (destination, 0))
    copied[destination_index] = expected[source_index]
    mode_pre = (words[layout['mode_before']] >> 12) & 255
    mode_post = (words[layout['mode_after']] >> 12) & 255
    constant_checks = [dict(word=r['word'], role=r['role'], expected=f'0x{r["expected_bits"]:08x}',
        observed=f'0x{words[r["word"]]:08x}', equal=words[r['word']] == r['expected_bits']) for r in layout['constant_words']]
    case_fields_match = words[layout['case_mode']] == case['mode_low_byte'] and words[layout['case_trap']] == int(case['trap'])
    complete = all(r['equal'] for r in constant_checks) and case_fields_match
    valid = before == expected and mode_pre == mode_byte(case['mode_low_byte'])
    if not complete:
        label = 'INVALID_PACKET_STRUCTURE'
    elif not valid:
        label = 'INVALID_INITIAL_STATE'
    elif after == expected:
        label = 'UNCHANGED'
    elif after == copied:
        label = 'PREDICTED_LANE0_CROSS_BANK_COPY'
    else:
        label = 'OTHER_TAG_CHANGE'
    return dict(classification=label, raw_words=[f'0x{x:08x}' for x in words],
        packet_structure_valid=complete, constant_word_checks=constant_checks,
        packet_case_fields_match=case_fields_match,
        intended_return_PC=f'0x{(words[27] << 32) | words[26]:016x}',
        returned_TTMP0=f'0x{words[28]:08x}', returned_TTMP1=f'0x{words[29]:08x}',
        EXEC_before=f'0x{words[30]:08x}', EXEC_after=f'0x{words[31]:08x}',
        EXEC_preserved=words[30] == words[31], EXEC_both_full_wave32=words[30] == words[31] == 0xffffffff,
        initialization_valid=valid, before_tags_equal_expected=before == expected,
        requested_MODE_19_12=f'0x{mode_byte(case["mode_low_byte"]):02x}',
        observed_MODE_19_12_before=f'0x{mode_pre:02x}', observed_MODE_19_12_after=f'0x{mode_post:02x}',
        MODE_bank_field_restored=mode_pre == mode_post,
        trap_enabled_before=bool(words[layout['status_before']] & 64),
        trap_enabled_after=bool(words[layout['status_after']] & 64),
        changed_tags=[dict(vgpr=t['vgpr'], lane=t['lane'], before=f'0x{a:08x}', after=f'0x{b:08x}')
            for t, a, b in zip(tags, before, after) if a != b],
        explicit_shader_trap_requested=case['trap'],
        interpretation='Tag classification is independent of HIP transport success. No-trap cases can still receive asynchronous traps.')


class HipFailure(RuntimeError):
    pass


class Hip:
    def __init__(self, library_path, journal):
        self.journal = journal
        self.events = []
        self.library = C.CDLL(library_path)
        v, i, u, z, p = C.c_void_p, C.c_int, C.c_uint, C.c_size_t, C.POINTER
        declarations = {
            'hipInit': [u], 'hipSetDevice': [i], 'hipGetDeviceCount': [p(i)],
            'hipRuntimeGetVersion': [p(i)], 'hipDriverGetVersion': [p(i)],
            'hipDeviceGetName': [v, i, i], 'hipDeviceSynchronize': [],
            'hipMalloc': [p(v), z], 'hipFree': [v], 'hipMemcpy': [v, v, z, i],
            'hipModuleLoadData': [p(v), v], 'hipModuleUnload': [v],
            'hipModuleGetFunction': [p(v), v, C.c_char_p],
            'hipModuleLaunchKernel': [v, u, u, u, u, u, u, u, v, p(v), p(v)],
        }
        for name, args in declarations.items():
            function = getattr(self.library, name)
            function.argtypes = args
            function.restype = i
        for name in ('hipGetErrorName', 'hipGetErrorString'):
            function = getattr(self.library, name)
            function.argtypes = [i]
            function.restype = C.c_char_p

    def call(self, name, *args, context, allow_error=False):
        started = now()
        self.journal.write(json.dumps(dict(phase='begin', index=len(self.events), function=name,
            context=context, started_utc=started), allow_nan=False) + '\n')
        self.journal.flush()
        os.fsync(self.journal.fileno())
        code = int(getattr(self.library, name)(*args))
        event = dict(phase='end', index=len(self.events), function=name, context=context, started_utc=started,
            finished_utc=now(), result_code=code)
        if code:
            event['error_name'] = (self.library.hipGetErrorName(code) or b'').decode(errors='replace')
            event['error_string'] = (self.library.hipGetErrorString(code) or b'').decode(errors='replace')
        self.events.append(event)
        self.journal.write(json.dumps(event, allow_nan=False) + '\n')
        self.journal.flush()
        os.fsync(self.journal.fileno())
        if code and not allow_error:
            raise HipFailure(f'{name}: {code} {event.get("error_name", "")}')
        return code


def capture_maps(output, label):
    raw = Path('/proc/self/maps').read_bytes()
    saved = write_bytes(output / f'{label}.maps.txt', raw)
    paths = set()
    for line in raw.decode().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            paths.add(fields[5])
    matches = {}
    for kind, expected in LIBRARY_PINS.items():
        prefix = 'libamdhip64.so' if kind == 'hip' else 'libhsa-runtime64.so'
        found = sorted(p for p in paths if Path(p).name.startswith(prefix))
        if len(found) != 1 or Path(found[0]).resolve() != Path(expected['path']).resolve():
            raise ValueError(f'Unexpected mapped {kind} library: {found}')
        pin(expected)
        matches[kind] = dict(mapped_path=found[0], pinned_file=expected)
    return dict(maps=saved, libraries=matches)


def run(cfg, identity, code, output):
    output.mkdir(parents=False, exist_ok=False)
    summary = dict(schema='MI450-CWSR-SENTINEL-RESULT-v1', started_utc=now(), configuration=identity,
        host_source=record(__file__), cases=[], runtime_initialized=False, transport_status='NOT_STARTED',
        required_environment={k: os.environ.get(k) for k in REQUIRED_ENV}, pid=os.getpid())
    write_json(output / 'start.json', summary)
    module, device_pointer = C.c_void_p(), C.c_void_p()
    hip = None
    exit_code = 1
    with (output / 'hip_calls.jsonl').open('x') as journal:
        try:
            if summary['required_environment'] != REQUIRED_ENV:
                raise ValueError('Required environment mismatch')
            prohibited = [k for k in os.environ if k.startswith('LINEAR_DW_') or k == 'LD_PRELOAD']
            if prohibited:
                raise ValueError(f'Prohibited launch environment keys: {prohibited}')
            hip = Hip(LIBRARY_PINS['hip']['path'], journal)
            hip.call('hipInit', 0, context='initialization')
            summary['runtime_initialized'] = True
            summary['maps_after_init'] = capture_maps(output, 'after_init')
            count, runtime, driver = C.c_int(), C.c_int(), C.c_int()
            hip.call('hipGetDeviceCount', C.byref(count), context='initialization')
            if not cfg['device'] < count.value:
                raise ValueError('Requested device is absent')
            hip.call('hipSetDevice', cfg['device'], context='initialization')
            hip.call('hipRuntimeGetVersion', C.byref(runtime), context='initialization')
            hip.call('hipDriverGetVersion', C.byref(driver), context='initialization')
            name = C.create_string_buffer(512)
            hip.call('hipDeviceGetName', name, len(name), cfg['device'], context='initialization')
            summary['device'] = dict(ordinal=cfg['device'], count=count.value, name=name.value.decode(errors='replace'),
                runtime_version=runtime.value, driver_version=driver.value)
            code_buffer = C.create_string_buffer(code, len(code))
            hip.call('hipModuleLoadData', C.byref(module), code_buffer, context='module_load')
            packet_bytes = 4 * cfg['packet_words']
            hip.call('hipMalloc', C.byref(device_pointer), packet_bytes, context='output_allocation')
            if not module.value or not device_pointer.value:
                raise ValueError('HIP returned a null handle after success')
            summary['module_handle'] = hex(module.value)
            summary['output_device_pointer'] = hex(device_pointer.value)
            for ordinal, case in enumerate(cfg['cases']):
                case_dir = output / f'case_{ordinal:02d}_{case["mode_low_byte"]:02x}_{int(case["trap"])}'
                case_dir.mkdir()
                row = dict(ordinal=ordinal, case=case, started_utc=now(), transport_status='STARTED',
                    call_index_begin=len(hip.events), output_device_pointer=hex(device_pointer.value))
                write_json(case_dir / 'start.json', row)
                try:
                    function = C.c_void_p()
                    hip.call('hipModuleGetFunction', C.byref(function), module, case['symbol'].encode(), context=ordinal)
                    if not function.value:
                        raise ValueError('HIP returned a null kernel handle after success')
                    row['function_handle'] = hex(function.value)
                    expected_prefill = bytes([cfg['prefill_byte']]) * packet_bytes
                    row['prefill_expected'] = write_bytes(case_dir / 'prefill_expected.bin', expected_prefill)
                    prefill = C.create_string_buffer(expected_prefill, packet_bytes)
                    hip.call('hipMemcpy', device_pointer, prefill, packet_bytes, 1, context=ordinal)
                    readback = C.create_string_buffer(packet_bytes)
                    hip.call('hipMemcpy', readback, device_pointer, packet_bytes, 2, context=ordinal)
                    row['prefill_actual'] = write_bytes(case_dir / 'prefill_actual.bin', readback.raw)
                    if readback.raw != expected_prefill:
                        raise ValueError('Device prefill readback mismatch')
                    # kernelParams is one pointer to the live eight-byte device-pointer
                    # object. Both it and the array remain live through synchronization.
                    kernel_argument = C.c_void_p(device_pointer.value)
                    kernel_params = (C.c_void_p * 1)(C.cast(C.byref(kernel_argument), C.c_void_p))
                    row['kernarg_output_pointer_bytes'] = bytes(kernel_argument).hex()
                    hip.call('hipModuleLaunchKernel', function, *cfg['grid'], *cfg['block'], cfg['shared_bytes'],
                        None, kernel_params, None, context=ordinal)
                    hip.call('hipDeviceSynchronize', context=ordinal)
                    hip.call('hipMemcpy', readback, device_pointer, packet_bytes, 2, context=ordinal)
                    row['output'] = write_bytes(case_dir / 'output.bin', readback.raw)
                    row['observation'] = classify(readback.raw, cfg, case)
                    row['transport_status'] = 'PASS'
                except BaseException as error:
                    row['transport_status'] = 'ERROR'
                    row['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                    raise
                finally:
                    row['finished_utc'] = now()
                    row['call_index_end_exclusive'] = len(hip.events)
                    row['calls'] = hip.events[row['call_index_begin']:]
                    summary['cases'].append(row)
                    write_json(case_dir / 'result.json', row)
            summary['transport_status'] = 'COMPLETE_ALL_14_CASES'
            exit_code = 0
        except BaseException as error:
            summary['transport_status'] = 'ERROR'
            summary['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        finally:
            if hip is not None:
                cleanup = []
                for name, handle in (('hipFree', device_pointer), ('hipModuleUnload', module)):
                    if handle.value:
                        try:
                            cleanup.append(dict(function=name, result=hip.call(name, handle, context='cleanup', allow_error=True)))
                        except BaseException as error:
                            cleanup.append(dict(function=name, exception=repr(error)))
                summary['cleanup'] = cleanup
                if any(r.get('result') != 0 for r in cleanup):
                    summary['transport_status'] = 'ERROR'
                    exit_code = 1
                try:
                    summary['maps_at_terminal'] = capture_maps(output, 'terminal')
                except BaseException as error:
                    summary['terminal_maps_error'] = repr(error)
                    summary['transport_status'] = 'ERROR'
                    exit_code = 1
                summary['hip_calls'] = hip.events
            summary['finished_utc'] = now()
            summary['host_exit_code'] = exit_code
            summary['completed_transport_cases'] = sum(r['transport_status'] == 'PASS' for r in summary['cases'])
            summary['process_wait_scope'] = 'Only the external root wrapper can certify process exit and reaping.'
            report = write_json(output / 'result.json', summary)
            print(json.dumps(report), flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--configuration-sha256', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg, identity, code = check_configuration(args.configuration, args.configuration_sha256)
    if args.check_only:
        print(json.dumps(dict(status='PASS_FILE_ONLY_PREFLIGHT', configuration=identity, module=cfg['module'],
            libraries=cfg['libraries'], cases=len(cfg['cases']), packet_bytes=4 * cfg['packet_words'],
            CDLL_called=False, HIP_initialized=False)), flush=True)
        return 0
    if args.output is None:
        parser.error('--output is required for execution')
    return run(cfg, identity, code, args.output)


if __name__ == '__main__':
    sys.exit(main())
