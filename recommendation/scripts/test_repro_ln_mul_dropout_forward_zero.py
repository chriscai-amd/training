#!/usr/bin/env python3
"""CPU contracts for the guarded exact-zero LN/multiply/dropout diagnostic."""
import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import repro_ln_mul_dropout_forward_zero as repro


def config(**changes):
    value = {**repro.DEFAULT_CONFIG, 'rows': 5, 'cols': 3, 'stride_x': 5,
             'stride_u': 7, 'guard_elements': 16}
    value.update(changes)
    return value


def backing(value):
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def correct_outputs(state, geometry):
    outputs = state['outputs']
    outputs['Y'].zero_()
    outputs['Mean'].zero_()
    eps32 = struct.unpack('f', struct.pack('f', geometry['eps']))[0]
    outputs['Rstd'].fill_(1.0 / math.sqrt(eps32))


def tiny_elf(text=b'actual-machine-code'):
    names = b'\x00.text\x00.shstrtab\x00'
    section_offset = 128
    value = bytearray(section_offset + 3 * 64)
    value[:6] = b'\x7fELF\x02\x01'
    value[64:64 + len(text)] = text
    string_offset = 64 + len(text)
    value[string_offset:string_offset + len(names)] = names
    struct.pack_into('<Q', value, 40, section_offset)
    struct.pack_into('<HHH', value, 58, 64, 3, 2)
    struct.pack_into('<IIQQQQIIQQ', value, section_offset + 64,
                     1, 1, 6, 0, 64, len(text), 0, 0, 4, 0)
    struct.pack_into('<IIQQQQIIQQ', value, section_offset + 128,
                     7, 3, 0, 0, string_offset, len(names), 0, 0, 1, 0)
    return bytes(value)


class LnMultiplyDropoutForwardZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()

    def test_guarded_buffers_preserve_strides_and_exact_input_recipe(self):
        geometry = config()
        state = repro.make_buffers(geometry, device='cpu')
        self.assertEqual(set(state['inputs']), {'X', 'U', 'W', 'B', 'RANDOM_MASK'})
        self.assertEqual(set(state['outputs']), {'Y', 'Mean', 'Rstd'})
        inputs, outputs = state['inputs'], state['outputs']
        self.assertEqual(inputs['X'].shape, (5, 3))
        self.assertEqual(inputs['X'].stride(), (5, 1))
        self.assertEqual(inputs['U'].stride(), (7, 1))
        self.assertEqual(outputs['Y'].shape, (5, 9))
        self.assertEqual(outputs['Y'].dtype, torch.bfloat16)
        self.assertEqual(outputs['Mean'].dtype, torch.float32)
        self.assertEqual(outputs['Rstd'].dtype, torch.float32)
        for name, expected in (('X', 0), ('U', 0), ('W', 1), ('B', 0), ('RANDOM_MASK', 7)):
            self.assertTrue(bool((inputs[name] == expected).all()))
        self.assertEqual(inputs['RANDOM_MASK'].dtype, torch.int8)
        for value in {**inputs, **outputs}.values():
            self.assertGreaterEqual(value.storage_offset(), geometry['guard_elements'])
            last = value.storage_offset() + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
            self.assertGreaterEqual(value.untyped_storage().nbytes() // value.element_size() - last - 1,
                                    geometry['guard_elements'])
        self.assertFalse(repro.check_inputs(inputs, chunk_elements=7)['failed'])

    def test_geometry_rejects_invalid_dimensions_limits_and_effective_fp32_scalars(self):
        for name, bad in (('rows', 0), ('rows', True), ('rows', 2**31),
                          ('cols', 0), ('stride_x', 2), ('stride_u', 1),
                          ('block_n', 3), ('num_warps', 3), ('guard_elements', -16),
                          ('guard_elements', 0), ('guard_elements', 1),
                          ('eps', 0.), ('eps', -1.), ('eps', 1e-100),
                          ('eps', float('inf')), ('eps', float('nan')),
                          ('dropout_ratio', 1.), ('dropout_ratio', -0.1),
                          ('dropout_ratio', 1. - 1e-12), ('dropout_ratio', float('nan'))):
            with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                repro.geometry_plan(config(**{name: bad}))
        repro.geometry_plan(config(eps=1.0))
        repro.geometry_plan(copy.deepcopy(repro.DEFAULT_CONFIG))

    def test_correct_zero_outputs_and_negative_zero_pass(self):
        for eps in (1e-6, 1.0):
            geometry = config(eps=eps)
            state = repro.make_buffers(geometry, device='cpu')
            repro.poison_outputs(state['outputs'])
            correct_outputs(state, geometry)
            state['outputs']['Y'].fill_(-0.)
            state['outputs']['Mean'].fill_(-0.)
            checked = repro.check_outputs(state['outputs'], geometry, chunk_elements=7)
            self.assertFalse(checked['failed'])

    def test_unwritten_outputs_and_each_concatenated_branch_are_checked(self):
        geometry = config()
        state = repro.make_buffers(geometry, device='cpu')
        repro.poison_outputs(state['outputs'])
        self.assertTrue(repro.check_outputs(state['outputs'], geometry, chunk_elements=7)['failed'])
        for col in (0, geometry['cols'], 2 * geometry['cols']):
            repro.poison_outputs(state['outputs'])
            correct_outputs(state, geometry)
            state['outputs']['Y'][-1, col] = 1
            with self.subTest(branch_column=col):
                self.assertTrue(repro.check_outputs(state['outputs'], geometry, chunk_elements=7)['failed'])

    def test_zero_oracle_catches_signed_subnormals_nan_and_inf(self):
        geometry = config()
        for name, dtype in (('Y', torch.int16), ('Mean', torch.int32)):
            for word in (1, torch.iinfo(dtype).min + 1):
                state = repro.make_buffers(geometry, device='cpu')
                repro.poison_outputs(state['outputs'])
                correct_outputs(state, geometry)
                value = state['outputs'][name]
                value.view(dtype)[(0,) * value.ndim] = word
                with self.subTest(name=name, word=word):
                    self.assertTrue(repro.check_outputs(state['outputs'], geometry, chunk_elements=1)['failed'])
        for name in ('Y', 'Mean', 'Rstd'):
            for number in (float('nan'), float('inf'), -float('inf')):
                state = repro.make_buffers(geometry, device='cpu')
                repro.poison_outputs(state['outputs'])
                correct_outputs(state, geometry)
                value = state['outputs'][name]
                value[(0,) * value.ndim] = number
                with self.subTest(name=name, number=number):
                    self.assertTrue(repro.check_outputs(state['outputs'], geometry, chunk_elements=7)['failed'])

    def test_rstd_all_rows_checked_for_zero_negative_and_finite_large_error(self):
        geometry = config()
        for number in (0., -1000., 1001.):
            state = repro.make_buffers(geometry, device='cpu')
            repro.poison_outputs(state['outputs'])
            correct_outputs(state, geometry)
            state['outputs']['Rstd'][-1] = number
            with self.subTest(number=number):
                self.assertTrue(repro.check_outputs(state['outputs'], geometry, chunk_elements=2)['failed'])

    def test_all_input_constants_and_guard_or_padding_mutations_are_checked(self):
        geometry = config()
        for name in ('X', 'U', 'W', 'B', 'RANDOM_MASK'):
            for kind in ('logical', 'prefix_guard', 'suffix_guard'):
                state = repro.make_buffers(geometry, device='cpu')
                tensor = state['inputs'][name]
                if kind == 'logical':
                    tensor[(0,) * tensor.ndim] += 1
                else:
                    value = backing(tensor)
                    index = 0 if kind == 'prefix_guard' else value.numel() - 1
                    value[index] ^= 1
                with self.subTest(name=name, kind=kind):
                    self.assertTrue(repro.check_inputs(state['inputs'], chunk_elements=3)['failed'])
        for name in ('X', 'U'):
            state = repro.make_buffers(geometry, device='cpu')
            tensor = state['inputs'][name]
            padding_byte = (tensor.storage_offset() + tensor.shape[1]) * tensor.element_size()
            backing(tensor)[padding_byte] ^= 1
            with self.subTest(padding=name):
                self.assertTrue(repro.check_inputs(state['inputs'], chunk_elements=3)['failed'])

    def test_output_guard_mutations_detected_separately_from_legal_zeros(self):
        geometry = config()
        for name in ('Y', 'Mean', 'Rstd'):
            state = repro.make_buffers(geometry, device='cpu')
            repro.poison_outputs(state['outputs'])
            correct_outputs(state, geometry)
            backing(state['outputs'][name])[0] ^= 1
            with self.subTest(name=name):
                self.assertTrue(repro.check_outputs(state['outputs'], geometry, chunk_elements=3)['failed'])

    def test_output_guards_are_finite_and_logical_poison_has_no_payload_assumption(self):
        geometry = config()
        state = repro.make_buffers(geometry, device='cpu')
        for _ in range(2):
            for name, tensor in state['outputs'].items():
                raw = repro.flat_storage(tensor)
                offset = tensor.storage_offset()
                end = offset + tensor.numel()
                with self.subTest(name=name):
                    self.assertTrue(bool((raw[:offset] == 23).all()))
                    self.assertTrue(bool((raw[end:] == 23).all()))
                    self.assertTrue(bool(torch.isnan(tensor).all()))
            correct_outputs(state, geometry)
            original_bits = repro.expected_bits
            def finite_bits_only(dtype, value):
                self.assertTrue(math.isfinite(value), 'Guard checks must not assume a CPU NaN payload')
                return original_bits(dtype, value)
            with mock.patch.object(repro, 'expected_bits', side_effect=finite_bits_only):
                self.assertFalse(repro.check_outputs(state['outputs'], geometry, chunk_elements=3)['failed'])
            repro.poison_outputs(state['outputs'])
        for name, integer, payload in (('Y', torch.int16, 0x7FC1),
                                        ('Mean', torch.int32, 0x7FC00001),
                                        ('Rstd', torch.int32, 0x7FFFFFFF)):
            correct_outputs(state, geometry)
            tensor = state['outputs'][name]
            tensor.view(integer)[(0,) * tensor.ndim] = payload
            checked = repro.check_outputs(state['outputs'], geometry, chunk_elements=3)
            with self.subTest(noncanonical_logical_nan=name):
                self.assertTrue(checked['failed'])
                self.assertEqual(checked['tensors'][name]['nonfinite_elements'], 1)
                self.assertTrue(all(v['changed_guard_elements'] == 0 for v in checked['tensors'].values()))

    def test_zero_nan_and_other_stores_into_finite_output_guards_are_detected(self):
        geometry = config()
        for name in ('Y', 'Mean', 'Rstd'):
            for side in ('prefix', 'suffix'):
                for number in (0., float('nan'), 24.):
                    state = repro.make_buffers(geometry, device='cpu')
                    correct_outputs(state, geometry)
                    raw = repro.flat_storage(state['outputs'][name])
                    raw[0 if side == 'prefix' else -1] = number
                    checked = repro.check_outputs(state['outputs'], geometry, chunk_elements=3)
                    with self.subTest(name=name, side=side, number=number):
                        self.assertTrue(checked['failed'])
                        self.assertEqual(checked['tensors'][name]['changed_guard_elements'], 1)
                        self.assertTrue(all(v['mismatched_elements'] == 0 for v in checked['tensors'].values()))

    def test_erroneous_zero_stores_into_input_guards_or_padding_are_detected(self):
        geometry = config()
        for name in ('X', 'U', 'W', 'B', 'RANDOM_MASK'):
            for region in ('prefix', 'suffix'):
                state = repro.make_buffers(geometry, device='cpu')
                flat = repro.flat_storage(state['inputs'][name])
                index = 0 if region == 'prefix' else flat.numel() - 1
                with self.subTest(name=name, region=region):
                    self.assertNotEqual(float(flat[index]), 0.)
                    flat[index] = 0
                    self.assertTrue(repro.check_inputs(state['inputs'], chunk_elements=3)['failed'])
        for name in ('X', 'U'):
            state = repro.make_buffers(geometry, device='cpu')
            tensor = state['inputs'][name]
            flat = repro.flat_storage(tensor)
            index = tensor.storage_offset() + tensor.shape[1]
            with self.subTest(name=name, region='padding'):
                self.assertNotEqual(float(flat[index]), 0.)
                flat[index] = 0
                self.assertTrue(repro.check_inputs(state['inputs'], chunk_elements=3)['failed'])

    def run_fixture(self, geometry, path, **kwargs):
        return repro.run_loop(geometry, failure_path=path, device='cpu',
                              max_resident_bytes=4 << 20, max_capture_bytes=4 << 20,
                              chunk_elements=7, chunk_bytes=31, **kwargs)

    def test_resident_loop_repoisons_outputs_and_publishes_each_completed_call(self):
        geometry = config()
        pointers, events = [], []
        def launch(state, actual_config):
            pointers.append(tuple(value.data_ptr() for value in state['outputs'].values()))
            self.assertEqual(actual_config, geometry)
            self.assertTrue(all(bool(torch.isnan(value).all()) for value in state['outputs'].values()))
            correct_outputs(state, actual_config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            result = self.run_fixture(geometry, path, repeats=4, launch=launch,
                                      before_call=lambda row: events.append(('before', row['iteration'])),
                                      progress=lambda row: events.append(('complete', row['iteration'])))
            self.assertFalse(path.exists())
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['final_input_hashes_match_initial'])
        self.assertEqual(len(set(pointers)), 1)
        self.assertEqual(events, [(stage, number) for number in range(1, 5) for stage in ('before', 'complete')])

    def test_first_fault_full_backings_and_initial_hashes_preserved_without_rerun(self):
        geometry = config()
        for fault in ('Y', 'Mean', 'Rstd', 'U_padding', 'RANDOM_MASK'):
            calls, observed = [], {}
            configuration = {'session': {'identity': 'before-dispatch'}}
            def launch(state, actual_config):
                calls.append(1)
                correct_outputs(state, actual_config)
                if len(calls) == 3:
                    configuration['session']['identity'] = 'changed-after-runtime-snapshot'
                    if fault == 'U_padding':
                        value = state['inputs']['U']
                        backing(value)[(value.storage_offset() + value.shape[1]) * value.element_size()] ^= 1
                    elif fault == 'RANDOM_MASK':
                        state['inputs'][fault][-1, -1] = 0
                    else:
                        value = state['outputs'][fault]
                        value[(0,) * value.ndim] = -1
                    observed.update({(group, name): backing(value).clone()
                                     for group, values in state.items() for name, value in values.items()})
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'failure.pt'
                result = self.run_fixture(geometry, path, repeats=9, launch=launch,
                                          configuration=configuration)
                artifact = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(len(calls), 3)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(artifact['format'], 'ln_mul_dropout_forward_zero_failure_v1')
            self.assertEqual(artifact['iteration'], 3)
            self.assertEqual(artifact['runtime']['configuration']['session']['identity'], 'before-dispatch')
            self.assertEqual(artifact['configuration'], geometry)
            self.assertEqual(artifact['runtime']['synthetic_recipe'], {'X': 0, 'U': 0, 'W': 1, 'B': 0, 'RANDOM_MASK': 7})
            for (group, name), expected in observed.items():
                self.assertTrue(torch.equal(backing(artifact['current_at_failure'][group][name]), expected))
            original = repro.make_buffers(geometry, device='cpu')
            self.assertEqual(repro.replay.digest_tree(original['inputs'], chunk_bytes=31), artifact['initial_input_digest'])
            self.assertEqual(artifact['observed_record']['input_hashes_match_initial'], fault in ('Y', 'Mean', 'Rstd'))

    def test_noop_launch_is_a_first_call_failure_with_actual_poison_retained(self):
        geometry = config()
        launch = mock.Mock()
        launch.evidence = {'injected_callable': True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            result = self.run_fixture(geometry, path, repeats=5, launch=launch)
            artifact = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(result['iterations_completed'], 1)
        self.assertTrue(all(bool(torch.isnan(value).all()) for value in artifact['current_at_failure']['outputs'].values()))

    def test_resource_and_existing_path_rejection_precedes_allocation_and_dispatch(self):
        geometry = config()
        launch = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            with mock.patch.object(repro, 'make_buffers') as allocate:
                for name in ('max_resident_bytes', 'max_capture_bytes'):
                    with self.subTest(resource=name), self.assertRaisesRegex(ValueError, 'budget'):
                        repro.run_loop(geometry, failure_path=path, launch=launch, **{name: 1})
                for suffix in ('', '.tmp'):
                    occupied = Path(str(path) + suffix)
                    occupied.write_bytes(b'preserve-existing')
                    with self.subTest(occupied=suffix), self.assertRaises(FileExistsError):
                        repro.run_loop(geometry, failure_path=path, launch=launch)
                    self.assertEqual(occupied.read_bytes(), b'preserve-existing')
                    occupied.unlink()
                allocate.assert_not_called()
        launch.assert_not_called()

    def test_no_implicit_cpu_kernel_and_no_duplicate_launcher_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            for kwargs in ({}, {'launch': mock.Mock(), 'launch_factory': mock.Mock()}):
                with self.assertRaisesRegex(ValueError, 'Exactly one'):
                    self.run_fixture(config(), path, repeats=1, **kwargs)

    def test_failed_artifact_write_cleans_owned_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            with mock.patch.object(torch, 'save', side_effect=RuntimeError('write failed')):
                with self.assertRaisesRegex(RuntimeError, 'write failed'):
                    self.run_fixture(config(), path, repeats=1, launch=lambda state, geometry: None)
            self.assertFalse(path.exists())
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_elf_text_identity_distinguishes_code_from_nontext_metadata(self):
        text = b'first-machine-code'
        binary = tiny_elf(text)
        expected = hashlib.sha256(text).hexdigest()
        self.assertEqual(repro.elf_text_hash(binary), expected)
        changed = bytearray(binary)
        changed[24] ^= 1
        self.assertEqual(repro.elf_text_hash(bytes(changed)), expected)
        changed[64] ^= 1
        self.assertNotEqual(repro.elf_text_hash(bytes(changed)), expected)
        for invalid in (b'', b'not-an-elf', binary[:32], binary[:150]):
            self.assertIsNone(repro.elf_text_hash(invalid))

    def test_original_inner_jit_archived_before_dispatch_with_exact_argument_order(self):
        @dataclass
        class Target:
            backend: str = 'hip'
            arch: str = 'gfx1250'
            warp_size: int = 32
        geometry = config()
        state = repro.make_buffers(geometry, device='cpu')
        binary = tiny_elf()
        runner = mock.Mock(side_effect=lambda *args: correct_outputs(state, geometry))
        class Compiled:
            name = repro.KERNEL
            hash = 'cache-key'
            metadata = {'target': Target(), 'shared': 64, 'num_warps': 1}
            asm = {'hsaco': binary, 'amdgcn': 'assembly text', 'ttir': 'IR text'}
            kernel = binary
            function = 12345
            n_regs = 1024
            n_spills = 174
            def __getitem__(self, grid):
                self.grid = grid
                return runner
        compiled = Compiled()
        jit = SimpleNamespace(arg_names=repro.ARGUMENTS, warmup=mock.Mock(return_value=compiled))
        wrapped = SimpleNamespace(fn=jit)
        module = SimpleNamespace(**{repro.KERNEL: wrapped})
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory) / 'code'
            with mock.patch.object(repro.importlib, 'import_module', return_value=module), \
                    mock.patch.object(repro, 'original_inner_jit', return_value=jit):
                launcher = repro.OriginalLauncher(state, geometry, code)
            runner.assert_not_called()
            self.assertEqual((code / (repro.KERNEL + '.hsaco')).read_bytes(), binary)
            evidence = json.loads((code / 'metadata.json').read_text())
            self.assertEqual(evidence['metadata']['target']['arch'], 'gfx1250')
            self.assertEqual(evidence['loaded_kernel_sha256'], hashlib.sha256(binary).hexdigest())
            self.assertEqual(evidence['hsaco_text_sha256'], hashlib.sha256(b'actual-machine-code').hexdigest())
            self.assertEqual(evidence['artifacts']['hsaco']['sha256'], hashlib.sha256(binary).hexdigest())
            root = Path(repro.__file__).resolve().parents[1]
            self.assertEqual((code / Path(repro.MODEL_SOURCE).name).read_bytes(), (root / repro.MODEL_SOURCE).read_bytes())
            self.assertEqual((code / Path(repro.__file__).name).read_bytes(), Path(repro.__file__).read_bytes())
            launcher(state, geometry)
            self.assertEqual(runner.call_count, 1)
            expected_tensors = [state[group][name] for group, name in
                                (('inputs', 'X'), ('inputs', 'U'), ('outputs', 'Y'),
                                 ('inputs', 'W'), ('inputs', 'B'), ('outputs', 'Mean'),
                                 ('outputs', 'Rstd'), ('inputs', 'RANDOM_MASK'))]
            actual = runner.call_args.args
            self.assertEqual(len(actual), 23)
            self.assertTrue(all(left is right for left, right in zip(actual[:8], expected_tensors)))
            self.assertEqual(actual[8:], (5, 3, 1e-6, .3, 5, 7, 9, 3,
                                          False, 4, 16, True, True, True, 'none'))
            self.assertEqual(jit.warmup.call_args.kwargs, {'grid': (1, 1, 1), 'num_warps': 1, 'num_stages': 1})
            with self.assertRaisesRegex(ValueError, 'changed'):
                launcher(state, config(block_n=8))

    def test_inner_jit_selection_bypasses_autotuner_warmup_and_rejects_cycles(self):
        from triton.runtime.jit import JITFunction
        jit = object.__new__(JITFunction)
        outer_warmup = mock.Mock(side_effect=AssertionError('Autotuner warmup must never run'))
        outer = SimpleNamespace(warmup=outer_warmup, fn=SimpleNamespace(fn=jit))
        self.assertIs(repro.original_inner_jit(outer), jit)
        outer_warmup.assert_not_called()
        cycle = SimpleNamespace()
        cycle.fn = cycle
        for invalid in (cycle, SimpleNamespace(warmup=outer_warmup)):
            with self.assertRaisesRegex(ValueError, 'inner JITFunction'):
                repro.original_inner_jit(invalid)

    def test_source_address_audit_rejects_old_int32_pointer_arithmetic(self):
        source = (Path(repro.__file__).resolve().parents[1] / repro.MODEL_SOURCE).read_text()
        checked = repro.audit_address_source(source)
        self.assertGreaterEqual(len(checked['verified_widened_pointer_products']), 4)
        for old, new in (('rows_i64[:, None] * stride_y', 'rows[:, None] * stride_y'),
                         ('rows.to(tl.int64)', 'rows.to(tl.int32)')):
            self.assertIn(old, source)
            with self.assertRaises(ValueError):
                repro.audit_address_source(source.replace(old, new))

    def test_cpu_cli_plans_production_without_allocation_or_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with mock.patch.object(repro, 'make_buffers') as allocate, mock.patch.object(repro, 'run_loop') as run, redirect_stdout(io.StringIO()):
                status = repro.main(['--report', str(path)])
            allocate.assert_not_called()
            run.assert_not_called()
            report = json.loads(path.read_text())
        self.assertEqual(status, 0)
        self.assertEqual(report['status'], 'CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH')
        self.assertEqual(report['geometry_plan']['grid'], [88956, 1, 1])
        self.assertTrue(report['geometry_plan']['address_guard']['crosses_old_signed32_y_product'])
        self.assertEqual(report['geometry_plan']['resource_plan']['resident_storage_bytes'], 8027339176)
        self.assertTrue(report['source_files_unchanged_during_run'])

    def test_cpu_cli_records_error_and_protects_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with mock.patch.object(repro, 'source_inventory', side_effect=RuntimeError('source audit failed')):
                with self.assertRaisesRegex(RuntimeError, 'source audit failed'):
                    repro.main(['--report', str(path)])
            report = json.loads(path.read_text())
            self.assertEqual(report['status'], 'ERROR')
            self.assertEqual(report['error']['type'], 'RuntimeError')
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                repro.main(['--report', str(path)])
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
