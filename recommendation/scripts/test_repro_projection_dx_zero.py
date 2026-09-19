"""No-GPU regressions for one-mm zero oracle, first fault and input evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import repro_projection_dx_zero as target


def bundle(rows=17, width=5, projected=7):
    w = torch.arange(width * projected, dtype=torch.float32).reshape(width, projected).div(1024).bfloat16()
    raw = memoryview(w.view(torch.uint8).numpy()).cast('B')
    return {'format': target.BUNDLE_FORMAT, 'w': w,
        'w_storage_sha256': hashlib.sha256(raw).hexdigest(),
        'dz_layout': {'shape': [rows, projected], 'stride': [projected, 1], 'dtype': 'torch.bfloat16',
                      'storage_offset': 0, 'storage_bytes': rows * projected * 2},
        'execution_controls': {'autocast': {'cpu': {'enabled': False, 'dtype': 'torch.bfloat16'},
            'cuda': {'enabled': False, 'dtype': 'torch.float16'}}, 'float32_matmul_precision': 'highest',
            'allow_tf32': False, 'allow_fp16_reduced_precision_reduction': True,
            'allow_bf16_reduced_precision_reduction': True},
        'blas_preference': '_BlasBackend.Cublaslt',
        'provenance': {'type': 'CPU_test_fixture'}}


class DirectMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()

    def setUp(self):
        guard = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_prepare_retains_weight_bytes_and_exact_original_zero_layout(self):
        b = bundle()
        p = target.prepare(b, chunk_bytes=1024)
        self.assertEqual(p['inputs']['dz'].shape, (17, 7))
        self.assertEqual(p['inputs']['dz'].stride(), (7, 1))
        self.assertEqual(p['inputs']['dz'].storage_offset(), 0)
        self.assertEqual(int(torch.count_nonzero(p['inputs']['dz'])), 0)
        self.assertIs(p['inputs']['w'], b['w'])
        self.assertEqual(p['input_digests']['inputs']['w']['logical_sha256'], b['w_storage_sha256'])

    def test_zero_checker_accepts_negative_zero_and_detects_subnormal_nan_inf(self):
        z = torch.zeros(3, 5, dtype=torch.bfloat16)
        z.view(torch.int16)[0, 0] = -32768
        self.assertFalse(target.check_zero(z, shape=z.shape, dtype=z.dtype, device=z.device, chunk_elements=4)['failed'])
        z.view(torch.int16)[1, 2] = 1
        z[2, 0], z[2, 1] = float('nan'), float('inf')
        result = target.check_zero(z, shape=z.shape, dtype=z.dtype, device=z.device, chunk_elements=4)
        self.assertEqual(result['nonzero_magnitude_bits_count'], 3)
        self.assertEqual((result['nan_count'], result['infinity_count']), (1, 1))
        self.assertTrue(result['failed'])

    def test_exactly_one_DX_mm_per_iteration_no_warmup_or_DB_DW(self):
        p = target.prepare(bundle(), chunk_bytes=1024)
        calls = []
        def mm(dz, transposed_w):
            calls.append((tuple(dz.shape), tuple(transposed_w.shape), tuple(transposed_w.stride())))
            return torch.mm(dz, transposed_w)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            r = target.run(p, failure_path=path, repeats=3, device='cpu', function=mm,
                           chunk_bytes=1024, chunk_elements=16)
            self.assertEqual(r['status'], 'PASS')
            self.assertEqual(calls, [((17, 7), (7, 5), (1, 7))] * 3)
            self.assertTrue(r['final_input_bytes_unchanged'])
            self.assertFalse(path.exists())

    def test_first_finite_or_nonfinite_corruption_retained_with_no_rerun(self):
        for corrupt in ('finite', 'subnormal', 'nan'):
            p = target.prepare(bundle(), chunk_bytes=1024)
            calls = []
            def mm(dz, transposed_w):
                calls.append(1)
                output = torch.mm(dz, transposed_w)
                if corrupt == 'finite': output[3, 2] = 0.03125
                elif corrupt == 'subnormal': output.view(torch.int16)[3, 2] = 1
                else: output[3, 2] = float('nan')
                return output
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'failure.pt'
                r = target.run(p, failure_path=path, repeats=3, device='cpu', function=mm,
                               chunk_bytes=1024, chunk_elements=16)
                self.assertEqual((r['status'], r['iterations_completed'], len(calls)), ('FAIL', 1, 1))
                a = torch.load(path, weights_only=False, map_location='cpu')
                self.assertTrue(a['input_backing_bytes_unchanged'])
                self.assertTrue(a['saved_DX_check_matches_observed'])
                self.assertFalse(a['capture_integrity_failure'])
                self.assertIn('scripts/repro_projection_dx_zero.py', a['source_sha256_at_failure'])
                self.assertEqual(a['runtime_versions']['torch'], str(torch.__version__))
                self.assertEqual(set(a['current_at_failure']['inputs']), {'w', 'dz'})
                self.assertEqual(a['saved_DX_zero_check']['nonzero_magnitude_bits_count'], 1)
                self.assertNotEqual(a['pristine_inputs']['w'].untyped_storage()._cdata,
                                    a['current_at_failure']['inputs']['w'].untyped_storage()._cdata)

    def test_input_mutation_fault_preserves_prior_and_current_storage(self):
        p = target.prepare(bundle(), chunk_bytes=1024)
        calls = []
        def mm(dz, w):
            calls.append(1)
            output = torch.mm(dz, w)
            dz[1, 2] = 1
            return output
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            r = target.run(p, failure_path=path, repeats=3, device='cpu', function=mm, chunk_bytes=1024)
            a = torch.load(path, weights_only=False, map_location='cpu')
            self.assertEqual(len(calls), 1)
            self.assertEqual(r['status'], 'FAIL')
            self.assertFalse(a['input_backing_bytes_unchanged'])
            self.assertEqual(a['pristine_inputs']['dz'][1, 2], 0)
            self.assertEqual(a['current_at_failure']['inputs']['dz'][1, 2], 1)
            self.assertFalse(a['saved_DX_zero_check']['failed'])

    def test_backend_preference_verified_and_restored_even_on_error(self):
        state = ['_BlasBackend.Default']
        calls = []
        def preferred(value=None):
            if value is not None:
                calls.append(value)
                state[0] = {v: k for k, v in target.BACKENDS.items()}[value]
            return state[0]
        with mock.patch.object(torch.backends.cuda, 'preferred_blas_library', side_effect=preferred):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with target.selected_backend('hipblaslt') as value:
                    self.assertEqual(value['observed'], '_BlasBackend.Cublaslt')
                    raise RuntimeError('injected')
        self.assertEqual(calls, ['hipblaslt', 'default'])
        self.assertEqual(state[0], '_BlasBackend.Default')

    def test_default_backend_can_resolve_to_concrete_ROCm_preference(self):
        state = ['_BlasBackend.Cublas']
        def preferred(value=None):
            if value is not None:
                state[0] = '_BlasBackend.Cublaslt' if value == 'default' else {v: k for k, v in target.BACKENDS.items()}[value]
            return state[0]
        with mock.patch.object(torch.backends.cuda, 'preferred_blas_library', side_effect=preferred):
            with target.selected_backend('default') as observed:
                self.assertEqual(observed['requested'], 'default')
                self.assertEqual(observed['observed'], '_BlasBackend.Cublaslt')
        self.assertEqual(state[0], '_BlasBackend.Cublas')

    def test_changed_saved_output_is_retained_but_marked_capture_integrity_failure(self):
        p = target.prepare(bundle(), chunk_bytes=1024)
        calls = []
        def mm(dz, w):
            calls.append(1)
            result = torch.mm(dz, w)
            result[0, 0] = 0.03125
            return result
        original_copy = target._cpu_copy_tree
        def changed_copy(*args, **kwargs):
            current, size = original_copy(*args, **kwargs)
            current['dx'].zero_()
            return current, size
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            with mock.patch.object(target, '_cpu_copy_tree', side_effect=changed_copy):
                r = target.run(p, failure_path=path, repeats=3, device='cpu', function=mm,
                               chunk_bytes=1024, configuration={'test_launch_identity': 'fixture1'})
            a = torch.load(path, weights_only=False, map_location='cpu')
            self.assertEqual(len(calls), 1)
            self.assertEqual(r['status'], 'CAPTURE_INTEGRITY_FAILURE')
            self.assertEqual(a['status'], 'CAPTURE_INTEGRITY_FAILURE')
            self.assertFalse(a['saved_DX_check_matches_observed'])
            self.assertTrue(a['capture_integrity_failure'])
            self.assertEqual(a['configuration']['test_launch_identity'], 'fixture1')

    def test_GPU_environment_guard_runs_before_input_load_and_GPU_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ['--bundle', directory + '/missing.pt', '--report', directory + '/report.json',
                    '--failure-dump', directory + '/failure.pt', '--gpu']
            for key in target.MANDATORY_GPU_ENVIRONMENT:
                environment = {name: '0' for name in target.MANDATORY_GPU_ENVIRONMENT}
                environment[key] = '1'
                with mock.patch.dict(os.environ, environment), \
                        mock.patch.object(torch, 'load', side_effect=AssertionError('Input load before guard')), \
                        self.assertRaises(SystemExit) as caught:
                    target.main(args)
                self.assertEqual(caught.exception.code, 2)

    def test_invalid_checksum_layout_nonfinite_and_budget_rejected_before_mm(self):
        for change in ('checksum', 'layout', 'nonfinite'):
            b = bundle()
            if change == 'checksum': b['w_storage_sha256'] = 'bad'
            elif change == 'layout': b['dz_layout']['stride'] = [14, 2]
            else:
                b['w'][0, 0] = float('nan')
                b['w_storage_sha256'] = hashlib.sha256(memoryview(b['w'].view(torch.uint8).numpy()).cast('B')).hexdigest()
            with self.assertRaises(ValueError):
                target.prepare(b)
        with self.assertRaisesRegex(ValueError, 'CPU storage budget'):
            target.prepare(bundle(), max_bytes=1)
        p = target.prepare(bundle(), chunk_bytes=1024)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'before dispatch'):
            target.run(p, failure_path=Path(directory) / 'bad.pt', function=lambda *args: self.fail('MM reached'),
                       repeats=1, device='cpu', max_resident_bytes=1)

    def test_cpu_only_cli_report_has_source_and_bundle_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            torch.save(bundle(), directory / 'bundle.pt')
            code = target.main(['--bundle', str(directory / 'bundle.pt'), '--report', str(directory / 'report.json'),
                                '--failure-dump', str(directory / 'fail.pt'), '--cpu-threads', '2', '--repeats', '1'])
            r = json.loads((directory / 'report.json').read_text())
            self.assertEqual(code, 0)
            self.assertEqual(r['status'], 'CPU_PREPARED')
            self.assertEqual(r['requested_backend'], 'hipblaslt')
            self.assertTrue(r['source_files_unchanged'])
            self.assertFalse((directory / 'fail.pt').exists())


if __name__ == '__main__':
    unittest.main()
