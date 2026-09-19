#!/usr/bin/env python3
"""CPU replay contracts, using real finite captures from the focused probe."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import capture_additional_linear as capture
import replay_additional_linear_capture as replay
from test_capture_additional_linear import TinyModel, input_tensor


def finite_capture(dtype=torch.float32, strided=False):
    model = TinyModel()
    with tempfile.TemporaryDirectory() as directory:
        probe = capture.AdditionalLinearProbe(model, Path(directory) / 'capture',
                                               expected_dims=(3, 4), chunk_bytes=1024,
                                               max_bytes=4 << 20, save_finite_step=1)
        try:
            probe.set_attempt('training', 0, 1)
            value = input_tensor(dtype=dtype, strided=strided)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                result = model(value)
            result.sum().backward()
            path, = probe.directory.glob('*.pt')
            return torch.load(path, weights_only=False, map_location='cpu')
        finally:
            probe.close()


def raw(value):
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,)).clone()


class AdditionalLinearReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()
        cls.controls = replay.helpers.execution_controls()
        cls.fixture = finite_capture()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert replay.helpers.execution_controls() == cls.controls
        assert not torch.cuda.is_initialized()

    def payload(self):
        return copy.deepcopy(self.fixture)

    def prepared(self, payload, zero=False):
        return replay.prepare_capture(payload, zero_dy=zero, chunk_bytes=1024, max_bytes=4 << 20)

    def run_capture(self, payload, prepared, path, **kwargs):
        return replay.run_replay(payload, prepared, failure_path=path, device='cpu',
                                 max_resident_bytes=8 << 20, max_capture_bytes=16 << 20,
                                 chunk_bytes=1024, **kwargs)

    def test_real_probe_payload_binds_actual_pre_node_operands_and_stages(self):
        payload = self.payload()
        bound = replay.validate_capture(payload)
        self.assertEqual(bound['inputs']['x'].dtype, torch.bfloat16)
        self.assertEqual(bound['inputs']['mat2'].stride(), (1, 3))
        self.assertEqual(bound['inputs']['dy'].stride(), (0, 0))
        self.assertEqual(bound['inputs']['dy'].untyped_storage().nbytes(), 2)
        prepared = self.prepared(payload)
        self.assertTrue(all(prepared['forward_to_backward_cast_value_bridge'].values()))
        self.assertEqual(prepared['original_digest'], prepared['prepared_digest'])
        self.assertIn('parameter_prior_weight', prepared['original']['stages'])
        self.assertIn('after_backward_return', prepared['original']['stages'])

    def test_strict_shape_provenance_precision_and_scalar_rejections(self):
        cases = []
        p = self.payload(); p['metadata']['input_provenance']['pristine'] = False; cases.append(p)
        p = self.payload(); p['metadata']['input_provenance']['operation_started'] = True; cases.append(p)
        p = self.payload(); del p['metadata']['input_provenance']; cases.append(p)
        for version in (True, 0, 2, '1'):
            p = self.payload(); p['format_version'] = version; cases.append(p)
        for kind in (None, 'MmBackward0', 'AddmmBackward1'):
            p = self.payload(); p['metadata']['node_kind'] = kind; cases.append(p)
        p = self.payload(); p['pristine_inputs']['dy'] = torch.ones(7, 5, dtype=torch.bfloat16); cases.append(p)
        p = self.payload(); p['pristine_inputs']['mat2'] = p['pristine_inputs']['mat2'].float(); cases.append(p)
        p = self.payload(); p['forward_inputs']['weight'] = torch.ones(3, 4); cases.append(p)
        p = self.payload(); p['stages']['raw_small']['dmat2'] = torch.ones(3, 4); cases.append(p)
        p = self.payload(); p['metadata']['execution_controls_backward']['autocast']['cpu']['enabled'] = True; cases.append(p)
        for scalar in ('alpha', 'beta'):
            for value in (0, True, float('nan')):
                p = self.payload(); p['metadata'][scalar] = value; cases.append(p)
        for index, payload in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                replay.validate_capture(payload)

    def test_blas_preference_restores_on_normal_return_and_operation_error(self):
        for fail in (False, True):
            current = ['_BlasBackend.Default']
            calls = []
            def preference(value=None):
                if value is not None:
                    calls.append(value)
                    current[0] = {'hipblaslt': '_BlasBackend.Cublaslt'}.get(value, value)
                return current[0]
            with self.subTest(operation_error=fail), mock.patch.object(torch.backends.cuda, 'preferred_blas_library', side_effect=preference):
                def execute():
                    with replay.restored_blas_preference({'preferred_blas_library': '_BlasBackend.Cublaslt'}) as evidence:
                        self.assertEqual(current[0], '_BlasBackend.Cublaslt')
                        self.assertTrue(evidence['restored'])
                        self.assertEqual(evidence['effective'], '_BlasBackend.Cublaslt')
                        if fail:
                            raise RuntimeError('operation failed')
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'operation failed'):
                        execute()
                else:
                    execute()
                self.assertEqual(current[0], '_BlasBackend.Default')
                self.assertEqual(calls, ['hipblaslt', '_BlasBackend.Default'])

    def test_blas_preference_refuses_silent_backend_change(self):
        calls = []
        def preference(value=None):
            if value is not None:
                calls.append(value)
            return '_BlasBackend.Default'
        with mock.patch.object(torch.backends.cuda, 'preferred_blas_library', side_effect=preference):
            with self.assertRaisesRegex(ValueError, 'restore exactly'):
                with replay.restored_blas_preference({'preferred_blas_library': '_BlasBackend.Cublaslt'}):
                    self.fail('Mismatched backend preference entered operation context')
        self.assertEqual(calls, ['hipblaslt', '_BlasBackend.Default'])

    def test_zero_dy_preserves_protected_full_backings_and_padding(self):
        payload = finite_capture(dtype=torch.bfloat16, strided=True)
        backing = torch.full((7, 11), 17., dtype=torch.bfloat16)
        payload['pristine_inputs']['dy'] = backing[:, 1:9:2]
        payload['pristine_inputs']['dy'].fill_(1)
        original_dy = raw(payload['pristine_inputs']['dy'])
        prepared = self.prepared(payload, True)
        transformed = prepared['prepared']
        self.assertTrue(prepared['transformation']['zero_dy'])
        self.assertTrue(torch.equal(raw(payload['pristine_inputs']['dy']), original_dy))
        dy = transformed['pristine_inputs']['dy']
        self.assertEqual(dy.stride(), (11, 2))
        self.assertEqual(dy.storage_offset(), 1)
        self.assertEqual(int(torch.count_nonzero(dy)), 0)
        after = raw(dy).view(torch.bfloat16).reshape(7, 11)
        before = original_dy.view(torch.bfloat16).reshape(7, 11)
        self.assertTrue(torch.equal(after[:, 0::2], before[:, 0::2]))
        self.assertTrue(torch.equal(after[:, 9:], before[:, 9:]))
        for group, names in (('pristine_inputs', ('x', 'mat2')), ('forward_inputs', ('x', 'weight', 'bias'))):
            for name in names:
                self.assertTrue(torch.equal(raw(transformed[group][name]), raw(payload[group][name])))
                self.assertNotEqual(transformed[group][name].untyped_storage()._cdata, payload[group][name].untyped_storage()._cdata)

    def test_zero_dy_rejects_nonfinite_protected_values_and_shared_backing(self):
        for group, name in (('pristine_inputs', 'x'), ('pristine_inputs', 'mat2'), ('forward_inputs', 'weight'), ('forward_inputs', 'bias')):
            payload = self.payload()
            value = payload[group][name]
            value[(0,) * value.ndim] = float('nan')
            with self.subTest(group=group, name=name), self.assertRaises(ValueError):
                self.prepared(payload, True)
        payload = self.payload()
        backing = torch.ones(40, dtype=torch.bfloat16)
        payload['pristine_inputs']['x'] = backing[:21].reshape(7, 3)
        payload['pristine_inputs']['dy'] = backing[:28].reshape(7, 4)
        with self.assertRaisesRegex(ValueError, 'share backing'):
            self.prepared(payload, True)

    def test_original_nonfinite_inputs_remain_propagation_evidence(self):
        payload = self.payload()
        payload['pristine_inputs']['x'][0, 0] = float('nan')
        prepared = self.prepared(payload)
        self.assertTrue(prepared['transformation']['original_inputs_nonfinite'])
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_capture(payload, prepared, Path(directory) / 'failure.pt', route='raw_dw', repeats=2)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['iterations_completed'], 1)
            self.assertFalse(result['runtime']['conditional_raw_dw_bound']['applicable'])

    def test_all_three_routes_cpu_node_mapping_and_zero_control(self):
        payload = self.payload()
        for zero in (False, True):
            prepared = self.prepared(payload, zero)
            for route in replay.ROUTES:
                resident, _ = replay.common.restore_raw_tree(prepared['prepared'], device='cpu', chunk_bytes=13)
                outputs = replay.execute_route(resident, prepared['metadata'], route)
                self.assertEqual(outputs['raw_dw'].shape, (3, 4))
                self.assertEqual(outputs['raw_dw'].dtype, torch.bfloat16)
                if route == 'autocast_linear':
                    self.assertEqual(outputs['dw'].dtype, torch.float32)
                    self.assertTrue(torch.equal(outputs['raw_dw'].T.float(), outputs['dw']))
                    self.assertTrue(torch.equal(outputs['db'].float(), outputs['leaf_db']))
                if zero:
                    self.assertTrue(all(torch.count_nonzero(value) == 0 for value in outputs.values()))
                else:
                    self.assertTrue(torch.equal(outputs['raw_dw'], payload['stages']['raw_small']['dmat2']))
                with tempfile.TemporaryDirectory() as directory:
                    result = self.run_capture(payload, prepared, Path(directory) / 'failure.pt', route=route, repeats=3)
                    self.assertEqual(result['status'], 'PASS')
                    self.assertTrue(result['final_input_bytes_match_prepared'])

    def test_autocast_route_rejects_forward_to_backward_operand_change(self):
        payload = self.payload()
        payload['pristine_inputs']['x'][0, 0] += 1
        prepared = self.prepared(payload)
        self.assertFalse(prepared['forward_to_backward_cast_value_bridge']['x'])
        with self.assertRaisesRegex(ValueError, 'saved operands differ'):
            replay.execute_route(prepared['prepared'], prepared['metadata'], 'autocast_linear')

    def test_zero_oracle_checks_every_output_integer_bits_and_signed_zero(self):
        for dtype, integer in ((torch.bfloat16, torch.int16), (torch.float32, torch.int32)):
            for word in (1, torch.iinfo(integer).min + 1):
                for name in ('raw_dw', 'dx', 'db', 'dw', 'leaf_db'):
                    outputs = {'raw_dw': torch.zeros(3, 4, dtype=dtype), name: torch.tensor([word], dtype=integer).view(dtype)}
                    checked = replay.scan_outputs(outputs, zero_dy=True, chunk_elements=1)
                    self.assertTrue(checked['failed'])
                    self.assertEqual(checked['outputs'][name]['nonzero_including_nonfinite_count'], 1)
            checked = replay.scan_outputs({'raw_dw': torch.full((3, 4), -0., dtype=dtype)}, zero_dy=True, chunk_elements=1)
            self.assertFalse(checked['failed'])

    def test_first_failing_actual_outputs_preserved_all_routes_without_rerun(self):
        for route in replay.ROUTES:
            for zero in (False, True):
                payload = self.payload()
                prepared = self.prepared(payload, zero)
                calls, pointers, saved_return = [], [], []
                def dispatch(resident, metadata, actual_route):
                    calls.append(1)
                    pointers.append(resident['pristine_inputs']['x'].data_ptr())
                    outputs = replay.execute_route(resident, metadata, actual_route)
                    if len(calls) == 3:
                        outputs['raw_dw'][0, 0] = 1e-30 if zero else 1e30
                        saved_return.append(outputs['raw_dw'].clone())
                    return outputs
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'failure.pt'
                    result = self.run_capture(payload, prepared, path, route=route, repeats=10, function=dispatch)
                    artifact = torch.load(path, weights_only=False, map_location='cpu')
                    self.assertEqual(len(calls), 3)
                    self.assertEqual(len(set(pointers)), 1)
                    self.assertEqual(result['status'], 'FAIL')
                    self.assertEqual(artifact['format'], 'additional_linear_replay_failure_v1')
                    self.assertEqual(artifact['iteration'], 3)
                    self.assertEqual(artifact['transformation']['zero_dy'], zero)
                    self.assertTrue(torch.equal(artifact['current_at_failure']['outputs']['raw_dw'], saved_return[0]))
                    self.assertTrue(torch.equal(artifact['original_capture']['stages']['raw_small']['dmat2'], payload['stages']['raw_small']['dmat2']))
                    self.assertEqual(artifact['prepared_pre_call']['pristine_inputs']['dy'].stride(), (0, 0))

    def test_finite_leaf_conversion_fault_captures_first_mismatch_below_threshold(self):
        for changed, mapping in (('dw', 'weight_matches_raw_transpose_cast'),
                                 ('leaf_db', 'bias_matches_raw_cast')):
            payload = self.payload()
            prepared = self.prepared(payload)
            calls, original_outputs = [], []
            def dispatch(resident, metadata, route):
                calls.append(1)
                outputs = replay.execute_route(resident, metadata, route)
                if len(calls) == 3:
                    value = outputs[changed]
                    value[(0,) * value.ndim] += 1
                    original_outputs.append({key: tensor.clone() for key, tensor in outputs.items()})
                return outputs
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'failure.pt'
                report = self.run_capture(payload, prepared, path, route='autocast_linear',
                                          repeats=8, function=dispatch)
                artifact = torch.load(path, weights_only=False, map_location='cpu')
            self.assertEqual(len(calls), 3)
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(artifact['iteration'], 3)
            record = artifact['observed_record']
            self.assertTrue(record['raw_to_leaf_conversion']['failed'])
            self.assertFalse(record['raw_to_leaf_conversion'][mapping])
            other = 'bias_matches_raw_cast' if changed == 'dw' else 'weight_matches_raw_transpose_cast'
            self.assertTrue(record['raw_to_leaf_conversion'][other])
            self.assertTrue(record['input_bytes_match_prepared'])
            self.assertTrue(all(value['nonfinite_count'] == value['extreme_count'] == 0
                                for value in record['outputs'].values()))
            self.assertFalse(artifact['saved_output_checks']['failed'])
            saved = artifact['current_at_failure']['outputs']
            self.assertTrue(all(torch.equal(saved[name], value) for name, value in original_outputs[0].items()))

    def test_input_mutation_and_final_raw_padding_mutation_preserved(self):
        for raw_only in (False, True):
            payload = self.payload()
            backing = torch.full((7, 11), 17., dtype=torch.bfloat16)
            payload['pristine_inputs']['dy'] = backing[:, 1:9:2]
            payload['pristine_inputs']['dy'].fill_(1)
            prepared = self.prepared(payload)
            calls = []
            def dispatch(resident, metadata, route):
                calls.append(1)
                outputs = replay.execute_route(resident, metadata, route)
                if raw_only:
                    value = resident['pristine_inputs']['dy']
                    direct = torch.empty(0, dtype=torch.bfloat16).set_(value.untyped_storage(), 0, (77,), (1,))
                    direct[0] = 42
                else:
                    resident['pristine_inputs']['x'].add_(1)
                return outputs
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'failure.pt'
                result = self.run_capture(payload, prepared, path, repeats=2, function=dispatch)
                artifact = torch.load(path, weights_only=False, map_location='cpu')
                self.assertEqual(len(calls), 2 if raw_only else 1)
                self.assertFalse(artifact['observed_record']['input_bytes_match_prepared'])
                self.assertIn('raw_dw', artifact['current_at_failure']['outputs'])
                if raw_only:
                    self.assertEqual(artifact['observed_record']['changed_input_versions'], [])
                    self.assertEqual(float(raw(artifact['current_at_failure']['inputs']['pristine_inputs']['dy']).view(torch.bfloat16)[0]), 42)

    def test_original_reference_difference_is_evidence_not_oracle(self):
        payload = self.payload()
        payload['stages']['raw_small']['dmat2'].add_(1)
        prepared = self.prepared(payload)
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_capture(payload, prepared, Path(directory) / 'failure.pt', repeats=1)
            self.assertEqual(result['status'], 'PASS')
            compared = result['iterations'][0]['original_reference_comparisons']['comparisons']['raw_dw']
            self.assertGreater(compared['different_numeric_elements'], 0)

    def test_failure_preserves_configuration_and_runtime_source_evidence(self):
        payload = self.payload()
        prepared = self.prepared(payload, True)
        configuration = {'capture': {'path': 'original.pt', 'sha256': 'a' * 64},
                         'context': {'path': 'context.json', 'sha256': 'b' * 64},
                         'session': {'identity': 'before-dispatch'}}
        expected = copy.deepcopy(configuration)
        def bad(resident, metadata, route):
            configuration['session']['identity'] = 'changed-by-caller'
            return {'raw_dw': torch.ones_like(resident['pristine_inputs']['mat2'])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            result = self.run_capture(payload, prepared, path, repeats=1, function=bad,
                                      configuration=configuration)
            artifact = torch.load(path, weights_only=False, map_location='cpu')
        self.assertEqual(result['runtime']['configuration'], expected)
        runtime = artifact['runtime']
        self.assertEqual(runtime['configuration'], expected)
        self.assertEqual(runtime['device'], 'cpu')
        self.assertTrue(runtime['injected_function'])
        self.assertEqual(runtime['versions']['torch'], str(torch.__version__))
        self.assertEqual(runtime['versions']['hip'], torch.version.hip)
        self.assertEqual(runtime['arguments']['repeats'], 1)
        self.assertEqual(runtime['arguments']['chunk_bytes'], 1024)
        self.assertEqual(runtime['captured_blas_preference'], payload['metadata']['preferred_blas_library'])
        before = runtime['source_sha256_before']
        self.assertEqual({Path(name).name for name in before}, set(replay.SOURCE_FILES))
        self.assertEqual(before, runtime['source_sha256_at_failure'])
        self.assertTrue(all(len(digest) == 64 for digest in before.values()))

    def test_runtime_identity_requires_torch_hip_and_checks_present_versions(self):
        versions = {'torch': 'torch-test', 'hip': 'hip-test', 'triton': 'triton-test',
                    'torch_distribution': 'torch-dist', 'triton_distribution': 'triton-dist',
                    'numpy_distribution': 'numpy-dist'}
        replay.validate_runtime({'torch': versions['torch'], 'hip': versions['hip']}, versions)
        replay.validate_runtime(copy.deepcopy(versions), versions)
        for key in versions:
            context = copy.deepcopy(versions)
            context[key] = 'incorrect-version'
            with self.subTest(mismatched=key), self.assertRaisesRegex(ValueError, key):
                replay.validate_runtime(context, versions)
        for key in ('torch', 'hip'):
            context = copy.deepcopy(versions)
            del context[key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, key):
                replay.validate_runtime(context, versions)

    def test_cpu_cli_inspects_without_dispatch_and_records_provenance(self):
        payload = self.payload()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'capture.pt'
            report_path = Path(directory) / 'report.json'
            torch.save(payload, source)
            with mock.patch.object(replay, 'run_replay') as dispatch, redirect_stdout(io.StringIO()):
                status = replay.main([str(source), '--report', str(report_path), '--cpu-threads', '2'])
            dispatch.assert_not_called()
            self.assertEqual(status, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report['status'], 'CPU_INSPECTED')
            self.assertEqual(report['configuration']['capture']['sha256'], replay.support.file_hash(source))
            self.assertEqual(report['configuration']['versions']['torch'], str(torch.__version__))
            self.assertTrue(report['source_files_unchanged_during_run'])
            self.assertTrue(report['conditional_raw_dw_bound']['applicable'])
            self.assertTrue(all(report['forward_to_backward_cast_value_bridge'].values()))
            self.assertFalse(report_path.with_name(report_path.name + '.tmp').exists())

    def test_cpu_cli_error_is_durably_recorded_before_reraising(self):
        payload = self.payload()
        payload['metadata']['input_provenance']['pristine'] = False
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'capture.pt'
            report_path = Path(directory) / 'report.json'
            torch.save(payload, source)
            with self.assertRaisesRegex(ValueError, 'pristine'), redirect_stdout(io.StringIO()):
                replay.main([str(source), '--report', str(report_path), '--cpu-threads', '2'])
            report = json.loads(report_path.read_text())
            self.assertEqual(report['status'], 'ERROR')
            self.assertEqual(report['error']['type'], 'ValueError')
            self.assertIn('pristine', report['error']['message'])
            self.assertIn('validate_capture', report['error']['traceback'])
            self.assertFalse(report['failure_artifact_exists'])
            self.assertFalse(report_path.with_name(report_path.name + '.tmp').exists())

    def test_cli_never_clobbers_existing_outputs_temps_or_source(self):
        for occupied in ('report.json', 'report.json.tmp', 'failure.pt', 'failure.pt.tmp'):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'capture.pt'
                target = Path(directory) / occupied
                target.write_bytes(b'preserve-existing')
                with self.subTest(occupied=occupied), mock.patch.object(torch, 'load') as load:
                    with self.assertRaises(FileExistsError):
                        replay.main([str(source), '--report', str(Path(directory) / 'report.json'),
                                     '--failure-dump', str(Path(directory) / 'failure.pt')])
                    load.assert_not_called()
                self.assertEqual(target.read_bytes(), b'preserve-existing')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'capture.pt'
            source.write_bytes(b'preserve-source')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                replay.main([str(source), '--report', str(source)])
            self.assertEqual(source.read_bytes(), b'preserve-source')
    def test_budget_and_existing_paths_rejected_before_dispatch(self):
        payload = self.payload()
        prepared = self.prepared(payload, True)
        function = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            for key in ('max_resident_bytes', 'max_capture_bytes'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    replay.run_replay(payload, prepared, failure_path=path, function=function, **{key: 1})
            path.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                replay.run_replay(payload, prepared, failure_path=path, function=function)
            self.assertEqual(path.read_bytes(), b'keep')
        function.assert_not_called()

    def test_failure_write_error_leaves_no_partial_final_or_owned_temp(self):
        payload = self.payload()
        prepared = self.prepared(payload, True)
        def bad(resident, metadata, route):
            return {'raw_dw': torch.ones_like(resident['pristine_inputs']['mat2'])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            with mock.patch.object(replay.common._torch(), 'save', side_effect=RuntimeError('save failed')):
                with self.assertRaisesRegex(RuntimeError, 'save failed'):
                    self.run_capture(payload, prepared, path, repeats=1, function=bad)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + '.tmp').exists())


if __name__ == '__main__':
    unittest.main()
