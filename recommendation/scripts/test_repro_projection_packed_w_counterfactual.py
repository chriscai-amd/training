"""CPU-only identity, environment and retained-failure tests for the W runner."""
import copy
import io
import json
import os
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import repro_projection_packed_w_counterfactual as target
from test_repro_projection_dx_zero import bundle as small_bundle


def fixture():
    value = small_bundle()
    value['provenance'] = {
        'W': 'Derived CPU test fixture',
        'source_bundle': {'sha256': target.SOURCE_BUNDLE_SHA256},
        'source_bundle_provenance': {'type': 'CPU_test_fixture'},
        'exporter': {'sha256': target.EXPORTER_SHA256},
        'counterfactual_intervention': {
            'format': target.INTERVENTION_FORMAT, 'operation': target.INTERVENTION_OPERATION,
        },
    }
    return value


class CounterfactualRunnerTests(unittest.TestCase):
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

    def arguments(self, directory, *extra):
        return ['--bundle', str(directory / 'bundle.pt'), '--report', str(directory / 'report.json'),
                '--failure-dump', str(directory / 'failure.pt'), '--cpu-threads', '2', *extra]

    def test_GPU_environment_gate_precedes_input_load_and_prepare(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for key in target.frozen.MANDATORY_GPU_ENVIRONMENT:
                for bad_value in (None, '1', ''):
                    environment = {name: '0' for name in target.frozen.MANDATORY_GPU_ENVIRONMENT}
                    if bad_value is None:
                        environment.pop(key)
                    else:
                        environment[key] = bad_value
                    with self.subTest(key=key, value=bad_value), \
                            mock.patch.dict(os.environ, environment, clear=True), \
                            mock.patch.object(torch, 'load', side_effect=AssertionError('Early load')), \
                            mock.patch.object(target.frozen, 'prepare', side_effect=AssertionError('Early prepare')), \
                            redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        target.main(self.arguments(directory, '--gpu'))
                    self.assertEqual(caught.exception.code, 2)
            self.assertFalse((directory / 'report.json').exists())

    def test_wrong_bundle_or_frozen_driver_hash_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / 'bundle.pt').write_bytes(b'wrong bundle')
            for bad_identity in ('bundle', 'driver'):
                pin = 'FROZEN_DRIVER_SHA256' if bad_identity == 'driver' else 'COUNTERFACTUAL_BUNDLE_SHA256'
                message = 'Frozen DX driver' if bad_identity == 'driver' else 'Counterfactual bundle'
                with self.subTest(identity=bad_identity), mock.patch.object(target, pin, 'bad'), \
                        mock.patch.object(torch, 'load', side_effect=AssertionError('Early load')), \
                        self.assertRaisesRegex(ValueError, message):
                    target.main(self.arguments(directory))
            self.assertFalse((directory / 'report.json').exists())

    def test_derived_provenance_rejected_before_full_DZ_prepare(self):
        original = fixture()
        changes = [
            lambda b: b.update(w_storage_sha256='bad'),
            lambda b: b['provenance']['counterfactual_intervention'].update(format='bad'),
            lambda b: b['provenance']['counterfactual_intervention'].update(operation='bad'),
            lambda b: b['provenance']['source_bundle'].update(sha256='bad'),
            lambda b: b['provenance']['exporter'].update(sha256='bad'),
            lambda b: b['provenance'].update(source_bundle_provenance={}),
        ]
        with mock.patch.object(target, 'COUNTERFACTUAL_W_SHA256', original['w_storage_sha256']):
            target.validate_counterfactual_bundle(original)
            for change in changes:
                value = copy.deepcopy(original)
                change(value)
                with self.assertRaises(ValueError):
                    target.validate_counterfactual_bundle(value)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            torch.save(value, directory / 'bundle.pt')
            digest = target.frozen.support.file_hash(directory / 'bundle.pt')
            with mock.patch.object(target, 'COUNTERFACTUAL_BUNDLE_SHA256', digest), \
                    mock.patch.object(target, 'COUNTERFACTUAL_W_SHA256', value['w_storage_sha256']), \
                    mock.patch.object(target.frozen, 'prepare', side_effect=AssertionError('Early prepare')), \
                    self.assertRaisesRegex(ValueError, 'source provenance'):
                target.main(self.arguments(directory))

    def test_CPU_main_uses_frozen_prepare_and_retains_derived_report(self):
        value = fixture()
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            torch.save(value, directory / 'bundle.pt')
            digest = target.frozen.support.file_hash(directory / 'bundle.pt')
            with mock.patch.object(target, 'COUNTERFACTUAL_BUNDLE_SHA256', digest), \
                    mock.patch.object(target, 'COUNTERFACTUAL_W_SHA256', value['w_storage_sha256']), \
                    mock.patch.object(target.frozen, 'prepare', wraps=target.frozen.prepare) as prepare, \
                    mock.patch.object(target.frozen, 'run', side_effect=AssertionError('CPU main dispatched')):
                code = target.main(self.arguments(directory, '--repeats', '3'))
            report = json.loads((directory / 'report.json').read_text())
            self.assertEqual((code, report['status']), (0, 'CPU_PREPARED'))
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs, {})
            self.assertEqual(report['bundle_provenance'], value['provenance'])
            self.assertEqual(report['counterfactual_intervention'], value['provenance']['counterfactual_intervention'])
            self.assertEqual(report['limits'], target.LIMITS)
            self.assertTrue(report['source_files_unchanged'])
            self.assertIn('scripts/repro_projection_packed_w_counterfactual.py', report['source_sha256_before'])
            self.assertEqual(report['source_sha256_before']['scripts/repro_projection_dx_zero.py'], target.FROZEN_DRIVER_SHA256)
            self.assertFalse((directory / 'failure.pt').exists())

    def test_frozen_run_handoff_retains_first_failure_with_derived_configuration(self):
        value = fixture()
        real_run = target.frozen.run
        prepared_objects, handed_off, calls = [], [], []
        real_prepare = target.frozen.prepare
        def prepare(bundle):
            prepared = real_prepare(bundle)
            prepared_objects.append(prepared)
            return prepared
        def mm(dz, transposed_w):
            calls.append(1)
            output = torch.mm(dz, transposed_w)
            if len(calls) == 2:
                output[3, 2] = 0.03125
            return output
        def cpu_handoff(prepared, **kwargs):
            handed_off.append((prepared, kwargs.copy()))
            kwargs['backend'] = None
            return real_run(prepared, device='cpu', function=mm, **kwargs)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            torch.save(value, directory / 'bundle.pt')
            digest = target.frozen.support.file_hash(directory / 'bundle.pt')
            environment = {name: '0' for name in target.frozen.MANDATORY_GPU_ENVIRONMENT}
            environment['TENSILE_STREAMK_FIXED_GRID'] = '460'
            with mock.patch.object(target, 'COUNTERFACTUAL_BUNDLE_SHA256', digest), \
                    mock.patch.object(target, 'COUNTERFACTUAL_W_SHA256', value['w_storage_sha256']), \
                    mock.patch.dict(os.environ, environment), \
                    mock.patch.object(torch.cuda, 'get_device_name', return_value='CPU mock, no GPU'), \
                    mock.patch.object(target.frozen.support, 'blas_preference', return_value='CPU mock'), \
                    mock.patch.object(target.frozen.support, 'loaded_blas_libraries', return_value=[]), \
                    mock.patch.object(target.frozen, 'prepare', side_effect=prepare), \
                    mock.patch.object(target.frozen, 'run', side_effect=cpu_handoff):
                code = target.main(self.arguments(directory, '--gpu', '--repeats', '3'))
            report = json.loads((directory / 'report.json').read_text())
            artifact = torch.load(directory / 'failure.pt', weights_only=False, map_location='cpu')
            self.assertEqual((code, report['status'], report['iterations_completed'], len(calls)), (1, 'FAIL', 2, 2))
            self.assertEqual(len(handed_off), 1)
            self.assertIs(handed_off[0][0], prepared_objects[0])
            self.assertEqual(set(handed_off[0][1]), {'failure_path', 'repeats', 'backend', 'configuration', 'progress'})
            self.assertEqual(handed_off[0][1]['backend'], 'hipblaslt')
            self.assertEqual(artifact['bundle_provenance'], value['provenance'])
            configuration = artifact['configuration']
            self.assertEqual(configuration['limits'], target.LIMITS)
            self.assertEqual(configuration['counterfactual_intervention'], value['provenance']['counterfactual_intervention'])
            self.assertEqual(configuration['source_sha256_before'], report['source_sha256_before'])
            self.assertEqual(configuration['environment']['TENSILE_STREAMK_FIXED_GRID'], '460')
            self.assertTrue(artifact['input_backing_bytes_unchanged'])
            self.assertTrue(artifact['saved_DX_check_matches_observed'])
            self.assertEqual(artifact['current_at_failure']['dx'][3, 2], 0.03125)


if __name__ == '__main__':
    unittest.main()
