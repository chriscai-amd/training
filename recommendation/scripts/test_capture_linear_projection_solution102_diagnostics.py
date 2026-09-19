"""File-only tests; production worker code is exercised with trainer stubs.

No Torch import, native loading, GPU execution or subprocess is needed.
Existing numerical queue behavior is covered by its frozen test suite.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent


def scoped(function, **overrides):
    result = types.FunctionType(function.__code__, {**function.__globals__, **overrides},
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class QueueSentinel:
    pass


class ProbeSentinel:
    def __init__(self, *args, **kwargs):
        self.frame = {'capture_variant': VARIANT, 'metadata': {}}
        self.scalar_queue = QueueSentinel()
        self.stage_save_calls = 0

    def set_attempt(self, *args):
        return self.scalar_queue

    def save_frame(self, trigger):
        self.stage_save_calls += 1
        self.frame['capture_variant'] = VARIANT
        self.frame['scalar_queue_diagnostics'] = {'unchanged_sentinel': self.scalar_queue}
        return self.frame


def fake_configure(options, environment):
    verify_sources()
    environment['NAN_TRAINING_OPTIONS'] = json.dumps({**options, 'monitor_variant': VARIANT,
        'scalar_queue_diagnostics_policy': {'frozen': True}, 'mitigation_control': {'frozen': True}})


def fake_validate(options, environment):
    verify_sources()
    if (options.get('monitor_variant') != VARIANT
            or options.get('scalar_queue_diagnostics_policy') != {'frozen': True}
            or options.get('mitigation_control') != {'frozen': True}):
        raise ValueError('Lost frozen worker options')


def load_target():
    training = types.SimpleNamespace(_archive_sources=lambda directory, context: {'frozen_archive': True})
    base = types.SimpleNamespace(_scoped=scoped, BoundaryProbeError=RuntimeError, base=training,
                                frozen=types.SimpleNamespace(main=lambda argv=None: None))
    capture = types.ModuleType('capture_linear_projection_deferred')
    capture.base, capture.LIMITS, capture.LinearProjectionProbe = base, [], object
    path = ROOT / 'capture_linear_projection_mitigated.py'
    spec = importlib.util.spec_from_file_location('_solution102_frozen_worker_fixture', path)
    mitigated = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {capture.__name__: capture}):
        spec.loader.exec_module(mitigated)
    stages = types.ModuleType('capture_linear_projection_diagnostics')
    stages.base, stages.mitigated = base, mitigated
    stages.VARIANT, stages.POLICY, stages.LIMITS = 'frozen_variant', {'frozen': True}, []
    stages.DiagnosticScalarQueue, stages.ProjectionProbe = QueueSentinel, ProbeSentinel
    stages.verify_sources = lambda: {}
    stages.configure_environment, stages.validate_worker = fake_configure, fake_validate
    stages.split_capture = lambda payload: payload
    spec = importlib.util.spec_from_file_location('_solution102_wrapper_under_test', ROOT / 'capture_linear_projection_solution102_diagnostics.py')
    target = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {stages.__name__: stages}):
        spec.loader.exec_module(target)
    return target


def forbid_native(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {'torch', 'triton', 'ctypes'}:
        raise AssertionError('Native/GPU library import forbidden in file-only tests')
    if event.startswith(('ctypes.', 'subprocess.')) or event in {'os.system', 'os.fork', 'os.posix_spawn'}:
        raise AssertionError('Native/process action forbidden in file-only tests')


sys.addaudithook(forbid_native)
target = load_target()


class PrivateCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.installed, self.private = self.root / 'installed', self.root / 'private'
        self.installed.mkdir()
        self.private.mkdir()
        self.target_name = target.TARGET_CATALOG_FILENAME
        names = ['TensileLibrary_lazy_gfx1250.dat.zlib', self.target_name.replace('.dat.zlib', '.co'), self.target_name]
        rows = []
        for index, name in enumerate(names):
            original, chosen = self.installed / name, self.private / name
            original.write_bytes(('original' + str(index)).encode())
            kind = 'original_symlink'
            if name == self.target_name:
                chosen.write_bytes(b'derivative')
                kind = 'derived_regular_file'
            else:
                chosen.symlink_to(original)
            row = {'name': name, 'kind': kind, 'original': target.file_pin(original), 'private': target.file_pin(chosen)}
            if kind == 'original_symlink':
                row['symlink_target'] = str(original)
            rows.append(row)
        source = self.root / 'prepare.py'
        source.write_text('# static fixture\n')
        self.manifest = {'schema_version': 1, 'private_library_path': str(self.private),
            'installed_library_path': str(self.installed), 'target_catalog_filename': self.target_name,
            'inventory': rows, 'catalog': {'derived': rows[2]['private']},
            'master': {k: rows[0][k] for k in ('original', 'private')},
            'native_dw_code_object': {k: rows[1][k] for k in ('original', 'private')},
            'preparation_source': target.file_pin(source)}
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        required = {**target.stages.mitigated.REQUIRED, 'HIPBLASLT_TENSILE_LIBPATH': str(self.private)}
        patcher = mock.patch.multiple(target, PRIVATE_LIBRARY_PATH=str(self.private),
            INSTALLED_LIBRARY_PATH=str(self.installed), MANIFEST_PATH=str(self.manifest_path),
            MANIFEST_SHA256=target.file_pin(self.manifest_path)['sha256'],
            DERIVED_SHA256=self.manifest['catalog']['derived']['sha256'], ENTRY_COUNT=3, REQUIRED=required)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.env = dict(required)

    def configured(self):
        env = dict(self.env)
        target.configure_environment({'directory': str(self.root / 'run'), 'steps': 2}, env)
        return env, json.loads(env['NAN_TRAINING_OPTIONS'])

    def worker_modules(self, bootstrap, trainer_main=None):
        gin = types.ModuleType('gin')
        gin.parse_config_file = lambda *args, **kwargs: None
        boot = types.ModuleType('generative_recommenders.dlrm_v4.train._env_bootstrap')
        boot.apply_env_bootstrap = bootstrap
        utils = types.SimpleNamespace(streaming_train_eval_loop=lambda **kwargs: kwargs)
        trainer = types.SimpleNamespace(_main_func=trainer_main or (lambda *args: None))
        modules = {'generative_recommenders.dlrm_v4.train.utils': utils,
                   'generative_recommenders.dlrm_v4.train.train_ranker': trainer}
        return gin, boot, utils, trainer, modules

    def run_worker(self):
        return target.diagnostic_worker(0, 1, 0, 1, 'localhost', 23456, 'unused.gin', 'streaming-train-eval')

    def test_complete_inventory_and_frozen_queue_identity(self):
        result = target.verify_private_catalog(self.env)
        self.assertEqual(result['verified_inventory_files'], 3)
        self.assertEqual((result['original_symlinks'], result['derived_regular_files']), (2, 1))
        self.assertIs(target.DiagnosticScalarQueue, target.stages.DiagnosticScalarQueue)
        self.assertIs(target.ProjectionProbe.set_attempt, target.stages.ProjectionProbe.set_attempt)
        target.verify_sources()  # Checks the three actual frozen source hashes without importing them.

    def test_manifest_tamper_rejected(self):
        self.manifest_path.write_text(self.manifest_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
            target.verify_private_catalog(self.env)

    def test_derived_catalog_tamper_rejected(self):
        (self.private / self.target_name).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            target.verify_private_catalog(self.env)

    def test_original_native_object_tamper_rejected(self):
        Path(self.manifest['native_dw_code_object']['original']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            target.verify_private_catalog(self.env)

    def test_same_bytes_wrong_symlink_target_rejected(self):
        chosen = Path(self.manifest['master']['private']['path'])
        alternate = self.root / 'alternate'
        alternate.write_bytes(chosen.read_bytes())
        chosen.unlink()
        chosen.symlink_to(alternate)
        with self.assertRaisesRegex(ValueError, 'symlink identity'):
            target.verify_private_catalog(self.env)

    def test_uncompressed_shadow_or_unlisted_entry_rejected(self):
        (self.private / self.target_name.removesuffix('.zlib')).write_bytes(b'shadow')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            target.verify_private_catalog(self.env)

    def test_catalog_directory_symlink_rejected(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.private, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'private HIPBLASLT'):
            target.verify_private_catalog(dict(self.env, HIPBLASLT_TENSILE_LIBPATH=str(alias)))

    def test_configuration_rejects_selector_overrides_without_mutation(self):
        for name in target.ABSENT:
            env = dict(self.env, **{name: '0'})
            before = copy.deepcopy(env)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unset'):
                target.configure_environment({}, env)
            self.assertEqual(env, before)

    def test_configuration_and_worker_options_record_actual_catalog(self):
        env, options = self.configured()
        self.assertEqual(target.validate_worker(options, env), options[target.CONTROL_KEY])
        self.assertEqual(options['monitor_variant'], target.VARIANT)
        self.assertEqual(options['mitigation_control'], {'frozen': True})
        bad = copy.deepcopy(options)
        bad[target.CONTROL_KEY]['derived_catalog']['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            target.validate_worker(bad, dict(env, NAN_TRAINING_OPTIONS=json.dumps(bad)))

    def test_real_frozen_worker_rejects_path_drift_after_bootstrap_before_trainer_import(self):
        env, _ = self.configured()
        def bootstrap():
            os.environ['HIPBLASLT_TENSILE_LIBPATH'] = str(self.installed)
        gin, boot, utils, trainer, modules = self.worker_modules(bootstrap)
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.dict(sys.modules, {'gin': gin, boot.__name__: boot}), \
                mock.patch.object(target.stages.mitigated.importlib, 'import_module', side_effect=modules.__getitem__) as imports:
            with self.assertRaisesRegex(ValueError, 'requires HIPBLASLT_TENSILE_LIBPATH'):
                self.run_worker()
            imports.assert_not_called()

    def test_real_frozen_worker_rejects_catalog_tamper_after_bootstrap(self):
        env, _ = self.configured()
        gin, boot, utils, trainer, modules = self.worker_modules(lambda: (self.private / self.target_name).write_bytes(b'changed'))
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.dict(sys.modules, {'gin': gin, boot.__name__: boot}), \
                mock.patch.object(target.stages.mitigated.importlib, 'import_module', side_effect=modules.__getitem__) as imports:
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.run_worker()
            imports.assert_not_called()

    def test_real_frozen_worker_two_checks_and_loop_restoration(self):
        env, _ = self.configured()
        checks = []
        original_validate = target.validate_worker
        def validate(options, environment):
            result = original_validate(options, environment)
            checks.append(result)
            return result
        gin, boot, utils, trainer, modules = self.worker_modules(lambda: None, lambda *args: (_ for _ in ()).throw(RuntimeError('trainer sentinel')))
        original_loop = utils.streaming_train_eval_loop
        frozen_globals = dict(target.stages.mitigated.diagnostic_worker.__globals__)
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.dict(sys.modules, {'gin': gin, boot.__name__: boot}), \
                mock.patch.object(target, 'validate_worker', side_effect=validate), \
                mock.patch.object(target.stages.mitigated.importlib, 'import_module', side_effect=modules.__getitem__):
            with self.assertRaisesRegex(RuntimeError, 'trainer sentinel'):
                self.run_worker()
        self.assertEqual(len(checks), 2)
        self.assertEqual(checks[0], checks[1])
        self.assertIs(utils.streaming_train_eval_loop, original_loop)
        self.assertEqual(frozen_globals, target.stages.mitigated.diagnostic_worker.__globals__)

    def test_frame_and_context_retain_catalog_and_bootstrap_records(self):
        control = target.verify_private_catalog(self.env)
        records = [{'phase': phase, 'control': control} for phase in ('before_bootstrap', 'after_bootstrap')]
        context = {'source_sha256': {}}
        with mock.patch.dict(os.environ, self.env, clear=True):
            probe = target.ProjectionProbe(catalog_bootstrap_verifications=records)
            queue = probe.scalar_queue
            frame = probe.save_frame({'sentinel': True})
            archive = target.archive_sources(self.root, context, records)
        self.assertEqual(probe.stage_save_calls, 1)
        self.assertIs(frame['scalar_queue_diagnostics']['unchanged_sentinel'], queue)
        self.assertEqual(frame['metadata'][target.CONTROL_KEY], control)
        self.assertEqual(frame['metadata']['private_catalog_bootstrap_verifications'], records)
        self.assertEqual(context[target.CONTROL_KEY], control)
        self.assertEqual(context['private_catalog_bootstrap_verifications'], records)
        self.assertEqual((self.root / 'dw_solution102_private_catalog_manifest.json').read_bytes(), self.manifest_path.read_bytes())
        self.assertEqual(archive, {'frozen_archive': True})
        self.assertEqual(target.split_capture(frame)['capture_variant'], target.stages.VARIANT)

    def test_missing_bootstrap_record_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'both bootstrap'):
            target.ProjectionProbe(catalog_bootstrap_verifications=[])


if __name__ == '__main__':
    unittest.main()
