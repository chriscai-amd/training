"""CPU validation of actual storage/provenance replay semantics and failure paths."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from nan_backward_training_extended import _argument_layouts
import replay_training_readwrite_boundary as replay


def fixture(operation=replay.ATTENTION, *, pristine=True, stage='input'):
    if operation == replay.ATTENTION:
        storage = torch.arange(64, dtype=torch.float64) / 32
        inputs = storage[3:21].view(3, 6)
        writes = torch.full((64,), float('nan'), dtype=torch.float64)[4:22].view(3, 6)
        kwargs = {'dout': torch.ones(3, 1, 2, dtype=torch.float64),
                  'q': inputs[:, :2].view(3, 1, 2), 'k': inputs[:, 2:4].view(3, 1, 2),
                  'v': inputs[:, 4:6].view(3, 1, 2),
                  'dq': writes[:, :2].view(3, 1, 2), 'dk': writes[:, 2:4].view(3, 1, 2),
                  'dv': writes[:, 4:6].view(3, 1, 2), 'seq_offsets': torch.tensor([0, 2, 3]),
                  'num_targets': torch.tensor([1, 1]), 'N': 2, 'alpha': 0.5,
                  'max_attn_len': 0, 'contextual_seq_len': 0,
                  'sort_by_length_indices': torch.tensor([0, 1]), 'enable_tma': False, 'num_softmax_heads': 0}
        function = attention
    else:
        kwargs = {'grad_output': torch.arange(12, dtype=torch.float64)[1::2].view(3, 2),
                  'self': torch._neg_view(torch.arange(20, dtype=torch.float64)[2:14:2].view(3, 2) / 5),
                  'grad_input': torch.full((20,), float('nan'), dtype=torch.float64)[3:15:2].view(3, 2)}
        function = torch.ops.aten.silu_backward
    outputs = None
    if stage != 'input':
        if pristine:
            fresh, _ = replay.common.restore_raw_tree(kwargs, device='cpu')
            function(**fresh)
            outputs = tuple(fresh[name] for name in replay.WRITES[operation])
        else:
            function(**kwargs)
            outputs = tuple(kwargs[name] for name in replay.WRITES[operation])
    controls = replay.helpers.execution_controls()
    event = {'operation': operation, 'input_snapshot_pristine': pristine, 'initial_destination_bytes_retained': pristine,
             'operation_executed': stage != 'input', 'destination_initial_values_scanned': False,
             'argument_roles': {'read': [name for name in kwargs if name not in replay.WRITES[operation]],
                                'write_only': list(replay.WRITES[operation])},
             'argument_layouts': _argument_layouts(kwargs), 'execution_controls': controls}
    if operation == replay.ATTENTION:
        event['scalar_arguments'] = {name: kwargs[name] for name in replay.SCALARS}
    return {'format': replay.FORMAT, 'format_version': 1, 'operation': operation,
            'stage': stage, 'probe_mode': 'pristine' if pristine else 'monitor_on_anomaly',
            'input_snapshot_pristine': pristine, 'initial_destination_bytes_retained': pristine,
            'execution_controls': controls, 'args': (), 'kwargs': kwargs, 'outputs': outputs, 'event': event}


def attention(**values):
    for index, name in enumerate(replay.WRITES[replay.ATTENTION]):
        values[name].fill_(index + 1)


class ReadwriteReplayTest(unittest.TestCase):
    def test_attention_fresh_full_storages_preserve_aliases_offsets_and_padding(self):
        payload = fixture(stage='output')
        seen = []
        saved = replay.helpers.input_digests(payload['kwargs'], chunk_bytes=31, threshold=1e6)

        def execute(**values):
            seen.append(values)
            self.assertEqual(values['dq'].untyped_storage()._cdata, values['dk'].untyped_storage()._cdata)
            self.assertEqual(values['dk'].untyped_storage()._cdata, values['dv'].untyped_storage()._cdata)
            self.assertEqual(values['q'].storage_offset(), 3)
            self.assertEqual(values['q'].stride(), payload['kwargs']['q'].stride())
            self.assertEqual(values['q'].untyped_storage().nbytes(), 64 * 8)
            self.assertTrue(torch.isnan(values['dq']).all())
            attention(**values)

        reports = replay.replay_capture(payload, device='cpu', function=execute, repeats=2, chunk_bytes=31)
        self.assertNotEqual(seen[0]['dq'].untyped_storage()._cdata, seen[1]['dq'].untyped_storage()._cdata)
        for report in reports:
            self.assertTrue(report['matches_prepared_argument_bytes_and_layouts'])
            self.assertTrue(report['matches_original_capture_argument_bytes_and_layouts'])
            self.assertTrue(report['read_logical_values_unchanged'])
            self.assertFalse(report['all_argument_backing_bytes_unchanged'])
            self.assertEqual(report['matches_captured_logical_bytes'], [True] * 3)
            self.assertFalse(report['output_anomaly'])
        self.assertEqual(reports[1]['matches_first_repeat_logical_bytes'], [True] * 3)
        self.assertEqual(saved, replay.helpers.input_digests(payload['kwargs'], chunk_bytes=31, threshold=1e6))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_silu_cpu_api_preserves_negative_view_and_noncontiguous_destination(self):
        payload = fixture(replay.SILU, stage='output')
        seen = []

        def execute(**values):
            seen.append(values)
            self.assertTrue(values['self'].is_neg())
            self.assertFalse(values['grad_input'].is_contiguous())
            return torch.ops.aten.silu_backward(**values)

        reports = replay.replay_capture(payload, device='cpu', function=execute, repeats=2, chunk_bytes=33)
        self.assertTrue(all(report['matches_captured_logical_bytes'] == [True] for report in reports))
        self.assertTrue(all(report['read_logical_values_unchanged'] for report in reports))
        self.assertNotEqual(seen[0]['grad_input'].data_ptr(), seen[1]['grad_input'].data_ptr())

    def test_postcall_rejected_before_operation_import_and_explicitly_labeled(self):
        payload = fixture(pristine=False, stage='output')
        with patch.object(replay.importlib, 'import_module', side_effect=AssertionError('GPU import')):
            with self.assertRaisesRegex(ValueError, '--allow-postcall-inputs'):
                replay.replay_capture(payload)
        reports = replay.replay_capture(payload, allow_postcall_inputs=True, device='cpu', function=attention, repeats=1)
        self.assertFalse(reports[0]['input_snapshot_pristine'])
        self.assertFalse(reports[0]['initial_destination_bytes_retained'])
        self.assertIn('conditional', reports[0]['replay_interpretation'])
        self.assertEqual(reports[0]['matches_captured_logical_bytes'], [True] * 3)

    def test_certified_monitor_input_and_all_pristine_stages(self):
        payload = fixture()
        payload['probe_mode'] = 'monitor_on_anomaly'
        self.assertTrue(replay.validate_capture(payload)[2]['input_snapshot_pristine'])
        for operation in (replay.ATTENTION, replay.SILU):
            for stage in ('input', 'output', 'finite'):
                with self.subTest(operation=operation, stage=stage):
                    replay.validate_capture(fixture(operation, stage=stage))

    def test_rejects_inconsistent_provenance_roles_aliases_and_scalars(self):
        mutations = [
            lambda p: p.update(format='other'),
            lambda p: p['event'].update(operation='hstu_silu_bwd'),
            lambda p: p['event'].update(input_snapshot_pristine=False),
            lambda p: p['event'].update(operation_executed=True),
            lambda p: p['event']['argument_roles'].update(write_only=['dq']),
            lambda p: p['event']['argument_layouts']['dq'].update(storage_offset=0),
            lambda p: p['event']['argument_layouts']['dk'].update(storage_group=999),
            lambda p: p['event']['scalar_arguments'].update(N=3),
            lambda p: p['kwargs'].update(N=1),
            lambda p: p['kwargs'].update(sort_by_length_indices=torch.tensor([0, 0])),
            lambda p: p['kwargs'].update(num_targets=torch.tensor([3, 1])),
        ]
        for change in mutations:
            payload = fixture()
            change(payload)
            with self.subTest(change=change), self.assertRaises(ValueError):
                replay.validate_capture(payload)
        payload = fixture(pristine=False, stage='output')
        payload['outputs'] = tuple(value.clone() for value in payload['outputs'])
        with self.assertRaisesRegex(ValueError, 'alias'):
            replay.validate_capture(payload, allow_postcall_inputs=True)

    def test_corrupt_restored_padding_stops_before_execution(self):
        payload = fixture()
        restore = replay.common.restore_raw_tree

        def corrupt(*args, **kwargs):
            values, size = restore(*args, **kwargs)
            raw = torch.empty(0, dtype=torch.float64).set_(values['q'].untyped_storage(), 0, (64,), (1,))
            raw[0] += 1
            return values, size

        with patch.object(replay.common, 'restore_raw_tree', side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, 'hashes or layouts differ'):
                replay.replay_capture(payload, device='cpu', function=lambda **kw: self.fail('operation executed'))

    def test_read_mutation_is_reported_and_execution_controls_restore_after_failure(self):
        payload = fixture()
        previous = replay.helpers.execution_controls()
        changed = copy.deepcopy(previous)
        changed['autocast']['cpu'] = {'enabled': True, 'dtype': 'torch.bfloat16'}
        payload['execution_controls'] = payload['event']['execution_controls'] = changed
        progress = []

        def corrupt(**values):
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            attention(**values)
            values['q'].add_(1)

        with self.assertRaisesRegex(RuntimeError, 'changed captured read-input'):
            replay.replay_capture(payload, device='cpu', function=corrupt, progress=progress.append)
        self.assertFalse(progress[-1]['attempts'][0]['read_logical_values_unchanged'])
        self.assertEqual(replay.helpers.execution_controls(), previous)
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            replay.replay_capture(payload, device='cpu', function=lambda **kw: (_ for _ in ()).throw(RuntimeError('injected failure')))
        self.assertEqual(replay.helpers.execution_controls(), previous)

    def test_destination_contract_and_output_anomaly(self):
        with self.assertRaisesRegex(RuntimeError, 'unexpectedly returned'):
            replay.replay_capture(fixture(), device='cpu', function=lambda **kw: torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, 'original destination'):
            replay.replay_capture(fixture(replay.SILU), device='cpu', function=lambda **kw: torch.zeros(3, 2))

        def bad(**values):
            attention(**values)
            values['dq'][0, 0, 0] = float('nan')
            values['dv'][1, 0, 1] = 1e30

        report = replay.replay_capture(fixture(), device='cpu', function=bad, repeats=1)[0]
        self.assertTrue(report['output_anomaly'])
        self.assertEqual(report['outputs']['dq']['nonfinite_count'], 1)
        self.assertEqual(report['outputs']['dv']['extreme_count'], 1)
        self.assertIsNone(report['matches_captured_logical_bytes'])

    def test_cpu_cli_roundtrip_and_no_gpu_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            capture, report = path / 'capture.pt', path / 'report.json'
            torch.save(fixture(), capture)
            with patch.object(replay.importlib, 'import_module', side_effect=AssertionError('GPU import')), redirect_stdout(io.StringIO()):
                replay.main([str(capture), '--report', str(report), '--chunk-mib', '1'])
            saved = json.loads(report.read_text())
            self.assertEqual(saved['operation'], replay.ATTENTION)
            self.assertIn('not requested', saved['replay_status'])
            self.assertTrue(saved['provenance']['input_snapshot_pristine'])
            self.assertFalse(torch.cuda.is_initialized())

    def test_actual_extended_and_all3_generated_capture_formats(self):
        from nan_backward_boundaries import BoundaryAnomalyError
        from nan_backward_training_all_layers import AllLayersBatchedTrainingBackwardBoundaryProbe
        from nan_backward_training_extended import ExtendedTrainingBackwardBoundaryProbe
        from test_nan_backward_training_extended import Layer, make_full_modules
        from test_nan_backward_training_upstream import Stack
        for use_all3, operation, mode in ((False, replay.ATTENTION, 'pristine'),
                                         (False, replay.SILU, 'pristine'),
                                         (True, replay.ATTENTION, 'monitor_on_anomaly'),
                                         (True, replay.SILU, 'monitor_on_anomaly')):
            with self.subTest(all3=use_all3, operation=operation), tempfile.TemporaryDirectory() as directory:
                compute, preprocess, output, state = make_full_modules()
                model = Stack(compute)
                setattr(state, 'extreme_attention' if operation == replay.ATTENTION else 'extreme_silu', True)
                environment = {'NAN_BACKWARD_TARGET': '*', 'NAN_BACKWARD_SAVE_ALL': '0',
                               'NAN_BACKWARD_ABS_THRESHOLD': '1e20', 'NAN_BACKWARD_CHUNK_MIB': '1',
                               'NAN_BACKWARD_MAX_CAPTURE_GIB': '32'}
                with patch.dict(os.environ, environment):
                    cls = AllLayersBatchedTrainingBackwardBoundaryProbe if use_all3 else ExtendedTrainingBackwardBoundaryProbe
                    kwargs = {} if use_all3 else {'selected_operations': (operation,)}
                    probe = cls(model, directory, mode=mode, preprocess_module=preprocess,
                                compute_module=compute, output_module=output, **kwargs)
                    try:
                        probe.set_attempt('training', 0, 1)
                        x = (torch.arange(12, dtype=torch.float64).reshape(3, 4) / 64).requires_grad_()
                        with self.assertRaises(BoundaryAnomalyError):
                            model(x).backward(torch.full_like(x, .5))
                    finally:
                        probe.close()
                path, = Path(directory).glob('*.pt')
                payload = torch.load(path, map_location='cpu', weights_only=False)
                values, reads, provenance = replay.validate_capture(payload, allow_postcall_inputs=use_all3)
                self.assertEqual(payload['operation'], operation)
                self.assertEqual(provenance['input_snapshot_pristine'], not use_all3)
                function = attention if operation == replay.ATTENTION else torch.ops.aten.silu_backward
                result = replay.replay_capture(payload, allow_postcall_inputs=use_all3, device='cpu',
                                                function=function, repeats=1, chunk_bytes=127)
                self.assertTrue(result[0]['read_logical_values_unchanged'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_zero_oracle_preserves_original_k_and_every_unused_backing_byte(self):
        payload = fixture()
        original = replay.argument_digests(payload['kwargs'], chunk_bytes=31, threshold=1e6)
        prepared = replay.prepare_arguments(payload, attention_zero_oracle=True, chunk_bytes=31)
        values = prepared['values']
        self.assertTrue(prepared['provenance']['arguments_transformed'])
        self.assertEqual(original['inputs']['k'], prepared['prepared_argument_digests']['inputs']['k'])
        zeroed = set(prepared['transformation']['zeroed_logical_arguments'])
        modified_bytes = {}
        for name in zeroed:
            source = payload['kwargs'][name]
            identity = source.untyped_storage()._cdata
            addresses = modified_bytes.setdefault(identity, set())
            for coordinate in itertools.product(*(range(size) for size in source.shape)):
                offset = source.storage_offset() + sum(index * stride for index, stride in zip(coordinate, source.stride()))
                addresses.update(range(offset * source.element_size(), (offset + 1) * source.element_size()))
            self.assertEqual(int(torch.count_nonzero(values[name])), 0)
        seen = set()
        for name, source in payload['kwargs'].items():
            if not isinstance(source, torch.Tensor) or source.untyped_storage()._cdata in seen:
                continue
            identity = source.untyped_storage()._cdata
            seen.add(identity)
            size = source.untyped_storage().nbytes()
            before = torch.empty(0, dtype=torch.uint8).set_(source.untyped_storage(), 0, (size,), (1,))
            after = torch.empty(0, dtype=torch.uint8).set_(values[name].untyped_storage(), 0, (size,), (1,))
            protected = torch.tensor([index not in modified_bytes.get(identity, set()) for index in range(size)])
            self.assertTrue(torch.equal(before[protected], after[protected]))
        self.assertEqual(original, replay.argument_digests(payload['kwargs'], chunk_bytes=31, threshold=1e6))

    def test_zero_oracle_accepts_signed_zero_and_stops_on_tiny_nonzero_or_nan(self):
        payload = fixture(stage='output')

        def zero(**values):
            for name in replay.WRITES[replay.ATTENTION]:
                values[name].zero_()
            values['dk'][0, 0, 0] = -0.0

        reports = replay.replay_capture(payload, attention_zero_oracle=True, device='cpu', function=zero, repeats=2)
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(not report['zero_oracle_violation'] for report in reports))
        self.assertTrue(all(report['matches_prepared_argument_bytes_and_layouts'] for report in reports))
        self.assertTrue(all(not report['matches_original_capture_argument_bytes_and_layouts'] for report in reports))
        self.assertTrue(all(report['matches_captured_logical_bytes'] is None for report in reports))
        for value in (1e-200, float('nan'), float('inf')):
            calls = []

            def corrupt(**values):
                calls.append(1)
                zero(**values)
                values['dq'][0, 0, 0] = value

            reports = replay.replay_capture(payload, attention_zero_oracle=True, device='cpu',
                                            function=corrupt, repeats=3, threshold=1e6)
            self.assertEqual(len(calls), 1)
            self.assertTrue(reports[0]['zero_oracle_violation'])
            self.assertTrue(reports[0]['output_anomaly'])
            self.assertEqual(reports[0]['outputs']['dq']['nonzero_count_including_nonfinite'], 1)
            self.assertEqual(reports[0]['outputs']['dq']['extreme_count'], 0)

    def test_zero_oracle_rejects_nonfinite_k_softmax_and_destructive_aliases(self):
        for value in (float('nan'), float('inf'), 1e300):
            payload = fixture()
            payload['kwargs']['k'][0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, 'finite K'):
                replay.prepare_arguments(payload, attention_zero_oracle=True)
        payload = fixture()
        payload['kwargs']['num_softmax_heads'] = payload['event']['scalar_arguments']['num_softmax_heads'] = 1
        with self.assertRaisesRegex(ValueError, 'num_softmax_heads=0'):
            replay.prepare_arguments(payload, attention_zero_oracle=True)
        payload = fixture()
        payload['kwargs']['k'] = payload['kwargs']['q']
        payload['event']['argument_layouts'] = _argument_layouts(payload['kwargs'])
        with self.assertRaisesRegex(ValueError, 'protected K'):
            replay.prepare_arguments(payload, attention_zero_oracle=True)
        with self.assertRaisesRegex(ValueError, 'requires attention'):
            replay.prepare_arguments(fixture(replay.SILU), attention_zero_oracle=True)

    def test_zero_oracle_postcall_labels_and_cpu_cli_transform_report(self):
        payload = fixture(pristine=False, stage='output')
        with self.assertRaisesRegex(ValueError, '--allow-postcall-inputs'):
            replay.prepare_arguments(payload, attention_zero_oracle=True)
        prepared = replay.prepare_arguments(payload, attention_zero_oracle=True, allow_postcall_inputs=True)
        self.assertFalse(prepared['provenance']['input_snapshot_pristine'])
        self.assertIn('conditional post-call', prepared['provenance']['replay_interpretation'])
        with tempfile.TemporaryDirectory() as directory:
            capture, report = Path(directory) / 'capture.pt', Path(directory) / 'report.json'
            torch.save(payload, capture)
            with patch.object(replay.importlib, 'import_module', side_effect=AssertionError('GPU import')), redirect_stdout(io.StringIO()):
                replay.main([str(capture), '--report', str(report), '--attention-zero-oracle', '--allow-postcall-inputs'])
            saved = json.loads(report.read_text())
            self.assertFalse(saved['provenance']['input_snapshot_pristine'])
            self.assertEqual(saved['transformation']['mode'], 'attention_zero_oracle_v1')
            self.assertNotEqual(saved['argument_digests'], saved['original_argument_digests'])
            self.assertEqual(saved['argument_digests']['inputs']['k'], saved['original_argument_digests']['inputs']['k'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_zero_failure_saved_before_coincident_read_mutation_error(self):
        payload = fixture(pristine=False, stage='output')
        prepared = replay.prepare_arguments(payload, allow_postcall_inputs=True, attention_zero_oracle=True)
        progress, calls = [], []

        def corrupt(**values):
            calls.append(1)
            values['dq'][1, 0, 1] = 1e-200
            values['q'].add_(1)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'failure.pt'
            with self.assertRaisesRegex(RuntimeError, 'changed captured read-input'):
                replay.replay_capture(payload, allow_postcall_inputs=True, attention_zero_oracle=True,
                                        device='cpu', function=corrupt, repeats=3, failure_dump=target,
                                        _prepared=prepared, progress=progress.append,
                                        replay_context={'source': 'CPU test'})
            self.assertEqual(len(calls), 1)
            capture = torch.load(target, map_location='cpu', weights_only=False)
            self.assertEqual(capture['format'], 'training_readwrite_replay_failure_v1')
            self.assertFalse(capture['original_training_capture']['input_snapshot_pristine'])
            pre, post = capture['prepared_pre_call'], capture['post_call']
            self.assertTrue(pre['input_snapshot_pristine_for_this_prepared_replay'])
            self.assertTrue(pre['matches_actual_replay_pre_call_hashes'])
            self.assertFalse(pre['matches_original_training_capture_bytes'])
            self.assertEqual(pre['argument_digests'], prepared['prepared_argument_digests'])
            self.assertEqual(int(torch.count_nonzero(pre['kwargs']['q'])), 0)
            self.assertTrue(torch.equal(post['arguments']['q'], torch.ones_like(post['arguments']['q'])))
            self.assertEqual(post['outputs']['dq'][1, 0, 1], 1e-200)
            self.assertEqual(post['outputs']['dq'].untyped_storage()._cdata, post['arguments']['dq'].untyped_storage()._cdata)
            self.assertEqual(post['outputs']['dq'].untyped_storage()._cdata, post['outputs']['dk'].untyped_storage()._cdata)
            self.assertNotEqual(pre['kwargs']['dq'].untyped_storage()._cdata, post['outputs']['dq'].untyped_storage()._cdata)
            self.assertEqual(post['outputs']['dq'].stride(), payload['kwargs']['dq'].stride())
            self.assertEqual(post['outputs']['dq'].untyped_storage().nbytes(), 64 * 8)
            self.assertTrue(post['matches_observed_post_call_hashes'])
            self.assertTrue(capture['trigger_persists_in_cpu_snapshot'])
            self.assertEqual(capture['failure_reasons'], ['zero_oracle_violation', 'read_input_mutation'])
            self.assertEqual(capture['execution_controls'], capture['observed_attempt']['execution_controls'])
            self.assertEqual(capture['replay_context'], {'source': 'CPU test'})
            self.assertIn('failure_dump_complete', [event['phase'] for event in progress])
            self.assertEqual(progress[-1]['attempts'][0]['failure_dump']['path'], str(target))
            self.assertFalse(target.with_name(target.name + '.tmp').exists())

    def test_ordinary_silu_failure_preserves_negative_views_and_exact_budget(self):
        payload = fixture(replay.SILU)
        initial = replay.argument_digests(payload['kwargs'], chunk_bytes=29, threshold=1e6)
        expected_size = 2 * sum(item['bytes'] for item in initial['raw_storages'])
        calls = []

        def corrupt(**values):
            calls.append(1)
            result = torch.ops.aten.silu_backward(**values)
            values['grad_input'][2, 1] = 1e30
            return result

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'silu.pt'
            reports = replay.replay_capture(payload, device='cpu', function=corrupt, repeats=3,
                                             failure_dump=target, max_capture_bytes=expected_size, chunk_bytes=29)
            self.assertEqual(len(calls), 1)
            self.assertTrue(reports[0]['output_anomaly'])
            saved = torch.load(target, map_location='cpu', weights_only=False)
            pre, post = saved['prepared_pre_call'], saved['post_call']
            self.assertTrue(pre['kwargs']['self'].is_neg())
            self.assertTrue(post['arguments']['self'].is_neg())
            self.assertEqual(post['outputs']['du'].storage_offset(), payload['kwargs']['grad_input'].storage_offset())
            self.assertEqual(post['outputs']['du'].stride(), payload['kwargs']['grad_input'].stride())
            self.assertEqual(post['outputs']['du'].untyped_storage()._cdata, post['arguments']['grad_input'].untyped_storage()._cdata)
            self.assertEqual(float(post['outputs']['du'][2, 1]), 1e30)
            self.assertEqual(saved['capture_budget']['total_unique_snapshot_storage_bytes'], expected_size)
            self.assertEqual(reports[0]['failure_dump']['capture_storage_bytes'], expected_size)
            self.assertTrue(pre['matches_original_training_capture_bytes'])
            self.assertEqual(initial, replay.argument_digests(payload['kwargs'], chunk_bytes=29, threshold=1e6))

    def test_failure_budget_and_existing_paths_reject_before_dispatch(self):
        payload = fixture()
        prepared = replay.prepare_arguments(payload)
        required = 2 * sum(item['bytes'] for item in prepared['prepared_argument_digests']['raw_storages'])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'failure.pt'
            with self.assertRaisesRegex(ValueError, 'Failure capture needs'):
                replay.replay_capture(payload, device='cpu', function=lambda **kw: self.fail('dispatched'),
                                        failure_dump=target, max_capture_bytes=required - 1)
            self.assertFalse(target.exists())
            for existing in (target, target.with_name(target.name + '.tmp')):
                existing.write_bytes(b'owned by another run')
                with self.assertRaisesRegex(ValueError, 'must be new paths'):
                    replay.replay_capture(payload, device='cpu', function=lambda **kw: self.fail('dispatched'), failure_dump=target)
                self.assertEqual(existing.read_bytes(), b'owned by another run')
                existing.unlink()

    def test_no_failure_copy_or_file_on_healthy_replay(self):
        import nan_backward_boundaries
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'unused.pt'
            with patch.object(nan_backward_boundaries, '_cpu_copy_tree', side_effect=AssertionError('healthy snapshot')):
                reports = replay.replay_capture(fixture(), device='cpu', function=attention,
                                                 failure_dump=target, repeats=2)
            self.assertEqual(len(reports), 2)
            self.assertTrue(all('failure_dump' not in report for report in reports))
            self.assertFalse(target.exists())

    def test_failure_atomic_publish_fsync_order_and_incomplete_temp_cleanup(self):
        payload = fixture()

        def corrupt(**values):
            attention(**values)
            values['dq'][0, 0, 0] = float('nan')

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'atomic.pt'
            order = []
            actual_sync, actual_replace = replay.os.fsync, replay.os.replace

            def sync(descriptor):
                order.append('fsync')
                actual_sync(descriptor)

            def replace(source, destination):
                self.assertEqual(order, ['fsync'])
                self.assertFalse(target.exists())
                self.assertEqual(torch.load(source, map_location='cpu', weights_only=False)['format'], 'training_readwrite_replay_failure_v1')
                order.append('replace')
                actual_replace(source, destination)

            with patch.object(replay.os, 'fsync', side_effect=sync), patch.object(replay.os, 'replace', side_effect=replace):
                replay.replay_capture(payload, device='cpu', function=corrupt, failure_dump=target)
            self.assertEqual(order, ['fsync', 'replace', 'fsync'])
            target.unlink()

            def incomplete(value, stream):
                stream.write(b'incomplete snapshot')
                raise OSError('injected serialization failure')

            progress = []
            with patch.object(torch, 'save', side_effect=incomplete), self.assertRaisesRegex(OSError, 'injected serialization failure'):
                replay.replay_capture(payload, device='cpu', function=corrupt, failure_dump=target, progress=progress.append)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name(target.name + '.tmp').exists())
            self.assertNotIn('failure_dump_complete', [event['phase'] for event in progress])

    def test_cli_rejects_failure_report_temporary_and_input_path_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, report, context = root / 'capture.pt', root / 'report.json', root / 'context.json'
            torch.save(fixture(), capture)
            for failure in (capture, report, report.with_name(report.name + '.tmp')):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    replay.main([str(capture), '--report', str(report), '--context', str(context),
                                 '--gpu', '--failure-dump', str(failure)])
            existing = report.with_name(report.name + '.tmp')
            existing.write_bytes(b'preexisting temporary')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                replay.main([str(capture), '--report', str(report)])
            self.assertEqual(existing.read_bytes(), b'preexisting temporary')


if __name__ == '__main__':
    unittest.main()
