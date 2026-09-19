"""CPU graph, fault-retention and mocked stream checks; no GPU initialization."""
from contextlib import contextmanager, nullcontext
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import torch
import capture_additional_linear as frozen
import capture_additional_linear_deferred as capture
import nan_backward_training_deferred as scalar
from test_capture_additional_linear import TinyModel, input_tensor, flat_storage
from test_nan_backward_training_extended import make_full_modules
from test_nan_backward_training_upstream import Stack


class DeferredLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()
        cls.original_backward = torch.autograd.backward
        cls.sources = capture.verify_sources()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert torch.autograd.backward is cls.original_backward
        assert not torch.cuda.is_initialized()
        assert cls.sources == capture.verify_sources()

    def setUp(self):
        guard = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU initialization forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    @contextmanager
    def probe(self, model=None, dims=(3, 4), **options):
        model = model or TinyModel(dims)
        with tempfile.TemporaryDirectory() as directory:
            probe = capture.DeferredAdditionalLinearProbe(model, Path(directory) / 'capture',
                expected_dims=dims, chunk_bytes=4096, max_bytes=64 << 20,
                monitor_hstu=options.pop('monitor_hstu', False), **options)
            try:
                probe.set_attempt('training', 0, 1)
                yield model, probe
            finally:
                probe.close()

    def forward(self, model, value):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            return model(value)

    def payload(self, probe):
        paths = list(probe.directory.glob('*.pt'))
        self.assertEqual(len(paths), 1)
        return torch.load(paths[0], weights_only=False, map_location='cpu')

    def events(self, probe):
        return [json.loads(line) for line in (probe.directory / 'events.jsonl').read_text().splitlines()]

    def test_actual_bf16_graph_retains_raw_dx_and_complete_stage_mapping(self):
        with self.probe(save_finite_step=1) as (model, probe):
            self.forward(model, input_tensor(dtype=torch.bfloat16, strided=True)).sum().backward()
            payload = self.payload(probe)
            stages = payload['stages']
            self.assertEqual(payload['format'], frozen.FORMAT)
            self.assertEqual(payload['capture_variant'], capture.VARIANT)
            self.assertEqual(stages['raw_dx'].shape, (7, 3))
            self.assertEqual(stages['raw_dx'].dtype, torch.bfloat16)
            self.assertEqual(payload['pristine_inputs']['dy'].stride(), (0, 0))
            self.assertEqual(payload['pristine_inputs']['dy'].untyped_storage().nbytes(), 2)
            self.assertEqual(payload['pristine_inputs']['mat2'].stride(), (1, 3))
            self.assertTrue(torch.equal(stages['raw_small']['dmat2'].T.float(), stages['parameter_pre_weight']))
            self.assertTrue(torch.equal(stages['parameter_pre_weight'], stages['parameter_post_weight']))
            self.assertTrue(torch.equal(stages['parameter_post_weight'], stages['after_backward_return']['weight']))
            self.assertTrue(payload['metadata']['actual_backward_return_observed'])
            self.assertIsNone(payload['metadata']['pre_node_copy_completed_time'])
            self.assertTrue(all(not tensor.requires_grad and tensor.grad_fn is None for _, tensor in capture._tensors(payload)))
            self.assertIsNone(probe.frame)
            self.assertIsNone(probe.snapshots)

    def test_retained_payload_is_accepted_by_existing_capture_replay_validator(self):
        import replay_additional_linear_capture as replay
        with self.probe(dims=(256, 512), save_finite_step=1) as (model, probe):
            self.forward(model, input_tensor((256, 512), dtype=torch.bfloat16)).sum().backward()
            payload = self.payload(probe)
            bound = replay.validate_capture(payload)
            self.assertEqual(bound['inputs']['mat2'].shape, (256, 512))
            self.assertIn('raw_dx', bound['stages'])
            self.assertTrue(torch.equal(bound['inputs']['x'], payload['forward_inputs']['x']))

    def test_device_snapshot_preserves_entire_backing_aliases_offsets_and_independence(self):
        ledger = capture.DeviceSnapshots(4096)
        backing = torch.arange(30, dtype=torch.float32)
        views = {'a': backing[2:14:2], 'b': backing[4:16:2], 'expanded': backing[0].expand(5)}
        clone = ledger.copy(views)
        self.assertEqual(ledger.bytes, backing.untyped_storage().nbytes())
        self.assertEqual(clone['a'].storage_offset(), 2)
        self.assertEqual(clone['a'].stride(), (2,))
        self.assertEqual(clone['a'].untyped_storage()._cdata, clone['b'].untyped_storage()._cdata)
        self.assertEqual(clone['expanded'].stride(), (0,))
        expected = flat_storage(backing).clone()
        backing.fill_(99)
        self.assertTrue(torch.equal(flat_storage(clone['a']), expected))
        clone2 = ledger.copy(clone)
        self.assertNotEqual(clone2['a'].untyped_storage()._cdata, clone['a'].untyped_storage()._cdata)

    def test_no_hook_readback_or_cpu_copy_and_one_resolution_after_real_return(self):
        with self.probe() as (model, probe):
            reader = scalar._read_endpoint_batch
            stages = []
            def read(batch):
                self.assertFalse(probe.in_backward)
                self.assertEqual(probe.post_leaves, {'weight', 'bias'})
                stages.append('read')
                return reader(batch)
            with mock.patch.object(scalar, '_read_endpoint_batch', side_effect=read), \
                    mock.patch.object(capture, '_cpu_copy_tree', side_effect=AssertionError('CPU copy on finite step')), \
                    mock.patch.object(frozen, '_cpu_copy_tree', side_effect=AssertionError('Frozen CPU copy')), \
                    mock.patch.object(torch, 'equal', side_effect=AssertionError('Host equality')), \
                    mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host item')):
                result = self.forward(model, input_tensor())
                self.assertEqual(stages, [])
                result.sum().backward()
            self.assertEqual(stages, ['read'])
            self.assertTrue(probe.backward_complete)
            self.assertFalse(list(probe.directory.glob('*.pt')))

    def test_raw_nan_retains_exact_fault_outputs_and_pristine_inputs_without_rerun(self):
        model = TinyModel()
        calls, handles = [], []
        def install(module, args, output):
            def corrupt(inputs, outputs):
                calls.append(1)
                changed = list(inputs)
                changed[2] = changed[2].clone()
                changed[2][0, 0] = float('nan')
                return tuple(changed)
            handles.append(output.grad_fn.register_hook(corrupt))
        hook = model.linear.register_forward_hook(install)
        try:
            with self.probe(model) as (_, probe):
                result = self.forward(model, input_tensor())
                with self.assertRaises(capture.BoundaryAnomalyError):
                    result.sum().backward()
                payload = self.payload(probe)
                self.assertEqual(calls, [1])
                self.assertEqual(payload['stage'], 'raw_addmm_outputs')
                self.assertTrue(torch.isnan(payload['stages']['raw_small']['dmat2'][0, 0]))
                self.assertTrue(torch.isfinite(payload['pristine_inputs']['x']).all())
                self.assertIn('raw_dx', payload['stages'])
                self.assertIn('after_backward_return', payload['stages'])
                self.assertTrue(payload['trigger_persists_in_cpu_snapshot'])
        finally:
            hook.remove()
            for handle in handles:
                handle.remove()

    def test_raw_dx_only_fault_is_retained_after_downstream_backward(self):
        model = TinyModel()
        handles = []
        def install(module, args, output):
            def corrupt(inputs, outputs):
                changed = list(inputs)
                changed[1] = changed[1].clone()
                changed[1][1, 2] = float('nan')
                return tuple(changed)
            handles.append(output.grad_fn.register_hook(corrupt))
        hook = model.linear.register_forward_hook(install)
        try:
            with self.probe(model) as (_, probe):
                with self.assertRaises(capture.BoundaryAnomalyError):
                    self.forward(model, input_tensor()).sum().backward()
                payload = self.payload(probe)
                self.assertEqual(payload['stage'], 'raw_addmm_outputs')
                self.assertTrue(torch.isnan(payload['stages']['raw_dx'][1, 2]))
                self.assertTrue(torch.isfinite(payload['stages']['raw_small']['dmat2']).all())
        finally:
            hook.remove()
            for handle in handles:
                handle.remove()

    def test_post_node_input_mutation_keeps_before_and_after_backing_bytes(self):
        model = TinyModel()
        handles = []
        def install_valid(module, args, output):
            x = output.grad_fn._saved_mat1
            def mutate(inputs, outputs):
                x.data.fill_(42)
            handles.append(output.grad_fn.register_hook(mutate))
        hook = model.linear.register_forward_hook(install_valid)
        try:
            with self.probe(model) as (_, probe):
                with self.assertRaises(capture.BoundaryAnomalyError):
                    self.forward(model, input_tensor()).sum().backward()
                payload = self.payload(probe)
                self.assertEqual(payload['stage'], 'input_integrity_x')
                self.assertTrue(torch.all(payload['post_node_inputs']['x'] == 42))
                self.assertTrue(torch.all(payload['pristine_inputs']['x'] != 42))
        finally:
            hook.remove()
            for handle in handles:
                handle.remove()

    def test_finite_conversion_change_is_detected_below_threshold(self):
        model = TinyModel()
        hook = model.linear.weight.register_hook(lambda gradient: gradient + 1)
        try:
            with self.probe(model) as (_, probe):
                with self.assertRaises(capture.BoundaryAnomalyError):
                    self.forward(model, input_tensor()).sum().backward()
                payload = self.payload(probe)
                self.assertEqual(payload['stage'], 'raw_to_parameter_pre_weight')
                self.assertTrue(torch.equal(payload['stages']['parameter_pre_weight'],
                                            payload['stages']['raw_small']['dmat2'].T.float() + 1))
        finally:
            hook.remove()

    def test_accumulation_corruption_has_separate_snapshot(self):
        model = TinyModel()
        def mutate(parameter):
            parameter.grad.data[0, 0] += 1
        hook = model.linear.weight.register_post_accumulate_grad_hook(mutate)
        try:
            with self.probe(model) as (_, probe):
                with self.assertRaises(capture.BoundaryAnomalyError):
                    self.forward(model, input_tensor()).sum().backward()
                payload = self.payload(probe)
                self.assertEqual(payload['stage'], 'leaf_accumulation_weight')
                stages = payload['stages']
                self.assertTrue(torch.equal(stages['raw_small']['dmat2'].T.float(), stages['parameter_pre_weight']))
                self.assertFalse(torch.equal(stages['parameter_pre_weight'], stages['parameter_post_weight']))
        finally:
            hook.remove()

    def test_late_engine_writer_observed_after_return_without_early_callback(self):
        for nonfinite in (False, True):
            with self.probe() as (model, probe):
                calls = []
                def queue_late(parameter):
                    def mutate():
                        calls.append(1)
                        parameter.grad.data[0, 0] = float('nan') if nonfinite else parameter.grad[0, 0] + 1
                    torch.autograd.Variable._execution_engine.queue_callback(mutate)
                hook = model.linear.weight.register_post_accumulate_grad_hook(queue_late)
                try:
                    with self.assertRaises(capture.BoundaryAnomalyError):
                        self.forward(model, input_tensor()).sum().backward()
                    payload = self.payload(probe)
                    self.assertEqual(calls, [1])
                    self.assertEqual(payload['stage'], 'after_backward_return_selected' if nonfinite else 'post_accumulate_to_final_weight')
                    self.assertTrue(torch.isfinite(payload['stages']['parameter_post_weight']).all())
                    self.assertIn('final_dense', payload)
                finally:
                    hook.remove()

    def test_prior_gradient_is_observed_at_leaf_entry_and_exact_accumulation_passes(self):
        with self.probe(save_finite_step=1) as (model, probe):
            result = self.forward(model, input_tensor())
            model.linear.weight.grad = torch.full_like(model.linear.weight, .25)
            model.linear.bias.grad = torch.full_like(model.linear.bias, .5)
            result.sum().backward()
            stages = self.payload(probe)['stages']
            self.assertIsNone(stages['preexisting_grad']['weight'])
            self.assertTrue(torch.all(stages['parameter_prior_weight'] == .25))
            self.assertTrue(torch.equal(stages['parameter_post_weight'],
                                        stages['parameter_prior_weight'] + stages['parameter_pre_weight']))

    def test_shared_bucket_is_preserved_once_per_observation_and_not_aliased_across_stages(self):
        with self.probe(save_finite_step=1) as (model, probe):
            storage = torch.zeros(100)
            model.linear.weight.grad = storage[7:19].view_as(model.linear.weight)
            model.linear.bias.grad = storage[23:27]
            self.forward(model, input_tensor()).sum().backward()
            stages = self.payload(probe)['stages']
            final = stages['after_backward_return']
            self.assertEqual(final['weight'].untyped_storage().nbytes(), 400)
            self.assertEqual(final['weight'].storage_offset(), 7)
            self.assertEqual(final['weight'].untyped_storage()._cdata, final['bias'].untyped_storage()._cdata)
            self.assertNotEqual(final['weight'].untyped_storage()._cdata, stages['parameter_post_weight'].untyped_storage()._cdata)

    def test_first_fault_stops_clip_and_optimizer_after_all_stages(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        hook = model.linear.weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        try:
            with self.probe(model, optimizer=optimizer) as (_, probe):
                with mock.patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=AssertionError('clip reached')), \
                        mock.patch.object(optimizer, 'step', side_effect=AssertionError('update reached')), \
                        self.assertRaises(capture.BoundaryAnomalyError):
                    self.forward(model, input_tensor()).sum().backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    probe.optimizer_proxy.step()
                self.assertEqual(probe.post_leaves, {'weight', 'bias'})
                with self.assertRaises(capture.BoundaryProbeError):
                    probe.optimizer_proxy.step()
        finally:
            hook.remove()

    def test_optimizer_lifecycle_next_attempt_and_cleanup_restore_hooks(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with self.probe(model, optimizer=optimizer) as (_, probe):
            with self.assertRaises(capture.BoundaryProbeError):
                probe.optimizer_proxy.step()
            result = self.forward(model, input_tensor())
            ref = weakref.ref(result)
            result.sum().backward()
            with self.assertRaises(capture.BoundaryProbeError):
                probe.set_attempt('training', 0, 2)
            probe.optimizer_proxy.step()
            optimizer.zero_grad(set_to_none=True)
            probe.set_attempt('training', 0, 2)
            self.forward(model, input_tensor()).sum().backward()
            probe.optimizer_proxy.step()
            del result
        gc.collect()
        self.assertIsNone(ref())
        self.assertIs(torch.autograd.backward, type(self).original_backward)
        self.assertFalse(model.linear._forward_hooks)
        self.assertFalse(model.linear._forward_pre_hooks)
        self.assertFalse(model.linear.weight._backward_hooks)
        self.assertIsNone(probe.scalar_queue)

    def test_budget_failure_happens_before_copy_or_linear_and_bad_budget_rejected(self):
        ledger = capture.DeviceSnapshots(1)
        with mock.patch.object(torch.Tensor, 'clone', side_effect=AssertionError('copy reached')), \
                self.assertRaises(capture.BoundaryProbeError):
            ledger.copy(torch.ones(2))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                capture.DeferredAdditionalLinearProbe(TinyModel(), Path(directory) / 'bad', max_bytes=0, monitor_hstu=False)

    def test_original_backward_error_cleans_up_without_resolving_partial_attempt(self):
        model = TinyModel()
        with self.probe(model) as (_, probe):
            result = self.forward(model, input_tensor())
            with mock.patch.object(scalar, '_read_endpoint_batch', side_effect=AssertionError('Partial flush')), \
                    self.assertRaises(RuntimeError):
                result.backward(torch.ones(2))
            self.assertFalse(probe.backward_complete)
            self.assertFalse(probe.scalar_queue.flushed)
        self.assertIs(torch.autograd.backward, type(self).original_backward)
        self.assertFalse(model.linear._forward_hooks)
        self.assertFalse(model.linear.weight._backward_hooks)

    def test_cli_rejects_invalid_bounds_before_trainer_import(self):
        for steps, finite in ((0, 0), (2, 3), (2, -1)):
            with tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(frozen.importlib, 'import_module', side_effect=AssertionError('Trainer import')), \
                        self.assertRaises(SystemExit) as error:
                    capture.main(['--directory', str(Path(directory) / 'new'), '--steps', str(steps),
                                  '--save-finite-step', str(finite)])
                self.assertEqual(error.exception.code, 2)

    def test_existing_directory_and_foreign_temporary_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                capture.DeferredAdditionalLinearProbe(TinyModel(), directory, expected_dims=(3, 4), monitor_hstu=False)
            target = Path(directory) / 'fault.pt'
            temporary = target.with_suffix('.pt.tmp')
            temporary.write_bytes(b'foreign')
            with self.assertRaises(FileExistsError):
                frozen.atomic_save(target, {'x': 1})
            self.assertEqual(temporary.read_bytes(), b'foreign')

    def test_mocked_copy_and_scalar_streams_are_joined_before_final_reads(self):
        ledger = capture.DeviceSnapshots(64)
        queue = scalar.DeferredScalarQueue(chunk_bytes=32, abs_threshold=10)
        ledger.streams['cuda:0'][11] = 'copy-stream'
        queue.streams['cuda:0'][12] = 'scan-stream'
        order = []
        class Stream:
            cuda_stream = 99
            def wait_event(self, event):
                order.append(('wait', event.producer))
        class Event:
            def record(self, producer):
                self.producer = producer
                order.append(('record', producer))
        with mock.patch.object(torch.cuda, 'current_stream', return_value=Stream()), \
                mock.patch.object(torch.cuda, 'Event', Event):
            records = ledger.join(queue)
        self.assertEqual(order, [('record', 'copy-stream'), ('wait', 'copy-stream'),
                                 ('record', 'scan-stream'), ('wait', 'scan-stream')])
        self.assertEqual([record['producer_stream_id'] for record in records], [11, 12])

    def test_mocked_cross_stream_snapshot_read_waits_for_copy_event(self):
        ledger = capture.DeviceSnapshots(64)
        tensor = torch.ones(2)
        producer = type('Producer', (), {'cuda_stream': 11})()
        order = []
        class Consumer:
            cuda_stream = 12
            def wait_event(self, event):
                order.append(('wait', event))
        ledger.producers[capture._storage_key(tensor)] = producer, 'copy-event'
        with mock.patch.object(torch.cuda, 'current_stream', return_value=Consumer()), \
                mock.patch.object(torch.Tensor, 'record_stream', side_effect=lambda stream: order.append(('record_stream', stream.cuda_stream))):
            ledger.wait_reads({'a': tensor, 'b': tensor})
        self.assertEqual(order, [('wait', 'copy-event'), ('record_stream', 12)])

    def test_all42_hstu_endpoints_share_final_resolution_without_completion_callback(self):
        compute, preprocess, output, state = make_full_modules()
        # Reconciled source added the nullcontext used when B0 tripwires are off.
        preprocess.nullcontext = nullcontext
        model = TinyModel((4, 4)).double()
        model.stack = Stack(compute).double()
        model.forward = lambda value: model.stack(model._additional_embedding_mlp(value))
        with mock.patch.dict(os.environ, {'NAN_BACKWARD_TARGET': '*', 'NAN_BACKWARD_SAVE_ALL': '0'}):
            with self.probe(model, dims=(4, 4), monitor_hstu=True,
                            hstu_modules={'compute_module': compute, 'preprocess_module': preprocess, 'output_module': output}) as (_, probe):
                with mock.patch.object(scalar, '_read_endpoint_batch', wraps=scalar._read_endpoint_batch) as reader:
                    x = torch.arange(12, dtype=torch.float64).reshape(3, 4).div(64).requires_grad_()
                    model(x).sum().backward()
                self.assertEqual(reader.call_count, 1)
                self.assertEqual((state.attention_calls, state.silu_calls), (3, 3))
                flush = [event for event in self.events(probe) if event['event'] == 'deferred_flush'][0]
                operations = [scan for scan in flush['scans'] if scan['kind'] == 'operation']
                self.assertEqual(len(operations), 42)
                self.assertIsNone(probe.hstu.sentinel)
                self.assertTrue(flush['actual_backward_return_observed'])

    def test_source_verification_and_scoped_launcher_leave_frozen_globals_unchanged(self):
        before = dict(frozen.configure_environment.__globals__)
        environment = {}
        options = {'directory': '/tmp/new-deferred-capture', 'steps': 1000, 'abs_threshold': 1e6,
                   'dataset': 'yambda-5b', 'save_finite_step': 1}
        capture.configure_environment(options, environment)
        recorded = json.loads(environment['NAN_TRAINING_OPTIONS'])
        self.assertEqual(recorded['monitor_variant'], capture.VARIANT)
        self.assertEqual(recorded['hstu_expected_endpoints'], 42)
        self.assertEqual(environment['NAN_BACKWARD_SAVE_ALL'], '0')
        self.assertEqual(environment['NAN_BACKWARD_TARGET'], '*')
        self.assertEqual(before, frozen.configure_environment.__globals__)
        with mock.patch.dict(capture.FROZEN_SOURCES, {'capture_additional_linear.py': 'bad'}), \
                self.assertRaises(capture.BoundaryProbeError):
            capture.verify_sources()


if __name__ == '__main__':
    unittest.main()
