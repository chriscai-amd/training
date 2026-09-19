#!/usr/bin/env python3
"""CPU graph/ordering/storage fixtures for the focused additional Linear probe."""
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import torch

import capture_additional_linear as capture


class TinyModel(torch.nn.Module):
    def __init__(self, dims=(3, 4)):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            linear = torch.nn.Linear(*dims)
        with torch.no_grad():
            linear.weight.copy_(torch.arange(linear.weight.numel()).reshape_as(linear.weight) / 1024)
            linear.bias.fill_(.125)
        self._additional_embedding_mlp = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), linear)

    @property
    def linear(self):
        return self._additional_embedding_mlp[2]

    def forward(self, value):
        return self._additional_embedding_mlp(value)


def input_tensor(dims=(3, 4), dtype=torch.float32, strided=False):
    width = dims[0]
    if strided:
        raw = torch.arange(7 * (2 * width + 3), dtype=dtype).reshape(7, 2 * width + 3) / 128
        return raw[:, 1:1 + 2 * width:2].detach().requires_grad_()
    return (torch.arange(7 * width, dtype=dtype).reshape(7, width) / 128).requires_grad_()


def flat_storage(value):
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


class FocusedLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()
        cls.original_backward = torch.autograd.backward

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert torch.autograd.backward is cls.original_backward
        assert not torch.cuda.is_initialized()

    @contextmanager
    def probe(self, model=None, dims=(3, 4), **options):
        model = model or TinyModel(dims)
        with tempfile.TemporaryDirectory() as directory:
            probe = capture.AdditionalLinearProbe(model, Path(directory) / 'capture',
                                                   expected_dims=dims, chunk_bytes=4096,
                                                   max_bytes=64 << 20, **options)
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

    def test_bf16_autocast_actual_node_mapping_and_fp32_leaf_conversion(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.probe(dims=(256, 512), save_finite_step=1) as (model, probe):
                value = input_tensor((256, 512), dtype=dtype)
                result = self.forward(model, value)
                result.sum().backward()
                self.assertTrue(probe.backward_complete)
                self.assertEqual(probe.call_handles, [])
                frame = probe.frame
                self.assertEqual(frame['metadata']['node_kind'], 'AddmmBackward0')
                self.assertTrue(frame['metadata']['execution_controls_forward']['autocast']['cpu']['enabled'])
                self.assertFalse(frame['metadata']['execution_controls_backward']['autocast']['cpu']['enabled'])
                x, mat2, dy = (frame['pristine_inputs'][name] for name in ('x', 'mat2', 'dy'))
                self.assertEqual(x.dtype, torch.bfloat16)
                self.assertEqual(mat2.shape, (256, 512))
                self.assertEqual(mat2.stride(), (1, 256))
                raw = frame['stages']['raw_small']
                self.assertEqual(raw['dmat2'].dtype, torch.bfloat16)
                self.assertEqual(raw['dmat2'].shape, (256, 512))
                self.assertEqual(frame['stages']['parameter_pre_weight'].dtype, torch.float32)
                self.assertTrue(torch.equal(raw['dmat2'].T.float(), frame['stages']['parameter_pre_weight']))
                self.assertTrue(torch.equal(raw['bias'].float(), frame['stages']['parameter_pre_bias']))
                self.assertTrue(torch.equal(x, value.detach().bfloat16()))
                self.assertTrue(torch.equal(mat2, model.linear.weight.detach().bfloat16().T))
                saved = self.payload(probe)
                self.assertEqual(saved['format'], capture.FORMAT)
                self.assertEqual(saved['session'], probe.session)
                self.assertEqual(saved['metadata']['input_provenance'], {'pristine': True, 'operation_started': False})
                self.assertEqual(saved['stage'], 'after_backward_return')
                self.assertEqual(saved['pristine_inputs']['dy'].stride(), (0, 0))
                self.assertEqual(saved['pristine_inputs']['dy'].untyped_storage().nbytes(), 2)

    def test_strided_forward_input_snapshot_retains_whole_backing_and_is_independent(self):
        with self.probe() as (model, probe):
            value = input_tensor(dtype=torch.bfloat16, strided=True)
            original_raw = flat_storage(value).clone()
            result = self.forward(model, value)
            original = probe.frame['forward_inputs']['x']
            self.assertEqual(original.stride(), value.stride())
            self.assertEqual(original.storage_offset(), value.storage_offset())
            self.assertTrue(torch.equal(flat_storage(original), original_raw))
            self.assertNotEqual(original.untyped_storage()._cdata, value.untyped_storage()._cdata)
            result.sum().backward()
            self.assertTrue(torch.equal(flat_storage(original), original_raw))
            self.assertTrue(all(not tensor.requires_grad and tensor.grad_fn is None
                                for _, tensor in capture._tensors(probe.frame)))

    def test_prior_gradient_is_observed_at_leaf_entry_after_forward(self):
        with self.probe() as (model, probe):
            result = self.forward(model, input_tensor())
            self.assertIsNone(probe.frame['stages']['preexisting_grad']['weight'])
            model.linear.weight.grad = torch.full_like(model.linear.weight, .25)
            model.linear.bias.grad = torch.full_like(model.linear.bias, .5)
            result.sum().backward()
            for name, prior in (('weight', .25), ('bias', .5)):
                stages = probe.frame['stages']
                self.assertTrue(torch.equal(stages[f'parameter_prior_{name}'], torch.full_like(getattr(model.linear, name), prior)))
                self.assertTrue(torch.equal(stages[f'parameter_post_{name}'],
                                            stages[f'parameter_prior_{name}'] + stages[f'parameter_pre_{name}']))

    def test_nan_prior_gradient_stops_before_leaf_accumulation_not_as_gemm_fault(self):
        with self.probe() as (model, probe):
            result = self.forward(model, input_tensor())
            model.linear.weight.grad = torch.zeros_like(model.linear.weight)
            model.linear.weight.grad[0, 0] = float('nan')
            with self.assertRaises(capture.BoundaryAnomalyError):
                result.sum().backward()
            payload = self.payload(probe)
            self.assertEqual(payload['session'], probe.session)
            self.assertEqual(payload['stage'], 'parameter_prior_weight')
            self.assertTrue(torch.isfinite(payload['stages']['raw_small']['dmat2']).all())
            self.assertTrue(torch.isfinite(payload['stages']['parameter_pre_weight']).all())
            self.assertTrue(torch.isnan(payload['stages']['parameter_prior_weight'][0, 0]))
            self.assertNotIn('parameter_post_weight', payload['stages'])

    def test_raw_node_fault_retains_pristine_and_current_inputs_without_rerun(self):
        model = TinyModel()
        calls = []
        injected_handles = []
        def install(module, args, output):
            node = output.grad_fn
            saved_x = node._saved_mat1
            def corrupt(grad_inputs, grad_outputs):
                calls.append(1)
                changed = list(grad_inputs)
                changed[2] = changed[2].clone()
                changed[2][0, 0] = float('nan')
                saved_x.data[0, 0] = 42
                return tuple(changed)
            injected_handles.append(node.register_hook(corrupt))
        outer = model.linear.register_forward_hook(install)
        try:
            with self.probe(model) as (_, probe):
                result = self.forward(model, input_tensor())
                with self.assertRaises(capture.BoundaryAnomalyError):
                    result.sum().backward()
                payload = self.payload(probe)
                self.assertEqual(calls, [1])
                self.assertEqual(payload['stage'], 'raw_addmm_outputs')
                self.assertTrue(torch.isnan(payload['fault_snapshot']['dmat2'][0, 0]))
                self.assertEqual(float(payload['fault_snapshot']['current_inputs']['x'][0, 0]), 42)
                self.assertNotEqual(float(payload['pristine_inputs']['x'][0, 0]), 42)
                self.assertTrue(payload['trigger_persists_in_cpu_snapshot'])
                self.assertNotIn('parameter_pre_weight', payload['stages'])
        finally:
            outer.remove()
            for handle in injected_handles:
                handle.remove()

    def test_finite_post_conversion_hook_change_is_separate_from_raw_dmat2(self):
        model = TinyModel()
        def alter(gradient):
            changed = gradient.clone()
            changed[0, 0] += 1
            return changed
        handle = model.linear.weight.register_hook(alter)
        try:
            with self.probe(model) as (_, probe):
                result = self.forward(model, input_tensor())
                with self.assertRaises(capture.BoundaryAnomalyError):
                    result.sum().backward()
                payload = self.payload(probe)
                self.assertEqual(payload['stage'], 'parameter_pre_weight_mismatch')
                self.assertIsNone(payload['trigger_persists_in_cpu_snapshot'])
                self.assertTrue(torch.isfinite(payload['stages']['raw_small']['dmat2']).all())
                self.assertFalse(torch.equal(payload['stages']['raw_small']['dmat2'].T.float(), payload['stages']['parameter_pre_weight']))
        finally:
            handle.remove()

    def test_final_backward_return_observes_late_engine_callback_before_clip(self):
        for nonfinite in (True, False):
            with self.probe() as (model, probe):
                queued = []
                def queue_late(parameter):
                    def mutate():
                        queued.append(1)
                        parameter.grad.data[0, 0] = float('nan') if nonfinite else parameter.grad[0, 0] + 1
                    torch.autograd.Variable._execution_engine.queue_callback(mutate)
                handle = model.linear.weight.register_post_accumulate_grad_hook(queue_late)
                clipped = []
                try:
                    result = self.forward(model, input_tensor())
                    with self.assertRaises(capture.BoundaryAnomalyError):
                        result.sum().backward()
                        clipped.append(1)
                    payload = self.payload(probe)
                    self.assertEqual(queued, [1])
                    self.assertEqual(clipped, [])
                    self.assertTrue(torch.isfinite(payload['stages']['parameter_post_weight']).all())
                    self.assertEqual(payload['stage'], 'after_backward_return_selected' if nonfinite else 'after_backward_return_weight_mismatch')
                    self.assertFalse(probe.backward_complete)
                finally:
                    handle.remove()

    def test_shared_grad_bucket_backing_offsets_and_aliases_retained(self):
        model = TinyModel()
        backing = torch.full((64,), 17., dtype=torch.float32)
        model.linear.weight.grad = backing[7:19].reshape(4, 3)
        model.linear.bias.grad = backing[31:35]
        model.linear.weight.grad.zero_()
        model.linear.bias.grad.zero_()
        with self.probe(model, save_finite_step=1) as (_, probe):
            self.forward(model, input_tensor()).sum().backward()
            payload = self.payload(probe)
            final = payload['stages']['after_backward_return']
            self.assertEqual(final['weight'].storage_offset(), 7)
            self.assertEqual(final['bias'].storage_offset(), 31)
            self.assertEqual(final['weight'].untyped_storage()._cdata, final['bias'].untyped_storage()._cdata)
            self.assertEqual(final['weight'].untyped_storage().nbytes(), backing.untyped_storage().nbytes())
            self.assertTrue(torch.equal(flat_storage(final['weight']), flat_storage(backing)))
            self.assertTrue(torch.equal(payload['stages']['parameter_prior_weight'], torch.zeros(4, 3)))
            self.assertEqual(float(flat_storage(final['weight']).view(torch.float32)[0]), 17.)

    def test_one_attempt_lifecycle_rejects_duplicate_forward_and_second_backward(self):
        with self.probe() as (model, probe):
            value = input_tensor()
            result = self.forward(model, value)
            with self.assertRaises(capture.BoundaryProbeError):
                self.forward(model, value)
            result.sum().backward(retain_graph=True)
            with self.assertRaises(capture.BoundaryProbeError):
                result.sum().backward()
        self.assertIs(torch.autograd.backward, FocusedLinearTests.original_backward)

    def test_success_and_error_close_remove_hooks_and_restore_dispatch(self):
        for fail in (False, True):
            model = TinyModel()
            with self.probe(model) as (_, probe):
                value = input_tensor()
                result = self.forward(model, value)
                reference = weakref.ref(result)
                if fail:
                    with self.assertRaises(RuntimeError):
                        result.backward(torch.ones(2))
                else:
                    result.sum().backward()
                del result
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(probe.call_handles, [])
            self.assertEqual(probe.handles, [])
            self.assertIsNone(probe.frame)
            self.assertEqual(len(model.linear._forward_hooks), 0)
            self.assertEqual(len(model.linear._forward_pre_hooks), 0)
            self.assertIs(torch.autograd.backward, FocusedLinearTests.original_backward)

    def test_optimizer_gate_and_next_attempt_require_completed_step(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with self.probe(model, optimizer=optimizer) as (_, probe):
            with self.assertRaises(capture.BoundaryProbeError):
                probe.optimizer_proxy.step()
            self.forward(model, input_tensor()).sum().backward()
            with self.assertRaises(capture.BoundaryProbeError):
                probe.set_attempt('training', 0, 2)
            probe.optimizer_proxy.step()
            with self.assertRaises(capture.BoundaryProbeError):
                probe.optimizer_proxy.step()
            optimizer.zero_grad(set_to_none=True)
            probe.set_attempt('training', 0, 2)
            self.forward(model, input_tensor()).sum().backward()
            probe.optimizer_proxy.step()
            self.assertTrue(probe.optimizer_complete)

    def test_small_copy_budget_rejects_before_linear_executes(self):
        model = TinyModel()
        count = []
        with tempfile.TemporaryDirectory() as directory:
            probe = capture.AdditionalLinearProbe(model, Path(directory) / 'capture',
                                                   expected_dims=(3, 4), chunk_bytes=16, max_bytes=1)
            probe.set_attempt('training', 0, 1)
            hook = model.linear.register_forward_hook(lambda *args: count.append(1))
            try:
                with self.assertRaises(capture.BoundaryProbeError):
                    self.forward(model, input_tensor())
                self.assertEqual(count, [])
                self.assertEqual(list(probe.directory.glob('*.pt')), [])
            finally:
                hook.remove()
                probe.close()

    def test_atomic_save_preserves_foreign_temporary_and_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'fault.pt'
            temporary = target.with_suffix('.pt.tmp')
            temporary.write_bytes(b'foreign')
            with self.assertRaises(FileExistsError):
                capture.atomic_save(target, {'value': torch.ones(2)})
            self.assertEqual(temporary.read_bytes(), b'foreign')
            temporary.unlink()
            target.write_bytes(b'prior')
            with self.assertRaises(FileExistsError):
                capture.atomic_save(target, {'value': torch.ones(2)})
            self.assertEqual(target.read_bytes(), b'prior')
            self.assertFalse(temporary.exists())

    def test_atomic_save_cleanup_on_write_error_and_fsyncs_both_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'fault.pt'
            with mock.patch.object(capture.torch, 'save', side_effect=RuntimeError('save failed')):
                with self.assertRaises(RuntimeError):
                    capture.atomic_save(target, {})
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.pt.tmp').exists())
            with mock.patch.object(capture.os, 'fsync', wraps=os.fsync) as fsync:
                capture.atomic_save(target, {'value': torch.ones(2)})
            self.assertEqual(fsync.call_count, 2)


if __name__ == '__main__':
    unittest.main()
