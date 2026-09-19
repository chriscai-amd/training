import unittest
from unittest.mock import patch

import torch
import reference_attention_dq_rows as ref


def fixture():
    q = ((torch.arange(8*2*4).reshape(8, 2, 4) % 17 - 8) / 16).bfloat16()
    k = q.flip(0).clone()
    v = ((torch.arange(8*2*3).reshape(8, 2, 3) % 13 - 6) / 16).bfloat16()
    return {'q': q, 'k': k, 'v': v, 'dout': v.flip(0).clone(),
            'seq_offsets': torch.tensor([0, 3, 8]), 'num_targets': torch.tensor([1, 2])}, {
            'N': 9, 'alpha': .125, 'enable_tma': False, 'num_softmax_heads': 0,
            'max_attn_len': 0, 'contextual_seq_len': 0}


class QueryReferenceTests(unittest.TestCase):
    def setUp(self):
        self.trap = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        self.trap.start()
        self.addCleanup(self.trap.stop)

    def test_full_forward_autograd_all_queries_and_masks(self):
        inputs, scalar = fixture()
        for context, window in ((0, 0), (0, 2), (1, 0), (2, 3)):
            scalar.update(contextual_seq_len=context, max_attn_len=window)
            for sequence, (start, stop) in enumerate(((0, 3), (3, 8))):
                for head in range(2):
                    q = inputs['q'][start:stop, head].double().requires_grad_()
                    k, v, g = (inputs[n][start:stop, head].double() for n in ('k', 'v', 'dout'))
                    mask = ref.mask_helper()(torch.device('cpu'), True, scalar['N'],
                        torch.tensor([stop-start]), inputs['num_targets'][sequence:sequence+1],
                        window, context)[0, :stop-start, :stop-start]
                    output = ((torch.nn.functional.silu(q @ k.T * scalar['alpha']) / scalar['N']) * mask) @ v
                    expected = torch.autograd.grad((output*g).sum(), q)[0]
                    for local in range(stop-start):
                        actual = ref.reference_query(inputs, scalar, start+local, head)['dq_fp64']
                        torch.testing.assert_close(actual, expected[local], rtol=2e-14, atol=1e-16)

    def test_zero_query_incoming_gradient_exact(self):
        inputs, scalar = fixture()
        inputs['dout'][4, 1].zero_()
        result = ref.reference_query(inputs, scalar, 4, 1)
        self.assertTrue(result['zero_incoming_query_gradient'])
        self.assertTrue(bool((result['dq_fp64'] == 0).all()))

    def test_unrelated_query_gradient_cannot_change_dq(self):
        inputs, scalar = fixture()
        first = ref.reference_query(inputs, scalar, 4, 1)['dq_fp64']
        inputs['dout'][3, 1].fill_(100)
        self.assertTrue(torch.equal(first, ref.reference_query(inputs, scalar, 4, 1)['dq_fp64']))

    def test_normalization_uses_global_n(self):
        inputs, scalar = fixture()
        first = ref.reference_query(inputs, scalar, 4, 1)['dq_fp64']
        scalar['N'] *= 2
        self.assertTrue(torch.equal(first/2, ref.reference_query(inputs, scalar, 4, 1)['dq_fp64']))

    def test_invalid_contexts_rejected(self):
        for change in ({'N': 4}, {'enable_tma': True}, {'num_softmax_heads': 1}, {'alpha': float('nan')}, {'max_attn_len': -1}):
            inputs, scalar = fixture()
            with self.assertRaises(ValueError): ref.reference_query(inputs, {**scalar, **change}, 4, 1)
        inputs, scalar = fixture()
        inputs['seq_offsets'][1] = -1
        with self.assertRaises(ValueError): ref.reference_query(inputs, scalar, 4, 1)

    def test_dependencies_must_be_finite(self):
        inputs, scalar = fixture()
        inputs['k'][3, 1, 2] = float('nan')
        with self.assertRaises(ValueError): ref.reference_query(inputs, scalar, 4, 1)

    def test_reference_overflow_is_rejected(self):
        inputs, scalar = fixture()
        inputs['q'].fill_(1e30)
        inputs['k'].fill_(1e30)
        scalar['alpha'] = 1e300
        with self.assertRaisesRegex(ValueError, 'FP64 score overflow'):
            ref.reference_query(inputs, scalar, 4, 1)


if __name__ == '__main__': unittest.main()
