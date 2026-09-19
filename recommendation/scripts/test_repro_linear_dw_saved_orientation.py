#!/usr/bin/env python3
"""CPU contract checks for the isolated saved-orientation DW experiment."""
import unittest
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import repro_linear_dw_saved_orientation as driver


class MMTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function == torch.ops.aten.mm.default:
            self.calls.append([(tuple(t.shape), tuple(t.stride())) for t in args])
        if function in (torch.ops.aten.addmm.default, torch.ops.aten.sum.default):
            raise AssertionError('Forward/bias operation unexpectedly executed')
        return function(*args, **(kwargs or {}))


class SavedOrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()
        cls.controls = driver.frozen.helpers.execution_controls()
        cls.metadata = {'preferred_blas_library': None,
                        'execution_controls_backward': cls.controls}

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert cls.controls == driver.frozen.helpers.execution_controls()

    def inputs(self, zero=False, device='cpu'):
        x = torch.tensor([[1, -2, 3, 0], [2, 1, -1, 4], [0, 3, 2, -1]], dtype=torch.bfloat16, device=device)
        dy = torch.tensor([[1, 0], [-2, 3], [4, -1]], dtype=torch.bfloat16, device=device)
        if zero:
            dy.zero_()
        return {'x': x, 'dy': dy, 'mat2': torch.zeros((2, 4), dtype=torch.bfloat16, device=device).t()}

    def execute(self, values):
        return driver.execute_saved_orientation({'pristine_inputs': values}, self.metadata, 'raw_dw')['raw_dw']

    def test_one_mm_matches_independent_integer_products_and_layout(self):
        values = self.inputs()
        saved = {k: t.clone() for k, t in values.items()}
        with MMTrace() as trace:
            output = self.execute(values)
        self.assertEqual(trace.calls, [[((2, 3), (1, 2)), ((3, 4), (4, 1))]])
        expected = torch.tensor([[-3, 6], [8, 0], [13, -5], [-12, 13]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(output.stride(), (1, 4))
        for k in saved:
            self.assertTrue(torch.equal(values[k], saved[k]))

    def test_zero_dy_exact_oracle(self):
        output = self.execute(self.inputs(zero=True))
        scan = driver.frozen.scan_outputs({'raw_dw': output}, zero_dy=True)
        self.assertFalse(scan['failed'])
        self.assertEqual(scan['outputs']['raw_dw']['nonzero_including_nonfinite_count'], 0)

    def test_frozen_oracle_rejects_tiny_nonzero(self):
        output = self.execute(self.inputs(zero=True))
        output[1, 1] = 2.0 ** -110
        self.assertTrue(driver.frozen.scan_outputs({'raw_dw': output}, zero_dy=True)['failed'])

    def test_row_major_mat2_rejected_before_mm(self):
        values = self.inputs(); values['mat2'] = values['mat2'].contiguous()
        with mock.patch.object(torch, 'mm', side_effect=AssertionError('must not call')):
            with self.assertRaisesRegex(ValueError, 'column-major'):
                self.execute(values)

    def test_noncontiguous_dy_rejected(self):
        values = self.inputs(); values['dy'] = values['dy'].t().contiguous().t()
        with self.assertRaisesRegex(ValueError, 'column-major'):
            self.execute(values)

    def test_wrong_harness_route_rejected(self):
        with self.assertRaisesRegex(ValueError, 'raw_dw'):
            driver.execute_saved_orientation({'pristine_inputs': self.inputs()}, self.metadata, 'addmm')

    def test_full_capture_shape_meta_has_same_single_mm(self):
        values = {'x': torch.empty((2324351, 256), device='meta', dtype=torch.bfloat16),
                  'dy': torch.empty((2324351, 512), device='meta', dtype=torch.bfloat16),
                  'mat2': torch.empty((512, 256), device='meta', dtype=torch.bfloat16).t()}
        with MMTrace() as trace:
            output = self.execute(values)
        self.assertEqual(trace.calls, [[((512, 2324351), (1, 512)), ((2324351, 256), (256, 1))]])
        self.assertEqual(tuple(output.shape), (256, 512))
        self.assertEqual(output.stride(), (1, 256))

    def test_frozen_source_pin_matches(self):
        self.assertEqual(driver.frozen.support.file_hash(driver.frozen.__file__), driver.FROZEN_SHA256)


if __name__ == '__main__':
    unittest.main()
