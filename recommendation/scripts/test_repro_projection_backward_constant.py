import ast
from pathlib import Path
import unittest
from unittest import mock

import torch
import repro_projection_backward_constant as target


class ConstantProjectionTests(unittest.TestCase):
    def test_exact_oracles_through_checked_in_function(self):
        path = Path(target.__file__).resolve().parents[1] / 'generative_recommenders/ops/triton/triton_addmm.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'triton_addmm_bwd')
        namespace = {'torch': torch, 'Tuple': tuple}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        rng = torch.get_rng_state().clone()
        with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden')):
            for case in ('zero', 'one_column'):
                for rows in (1, 17, 259):
                    with self.subTest(case=case, rows=rows):
                        prepared = target.prepare_constant(rows, 5, 7, case, chunk_bytes=1024)
                        actual = namespace['triton_addmm_bwd'](**prepared['prepared']['inputs'])
                        for result, expected in zip(actual, prepared['prepared']['references']):
                            self.assertTrue(torch.equal(result, expected))
                        # Independently evaluate the real mathematical products in FP64.
                        values = prepared['prepared']['inputs']
                        expected = ((values['dz'].double() @ values['w'].double().T).bfloat16(),
                                    (values['x'].double().T @ values['dz'].double()).bfloat16(),
                                    values['dz'].double().sum(0).bfloat16())
                        for result, ref in zip(actual, expected):
                            self.assertTrue(torch.equal(result, ref))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_dimensions_and_budget(self):
        for args in ((0, 5, 7, 'zero'), (True, 5, 7, 'zero'), (2**24+1, 5, 7, 'zero'),
                     (1, 5, 7, 'bad')):
            with self.assertRaises(ValueError):
                target.prepare_constant(*args)
        with self.assertRaises(ValueError):
            target.prepare_constant(100, 512, 2048, 'zero', max_bytes=100)

    def test_all_dx_elements_have_nonzero_reference(self):
        p = target.prepare_constant(9, 3, 4, 'one_column')
        dx, dw, db = p['prepared']['references']
        self.assertEqual(int(torch.count_nonzero(dx)), 27)
        self.assertEqual(float(dx[0, 0]), 2**-25)
        self.assertEqual(int(torch.count_nonzero(dw)), 0)
        self.assertEqual(int(torch.count_nonzero(db)), 1)
        self.assertFalse(p['transformation']['zero_oracle'])


if __name__ == '__main__':
    unittest.main()
