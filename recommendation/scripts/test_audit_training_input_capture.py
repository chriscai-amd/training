"""Known BF16 bit-pattern fixtures for the complete input-capture scan."""
import hashlib
import unittest

import numpy as np
import torch

from audit_training_input_capture import scan_bf16


class InputCaptureScanTests(unittest.TestCase):
    def test_known_zero_subnormal_normal_inf_and_nan_bits(self):
        bits = np.array([[0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80],
                         [0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0xff81]], dtype=np.uint16)
        value = torch.from_numpy(bits).view(torch.bfloat16)
        result = scan_bf16(value, chunk_rows=1)
        self.assertEqual(result['bit_classes'], {
            'positive_zero': 1, 'negative_zero': 1, 'positive_infinity': 1,
            'negative_infinity': 1, 'positive_nan': 2, 'negative_nan': 2,
            'quiet_nan': 2, 'signaling_nan': 2, 'subnormal': 2, 'normal': 2})
        self.assertEqual(result['max_abs_finite'], 1.0)
        self.assertEqual(result['counts']['nonfinite'], 6)
        self.assertEqual(result['rows']['nonfinite']['range'], [1, 1])
        self.assertEqual(result['logical_and_complete_backing_sha256'], hashlib.sha256(bits.tobytes()).hexdigest())

    def test_finite_thresholds_exclude_nonfinite_and_are_strict(self):
        value = torch.tensor([[1e6, 2e6], [1e21, 1e31], [float('inf'), float('nan')]], dtype=torch.bfloat16)
        result = scan_bf16(value, chunk_rows=1)
        self.assertEqual(result['counts']['finite_extreme'], 3)
        self.assertEqual(result['counts']['finite_gt_1e20'], 2)
        self.assertEqual(result['counts']['finite_gt_1e30'], 1)
        self.assertEqual(result['counts']['any_anomaly'], 5)
        self.assertEqual(result['feature_counts']['nonfinite'], [1, 1])

    def test_chunk_boundaries_preserve_histogram_hash_rows_and_counts(self):
        value = torch.tensor([[0, 1], [2e7, -2e7], [0, -0.0], [float('inf'), 2], [1e35, 3]], dtype=torch.bfloat16)
        first, second = scan_bf16(value, chunk_rows=1), scan_bf16(value, chunk_rows=4)
        first.pop('chunk_rows'); second.pop('chunk_rows')
        self.assertEqual(first, second)
        self.assertEqual(first['rows']['any_anomaly']['contiguous_ranges'], [[1, 1], [3, 4]])


if __name__ == '__main__':
    unittest.main()
