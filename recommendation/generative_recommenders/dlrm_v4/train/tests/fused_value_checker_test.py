"""CPU checks for fused-checker layout, fallback, and launch metadata contracts.

Kernel launches are mocked; these tests do not establish GPU correctness.
"""

import dataclasses
import importlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from generative_recommenders.dlrm_v4.train import fused_value_checker as helper


class FakeKernel:
    def __init__(self, status):
        self.status = status
        self.launch = None

    def __getitem__(self, grid):
        def run(tensor, output, numel, shape, strides, **options):
            self.launch = dict(grid=grid, tensor=tensor, numel=numel, shape=shape,
                               strides=strides, options=options)
            output.fill_(self.status)
            return SimpleNamespace(hash="exact-launch-hash", name="check_value_blocks",
                                   metadata=SimpleNamespace(num_warps=4, num_stages=1))
        return run


class CheckerTest(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_layout_views_including_actual_stride_forms(self):
        base = torch.arange(9 * 1024, dtype=torch.float32).reshape(9, 1024)
        values = (base, base.t(), base[:, 512:], base[:, :512], base[1::2, 1::3],
                  torch.tensor(2.).expand(17, 5), torch.tensor(2.), torch.empty(2, 0, 3))
        for value in values:
            metadata = helper.layout_metadata(tuple(value.shape), tuple(value.stride()),
                                              value.storage_offset(), value.element_size(),
                                              value.untyped_storage().nbytes())
            self.assertEqual(metadata["numel"], value.numel())
            self.assertEqual(metadata["storage_offset"], value.storage_offset())
            self.assertEqual(metadata["status_bytes"], (value.numel() + 4095) // 4096)
            if value.numel():
                raw = torch.empty(0, dtype=value.dtype).set_(value.untyped_storage(), 0,
                        (value.untyped_storage().nbytes() // value.element_size(),), (1,))
                last = value.storage_offset() + metadata["maximum_relative_element_offset"]
                self.assertEqual(float(raw[last]), float(value[tuple(n - 1 for n in value.shape)]))

    def test_signed_64_bit_metadata_without_large_allocations(self):
        # This is a logical broadcast of >2**31 elements from one stored value.
        metadata = helper.layout_metadata((65537, 32769), (0, 0), 0, 4, 4)
        self.assertGreater(metadata["numel"], 2**31)
        self.assertEqual(metadata["maximum_relative_element_offset"], 0)
        metadata = helper.layout_metadata((2, 3), (2**32, 1), 0, 2, 2 * (2**32 + 3))
        self.assertGreater(metadata["maximum_relative_element_offset"], 2**31)

    def test_invalid_layout_falls_out_before_addressing(self):
        for arguments in (((2,), (-1,), 0, 4, 8), ((2,), (2,), 0, 4, 8),
                          ((2,), (1,), 2, 4, 8), ((2**63,), (0,), 0, 4, 4)):
            with self.assertRaises(ValueError):
                helper.layout_metadata(*arguments)

    def test_all_packed_states_and_separate_bits(self):
        for status, expected in (([], (True, True)), ([0], (True, True)),
                                 ([1], (False, False)), ([2], (True, False)),
                                 ([3], (False, False)), ([1, 2], (False, False))):
            finite, bound = helper.reduce_status(torch.tensor(status, dtype=torch.uint8), True)
            self.assertEqual((bool(finite), bool(bound)), expected)
            finite_only, absent = helper.reduce_status(torch.tensor(status, dtype=torch.uint8), False)
            self.assertEqual(bool(finite_only), expected[0])
            self.assertIsNone(absent)

    def test_default_disabled_and_unsupported_never_import_triton(self):
        checker = helper.FusedValueChecker()
        with patch.object(importlib, "import_module", side_effect=AssertionError("unexpected import")):
            self.assertIsNone(checker.try_check(torch.ones(4), 1e20))
            checker.enabled = True
            self.assertIsNone(checker.try_check(torch.ones(4), 1e20))
            self.assertIsNone(checker.try_check(torch.ones(4, dtype=torch.complex64), 1e20))
            self.assertIsNone(checker.try_check(torch.ones(4, dtype=torch.int64), 1e20))
            self.assertIsNone(checker.try_check(torch.ones(4).to_sparse(), 1e20))
        self.assertEqual(checker.calls, 0)
        self.assertIsNone(checker._kernel)

    def test_mock_launch_preserves_view_and_records_exact_hash(self):
        checker = helper.FusedValueChecker(enabled=True)
        kernel = FakeKernel(2)
        checker._kernel = kernel
        value = torch.ones(9, 1024)[:, 512:]
        before = value.clone()
        with patch.object(helper, "unsupported_reason", return_value=None):
            finite, bound = checker.try_check(value, 1e20)
        self.assertTrue(bool(finite))
        self.assertFalse(bool(bound))
        self.assertTrue(torch.equal(value, before))
        self.assertIs(kernel.launch["tensor"], value)
        self.assertEqual(kernel.launch["strides"], (1024, 1))
        self.assertEqual(kernel.launch["options"]["BLOCK"], 4096)
        self.assertEqual(kernel.launch["options"]["num_warps"], 4)
        self.assertEqual(kernel.launch["options"]["num_stages"], 1)
        metadata = checker.metadata()
        self.assertEqual(metadata["last_launch"]["storage_offset"], 512)
        self.assertEqual(metadata["compiled"][0]["hash"], "exact-launch-hash")
        json.dumps(metadata, allow_nan=False)

    def test_empty_supported_path_skips_kernel_load(self):
        checker = helper.FusedValueChecker(enabled=True)
        with patch.object(helper, "unsupported_reason", return_value=None), \
                patch.object(checker, "_load_kernel", side_effect=AssertionError("empty launch")):
            finite, bound = checker.try_check(torch.empty(2, 0, 3), 1e20)
        self.assertTrue(bool(finite))
        self.assertTrue(bool(bound))
        self.assertEqual(checker.empty_calls, 1)

    def test_scalar_and_bound_disabled_mock_launch(self):
        checker = helper.FusedValueChecker(enabled=True)
        checker._kernel = FakeKernel(1)
        with patch.object(helper, "unsupported_reason", return_value=None):
            finite, bound = checker.try_check(torch.tensor(float("nan")), None)
        self.assertFalse(bool(finite))
        self.assertIsNone(bound)
        self.assertEqual(checker._kernel.launch["shape"], ())
        self.assertEqual(checker._kernel.launch["grid"], (1,))

    def test_metadata_handles_namedtuple_and_target_dataclass(self):
        @dataclasses.dataclass
        class Target:
            backend: str = "hip"
            arch: str = "gfx1250"
        encoded = helper.json_metadata({"target": Target(), "nested": (1, 2)})
        self.assertEqual(encoded["target"]["arch"], "gfx1250")
        json.dumps(encoded, allow_nan=False)

    def test_invalid_bounds_rejected_when_enabled(self):
        checker = helper.FusedValueChecker(enabled=True)
        for value in (True, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                checker.try_check(torch.ones(1), value)

    def test_out_of_range_fp32_compute_threshold_falls_back(self):
        checker = helper.FusedValueChecker(enabled=True)
        with patch.object(helper, "unsupported_reason", return_value=None), \
                patch.object(checker, "_load_kernel", side_effect=AssertionError("unexpected launch")):
            self.assertIsNone(checker.try_check(torch.tensor(float("inf")), 1e100))
        self.assertEqual(checker.fallbacks["threshold_outside_compute_dtype"], 1)

    def test_loaded_jit_source_hash_is_cached(self):
        checker = helper.FusedValueChecker(enabled=True)
        fake = SimpleNamespace(src="loaded kernel source")
        with patch.object(importlib, "import_module", return_value=SimpleNamespace(check_value_blocks=fake)):
            checker._load_kernel()
        before = checker.metadata()["kernel_jit_source_sha256"]
        fake.src = "later source change"
        self.assertEqual(checker.metadata()["kernel_jit_source_sha256"], before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
