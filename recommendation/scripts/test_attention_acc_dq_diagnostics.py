"""CPU checks that diagnostic JIT variants cannot mutate production dependencies."""

import unittest

import torch

from generative_recommenders.ops.triton import triton_hstu_attention as attention

from attention_acc_dq_diagnostics import VARIANTS, make_diagnostic_kernel


class DiagnosticKernelTests(unittest.TestCase):
    def test_baseline_preserves_signature_source_and_cache_key(self):
        original = attention._hstu_attn_bwd.fn
        clone = make_diagnostic_kernel(attention, "baseline")
        self.assertIsNot(clone, original)
        self.assertEqual(clone.signature, original.signature)
        self.assertEqual(clone.src, original.src)
        self.assertEqual(clone.cache_key, original.cache_key)
        self.assertIsNot(clone.device_caches, original.device_caches)

    def test_variants_are_stable_and_independent_of_build_order(self):
        kernels = {name: make_diagnostic_kernel(attention, name) for name in VARIANTS}
        self.assertEqual(len({fn.cache_key for fn in kernels.values()}), len(VARIANTS))
        for name in reversed(VARIANTS):
            repeated = make_diagnostic_kernel(attention, name)
            self.assertEqual(repeated.cache_key, kernels[name].cache_key)
            self.assertIsNot(repeated.fn.__globals__, kernels[name].fn.__globals__)

    def test_production_references_and_gpu_state_are_preserved(self):
        wrapper = attention._hstu_attn_bwd
        raw = wrapper.fn
        block = attention._hstu_attn_bwd_one_block
        col = attention._hstu_attn_bwd_one_col_block
        acc = block.fn.__globals__["acc_dq"]
        prehooks = [cfg.pre_hook for cfg in wrapper.configs]
        rng = torch.random.get_rng_state().clone()
        self.assertFalse(torch.cuda.is_initialized())
        for variant in VARIANTS:
            make_diagnostic_kernel(attention, variant)
        self.assertIs(attention._hstu_attn_bwd, wrapper)
        self.assertIs(wrapper.fn, raw)
        self.assertIs(raw.fn.__globals__["_hstu_attn_bwd_one_col_block"], col)
        self.assertIs(col.fn.__globals__["_hstu_attn_bwd_one_block"], block)
        self.assertIs(block.fn.__globals__["acc_dq"], acc)
        self.assertEqual([cfg.pre_hook for cfg in wrapper.configs], prehooks)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertFalse(torch.cuda.is_initialized())

    def test_reject_unknown_variant_before_touching_kernel(self):
        with self.assertRaisesRegex(ValueError, "unknown DQ diagnostic"):
            make_diagnostic_kernel(None, "mistyped")


if __name__ == "__main__":
    unittest.main()
