"""CPU checks for stage capture layout, exact decoding and isolated launch state."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from generative_recommenders.ops.triton import triton_hstu_attention as attention

from attention_dq_stage_probe import (
    GLOBAL_WORDS, MAGIC, STAGES, LaunchProxy, StageController, StageProbePositive,
    _stage_acc_dq, decode_workspace, make_stage_probe_kernel, workspace_spec,
)


def synthetic_capture(*, stage="dot", pid=3, sorted_sequences=True):
    spec = workspace_spec(4, 4, 4, 2)
    pristine = {
        "k": torch.arange(28).reshape(7, 2, 2).to(torch.bfloat16),
        "seq_offsets": torch.tensor([0, 3, 7]),
        "sort_by_length_indices": torch.tensor([1, 0]) if sorted_sequences else None,
    }
    raw = torch.zeros(spec["total_words"], dtype=torch.int32)
    raw[0], raw[1] = pid + 1, 1
    row = raw[GLOBAL_WORDS:].view(4, spec["program_stride_words"])[pid]
    seq = (1, 0)[pid // 2] if sorted_sequences else pid // 2
    start = int(pristine["seq_offsets"][seq])
    length = int(pristine["seq_offsets"][seq + 1]) - start
    query, key = length - 2, length - 1
    row[:7] = torch.tensor([STAGES.index(stage) + 1, pid, query, key, length, 2, MAGIC])
    panels = {}
    for panel in spec["panels"]:
        dtype = torch.bfloat16 if panel["dtype"] == "torch.bfloat16" else torch.float32
        value = torch.zeros(panel["shape"], dtype=dtype)
        if panel["name"] == "k":
            value[0] = pristine["k"][start + key, pid % 2]
        elif panel["name"] == stage:
            value[0, 0] = float("nan")
            value[1, 1] = 0.5
            # Invalid query lanes must not be counted, including NaNs there.
            value[0, 3] = float("nan")
        panels[panel["name"]] = value
        bits = value.view(torch.int16).to(torch.int32) & 65535 if dtype == torch.bfloat16 else value.view(torch.int32)
        offset = panel["offset_words"]
        row[offset:offset + panel["words"]] = bits.flatten()
    row[8 + STAGES.index(stage)] = 2
    return raw, spec, pristine, panels


class WorkspaceTests(unittest.TestCase):
    def test_layout_has_contiguous_panels_and_expected_bound(self):
        spec = workspace_spec(128, 32, 64, 128)
        self.assertEqual(spec["program_stride_words"], 16 + 64 * 128 + 64 * 32 + 6 * 128 * 32)
        previous_end = 16
        for panel in spec["panels"]:
            self.assertEqual(panel["offset_words"], previous_end)
            self.assertEqual(panel["words"], panel["shape"][0] * panel["shape"][1])
            previous_end += panel["words"]
        self.assertEqual(previous_end, spec["program_stride_words"])
        self.assertEqual(spec["bytes"], 4 * (8 + 128 * previous_end))
        with self.assertRaisesRegex(ValueError, "positive"):
            workspace_spec(0, 32, 64, 128)

    def test_exact_panels_stage_counts_masking_and_sequence_attribution(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                raw, spec, pristine, panels = synthetic_capture(stage=stage)
                decoded = decode_workspace(raw, spec, pristine)
                self.assertEqual(decoded["first_claimed_program"], 3)
                record, = decoded["records"]
                self.assertEqual((record["sequence"], record["head"]), (0, 1))
                self.assertEqual((record["query_start"], record["key_start"]), (1, 2))
                self.assertEqual(record["first_checked_nonzero_stage"], stage)
                self.assertEqual(record["stage_counts"][stage], 2)
                self.assertTrue(record["k_tile_matches_pristine_global_input"])
                self.assertEqual(len(record["samples"]), 2)
                for name, original in panels.items():
                    actual = record["panels"][name]
                    view_dtype = torch.int16 if original.dtype == torch.bfloat16 else torch.int32
                    self.assertTrue(torch.equal(actual.view(view_dtype), original.view(view_dtype)), name)

    def test_unsorted_attribution_and_k_mismatch_are_explicit(self):
        raw, spec, pristine, _ = synthetic_capture(sorted_sequences=False)
        pristine["k"][6, 1, 0] += 1
        record, = decode_workspace(raw, spec, pristine)["records"]
        self.assertEqual(record["sequence"], 1)
        self.assertEqual(record["global_query_start"], 5)
        self.assertFalse(record["k_tile_matches_pristine_global_input"])

    def test_empty_workspace(self):
        raw, spec, pristine, _ = synthetic_capture()
        raw.zero_()
        self.assertEqual(decode_workspace(raw, spec, pristine), {"first_claimed_program": None, "records": []})
        raw[0] = 1
        with self.assertRaisesRegex(ValueError, "claims disagree"):
            decode_workspace(raw, spec, pristine)

    def test_rejects_incomplete_headers_bad_counts_and_bit_ranges(self):
        cases = ((0, -1, "header"), (1, 0, "header"), (4, 999, "coordinates"),
                 (5, 3, "mask"), (6, 0, "header"), (10, 1, "counts"),
                 (0, 4, "counts"), (16, 65536, "high bits"))
        for offset, value, message in cases:
            with self.subTest(offset=offset, value=value):
                raw, spec, pristine, _ = synthetic_capture()
                raw[GLOBAL_WORDS + 3 * spec["program_stride_words"] + offset] = value
                with self.assertRaisesRegex(ValueError, message):
                    decode_workspace(raw, spec, pristine)
        raw, spec, pristine, _ = synthetic_capture()
        raw[1] = 2
        with self.assertRaisesRegex(ValueError, "claims disagree"):
            decode_workspace(raw, spec, pristine)
        with self.assertRaisesRegex(ValueError, "layout"):
            decode_workspace(raw.to(torch.int64), spec, pristine)


class CloneTests(unittest.TestCase):
    def test_clone_signature_metadata_and_unchanged_production_dependencies(self):
        wrapper, raw = attention._hstu_attn_bwd, attention._hstu_attn_bwd.fn
        block, col = attention._hstu_attn_bwd_one_block, attention._hstu_attn_bwd_one_col_block
        acc = block.fn.__globals__["acc_dq"]
        source, key = block.src, raw.cache_key
        prehooks = [config.pre_hook for config in wrapper.configs]
        rng = torch.random.get_rng_state().clone()
        self.assertFalse(torch.cuda.is_initialized())
        clone = make_stage_probe_kernel(attention)
        repeated = make_stage_probe_kernel(attention)
        self.assertEqual(clone.signature, raw.signature)
        self.assertEqual(clone.arg_names, raw.arg_names)
        self.assertEqual(clone.cache_key, repeated.cache_key)
        self.assertNotEqual(clone.cache_key, key)
        cloned_col = clone.fn.__globals__["_hstu_attn_bwd_one_col_block"]
        cloned_block = cloned_col.fn.__globals__["_hstu_attn_bwd_one_block"]
        self.assertIn("key_start=tl.min(offs_n, 0)", cloned_block.src)
        self.assertIn("seq_len=seq_len", cloned_block.src)
        self.assertIsNot(cloned_block.fn.__globals__["acc_dq"], acc)
        self.assertIs(attention._hstu_attn_bwd, wrapper)
        self.assertIs(wrapper.fn, raw)
        self.assertIs(raw.fn.__globals__["_hstu_attn_bwd_one_col_block"], col)
        self.assertIs(col.fn.__globals__["_hstu_attn_bwd_one_block"], block)
        self.assertIs(block.fn.__globals__["acc_dq"], acc)
        self.assertEqual(block.src, source)
        self.assertEqual(raw.cache_key, key)
        self.assertEqual([config.pre_hook for config in wrapper.configs], prehooks)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertFalse(torch.cuda.is_initialized())

    def test_install_restores_wrapper_after_exception(self):
        wrapper, raw = attention._hstu_attn_bwd, attention._hstu_attn_bwd.fn
        controller = StageController("unused.pt", 1 << 20)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with controller.install(attention, "production"):
                self.assertIsInstance(attention._hstu_attn_bwd, LaunchProxy)
                self.assertIsNot(wrapper.fn, raw)
                raise RuntimeError("stop")
        self.assertIs(attention._hstu_attn_bwd, wrapper)
        self.assertIs(wrapper.fn, raw)
        with self.assertRaisesRegex(ValueError, "separate"):
            with controller.install(attention, "dot_only"):
                pass


class LaunchTests(unittest.TestCase):
    def configure(self, path):
        raw, spec, pristine, _ = synthetic_capture()
        controller = StageController(path, 1 << 20)
        controller.pristine = pristine
        controller.config = SimpleNamespace(kwargs={"BLOCK_M": 4, "BLOCK_N": 4, "SEQUENCE_PARALLEL": False})
        controller.attention = SimpleNamespace(_hstu_attn_bwd=SimpleNamespace(device_caches={}))
        controller.source_capture, controller.expected_digests, controller.configuration = "fake.pt", {}, {}
        kwargs = {"Q": torch.zeros(7, 2, 2, dtype=torch.bfloat16), "Z": 2, "H": 2, "DimQ": 2,
                  "LOCK": torch.ones(1, dtype=torch.int32)}
        return controller, raw, spec, kwargs

    def test_cpu_fake_launcher_stops_on_first_positive_and_preserves_bits(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stage.pt"
            controller, raw, spec, kwargs = self.configure(path)
            calls = []

            def fake(**actual):
                self.assertEqual(int(actual["LOCK"].count_nonzero()), 0)
                self.assertIsNot(actual["LOCK"], kwargs["LOCK"])
                calls.append(1)
                if len(calls) == 2:
                    actual["LOCK"].copy_(raw)
                return "completed"

            self.assertEqual(controller.launch(fake, kwargs), "completed")
            self.assertFalse(path.exists())
            with self.assertRaises(StageProbePositive):
                for _ in range(10):
                    controller.launch(fake, kwargs)
            self.assertEqual(len(calls), 2)
            saved = torch.load(path, weights_only=False)
            self.assertTrue(torch.equal(saved["workspace_raw_int32"], raw))
            self.assertEqual(saved["workspace_spec"], spec)
            self.assertEqual(controller.positive["iteration"], 2)
            self.assertEqual(int(kwargs["LOCK"][0]), 1)
            self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_workspace_is_saved_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.pt"
            controller, raw, _, kwargs = self.configure(path)
            raw[1] = 2
            with self.assertRaises(StageProbePositive):
                controller.launch(lambda **actual: actual["LOCK"].copy_(raw), kwargs)
            saved = torch.load(path, weights_only=False)
            self.assertTrue(torch.equal(saved["workspace_raw_int32"], raw))
            self.assertIn("claims disagree", saved["decode_error"])
            self.assertIn("decode_error", controller.positive)

    def test_workspace_budget_rejects_before_launch(self):
        controller, _, _, kwargs = self.configure("unused.pt")
        controller.max_workspace_bytes = 1
        with self.assertRaisesRegex(ValueError, "workspace needs"):
            controller.launch(lambda **_: self.fail("must not launch"), kwargs)

    def test_proxy_forwards_grid_attributes_and_named_arguments(self):
        class FakeKernel:
            name = "fake"

            def __getitem__(self, grid):
                self.grid = grid
                return lambda **kwargs: kwargs

        original = FakeKernel()
        controller = SimpleNamespace(launch=lambda fn, kwargs: fn(**kwargs))
        proxy = LaunchProxy(original, controller)
        self.assertEqual(proxy.name, "fake")
        self.assertEqual(proxy[(4, 1)](Q="q"), {"Q": "q"})
        self.assertEqual(original.grid, (4, 1))
        with self.assertRaisesRegex(ValueError, "named"):
            proxy[(4, 1)]("q")


@triton.jit
def offline_compile_entry(DQ, K, G, LOCK, seq_len, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    """Compile-only entry; this test module never invokes a GPU launch."""
    m, n, d = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, D)
    k = tl.load(K + n[:, None] * D + d[None, :])
    g = tl.load(G + n[:, None] * BM + m[None, :])
    pointers = DQ + d[:, None] + m[None, :] * D
    _stage_acc_dq(pointers, 0, D, k, g, 0.125, m < seq_len, seq_len,
                  LOCK, BM, False, False, 0, seq_len)


if __name__ == "__main__":
    unittest.main()
