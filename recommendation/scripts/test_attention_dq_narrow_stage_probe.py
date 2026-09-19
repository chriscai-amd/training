"""CPU verification of narrow stage bits, workspace bounds and launch semantics."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import triton
import triton.language as tl

from generative_recommenders.ops.triton import triton_hstu_attention as attention

import attention_dq_narrow_stage_probe as narrow


def fixture(stage, *, forced=False, pid=3):
    spec = narrow.workspace_spec(4, 4, 4, 2, stage)
    stage = spec["selected_stage"]
    pristine = {"k": torch.arange(28).reshape(7, 2, 2).to(torch.bfloat16),
                "seq_offsets": torch.tensor([0, 3, 7]),
                "sort_by_length_indices": torch.tensor([1, 0])}
    raw = torch.zeros(spec["total_words"], dtype=torch.int32)
    raw[0], raw[1] = pid + 1, 1
    row = raw[8:].view(4, spec["program_stride_words"])[pid]
    seq = (1, 0)[pid // 2]
    start = int(pristine["seq_offsets"][seq])
    length = int(pristine["seq_offsets"][seq + 1]) - start
    query, key = (0, 0) if forced else (length - 2, length - 1)
    valid_m, valid_n = min(4, length - query), min(4, length - key)
    row[:9] = torch.tensor([narrow.STAGES.index(stage) + 1, pid, query, key,
                            length, valid_m, narrow.MAGIC, int(forced), 1 if forced else 2])
    panels = {}
    for panel in spec["panels"]:
        dtype = torch.float32 if panel["dtype"] == "torch.float32" else torch.bfloat16
        value = torch.zeros(panel["shape"], dtype=dtype)
        if panel["name"] == "k":
            value[:valid_n] = pristine["k"][start + key:start + key + valid_n, pid % 2]
        elif forced:
            value[0, 0] = 1
        else:
            value[0, 0], value[1, 1] = float("nan"), -0.5
            value[0, 3] = float("nan")  # Masked query lane does not count.
        panels[panel["name"]] = value
        bits = ((value.view(torch.int16).to(torch.int32) & 65535)
                if dtype == torch.bfloat16 else value.view(torch.int32))
        offset = panel["offset_words"]
        row[offset:offset + panel["words"]] = bits.flatten()
    return raw, spec, pristine, panels


class NarrowWorkspaceTests(unittest.TestCase):
    def test_production_size_and_last_slot_end_are_bounded(self):
        for stage in narrow.STAGES:
            spec = narrow.workspace_spec(128, 32, 64, 128, stage)
            selected_words = 64 * 32 if stage == "dqk_trans" else 128 * 32
            self.assertEqual(spec["program_stride_words"], 16 + 64 * 128 + selected_words)
            panel = spec["panels"][-1]
            last = 8 + 127 * spec["program_stride_words"] + panel["offset_words"] + panel["words"]
            self.assertEqual(last, spec["total_words"])
            self.assertLess(spec["bytes"], 7 << 20)
        self.assertEqual(narrow.workspace_spec(128, 32, 64, 128, "dqk")["bytes"], 5251104)
        with self.assertRaises(ValueError):
            narrow.workspace_spec(0, 32, 64, 128, "dot")

    def test_exact_bits_counts_masks_and_k_attribution(self):
        for stage in narrow.STAGES:
            raw, spec, pristine, panels = fixture(stage)
            decoded = narrow.decode_workspace(raw, spec, pristine)
            record, = decoded["records"]
            self.assertEqual(decoded["first_claimed_program"], 3)
            self.assertEqual((record["sequence"], record["head"]), (0, 1))
            self.assertEqual(record["selected_stage"], stage)
            self.assertEqual(record["nonzero_count"], 2)
            self.assertEqual(len(record["samples"]), 2)
            self.assertTrue(record["k_tile_matches_pristine_global_input"])
            for name, expected in panels.items():
                view = torch.int32 if expected.dtype == torch.float32 else torch.int16
                self.assertTrue(torch.equal(record["panels"][name].view(view), expected.view(view)))

    def test_forced_panel_is_explicitly_labeled_and_not_a_spontaneous_failure(self):
        for stage in narrow.STAGES:
            raw, spec, pristine, _ = fixture(stage, forced=True)
            result = narrow.decode_workspace(raw, spec, pristine, force_positive_control=True)
            record, = result["records"]
            self.assertTrue(record["force_positive_control"])
            self.assertTrue(record["control_sentinel_only"])
            with self.assertRaisesRegex(ValueError, "header"):
                narrow.decode_workspace(raw, spec, pristine)
            self.assertTrue(narrow.valid_control_capture(result, expected_programs=1, calls=1))
            self.assertFalse(narrow.valid_control_capture(result, expected_programs=4, calls=1))
            self.assertFalse(narrow.valid_control_capture(result, expected_programs=1, calls=2))
            record["k_tile_matches_pristine_global_input"] = False
            self.assertFalse(narrow.valid_control_capture(result, expected_programs=1, calls=1))

    def test_incomplete_count_and_payload_corruption_are_rejected(self):
        for offset, value, message in ((0, -1, "header"), (1, 0, "header"),
                                        (4, 999, "coordinates"), (5, 3, "mask"),
                                        (7, 1, "header"), (8, 1, "count"),
                                        (16, 65536, "high bits")):
            raw, spec, pristine, _ = fixture("dqk")
            raw[8 + 3 * spec["program_stride_words"] + offset] = value
            with self.assertRaisesRegex(ValueError, message):
                narrow.decode_workspace(raw, spec, pristine)
        raw, spec, pristine, _ = fixture("dot")
        raw.zero_()
        self.assertEqual(narrow.decode_workspace(raw, spec, pristine)["records"], [])
        raw[0] = 1
        with self.assertRaisesRegex(ValueError, "claims"):
            narrow.decode_workspace(raw, spec, pristine)


class NarrowCloneTests(unittest.TestCase):
    def test_clones_are_distinct_reproducible_and_leave_original_frozen(self):
        wrapper = attention._hstu_attn_bwd
        original = wrapper.fn
        source = attention._hstu_attn_bwd_one_block.src
        rng = torch.get_rng_state().clone()
        keys = []
        for stage in narrow.STAGES:
            for force in (False, True):
                clone = narrow.make_narrow_probe_kernel(attention, stage, force_positive_control=force)
                repeat = narrow.make_narrow_probe_kernel(attention, stage, force_positive_control=force)
                self.assertEqual(clone.arg_names, original.arg_names)
                self.assertEqual(clone.cache_key, repeat.cache_key)
                self.assertEqual(clone.diagnostic_metadata["selected_stage"], stage)
                self.assertEqual(clone.diagnostic_metadata["force_positive_control"], force)
                block = clone.fn.__globals__["_hstu_attn_bwd_one_col_block"].fn.__globals__["_hstu_attn_bwd_one_block"]
                self.assertIn(f"STAGE={narrow.STAGES.index(stage)}", block.src)
                self.assertIn(f"FORCE={force}", block.src)
                keys.append(clone.cache_key)
        self.assertEqual(len(set(keys)), 6)
        self.assertIs(wrapper.fn, original)
        self.assertEqual(attention._hstu_attn_bwd_one_block.src, source)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(torch.cuda.is_initialized())

    def test_scoped_install_restores_after_failure(self):
        original, raw = attention._hstu_attn_bwd, attention._hstu_attn_bwd.fn
        controller = narrow.NarrowStageController("unused.pt", 1 << 20, "dqk")
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with controller.install(attention, "production"):
                self.assertIsInstance(attention._hstu_attn_bwd, narrow.NarrowLaunchProxy)
                raise RuntimeError("stop")
        self.assertIs(attention._hstu_attn_bwd, original)
        self.assertIs(original.fn, raw)


class NarrowLaunchTests(unittest.TestCase):
    def configure(self, path, *, forced=False, stage="dqk"):
        raw, spec, pristine, _ = fixture(stage, forced=forced)
        controller = narrow.NarrowStageController(path, 1 << 20, stage, force_positive_control=forced)
        controller.pristine = pristine
        controller.config = SimpleNamespace(kwargs={"BLOCK_M": 4, "BLOCK_N": 4, "SEQUENCE_PARALLEL": False})
        controller.attention = SimpleNamespace(_hstu_attn_bwd=SimpleNamespace(device_caches={}))
        controller.source_capture, controller.expected_digests, controller.configuration = "fake.pt", {}, {}
        kwargs = {"Q": torch.zeros(7, 2, 2, dtype=torch.bfloat16), "Z": 2, "H": 2, "DimQ": 2,
                  "LOCK": torch.ones(1, dtype=torch.int32)}
        return controller, raw, spec, kwargs

    def test_metadata_published_before_launch_and_first_positive_bytes_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "narrow.pt"
            controller, raw, spec, kwargs = self.configure(path)
            events = []
            controller.report = {}
            controller.publish = events.append
            launches = []

            def fake(**actual):
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["launch_grid"], [4, 1])
                pointer = events[0]["kernel_arg_addresses"]["LOCK"]
                self.assertEqual(pointer["data_ptr"], actual["LOCK"].data_ptr())
                self.assertEqual(pointer["storage_bytes"], spec["bytes"])
                self.assertEqual(int(actual["LOCK"].count_nonzero()), 0)
                launches.append(1)
                if len(launches) == 2:
                    actual["LOCK"].copy_(raw)
                return "ok"

            grid = lambda meta: (meta["Z"] * meta["H"], 1)
            self.assertEqual(controller.launch(fake, kwargs, grid=grid), "ok")
            with self.assertRaises(narrow.StageProbePositive):
                controller.launch(fake, kwargs, grid=grid)
            saved = torch.load(path, weights_only=False)
            self.assertTrue(torch.equal(saved["workspace_raw_int32"], raw))
            self.assertEqual(saved["format"], "hstu_dq_narrow_stage_probe_v1")
            self.assertEqual(saved["iteration"], 2)
            self.assertEqual(saved["launch_metadata"], controller.report["narrow_stage_allocation"])
            self.assertFalse(saved["force_positive_control"])
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_forced_positive_capture_is_labeled_and_invalid_workspace_is_preserved(self):
        for forced, corrupt in ((True, False), (False, True)):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "narrow.pt"
                controller, raw, _, kwargs = self.configure(path, forced=forced, stage="dot")
                if corrupt:
                    raw[1] = 2
                with self.assertRaises(narrow.StageProbePositive):
                    controller.launch(lambda **actual: actual["LOCK"].copy_(raw), kwargs, grid=(4, 1))
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved["force_positive_control"], forced)
                self.assertEqual("decode_error" in saved, corrupt)
                if forced:
                    self.assertIn("FORCED DIAGNOSTIC CONTROL", saved["scope"])
                self.assertTrue(torch.equal(saved["workspace_raw_int32"], raw))

    def test_grid_and_budget_reject_before_launch(self):
        controller, _, _, kwargs = self.configure("unused.pt")
        for grid in ((5, 1), (4, 2), (4, 1, 2)):
            with self.assertRaisesRegex(ValueError, "grid"):
                controller.launch(lambda **_: self.fail("must not launch"), kwargs, grid=grid)
        controller.max_workspace_bytes = 1
        with self.assertRaisesRegex(ValueError, "workspace needs"):
            controller.launch(lambda **_: self.fail("must not launch"), kwargs, grid=(4, 1))

    def test_cli_stage_alias_control_and_delegated_args(self):
        with tempfile.TemporaryDirectory() as temp:
            parsed = narrow.arguments(["--stage", "dqk", "--force-positive-control",
                                       "--stage-dump", str(Path(temp) / "control.pt"), "--",
                                       "capture.pt", "--repeat", "1", "--dout-zero"])
        self.assertEqual(parsed.stage, "dqk_trans")
        self.assertTrue(parsed.force_positive_control)
        self.assertEqual(parsed.raw_args, ["capture.pt", "--repeat", "1", "--dout-zero"])


@triton.jit
def offline_compile_entry(DQ, K, G, LOCK, seq_len, BM: tl.constexpr, BN: tl.constexpr,
                           D: tl.constexpr, STAGE: tl.constexpr, FORCE: tl.constexpr):
    """Compile-only entry; this test module never invokes a GPU launch."""
    m, n, d = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, D)
    k = tl.load(K + n[:, None] * D + d[None, :])
    g = tl.load(G + n[:, None] * BM + m[None, :])
    pointers = DQ + d[:, None] + m[None, :] * D
    narrow._narrow_acc_dq(pointers, 0, D, k, g, 0.125, m < seq_len, seq_len,
                          LOCK, BM, False, False, 0, seq_len, STAGE, FORCE)


if __name__ == "__main__":
    unittest.main()
