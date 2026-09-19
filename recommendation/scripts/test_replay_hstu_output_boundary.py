#!/usr/bin/env python3
"""CPU-only checks of independently replayable HSTU output-stage evidence."""
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import replay_hstu_output_boundary as replay


def norm_fixture(concat_u=True, concat_x=True, silu_u=False, activation="none", training=True):
    n, d = 5, 7
    raw = torch.arange(n * d * 3, dtype=torch.float64).reshape(n, d * 3)
    x = (raw[:, 1::3] - 30) / 19
    u = (raw[:, 2::3] - 43) / 23
    weight = torch.linspace(-0.4, 1.1, d, dtype=torch.float64)
    bias = torch.linspace(-0.1, 0.2, d, dtype=torch.float64)
    p = d * (1 + concat_u + concat_x)
    dy = torch.arange(n * p * 2, dtype=torch.float64).reshape(n, p * 2)[:, 1::2] / 300 - .3
    packed = (torch.arange(n * d).reshape(n, d) % (1 << (1 + concat_u + concat_x))).to(torch.int8)
    eps = 1e-5
    values = dict(x=x, u=u, weight=weight, bias=bias, dy=dy,
                  mean=x.mean(1), rstd=torch.rsqrt(x.var(1, unbiased=False) + eps),
                  BLOCK_D=8, num_warps=1, eps=eps, training=training,
                  dropout_ratio=.25, seed=417, silu_u=silu_u, concat_u=concat_u,
                  concat_x=concat_x, mul_u_activation_type=activation,
                  compute_y=True, random_mask=packed if training else None)
    # Independent differentiable forward; autograd supplies the full gradient.
    xx, uu, ww, bb = [value.detach().clone().requires_grad_() for value in (x, u, weight, bias)]
    ln = torch.nn.functional.layer_norm(xx, (d,), ww, bb, eps)
    mul = {"none": lambda z: z, "silu": torch.nn.functional.silu, "sigmoid": torch.sigmoid}[activation](uu)
    parts = []
    if concat_u:
        uv = torch.nn.functional.silu(uu) if silu_u else uu
        if training:
            uv = torch.where((packed & (4 if concat_x else 2)) != 0, uv / .75, 0.)
        parts.append(uv)
    if concat_x:
        xv = torch.where((packed & 2) != 0, xx / .75, 0.) if training else xx
        parts.append(xv)
    yv = ln * mul
    if training:
        yv = torch.where((packed & 1) != 0, yv / .75, 0.)
    parts.append(yv)
    y = torch.cat(parts, 1)
    grads = torch.autograd.grad(y, (xx, uu, ww, bb), dy)
    return {"operation": replay.NORM, "args": (), "kwargs": values,
            "outputs": (*grads, y.detach()), "stage": "output"}


class OutputBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state()), "CPU analysis consumed global RNG"
        assert not torch.cuda.is_initialized(), "CPU tests initialized CUDA"

    def test_norm_against_autograd_all_production_options(self):
        for concat_u, concat_x, silu_u, training, activation in itertools.product(
                (False, True), (False, True), (False, True), (False, True), ("none", "silu", "sigmoid")):
            with self.subTest(concat_u=concat_u, concat_x=concat_x, silu_u=silu_u,
                              training=training, activation=activation):
                payload = norm_fixture(concat_u, concat_x, silu_u, activation, training)
                report = replay.analyze(payload, rows="all", max_rows=0, row_chunk=2, chunk_bytes=16384)
                self.assertTrue(report["supported"])
                for summary in report["references"].values():
                    self.assertLess(summary["max_abs_error_vs_fp64_finite_pairs"], 1e-12)
                    self.assertEqual(summary["elements"], summary["compared_elements"])

    def test_partial_reductions_are_not_compared_with_full_gradients(self):
        report = replay.analyze(norm_fixture(), rows="1,3:5", row_chunk=1, chunk_bytes=16384)
        self.assertEqual(report["selection"]["analyzed_rows"], 3)
        self.assertEqual(report["references"]["d_attn"]["compared_elements"], 21)
        for name in ("d_norm_weight", "d_norm_bias"):
            self.assertEqual(report["references"][name]["compared_elements"], 0)
            self.assertIn("partial", report["references"][name]["scope"])

    def test_input_trigger_and_optional_recomputed_y(self):
        payload = norm_fixture()
        payload["outputs"] = None
        payload["stage"] = "input"
        payload["kwargs"]["compute_y"] = False
        report = replay.analyze(payload, rows="0:2", chunk_bytes=16384)
        self.assertIsNone(report["references"]["y"])
        self.assertEqual(report["references"]["d_attn"]["compared_elements"], 0)
        payload["outputs"] = (torch.zeros(5, 7),) * 4
        with self.assertRaisesRegex(ValueError, "Expected 5"):
            replay.bind_inputs(payload)

    def test_training_mask_is_required_even_when_ratio_zero(self):
        payload = norm_fixture()
        for mask in (None, torch.empty(0)):
            for ratio in (.25, 0.):
                payload["kwargs"].update(random_mask=mask, dropout_ratio=ratio)
                report = replay.analyze(payload, rows="0:1", chunk_bytes=16384)
                self.assertFalse(report["supported"])
                self.assertIn("GPU stage replay remains available", report["limitation"])
                with self.assertRaisesRegex(ValueError, "saved packed random_mask"):
                    replay.norm_reference_rows(payload["kwargs"], torch.tensor([0]))

    def test_saved_statistics_are_used(self):
        payload = norm_fixture(training=False)
        ids = torch.tensor([0, 2])
        baseline = replay.norm_reference_rows(payload["kwargs"], ids)
        payload["kwargs"]["eps"] = 99.0
        unchanged = replay.norm_reference_rows(payload["kwargs"], ids)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, unchanged)))
        payload["kwargs"]["rstd"] = payload["kwargs"]["rstd"] * 2
        changed = replay.norm_reference_rows(payload["kwargs"], ids)
        self.assertFalse(torch.equal(baseline[0], changed[0]))

    def test_mask_bits_and_production_u_is_not_activated_twice(self):
        payload = norm_fixture()
        values = payload["kwargs"]
        values.update(x=torch.tensor([[1., -1.]], dtype=torch.float64),
                      u=torch.tensor([[2., 3.]], dtype=torch.float64),
                      dy=torch.tensor([[11., 13., 17., 19., 23., 29.]], dtype=torch.float64),
                      weight=torch.ones(2, dtype=torch.float64), bias=torch.zeros(2, dtype=torch.float64),
                      mean=torch.zeros(1, dtype=torch.float64), rstd=torch.ones(1, dtype=torch.float64),
                      random_mask=torch.tensor([[4, 2]], dtype=torch.int8), dropout_ratio=0.)
        dx, du, dw, db, y = replay.norm_reference_rows(values, torch.tensor([0]))
        torch.testing.assert_close(du, torch.tensor([[11., 0.]], dtype=torch.float64))
        torch.testing.assert_close(dx, torch.tensor([[0., 19.]], dtype=torch.float64))
        self.assertEqual(float(dw.abs().sum() + db.abs().sum()), 0.)
        torch.testing.assert_close(y, torch.tensor([[2., 0., 0., -1., 0., 0.]], dtype=torch.float64))

    def test_mm_reference_and_noncontiguous_layout(self):
        raw = torch.arange(200, dtype=torch.float64).reshape(10, 20) / 7
        dout, weight = raw[:6, 1:13:3], raw[6:, 2:14:3]
        payload = {"operation": replay.MM, "args": (), "kwargs": dict(dout=dout, output_weight=weight),
                   "outputs": (dout @ weight.T,), "stage": "finite"}
        report = replay.analyze(payload, rows="all", max_rows=0, row_chunk=2, chunk_bytes=8192)
        self.assertLess(report["references"]["dy"]["max_abs_error_vs_fp64_finite_pairs"], 1e-12)
        fresh, _ = replay.common.restore_raw_tree(payload["kwargs"], device="cpu", chunk_bytes=13)
        self.assertEqual(fresh["dout"].stride(), dout.stride())
        self.assertEqual(fresh["dout"].storage_offset(), dout.storage_offset())
        self.assertEqual(fresh["dout"].untyped_storage()._cdata, fresh["output_weight"].untyped_storage()._cdata)
        pristine = replay.input_digests(payload["kwargs"], chunk_bytes=64, threshold=1e20)
        self.assertEqual(pristine, replay.input_digests(fresh, chunk_bytes=64, threshold=1e20))
        full = torch.empty(0, dtype=torch.float64).set_(fresh["dout"].untyped_storage(), 0, (200,), (1,))
        full[0] = -3  # Outside both logical views: raw digest must detect this.
        mutated = replay.input_digests(fresh, chunk_bytes=64, threshold=1e20)
        self.assertEqual(pristine["inputs"], mutated["inputs"])
        self.assertNotEqual(pristine["raw_storages"], mutated["raw_storages"])
        again, _ = replay.common.restore_raw_tree(payload["kwargs"], device="cpu", chunk_bytes=19)
        self.assertEqual(pristine, replay.input_digests(again, chunk_bytes=64, threshold=1e20))

    def test_extreme_inputs_show_expected_overflow(self):
        dtype = torch.bfloat16
        payload = {"operation": replay.MM, "args": (), "kwargs": {
            "dout": torch.tensor([[1e37, 1e37]], dtype=dtype),
            "output_weight": torch.full((3, 2), 100., dtype=dtype)}, "outputs": None}
        report = replay.analyze(payload, chunk_bytes=8192)
        self.assertEqual(report["selection"]["first_sample_rows"], [0])
        self.assertEqual(report["references"]["dy"]["cast_nonfinite"], 3)
        self.assertEqual(report["references"]["dy"]["compared_elements"], 0)

    def test_replay_orchestration_fresh_storage_hashes_and_corruption_stop(self):
        dout = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10
        weight = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 7
        payload = {"operation": replay.MM, "args": (), "kwargs": dict(dout=dout, output_weight=weight),
                   "outputs": (dout @ weight.T,)}
        raw_restore = replay.common.restore_raw_tree
        saved_pointers = []
        retained = []

        def cpu_restore(tree, **kwargs):
            result = raw_restore(tree, **{**kwargs, "device": "cpu"})
            saved_pointers.append(result[0]["dout"].data_ptr())
            retained.append(result[0])
            return result

        events = []
        with mock.patch.object(replay.common, "restore_raw_tree", side_effect=cpu_restore), mock.patch.object(torch.cuda, "synchronize"):
            reports = replay.replay_gpu(payload, repeats=2, rows="all", max_rows=0,
                                         chunk_bytes=8192, progress=events.append)
        self.assertNotEqual(saved_pointers[0], saved_pointers[1])
        for report in reports:
            self.assertTrue(report["matches_pristine_inputs"])
            self.assertEqual(report["pristine_input_digests"], report["input_digests"])
            self.assertEqual(report["matches_captured_logical_bytes"], [True])
            self.assertTrue(report["inputs_unchanged_by_operation"])
            self.assertLess(report["fp64_reference_comparison"]["references"]["dy"]["max_abs_error_vs_fp64_finite_pairs"], 1e-12)
        self.assertEqual(reports[1]["matches_first_repeat_logical_bytes"], [True])

        def corrupt_restore(tree, **kwargs):
            result = cpu_restore(tree, **kwargs)
            result[0]["dout"][0, 0] += 1
            return result

        with mock.patch.object(replay.common, "restore_raw_tree", side_effect=corrupt_restore), mock.patch.object(torch.cuda, "synchronize"):
            with self.assertRaisesRegex(RuntimeError, "hashes differ"):
                replay.replay_gpu(payload, repeats=1, rows="all", max_rows=0, chunk_bytes=8192)
        self.assertEqual(float(dout[0, 0]), 0.)

    def test_controls_restore_and_signed_infinity_comparison(self):
        before = replay.execution_controls()
        changed = {**before, "autocast": {**before["autocast"], "cpu": {
            "enabled": True, "dtype": "torch.bfloat16"}}}
        with replay.restored_execution_controls(changed) as info:
            self.assertTrue(info["verified"])
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertEqual(replay.execution_controls(), before)
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with replay.restored_execution_controls(changed):
                raise RuntimeError("test exception")
        self.assertEqual(replay.execution_controls(), before)
        summary = replay.ReferenceSummary(torch.bfloat16, scope="test", observed_available=True)
        summary.add(torch.tensor([[1e40, -1e40]], dtype=torch.float64),
                    torch.tensor([[-float("inf"), -float("inf")]], dtype=torch.bfloat16))
        self.assertEqual(summary.result()["nonfinite_pattern_mismatches_vs_cast_reference"], 1)

    def test_invalid_mask_is_not_a_computation_reference(self):
        payload = norm_fixture(concat_u=False, concat_x=False)
        payload["kwargs"]["random_mask"][0, 0] = 2
        report = replay.analyze(payload, rows="all", max_rows=0)
        self.assertFalse(report["supported"])
        self.assertIn("invalid bits", report["limitation"])
        payload = norm_fixture()
        payload["kwargs"]["weight"] = payload["kwargs"]["weight"].float()
        payload["outputs"] = None
        report = replay.analyze(payload, rows="0:1")
        self.assertEqual(report["references"]["d_norm_bias"]["output_dtype"], "torch.float32")

    def test_mask_and_selection_validation(self):
        payload = norm_fixture()
        payload["kwargs"]["random_mask"] = torch.ones(5, 7)
        with self.assertRaisesRegex(ValueError, "packed random_mask"):
            replay.bind_inputs(payload)
        for rows in ("-1", "0:6", "1:2:3"):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replay.analyze(norm_fixture(), rows=rows)
        with self.assertRaisesRegex(ValueError, "budget"):
            replay.analyze(norm_fixture(), rows="0:1", chunk_bytes=10)

    def test_roundtrip_and_cli_cpu_report(self):
        payload = norm_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            dump, report_path = Path(temporary) / "dump.pt", Path(temporary) / "report.json"
            torch.save(payload, dump)
            reloaded = torch.load(dump, map_location="cpu", weights_only=False, mmap=True)
            actual = replay.analyze(reloaded, rows="all", max_rows=0)
            self.assertLess(actual["references"]["d_u"]["max_abs_error_vs_fp64_finite_pairs"], 1e-12)
            with mock.patch("sys.argv", ["replay_hstu_output_boundary.py", str(dump), "--rows", "0:2", "--report", str(report_path)]), mock.patch("builtins.print"):
                replay.main()
            report = json.loads(report_path.read_text())
            self.assertEqual(report["gpu_validation"], "not requested")
            self.assertEqual(report["cpu_reference"]["selection"]["analyzed_rows"], 2)

    def test_allocator_context_default_and_explicit_empty_override(self):
        captured = {"PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "AMDGCN_USE_BUFFER_OPS": "0"}
        original = dict(captured)
        with mock.patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": "previous"}, clear=True):
            report = replay.restore_replay_environment(captured)
            self.assertEqual(report["environment_overrides"], {})
            self.assertEqual(report["effective_allocator_environment"], {
                "PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            self.assertNotIn("PYTORCH_ALLOC_CONF", os.environ)
            self.assertEqual(os.environ["AMDGCN_USE_BUFFER_OPS"], "0")
            report = replay.restore_replay_environment(captured, "")
            expected = {"PYTORCH_ALLOC_CONF": "", "PYTORCH_CUDA_ALLOC_CONF": ""}
            self.assertEqual(report["environment_overrides"], expected)
            self.assertEqual(report["effective_allocator_environment"], expected)
            self.assertEqual({key: os.environ[key] for key in expected}, expected)
            self.assertEqual(captured, original)
            # A later default invocation restores capture values, not prior overrides.
            self.assertEqual(replay.restore_replay_environment(captured)["environment_overrides"], {})
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_allocator_override_requires_gpu_before_torch_access(self):
        with mock.patch("sys.argv", ["replay_hstu_output_boundary.py", "unused.pt", "--allocator-config", ""]), \
                mock.patch.object(replay.common, "_torch", side_effect=AssertionError("Torch accessed before validation")), \
                mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                replay.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
