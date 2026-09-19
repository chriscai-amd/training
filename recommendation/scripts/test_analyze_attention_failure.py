#!/usr/bin/env python3
"""CPU-only tests for mathematical isolation of corrupted history DQ."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import analyze_attention_failure as analyzer


def fixture():
    n, d, h, a, v = 6, 4, 1, 2, 2
    x = (torch.arange(n * d, dtype=torch.float64).reshape(n, d) - 11) / 9
    gamma = torch.tensor([.75, 1.25, -.5, 1.], dtype=torch.float64)
    beta = torch.tensor([.125, -.25, .375, .5], dtype=torch.float64)
    eps = 1e-5
    mean, rstd = x.mean(1), torch.rsqrt(x.var(1, unbiased=False) + eps)
    weight = (torch.arange(d * 8, dtype=torch.float64).reshape(d, 8) - 16) / 31
    offsets, targets, sort = torch.tensor([0, 3, 6]), torch.ones(2, dtype=torch.int64), torch.tensor([0, 1])
    bias = torch.zeros(8, dtype=torch.float64)
    dout = torch.zeros(n, h, v, dtype=torch.float64)
    dout[[2, 5]] = .125
    saved = [x, gamma, beta, mean, rstd, weight, offsets, targets, bias, sort]
    tensors = [*saved, dout]
    storages, descriptions = {}, []
    for index, tensor in enumerate(tensors):
        storages[index] = tensor.contiguous().reshape(-1).view(torch.uint8).clone()
        descriptions.append({"__tripwire_tensor__": index, "dtype": str(tensor.dtype),
                             "shape": list(tensor.shape), "stride": list(tensor.stride()),
                             "storage_offset": 0, "device": "cuda:0"})
    ctx = dict(attn_alpha=.125, has_multiple_targets=True, max_seq_len=3, max_attn_len=0,
               recompute_normed_x_in_backward=True, recompute_uvqk_in_backward=True,
               hidden_dim=v, attn_dim=a, num_heads=h, uvqk_bias_1d=True, norm_eps=eps,
               norm_BLOCK_D=d, contextual_seq_len=0, sort_by_length=True,
               enable_tma=False, num_softmax_heads=0)
    runtime = {"common": {"STATIC_MAX_SEQ_LENS": [4], "USE_RUNTIME_MAX_SEQ_LEN": False},
               "float32_matmul_precision": "highest"}
    replay = {"class": "_HSTUPreprocessAndAttentionFunction", "direction": "backward", "ctx": ctx,
              "saved_tensors": descriptions[:10], "args": (None, descriptions[10]), "kwargs": {},
              "runtime_state": runtime}
    prefix = {"format": analyzer.PREFIX_FORMAT, "format_version": analyzer.PREFIX_VERSION,
              "payload": {"replay": replay}, "storages": storages}
    before = {"q": torch.full((n, h, a), .25, dtype=torch.float64),
              "k": torch.full((n, h, a), .5, dtype=torch.float64),
              "v": torch.full((n, h, v), .75, dtype=torch.float64), "dout": dout.clone(),
              "seq_offsets": offsets.clone(), "num_targets": targets.clone(),
              "sort_by_length_indices": sort.clone()}
    dq = torch.zeros(n, h, a, dtype=torch.float64)
    dq[1, 0] = torch.tensor([.75, -.125])
    dq[3, 0, 1] = -.5
    dq[2, 0, 0] = 99.0  # Valid target DQ is deliberately large and must be excluded.
    failure = {"format": "hstu_attention_backward_failure", "format_version": 1,
               "iteration": 7, "failures": [{"iteration": 7, "check": "dq_history_zero"}],
               "configuration": {"context": copy.deepcopy(ctx), "captured_runtime_state": copy.deepcopy(runtime),
                                 "static_max_seq_lens": [4], "use_runtime_max_seq_len": False,
                                 "pre_hook": "_bwd_pre_hook", "input_artifact": "prefix.pt", "input_artifact_sha256": "fixture"},
               "pristine_inputs": before, "inputs_at_failure": {k: v.clone() for k, v in before.items()},
               "outputs": {"dq": dq, "dk": torch.zeros_like(dq), "dv": torch.zeros(n, h, v, dtype=torch.float64)}}
    return failure, prefix, saved


class AttentionFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rng = torch.get_rng_state().clone()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng, torch.get_rng_state())
        assert not torch.cuda.is_initialized()

    def test_propagation_matches_independent_autograd(self):
        failure, prefix, saved = fixture()
        report, tensors = analyzer.analyze(failure, prefix, prefix_sha256="fixture", row_chunk=1)
        self.assertEqual(report["history_dq"]["violating_elements"], 3)
        self.assertEqual(report["history_dq"]["violating_rows"], [1, 3])
        self.assertEqual(report["history_dq"]["max_abs_finite"], .75)
        self.assertTrue(report["propagation"]["all_violating_rows_included"])
        x, gamma, beta, mean, rstd, weight = saved[:6]
        xx, gg, bb, ww = [t.clone().requires_grad_() for t in (x, gamma, beta, weight)]
        normed = torch.nn.functional.layer_norm(xx, (4,), gg, bb, 1e-5)
        error = torch.zeros(6, 2, dtype=torch.float64)
        error[[1, 3]] = failure["outputs"]["dq"][[1, 3]].reshape(2, 2)
        result = torch.autograd.grad((normed @ ww)[:, 4:6], (xx, gg, bb, ww), error)
        torch.testing.assert_close(tensors["delta_d_input_x_fp64"], result[0][tensors["rows"]], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(tensors["delta_uvqk_weight_qblock_fp64"], result[3][:, 4:6], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(tensors["delta_uvqk_bias_qblock_fp64"], error.sum(0))
        incoming = (error @ weight[:, 4:6].T).bfloat16().double()
        stage_grads = torch.autograd.grad(torch.nn.functional.layer_norm(xx, (4,), gg, bb, 1e-5), (xx, gg, bb), incoming)
        torch.testing.assert_close(tensors["delta_d_input_x_from_bf16_projection_fp64"], stage_grads[0][tensors["rows"]], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(tensors["delta_norm_weight_from_bf16_projection_fp64"], stage_grads[1], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(tensors["delta_norm_bias_from_bf16_projection_fp64"], stage_grads[2], rtol=1e-12, atol=1e-12)

    def test_bf16_overflow_and_selected_rows_are_explicit(self):
        failure, prefix, saved = fixture()
        failure["outputs"]["dq"][1, 0] = torch.tensor([1e38, -1e38], dtype=torch.float64)
        # Raise only the downstream consumer weights, retaining bounded attention inputs.
        storage = prefix["storages"][5].view(torch.float64).reshape(4, 8)
        storage[:, 4] = 100.
        storage[:, 5] = -100.
        report, tensors = analyzer.analyze(failure, prefix, prefix_sha256="fixture", max_propagation_rows=1)
        self.assertEqual(report["history_dq"]["finite_elements_above_1e20"], 2)
        self.assertFalse(report["propagation"]["all_violating_rows_included"])
        self.assertGreater(report["propagation"]["stage_references"]["delta_d_normed_x_fp64"]["cast_nonfinite"], 0)
        self.assertTrue(torch.isinf(tensors["delta_d_normed_x_bf16"]).any())

    def test_history_gradient_or_input_mutation_invalidates_oracle(self):
        failure, prefix, saved = fixture()
        for where in ("pristine_inputs", "inputs_at_failure"):
            failure[where]["dout"][0, 0, 0] = 1.
        prefix["storages"][10].view(torch.float64)[0] = 1.
        with self.assertRaisesRegex(ValueError, "not exactly zero"):
            analyzer.analyze(failure, prefix, prefix_sha256="fixture")
        failure, prefix, _ = fixture()
        failure["inputs_at_failure"]["q"][0, 0, 0] += .125
        with self.assertRaisesRegex(ValueError, "changed logically or in raw storage"):
            analyzer.analyze(failure, prefix, prefix_sha256="fixture")

    def test_context_provenance_and_bounded_inputs_are_checked(self):
        failure, prefix, _ = fixture()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            analyzer.analyze(failure, prefix, prefix_sha256="wrong")
        for field, value in (("pre_hook", "other_hook"), ("static_max_seq_lens", [2])):
            changed = copy.deepcopy(failure)
            changed["configuration"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyzer.analyze(changed, prefix, prefix_sha256="fixture")
        for where in ("pristine_inputs", "inputs_at_failure"):
            failure[where]["q"][0, 0, 0] = 1e30
        with self.assertRaisesRegex(ValueError, "conservative input magnitude"):
            analyzer.analyze(failure, prefix, prefix_sha256="fixture")

    def test_equal_length_sort_ties_may_differ(self):
        failure, prefix, _ = fixture()
        for where in ("pristine_inputs", "inputs_at_failure"):
            failure[where]["sort_by_length_indices"] = torch.tensor([1, 0])
        report, _ = analyzer.analyze(failure, prefix, prefix_sha256="fixture")
        self.assertFalse(report["validation"]["sort_matches_prefix"])

    def test_no_violation_and_matrix_limit(self):
        failure, prefix, _ = fixture()
        with self.assertRaisesRegex(ValueError, "matrices exceed"):
            analyzer.analyze(failure, prefix, prefix_sha256="fixture", max_matrix_bytes=1)
        failure["outputs"]["dq"][[1, 3]] = 0
        report, tensors = analyzer.analyze(failure, prefix, prefix_sha256="fixture")
        self.assertEqual(report["propagation"]["status"], "no history-DQ violation")
        self.assertEqual(tensors["all_violating_rows"].numel(), 0)

    def test_cli_roundtrip_and_source_hash(self):
        failure, prefix, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pp, fp, rp = directory / "prefix.pt", directory / "failure.pt", directory / "analysis.json"
            torch.save(prefix, pp)
            failure["configuration"]["input_artifact_sha256"] = analyzer.file_sha256(pp)
            failure["configuration"]["input_artifact"] = str(pp)
            # Compiler metadata may contain opaque target descriptors.
            failure["compiled_kernels"] = [{"target": Path("gfx1250") }]
            torch.save(failure, fp)
            with mock.patch("sys.argv", ["analyze_attention_failure.py", str(fp), "--report", str(rp)]), mock.patch("builtins.print"):
                analyzer.main()
            report = json.loads(rp.read_text())
            self.assertFalse(report["cuda_initialized"])
            self.assertTrue(report["rng_unchanged"])
            self.assertEqual(report["history_dq"]["violating_elements"], 3)
            sidecar = torch.load(report["tensor_report"], weights_only=True)
            self.assertIn("delta_uvqk_weight_qblock_fp64", sidecar["tensors"])


if __name__ == "__main__":
    unittest.main()
