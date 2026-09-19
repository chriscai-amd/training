"""CPU checks using the real preprocessing/output autograd Python classes."""

import ast
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from nan_backward_training_extended import ExtendedTrainingBackwardBoundaryProbe, EXTENDED_OPERATIONS
from nan_replay_state import capture_rng
from capture_training_backward import _rng_equal
from test_nan_backward_output import make_modules


def make_full_modules():
    compute, _, output = make_modules()
    state = types.SimpleNamespace(attention_calls=0, silu_calls=0, projection_calls=0,
                                  extreme_attention=False, extreme_silu=False,
                                  mutate_attention_inputs=False, corrupt_recomputed_q=False,
                                  attention_inputs=None)
    preprocess = types.ModuleType("cpu_full_preprocess")

    def silu_backward(grad_output, value, *, grad_input):
        state.silu_calls += 1
        result = torch.ops.aten.silu_backward(grad_output, value, grad_input=grad_input)
        if state.extreme_silu:
            grad_input[1, 2] = 1e25
        return result

    class LocalTorch:
        ops = types.SimpleNamespace(aten=types.SimpleNamespace(silu_backward=silu_backward))

        def __getattr__(self, name):
            return getattr(torch, name)

        def empty_like(self, value):
            # Packed write-only buffer begins poisoned. Before attention, all
            # four slices are unwritten. DU remains poisoned until SiLU writes.
            return torch.full_like(value, float("nan"))

    preprocess.torch = LocalTorch()
    preprocess.Tuple = tuple
    preprocess.Optional = __import__("typing").Optional
    preprocess.F = torch.nn.functional
    preprocess.compute_BLOCK_D = lambda x: 4

    def norm_forward(x, weight, bias, eps, mean=None, rstd=None):
        mean = torch.zeros(x.shape[0], dtype=torch.float32) if mean is None else mean.clone()
        rstd = torch.ones(x.shape[0], dtype=torch.float32) if rstd is None else rstd.clone()
        return x.clone(), mean, rstd

    def projection_forward(x, w, y):
        state.projection_calls += 1
        result = x @ w + y
        if state.corrupt_recomputed_q and state.projection_calls == 2:
            result[:, 8:12] = 1e25
        return result

    def attention_forward(N, alpha, q, k, v, seq_offsets, num_targets,
                          max_attn_len, contextual_seq_len, sort_by_length_indices,
                          enable_tma, num_softmax_heads):
        return q + v

    def attention_backward(dout, q, k, v, dq, dk, dv, seq_offsets, num_targets,
                           N, alpha, max_attn_len, contextual_seq_len,
                           sort_by_length_indices, enable_tma, num_softmax_heads):
        state.attention_calls += 1
        state.attention_inputs = {"q": q.clone(), "k": k.clone(), "v": v.clone(),
                                  "dout": dout.clone(), "dq": dq, "dk": dk, "dv": dv}
        shaped = dout.reshape_as(dq)
        dq.copy_(shaped)
        dk.copy_(shaped * 2)
        dv.copy_(shaped * 3)
        if state.mutate_attention_inputs:
            q.fill_(42.)
        if state.extreme_attention:
            dq[1, 0, 2] = 1e25

    def projection_backward(x, w, dz, is_y_1d):
        return dz @ w.t(), x.t() @ dz, dz.sum(dim=0)

    def norm_backward(dy, x, weight, bias, mean, rstd, learnable, eps, BLOCK_D):
        return dy.clone(), dy.sum(dim=0), dy.sum(dim=0)

    preprocess.triton_weighted_layer_norm_fwd = norm_forward
    preprocess.maybe_triton_addmm_fwd = projection_forward
    preprocess.triton_hstu_attention_fwd = attention_forward
    class OpaqueCustomOp:
        def __init__(self, function):
            self._init_fn = function

        def __call__(self, *args, **kwargs):
            return self._init_fn(*args, **kwargs)

    # Real torch.library.CustomOpDef exposes the same variadic public call.
    preprocess.triton_hstu_attention_bwd = OpaqueCustomOp(attention_backward)
    preprocess.triton_addmm_bwd = projection_backward
    preprocess.triton_weighted_layer_norm_bwd = norm_backward
    source = Path(__file__).resolve().parents[1] / "generative_recommenders/ops/triton/triton_hstu_preprocess_and_attention.py"
    node, = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.ClassDef) and node.name == "_HSTUPreprocessAndAttentionFunction"]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), preprocess.__dict__)

    def preprocess_forward(x, norm_weight, norm_bias, uvqk_weight, uvqk_bias):
        return preprocess._HSTUPreprocessAndAttentionFunction.apply(
            x, norm_weight, norm_bias, 1e-6, 1, 4, 4, uvqk_weight, uvqk_bias,
            3, torch.tensor([0, 3]), 0.5, None, 0, 0, True, True, False, False)

    compute.triton_hstu_preprocess_and_attention = preprocess_forward
    return compute, preprocess, output, state


class Layer(torch.nn.Module):
    def __init__(self, compute):
        super().__init__()
        self.compute = compute
        self._input_norm_weight = torch.nn.Parameter(torch.ones(4))
        self._input_norm_bias = torch.nn.Parameter(torch.zeros(4))
        self._uvqk_weight = torch.nn.Parameter(torch.eye(4).repeat(1, 4))
        self._uvqk_beta = torch.nn.Parameter(torch.zeros(16))
        self._output_norm_weight = torch.nn.Parameter(torch.ones(4))
        self._output_norm_bias = torch.nn.Parameter(torch.zeros(4))
        self._output_weight = torch.nn.Parameter(torch.full((12, 4), 0.125))

    def forward(self, x):
        u, attn = self.compute.triton_hstu_preprocess_and_attention(
            x=x, norm_weight=self._input_norm_weight.to(x.dtype),
            norm_bias=self._input_norm_bias.to(x.dtype), uvqk_weight=self._uvqk_weight.to(x.dtype),
            uvqk_bias=self._uvqk_beta.to(x.dtype))
        return self.compute.triton_hstu_compute_output(
            attn=attn.view_as(x), u=u, x=x, norm_weight=self._output_norm_weight.to(x.dtype),
            norm_bias=self._output_norm_bias.to(x.dtype), output_weight=self._output_weight.to(x.dtype),
            eps=1e-6, dropout_ratio=0.125, training=True, concat_u=True, concat_x=True,
            mul_u_activation_type="none", seed=23, recompute_y_in_backward=True)


class ExtendedBoundaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        environment = patch.dict(os.environ, {"NAN_BACKWARD_TARGET": "_stu_layers.1.",
            "NAN_BACKWARD_SAVE_ALL": "0", "NAN_BACKWARD_ABS_THRESHOLD": "1e20",
            "NAN_BACKWARD_CHUNK_MIB": "1", "NAN_BACKWARD_MAX_CAPTURE_GIB": "32"})
        environment.start()
        self.addCleanup(environment.stop)

    def setup_probe(self, mode="monitor_on_anomaly", operations=EXTENDED_OPERATIONS):
        compute, preprocess, output, state = make_full_modules()
        model = torch.nn.Module()
        model._stu_layers = torch.nn.ModuleList([Layer(compute), Layer(compute)])
        original_torch = preprocess.torch
        original_backward = preprocess._HSTUPreprocessAndAttentionFunction.backward
        probe = ExtendedTrainingBackwardBoundaryProbe(model, self.directory, mode=mode,
            selected_operations=operations, compute_module=compute, preprocess_module=preprocess,
            output_module=output)
        self.addCleanup(probe.close)
        probe.set_attempt("training", 0, 12)
        return model, probe, state, original_torch, original_backward

    def execute(self, model, layer=1):
        x = (torch.arange(12, dtype=torch.float64).view(3, 4) / 16).requires_grad_()
        model._stu_layers[layer](x).backward(torch.full_like(x, 0.5))
        return x

    def dumps(self):
        return [torch.load(path, map_location="cpu", weights_only=False)
                for path in self.directory.glob("*.pt")]

    def events(self):
        return [json.loads(line) for line in (self.directory / "boundaries.jsonl").read_text().splitlines()]

    def test_real_backward_order_live_attribution_unwritten_destinations_and_rng(self):
        model, probe, state, original_torch, original_backward = self.setup_probe()
        initial_rng = capture_rng()
        global_silu = torch.ops.aten.silu_backward
        with patch("nan_backward_training_extended._cpu_copy_tree", side_effect=AssertionError("healthy copy")):
            self.execute(model)
        self.assertEqual((state.attention_calls, state.silu_calls, state.projection_calls), (1, 1, 2))
        after = [event for event in self.events() if event["event"] == "after"]
        self.assertEqual([event["operation"] for event in after], list(EXTENDED_OPERATIONS))
        self.assertTrue(all(event["layer"] == "model._stu_layers.1" for event in after))
        self.assertTrue(all(event["mapping"] == "forward_argument" for event in after))
        self.assertEqual(after[0]["argument_roles"]["write_only"], ["dq", "dk", "dv"])
        budget = after[0]["capture_budget"]
        self.assertEqual(budget["argument_unique_storage_bytes"], budget["maximum_snapshot_storage_bytes"])
        self.assertNotIn("dq", {item["name"] for item in after[0]["inputs"]})
        self.assertTrue(_rng_equal(initial_rng, capture_rng()))
        self.assertFalse(self.dumps())
        self.assertIs(torch.ops.aten.silu_backward, global_silu)
        probe.close()
        self.assertIs(probe.preprocess.torch, original_torch)
        self.assertIs(probe.preprocess._HSTUPreprocessAndAttentionFunction.backward, original_backward)
        self.assertIsNone(probe._local.preprocess_backward)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bad_recomputed_q_stops_before_attention_and_silu(self):
        model, probe, state, *_ = self.setup_probe()
        state.corrupt_recomputed_q = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertEqual(dump["operation"], "hstu_attention_bwd")
        self.assertEqual(dump["stage"], "input")
        self.assertEqual(dump["event"]["extreme_inputs"], ["q"])
        self.assertTrue(dump["input_snapshot_pristine"])
        self.assertTrue(dump["initial_destination_bytes_retained"])
        self.assertTrue(torch.isnan(dump["kwargs"]["dq"]).all())
        self.assertIsNone(dump["outputs"])
        self.assertEqual((state.attention_calls, state.silu_calls), (0, 0))
        self.assertIsNone(probe._local.preprocess_backward)

    def test_monitor_attention_fault_retains_written_views_and_postcall_input_provenance(self):
        model, probe, state, *_ = self.setup_probe()
        state.extreme_attention = state.mutate_attention_inputs = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertEqual(dump["operation"], "hstu_attention_bwd")
        self.assertFalse(dump["input_snapshot_pristine"])
        self.assertFalse(dump["initial_destination_bytes_retained"])
        self.assertTrue(torch.equal(dump["kwargs"]["q"], torch.full_like(dump["kwargs"]["q"], 42.)))
        before = {entry["name"]: entry for entry in dump["event"]["input_device_monitor"]}
        self.assertLess(before["q"]["max_abs_finite"], 42.)
        self.assertEqual(dump["event"]["extreme_outputs"], ["dq"])
        dq, dk, dv = dump["outputs"]
        self.assertEqual(dq.untyped_storage()._cdata, dk.untyped_storage()._cdata)
        self.assertEqual(dq.untyped_storage()._cdata, dv.untyped_storage()._cdata)
        self.assertEqual(dq.untyped_storage()._cdata, dump["kwargs"]["dq"].untyped_storage()._cdata)
        self.assertEqual(dq.stride(), state.attention_inputs["dq"].stride())
        self.assertEqual((state.attention_calls, state.silu_calls), (1, 0))
        self.assertTrue(probe.failed)

    def test_pristine_attention_captures_opaque_initial_destinations_and_inputs(self):
        model, _, state, *_ = self.setup_probe(mode="pristine")
        state.extreme_attention = state.mutate_attention_inputs = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertTrue(dump["input_snapshot_pristine"])
        self.assertTrue(torch.equal(dump["kwargs"]["q"], state.attention_inputs["q"]))
        self.assertTrue(torch.isnan(dump["kwargs"]["dq"]).all())
        self.assertEqual(dump["outputs"][0][1, 0, 2].item(), 1e25)
        self.assertNotEqual(dump["kwargs"]["dq"].untyped_storage()._cdata,
                            dump["outputs"][0].untyped_storage()._cdata)
        self.assertFalse(dump["event"]["destination_initial_values_scanned"])
        budget = dump["event"]["capture_budget"]
        self.assertEqual(budget["argument_unique_storage_bytes"] + budget["destination_unique_storage_bytes"],
                         budget["maximum_snapshot_storage_bytes"])

    def test_silu_fault_is_separate_from_attention_and_preprojection(self):
        model, _, state, *_ = self.setup_probe()
        state.extreme_silu = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertEqual(dump["operation"], "hstu_silu_bwd")
        self.assertEqual(dump["event"]["extreme_outputs"], ["du"])
        self.assertEqual(dump["event"]["argument_roles"]["write_only"], ["grad_input"])
        self.assertEqual(dump["outputs"][0][1, 2].item(), 1e25)
        self.assertEqual((state.attention_calls, state.silu_calls), (1, 1))
        self.assertFalse(any(event["event"] == "after" and event["operation"] == "triton_addmm_bwd"
                             for event in self.events()))

    def test_pristine_silu_preserves_initial_unwritten_du_and_written_neighbors(self):
        model, _, state, *_ = self.setup_probe(mode="pristine", operations=("hstu_silu_bwd",))
        state.extreme_silu = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        dump, = self.dumps()
        self.assertTrue(dump["initial_destination_bytes_retained"])
        self.assertTrue(torch.isnan(dump["kwargs"]["grad_input"]).all())
        self.assertFalse(torch.isnan(dump["outputs"][0]).any())
        self.assertNotEqual(dump["kwargs"]["grad_input"].untyped_storage()._cdata,
                            dump["outputs"][0].untyped_storage()._cdata)
        self.assertEqual(dump["event"]["bad_inputs"], [])

    def test_all_six_operations_observe_actual_backward_order(self):
        operations = ("hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd",
                      "hstu_attention_bwd", "hstu_silu_bwd", "triton_addmm_bwd",
                      "triton_weighted_layer_norm_bwd")
        model, _, _, *_ = self.setup_probe(operations=operations)
        self.execute(model)
        after = [event for event in self.events() if event["event"] == "after"]
        self.assertEqual([event["operation"] for event in after], list(operations))
        self.assertEqual([event["call"] for event in after], list(range(1, 7)))

    def test_layer_and_operation_filters_preserve_unselected_execution(self):
        model, _, state, *_ = self.setup_probe(operations=("hstu_silu_bwd",))
        state.extreme_attention = True
        with patch("nan_backward_training_extended.monitor_summaries", side_effect=AssertionError("unselected scan")):
            self.execute(model, layer=0)
        self.assertFalse(self.dumps())
        self.assertEqual((state.attention_calls, state.silu_calls), (1, 1))

    def test_four_existing_operations_do_not_install_extended_wrappers(self):
        model, probe, state, original_torch, original_backward = self.setup_probe(
            operations=("hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd", "triton_addmm_bwd", "triton_weighted_layer_norm_bwd"))
        self.assertIs(probe.preprocess.torch, original_torch)
        self.assertIs(probe.preprocess._HSTUPreprocessAndAttentionFunction.backward, original_backward)
        self.execute(model)
        after = [event for event in self.events() if event["event"] == "after"]
        self.assertEqual(len(after), 4)
        self.assertEqual((state.attention_calls, state.silu_calls), (1, 1))

    def test_capture_limit_failure_marks_probe_failed(self):
        model, probe, state, *_ = self.setup_probe()
        state.extreme_attention = True
        probe.max_bytes = 1
        with self.assertRaisesRegex(BoundaryProbeError, "exceeds remaining capture limit"):
            self.execute(model)
        self.assertTrue(probe.failed)
        self.assertFalse(self.dumps())
        self.assertIsNone(probe._local.preprocess_backward)
        self.assertEqual(state.attention_calls, 0)
        self.assertTrue(any(event["event"] == "capture_budget_exceeded" for event in self.events()))


if __name__ == "__main__":
    unittest.main()
