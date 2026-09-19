"""CPU checks of capture around the real output autograd function.

Extract the Python class from source and replace only its numerical helpers.
This exercises its real saved-tensor layout, GEMM order, and optional-mask path
without importing Triton, launching kernels, or initializing CUDA.
"""

import ast
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nan_backward_boundaries import (
    BackwardBoundaryProbe, BoundaryAnomalyError, BoundaryProbeError,
)


class MockTorch:
    def __init__(self):
        self.mm_calls = 0
        self.extreme_first_mm = False

    def __getattr__(self, name):
        return getattr(torch, name)

    def mm(self, left, right):
        self.mm_calls += 1
        result = torch.mm(left, right)
        if self.extreme_first_mm and self.mm_calls == 1:
            result[0, 1] = 1e25
        return result


def make_modules(*, random_mask=True):
    output = types.ModuleType("cpu_mock_hstu_linear")
    output.torch = MockTorch()
    output.Tuple = tuple
    output.Optional = __import__("typing").Optional
    output.norm_calls = []
    output.extreme_norm = False
    output.extreme_mean = False
    output.mutate_norm_inputs = False

    def features(x, u, concat_u, concat_x):
        return torch.cat(([u] if concat_u else []) + ([x] if concat_x else []) + [x * u], dim=1)

    def norm_forward(x, u, weight, bias, eps, dropout_ratio, training,
                     silu_u=False, concat_u=False, concat_x=False,
                     mul_u_activation_type="none", seed=None):
        mean = x.mean(dim=1).to(torch.float32)
        if output.extreme_mean:
            mean[1] = 1e25
        rstd = torch.ones(x.shape[0], dtype=torch.float32)
        mask = None
        if random_mask:
            # Both the nonzero offset and the interleaved unused storage must
            # survive the input snapshot, not merely its logical mask values.
            backing = torch.arange(x.numel() * 2 + 4, dtype=torch.int8) % 8
            mask = backing[2:2 + 2 * x.numel():2].view_as(x)
        return features(x, u, concat_u, concat_x), mean, rstd, 8, 4, seed, mask

    def norm_backward(dy, x, u, weight, bias, mean, rstd, BLOCK_D, num_warps,
                      eps, training, dropout_ratio, seed=None, silu_u=False,
                      concat_u=False, concat_x=False, mul_u_activation_type="none",
                      compute_y=False, random_mask=None):
        output.norm_calls.append({
            "dy": dy, "before_dy": dy.clone(), "random_mask": random_mask,
            "before_mask": None if random_mask is None else random_mask.clone(),
        })
        dx = dy[:, :x.shape[1]].clone()
        du = dx + 1
        dweight = dx.sum(dim=0).to(weight.dtype)
        dbias = du.sum(dim=0).to(bias.dtype)
        y = features(x, u, concat_u, concat_x) if compute_y else None
        if output.mutate_norm_inputs:
            dy.fill_(42)
            if random_mask is not None:
                random_mask.fill_(3)
        if output.extreme_norm:
            dx[2, 1] = 1e25
        return dx, du, dweight, dbias, y

    output.triton_layer_norm_mul_dropout_fwd = norm_forward
    output.triton_layer_norm_mul_dropout_bwd = norm_backward
    output.maybe_triton_addmm_fwd = lambda x, w, y: x @ w + y
    source = Path(__file__).resolve().parents[1] / (
        "generative_recommenders/ops/triton/triton_hstu_linear.py")
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in (
                 "HSTUComputeOutputFunction", "triton_hstu_compute_output")]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), output.__dict__)
    compute = types.SimpleNamespace(triton_hstu_compute_output=output.triton_hstu_compute_output)

    def preprocess_forward(x, norm_weight, norm_bias, uvqk_weight, uvqk_bias):
        return x

    compute.triton_hstu_preprocess_and_attention = preprocess_forward

    def addmm_backward(x, w, dz, is_y_1d):
        return dz @ w.t(), x.t() @ dz, dz.sum(dim=0)

    def layer_norm_backward(dy, x, weight, bias, mean, rstd, learnable, eps, BLOCK_D):
        return dy.clone(), dy.sum(dim=0), dy.sum(dim=0)

    preprocess = types.SimpleNamespace(triton_addmm_bwd=addmm_backward,
                                      triton_weighted_layer_norm_bwd=layer_norm_backward)
    return compute, preprocess, output


class Layer(torch.nn.Module):
    def __init__(self, compute, *, recompute=False):
        super().__init__()
        self.compute = compute
        self.recompute = recompute
        for name in ("_input_norm_weight", "_output_norm_weight"):
            setattr(self, name, torch.nn.Parameter(torch.ones(4)))
        for name in ("_input_norm_bias", "_output_norm_bias", "_uvqk_beta"):
            setattr(self, name, torch.nn.Parameter(torch.zeros(4)))
        self._uvqk_weight = torch.nn.Parameter(torch.eye(4))
        self._output_weight = torch.nn.Parameter(torch.full((12, 4), 0.125))

    def forward(self, x):
        attn = self.compute.triton_hstu_preprocess_and_attention(
            x=x, norm_weight=self._input_norm_weight.to(x.dtype),
            norm_bias=self._input_norm_bias.to(x.dtype),
            uvqk_weight=self._uvqk_weight.to(x.dtype), uvqk_bias=self._uvqk_beta.to(x.dtype))
        return self.compute.triton_hstu_compute_output(
            attn=attn, u=attn + 0.25, x=x,
            norm_weight=self._output_norm_weight.to(x.dtype),
            norm_bias=self._output_norm_bias.to(x.dtype),
            output_weight=self._output_weight.to(x.dtype), eps=1e-6,
            dropout_ratio=0.125, training=True, silu_u=True, concat_u=True,
            concat_x=True, mul_u_activation_type="silu", seed=23,
            recompute_y_in_backward=self.recompute)


class Model(torch.nn.Module):
    def __init__(self, compute, **kwargs):
        super().__init__()
        self._stu_layers = torch.nn.ModuleList([Layer(compute, **kwargs), Layer(compute, **kwargs)])

    def forward(self, x, layer=1):
        return self._stu_layers[layer](x)


class OutputBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "NAN_BACKWARD_TARGET": "_stu_layers.1.", "NAN_BACKWARD_SAVE_ALL": "1",
            "NAN_BACKWARD_ABS_THRESHOLD": "1e20", "NAN_BACKWARD_CHUNK_MIB": "1",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def setup_probe(self, **kwargs):
        compute, preprocess, output = make_modules(random_mask=kwargs.pop("random_mask", True))
        model = Model(compute, **kwargs)
        probe = BackwardBoundaryProbe(model, self.directory,
                                      compute_module=compute, preprocess_module=preprocess,
                                      output_module=output)
        self.addCleanup(probe.close)
        probe.set_attempt("current", 0, 0)
        return model, probe, output

    def execute(self, model, *, layer=1, extreme_dout=False):
        x = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 16
        x.requires_grad_(True)
        result = model(x, layer=layer)
        # The expanded incoming gradient is a zero-stride view with one storage
        # element; the dump must retain that layout for GEMM reproduction.
        dout = torch.tensor(1e25 if extreme_dout else 0.5, dtype=x.dtype).expand_as(result)
        result.backward(dout)
        return x

    def dumps(self):
        return [torch.load(path, map_location="cpu", weights_only=False)
                for path in sorted(self.directory.glob("boundary-*.pt"))]

    def events(self):
        return [json.loads(line) for line in (self.directory / "boundaries.jsonl").read_text().splitlines()]

    def test_finite_stages_preserve_actual_dy_attrs_and_rng(self):
        model, probe, output = self.setup_probe()
        random_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        self.execute(model)
        self.assertEqual(output.torch._original.mm_calls, 2)
        self.assertEqual(len(output.norm_calls), 1)
        mm, norm = self.dumps()
        self.assertEqual(mm["operation"], "hstu_output_grad_mm")
        self.assertEqual(norm["operation"], "triton_layer_norm_mul_dropout_bwd")
        self.assertEqual(mm["kwargs"]["dout"].stride(), (0, 0))
        self.assertEqual(mm["kwargs"]["output_weight"].dtype, torch.float64)
        self.assertEqual(mm["event"]["mapping"], "forward_argument")
        self.assertEqual(mm["execution_controls"]["autocast"]["cuda"]["enabled"], False)
        self.assertEqual(mm["execution_controls"]["float32_matmul_precision"],
                         torch.get_float32_matmul_precision())
        self.assertEqual(mm["execution_controls"]["autocast"]["cuda"]["dtype"],
                         str(torch.get_autocast_dtype("cuda")))
        observation = mm["event"]["input_observation"]
        self.assertEqual(observation["source"], "pristine_pre_call_cpu_snapshot")
        self.assertFalse(observation["operation_started"])
        self.assertLessEqual(observation["capture_started_unix_time"], observation["capture_completed_unix_time"])
        self.assertLessEqual(observation["capture_completed_unix_time"], observation["scan_completed_unix_time"])
        self.assertEqual(mm["event"]["parameter"], "model._stu_layers.1._output_weight")
        self.assertEqual(norm["event"]["parameter"], "model._stu_layers.1._output_norm_weight")
        self.assertTrue(torch.equal(mm["outputs"][0], norm["kwargs"]["dy"]))
        self.assertTrue(torch.equal(norm["kwargs"]["dy"], output.norm_calls[0]["before_dy"]))
        self.assertEqual(len(norm["outputs"]), 5)
        self.assertIsNone(norm["outputs"][4])
        for name, expected in {"seed": 23, "BLOCK_D": 8, "num_warps": 4,
                               "training": True, "dropout_ratio": 0.125, "eps": 1e-6,
                               "silu_u": True, "concat_u": True, "concat_x": True,
                               "mul_u_activation_type": "silu", "compute_y": False}.items():
            self.assertEqual(norm["kwargs"][name], expected)
        mask = norm["kwargs"]["random_mask"]
        original_mask = output.norm_calls[0]["random_mask"]
        self.assertEqual(mask.dtype, torch.int8)
        self.assertEqual(mask.stride(), original_mask.stride())
        self.assertEqual(mask.storage_offset(), 2)
        self.assertEqual(mask.untyped_storage().nbytes(), original_mask.untyped_storage().nbytes())
        self.assertTrue(torch.equal(mask, original_mask))
        self.assertEqual(random.getstate(), random_state)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_state[0])
        self.assertTrue(np.array_equal(current_numpy[1], numpy_state[1]))
        self.assertEqual(current_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(self.events()[0]["operations"], list(probe._OPERATIONS))
        self.assertIsNone(probe._local.output_backward)

    def test_recomputed_y_and_seed_only_dropout_are_saved(self):
        model, _, _ = self.setup_probe(recompute=True, random_mask=False)
        self.execute(model)
        _, norm = self.dumps()
        self.assertTrue(norm["kwargs"]["compute_y"])
        self.assertIsNone(norm["kwargs"]["random_mask"])
        self.assertEqual(norm["outputs"][4].shape, (3, 12))

    def test_extreme_dout_stops_before_gemm(self):
        model, probe, output = self.setup_probe()
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model, extreme_dout=True)
        (dump,) = self.dumps()
        self.assertEqual(dump["operation"], "hstu_output_grad_mm")
        self.assertEqual(dump["stage"], "input")
        self.assertEqual(dump["event"]["extreme_inputs"], ["dout"])
        self.assertIsNone(dump["outputs"])
        self.assertFalse(dump["event"]["operation_executed"])
        self.assertEqual(output.torch._original.mm_calls, 0)
        self.assertFalse(output.norm_calls)
        self.assertIsNone(probe._local.output_backward)
        with self.assertRaises(BoundaryProbeError):
            probe.set_attempt("current", 1, 0)

    def test_extreme_actual_gemm_output_stops_before_layer_norm(self):
        model, probe, output = self.setup_probe()
        output.torch._original.extreme_first_mm = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        (dump,) = self.dumps()
        self.assertEqual(dump["stage"], "output")
        self.assertEqual(dump["event"]["extreme_outputs"], ["dy"])
        self.assertEqual(dump["outputs"][0][0, 1].item(), 1e25)
        self.assertLess(dump["kwargs"]["dout"].max().item(), 1e20)
        self.assertEqual(output.torch._original.mm_calls, 1)
        self.assertFalse(output.norm_calls)
        self.assertIsNone(probe._local.output_backward)

    def test_norm_failure_retains_pristine_inputs_and_all_outputs(self):
        model, probe, output = self.setup_probe()
        output.extreme_norm = output.mutate_norm_inputs = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        mm, norm = self.dumps()
        self.assertEqual(norm["stage"], "output")
        self.assertEqual(norm["event"]["extreme_outputs"], ["d_attn"])
        self.assertEqual(len(norm["outputs"]), 5)
        self.assertTrue(torch.equal(mm["outputs"][0], norm["kwargs"]["dy"]))
        self.assertTrue(torch.equal(norm["kwargs"]["dy"], output.norm_calls[0]["before_dy"]))
        self.assertTrue(torch.equal(norm["kwargs"]["random_mask"], output.norm_calls[0]["before_mask"]))
        self.assertFalse(torch.equal(norm["kwargs"]["dy"], output.norm_calls[0]["dy"]))
        self.assertEqual(output.torch._original.mm_calls, 1)
        self.assertIsNone(probe._local.output_backward)

    def test_extreme_saved_mean_stops_before_norm(self):
        model, probe, output = self.setup_probe()
        output.extreme_mean = True
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        mm, norm = self.dumps()
        self.assertEqual(mm["stage"], "finite")
        self.assertEqual(norm["stage"], "input")
        self.assertEqual(norm["event"]["extreme_inputs"], ["mean"])
        self.assertIsNone(norm["outputs"])
        self.assertFalse(output.norm_calls)
        self.assertEqual(output.torch._original.mm_calls, 1)
        self.assertIsNone(probe._local.output_backward)

    def test_target_filter_and_no_save_all(self):
        model, _, output = self.setup_probe()
        self.execute(model, layer=0)
        self.assertFalse(self.dumps())
        skipped = [event for event in self.events() if event["event"] == "skipped"]
        self.assertEqual([event["operation"] for event in skipped],
                         ["hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd"])
        self.assertTrue(all(event["layer"] == "model._stu_layers.0" for event in skipped))
        self.assertEqual(output.torch._original.mm_calls, 2)

    def test_finite_calls_without_save_all_keep_summaries(self):
        with patch.dict(os.environ, {"NAN_BACKWARD_SAVE_ALL": "0"}):
            model, _, _ = self.setup_probe()
        self.execute(model)
        self.assertFalse(self.dumps())
        self.assertEqual(len([event for event in self.events() if event["event"] == "after"]), 2)

    def test_preprocess_stages_remain_captured(self):
        model, probe, _ = self.setup_probe()
        layer = model._stu_layers[1]
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 16
        dy = torch.ones_like(x)
        probe.preprocess.triton_addmm_bwd(x, layer._uvqk_weight, dy, True)
        probe.preprocess.triton_weighted_layer_norm_bwd(
            dy, x, layer._input_norm_weight, layer._input_norm_bias,
            x.mean(dim=1), torch.ones(3), True, 1e-6, 4)
        mm, norm = self.dumps()
        self.assertEqual(mm["operation"], "triton_addmm_bwd")
        self.assertEqual(norm["operation"], "triton_weighted_layer_norm_bwd")
        self.assertEqual(len(mm["outputs"]), 3)
        self.assertEqual(len(norm["outputs"]), 3)

    def test_proxy_is_local_and_staticmethod_restores_after_failure(self):
        compute, preprocess, output = make_modules()
        original_torch = output.torch
        original_mm = torch.mm
        original_backward = inspect.getattr_static(output.HSTUComputeOutputFunction, "backward")
        original_norm = output.triton_layer_norm_mul_dropout_bwd
        model = Model(compute)
        probe = BackwardBoundaryProbe(model, self.directory, compute_module=compute,
                                      preprocess_module=preprocess, output_module=output)
        self.addCleanup(probe.close)
        self.assertIs(torch.mm, original_mm)
        self.assertIsInstance(inspect.getattr_static(output.HSTUComputeOutputFunction, "backward"), staticmethod)
        # An unrelated module-local MM executes even before set_attempt.
        output.torch.mm(torch.ones(1, 1), torch.ones(1, 1))
        self.assertEqual(original_torch.mm_calls, 1)
        self.assertEqual(len(self.events()), 1)
        original_torch.mm_calls = 0
        original_torch.extreme_first_mm = True
        probe.set_attempt("current", 0, 0)
        with self.assertRaises(BoundaryAnomalyError):
            self.execute(model)
        probe.close()
        self.assertIs(output.torch, original_torch)
        self.assertIs(output.triton_layer_norm_mul_dropout_bwd, original_norm)
        self.assertIs(inspect.getattr_static(output.HSTUComputeOutputFunction, "backward"), original_backward)
        self.assertIs(torch.mm, original_mm)


if __name__ == "__main__":
    program = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU tests unexpectedly initialized CUDA"
    sys.exit(not program.result.wasSuccessful())
