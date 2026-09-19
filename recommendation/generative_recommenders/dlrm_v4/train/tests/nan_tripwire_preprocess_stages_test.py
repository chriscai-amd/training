"""CPU execution of the original preprocess class with its GPU helpers replaced."""

import ast
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
from typing import Optional, Tuple
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from generative_recommenders.dlrm_v4.train import nan_tripwire as tripwire_module
from scripts.replay_nan_tripwire import ReplayContext, decode_payload


PREPROCESS_MODULE = "generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention"
PREPROCESS_SOURCE = Path(__file__).resolve().parents[3] / "ops/triton/triton_hstu_preprocess_and_attention.py"


class FakeJIT:
    def __init__(self, name):
        self.name = name

    def run(self, *args, **kwargs):
        return SimpleNamespace(
            name=self.name, hash=f"{self.name}-bn{kwargs['BLOCK_N']}",
            metadata={"num_warps": kwargs["num_warps"], "num_stages": 1, "num_ctas": 1},
        )


class LaunchCaptureConcurrencyTest(unittest.TestCase):
    def test_overlapping_scopes_serialize_and_restore_after_worker_error(self):
        kernel = FakeJIT("shared_dx")
        original = kernel.run
        first = tripwire_module._OutputBackwardStages(None, "first", {})
        second = tripwire_module._OutputBackwardStages(None, "second", {})
        attempting, entered, finish = (threading.Event() for _ in range(3))
        results, errors = {}, []
        failure = RuntimeError("second original helper failed")

        def second_invocation():
            try:
                attempting.set()
                with second.capture_launches({"dx": kernel}, scope="second") as metadata:
                    results["metadata"] = metadata
                    entered.set()
                    if not finish.wait(5):
                        raise TimeoutError("test did not release second invocation")
                    kernel.run(BLOCK_N=8, BLOCK_D=512, num_warps=1)
                    raise failure
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=second_invocation, daemon=True)
        try:
            with first.capture_launches({"dx": kernel}, scope="first") as first_metadata:
                kernel.run(BLOCK_N=1, BLOCK_D=512, num_warps=1)
                worker.start()
                self.assertTrue(attempting.wait(5), "worker did not attempt its scope")
                self.assertFalse(entered.wait(0.1), "overlapping scope replaced live wrappers")
            self.assertTrue(entered.wait(5), "second scope did not acquire the released lock")
        finally:
            finish.set()
            worker.join(5)
        self.assertFalse(worker.is_alive(), "capture lock remained held")
        self.assertEqual(errors, [failure])
        self.assertEqual(kernel.run, original)
        self.assertEqual(first_metadata["dx"]["launch_count"], 1)
        self.assertEqual(first_metadata["dx"]["compiled_hash"], "shared_dx-bn1")
        self.assertEqual(results["metadata"]["dx"]["launch_count"], 1)
        self.assertEqual(results["metadata"]["dx"]["compiled_hash"], "shared_dx-bn8")
        # A later invocation must also acquire the lock after exception cleanup.
        with first.capture_launches({"dx": kernel}, scope="after_error") as later:
            kernel.run(BLOCK_N=4, BLOCK_D=512, num_warps=1)
        self.assertEqual(later["dx"]["compiled_hash"], "shared_dx-bn4")
        self.assertEqual(kernel.run, original)

    def test_same_thread_nested_scopes_restore_outer_wrapper(self):
        kernel = FakeJIT("shared_dx")
        original = kernel.run
        first = tripwire_module._OutputBackwardStages(None, "first", {})
        second = tripwire_module._OutputBackwardStages(None, "second", {})
        with first.capture_launches({"dx": kernel}, scope="outer") as outer:
            outer_run = kernel.run
            with second.capture_launches({"dx": kernel}, scope="inner") as inner:
                kernel.run(BLOCK_N=8, BLOCK_D=512, num_warps=1)
            self.assertIs(kernel.run, outer_run)
            kernel.run(BLOCK_N=1, BLOCK_D=512, num_warps=1)
        self.assertEqual(kernel.run, original)
        self.assertEqual(inner["dx"]["launch_count"], 1)
        self.assertEqual(outer["dx"]["launch_count"], 2)
        self.assertEqual(outer["dx"]["compiled_hash"], "shared_dx-bn1")


class OriginalPreprocessStagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(PREPROCESS_SOURCE.read_text(), filename=str(PREPROCESS_SOURCE))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "_HSTUPreprocessAndAttentionFunction")
        cls.class_code = compile(ast.Module(body=[original], type_ignores=[]), str(PREPROCESS_SOURCE), "exec")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.operations = []
        self.projection_inputs = None
        self.norm_inputs = None
        self.poison_projection = False
        self.poison_norm = False
        self.norm_error = None
        self.dx_kernel = SimpleNamespace(fn=FakeJIT("weighted_dx"), best_config="stale_bn99")
        self.reduction_kernel = SimpleNamespace(fn=FakeJIT("weighted_dwdb"))

        def norm_forward(x, weight, bias, eps, mean=None, rstd=None):
            n = x.shape[0]
            return x * weight + bias, torch.zeros(n), torch.ones(n)

        def attention_backward(**kwargs):
            self.operations.append("attention")
            for name, factor in (("dq", 2), ("dk", 3), ("dv", 4)):
                kwargs[name].copy_(kwargs["dout"] * factor)

        def projection_backward(**kwargs):
            self.operations.append("projection")
            self.projection_inputs = kwargs
            x, w, dz = (kwargs[name] for name in ("x", "w", "dz"))
            dx = dz @ w.t()
            if self.poison_projection:
                dx[0, 0] = float("nan")
            return dx, x.t() @ dz, dz.sum(0) if kwargs["is_y_1d"] else dz

        def norm_backward(**kwargs):
            self.operations.append("norm")
            self.norm_inputs = kwargs
            # Simulate tuning followed by the actual selected launch. A stale
            # best_config must not be used in place of these observed launches.
            self.dx_kernel.fn.run(BLOCK_N=1, BLOCK_D=kwargs["BLOCK_D"], num_warps=1)
            self.dx_kernel.fn.run(
                BLOCK_N=8, BLOCK_D=kwargs["BLOCK_D"], num_warps=1,
                llvm_fn_attrs="amdgpu-num-vgpr=192", maxnreg=192,
            )
            if self.norm_error is not None:
                raise self.norm_error
            self.reduction_kernel.fn.run(BLOCK_N=32, BLOCK_D=4, num_warps=8)
            dy = kwargs["dy"]
            return dy * (float("nan") if self.poison_norm else 2), dy.sum(0), dy.sum(0)

        self.preprocess = ModuleType(PREPROCESS_MODULE)
        self.preprocess.__dict__.update(
            torch=torch, Optional=Optional, Tuple=Tuple, F=F, nullcontext=nullcontext,
            compute_BLOCK_D=lambda x: x.shape[1],
            maybe_triton_addmm_fwd=lambda x, w, y: x @ w + y,
            triton_weighted_layer_norm_fwd=norm_forward,
            triton_hstu_attention_fwd=lambda **kwargs: kwargs["q"] + kwargs["k"] + kwargs["v"],
            triton_hstu_attention_bwd=attention_backward,
            triton_addmm_bwd=projection_backward,
            triton_weighted_layer_norm_bwd=norm_backward,
            _weighted_layer_norm_bwd_dx=self.dx_kernel,
            _layer_norm_bwd_dwdb=self.reduction_kernel,
        )
        exec(self.class_code, self.preprocess.__dict__)
        self.module_patch = patch.dict(sys.modules, {PREPROCESS_MODULE: self.preprocess})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.function = self.preprocess._HSTUPreprocessAndAttentionFunction
        self.x = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2] / 50
        self.weight = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 100
        self.dsilu_u = torch.ones(3, 4) * 0.2
        self.dout = torch.ones(3, 1, 4) * 0.3
        self.arguments = (
            self.x, torch.ones(4), torch.zeros(4), 1e-6, 1, 4, 4,
            self.weight, torch.zeros(16), 3, torch.tensor([0, 3]), 0.5,
            None, 0, 0, True, True, False, False,
        )

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def wire(self, enabled=True, **kwargs):
        wire = tripwire_module.NaNTripwire(
            self.temp.name, preprocess_backward_stages=enabled, **kwargs,
        )
        wire.patch_function(self.function)
        self.addCleanup(wire.uninstall)
        wire.begin(7)
        return wire

    def forward(self, wire):
        ctx = ReplayContext(needs_input_grad=(False,) * 19)
        self.function.forward(ctx, *self.arguments)
        wire.check("forward")
        return ctx

    def dump(self, wire, exception):
        with self.assertRaises(exception) as stopped:
            wire.check("backward")
        path = (stopped.exception.report_path if isinstance(stopped.exception, tripwire_module.CaptureComplete)
                else next(Path(self.temp.name).glob("*.json")))
        report = json.loads(path.read_text())
        capture = torch.load(report["capture_path"], weights_only=False)
        selected = next(e for e in report["events"] if e["sequence"] == report["selected_sequence"])
        return capture, selected

    def test_default_off_keeps_original_helper_calls_and_no_observer(self):
        wire = self.wire(False)
        ctx = self.forward(wire)
        with patch.object(tripwire_module, "_OutputBackwardStages", side_effect=AssertionError("disabled observer")):
            outputs = self.function.backward(ctx, self.dsilu_u, self.dout)
        self.assertEqual(self.operations, ["attention", "projection", "norm"])
        self.assertEqual(len(wire.events), 1)
        self.assertNotIn("stage", wire.events[0])
        self.assertNotIn("_nan_tripwire_preprocess_identity", ctx.__dict__)
        self.assertNotIn("_nan_tripwire_preprocess_stages", ctx.__dict__)
        self.assertTrue(torch.equal(outputs[0], self.norm_inputs["dy"] * 2))

    def test_original_stage_order_reference_identity_and_no_host_read(self):
        wire = self.wire()
        ctx = self.forward(wire)
        original_observe = wire._observe

        def observe(values):
            if isinstance(values, dict):
                if values.keys() == {"x", "w", "dz"}:
                    self.operations.append("observe_projection_inputs")
                elif values.keys() == {"d_normed_x", "d_uvqk_weight", "d_uvqk_bias"}:
                    self.operations.append("observe_projection_outputs")
                elif values.keys() == {"dy", "x", "weight", "bias", "mean", "rstd"}:
                    self.operations.append("observe_norm_inputs")
                elif values.keys() == {"d_x", "d_norm_weight", "d_norm_bias"}:
                    self.operations.append("observe_norm_outputs")
            return original_observe(values)

        with patch.object(wire, "_observe", side_effect=observe), \
                patch.object(torch.Tensor, "clone", side_effect=AssertionError("clone")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")), \
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronize")):
            self.function.backward(ctx, self.dsilu_u, self.dout)
        self.assertEqual(self.operations, [
            "attention", "observe_projection_inputs", "projection", "observe_projection_outputs",
            "observe_norm_inputs", "norm", "observe_norm_outputs",
        ])
        projection, norm, parent = wire.events
        self.assertEqual([event.get("stage", {}).get("name") for event in wire.events],
                         ["input_projection", "input_norm", None])
        self.assertEqual(projection["payload"]["outputs"]["d_normed_x"].data_ptr(), self.norm_inputs["dy"].data_ptr())
        self.assertEqual(norm["payload"]["inputs"]["dy"].data_ptr(), self.norm_inputs["dy"].data_ptr())
        uvqk = (self.x @ self.weight)
        expected_du = torch.ops.aten.silu_backward(self.dsilu_u, uvqk[:, :4])
        self.assertTrue(torch.equal(self.projection_inputs["dz"][:, :4], expected_du))
        self.assertEqual(norm["parent"]["sequence"], parent["sequence"])
        self.assertNotIn("_nan_tripwire_preprocess_stages", ctx.__dict__)

    def test_norm_capture_has_original_dy_and_actual_launch_config(self):
        wire = self.wire(capture_step=7, capture_function_pattern=r"\.input_norm$")
        ctx = self.forward(wire)
        original_dx, original_reduction = self.dx_kernel.fn.run, self.reduction_kernel.fn.run
        self.function.backward(ctx, self.dsilu_u, self.dout)
        capture, selected = self.dump(wire, tripwire_module.CaptureComplete)
        payload = decode_payload(capture, "cpu")
        self.assertTrue(torch.equal(payload["inputs"]["dy"], self.norm_inputs["dy"]))
        self.assertTrue(torch.equal(payload["outputs"]["d_x"], self.norm_inputs["dy"] * 2))
        stage = payload["stage"]
        self.assertEqual(stage["settings"], {"learnable": True, "eps": 1e-6, "BLOCK_D": 4})
        metadata = stage["launch_metadata"]
        self.assertEqual(metadata, selected["stage"]["launch_metadata"])
        self.assertEqual(metadata["dx"]["launch_count"], 2)
        self.assertEqual(metadata["dx"]["compiled_hash"], "weighted_dx-bn8")
        self.assertEqual(metadata["dx"]["launch_config"], {
            "BLOCK_N": 8, "BLOCK_D": 4, "num_warps": 1, "num_stages": 1,
            "num_ctas": 1, "maxnreg": 192, "llvm_fn_attrs": "amdgpu-num-vgpr=192",
        })
        self.assertEqual(metadata["reduction"]["launch_config"]["BLOCK_N"], 32)
        self.assertNotIn("_nan_tripwire_preprocess_stages", stage["parent_replay"]["ctx"])
        self.assertEqual(stage["identity"]["forward_order"], 0)
        self.assertEqual(self.dx_kernel.fn.run, original_dx)
        self.assertEqual(self.reduction_kernel.fn.run, original_reduction)

    def test_projection_failure_is_selected_before_norm_propagation(self):
        wire = self.wire()
        ctx = self.forward(wire)
        self.poison_projection = True
        self.function.backward(ctx, self.dsilu_u, self.dout)
        capture, selected = self.dump(wire, tripwire_module.NonFiniteError)
        self.assertEqual(selected["stage"]["name"], "input_projection")
        self.assertTrue(all(selected["input_finite"]))
        self.assertFalse(selected["output_finite"][0])
        self.assertTrue(torch.isnan(decode_payload(capture, "cpu")["outputs"]["d_normed_x"][0, 0]))

    def test_norm_failure_has_bounded_original_projection_output(self):
        wire = self.wire(max_abs=1e20)
        ctx = self.forward(wire)
        self.poison_norm = True
        self.function.backward(ctx, self.dsilu_u, self.dout)
        capture, selected = self.dump(wire, tripwire_module.NonFiniteError)
        self.assertEqual(selected["stage"]["name"], "input_norm")
        self.assertTrue(all(selected["input_finite"]))
        self.assertTrue(all(selected["input_within_bound"]))
        self.assertTrue(torch.isfinite(decode_payload(capture, "cpu")["inputs"]["dy"]).all())

    def test_later_exception_preserves_projection_and_restores_launch_methods(self):
        wire = self.wire(capture_step=7, capture_function_pattern=r"\.input_projection$")
        ctx = self.forward(wire)
        original_dx, original_reduction = self.dx_kernel.fn.run, self.reduction_kernel.fn.run
        self.norm_error = RuntimeError("original input norm failure")
        with self.assertRaises(RuntimeError) as stopped:
            self.function.backward(ctx, self.dsilu_u, self.dout)
        self.assertIs(stopped.exception, self.norm_error)
        self.assertEqual(self.dx_kernel.fn.run, original_dx)
        self.assertEqual(self.reduction_kernel.fn.run, original_reduction)
        self.assertNotIn("_nan_tripwire_preprocess_stages", ctx.__dict__)
        capture, selected = self.dump(wire, tripwire_module.CaptureComplete)
        self.assertFalse(selected["parent"]["completed"])
        payload = decode_payload(capture, "cpu")
        self.assertIsNone(payload["stage"]["parent_outputs"])
        self.assertEqual(payload["stage"]["parent_exception"]["message"], str(self.norm_error))
        self.assertTrue(torch.equal(payload["outputs"]["d_normed_x"],
                                    payload["inputs"]["dz"] @ payload["inputs"]["w"].t()))

    def test_identity_counters_are_separate_from_output_stage_counters(self):
        wire = self.wire(retain_payload=False, output_backward_stages=True)

        def forward(ctx, output_weight):
            return output_weight

        output = type("HSTUComputeOutputFunction", (torch.autograd.Function,), {
            "__module__": "generative_recommenders.ops.triton.triton_hstu_linear",
            "forward": staticmethod(forward), "backward": staticmethod(lambda ctx, dout: dout),
        })
        wire.patch_function(output)
        output.forward(ReplayContext(needs_input_grad=(False,)), self.weight)
        first, second = self.forward(wire), self.forward(wire)
        self.function.backward(second, self.dsilu_u, self.dout)
        self.function.backward(first, self.dsilu_u, self.dout)
        parents = [event for event in wire.events if "stage" not in event]
        self.assertEqual([event["invocation"]["forward_order"] for event in parents], [1, 0])
        self.assertEqual([event["invocation"]["backward_sequence"] for event in parents], [0, 1])
        self.assertEqual([event["invocation"]["observed_weight_ordinal"] for event in parents], [0, 0])
        self.assertEqual(wire._output_forward_sequence, 1)
        self.assertEqual(wire._output_backward_sequence, 0)
        self.assertEqual(wire._preprocess_forward_sequence, 2)

    def test_stage_env_requires_explicit_one(self):
        for value, expected in ((None, False), ("true", False), ("0", False), ("1", True)):
            env = {"NAN_TRIPWIRE_DIR": self.temp.name}
            if value is not None:
                env["NAN_TRIPWIRE_PREPROCESS_BACKWARD_STAGES"] = value
            with patch.dict(os.environ, env, clear=True), patch.object(tripwire_module.NaNTripwire, "install"):
                self.assertIs(tripwire_module.install_from_env().preprocess_backward_stages, expected)


if __name__ == "__main__":
    unittest.main()
