"""Execute the original HSTU class body on CPU with only its GPU helpers replaced."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Optional, Tuple
import unittest
from unittest.mock import patch

import torch

from generative_recommenders.dlrm_v4.train import nan_tripwire as tripwire_module
from scripts.replay_nan_tripwire import (
    ReplayContext, _execute_replay, decode_payload, replay_once, stress_replay,
)
from scripts.repro_output_backward_stages import original_stage_inputs


LINEAR_MODULE = "generative_recommenders.ops.triton.triton_hstu_linear"
LINEAR_SOURCE = Path(__file__).resolve().parents[3] / "ops/triton/triton_hstu_linear.py"


class OriginalOutputStagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(LINEAR_SOURCE.read_text(), filename=str(LINEAR_SOURCE))
        original = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "HSTUComputeOutputFunction")
        cls.class_code = compile(ast.Module(body=[original], type_ignores=[]), str(LINEAR_SOURCE), "exec")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.operations = []
        self.poison_norm = False
        self.raise_norm = False
        self.norm_inputs = None

        def norm_forward(**kwargs):
            x, u = kwargs["x"], kwargs["u"]
            return (x + u, torch.zeros(x.shape[0]), torch.ones(x.shape[0]),
                    x.shape[1], 4, kwargs["seed"], torch.ones_like(x, dtype=torch.int8))

        def norm_backward(**kwargs):
            self.operations.append("norm")
            self.norm_inputs = kwargs
            if self.raise_norm:
                raise RuntimeError("original norm failure")
            dy = kwargs["dy"]
            dattn = dy * (float("nan") if self.poison_norm else 2)
            return (dattn, dy * 3, dy.sum(0), dy.sum(0),
                    kwargs["x"] + kwargs["u"] if kwargs["compute_y"] else None)

        self.linear = ModuleType(LINEAR_MODULE)
        self.linear.__dict__.update(
            torch=torch, Optional=Optional, Tuple=Tuple,
            triton_layer_norm_mul_dropout_fwd=norm_forward,
            triton_layer_norm_mul_dropout_bwd=norm_backward,
            maybe_triton_addmm_fwd=lambda x, w, y: x @ w + y,
        )
        exec(self.class_code, self.linear.__dict__)
        self.module_patch = patch.dict(sys.modules, {LINEAR_MODULE: self.linear})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.function = self.linear.HSTUComputeOutputFunction
        backing = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        self.attn, self.u = backing[:, 4:], backing[:, :4]
        self.weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)[:, 1::2]
        self.dout = torch.full((3, 4), 0.5)[:, 1::2]
        self.arguments = (self.attn, self.u, torch.ones(3, 2), torch.ones(4),
                          torch.zeros(4), self.weight, 1e-5, 0.0, True)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def wire(self, enabled=True, **kwargs):
        wire = tripwire_module.NaNTripwire(self.temp.name, output_backward_stages=enabled, **kwargs)
        wire.patch_function(self.function)
        self.addCleanup(wire.uninstall)
        wire.begin(7)
        return wire

    def forward(self, wire, *, recompute=True):
        ctx = ReplayContext(needs_input_grad=(False,) * 18)
        self.function.forward(ctx, *self.arguments, seed=3, recompute_y_in_backward=recompute)
        wire.check("forward")
        return ctx

    def capture(self, wire, ctx, exception):
        result = self.function.backward(ctx, self.dout)
        with self.assertRaises(exception) as stopped:
            wire.check("backward")
        report_path = (stopped.exception.report_path if isinstance(stopped.exception, tripwire_module.CaptureComplete)
                       else next(Path(self.temp.name).glob("*.json")))
        report = json.loads(report_path.read_text())
        capture = torch.load(report["capture_path"], weights_only=False)
        return capture, report, result

    def test_default_off_runs_original_calls_without_stage_observer(self):
        wire = self.wire(False)
        ctx = self.forward(wire)
        expected_dy = self.dout @ self.weight.t()
        with patch.object(tripwire_module, "_OutputBackwardStages", side_effect=AssertionError("disabled observer")):
            result = self.function.backward(ctx, self.dout)
        self.assertEqual(self.operations, ["norm"])
        self.assertEqual(len(wire.events), 1)
        self.assertNotIn("stage", wire.events[0])
        self.assertNotIn("_nan_tripwire_output_identity", ctx.__dict__)
        self.assertNotIn("_nan_tripwire_output_stages", ctx.__dict__)
        self.assertTrue(torch.equal(result[0], expected_dy * 2))
        self.assertIs(result[2], self.dout)

    def test_original_stage_order_has_no_clone_or_host_read(self):
        wire = self.wire()
        ctx = self.forward(wire)
        original_mm, original_observe = torch.mm, wire._observe
        mm_calls = 0

        def mm(*args):
            nonlocal mm_calls
            self.operations.append("gradient_mm" if mm_calls == 0 else "weight_mm")
            mm_calls += 1
            return original_mm(*args)

        def observe(values):
            if isinstance(values, dict):
                if values.keys() == {"dout", "output_weight"}:
                    self.operations.append("observe_mm_inputs")
                elif values.keys() == {"dy"}:
                    self.operations.append("observe_dy")
                elif "random_mask" in values:
                    self.operations.append("observe_norm_inputs")
                elif "dattn" in values:
                    self.operations.append("observe_norm_outputs")
            return original_observe(values)

        with patch.object(torch, "mm", side_effect=mm), patch.object(wire, "_observe", side_effect=observe), \
                patch.object(torch.Tensor, "clone", side_effect=AssertionError("clone")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")), \
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronize")):
            self.function.backward(ctx, self.dout)
        self.assertEqual(self.operations, ["observe_mm_inputs", "gradient_mm", "observe_dy",
                                          "observe_norm_inputs", "norm", "observe_norm_outputs", "weight_mm"])
        self.assertEqual([event.get("stage", {}).get("name") for event in wire.events],
                         ["gradient_mm", "norm", None])
        self.assertEqual(self.norm_inputs["dy"].data_ptr(),
                         wire.events[0]["payload"]["outputs"]["dy"].data_ptr())

    def test_gemm_failure_selected_before_downstream_norm_propagation(self):
        wire = self.wire()
        ctx = self.forward(wire)
        original_mm = torch.mm
        calls = 0

        def bad_first_mm(*args):
            nonlocal calls
            result = original_mm(*args)
            if calls == 0:
                result[0, 0] = float("nan")
            calls += 1
            return result

        with patch.object(torch, "mm", side_effect=bad_first_mm):
            capture, report, _ = self.capture(wire, ctx, tripwire_module.NonFiniteError)
        selected = next(event for event in report["events"] if event["sequence"] == report["selected_sequence"])
        self.assertEqual(selected["stage"]["name"], "gradient_mm")
        self.assertTrue(report["all_inputs_finite"])
        self.assertFalse(report["all_outputs_finite"])
        self.assertFalse(report["events"][1]["input_finite"][0])
        fixed, outputs = original_stage_inputs(capture, "mm", "cpu")
        self.assertEqual(fixed["weight"].stride(), self.weight.stride())
        self.assertEqual(fixed["dout"].storage_offset(), self.dout.storage_offset())
        self.assertTrue(torch.isnan(outputs["dy"][0, 0]))

    def test_norm_failure_has_finite_original_dy_and_replayable_parent(self):
        wire = self.wire()
        ctx = self.forward(wire)
        self.poison_norm = True
        capture, report, result = self.capture(wire, ctx, tripwire_module.NonFiniteError)
        selected = next(event for event in report["events"] if event["sequence"] == report["selected_sequence"])
        self.assertEqual(selected["stage"]["name"], "norm")
        self.assertTrue(report["all_inputs_finite"])
        self.assertTrue(report["events"][0]["output_finite"][0])
        payload = decode_payload(capture, "cpu")
        self.assertIsNone(payload["replay"])
        parent = payload["stage"]["parent_replay"]
        self.assertNotIn("_nan_tripwire_output_stages", parent["ctx"])
        self.assertNotIn("_nan_tripwire_output_stages", ctx.__dict__)
        self.assertEqual(len(payload["stage"]["parent_outputs"]), 18)
        self.assertTrue(torch.equal(payload["stage"]["parent_outputs"][5], result[5]))
        self.assertTrue(all(info["version_changed"] is False for info in selected["parent"]["inputs"]))
        wire.uninstall()
        replayed = _execute_replay(parent, "cpu")
        self.assertTrue(torch.isnan(replayed[0]).all())
        self.assertTrue(torch.equal(replayed[5], result[5]))
        for replay in (lambda: replay_once(capture, "cpu"),
                       lambda: stress_replay(capture, "cpu", repeat=1)):
            with self.assertRaisesRegex(ValueError, "no custom operation"):
                replay()

    def test_requested_norm_stage_preserves_original_aliases_layouts_and_mask(self):
        wire = self.wire(capture_step=7, capture_function_pattern=r"\.norm$")
        ctx = self.forward(wire, recompute=False)
        capture, report, _ = self.capture(wire, ctx, tripwire_module.CaptureComplete)
        inputs, outputs = original_stage_inputs(capture, "norm", "cpu")
        self.assertEqual(inputs["x"].stride(), self.attn.stride())
        self.assertEqual(inputs["x"].storage_offset(), self.attn.storage_offset())
        self.assertEqual(inputs["x"].dtype, self.attn.dtype)
        self.assertEqual(inputs["x"].untyped_storage().data_ptr(), inputs["u"].untyped_storage().data_ptr())
        self.assertEqual(inputs["random_mask"].dtype, torch.int8)
        self.assertEqual(inputs["compute_y"], False)
        self.assertIsNone(outputs["y"])
        self.assertTrue(torch.equal(inputs["dy"], self.dout @ self.weight.t()))
        self.assertEqual(capture["payload"]["stage"]["identity"]["forward_order"], 0)
        self.assertTrue(report["output_backward_stages"])

    def test_identity_tracks_forward_order_and_reused_weight(self):
        wire = self.wire(retain_payload=False)
        first, second = self.forward(wire), self.forward(wire)
        self.function.backward(second, self.dout)
        self.function.backward(first, self.dout)
        parents = [event for event in wire.events if "stage" not in event]
        self.assertEqual([event["invocation"]["forward_order"] for event in parents], [1, 0])
        self.assertEqual([event["invocation"]["backward_sequence"] for event in parents], [0, 1])
        self.assertEqual([event["invocation"]["observed_weight_ordinal"] for event in parents], [0, 0])
        self.assertTrue(all(event["payload"] is None for event in wire.events))

    def test_completed_mm_stage_can_be_captured_when_original_norm_raises(self):
        wire = self.wire(capture_step=7, capture_function_pattern=r"\.gradient_mm$")
        ctx = self.forward(wire)
        self.raise_norm = True
        with self.assertRaisesRegex(RuntimeError, "original norm failure") as original_error:
            self.function.backward(ctx, self.dout)
        self.assertNotIn("_nan_tripwire_output_stages", ctx.__dict__)
        with self.assertRaises(tripwire_module.CaptureComplete) as stopped:
            wire.check("backward")
        capture = torch.load(stopped.exception.capture_path, weights_only=False)
        report = json.loads(stopped.exception.report_path.read_text())
        selected = next(event for event in report["events"] if event["sequence"] == report["selected_sequence"])
        self.assertEqual(selected["stage"]["name"], "gradient_mm")
        self.assertFalse(selected["parent"]["completed"])
        self.assertEqual(selected["parent"]["outputs"], [])
        self.assertEqual(selected["parent"]["exception"], {
            "type": "builtins.RuntimeError", "message": str(original_error.exception),
        })
        decoded = decode_payload(capture, "cpu")
        self.assertFalse(decoded["stage"]["parent_completed"])
        self.assertIsNone(decoded["stage"]["parent_outputs"])
        self.assertEqual(decoded["stage"]["parent_exception"], selected["parent"]["exception"])
        self.assertNotIn("_nan_tripwire_output_stages", decoded["stage"]["parent_replay"]["ctx"])
        fixed, outputs = original_stage_inputs(capture, "mm", "cpu")
        self.assertTrue(torch.equal(outputs["dy"], fixed["dout"] @ fixed["weight"].t()))

    def test_completed_norm_stage_survives_weight_mm_failure(self):
        wire = self.wire(capture_step=7, capture_function_pattern=r"\.norm$")
        ctx = self.forward(wire)
        original_mm = torch.mm
        failure = RuntimeError("original weight mm failure")
        calls = 0

        def mm(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise failure
            return original_mm(*args)

        with patch.object(torch, "mm", side_effect=mm), self.assertRaises(RuntimeError) as stopped:
            self.function.backward(ctx, self.dout)
        self.assertIs(stopped.exception, failure)
        with self.assertRaises(tripwire_module.CaptureComplete) as stopped:
            wire.check("backward")
        capture = torch.load(stopped.exception.capture_path, weights_only=False)
        self.assertFalse(capture["payload"]["stage"]["parent_completed"])
        fixed, outputs = original_stage_inputs(capture, "norm", "cpu")
        self.assertTrue(torch.equal(outputs["dattn"], fixed["dy"] * 2))
        self.assertTrue(torch.equal(outputs["du"], fixed["dy"] * 3))

    def test_partial_provenance_failure_does_not_mask_original_exception(self):
        wire = self.wire()
        ctx = self.forward(wire)
        self.raise_norm = True
        with patch.object(tripwire_module._OutputBackwardStages, "finish", side_effect=ValueError("diagnostic failure")), \
                self.assertLogs(tripwire_module.logger, level="WARNING") as logs, \
                self.assertRaisesRegex(RuntimeError, "original norm failure"):
            self.function.backward(ctx, self.dout)
        self.assertIn("could not retain partial output backward provenance", logs.output[0])
        self.assertNotIn("_nan_tripwire_output_stages", ctx.__dict__)

    def test_stage_env_requires_explicit_one(self):
        for value, expected in ((None, False), ("true", False), ("0", False), ("1", True)):
            env = {"NAN_TRIPWIRE_DIR": self.temp.name}
            if value is not None:
                env["NAN_TRIPWIRE_OUTPUT_BACKWARD_STAGES"] = value
            with patch.dict(os.environ, env, clear=True), patch.object(tripwire_module.NaNTripwire, "install"):
                self.assertIs(tripwire_module.install_from_env().output_backward_stages, expected)


if __name__ == "__main__":
    unittest.main()
