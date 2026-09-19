"""CPU tests of final-layer attribution, pristine B0 payloads and clean teardown."""
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import capture_replay_attention as adapter
import nan_backward_boundaries
from replay_nan_tripwire import changed_capture_inputs, decode_payload


class _HSTUPreprocessAndAttentionFunction(torch.autograd.Function):
    mutate = False
    huge = False

    @staticmethod
    def forward(ctx, x, norm_weight):
        ctx.save_for_backward(x, norm_weight)
        return x * norm_weight

    @staticmethod
    def backward(ctx, dsilu_u, dout):
        x, weight = ctx.saved_tensors
        if _HSTUPreprocessAndAttentionFunction.mutate:
            # Simulate a tracked source write after the pristine CPU snapshot.
            x.add_(5)
        result = x * weight
        if _HSTUPreprocessAndAttentionFunction.huge:
            result.fill_(1e30)
        return result, None


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._input_norm_weight = torch.nn.Parameter(torch.ones(3))
        self._uvqk_weight = torch.nn.Parameter(torch.ones(3, 3))

    def forward(self, x):
        ctx = SimpleNamespace(saved_tensors=(), needs_input_grad=(True, True))
        ctx.save_for_backward = lambda *values: setattr(ctx, "saved_tensors", values)
        result = _HSTUPreprocessAndAttentionFunction.forward(ctx, x, self._input_norm_weight)
        del ctx.save_for_backward  # Production ctx.__dict__ contains scalar controls only.
        return result, ctx


class AttentionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layers = {f"model._stu_layers.{index}": Layer() for index in (0, 1, 2)}
        self.original = {name: inspect.getattr_static(_HSTUPreprocessAndAttentionFunction, name)
                         for name in ("forward", "backward")}
        _HSTUPreprocessAndAttentionFunction.mutate = False
        _HSTUPreprocessAndAttentionFunction.huge = False
        self.probe = adapter.install(None, self.tmp.name,
                                     function_class=_HSTUPreprocessAndAttentionFunction,
                                     layer_modules=self.layers)
        self.probe.set_attempt("current", 0, 101)

    def tearDown(self):
        self.probe.close()
        self.probe.close()
        for name, expected in self.original.items():
            self.assertIs(inspect.getattr_static(_HSTUPreprocessAndAttentionFunction, name), expected)
        self.assertFalse(torch.cuda.is_initialized())
        self.tmp.cleanup()

    def capture(self, *, mutate=False, huge=False):
        raw = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        x = raw[:, 1:7:2]
        original = x.clone()
        _, ctx = self.layers[self.probe.target](x)
        _HSTUPreprocessAndAttentionFunction.mutate = mutate
        _HSTUPreprocessAndAttentionFunction.huge = huge
        dsilu, dout = x / 100, x / 200
        with self.assertRaises(self.probe.stop_type) as caught:
            _HSTUPreprocessAndAttentionFunction.backward(ctx, dsilu, dout)
        result = torch.load(caught.exception.dump_path, map_location="cpu", weights_only=False)
        return result, original, x, caught.exception

    def test_finite_capture_and_pristine_source_after_mutation(self):
        for mutate in (False, True):
            if mutate:
                self.probe.close()
                self.tmp.cleanup()
                self.setUp()
            result, original, source, stopped = self.capture(mutate=mutate)
            payload = decode_payload(result, "cpu")
            self.assertEqual(result["report"]["reason"], "requested_capture")
            self.assertEqual(result["report"]["metadata"]["layer"], "model._stu_layers.2")
            self.assertEqual(len(result["report"]["events"]), 1)
            self.assertEqual(result["report"]["events"][0]["direction"], "backward")
            self.assertEqual(changed_capture_inputs(result), [])
            saved_x = payload["replay"]["saved_tensors"][0]
            self.assertTrue(torch.equal(saved_x, original))
            self.assertEqual(saved_x.stride(), source.stride())
            self.assertEqual(saved_x.storage_offset(), source.storage_offset())
            self.assertEqual(saved_x.untyped_storage().nbytes(), source.untyped_storage().nbytes())
            self.assertEqual(bool(result["report"]["source_inputs_with_tracked_mutations"]), mutate)
            self.assertTrue(torch.equal(payload["outputs"][0], source))
            self.assertTrue(Path(stopped.report_path).is_file())
            self.assertTrue(self.probe.stopped)

    def test_magnitude_is_captured_as_anomaly(self):
        result, _, _, _ = self.capture(huge=True)
        self.assertEqual(result["report"]["reason"], "magnitude")
        self.assertTrue(result["report"]["all_inputs_within_bound"])
        self.assertFalse(result["report"]["all_outputs_within_bound"])

    def test_wrong_layer_is_rejected_before_execution(self):
        x = torch.ones(2, 3)
        _, ctx = self.layers["model._stu_layers.1"](x)
        with self.assertRaisesRegex(nan_backward_boundaries.BoundaryProbeError, "not the verified final"):
            _HSTUPreprocessAndAttentionFunction.backward(ctx, x, x)
        self.assertEqual(list(Path(self.tmp.name).glob("*.pt")), [])

    def test_changed_saved_weight_view_is_rejected(self):
        x = torch.ones(2, 3)
        _, ctx = self.layers[self.probe.target](x)
        ctx.saved_tensors = (x, torch.ones(3))
        with self.assertRaisesRegex(nan_backward_boundaries.BoundaryProbeError, "norm-weight view"):
            _HSTUPreprocessAndAttentionFunction.backward(ctx, x, x)

    def test_capture_failure_does_not_claim_payload(self):
        self.probe.tripwire.capture_limit_bytes = 1
        x = torch.ones(2, 3)
        _, ctx = self.layers[self.probe.target](x)
        with self.assertRaisesRegex(nan_backward_boundaries.BoundaryProbeError, "capture failed") as caught:
            _HSTUPreprocessAndAttentionFunction.backward(ctx, x, x)
        self.assertFalse(hasattr(caught.exception, "dump_path"))

    def test_no_global_rng_consumption(self):
        before = torch.get_rng_state().clone()
        self.capture()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
