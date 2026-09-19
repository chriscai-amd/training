"""CPU fidelity and independent-reference tests; no Triton/GPU invocation."""

import ast
from contextlib import redirect_stderr
import io
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scripts import replay_nan_tripwire as replay
from scripts import repro_output_backward_stages as stages


def norm_fixture(*, concat_u=True, concat_x=True, silu_u=False,
                 activation="none", training=True, compute_y=True):
    generator = torch.Generator().manual_seed(71)
    n, d = 5, 7

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    x, u, weight, bias = rand(n, d), rand(n, d), rand(d), rand(d)
    eps = 1e-5
    return {
        "x": x, "u": u, "weight": weight, "bias": bias,
        "mean": x.mean(dim=-1), "rstd": torch.rsqrt(x.var(dim=-1, correction=0) + eps),
        "dy": rand(n, d * (1 + int(concat_u) + int(concat_x))),
        "random_mask": torch.randint(0, 2 ** (1 + int(concat_u) + int(concat_x)),
                                     (n, d), generator=generator, dtype=torch.int8),
        "eps": eps, "dropout_ratio": 0.25, "training": training,
        "concat_u": concat_u, "concat_x": concat_x, "silu_u": silu_u,
        "mul_u_activation_type": activation, "compute_y": compute_y,
        "seed": 17, "BLOCK_D": 8, "num_warps": 1,
    }


def autograd_reference(inputs):
    """Differentiate native layer_norm/activation with explicit fixed masks."""
    x, u, weight, bias = [inputs[name].clone().requires_grad_()
                          for name in ("x", "u", "weight", "bias")]
    activation = inputs["mul_u_activation_type"]
    multiplier = F.silu(u) if activation == "silu" else torch.sigmoid(u) if activation == "sigmoid" else u
    normalized = F.layer_norm(x, (x.shape[1],), weight, bias, inputs["eps"])
    parts = []
    if inputs["concat_u"]:
        parts.append(F.silu(u) if inputs["silu_u"] else u)
    if inputs["concat_x"]:
        parts.append(x)
    parts.append(normalized * multiplier)
    if inputs["training"]:
        # Packed bits follow output segments in reverse order: y is always bit 0.
        parts = [torch.where((inputs["random_mask"] & (1 << (len(parts) - i - 1))) != 0,
                             part / (1 - inputs["dropout_ratio"]), 0.0)
                 for i, part in enumerate(parts)]
    y = torch.cat(parts, dim=1)
    gradients = torch.autograd.grad(y, (x, u, weight, bias), inputs["dy"])
    return (*gradients, y.detach() if inputs["compute_y"] else None)


def collect_reference(inputs, stage, rows):
    chunks = {}
    for name, start, value in stages.reference_chunks(inputs, stage, rows):
        chunks.setdefault(name, []).append(value)
    names = ("dy",) if stage == "mm" else stages.NORM_OUTPUT_NAMES
    return tuple(None if name not in chunks else chunks[name][0] if name.startswith("d_norm_")
                 else torch.cat(chunks[name], dim=0) for name in names)


class IndependentReferenceTest(unittest.TestCase):
    def test_norm_matches_native_autograd_for_all_supported_paths(self):
        for concat_u, concat_x, silu_u, activation, training, compute_y in itertools.product(
            (False, True), (False, True), (False, True), ("none", "silu", "sigmoid"),
            (False, True), (False, True),
        ):
            with self.subTest(concat_u=concat_u, concat_x=concat_x, silu_u=silu_u,
                              activation=activation, training=training, compute_y=compute_y):
                inputs = norm_fixture(concat_u=concat_u, concat_x=concat_x, silu_u=silu_u,
                                      activation=activation, training=training, compute_y=compute_y)
                expected = autograd_reference(inputs)
                actual = collect_reference(inputs, "norm", rows=2)
                for left, right in zip(actual, expected):
                    if right is None:
                        self.assertIsNone(left)
                    else:
                        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

    def test_mm_noncontiguous_inputs_and_partial_final_chunk(self):
        inputs = {"dout": torch.arange(30, dtype=torch.float64).reshape(6, 5).t(),
                  "weight": torch.arange(24, dtype=torch.float64).reshape(6, 4).t()}
        actual, = collect_reference(inputs, "mm", rows=2)
        torch.testing.assert_close(actual, inputs["dout"] @ inputs["weight"].t(), rtol=0, atol=0)

    def test_mm_reference_applies_captured_autocast_input_rounding(self):
        inputs = {"dout": torch.tensor([[1.001, 0.2222]], dtype=torch.float32),
                  "weight": torch.tensor([[1.004, 3.001]], dtype=torch.float32)}
        _, _, actual = next(stages.reference_chunks(inputs, "mm", 1, mm_input_dtype=torch.bfloat16))
        expected = inputs["dout"].bfloat16().double() @ inputs["weight"].bfloat16().double().t()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, inputs["dout"].double() @ inputs["weight"].double().t()))

    def test_reference_detects_finite_corruption_in_norm_weight_gradient(self):
        inputs = norm_fixture()
        outputs = list(autograd_reference(inputs))
        outputs[2][3] = 1e30
        report = stages.compare_reference(inputs, "norm", outputs, rows=2, rtol=1e-10, atol=1e-10)
        self.assertFalse(report["passed"])
        self.assertEqual(report["outputs"]["d_norm_weight"]["mismatched_elements"], 1)
        self.assertEqual(report["outputs"]["d_norm_weight"]["nonfinite_elements"], 0)

    def test_reference_reports_nonfinite_output(self):
        inputs = {"dout": torch.ones(1, 2), "weight": torch.ones(1, 2)}
        report = stages.compare_reference(inputs, "mm", (torch.tensor([[float("inf")]]),),
                                          rows=1, rtol=0, atol=0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["outputs"]["dy"]["nonfinite_elements"], 1)
        self.assertEqual(report["outputs"]["dy"]["max_abs_diff"], float("inf"))

    def test_reference_preserves_saved_statistics(self):
        inputs = norm_fixture()
        before = collect_reference(inputs, "norm", rows=2)
        inputs["rstd"] *= 3
        after = collect_reference(inputs, "norm", rows=2)
        self.assertFalse(torch.equal(before[0], after[0]))
        self.assertFalse(torch.equal(before[2], after[2]))

    def test_training_without_original_mask_rejects_reference(self):
        for mask in (None, torch.empty(0)):
            inputs = norm_fixture()
            inputs["random_mask"] = mask
            with self.assertRaisesRegex(ValueError, "captured packed dropout mask"):
                list(stages.reference_chunks(inputs, "norm", 2))


class PreparationFidelityTest(unittest.TestCase):
    def test_real_backward_tensor_mapping_and_preparation_dy_identity(self):
        # Execute the repository's actual backward body on CPU with only its norm
        # helper replaced. This verifies saved tensor indexing without importing
        # the Triton module or copying its backward implementation into the test.
        path = Path(stages.__file__).parents[1] / "generative_recommenders/ops/triton/triton_hstu_linear.py"
        module = ast.parse(path.read_text())
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "HSTUComputeOutputFunction")
        backward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "backward")
        backward.decorator_list = []
        source = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), backward], type_ignores=[])
        ast.fix_missing_locations(source)
        for recompute, has_mask in itertools.product((False, True), repeat=2):
            with self.subTest(recompute=recompute, has_mask=has_mask):
                inputs = norm_fixture(compute_y=recompute)
                outputs = autograd_reference({**inputs, "compute_y": True})
                dout = torch.arange(15, dtype=torch.float64).reshape(5, 3)
                output_weight = torch.arange(63, dtype=torch.float64).reshape(21, 3)
                seen_dy = []

                def helper(dy, *, compute_y=False, random_mask=None, **kwargs):
                    seen_dy.append(dy)
                    return (*outputs[:4], outputs[4] if compute_y else None)

                linear = SimpleNamespace(triton_layer_norm_mul_dropout_bwd=helper)
                namespace = {"torch": torch, "triton_layer_norm_mul_dropout_bwd": lambda *args, **kwargs: linear.triton_layer_norm_mul_dropout_bwd(*args, **kwargs)}
                exec(compile(source, str(path), "exec"), namespace)
                saved = [inputs[name] for name in ("x", "u", "weight", "bias", "mean", "rstd")] + [output_weight]
                if not recompute:
                    saved.append(outputs[4])
                if has_mask:
                    saved.append(inputs["random_mask"])
                ctx = SimpleNamespace(**{name: value for name, value in inputs.items() if not isinstance(value, torch.Tensor)},
                                      saved_tensors=tuple(saved), recompute_y_in_backward=recompute,
                                      has_random_mask=has_mask, group_norm=False)
                decoded = {"args": (dout,), "saved_tensors": tuple(saved)}

                def replay_once(*args, **kwargs):
                    actual = namespace["backward"](ctx, dout)
                    self.assertIs(actual[2], dout)
                    torch.testing.assert_close(actual[5], outputs[4].t() @ dout)
                    return actual, {"replay": decoded, "outputs": actual}

                with patch.object(replay, "replay_once", side_effect=replay_once) as execute:
                    retained, decoded_result, comparison = stages.prepare({}, linear)
                self.assertIs(retained["dy"], seen_dy[0])
                torch.testing.assert_close(retained["dy"], dout @ output_weight.t())
                self.assertIs(retained["random_mask"], inputs["random_mask"] if has_mask else None)
                self.assertEqual(retained["compute_y"], recompute)
                self.assertIs(decoded_result["saved_tensors"][6], output_weight)
                self.assertTrue(comparison["passed"])
                self.assertIs(linear.triton_layer_norm_mul_dropout_bwd, helper)
                self.assertEqual(execute.call_args.kwargs, {"snapshot_original_outputs": True})

    def test_preparation_restores_helper_after_exception_or_extra_call(self):
        def helper(dy):
            return dy

        linear = SimpleNamespace(triton_layer_norm_mul_dropout_bwd=helper)

        def twice(*args, **kwargs):
            linear.triton_layer_norm_mul_dropout_bwd(torch.ones(1))
            linear.triton_layer_norm_mul_dropout_bwd(torch.ones(1))

        for side_effect in (RuntimeError("execution failed"), twice):
            with patch.object(replay, "replay_once", side_effect=side_effect):
                with self.assertRaises((RuntimeError, ValueError)):
                    stages.prepare({}, linear)
            self.assertIs(linear.triton_layer_norm_mul_dropout_bwd, helper)


class OriginalStageTest(unittest.TestCase):
    def encode(self, inputs, outputs, stage, settings=None):
        storages, indices = [], {}

        def encode_tensor(tensor):
            if tensor is None:
                return None
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key not in indices:
                indices[key] = len(storages)
                storages.append(torch.empty(0, dtype=torch.uint8).set_(
                    storage, 0, (storage.nbytes(),), (1,),
                ).clone())
            return {"__tripwire_tensor__": indices[key], "dtype": str(tensor.dtype),
                    "device": "cpu", "shape": tuple(tensor.shape),
                    "stride": tuple(tensor.stride()), "storage_offset": tensor.storage_offset()}

        return {"storages": storages, "payload": {
            "replay": None,
            "stage": {"format_version": 1, "name": stage, "settings": settings or {},
                      # This intentionally invalid descriptor must never be decoded.
                      "parent_replay": {"saved_tensors": [{"__tripwire_tensor__": 999, "device": "cuda:0"}]}},
            "inputs": {name: encode_tensor(value) for name, value in inputs.items()},
            "outputs": {name: encode_tensor(value) for name, value in outputs.items()},
        }}

    def test_original_mm_decodes_strides_without_parent_preparation(self):
        dout = torch.arange(15.0).reshape(3, 5).t()
        weight = torch.ones(7, 3)
        capture = self.encode({"dout": dout, "output_weight": weight}, {"dy": dout @ weight.t()}, "gradient_mm")
        with patch.object(replay, "replay_once", side_effect=AssertionError("must not prepare")):
            fixed, original = stages.original_stage_inputs(capture, "mm", "cpu")
        self.assertEqual(fixed["dout"].stride(), dout.stride())
        torch.testing.assert_close(fixed["dout"], dout)
        torch.testing.assert_close(fixed["weight"], weight)
        torch.testing.assert_close(original["dy"], dout @ weight.t())

    def test_original_norm_preserves_mask_and_scalar_settings(self):
        values = norm_fixture(compute_y=False)
        tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
        settings = {name: value for name, value in values.items() if not isinstance(value, torch.Tensor)}
        outputs = dict(zip(stages.NORM_OUTPUT_NAMES, autograd_reference(values)))
        capture = self.encode(tensors, outputs, "norm", settings)
        fixed, original = stages.original_stage_inputs(capture, "norm", "cpu")
        self.assertEqual(fixed.keys(), values.keys())
        self.assertEqual(fixed["seed"], values["seed"])
        torch.testing.assert_close(fixed["random_mask"], values["random_mask"])
        self.assertIsNone(original["y"])
        report = stages.compare_reference(fixed, "norm", tuple(original.values()), rows=2, rtol=1e-10, atol=1e-10)
        self.assertTrue(report["passed"])

    def test_original_output_alias_is_frozen_before_replay(self):
        tensor = torch.ones(2, 2)
        capture = self.encode({"dout": tensor, "output_weight": torch.eye(2)}, {"dy": tensor}, "gradient_mm")
        fixed, original = stages.original_stage_inputs(capture, "mm", "cpu")
        fixed["dout"].zero_()
        torch.testing.assert_close(original["dy"], tensor)

    def test_wrong_stage_is_rejected(self):
        capture = {"payload": {"stage": {"format_version": 1, "name": "gradient_mm"}}}
        with self.assertRaisesRegex(ValueError, "requested original stage: norm"):
            stages.original_stage_inputs(capture, "norm", "cpu")


class InputGuardTest(unittest.TestCase):
    def test_byte_comparison_handles_views_and_signed_zero(self):
        for view in (torch.arange(24.0).reshape(4, 6).t(), torch.arange(24.0)[2::3],
                     torch.ones(1).expand(4, 3), torch.empty(0), torch.tensor(0.0)):
            with self.subTest(shape=view.shape, stride=view.stride()):
                self.assertTrue(stages.all_equal(torch, view, view.clone(), chunk_size=3, bytewise=True))
        self.assertTrue(stages.all_equal(torch, torch.tensor(0.0), torch.tensor(-0.0)))
        self.assertFalse(stages.all_equal(torch, torch.tensor(0.0), torch.tensor(-0.0), bytewise=True))

    def test_report_cannot_overwrite_capture_through_hardlink(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.pt"
            capture.write_bytes(b"immutable")
            alias = Path(directory) / "alias.pt"
            alias.hardlink_to(capture)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                stages.arguments([str(capture), "--stage", "mm", "--report", str(alias)])
            self.assertEqual(capture.read_bytes(), b"immutable")

    def test_no_gpu_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
