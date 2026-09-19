"""CPU checks for the opt-in weighted LN backward DX configuration pin."""

import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "generative_recommenders/ops/triton/triton_layer_norm.py"
PINNING_SOURCE = SOURCE.with_name("_autotune_pinning.py")


class FakeConfig:
    def __init__(self, kwargs, num_warps=4):
        self.kwargs, self.num_warps = kwargs, num_warps


class WeightedLayerNormConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
        helpers = {
            "_get_layer_norm_fwd_configs", "_get_bwd_dwdb_configs",
            "_get_weighted_layer_norm_bwd_dx_configs",
        }
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helpers]
        pinning = ast.parse(PINNING_SOURCE.read_text(), filename=str(PINNING_SOURCE))
        nodes += [node for node in pinning.body if isinstance(node, ast.FunctionDef) and node.name == "pinned_or_full"]
        future = ast.parse("from __future__ import annotations").body
        cls.helper_code = compile(ast.Module(body=future + nodes, type_ignores=[]), str(SOURCE), "exec")
        cls.config_code = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "triton_autotune":
                        configs = next(arg.value for arg in decorator.keywords if arg.arg == "configs")
                        cls.config_code[node.name] = compile(ast.Expression(configs), str(SOURCE), "eval")

    def namespace(self, *, sm100=False):
        values = {"os": os, "triton": SimpleNamespace(Config=FakeConfig),
                  "torch": SimpleNamespace(ops=SimpleNamespace(hip=True)),
                  "is_sm100_plus": Mock(return_value=sm100)}
        exec(self.helper_code, values)
        return values

    def configs(self, namespace, kernel="_weighted_layer_norm_bwd_dx"):
        # Evaluate the actual kernel decorator expression, so a disconnected
        # helper does not make these configuration-selection checks pass.
        return [(config.kwargs, config.num_warps)
                for config in eval(self.config_code[kernel], namespace)]

    def test_unset_and_zero_preserve_existing_pinned_candidates(self):
        for value in (None, "0"):
            for full in (None, "0"):
                environment = {}
                if value is not None:
                    environment["WEIGHTED_LN_BWD_BLOCK_N"] = value
                if full is not None:
                    environment["TRITON_FULL_AUTOTUNE"] = full
                with self.subTest(value=value, full=full), patch.dict(os.environ, environment, clear=True):
                    namespace = self.namespace()
                    self.assertEqual(self.configs(namespace), [({"BLOCK_N": 1}, 1), ({"BLOCK_N": 8}, 1)])
                    namespace["is_sm100_plus"].assert_not_called()

    def test_unset_and_zero_preserve_full_architecture_search(self):
        for value in (None, "0"):
            for sm100, block_ns in ((False, (1, 2, 4, 8)), (True, (4, 8, 16))):
                environment = {"TRITON_FULL_AUTOTUNE": "1"}
                if value is not None:
                    environment["WEIGHTED_LN_BWD_BLOCK_N"] = value
                with self.subTest(value=value, sm100=sm100), patch.dict(os.environ, environment, clear=True):
                    namespace = self.namespace(sm100=sm100)
                    self.assertEqual(self.configs(namespace),
                                     [({"BLOCK_N": block_n}, warps) for block_n in block_ns for warps in (1, 2, 4, 8)])
                    namespace["is_sm100_plus"].assert_called_once_with()

    def test_explicit_pin_wins_over_full_autotune_without_architecture_query(self):
        for value in ("1", "8"):
            for full in ("0", "1"):
                with self.subTest(value=value, full=full), patch.dict(os.environ, {
                    "WEIGHTED_LN_BWD_BLOCK_N": value, "TRITON_FULL_AUTOTUNE": full,
                }, clear=True):
                    namespace = self.namespace()
                    self.assertEqual(self.configs(namespace), [({"BLOCK_N": int(value)}, 1)])
                    namespace["is_sm100_plus"].assert_not_called()

    def test_invalid_values_fail_clearly_even_with_full_autotune(self):
        for value in ("", "2", "-1", "16", "1.0", "true", "invalid"):
            for full in ("0", "1"):
                with self.subTest(value=value, full=full), patch.dict(os.environ, {
                    "WEIGHTED_LN_BWD_BLOCK_N": value, "TRITON_FULL_AUTOTUNE": full,
                }, clear=True):
                    with self.assertRaisesRegex(ValueError, "WEIGHTED_LN_BWD_BLOCK_N must be 0 \\(default\\), 1, or 8"):
                        self.configs(self.namespace())

    def test_dx_pin_leaves_forward_and_parameter_reduction_selection_unchanged(self):
        kernels = ("_layer_norm_fwd", "_weighted_layer_norm_fwd", "_layer_norm_bwd_dwdb")
        for full in ("0", "1"):
            with patch.dict(os.environ, {"TRITON_FULL_AUTOTUNE": full}, clear=True):
                baseline = {kernel: self.configs(self.namespace(), kernel) for kernel in kernels}
                for value in ("0", "1", "8"):
                    os.environ["WEIGHTED_LN_BWD_BLOCK_N"] = value
                    for kernel in kernels:
                        with self.subTest(full=full, value=value, kernel=kernel):
                            self.assertEqual(self.configs(self.namespace(), kernel), baseline[kernel])


class ConfigurationProvenanceTest(unittest.TestCase):
    def test_tripwire_report_retains_explicit_dx_pin(self):
        import torch
        from generative_recommenders.dlrm_v4.train.nan_tripwire import NaNTripwire, NonFiniteError

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"WEIGHTED_LN_BWD_BLOCK_N": "1"}):
            tripwire = NaNTripwire(directory)
            tripwire.begin(1)
            tripwire.watch("cpu_bad_output", torch.tensor([float("nan")]))
            with self.assertRaises(NonFiniteError):
                tripwire.check("backward")
            report_path, = Path(directory).glob("*.json")
            report = json.loads(report_path.read_text())
        self.assertEqual(report["environment"]["WEIGHTED_LN_BWD_BLOCK_N"], "1")
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
