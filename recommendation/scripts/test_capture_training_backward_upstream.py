"""CPU launch/provenance checks for the adjacent-layer upstream monitor."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import capture_training_backward as base
import capture_training_backward_upstream as entry


class UpstreamEntryTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.options = {"directory": str(self.root / "new"), "steps": 1000,
                        "capture_mode": "monitor_on_anomaly", "upper_layer": 2,
                        "lower_layer": 1, "abs_threshold": 1e6, "dataset": "yambda-5b"}

    def test_records_exact_layer_scope_with_uncapped_environment_and_preserves_allocator(self):
        environment = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "BATCH_SIZE": "1024"}
        original = base.configure_environment
        entry.configure_environment(self.options, environment)
        self.assertIs(base.configure_environment, original)
        self.assertEqual(environment["HSTU_BWD_MAX_VGPR"], "0")
        self.assertEqual(environment["NAN_BACKWARD_TARGET"], "*")
        self.assertEqual(environment["DIE_AT_STEP"], "1000")
        self.assertEqual(environment["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        options = json.loads(environment["NAN_TRAINING_OPTIONS"])
        self.assertEqual(options["operations"], list(base.ALL_OPERATIONS))
        self.assertEqual(options["layer_operation_selection"], {
            "_stu_layers.2.": list(base.ALL_OPERATIONS), "_stu_layers.1.": ["hstu_output_grad_mm"]})
        self.assertTrue(options["lower_witness_input_only"])
        self.assertEqual(options["attention_vgpr_cap"], 0)
        self.assertNotIn("operations", self.options)

    def test_environment_conflicts_and_invalid_scope_are_atomic(self):
        for environment in ({"HSTU_BWD_MAX_VGPR": "256"}, {"START_TS": "1"},
                            {"NAN_REPLAY_DIR": "/other"}, {"CKPT_PATH": "/checkpoint"},
                            {"NAN_MODULE_PROBE": "1"}, {"DIE_AT_STEP": "600"}):
            with self.subTest(environment=environment):
                before = dict(environment)
                with self.assertRaises(ValueError):
                    entry.configure_environment(self.options, environment)
                self.assertEqual(environment, before)
        for change in ({"upper_layer": 1}, {"lower_layer": -1}, {"upper_layer": 3},
                       {"capture_mode": "pristine"}, {"steps": 0}, {"abs_threshold": float("nan")}):
            with self.subTest(change=change):
                environment = {"HSTU_BWD_MAX_VGPR": "0"}
                with self.assertRaises(ValueError):
                    entry.configure_environment({**self.options, **change}, environment)
                self.assertEqual(environment, {"HSTU_BWD_MAX_VGPR": "0"})

    def test_worker_bootstraps_before_imports_and_passes_layer_indices_to_factory(self):
        from nan_backward_training_upstream import UpstreamBatchedTrainingBackwardBoundaryProbe
        order = []
        gin = ModuleType("gin")
        gin.parse_config_file = lambda *args, **kwargs: order.append("gin")
        bootstrap = ModuleType("generative_recommenders.dlrm_v4.train._env_bootstrap")
        bootstrap.apply_env_bootstrap = lambda: order.append("bootstrap")
        original_loop = lambda **kwargs: None
        utils = SimpleNamespace(streaming_train_eval_loop=original_loop)

        def train(*args):
            self.assertEqual(order, ["gin", "bootstrap", "utils", "trainer"])
            self.assertEqual(utils.streaming_train_eval_loop(model="fake"), 77)
            raise RuntimeError("trainer fault")

        trainer = SimpleNamespace(_main_func=train)

        def imported(name):
            key = name.rsplit(".", 1)[-1]
            order.append("trainer" if key == "train_ranker" else key)
            return trainer if key == "train_ranker" else utils

        environment = {}
        entry.configure_environment(self.options, environment)
        with patch.dict(os.environ, environment), patch.dict(sys.modules, {"gin": gin, bootstrap.__name__: bootstrap}), \
                patch.object(entry.importlib, "import_module", side_effect=imported), \
                patch.object(base, "run_instrumented_loop", return_value=77) as wrapped, \
                self.assertRaisesRegex(RuntimeError, "trainer fault"):
            entry.diagnostic_worker(0, 1, 0, 1, "localhost", "29883", "test.gin", "streaming-train-eval")
        self.assertIs(utils.streaming_train_eval_loop, original_loop)
        args, kwargs = wrapped.call_args
        self.assertEqual(args[1]["layer_operation_selection"], entry.layer_selection(2, 1))
        factory = kwargs["probe_factory"]
        self.assertIs(factory.func, UpstreamBatchedTrainingBackwardBoundaryProbe)
        self.assertEqual(factory.keywords, {"upper_layer": 2, "lower_layer": 1})

    def test_worker_rejects_lost_scope_or_cap_before_runtime_import(self):
        environment = {}
        entry.configure_environment(self.options, environment)
        changed = dict(environment)
        options = json.loads(changed["NAN_TRAINING_OPTIONS"])
        options["layer_operation_selection"]["_stu_layers.1."] = []
        changed["NAN_TRAINING_OPTIONS"] = json.dumps(options)
        for candidate in ({**environment, "HSTU_BWD_MAX_VGPR": "256"},
                          {**environment, "NAN_BACKWARD_TARGET": "_stu_layers.1."}, changed):
            with self.subTest(candidate=candidate), patch.dict(os.environ, candidate), \
                    patch.object(entry.importlib, "import_module", side_effect=AssertionError("runtime import")), \
                    self.assertRaisesRegex(ValueError, "per-layer provenance"):
                entry.diagnostic_worker(0, 1, 0, 1, "localhost", "29883", "test.gin", "streaming-train-eval")

    def test_main_invalid_scope_and_existing_output_precede_trainer_import(self):
        prefix = ["--directory", str(self.root / "new"), "--steps", "1000"]
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()), \
                patch.object(entry.importlib, "import_module", side_effect=AssertionError("trainer import")):
            for arguments in (prefix + ["--upper-layer", "3"], prefix + ["--steps", "0"],
                              prefix + ["--directory", str(self.root)]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    entry.main(arguments)

    def test_main_restores_worker_and_argv_after_trainer_failure(self):
        original_worker = lambda: None
        previous_argv = sys.argv
        trainer = SimpleNamespace(_main_func=original_worker)

        def run():
            self.assertIs(trainer._main_func, entry.diagnostic_worker)
            self.assertEqual(os.environ["HSTU_BWD_MAX_VGPR"], "0")
            self.assertEqual(os.environ["NAN_BACKWARD_TARGET"], "*")
            raise RuntimeError("launch failed")

        trainer.main = run
        with patch.dict(os.environ, {}, clear=True), patch.object(entry.importlib, "import_module", return_value=trainer), \
                self.assertRaisesRegex(RuntimeError, "launch failed"):
            entry.main(["--directory", str(self.root / "new"), "--steps", "1000"])
        self.assertIs(trainer._main_func, original_worker)
        self.assertIs(sys.argv, previous_argv)


if __name__ == "__main__":
    unittest.main()
