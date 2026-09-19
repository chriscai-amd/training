"""CPU checks for isolated uncapped batched-monitor launch provenance."""
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
import capture_training_backward_batched as entry


class BatchedTrainingEntryTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.options = {"directory": str(self.root / "new_capture"), "steps": 3,
                        "capture_mode": "monitor_on_anomaly", "target": "_stu_layers.1.",
                        "operations": list(entry.OUTPUT_OPERATIONS),
                        "abs_threshold": 1e6, "dataset": "yambda-5b"}

    def test_explicit_uncapped_options_and_other_training_environment_preserved(self):
        environment = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "BATCH_SIZE": "1024"}
        original = base.configure_environment
        entry.configure_environment(self.options, environment)
        self.assertIs(base.configure_environment, original)
        self.assertEqual(environment["HSTU_BWD_MAX_VGPR"], "0")
        self.assertEqual(environment["START_TS"], "0")
        self.assertEqual(environment["CKPT_PATH"], "")
        self.assertEqual(environment["DIE_AT_STEP"], "3")
        self.assertEqual(environment["NAN_BACKWARD_TARGET"], "_stu_layers.1.")
        self.assertEqual(environment["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(environment["BATCH_SIZE"], "1024")
        options = json.loads(environment["NAN_TRAINING_OPTIONS"])
        self.assertEqual(options["monitor_variant"], entry.VARIANT)
        self.assertEqual(options["attention_vgpr_cap"], 0)
        self.assertEqual(options["operations"], list(entry.OUTPUT_OPERATIONS))
        self.assertNotIn("monitor_variant", self.options)

    def test_environment_conflicts_are_atomic(self):
        conflicts = ({"HSTU_BWD_MAX_VGPR": "256"}, {"START_TS": "150"},
                     {"NAN_REPLAY_DIR": "/other"}, {"NAN_TRIPWIRE_DIR": "/other"},
                     {"CKPT_PATH": "/checkpoint"}, {"NAN_MODULE_PROBE": "1"},
                     {"AMDGCN_USE_BUFFER_OPS": "1"}, {"DIE_AT_STEP": "17"})
        for environment in conflicts:
            with self.subTest(environment=environment):
                before = dict(environment)
                with self.assertRaises(ValueError):
                    entry.configure_environment(self.options, environment)
                self.assertEqual(environment, before)

    def test_invalid_scope_mode_and_bounds_fail_without_environment_changes(self):
        for changes in ({"capture_mode": "pristine"}, {"operations": ["hstu_attention_bwd"]},
                        {"operations": []}, {"operations": "hstu_output_grad_mm"},
                        {"operations": [entry.OUTPUT_OPERATIONS[0]] * 2},
                        {"steps": 0}, {"abs_threshold": float("nan")}):
            with self.subTest(changes=changes):
                environment = {"HSTU_BWD_MAX_VGPR": "0"}
                with self.assertRaises(ValueError):
                    entry.configure_environment({**self.options, **changes}, environment)
                self.assertEqual(environment, {"HSTU_BWD_MAX_VGPR": "0"})

    def test_worker_preserves_bootstrap_order_injects_only_new_probe_and_restores_loop(self):
        from nan_backward_training_batched import BatchedTrainingBackwardBoundaryProbe

        order = []
        gin = ModuleType("gin")
        gin.parse_config_file = lambda *args, **kwargs: order.append("gin")
        bootstrap = ModuleType("generative_recommenders.dlrm_v4.train._env_bootstrap")
        bootstrap.apply_env_bootstrap = lambda: order.append("bootstrap")
        original_loop = lambda **kwargs: None
        utils = SimpleNamespace(streaming_train_eval_loop=original_loop)

        def train(*args):
            self.assertEqual(order, ["gin", "bootstrap", "utils", "trainer"])
            self.assertEqual(args, (0, 1, 0, 1, "localhost", "12345", "test.gin", "streaming-train-eval"))
            self.assertEqual(utils.streaming_train_eval_loop(model="fake"), 123)
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
                patch.object(base, "run_instrumented_loop", return_value=123) as wrapped, \
                self.assertRaisesRegex(RuntimeError, "trainer fault"):
            entry.diagnostic_worker(0, 1, 0, 1, "localhost", "12345", "test.gin", "streaming-train-eval")
        self.assertIs(utils.streaming_train_eval_loop, original_loop)
        args, kwargs = wrapped.call_args
        self.assertIs(args[0], original_loop)
        self.assertEqual(args[1]["attention_vgpr_cap"], 0)
        self.assertEqual(args[3], {"model": "fake"})
        self.assertIs(kwargs["probe_factory"], BatchedTrainingBackwardBoundaryProbe)

    def test_worker_rejects_lost_uncapped_provenance_before_runtime_import(self):
        environment = {}
        entry.configure_environment(self.options, environment)
        for cap in ("256", "512", ""):
            with self.subTest(cap=cap), patch.dict(os.environ, {**environment, "HSTU_BWD_MAX_VGPR": cap}), \
                    patch.object(entry.importlib, "import_module", side_effect=AssertionError("runtime import")), \
                    self.assertRaisesRegex(ValueError, "uncapped provenance"):
                entry.diagnostic_worker(0, 1, 0, 1, "localhost", "12345", "test.gin", "streaming-train-eval")

    def test_main_invalid_arguments_do_not_import_trainer(self):
        prefix = ["--directory", str(self.root / "new"), "--steps", "3"]
        bad = [prefix + ["--steps", "0"], prefix + ["--operations", "hstu_silu_bwd"],
               prefix + ["--capture-mode", "pristine"], prefix + ["--directory", str(self.root)]]
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()), \
                patch.object(entry.importlib, "import_module", side_effect=AssertionError("trainer import")):
            for args in bad:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    entry.main(args)

    def test_main_restores_spawn_target_and_argv_after_trainer_failure(self):
        original_worker = lambda: None
        previous_argv = sys.argv
        trainer = SimpleNamespace(_main_func=original_worker)

        def run():
            self.assertIs(trainer._main_func, entry.diagnostic_worker)
            self.assertEqual(os.environ["HSTU_BWD_MAX_VGPR"], "0")
            self.assertEqual(sys.argv[1:], ["--dataset", "yambda-5b", "--mode", "streaming-train-eval"])
            raise RuntimeError("launch failed")

        trainer.main = run
        with patch.dict(os.environ, {}, clear=True), patch.object(entry.importlib, "import_module", return_value=trainer), \
                self.assertRaisesRegex(RuntimeError, "launch failed"):
            entry.main(["--directory", str(self.root / "new"), "--steps", "3"])
        self.assertIs(trainer._main_func, original_worker)
        self.assertIs(sys.argv, previous_argv)


if __name__ == "__main__":
    unittest.main()
