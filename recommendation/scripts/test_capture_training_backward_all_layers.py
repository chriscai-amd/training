"""CPU entry checks for uncapped scope, loop wiring, and bootstrap order."""
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

import torch

import capture_training_backward as base
import capture_training_backward_all_layers as entry
import nan_backward_training_all_layers as monitor


class AllLayerEntryTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.options = {"directory": str(self.root / "new"), "steps": 2,
                        "capture_mode": "monitor_on_anomaly", "abs_threshold": 1e6,
                        "dataset": "yambda-5b"}

    def test_environment_records_all42_endpoints_uncapped_and_dense_phases(self):
        environment = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "BATCH_SIZE": "1024"}
        original = base.configure_environment
        entry.configure_environment(self.options, environment)
        self.assertIs(base.configure_environment, original)
        self.assertEqual(environment["HSTU_BWD_MAX_VGPR"], "0")
        self.assertEqual(environment["NAN_BACKWARD_TARGET"], "*")
        self.assertEqual(environment["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        recorded = json.loads(environment["NAN_TRAINING_OPTIONS"])
        self.assertEqual(recorded["operations"], list(base.ALL_OPERATIONS))
        self.assertEqual(recorded["layer_indices"], [0, 1, 2])
        self.assertEqual(recorded["expected_operation_endpoints"], 42)
        self.assertTrue(recorded["dense_phase_sentinels"])
        self.assertEqual(recorded["attention_vgpr_cap"], 0)
        self.assertNotIn("operations", self.options)

    def test_conflicts_rejected_without_partial_environment_mutation(self):
        for environment in ({"HSTU_BWD_MAX_VGPR": "256"}, {"START_TS": "1"},
                            {"NAN_REPLAY_DIR": "/other"}, {"CKPT_PATH": "/checkpoint"},
                            {"NAN_MODULE_PROBE": "1"}, {"DIE_AT_STEP": "600"}):
            with self.subTest(environment=environment):
                before = dict(environment)
                with self.assertRaises(ValueError):
                    entry.configure_environment(self.options, environment)
                self.assertEqual(environment, before)

    def test_loop_uses_local_optimizer_proxy_and_restores_forward_at_bound(self):
        model = torch.nn.Linear(1, 1)
        metrics = SimpleNamespace(global_step={"train": 0})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer_proxy = object()
        probes = []
        class FakeProbe:
            def __init__(self, actual_model, directory, **options):
                self.sentinel = SimpleNamespace(optimizer_proxy=optimizer_proxy)
                self.options, self.attempts, self.closed = options, [], False
                probes.append(self)
            def set_attempt(self, *attempt):
                self.attempts.append(attempt)
            def close(self):
                self.closed = True
        environment = {}
        entry.configure_environment(self.options, environment)
        options = json.loads(environment["NAN_TRAINING_OPTIONS"])
        original_forward, original_step = model.forward, optimizer.step
        def loop(**kwargs):
            self.assertIs(kwargs["optimizer"], optimizer_proxy)
            self.assertEqual(kwargs["die_at_step"], 2)
            for _ in range(2):
                kwargs["model"].forward(torch.ones(1, 1))
                metrics.global_step["train"] += 1
            raise SystemExit(42)
        original_runner = base.run_instrumented_loop
        def cpu_runner(*args, **kwargs):
            return original_runner(*args, **kwargs,
                                   context_factory=lambda: {"source_sha256": {}},
                                   archive_factory=lambda *args: {"test": True})
        kwargs = {"model": model, "optimizer": optimizer, "metric_logger": metrics,
                  "rank": 0, "resume_train_ts": None, "resume_cold_start": True}
        with patch.object(monitor, "AllLayersBatchedTrainingBackwardBoundaryProbe", FakeProbe), \
                patch.object(base, "run_instrumented_loop", side_effect=cpu_runner), \
                self.assertRaises(SystemExit) as raised:
            entry.run_instrumented_loop(loop, options, (), kwargs)
        self.assertEqual(raised.exception.code, 42)
        self.assertIs(probes[0].options["optimizer"], optimizer)
        self.assertEqual(probes[0].attempts, [("training", 0, 1), ("training", 0, 2)])
        self.assertTrue(probes[0].closed)
        self.assertEqual(model.forward, original_forward)
        self.assertEqual(optimizer.step, original_step)
        self.assertEqual(json.loads((self.root / "new/outcome.json").read_text())["status"], "bounded_complete")
        self.assertFalse(torch.cuda.is_initialized())

    def test_worker_bootstraps_before_runtime_imports_and_restores_loop(self):
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
                patch.object(entry, "run_instrumented_loop", return_value=77), \
                self.assertRaisesRegex(RuntimeError, "trainer fault"):
            entry.diagnostic_worker(0, 1, 0, 1, "localhost", "29884", "test.gin", "streaming-train-eval")
        self.assertIs(utils.streaming_train_eval_loop, original_loop)

    def test_worker_rejects_lost_scope_or_cap_before_runtime_import(self):
        environment = {}
        entry.configure_environment(self.options, environment)
        changed = dict(environment)
        options = json.loads(changed["NAN_TRAINING_OPTIONS"])
        options["dense_phase_sentinels"] = False
        changed["NAN_TRAINING_OPTIONS"] = json.dumps(options)
        for candidate in ({**environment, "HSTU_BWD_MAX_VGPR": "256"},
                          {**environment, "NAN_BACKWARD_TARGET": "_stu_layers.1."}, changed):
            with self.subTest(candidate=candidate), patch.dict(os.environ, candidate), \
                    patch.object(entry.importlib, "import_module", side_effect=AssertionError("runtime import")), \
                    self.assertRaisesRegex(ValueError, "scope provenance"):
                entry.diagnostic_worker(0, 1, 0, 1, "localhost", "29884", "test.gin", "streaming-train-eval")

    def test_main_validation_and_failure_restore_worker_and_argv(self):
        original_worker = lambda: None
        previous_argv = sys.argv
        trainer = SimpleNamespace(_main_func=original_worker)
        def run():
            self.assertIs(trainer._main_func, entry.diagnostic_worker)
            self.assertEqual(os.environ["HSTU_BWD_MAX_VGPR"], "0")
            raise RuntimeError("launch failed")
        trainer.main = run
        with patch.dict(os.environ, {}, clear=True), patch.object(entry.importlib, "import_module", return_value=trainer), \
                self.assertRaisesRegex(RuntimeError, "launch failed"):
            entry.main(["--directory", str(self.root / "new"), "--steps", "2"])
        self.assertIs(trainer._main_func, original_worker)
        self.assertIs(sys.argv, previous_argv)
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()), \
                patch.object(entry.importlib, "import_module", side_effect=AssertionError("trainer import")), \
                self.assertRaises(SystemExit):
            entry.main(["--directory", str(self.root), "--steps", "2"])


if __name__ == "__main__":
    unittest.main()
