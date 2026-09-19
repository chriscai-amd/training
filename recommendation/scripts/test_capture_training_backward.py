"""CPU-only integration checks for the bounded production-loop wrapper."""

import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from capture_training_backward import configure_environment, diagnostic_worker, run_instrumented_loop
from nan_backward_boundaries import BoundaryAnomalyError, BoundaryProbeError
from nan_replay_state import capture_rng
from capture_training_backward import _rng_equal


class FakeProbe:
    instances = []

    def __init__(self, model, directory, **options):
        self.attempts = []
        self.closed = False
        self.options = options
        self.instances.append(self)

    def set_attempt(self, mode, repeat, step):
        self.attempts.append((mode, repeat, step))

    def close(self):
        self.closed = True


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value * 2


class TrainingBoundaryIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "capture"
        self.options = {"directory": str(self.directory), "steps": 3,
                        "capture_mode": "monitor_on_anomaly", "target": "*",
                        "operations": ["hstu_output_grad_mm", "triton_layer_norm_mul_dropout_bwd"],
                        "abs_threshold": 1e20, "dataset": "yambda-5b"}
        self.model = Model()
        self.metrics = types.SimpleNamespace(global_step={"train": 0})
        self.kwargs = {"model": self.model, "metric_logger": self.metrics, "rank": 0,
                       "resume_train_ts": None, "resume_cold_start": True}
        FakeProbe.instances.clear()

    def run_loop(self, loop, **extra):
        return run_instrumented_loop(loop, self.options, (), self.kwargs,
                                     probe_factory=extra.get("probe_factory", FakeProbe),
                                     context_factory=lambda: {"source_sha256": {}, "limits": ["wrong"]},
                                     archive_factory=lambda *args: {"test": True})

    def outcome(self):
        return json.loads((self.directory / "outcome.json").read_text())

    def test_direct_forward_step_attribution_bound_rng_and_restoration(self):
        initial = capture_rng()
        original_forward = self.model.forward

        def loop(**kwargs):
            self.assertEqual(kwargs["die_at_step"], 3)
            self.assertEqual(kwargs["start_ts"], 0)
            self.assertEqual(kwargs["eval_every_data_pct"], 0)
            self.assertEqual(kwargs["eval_every_n_windows"], 0)
            for _ in range(3):
                # Production invokes .forward directly, bypassing root hooks.
                self.assertEqual(kwargs["model"].forward(torch.tensor(2.)).item(), 4.)
                kwargs["metric_logger"].global_step["train"] += 1
            raise SystemExit(42)

        with self.assertRaises(SystemExit) as raised:
            self.run_loop(loop)
        self.assertEqual(raised.exception.code, 42)
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(self.model.forward, original_forward)
        self.assertNotIn("forward", self.model.__dict__)
        self.assertEqual(FakeProbe.instances[0].attempts,
                         [("training", 0, 1), ("training", 0, 2), ("training", 0, 3)])
        self.assertTrue(FakeProbe.instances[0].closed)
        self.assertTrue(_rng_equal(initial, capture_rng()))
        self.assertEqual(self.outcome()["status"], "bounded_complete")
        self.assertEqual(self.outcome()["completed_steps"], 3)
        context = json.loads((self.directory / "context.json").read_text())
        self.assertTrue(context["installation_rng_unchanged"])
        self.assertNotIn("wrong", context["limits"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_anomaly_metadata_saved_before_trainer_wraps_error(self):
        fault_path = self.directory / "fault.pt"

        def loop(**kwargs):
            kwargs["model"].forward(torch.tensor(1.))
            raise BoundaryAnomalyError(fault_path, "hstu_output_grad_mm", "layer1", "input")

        with self.assertRaises(BoundaryAnomalyError):
            self.run_loop(loop)
        self.assertEqual(self.outcome()["status"], "anomaly")
        self.assertEqual(self.outcome()["dump_path"], str(fault_path))
        self.assertEqual(self.outcome()["last_attempted_step"], 1)
        self.assertEqual(self.outcome()["completed_steps"], 0)
        self.assertTrue(FakeProbe.instances[0].closed)
        self.assertNotIn("forward", self.model.__dict__)

    def test_preserves_instance_forward_and_ordinary_error(self):
        custom_forward = lambda value: value + 1
        self.model.forward = custom_forward

        def loop(**kwargs):
            self.assertEqual(kwargs["model"].forward(torch.tensor(1.)).item(), 2.)
            raise RuntimeError("original failure")

        with self.assertRaisesRegex(RuntimeError, "original failure"):
            self.run_loop(loop)
        self.assertIs(self.model.forward, custom_forward)
        self.assertEqual(self.outcome()["status"], "error")
        self.assertIn("original failure", self.outcome()["traceback"])

    def test_installation_that_consumes_rng_fails_and_closes(self):
        class ConsumingProbe(FakeProbe):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                torch.rand(1)

        with self.assertRaisesRegex(BoundaryProbeError, "changed default RNG"):
            self.run_loop(lambda **kwargs: None, probe_factory=ConsumingProbe)
        self.assertTrue(FakeProbe.instances[0].closed)
        self.assertEqual(self.model.calls, 0)

    def test_unexpected_exit_and_duplicate_forward_do_not_claim_success(self):
        def loop(**kwargs):
            kwargs["model"].forward(torch.tensor(1.))
            kwargs["model"].forward(torch.tensor(1.))

        with self.assertRaisesRegex(BoundaryProbeError, "Unexpected root training forward"):
            self.run_loop(loop)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.outcome()["status"], "error")

    def test_premature_exit_42_is_failure(self):
        with self.assertRaises(SystemExit):
            self.run_loop(lambda **kwargs: (_ for _ in ()).throw(SystemExit(42)))
        self.assertEqual(self.outcome()["status"], "unexpected_exit")

    def test_resume_rejected_before_directory_or_probe(self):
        self.kwargs["resume_cold_start"] = False
        with self.assertRaisesRegex(ValueError, "resumed checkpoint"):
            self.run_loop(lambda **kwargs: None)
        self.assertFalse(self.directory.exists())
        self.assertFalse(FakeProbe.instances)

    def test_configuration_is_explicit_and_rejects_conflicts(self):
        environment = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "BATCH_SIZE": "1024"}
        configure_environment(self.options, environment)
        self.assertEqual(environment["HSTU_BWD_MAX_VGPR"], "256")
        self.assertEqual(environment["START_TS"], "0")
        self.assertEqual(environment["CKPT_PATH"], "")
        self.assertEqual(environment["DIE_AT_STEP"], "3")
        self.assertEqual(environment["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(environment["NAN_BACKWARD_TARGET"], "*")
        for key, value in (("START_TS", "150"), ("NAN_REPLAY_DIR", "/data/other"),
                           ("CKPT_PATH", "/data/checkpoint"), ("HSTU_BWD_MAX_VGPR", "0")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                configure_environment(self.options, {key: value})

    def test_spawn_worker_bootstraps_before_utils_and_restores_patch(self):
        order = []
        gin = types.ModuleType("gin")
        gin.parse_config_file = lambda *args, **kwargs: order.append("gin")
        bootstrap = types.ModuleType("generative_recommenders.dlrm_v4.train._env_bootstrap")
        bootstrap.apply_env_bootstrap = lambda: order.append("bootstrap")
        original = lambda **kwargs: None
        utils = types.SimpleNamespace(streaming_train_eval_loop=original)

        def train(*args):
            self.assertEqual(order, ["gin", "bootstrap", "utils", "trainer"])
            self.assertIsNot(utils.streaming_train_eval_loop, original)
            self.assertEqual(args, (0, 1, 0, 1, "localhost", "12345", "test.gin", "streaming-train-eval"))
            raise RuntimeError("worker original exception")

        trainer = types.SimpleNamespace(_main_func=train)

        def load(name):
            key = name.rsplit(".", 1)[-1]
            order.append("trainer" if key == "train_ranker" else key)
            return trainer if key == "train_ranker" else utils

        with patch.dict(os.environ, {"NAN_TRAINING_OPTIONS": json.dumps(self.options)}), \
             patch.dict("sys.modules", {"gin": gin, bootstrap.__name__: bootstrap}), \
             patch("capture_training_backward.importlib.import_module", side_effect=load), \
             self.assertRaisesRegex(RuntimeError, "worker original exception"):
            diagnostic_worker(0, 1, 0, 1, "localhost", "12345", "test.gin", "streaming-train-eval")
        self.assertIs(utils.streaming_train_eval_loop, original)


if __name__ == "__main__":
    unittest.main()
