"""CPU integration checks for the streaming diagnostic ordering.

Execute the production step statements with stand-ins for model/distributed
services, so these checks need neither Torch nor a GPU. Tensor/capture semantics
are covered separately by nan_tripwire_test.py and scripts/test_nan_replay.py.
"""

import ast
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
TRAINER = ROOT / "generative_recommenders/dlrm_v4/train/utils.py"


def _streaming_step_code():
    tree = ast.parse(TRAINER.read_text(), filename=str(TRAINER))
    window = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_run_train_window")
    loop = next(node for node in window.body if isinstance(node, ast.While))
    start = next(i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "_apply_lr_warmup")
    end = next(i for i, node in enumerate(loop.body)
               if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
               and node.target.id == "train_batch_idx")
    return compile(ast.Module(body=loop.body[start:end + 1], type_ignores=[]),
                   str(TRAINER), "exec")


class _TripwireStop(RuntimeError):
    pass


class _StepHarness:
    """Only external services are mocked; branch/order logic is production code."""

    def __init__(self, *, b0=False, defer=0, frequency=1, replay_stop=None,
                 tripwire_stop=None, no_grad=False, source=TRAINER):
        self.events = []
        self.import_paths = []
        self.replay_stop = replay_stop
        self.tripwire_stop = tripwire_stop
        self.parameter = SimpleNamespace(grad=None)
        events = self.events

        class Loss:
            def __radd__(loss, value):
                if value != 0:
                    raise AssertionError("unexpected loss addition")
                return loss

            def backward(loss):
                events.append("backward")
                if not no_grad:
                    self.parameter.grad = "dense-gradient"

        class Model:
            def forward(model, uih, candidates):
                events.append("forward")
                return None, None, {"task": Loss()}, "preds", "labels", "weights"

            def __call__(model, *args):
                events.append("module_call")
                return model.forward(*args)

            def parameters(model):
                return [self.parameter]

        class Candidates:
            def lengths(candidates):
                return candidates

            def view(candidates, *shape):
                return [[1]]

            def keys(candidates):
                return ["item"]

        self.model = Model()
        self.metric = SimpleNamespace(global_step={"train": 0})

        def update(**kwargs):
            events.append("metrics_update")
            self.metric.global_step["train"] += 1

        self.metric.update = update
        self.metric.maybe_log_mlperf_train_loss = lambda *args, **kw: events.append("mlperf")
        self.metric.compute_and_log = lambda **kw: events.append("metrics_log")
        self.optimizer = SimpleNamespace(
            zero_grad=lambda: (events.append("zero_grad"), setattr(self.parameter, "grad", None)),
            step=lambda: events.append("optimizer"),
        )
        tripwire = SimpleNamespace(
            active=True,
            begin=lambda step, **kw: events.append(("b0_begin", step)),
            watch=lambda name, value: events.append(("b0_watch", name)),
            check=self.check,
            end=lambda: events.append("b0_end"),
        ) if b0 else None
        self.namespace = dict(
            __file__=str(source), os=os, model=self.model, optimizer=self.optimizer,
            sample=SimpleNamespace(uih_features_kjt="uih", candidates_features_kjt=Candidates(),
                                   to=lambda device: events.append("sample_to")),
            device="cpu", metric_logger=self.metric,
            _apply_lr_warmup=lambda step: events.append(("warmup", step)),
            tripwire=tripwire, _tripwire_defer=defer,
            _dense_diagnostic_params={"weight": self.parameter},
            train_ts=7, train_batch_idx=0, grad_clip_norm=1.0,
            metric_log_frequency=frequency, mlperf_loss_every=frequency,
            torch=SimpleNamespace(nn=SimpleNamespace(utils=SimpleNamespace(
                clip_grad_norm_=lambda *args, **kw: events.append("clip")))),
        )
        self.modules = {name: ModuleType(name) for name in
                        ("nan_replay_capture", "nan_module_probe", "nan_capture")}
        self.modules["nan_replay_capture"].before_step = self.before_step
        self.modules["nan_module_probe"].install = self.install_probe
        self.modules["nan_capture"].step_hook = self.legacy_capture
        self.probe = SimpleNamespace(
            report_if_bad=lambda: events.append("probe_poll"),
            set_step=lambda step: events.append(("probe_step", step)),
            pump=lambda: events.append("probe_pump"),
            pause=lambda: events.append("probe_pause"),
        )

    def before_step(self, model, optimizer, step, sample, grad_clip_norm):
        self.events.append(("a0_before", step))
        self.import_paths.append(sys.path[0])
        if self.replay_stop == "before":
            raise OSError("capture storage unavailable")
        return SimpleNamespace(after_forward=lambda *args: self.replay_boundary("forward"),
                               after_backward=lambda: self.replay_boundary("backward"))

    def replay_boundary(self, phase):
        self.events.append(("a0", phase))
        if self.replay_stop == phase:
            raise SystemExit(86)

    def install_probe(self, model):
        self.events.append("probe_install")
        self.import_paths.append(sys.path[0])
        return self.probe

    def legacy_capture(self, step, *args):
        self.events.append(("legacy_capture", step))
        self.import_paths.append(sys.path[0])

    def check(self, phase):
        self.events.append(("b0_check", phase))
        if self.tripwire_stop == phase:
            raise _TripwireStop(phase)

    def run(self, environment=None, steps=1):
        with patch.dict(os.environ, environment or {}, clear=True), \
             patch.dict(sys.modules, self.modules), patch.object(sys, "path", list(sys.path)):
            for _ in range(steps):
                exec(_streaming_step_code(), self.namespace)


class StreamingDiagnosticsCompatibilityTest(unittest.TestCase):
    def assert_before(self, events, earlier, later):
        self.assertLess(events.index(earlier), events.index(later), events)

    def test_diagnostics_off_performs_one_training_update(self):
        run = _StepHarness()
        run.run()
        for action in ("forward", "backward", "zero_grad", "optimizer", "metrics_update"):
            self.assertEqual(run.events.count(action), 1)
        self.assertEqual(run.metric.global_step["train"], 1)
        self.assertEqual(run.import_paths, [])
        self.assertNotIn("module_call", run.events)
        self.assert_before(run.events, "backward", "clip")
        self.assert_before(run.events, "clip", "optimizer")

    def test_a0_replay_alone_snapshots_the_warmed_pre_forward_state(self):
        run = _StepHarness()
        run.run({"NAN_REPLAY_DIR": "/unused/capture"})
        for action in (("warmup", 0), "zero_grad", "sample_to"):
            self.assert_before(run.events, action, ("a0_before", 1))
        self.assert_before(run.events, ("a0_before", 1), "forward")
        self.assert_before(run.events, "forward", ("a0", "forward"))
        self.assert_before(run.events, ("a0", "forward"), "backward")
        self.assert_before(run.events, ("a0", "backward"), "clip")
        self.assertFalse(any(isinstance(event, tuple) and event[0].startswith("b0")
                             for event in run.events))

    def test_b0_alone_checks_gradients_before_clipping_and_parameters_after_update(self):
        run = _StepHarness(b0=True)
        run.run({"NAN_TRIPWIRE_CHECK_FORWARD": "1"})
        for earlier, later in (("forward", ("b0_check", "forward")),
                               (("b0_check", "forward"), "backward"),
                               ("backward", ("b0_check", "backward")),
                               (("b0_check", "backward"), "clip"),
                               ("optimizer", ("b0_check", "optimizer")),
                               (("b0_check", "optimizer"), "metrics_update")):
            self.assert_before(run.events, earlier, later)
        self.assertEqual(run.import_paths, [])

    def test_combined_nonterminating_hooks_complete_one_step_in_priority_order(self):
        run = _StepHarness(b0=True)
        run.run({"NAN_REPLAY_DIR": "/unused", "NAN_CAPTURE_STEP": "1",
                 "NAN_TRIPWIRE_CHECK_FORWARD": "1"})
        for earlier, later in ((("a0", "forward"), ("legacy_capture", 1)),
                               (("legacy_capture", 1), ("b0_check", "forward")),
                               (("a0", "backward"), ("b0_check", "backward"))):
            self.assert_before(run.events, earlier, later)
        self.assertEqual(run.events.count("optimizer"), 1)
        self.assertEqual(run.metric.global_step["train"], 1)

    def test_full_state_capture_terminates_before_b0_at_each_completed_boundary(self):
        for phase in ("forward", "backward"):
            with self.subTest(phase=phase):
                run = _StepHarness(b0=True, replay_stop=phase, tripwire_stop=phase)
                with self.assertRaises(SystemExit) as error:
                    run.run({"NAN_REPLAY_DIR": "/unused", "NAN_TRIPWIRE_CHECK_FORWARD": "1"})
                self.assertEqual(error.exception.code, 86)
                self.assertNotIn(("b0_check", phase), run.events)
                self.assertNotIn("clip", run.events)
                self.assertNotIn("optimizer", run.events)

    def test_b0_failure_stops_updates_when_full_state_capture_does_not_trigger(self):
        for phase in ("forward", "backward"):
            with self.subTest(phase=phase):
                run = _StepHarness(b0=True, tripwire_stop=phase)
                with self.assertRaises(_TripwireStop):
                    run.run({"NAN_REPLAY_DIR": "/unused", "NAN_TRIPWIRE_CHECK_FORWARD": "1"})
                self.assert_before(run.events, ("a0", phase), ("b0_check", phase))
                self.assertNotIn("clip", run.events)
                self.assertNotIn("optimizer", run.events)

    def test_replay_setup_errors_fail_closed_before_forward(self):
        run = _StepHarness(b0=True, replay_stop="before")
        with self.assertRaises(OSError):
            run.run({"NAN_REPLAY_DIR": "/unused"})
        self.assertNotIn("forward", run.events)
        self.assertNotIn("optimizer", run.events)

    def test_missing_dense_gradients_stop_before_optimizer(self):
        run = _StepHarness(b0=True, no_grad=True)
        with self.assertRaisesRegex(RuntimeError, "no dense gradients"):
            run.run()
        self.assertNotIn("clip", run.events)
        self.assertNotIn("optimizer", run.events)

    def test_deferred_b0_checks_only_at_global_metric_boundary_probe_polls_every_step(self):
        run = _StepHarness(b0=True, defer=3, frequency=3)
        run.run({"NAN_MODULE_PROBE": "1", "NAN_TRIPWIRE_CHECK_FORWARD": "1"}, steps=4)
        self.assertEqual([event for event in run.events if isinstance(event, tuple)
                          and event[0] == "b0_check"], [("b0_check", "metric_boundary")])
        self.assertEqual(run.events.count("b0_end"), 1)
        self.assertEqual(run.events.count("metrics_log"), 1)
        self.assertEqual(run.events.count("probe_poll"), 8)
        self.assertEqual(run.events.count("probe_pump"), 4)
        self.assertEqual(run.events.count("probe_pause"), 4)
        self.assertEqual(run.events.count("module_call"), 4)
        self.assertEqual(run.events.count("optimizer"), 4)
        self.assert_before(run.events, "metrics_log", ("b0_check", "metric_boundary"))

    def test_a0_hook_imports_follow_checkout_location_on_either_host(self):
        for source in (TRAINER, Path("/tmp/a0 host/recommendation/generative_recommenders/dlrm_v4/train/utils.py")):
            for environment in ({"NAN_REPLAY_DIR": "/unused"}, {"NAN_MODULE_PROBE": "1"},
                                {"NAN_CAPTURE_STEP": "1"}):
                with self.subTest(source=source, environment=environment):
                    run = _StepHarness(source=source)
                    run.run(environment)
                    self.assertEqual(run.import_paths,
                                     [str(source.resolve().parents[3] / "scripts")])


class DiagnosticEnvironmentCompatibilityTest(unittest.TestCase):
    def test_legacy_enable_flag_does_not_enable_b0_or_override_its_step_range(self):
        source = TRAINER.with_name("nan_tripwire.py")
        tree = ast.parse(source.read_text(), filename=str(source))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "install_from_env")
        future = ast.parse("from __future__ import annotations").body
        namespace = {"os": os}
        created = []

        class FakeTripwire:
            def __init__(self, directory, **kwargs):
                self.directory, self.options, self.installed = directory, kwargs, False
                created.append(self)

            def install(self):
                self.installed = True

        namespace["NaNTripwire"] = FakeTripwire
        exec(compile(ast.Module(body=future + [function], type_ignores=[]), str(source), "exec"), namespace)
        with patch.dict(os.environ, {"NAN_TRIPWIRE": "1", "NAN_TRIPWIRE_ARM_STEP": "9"}, clear=True):
            self.assertIsNone(namespace["install_from_env"]())
            self.assertEqual(created, [])
            os.environ["NAN_TRIPWIRE_DIR"] = "/unused/b0"
            tripwire = namespace["install_from_env"]()
        self.assertTrue(tripwire.installed)
        self.assertEqual(tripwire.directory, "/unused/b0")
        self.assertEqual(tripwire.options["start_step"], 0)
        self.assertIsNone(tripwire.options["end_step"])


ATTENTION_SOURCE = ROOT / "generative_recommenders/ops/triton/triton_hstu_attention.py"


class _FakeConfig:
    def __init__(self, kwargs, **options):
        self.kwargs = kwargs
        self.options = options


class AttentionPinnedConfigCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ATTENTION_SOURCE.read_text()
        tree = ast.parse(source, filename=str(ATTENTION_SOURCE))
        matches = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_bw_pinned_configs"
        ]
        if len(matches) != 1:
            raise AssertionError("Expected exactly one backward pinned-config function")
        cls.function_code = compile(
            ast.Module(body=matches, type_ignores=[]),
            str(ATTENTION_SOURCE),
            "exec",
        )

    def _evaluate(self, *, arch="gfx1250", version="3.8.0+git7ff97e31",
                  hip=True, env=None, properties_error=None):
        def properties(index):
            self.assertEqual(index, 0)
            if properties_error is not None:
                raise properties_error
            return SimpleNamespace(gcnArchName=arch)

        pre_hook = object()
        fallback = object()
        namespace = {
            "List": list,
            "os": SimpleNamespace(environ={} if env is None else dict(env)),
            "torch": SimpleNamespace(
                version=SimpleNamespace(hip="7.1" if hip else None),
                cuda=SimpleNamespace(get_device_properties=properties),
            ),
            "triton": SimpleNamespace(__version__=version, Config=_FakeConfig),
            "_bwd_pre_hook": pre_hook,
            "_get_bw_configs": lambda: fallback,
        }
        exec(self.function_code, namespace)
        return namespace["_get_bw_pinned_configs"](), pre_hook, fallback

    def _assert_pin(self, result, pre_hook, *, block_n, max_vgpr=None):
        self.assertEqual(len(result), 1)
        expected = {
            "BLOCK_M": 32,
            "BLOCK_N": block_n,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": 0,
            "SEQUENCE_PARALLEL": False,
            "UNROLL": 1,
        }
        if max_vgpr is not None:
            expected["llvm_fn_attrs"] = f"amdgpu-num-vgpr={max_vgpr}"
        self.assertEqual(result[0].kwargs, expected)
        self.assertEqual(result[0].options, {
            "num_stages": 1, "num_warps": 4, "pre_hook": pre_hook,
        })

    def test_gfx1250_unset_preserves_uncapped_default_and_tile(self):
        result, hook, _ = self._evaluate()
        self._assert_pin(result, hook, block_n=64)

    def test_gfx1250_explicit_zero_preserves_uncapped_default(self):
        result, hook, _ = self._evaluate(env={"HSTU_BWD_MAX_VGPR": "0"})
        self._assert_pin(result, hook, block_n=64)

    def test_gfx1250_explicit_256_applies_register_limit(self):
        result, hook, _ = self._evaluate(
            arch="gfx1250:sramecc+:xnack-",
            env={"HSTU_BWD_MAX_VGPR": "256"},
        )
        self._assert_pin(result, hook, block_n=64, max_vgpr=256)

    def test_gfx1250_negative_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HSTU_BWD_MAX_VGPR must be nonnegative"):
            self._evaluate(env={"HSTU_BWD_MAX_VGPR": "-1"})

    def test_gfx1250_noninteger_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            self._evaluate(env={"HSTU_BWD_MAX_VGPR": "invalid"})

    def test_gfx950_ignores_register_override_and_preserves_pin(self):
        for override in ("256", "-1", "invalid"):
            with self.subTest(override=override):
                result, hook, _ = self._evaluate(
                    arch="gfx950", env={"HSTU_BWD_MAX_VGPR": override},
                )
                self._assert_pin(result, hook, block_n=128)

    def test_old_triton_ignores_register_override_and_preserves_pin(self):
        for override in ("256", "invalid"):
            with self.subTest(override=override):
                result, hook, _ = self._evaluate(
                    version="3.7.0", env={"HSTU_BWD_MAX_VGPR": override},
                )
                self._assert_pin(result, hook, block_n=128)

    def test_block_n_override_remains_independent(self):
        result, hook, _ = self._evaluate(env={
            "HSTU_BWD_BLOCK_N": "128", "HSTU_BWD_MAX_VGPR": "256",
        })
        self._assert_pin(result, hook, block_n=128, max_vgpr=256)

    def test_nonhip_uses_full_config_fallback(self):
        result, _, fallback = self._evaluate(
            hip=False, env={"HSTU_BWD_MAX_VGPR": "invalid"},
        )
        self.assertIs(result, fallback)

    def test_failed_device_lookup_preserves_uncapped_baseline(self):
        result, hook, _ = self._evaluate(
            properties_error=RuntimeError("no device"),
            env={"HSTU_BWD_MAX_VGPR": "invalid"},
        )
        self._assert_pin(result, hook, block_n=128)


if __name__ == "__main__":
    unittest.main()
