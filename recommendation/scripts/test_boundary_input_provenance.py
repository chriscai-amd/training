"""CPU checks that fault-time input snapshots cannot masquerade as pristine."""

import unittest
from unittest.mock import patch

import torch

import replay_backward_boundary as preprocess
import replay_hstu_output_boundary as output


def fixture(consumer):
    if consumer is preprocess:
        return {"operation": "triton_addmm_bwd", "args": (), "outputs": None,
                "kwargs": {"x": torch.ones(1, 2), "w": torch.ones(2, 3),
                           "dz": torch.ones(1, 3), "is_y_1d": True}, "stage": "input"}
    return {"operation": output.MM, "args": (), "outputs": None,
            "kwargs": {"dout": torch.ones(1, 2), "output_weight": torch.ones(3, 2)},
            "stage": "input"}


class InputProvenanceTest(unittest.TestCase):
    def test_legacy_and_certified_pre_call_monitor_inputs_remain_supported(self):
        for consumer in (preprocess, output):
            with self.subTest(consumer=consumer.__name__):
                payload = fixture(consumer)
                self.assertTrue(consumer.bind_inputs(payload))
                payload.update(format="training_backward_monitor_v1", input_snapshot_pristine=True)
                self.assertTrue(consumer.bind_inputs(payload))
                report = consumer.analyze(payload, rows="all", max_rows=0,
                                          chunk_bytes=4096, row_chunk=1)
                self.assertTrue(report["references"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_post_call_or_uncertified_monitor_inputs_are_rejected(self):
        cases = (
            {"input_snapshot_pristine": False},
            {"event": {"input_snapshot_pristine": False}},
            {"format": "training_backward_monitor_v1", "stage": "output"},
            {"format": "training_backward_monitor_v1", "stage": "output",
             "input_snapshot_pristine": True},
            {"format": "training_backward_monitor_v1", "stage": "input"},
        )
        for consumer in (preprocess, output):
            for case in cases:
                with self.subTest(consumer=consumer.__name__, case=case):
                    payload = {**fixture(consumer), **case}
                    with self.assertRaisesRegex(ValueError, "Post-call input snapshots"):
                        consumer.bind_inputs(payload)
                    with self.assertRaisesRegex(ValueError, "Post-call input snapshots"):
                        consumer.analyze(payload)

    def test_replay_rejects_post_call_snapshot_before_importing_gpu_operations(self):
        for consumer in (preprocess, output):
            with self.subTest(consumer=consumer.__name__):
                payload = {**fixture(consumer), "input_snapshot_pristine": False}
                with patch.object(consumer.importlib, "import_module",
                                  side_effect=AssertionError("GPU operation imported")):
                    with self.assertRaisesRegex(ValueError, "Post-call input snapshots"):
                        consumer.replay_gpu(payload)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
