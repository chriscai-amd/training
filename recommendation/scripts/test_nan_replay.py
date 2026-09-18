"""CPU-only regression checks for the replay evidence, not model numerics.

Run in the training image: python scripts/test_nan_replay.py
"""
import tempfile
import sys
from pathlib import Path
import unittest

import torch

from nan_replay_capture import (
    pack_small_state, restore_small_state, compare_small_state,
    nonfinite_small_state,
)
from nan_replay_state import collect_state, restore_controls, capture_rng, restore_rng
from nan_replay_storage import RollingTensorSnapshot, restore_snapshot, compare_snapshot
from replay_training_step import differences


class ReplayEvidenceTest(unittest.TestCase):
    def test_undo_recovers_writes_outside_tensor_views(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = torch.arange(2051, dtype=torch.float32)
            # Deliberately expose only part of storage; a wild store elsewhere
            # must still be reversible, including a final partial page.
            refs = {"visible": raw[21:800:3], "alias": raw[70:75]}
            with RollingTensorSnapshot(refs, Path(directory) / "rolling", chunk_bytes=1024, page_bytes=256) as snapshot:
                original = raw.clone()
                self.assertEqual(snapshot.nonfinite_floating_ranges({0: torch.float32}), [])
                raw[0] = float("nan")
                raw[-1] = -17
                snapshot.update(refs, boundary_id=2)
                self.assertEqual(snapshot.nonfinite_floating_ranges({0: torch.float32}), [{"storage_id": 0, "byte_offset": 0}])
                target = Path(directory) / "saved"
                snapshot.persist(target)
                current = raw.clone()
                raw.zero_()
                restore_snapshot(target, refs, boundary="previous", chunk_bytes=2048)
                self.assertTrue(torch.equal(raw, original))
                restore_snapshot(target, refs, boundary="current", chunk_bytes=512)
                self.assertEqual(differences(raw, current), [])
                self.assertTrue(compare_snapshot(target, refs)["equal"])
                raw[0] = 99
                snapshot.update(refs, boundary_id=3)
                self.assertEqual(snapshot.nonfinite_floating_ranges({0: torch.float32}), [])
                self.assertTrue(snapshot.compare_current(refs)["equal"])

    def test_small_dynamic_state_keeps_aliases_and_nan_bits(self):
        raw = torch.arange(100, dtype=torch.float32)
        source = {"a": raw[7:55:2], "b": raw[33:40], "c": torch.tensor(float("nan"))}
        saved = pack_small_state(source)
        dest = {"a": torch.empty(24), "b": torch.empty(7), "c": torch.tensor(0.)}
        restore_small_state(dest, saved)
        self.assertTrue(compare_small_state(dest, saved)["equal"])
        self.assertEqual(dest["a"].untyped_storage()._cdata, dest["b"].untyped_storage()._cdata)
        self.assertEqual(nonfinite_small_state(saved), ["c"])
        self.assertEqual(differences(saved, pack_small_state(dest)), [])

    def test_adam_and_rng_resume(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        sample = torch.ones(1, 3)
        model(sample).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        refs, controls = collect_state(model, optimizer)
        saved = pack_small_state(refs)
        rng = capture_rng()
        expected_random = torch.rand(10)
        expected_output = model(sample).detach().clone()
        model(sample).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        restore_controls(model, optimizer, controls)
        refs, _ = collect_state(model, optimizer)
        restore_small_state(refs, saved)
        restore_rng(rng)
        self.assertTrue(compare_small_state(refs, saved)["equal"])
        self.assertTrue(torch.equal(expected_output, model(sample)))
        self.assertTrue(torch.equal(expected_random, torch.rand(10)))


if __name__ == "__main__":
    program = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU tests unexpectedly initialized CUDA"
    sys.exit(not program.result.wasSuccessful())
