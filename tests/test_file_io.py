# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for OBB CSV read/write round-trips in file_io.py."""

import os
import tempfile
import unittest

import numpy as np
import torch

from utils.file_io import ObbCsvWriter2, load_bb2d_csv, read_obb_csv, save_bb2d_csv
from utils.tw.obb import make_obb
from utils.tw.tensor_utils import pad_string, string2tensor


def _make_test_obb(
    position, sz=(1.0, 1.0, 1.0), yaw=0.0, prob=0.9, text="chair", sem_id=5, inst_id=-1
):
    """Create an ObbTW with text and semantic label set."""
    obb = make_obb(sz=list(sz), position=list(position), prob=prob, yaw=yaw)
    text_tensor = string2tensor(pad_string(text, max_len=128))
    obb.set_text(text_tensor)
    obb.set_sem_id(torch.tensor(sem_id))
    obb.set_inst_id(torch.tensor(inst_id))
    return obb


class TestObbCsvRoundTrip(unittest.TestCase):
    """Test that OBB CSV write -> read preserves data."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "test_obbs.csv")

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_single_obb_roundtrip(self):
        """Write one OBB and read it back, verify fields match."""
        obb = _make_test_obb(
            [1.0, 2.0, 3.0], sz=(0.5, 1.0, 1.5), prob=0.85, text="table"
        )
        obbs = torch.stack([obb])

        writer = ObbCsvWriter2(self.csv_path)
        writer.write(obbs, timestamps_ns=1000)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        self.assertIn(1000, timed_obbs)
        loaded = timed_obbs[1000]
        self.assertEqual(loaded.shape[0], 1)

        # Check position
        orig_pos = obb.T_world_object.t
        loaded_pos = loaded[0].T_world_object.t
        self.assertTrue(torch.allclose(orig_pos, loaded_pos, atol=1e-4))

        # Check size (diagonal = [sx, sy, sz])
        orig_diag = obb.bb3_diagonal
        loaded_diag = loaded[0].bb3_diagonal
        self.assertTrue(torch.allclose(orig_diag, loaded_diag, atol=1e-4))

        # Check probability
        self.assertAlmostEqual(obb.prob.item(), loaded[0].prob.item(), places=4)

    def test_multiple_obbs_same_timestamp(self):
        """Write multiple OBBs at one timestamp."""
        obbs = [
            _make_test_obb([0.0, 0.0, 0.5], text="chair", sem_id=1),
            _make_test_obb([3.0, 0.0, 0.5], text="table", sem_id=2),
            _make_test_obb([6.0, 0.0, 0.5], text="lamp", sem_id=3),
        ]
        stacked = torch.stack(obbs)

        writer = ObbCsvWriter2(self.csv_path)
        writer.write(stacked, timestamps_ns=5000)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        self.assertEqual(len(timed_obbs), 1)
        self.assertEqual(timed_obbs[5000].shape[0], 3)

    def test_multiple_timestamps(self):
        """Write OBBs at different timestamps and verify grouping."""
        writer = ObbCsvWriter2(self.csv_path)

        obb1 = torch.stack([_make_test_obb([0.0, 0.0, 0.5], text="chair")])
        obb2 = torch.stack(
            [
                _make_test_obb([1.0, 0.0, 0.5], text="table"),
                _make_test_obb([2.0, 0.0, 0.5], text="lamp"),
            ]
        )
        writer.write(obb1, timestamps_ns=100)
        writer.write(obb2, timestamps_ns=200)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        self.assertEqual(len(timed_obbs), 2)
        self.assertEqual(timed_obbs[100].shape[0], 1)
        self.assertEqual(timed_obbs[200].shape[0], 2)

    def test_rotation_roundtrip(self):
        """Verify rotation (quaternion) survives the CSV round-trip."""
        import math

        yaw = math.pi / 3
        obb = _make_test_obb([1.0, 2.0, 0.5], yaw=yaw, text="sofa")
        stacked = torch.stack([obb])

        writer = ObbCsvWriter2(self.csv_path)
        writer.write(stacked, timestamps_ns=0)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        loaded = timed_obbs[0][0]

        # Compare rotation matrices
        orig_R = obb.T_world_object.R
        loaded_R = loaded.T_world_object.R
        self.assertTrue(torch.allclose(orig_R, loaded_R, atol=1e-4))

    def test_text_label_roundtrip(self):
        """Verify text labels survive the CSV round-trip."""
        labels = ["dining chair", "coffee table", "floor lamp"]
        obbs = [
            _make_test_obb([i * 3.0, 0.0, 0.5], text=label)
            for i, label in enumerate(labels)
        ]
        stacked = torch.stack(obbs)

        writer = ObbCsvWriter2(self.csv_path)
        writer.write(stacked, timestamps_ns=0)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        loaded = timed_obbs[0]
        loaded_labels = loaded.text_string()
        for orig, loaded_label in zip(labels, loaded_labels):
            self.assertEqual(orig, loaded_label)

    def test_empty_write(self):
        """Writing zero OBBs should not crash and produce a readable file."""
        obb = _make_test_obb([0.0, 0.0, 0.5])
        empty = torch.stack([obb])[:0]

        writer = ObbCsvWriter2(self.csv_path)
        writer.write(empty, timestamps_ns=0)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        self.assertEqual(len(timed_obbs), 0)

    def test_empty_file_read(self):
        """Reading a header-only CSV should return empty dict."""
        writer = ObbCsvWriter2(self.csv_path)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        self.assertEqual(len(timed_obbs), 0)

    def test_append_mode(self):
        """Appending to existing CSV should accumulate rows."""
        obb1 = torch.stack([_make_test_obb([0.0, 0.0, 0.5])])

        writer = ObbCsvWriter2(self.csv_path)
        writer.write(obb1, timestamps_ns=100)
        writer.close()

        obb2 = torch.stack([_make_test_obb([5.0, 0.0, 0.5])])
        writer2 = ObbCsvWriter2(self.csv_path, append=True)
        writer2.write(obb2, timestamps_ns=200)
        writer2.close()

        timed_obbs = read_obb_csv(self.csv_path)
        self.assertEqual(len(timed_obbs), 2)
        self.assertIn(100, timed_obbs)
        self.assertIn(200, timed_obbs)

    def test_sem_id_to_name_fallback(self):
        """When text is empty, sem_id_to_name should be used."""
        obb = _make_test_obb([0.0, 0.0, 0.5], text="", sem_id=42)
        stacked = torch.stack([obb])

        sem_map = {42: "bookshelf"}
        writer = ObbCsvWriter2(self.csv_path)
        writer.write(stacked, timestamps_ns=0, sem_id_to_name=sem_map)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        loaded = timed_obbs[0]
        self.assertEqual(loaded.text_string()[0], "bookshelf")

    def test_large_batch(self):
        """Write and read back a larger batch of OBBs."""
        num = 50
        obbs = [
            _make_test_obb(
                [float(i), float(i % 5), 0.5],
                sz=(0.5 + i * 0.01, 1.0, 1.5),
                prob=0.5 + i * 0.01,
                text=f"obj_{i}",
                sem_id=i % 10,
            )
            for i in range(num)
        ]
        stacked = torch.stack(obbs)

        writer = ObbCsvWriter2(self.csv_path)
        writer.write(stacked, timestamps_ns=999)
        writer.close()

        timed_obbs = read_obb_csv(self.csv_path)
        self.assertEqual(timed_obbs[999].shape[0], num)

        # Spot-check a few positions
        for i in [0, 10, 49]:
            orig_pos = obbs[i].T_world_object.t
            loaded_pos = timed_obbs[999][i].T_world_object.t
            self.assertTrue(torch.allclose(orig_pos, loaded_pos, atol=1e-3))


class TestBb2dCsvRoundTrip(unittest.TestCase):
    """Test 2D bounding box CSV write -> read round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "test_bb2d.csv")

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_basic_roundtrip(self):
        """Write 2D detections and read them back."""
        bb2d = np.array([[10.0, 20.0, 50.0, 60.0], [100.0, 200.0, 300.0, 400.0]])
        scores = np.array([0.95, 0.80])
        labels = ["chair", "table"]

        save_bb2d_csv(
            self.csv_path,
            frame_id=0,
            bb2d=bb2d,
            scores=scores,
            labels=labels,
            append=False,
            time_ns=1000,
            img_width=640,
            img_height=480,
        )

        groups = load_bb2d_csv(self.csv_path)
        self.assertIn(1000, groups)
        g = groups[1000]
        np.testing.assert_allclose(g["bb2d"], bb2d, atol=0.01)
        np.testing.assert_allclose(g["scores"], scores, atol=1e-5)
        self.assertEqual(g["labels"], labels)
        self.assertEqual(g["img_width"], 640)
        self.assertEqual(g["img_height"], 480)

    def test_torch_tensor_input(self):
        """Verify torch tensor inputs are handled correctly."""
        bb2d = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
        scores = torch.tensor([0.9])
        labels = ["lamp"]

        save_bb2d_csv(
            self.csv_path,
            frame_id=0,
            bb2d=bb2d,
            scores=scores,
            labels=labels,
            append=False,
            time_ns=500,
        )

        groups = load_bb2d_csv(self.csv_path)
        self.assertIn(500, groups)
        self.assertEqual(groups[500]["bb2d"].shape, (1, 4))

    def test_multiple_frames_append(self):
        """Appending multiple frames should group by time_ns."""
        bb1 = np.array([[10.0, 20.0, 30.0, 40.0]])
        bb2 = np.array([[50.0, 60.0, 70.0, 80.0]])

        save_bb2d_csv(
            self.csv_path,
            frame_id=0,
            bb2d=bb1,
            scores=[0.9],
            labels=["a"],
            append=False,
            time_ns=100,
        )
        save_bb2d_csv(
            self.csv_path,
            frame_id=1,
            bb2d=bb2,
            scores=[0.8],
            labels=["b"],
            append=True,
            time_ns=200,
        )

        groups = load_bb2d_csv(self.csv_path)
        self.assertEqual(len(groups), 2)
        self.assertIn(100, groups)
        self.assertIn(200, groups)


if __name__ == "__main__":
    unittest.main()
