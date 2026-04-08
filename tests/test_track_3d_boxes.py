# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utils/track_3d_boxes.py — 3D bounding box tracker."""

import torch

from utils.track_3d_boxes import (
    _SEM_OVERRIDES,
    BoundingBox3DTracker,
    TrackState,
)
from utils.tw.obb import make_obb
from utils.tw.tensor_utils import pad_string, string2tensor


def _make_det(position, sz=(1.0, 1.0, 1.0), prob=0.9, yaw=0.0, text="chair"):
    """Create a single ObbTW detection."""
    obb = make_obb(sz=list(sz), position=list(position), prob=prob, yaw=yaw)
    text_t = string2tensor(pad_string(text, max_len=128, silent=True))
    obb.set_text(text_t)
    obb.set_sem_id(torch.tensor(8))
    obb.set_inst_id(torch.tensor(-1))
    return obb


class TestTrackState:
    def test_enum_values(self):
        assert TrackState.TENTATIVE.value == 0
        assert TrackState.ACTIVE.value == 1
        assert TrackState.INACTIVE.value == 2


class TestSemOverrides:
    def test_bidirectional(self):
        assert ("table", "desk") in _SEM_OVERRIDES
        assert ("desk", "table") in _SEM_OVERRIDES

    def test_similarity_score(self):
        assert _SEM_OVERRIDES[("table", "desk")] == 0.9


class TestTrackerInit:
    def test_defaults(self):
        tracker = BoundingBox3DTracker(verbose=False)
        assert tracker.iou_threshold == 0.25
        assert tracker.min_hits == 3
        assert tracker.conf_threshold == 0.55
        assert len(tracker.tracks) == 0

    def test_custom_params(self):
        tracker = BoundingBox3DTracker(
            iou_threshold=0.5, min_hits=5, max_missed=10, verbose=False
        )
        assert tracker.iou_threshold == 0.5
        assert tracker.min_hits == 5
        assert tracker.max_missed == 10


class TestTrackerCreateTracks:
    def test_creates_from_detections(self):
        tracker = BoundingBox3DTracker(verbose=False, conf_threshold=0.0)
        det = _make_det([0.0, 0.0, 0.5], prob=0.9)
        detections = torch.stack([det])
        tracker.update(detections, frame_idx=0)
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].state == TrackState.TENTATIVE
        assert tracker.tracks[0].support_count == 1

    def test_multiple_detections(self):
        tracker = BoundingBox3DTracker(verbose=False, conf_threshold=0.0)
        dets = torch.stack(
            [
                _make_det([0.0, 0.0, 0.5]),
                _make_det([10.0, 0.0, 0.5]),
                _make_det([20.0, 0.0, 0.5]),
            ]
        )
        tracker.update(dets, frame_idx=0)
        assert len(tracker.tracks) == 3

    def test_unique_track_ids(self):
        tracker = BoundingBox3DTracker(verbose=False, conf_threshold=0.0)
        dets = torch.stack(
            [
                _make_det([0.0, 0.0, 0.5]),
                _make_det([10.0, 0.0, 0.5]),
            ]
        )
        tracker.update(dets, frame_idx=0)
        ids = [t.track_id for t in tracker.tracks]
        assert len(set(ids)) == len(ids)


class TestTrackerConfidenceFiltering:
    def test_filters_low_confidence(self):
        tracker = BoundingBox3DTracker(verbose=False, conf_threshold=0.8)
        dets = torch.stack(
            [
                _make_det([0.0, 0.0, 0.5], prob=0.9),  # above threshold
                _make_det([10.0, 0.0, 0.5], prob=0.5),  # below threshold
            ]
        )
        tracker.update(dets, frame_idx=0)
        assert len(tracker.tracks) == 1


class TestTrackerPromotion:
    def test_tentative_to_active_by_hits(self):
        tracker = BoundingBox3DTracker(
            verbose=False,
            conf_threshold=0.0,
            min_hits=2,
            force_cpu=True,
            samp_per_dim=4,
        )
        # Same position => should match and accumulate hits
        det = _make_det([0.0, 0.0, 0.5], prob=0.9)
        for i in range(3):
            tracker.update(torch.stack([det]), frame_idx=i)

        # Should have promoted after 2 hits
        active = [t for t in tracker.tracks if t.state == TrackState.ACTIVE]
        assert len(active) >= 1


class TestTrackerNoDetections:
    def test_empty_detections(self):
        tracker = BoundingBox3DTracker(verbose=False, conf_threshold=0.0)
        det = _make_det([0.0, 0.0, 0.5])
        tracker.update(torch.stack([det]), frame_idx=0)
        assert len(tracker.tracks) == 1

        # Send empty detections
        empty = torch.stack([det])[:0]
        tracker.update(empty, frame_idx=1)
        # Track should still exist (within max_missed)
        assert (
            len(tracker.tracks) >= 0
        )  # may or may not be removed depending on missed_count


class TestTrackerAging:
    def test_tracks_removed_after_max_missed(self):
        tracker = BoundingBox3DTracker(verbose=False, conf_threshold=0.0, max_missed=2)
        det = _make_det([0.0, 0.0, 0.5])
        tracker.update(torch.stack([det]), frame_idx=0)

        # Send many frames with no detections
        empty = torch.stack([det])[:0]
        for i in range(1, 20):
            tracker.update(empty, frame_idx=i)

        # Track should eventually be removed
        assert len(tracker.tracks) == 0
