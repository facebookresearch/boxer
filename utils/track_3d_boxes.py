#!/usr/bin/env python3
# pyre-ignore-all-errors

"""
Online 3D Bounding Box Tracker

Maintains a set of tracked instances and matches them against each frame's
detections using a VxN IoU matrix (V visible tracks x N detections), where
V << total tracks M. Invisible tracks (behind walls, out of FOV) are skipped
in the IoU computation and sent directly to aging. Uses Hungarian assignment
for optimal matching.

Algorithm per frame:
1. Filter detections by confidence threshold
2. Partition tracks into visible and invisible using cached last_visible flag
3. Compute VxN IoU between visible tracks and new detections
4. Hungarian assignment with IoU threshold gating
5. Update matched tracks (confidence-weighted averaging)
6. Create new tracks from unmatched detections
7. Age and remove stale tracks (unmatched visible + all invisible)
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
from tw.camera import CameraTW
from utils.file_io import ObbCsvWriter2, read_obb_csv
from utils.fuse_3d_boxes import (
    align_boxes_r90,
    angular_distance,
    weighted_yaw_mean,
)
from tw.obb import ObbTW, iou_mc7
from tw.pose import PoseTW, rotation_from_euler
from tw.tensor_utils import (
    pad_string,
    string2tensor,
    tensor2string,
    unpad_string,
)


class TrackState(Enum):
    TENTATIVE = 0  # New, not yet confirmed (support_count < min_hits)
    ACTIVE = 1  # Confirmed track
    INACTIVE = 2  # Lost (not matched for some frames)


# Pairs of semantic labels that should be treated as near-identical for merging.
# Format: (text_a, text_b, similarity_score). Matching is case-insensitive.
_SEM_OVERRIDE_PAIRS = [
    ("table", "desk", 0.9),
    ("coffee table", "desk", 0.9),
]

# Build bidirectional lookup dict (computed once at import time)
_SEM_OVERRIDES: dict[tuple[str, str], float] = {}
for _a, _b, _s in _SEM_OVERRIDE_PAIRS:
    _SEM_OVERRIDES[(_a.lower(), _b.lower())] = _s
    _SEM_OVERRIDES[(_b.lower(), _a.lower())] = _s


@dataclass
class TrackedInstance:
    """A tracked 3D object instance."""

    obb: ObbTW  # Current fused OBB
    track_id: int  # Unique monotonic ID
    support_count: int  # Total matched detections
    last_seen_frame: int  # Last frame index where matched
    first_seen_frame: int  # First frame index where created
    state: TrackState  # Current track state
    accumulated_weight: float  # Running sum of confidence weights
    missed_count: int = 0  # Frames where visible but unmatched
    last_visible: bool = True  # Whether track was visible in last aging check
    cached_text: str = "?"  # Cached text label (updated on match)


class BoundingBox3DTracker:
    """Online 3D bounding box tracker using Hungarian assignment."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        min_hits: int = 3,
        conf_threshold: float = 0.55,
        ema_decay: float = 0.0,
        samp_per_dim: int = 8,
        max_missed: int = 5,
        force_cpu: bool = False,
        merge_iou_threshold: float = 0.5,
        merge_semantic_threshold: float = 0.7,
        merge_iou_2d_threshold: float = 0.7,
        merge_interval: int = 5,
        min_confidence_mass: float = 4.0,
        min_obs_points: int = 2,
        verbose: bool = True,
    ) -> None:
        """
        Initialize tracker.

        Args:
            iou_threshold: Minimum IoU for valid match
            min_hits: Matches to promote TENTATIVE -> ACTIVE
            conf_threshold: Minimum detection confidence
            ema_decay: 0 = confidence-weighted running avg; >0 = EMA
            samp_per_dim: IoU sampling density (lower = faster for online)
            max_missed: Visible-but-unmatched frames before track removal
            force_cpu: Force IoU computation on CPU (avoids MPS scheduling jitter)
            merge_iou_threshold: Min IoU between tracks to consider merging (0 = disabled)
            merge_semantic_threshold: Min cosine similarity between track labels to allow merge
            merge_iou_2d_threshold: Min 2D IoU (projected) between tracks to consider merging as secondary criterion
            merge_interval: Run merge every N frames (1 = every frame, 10 = every 10th frame)
            min_confidence_mass: Accumulated confidence to promote TENTATIVE -> ACTIVE (alternative to min_hits)
        """
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.conf_threshold = conf_threshold
        self.ema_decay = ema_decay
        self.samp_per_dim = samp_per_dim
        self.max_missed = max_missed
        self.force_cpu = force_cpu
        self.merge_iou_threshold = merge_iou_threshold
        self.merge_semantic_threshold = merge_semantic_threshold
        self.merge_iou_2d_threshold = merge_iou_2d_threshold
        self.merge_interval = merge_interval
        self.min_confidence_mass = min_confidence_mass
        self.min_obs_points = min_obs_points
        self.verbose = verbose

        self.tracks: list[TrackedInstance] = []
        self._next_id: int = 0
        self.last_iou_matrix_size: tuple[int, int] = (0, 0)
        self._embedding_cache: dict[str, torch.Tensor] = {}
        self._embed_model = None

    def _get_next_id(self) -> int:
        """Get next unique track ID."""
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def update(
        self,
        detections: ObbTW,
        frame_idx: int,
        cam: Optional[CameraTW] = None,
        T_world_rig: Optional[PoseTW] = None,
        observed_points: Optional[torch.Tensor] = None,
    ) -> List[TrackedInstance]:
        """
        Process one frame of detections and update tracks.

        Args:
            detections: ObbTW tensor of detections for this frame
            frame_idx: Current frame index
            cam: Optional camera for visibility-aware aging
            T_world_rig: Optional rig pose for visibility-aware aging
            observed_points: Optional (K, 3) semidense points for occlusion-aware aging

        Returns:
            List of active tracked instances
        """
        # Filter detections by confidence threshold
        if len(detections) > 0 and self.conf_threshold > 0:
            conf_mask = (detections.prob >= self.conf_threshold).reshape(-1)
            detections = detections[conf_mask]

        N = len(detections)
        M = len(self.tracks)

        # Case 1: No tracks exist -> create new tracks from all detections
        if M == 0:
            for i in range(N):
                self._create_track(detections[i], frame_idx)
            return self._get_active_tracks()

        # Case 2: No detections -> age all tracks
        if N == 0:
            t0 = time.perf_counter()
            self._age_tracks(frame_idx, cam, T_world_rig, observed_points)
            if self.verbose:
                print(
                    f"  [BENCH] tracker.age_all ({M} tracks): {(time.perf_counter() - t0) * 1000:.1f}ms"
                )
            return self._get_active_tracks()

        # Case 3: Both tracks and detections exist -> match

        # Partition tracks into visible and invisible using cached flag.
        # Invisible tracks (behind walls, out of FOV) cannot match any
        # detection, so we skip them in the IoU computation entirely.
        visible_indices = [i for i, t in enumerate(self.tracks) if t.last_visible]
        invisible_indices = [i for i, t in enumerate(self.tracks) if not t.last_visible]
        V = len(visible_indices)

        t0 = time.perf_counter()
        matched_tracks = set()
        matched_detections = set()

        if V > 0:
            # Stack only visible track OBBs for batch IoU (V×N instead of M×N)
            visible_obbs = torch.stack([self.tracks[i].obb for i in visible_indices])

            # Move to GPU if available for IoU computation
            if self.force_cpu:
                visible_obbs_gpu = visible_obbs
                detections_gpu = detections
            elif torch.backends.mps.is_available():
                visible_obbs_gpu = visible_obbs.to("mps")
                detections_gpu = detections.to("mps")
            elif torch.cuda.is_available():
                visible_obbs_gpu = visible_obbs.to("cuda")
                detections_gpu = detections.to("cuda")
            else:
                visible_obbs_gpu = visible_obbs
                detections_gpu = detections

            # Compute V×N IoU matrix (visible tracks only)
            iou_result = iou_mc7(
                visible_obbs_gpu,
                detections_gpu,
                samp_per_dim=self.samp_per_dim,
                verbose=False,
            )
            iou_matrix = iou_result.cpu()  # (V, N)
            t_iou = time.perf_counter()

            # Build cost matrix for Hungarian assignment
            cost_matrix = 1.0 - iou_matrix.numpy()

            # Gate invalid pairs (IoU below threshold)
            invalid_mask = iou_matrix.numpy() < self.iou_threshold
            cost_matrix[invalid_mask] = 1e6

            # Hungarian assignment
            from utils.fuse_3d_boxes import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            t_hungarian = time.perf_counter()

            # Map Hungarian row indices back to self.tracks indices
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1.0:  # Valid match
                    original_idx = visible_indices[r]
                    self._update_track(
                        self.tracks[original_idx], detections[c], frame_idx
                    )
                    matched_tracks.add(original_idx)
                    matched_detections.add(c)
        else:
            t_iou = t0
            t_hungarian = t0

        self.last_iou_matrix_size = (V, N)

        # Create new tracks from unmatched detections
        for j in range(N):
            if j not in matched_detections:
                self._create_track(detections[j], frame_idx)

        # Age unmatched tracks: unmatched visible + all invisible
        unmatched_indices = [
            i for i in visible_indices if i not in matched_tracks
        ] + invisible_indices
        if unmatched_indices:
            self._age_tracks_batched(
                unmatched_indices, frame_idx, cam, T_world_rig, observed_points
            )
        t_age = time.perf_counter()

        # Remove dead tracks
        if self.verbose:
            for t in self.tracks:
                if self._should_remove(t, frame_idx):
                    frames_since = frame_idx - t.last_seen_frame
                    avg_p = t.accumulated_weight / max(t.support_count, 1)
                    print(
                        f"  [TRACK] REMOVE id={t.track_id} '{t.cached_text}' "
                        f"state={t.state.name} support={t.support_count} "
                        f"missed={t.missed_count} frames_since_seen={frames_since} "
                        f"avg_conf={avg_p:.2f}"
                    )
        self.tracks = [t for t in self.tracks if not self._should_remove(t, frame_idx)]
        t_remove = time.perf_counter()

        # Merge duplicate tracks (every N frames to amortize cost)
        if self.merge_interval <= 1 or frame_idx % self.merge_interval == 0:
            self._merge_duplicate_tracks(cam=cam, T_world_rig=T_world_rig)
        t_merge = time.perf_counter()

        if self.verbose:
            n_tent = sum(1 for t in self.tracks if t.state == TrackState.TENTATIVE)
            n_active = sum(1 for t in self.tracks if t.state == TrackState.ACTIVE)
            n_inactive = sum(1 for t in self.tracks if t.state == TrackState.INACTIVE)
            print(
                f"  [BENCH] tracker.update ({V}v/{M}x{N}): "
                f"iou={(t_iou - t0) * 1000:.1f}ms  "
                f"hungarian={(t_hungarian - t_iou) * 1000:.1f}ms  "
                f"age({len(unmatched_indices)})={(t_age - t_hungarian) * 1000:.1f}ms  "
                f"merge={(t_merge - t_remove) * 1000:.1f}ms  "
                f"tracks={len(self.tracks)}(T={n_tent}/A={n_active}/I={n_inactive}) "
                f"matched={len(matched_tracks)}/{V}vis"
            )

        return self._get_active_tracks()

    def _create_track(self, detection: ObbTW, frame_idx: int) -> None:
        """Create a new track from a detection."""
        try:
            text_label = detection.reshape(1, -1).text_string()[0]
        except Exception:
            text_label = "?"
        new_id = self._get_next_id()
        prob = float(detection.prob.item())
        track = TrackedInstance(
            obb=detection.clone(),
            track_id=new_id,
            support_count=1,
            last_seen_frame=frame_idx,
            first_seen_frame=frame_idx,
            state=TrackState.TENTATIVE,
            accumulated_weight=prob,
            cached_text=text_label,
        )
        self.tracks.append(track)
        if self.verbose:
            print(
                f"  [TRACK] CREATE id={new_id} '{text_label}' "
                f"prob={prob:.2f} frame={frame_idx}"
            )

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get normalized text embedding, using persistent cache."""
        if text not in self._embedding_cache:
            if self._embed_model is None:
                try:
                    from owl.clip_tokenizer import TextEmbedder
                except ImportError:
                    raise ImportError("condense_text module not available")

                self._embed_model = TextEmbedder()
            self._embedding_cache[text] = self._embed_model.forward([text])[0]  # (embed_dim,)
        return self._embedding_cache[text]

    def _update_track(
        self,
        track: TrackedInstance,
        detection: ObbTW,
        frame_idx: int,
    ) -> None:
        """Fuse a new detection into an existing track."""
        track.support_count += 1
        track.last_seen_frame = frame_idx
        track.missed_count = 0
        track.last_visible = True

        # Promote TENTATIVE -> ACTIVE if enough hits or accumulated confidence
        if track.state == TrackState.TENTATIVE and (
            track.support_count >= self.min_hits
            or track.accumulated_weight >= self.min_confidence_mass
        ):
            track.state = TrackState.ACTIVE
            if self.verbose:
                print(
                    f"  [TRACK] PROMOTE id={track.track_id} '{track.cached_text}' "
                    f"support={track.support_count} conf_mass={track.accumulated_weight:.2f}"
                )
        elif track.state == TrackState.INACTIVE:
            track.state = TrackState.ACTIVE

        # Compute new weight
        new_weight = float(detection.prob.item())

        if self.ema_decay > 0:
            # EMA update
            alpha = self.ema_decay
            old_obb = track.obb
            new_obb = detection

            # Fuse translation via EMA
            old_t = old_obb.T_world_object.t  # (3,)
            new_t = new_obb.T_world_object.t  # (3,)
            fused_t = (1 - alpha) * old_t + alpha * new_t

            # Fuse sizes via EMA
            old_extents = old_obb.bb3_object.reshape(-1)  # (6,)
            new_extents = new_obb.bb3_object.reshape(-1)  # (6,)
            old_sizes = torch.tensor(
                [
                    old_extents[1] - old_extents[0],
                    old_extents[3] - old_extents[2],
                    old_extents[5] - old_extents[4],
                ]
            )
            new_sizes = torch.tensor(
                [
                    new_extents[1] - new_extents[0],
                    new_extents[3] - new_extents[2],
                    new_extents[5] - new_extents[4],
                ]
            )
            fused_sizes = (1 - alpha) * old_sizes + alpha * new_sizes

            # Fuse yaw with alignment
            old_eulers = old_obb.T_world_object.to_euler().reshape(-1)  # (3,)
            new_eulers = new_obb.T_world_object.to_euler().reshape(-1)  # (3,)
            old_yaw = old_eulers[2]
            new_yaw = new_eulers[2]

            sizes_pair = torch.stack([old_sizes, new_sizes])
            yaws_pair = torch.stack(
                [old_yaw.reshape(-1), new_yaw.reshape(-1)]
            ).squeeze()
            weights_pair = torch.tensor([1 - alpha, alpha])

            aligned_sizes, aligned_yaws = align_boxes_r90(
                sizes_pair, yaws_pair, weights_pair
            )
            fused_yaw, _ = weighted_yaw_mean(aligned_yaws, weights_pair)
            fused_sizes = (aligned_sizes * weights_pair.unsqueeze(1)).sum(dim=0)
        else:
            # Confidence-weighted running average
            total_weight = track.accumulated_weight + new_weight
            w_old = track.accumulated_weight / total_weight
            w_new = new_weight / total_weight

            old_obb = track.obb
            new_obb = detection

            # Fuse translation
            old_t = old_obb.T_world_object.t  # (3,)
            new_t = new_obb.T_world_object.t  # (3,)
            fused_t = w_old * old_t + w_new * new_t

            # Extract sizes - flatten to 1D to handle both (6,) and (1, 6)
            old_extents = old_obb.bb3_object.reshape(-1)  # (6,)
            new_extents = new_obb.bb3_object.reshape(-1)  # (6,)
            old_sizes = torch.tensor(
                [
                    old_extents[1] - old_extents[0],
                    old_extents[3] - old_extents[2],
                    old_extents[5] - old_extents[4],
                ]
            )
            new_sizes = torch.tensor(
                [
                    new_extents[1] - new_extents[0],
                    new_extents[3] - new_extents[2],
                    new_extents[5] - new_extents[4],
                ]
            )

            # Fuse yaw with 90-degree alignment
            old_eulers = old_obb.T_world_object.to_euler().reshape(-1)  # (3,)
            new_eulers = new_obb.T_world_object.to_euler().reshape(-1)  # (3,)
            old_yaw = old_eulers[2]
            new_yaw = new_eulers[2]

            sizes_pair = torch.stack([old_sizes, new_sizes])
            yaws_pair = torch.stack(
                [old_yaw.reshape(-1), new_yaw.reshape(-1)]
            ).squeeze()
            weights_pair = torch.tensor([w_old, w_new])

            aligned_sizes, aligned_yaws = align_boxes_r90(
                sizes_pair, yaws_pair, weights_pair
            )
            fused_yaw, _ = weighted_yaw_mean(aligned_yaws, weights_pair)
            fused_sizes = (aligned_sizes * weights_pair.unsqueeze(1)).sum(dim=0)

        track.accumulated_weight += new_weight

        # Build fused OBB
        bb3_object = torch.stack(
            [
                -fused_sizes[0] / 2,
                fused_sizes[0] / 2,
                -fused_sizes[1] / 2,
                fused_sizes[1] / 2,
                -fused_sizes[2] / 2,
                fused_sizes[2] / 2,
            ]
        )

        new_euler = torch.tensor([0, 0, fused_yaw]).to(fused_t)
        new_euler = new_euler.reshape(1, 3)
        fused_rotation = rotation_from_euler(new_euler)[0]
        fused_pose = PoseTW.from_Rt(fused_rotation, fused_t)

        # Fuse confidence
        if self.ema_decay > 0:
            fused_prob = (
                1 - self.ema_decay
            ) * old_obb.prob + self.ema_decay * new_obb.prob
        else:
            fused_prob = w_old * old_obb.prob + w_new * new_obb.prob

        # Take text label from highest-confidence observation
        if new_weight >= track.accumulated_weight - new_weight:
            # New detection has higher cumulative contribution
            try:
                text_label = new_obb.reshape(1, -1).text_string()[0]
            except Exception:
                text_label = "Unknown"
        else:
            try:
                text_label = old_obb.reshape(1, -1).text_string()[0]
            except Exception:
                text_label = "Unknown"
        text_padded = string2tensor(pad_string(text_label, max_len=128))
        track.cached_text = text_label

        # Get sem_id from detection with highest confidence
        sem_id = (
            new_obb.sem_id
            if new_weight >= (track.accumulated_weight - new_weight)
            else old_obb.sem_id
        )

        track.obb = ObbTW.from_lmc(
            bb3_object=bb3_object,
            prob=fused_prob.reshape(1, 1),
            T_world_object=fused_pose,
            text=text_padded,
            sem_id=sem_id,
        )

    def _merge_track_pair(
        self, absorber: TrackedInstance, absorbed: TrackedInstance
    ) -> None:
        """Merge absorbed track into absorber using accumulated-weight fusion."""
        w_a = absorber.accumulated_weight
        w_b = absorbed.accumulated_weight
        total = w_a + w_b
        f_a, f_b = w_a / total, w_b / total

        old_obb = absorber.obb
        new_obb = absorbed.obb

        # Fuse translation
        fused_t = f_a * old_obb.T_world_object.t + f_b * new_obb.T_world_object.t

        # Extract sizes
        old_extents = old_obb.bb3_object.reshape(-1)  # (6,)
        new_extents = new_obb.bb3_object.reshape(-1)  # (6,)
        old_sizes = torch.tensor(
            [
                old_extents[1] - old_extents[0],
                old_extents[3] - old_extents[2],
                old_extents[5] - old_extents[4],
            ]
        )
        new_sizes = torch.tensor(
            [
                new_extents[1] - new_extents[0],
                new_extents[3] - new_extents[2],
                new_extents[5] - new_extents[4],
            ]
        )

        # Fuse yaw with 90-degree alignment
        old_eulers = old_obb.T_world_object.to_euler().reshape(-1)  # (3,)
        new_eulers = new_obb.T_world_object.to_euler().reshape(-1)  # (3,)
        old_yaw = old_eulers[2]
        new_yaw = new_eulers[2]

        sizes_pair = torch.stack([old_sizes, new_sizes])
        yaws_pair = torch.stack([old_yaw.reshape(-1), new_yaw.reshape(-1)]).squeeze()
        weights_pair = torch.tensor([f_a, f_b])

        aligned_sizes, aligned_yaws = align_boxes_r90(
            sizes_pair, yaws_pair, weights_pair
        )
        fused_yaw, _ = weighted_yaw_mean(aligned_yaws, weights_pair)
        fused_sizes = (aligned_sizes * weights_pair.unsqueeze(1)).sum(dim=0)

        # Build fused OBB
        bb3_object = torch.stack(
            [
                -fused_sizes[0] / 2,
                fused_sizes[0] / 2,
                -fused_sizes[1] / 2,
                fused_sizes[1] / 2,
                -fused_sizes[2] / 2,
                fused_sizes[2] / 2,
            ]
        )

        new_euler = torch.tensor([0, 0, fused_yaw]).to(fused_t)
        new_euler = new_euler.reshape(1, 3)
        fused_rotation = rotation_from_euler(new_euler)[0]
        fused_pose = PoseTW.from_Rt(fused_rotation, fused_t)

        # Fuse confidence
        fused_prob = f_a * old_obb.prob + f_b * new_obb.prob

        # Keep text/sem_id from track with higher accumulated weight
        if w_a >= w_b:
            text_label = absorber.cached_text
            sem_id = old_obb.sem_id
        else:
            text_label = absorbed.cached_text
            sem_id = new_obb.sem_id
            absorber.cached_text = text_label

        text_padded = string2tensor(pad_string(text_label, max_len=128))

        absorber.obb = ObbTW.from_lmc(
            bb3_object=bb3_object,
            prob=fused_prob.reshape(1, 1),
            T_world_object=fused_pose,
            text=text_padded,
            sem_id=sem_id,
        )

        # Update metadata
        absorber.accumulated_weight = total
        absorber.support_count += absorbed.support_count
        absorber.first_seen_frame = min(
            absorber.first_seen_frame, absorbed.first_seen_frame
        )
        absorber.last_seen_frame = max(
            absorber.last_seen_frame, absorbed.last_seen_frame
        )

    def _merge_duplicate_tracks(self, cam=None, T_world_rig=None) -> None:
        """Merge overlapping tracks with similar labels (greedy, non-TENTATIVE only).

        Uses V×N IoU (visible × all) instead of N×N to avoid comparing pairs of
        invisible tracks. A merge only matters when at least one track is visible,
        since that's when the user is looking at an area where duplicates could be
        detected.

        Uses 3D IoU as the primary merge criterion. When cam/pose are available,
        also computes 2D IoU (projected bounding boxes) as a secondary criterion
        to catch duplicates with slightly different 3D positions but heavy 2D overlap.
        """
        if self.merge_iou_threshold <= 0:
            return

        all_candidates = [t for t in self.tracks if t.state != TrackState.TENTATIVE]
        if len(all_candidates) < 2:
            return

        visible_candidates = [t for t in all_candidates if t.last_visible]
        if len(visible_candidates) == 0:
            return

        # Pre-filter: drop all_candidates whose centroid is > 4m from every visible centroid.
        # This reduces N cheaply before the expensive IoU computation.
        vis_centroids = torch.stack(
            [t.obb.T_world_object.t.reshape(3) for t in visible_candidates]
        )  # (V, 3)
        all_centroids = torch.stack(
            [t.obb.T_world_object.t.reshape(3) for t in all_candidates]
        )  # (N, 3)
        dists = torch.cdist(vis_centroids, all_centroids)  # (V, N)
        min_dist_per_col = dists.min(dim=0).values  # (N,)
        nearby_mask = (
            min_dist_per_col <= 4.0
        )  # keep those within 4m of any visible track
        # Always keep visible tracks themselves (they may be > 4m from each other)
        vis_set = {t.track_id for t in visible_candidates}
        for j, t in enumerate(all_candidates):
            if t.track_id in vis_set:
                nearby_mask[j] = True
        nearby_indices = torch.where(nearby_mask)[0].tolist()

        all_candidates = [all_candidates[j] for j in nearby_indices]

        if len(all_candidates) < 2:
            return

        V = len(visible_candidates)
        N = len(all_candidates)

        # Map visible indices back to all_candidates indices (for self-pair skipping)
        all_id_to_idx = {t.track_id: idx for idx, t in enumerate(all_candidates)}
        vis_to_all_idx = [all_id_to_idx[t.track_id] for t in visible_candidates]

        # V×N 3D IoU (always request GIoU so we can use it as fallback)
        visible_stacked = ObbTW(
            torch.stack([t.obb._data.squeeze() for t in visible_candidates])
        )
        all_stacked = ObbTW(
            torch.stack([t.obb._data.squeeze() for t in all_candidates])
        )
        iou_matrix = iou_mc7(
            visible_stacked,
            all_stacked,
            samp_per_dim=self.samp_per_dim,
        )

        # V×N semantic similarity (cached embeddings, cheap)
        vis_embeddings = torch.stack(
            [self._get_text_embedding(t.cached_text) for t in visible_candidates]
        )
        all_embeddings = torch.stack(
            [self._get_text_embedding(t.cached_text) for t in all_candidates]
        )
        sem_matrix = vis_embeddings @ all_embeddings.t()  # (V, N)

        # Apply semantic similarity overrides for known-similar label pairs
        for i in range(V):
            text_i = visible_candidates[i].cached_text.lower()
            for j in range(N):
                override = _SEM_OVERRIDES.get(
                    (text_i, all_candidates[j].cached_text.lower())
                )
                if override is not None:
                    sem_matrix[i, j] = override

        # V×N 2D IoU as secondary merge criterion
        iou_2d_matrix = None
        if (
            cam is not None
            and T_world_rig is not None
            and self.merge_iou_2d_threshold > 0
        ):
            bb2s, bb2s_valid = all_stacked.get_pseudo_bb2(
                cam.unsqueeze(0),
                T_world_rig.unsqueeze(0),
                num_samples_per_edge=1,  # corners only -- fast
                valid_ratio=0.1667,
            )
            # bb2s shape: (1, N, 4) in [xmin, xmax, ymin, ymax] format
            bb2s = bb2s.squeeze(0)  # (N, 4)
            bb2s_valid = bb2s_valid.squeeze(0)  # (N,)

            # Slice visible rows from the N projections
            v_bb2s = bb2s[vis_to_all_idx]  # (V, 4)
            v_valid = bb2s_valid[vis_to_all_idx]  # (V,)

            # Vectorized V×N axis-aligned 2D IoU
            xmin, xmax, ymin, ymax = bb2s[:, 0], bb2s[:, 1], bb2s[:, 2], bb2s[:, 3]
            v_xmin, v_xmax, v_ymin, v_ymax = (
                v_bb2s[:, 0],
                v_bb2s[:, 1],
                v_bb2s[:, 2],
                v_bb2s[:, 3],
            )
            # Intersection
            ix_min = torch.maximum(v_xmin[:, None], xmin[None, :])  # (V, N)
            ix_max = torch.minimum(v_xmax[:, None], xmax[None, :])
            iy_min = torch.maximum(v_ymin[:, None], ymin[None, :])
            iy_max = torch.minimum(v_ymax[:, None], ymax[None, :])
            iw = (ix_max - ix_min).clamp(min=0)
            ih = (iy_max - iy_min).clamp(min=0)
            inter = iw * ih
            # Union
            v_area = (v_xmax - v_xmin) * (v_ymax - v_ymin)  # (V,)
            a_area = (xmax - xmin) * (ymax - ymin)  # (N,)
            union = v_area[:, None] + a_area[None, :] - inter  # (V, N)
            iou_2d_matrix = inter / (union + 1e-8)

            # Zero out pairs where either box has invalid 2D projection
            valid_mask = v_valid[:, None] & bb2s_valid[None, :]  # (V, N)
            iou_2d_matrix[~valid_mask] = 0.0

        # Greedy merge: i indexes visible_candidates (rows), j indexes all_candidates (cols)
        absorbed_ids: set[int] = set()

        for i in range(V):
            if visible_candidates[i].track_id in absorbed_ids:
                continue
            for j in range(N):
                if all_candidates[j].track_id in absorbed_ids:
                    continue
                # Skip self-pairs (visible_candidates[i] is also in all_candidates)
                if vis_to_all_idx[i] == j:
                    continue

                # Primary: 3D IoU. Secondary: 2D IoU.
                has_3d_overlap = iou_matrix[i, j] >= self.merge_iou_threshold
                has_2d_overlap = (
                    iou_2d_matrix is not None
                    and iou_2d_matrix[i, j] >= self.merge_iou_2d_threshold
                )
                if not has_3d_overlap and not has_2d_overlap:
                    continue

                # Semantic check
                if sem_matrix[i, j] < self.merge_semantic_threshold:
                    continue

                # Absorb weaker (lower accumulated_weight) into stronger
                if (
                    visible_candidates[i].accumulated_weight
                    >= all_candidates[j].accumulated_weight
                ):
                    self._merge_track_pair(visible_candidates[i], all_candidates[j])
                    absorbed_ids.add(all_candidates[j].track_id)
                else:
                    self._merge_track_pair(all_candidates[j], visible_candidates[i])
                    absorbed_ids.add(visible_candidates[i].track_id)
                    break  # i is gone

        if absorbed_ids:
            self.tracks = [t for t in self.tracks if t.track_id not in absorbed_ids]

    def _age_tracks_batched(
        self,
        track_indices: list[int],
        frame_idx: int,
        cam: Optional[CameraTW] = None,
        T_world_rig: Optional[PoseTW] = None,
        observed_points: Optional[torch.Tensor] = None,
    ) -> None:
        """Age a batch of unmatched tracks with a single batched visibility check."""
        tracks_to_age = [self.tracks[i] for i in track_indices]

        # Stack OBBs once for reuse in both visibility and containment checks
        stacked_obbs = (
            torch.stack([t.obb for t in tracks_to_age])
            if len(tracks_to_age) > 0
            else None
        )

        # Compute visibility for all tracks in one batched call
        if cam is not None and T_world_rig is not None and stacked_obbs is not None:
            _, bb2_valid = stacked_obbs.get_pseudo_bb2(
                cam.unsqueeze(0),
                T_world_rig.unsqueeze(0),
                num_samples_per_edge=10,
                valid_ratio=0.16667,
            )
            is_visible = bb2_valid.squeeze(0)  # (K,) bool tensor
        else:
            is_visible = None

        # Semidense containment check: count observed points inside each track's 3D box
        if observed_points is not None and stacked_obbs is not None:
            K = len(stacked_obbs)
            N_pts = observed_points.shape[0]
            pts_expanded = observed_points.unsqueeze(0).expand(K, N_pts, 3)  # (K, N, 3)
            inside = stacked_obbs.batch_points_inside_bb3(pts_expanded)  # (K, N) bool
            points_inside_count = inside.sum(dim=1)  # (K,)
            has_enough_points = points_inside_count >= self.min_obs_points  # (K,) bool
        else:
            has_enough_points = None

        for k, track in enumerate(tracks_to_age):
            frames_since_seen = frame_idx - track.last_seen_frame
            if track.state == TrackState.ACTIVE and frames_since_seen >= 1:
                track.state = TrackState.INACTIVE

            # Determine if track is projection-visible
            if is_visible is not None:
                projected_visible = is_visible[k].item()
            else:
                projected_visible = True  # conservative fallback

            # Determine if track contains enough observed semidense points
            if has_enough_points is not None:
                contains_points = has_enough_points[k].item()
            else:
                contains_points = True  # conservative fallback (no point data)

            # last_visible requires both frustum visibility and point support.
            # This is the "strictly visible" definition used by viz + matching.
            track.last_visible = projected_visible and contains_points
            # Option B behavior: age only when the track is both in-FOV AND has
            # enough currently observed points. Projected-but-unsupported tracks
            # should persist (occluded / sparse observation) instead of being
            # deleted by missed-count accumulation.
            if projected_visible and contains_points:
                track.missed_count += 1

    def _age_single_track(
        self,
        track: TrackedInstance,
        frame_idx: int,
        cam: Optional[CameraTW] = None,
        T_world_rig: Optional[PoseTW] = None,
        observed_points: Optional[torch.Tensor] = None,
    ) -> None:
        """Age a single unmatched track, using visibility-aware missed counting."""
        frames_since_seen = frame_idx - track.last_seen_frame
        if track.state == TrackState.ACTIVE and frames_since_seen >= 1:
            track.state = TrackState.INACTIVE

        # Visibility-aware missed counting
        if cam is not None and T_world_rig is not None:
            # get_pseudo_bb2 expects: obb (N, 165), cam (T, C), T_world_rig (T, 12)
            obb_batched = track.obb.reshape(1, -1)  # (1, 165)
            cam_batched = cam.unsqueeze(0)  # (1, C) — T=1
            pose_batched = T_world_rig.unsqueeze(0)  # (1, 12) — T=1
            _, bb2_valid = obb_batched.get_pseudo_bb2(
                cam_batched,
                pose_batched,
                num_samples_per_edge=10,
                valid_ratio=0.16667,
            )
            if bb2_valid.any().item():
                track.last_visible = True
                track.missed_count += 1
            else:
                track.last_visible = False
                # If not visible, don't increment missed_count
        else:
            # Fallback: no camera info, always count as missed
            track.last_visible = True
            track.missed_count += 1

    def _age_tracks(
        self,
        frame_idx: int,
        cam: Optional[CameraTW] = None,
        T_world_rig: Optional[PoseTW] = None,
        observed_points: Optional[torch.Tensor] = None,
    ) -> None:
        """Age all tracks (called when no detections)."""
        all_indices = list(range(len(self.tracks)))
        if all_indices:
            self._age_tracks_batched(
                all_indices, frame_idx, cam, T_world_rig, observed_points
            )
        self.tracks = [t for t in self.tracks if not self._should_remove(t, frame_idx)]

    def _should_remove(self, track: TrackedInstance, frame_idx: int) -> bool:
        """Determine if a track should be removed."""
        frames_since_seen = frame_idx - track.last_seen_frame

        # Visibility-aware removal — scale patience by avg confidence
        avg_prob = track.accumulated_weight / max(track.support_count, 1)
        effective_max_missed = int(self.max_missed * (0.5 + avg_prob))
        if track.missed_count >= effective_max_missed:
            return True

        # Remove TENTATIVE tracks that missed 2 frames
        if track.state == TrackState.TENTATIVE and frames_since_seen >= 2:
            return True

        return False

    def _get_active_tracks(self) -> list[TrackedInstance]:
        """Return tracks that are ACTIVE or INACTIVE (still visible during brief occlusions)."""
        return [
            t
            for t in self.tracks
            if t.state in (TrackState.ACTIVE, TrackState.INACTIVE)
        ]

    def get_all_tracks(self) -> list[TrackedInstance]:
        """Return all tracks (including TENTATIVE and INACTIVE)."""
        return list(self.tracks)

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self._next_id = 0


def track_obbs_from_csv(
    input_path: str,
    output_path: Optional[str] = None,
    iou_threshold: float = 0.25,
    min_hits: int = 8,
    conf_threshold: float = 0.55,
    samp_per_dim: int = 8,
    max_missed: int = 90,
) -> list[TrackedInstance]:
    """
    Load OBBs from CSV, run online tracking frame-by-frame, save results.

    Args:
        input_path: Path to input obb.csv file
        output_path: Path to output obb_tracked.csv file (default: input with _tracked suffix)
        iou_threshold: IoU threshold for matching
        min_hits: Min matches to confirm a track
        conf_threshold: Min detection confidence
        samp_per_dim: IoU sampling density
        max_missed: Visible-but-unmatched frames before track removal

    Returns:
        List of final active tracked instances
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_tracked{ext}"

    print(f"==> Loading OBBs from {input_path}")
    timed_obbs = read_obb_csv(input_path)

    if len(timed_obbs) == 0:
        print("==> No OBBs found in input file, nothing to track")
        return []

    # Sort frames by timestamp
    sorted_frames = sorted(timed_obbs.keys())
    total_frames = len(sorted_frames)
    print(f"==> Loaded {total_frames} frames from {input_path}")

    # Create tracker
    tracker = BoundingBox3DTracker(
        iou_threshold=iou_threshold,
        min_hits=min_hits,
        conf_threshold=conf_threshold,
        samp_per_dim=samp_per_dim,
        max_missed=max_missed,
        verbose=False,
    )

    # Process frame by frame
    print(f"\n{'=' * 60}")
    print(f"TRACKING {total_frames} FRAMES")
    print(f"{'=' * 60}")

    start_time = time.time()
    for frame_idx, ts in enumerate(sorted_frames):
        detections = timed_obbs[ts]
        active_tracks = tracker.update(detections, frame_idx)

        if (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
            total_tracks = len(tracker.get_all_tracks())
            print(
                f"  Frame {frame_idx + 1}/{total_frames}: "
                f"{len(detections)} detections, "
                f"{len(active_tracks)} active / {total_tracks} total tracks"
            )

    elapsed = time.time() - start_time
    print(f"\n==> Tracking complete in {elapsed:.2f}s")

    # Get active tracks (confirmed and not removed)
    active_tracks = tracker._get_active_tracks()
    print(f"==> {len(active_tracks)} active tracks")

    if len(active_tracks) == 0:
        print("==> No tracks, skipping output")
        return []

    # Write tracked OBBs
    tracked_obbs = torch.stack([t.obb for t in active_tracks])

    # Assign track IDs as instance IDs
    ids = torch.tensor([t.track_id for t in active_tracks], dtype=torch.int32)
    tracked_obbs.set_inst_id(ids)

    # Round prob
    rounded_prob = torch.round(tracked_obbs.prob * 100) / 100
    tracked_obbs.set_prob(rounded_prob.squeeze(-1), use_mask=False)

    # Build sem_id_to_name mapping
    sem_id_to_name = {}
    for obb in tracked_obbs:
        sem_id = int(obb.sem_id.item())
        if sem_id not in sem_id_to_name:
            text = unpad_string(tensor2string(obb.text.int()))
        else:
            text = sem_id_to_name[sem_id]
        sem_id_to_name[sem_id] = text

    writer = ObbCsvWriter2(output_path)
    writer.write(tracked_obbs, timestamps_ns=0, sem_id_to_name=sem_id_to_name)
    writer.close()
    print(f"==> Saved {len(active_tracks)} tracked OBBs to {output_path}")

    return active_tracks
