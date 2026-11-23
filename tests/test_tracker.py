"""SORT tracker smoke test on synthetic boxes (Kalman + Hungarian)."""

from __future__ import annotations

import numpy as np

from motion_detect.tracker import SortTracker, center_distance, iou


def test_iou_basic() -> None:
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # half overlap in both axes -> inter 25, union 175
    assert abs(iou([0, 0, 10, 10], [5, 5, 15, 15]) - 25.0 / 175.0) < 1e-9


def test_center_distance() -> None:
    assert center_distance([0, 0, 2, 2], [0, 0, 2, 2]) == 0.0
    # centers (1,1) and (4,5) -> distance 5
    assert abs(center_distance([0, 0, 2, 2], [3, 4, 5, 6]) - 5.0) < 1e-9


def test_tracker_keeps_stable_id_for_moving_box() -> None:
    tracker = SortTracker(max_age=15, min_hits=2, iou_threshold=0.05, max_dist=80)
    seen_ids = set()
    for frame in range(12):
        x = 10 + frame  # box moving right by 1 px/frame
        tracks = tracker.update([[x, 10, x + 8, 18]])
        active = [t for t in tracks if t.time_since_update == 0]
        if active:
            seen_ids.add(active[0].track_id)
    assert len(seen_ids) == 1  # one stable track


def test_tracker_creates_and_expires_tracks() -> None:
    tracker = SortTracker(max_age=2, min_hits=2, iou_threshold=0.05, max_dist=80)
    tracker.update([[0, 0, 10, 10]])
    tracker.update([])
    tracker.update([])
    tracker.update([])  # beyond max_age -> pruned
    assert tracker.tracks == []
    # a far-away new detection starts a new id
    tracker.update([[500, 500, 510, 510]])
    assert tracker.tracks[-1].track_id != 1


def test_hungarian_assignment_prefers_iou() -> None:
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.05, max_dist=1000)
    tracker.update([[0, 0, 10, 10], [100, 100, 110, 110]])
    # swap: detection near second track only
    tracks = tracker.update([[101, 101, 111, 111]])
    matched = [t for t in tracks if t.time_since_update == 0]
    assert len(matched) == 1
    assert matched[0].track_id == 2
