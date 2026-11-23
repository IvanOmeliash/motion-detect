"""Compensation pipeline on a synthetic static video -> near-identity homography."""

from __future__ import annotations

import numpy as np
import pytest

from motion_detect.compensation import (
    DEFAULT_COMPENSATION_CONFIG,
    CompensationPipeline,
    detect_points_grid,
    estimate_camera_motion,
)


def _static_frame(rng: np.random.Generator, size: tuple[int, int] = (360, 640)) -> np.ndarray:
    """Textured but static frame (fixed random noise)."""
    h, w = size
    gray = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    # add some structure so GFTT finds corners
    gray[100:260, 150:420] = (gray[100:260, 150:420] // 2 + 128).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1).copy()


def test_config_validation_rejects_missing_keys() -> None:
    with pytest.raises(KeyError):
        CompensationPipeline({"max_width": 640}, step=1)


def test_static_video_yields_identity_motion() -> None:
    rng = np.random.default_rng(0)
    frame = _static_frame(rng)
    pipeline = CompensationPipeline(dict(DEFAULT_COMPENSATION_CONFIG), step=1)

    samples = [pipeline.process_frame(frame) for _ in range(8)]
    valid_samples = [s for s in samples if s.valid]
    assert valid_samples, "static video should produce valid compensated samples"

    identity = np.eye(3, dtype=np.float32)
    for frame_index, matrix in pipeline.matrices.items():
        if frame_index < 2:
            continue
        assert np.allclose(matrix, identity, atol=2.0), (
            f"frame {frame_index}: expected ~identity homography, got\n{matrix}"
        )
    for sample in valid_samples:
        assert sample.tensor.shape[0] == 4  # 3 frames + validity channel
        assert sample.inlier_ratio >= float(DEFAULT_COMPENSATION_CONFIG["min_inlier_ratio"])


def test_estimate_camera_motion_direct_identity() -> None:
    rng = np.random.default_rng(1)
    frame = _static_frame(rng)
    gray = frame[:, :, 0]
    points = detect_points_grid(gray, DEFAULT_COMPENSATION_CONFIG)
    assert len(points) >= 4
    result = estimate_camera_motion(points, points.copy(), DEFAULT_COMPENSATION_CONFIG, gray.shape)
    assert result["valid"]
    assert np.allclose(result["matrix"], np.eye(3, dtype=np.float32), atol=1.0)


def test_translated_frame_gives_pure_translation() -> None:
    rng = np.random.default_rng(2)
    base = _static_frame(rng)[:, :, 0]
    shifted = np.roll(base, shift=(3, 5), axis=(0, 1))  # dy=3, dx=5
    points = detect_points_grid(base, DEFAULT_COMPENSATION_CONFIG)
    # emulate LK output: same points shifted by (dx=5, dy=3)
    shifted_points = points.copy()
    shifted_points[:, 0, 0] += 5
    shifted_points[:, 0, 1] += 3
    result = estimate_camera_motion(points, shifted_points, DEFAULT_COMPENSATION_CONFIG, base.shape)
    assert result["valid"]
    matrix = result["matrix"]
    assert abs(matrix[0, 2] - 5.0) < 1.0
    assert abs(matrix[1, 2] - 3.0) < 1.0
    assert abs(matrix[0, 0] - 1.0) < 0.1 and abs(matrix[1, 1] - 1.0) < 0.1
