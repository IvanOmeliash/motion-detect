"""Production camera-motion compensation for neural-network input tensors.

The module intentionally contains only the classical operations required by the
runtime path:

    resize/grayscale -> grid GFTT -> forward/backward LK -> RANSAC -> grayscale warp

Candidate generation, residual classification, blob detection, visualization,
CSV metrics, and F1 evaluation remain in the validation notebook.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Deque, Dict, Iterator, Mapping, Optional, Tuple

import cv2
import numpy as np


DEFAULT_COMPENSATION_CONFIG = {
    # Processing resolution.
    "max_width": 640,
    # Phase 3 structural speed controls.
    "redetect_interval": 8,
    "request_decoder_scaling": True,
    "preprocess_gray_first": True,

    # Grid GFTT detector.
    "grid_rows": 10,
    "grid_cols": 10,
    "points_per_cell": 3,
    "quality_level": 0.01,
    "min_distance": 7,
    "block_size": 7,

    # Forward/backward Lucas-Kanade.
    "lk_win_size": 11,
    "lk_max_level": 2,
    "min_eig_threshold": 1e-4,
    "fb_threshold": 0.75,
    "min_correspondences": 50,

    # Dominant camera-motion model.
    "motion_model": "homography",
    "ransac_threshold": 2.0,
    "min_ransac_inliers": 25,
    "min_inlier_ratio": 0.50,
    "low_confidence_inlier_ratio": 0.65,
    "min_model_scale": 0.50,
    "max_model_scale": 2.00,
}

_REQUIRED_CONFIG_KEYS = frozenset(DEFAULT_COMPENSATION_CONFIG)
_EMPTY_TENSOR = np.empty((0, 0, 0), dtype=np.float32)
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class CompensatedSample:
    """One network input aligned to ``frame_index``.

    ``tensor`` has shape ``(4, H, W)`` for valid samples. For invalid samples it
    is an empty float32 array; callers must check ``valid`` before inference.
    """

    tensor: np.ndarray
    frame_index: int
    valid: bool
    inlier_ratio: float
    skip_reason: Optional[str]


@dataclass(frozen=True)
class _MotionCacheEntry:
    valid: bool
    matrix: Optional[np.ndarray]
    inlier_ratio: float
    failure_reason: Optional[str]


def _validate_config(config: Mapping[str, object]) -> dict:
    missing = sorted(_REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise KeyError(f"Missing production config keys: {', '.join(missing)}")

    validated = {key: config[key] for key in DEFAULT_COMPENSATION_CONFIG}

    positive_integer_keys = (
        "max_width",
        "grid_rows",
        "grid_cols",
        "points_per_cell",
        "block_size",
        "lk_win_size",
        "min_correspondences",
        "min_ransac_inliers",
        "redetect_interval",
    )
    for key in positive_integer_keys:
        if int(validated[key]) <= 0:
            raise ValueError(f"config[{key!r}] must be positive")

    if int(validated["lk_max_level"]) < 0:
        raise ValueError("config['lk_max_level'] must be non-negative")

    positive_float_keys = (
        "quality_level",
        "min_distance",
        "min_eig_threshold",
        "fb_threshold",
        "ransac_threshold",
        "min_model_scale",
        "max_model_scale",
    )
    for key in positive_float_keys:
        if float(validated[key]) <= 0:
            raise ValueError(f"config[{key!r}] must be positive")

    for key in ("min_inlier_ratio", "low_confidence_inlier_ratio"):
        value = float(validated[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"config[{key!r}] must be in [0, 1]")

    if float(validated["min_model_scale"]) > float(validated["max_model_scale"]):
        raise ValueError("min_model_scale must not exceed max_model_scale")

    if str(validated["motion_model"]) not in {
        "affine_partial",
        "affine_full",
        "homography",
    }:
        raise ValueError(
            "motion_model must be affine_partial, affine_full or homography"
        )

    return validated



def _scaled_size(width: int, height: int, max_width: int) -> Tuple[Tuple[int, int], float]:
    """Return target (width, height) and scale while preserving aspect ratio."""
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")
    if int(max_width) <= 0:
        raise ValueError("max_width must be positive")

    scale = min(1.0, float(max_width) / float(width))
    output_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return output_size, scale


def preprocess_frame(
    frame: np.ndarray,
    max_width: int,
    gray_first: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Downscale a BGR frame and return grayscale plus scale.

    Phase 3.3 uses grayscale-first preprocessing. cvtColor and INTER_AREA resize
    are linear operations, so this preserves the signal up to uint8 rounding
    while avoiding a three-channel resize.
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape H×W×3")
    if frame.dtype != np.uint8:
        raise ValueError("frame must have dtype uint8")
    if max_width is None or int(max_width) <= 0:
        raise ValueError("max_width must be positive")

    height, width = frame.shape[:2]
    output_size, scale = _scaled_size(width, height, int(max_width))

    if bool(gray_first):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                output_size,
                interpolation=cv2.INTER_AREA,
            )
    else:
        resized = (
            cv2.resize(
                frame,
                output_size,
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else frame
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return np.ascontiguousarray(gray), scale


def _request_decoder_scaling(
    capture: cv2.VideoCapture,
    max_width: int,
) -> Optional[Tuple[int, int]]:
    """
    Ask the backend (notably AVFoundation) for a smaller decoded frame.

    Many file backends ignore CAP_PROP_FRAME_WIDTH/HEIGHT. The caller verifies
    the actual decoded size and falls back to preprocess_frame when needed.
    """
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        return None

    target_size, scale = _scaled_size(
        source_width,
        source_height,
        int(max_width),
    )
    if scale >= 1.0:
        return target_size

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(target_size[0]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(target_size[1]))
    return target_size


def detect_points_grid(gray: np.ndarray, config: Mapping[str, object]) -> np.ndarray:
    """Detect GFTT corners per grid cell and enforce global point spacing."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a non-empty uint8 image with shape H×W")
    if gray.size == 0:
        raise ValueError("gray must not be empty")

    grid_rows = int(config["grid_rows"])
    grid_cols = int(config["grid_cols"])
    points_per_cell = int(config["points_per_cell"])
    min_distance = float(config["min_distance"])
    block_size = int(config["block_size"])
    if min(grid_rows, grid_cols, points_per_cell, block_size) <= 0:
        raise ValueError(
            "grid dimensions, points_per_cell, and block_size must be positive"
        )

    height, width = gray.shape
    x_edges = np.linspace(0, width, grid_cols + 1, dtype=np.int32)
    y_edges = np.linspace(0, height, grid_rows + 1, dtype=np.int32)
    candidates = []

    for row in range(grid_rows):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for col in range(grid_cols):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            cell = gray[y0:y1, x0:x1]
            if cell.shape[0] < block_size or cell.shape[1] < block_size:
                continue

            local_points = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=points_per_cell,
                qualityLevel=float(config["quality_level"]),
                minDistance=min_distance,
                blockSize=block_size,
            )
            if local_points is None:
                continue

            local_points = local_points.reshape(-1, 2).astype(np.float32)
            local_points[:, 0] += x0
            local_points[:, 1] += y0
            candidates.extend(local_points)

    if not candidates:
        return np.empty((0, 1, 2), dtype=np.float32)

    # Spatial buckets preserve the notebook's first-point-wins behaviour while
    # avoiding an O(n²) global duplicate check.
    accepted = []
    min_distance_squared = min_distance**2
    bucket_size = max(min_distance, 1e-6)
    buckets: Dict[Tuple[int, int], list[int]] = {}

    def bucket_key(x: float, y: float) -> Tuple[int, int]:
        return (
            int(np.floor(x / bucket_size)),
            int(np.floor(y / bucket_size)),
        )

    for point in np.asarray(candidates, dtype=np.float32):
        if not np.isfinite(point).all():
            continue

        bucket_x, bucket_y = bucket_key(float(point[0]), float(point[1]))
        too_close = False
        for delta_x in (-1, 0, 1):
            if too_close:
                break
            for delta_y in (-1, 0, 1):
                neighbour_indices = buckets.get(
                    (bucket_x + delta_x, bucket_y + delta_y)
                )
                if not neighbour_indices:
                    continue
                for other_index in neighbour_indices:
                    other_point = accepted[other_index]
                    diff_x = float(other_point[0] - point[0])
                    diff_y = float(other_point[1] - point[1])
                    if diff_x * diff_x + diff_y * diff_y < min_distance_squared:
                        too_close = True
                        break
                if too_close:
                    break

        if too_close:
            continue

        buckets.setdefault((bucket_x, bucket_y), []).append(len(accepted))
        accepted.append(point)

    if not accepted:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.asarray(accepted, dtype=np.float32).reshape(-1, 1, 2)



def _merge_spaced_points(
    existing_points: np.ndarray,
    candidate_points: np.ndarray,
    min_distance: float,
) -> np.ndarray:
    """Merge points while preserving existing tracks and global spacing."""
    existing = np.asarray(
        existing_points,
        dtype=np.float32,
    ).reshape(-1, 2)
    candidates = np.asarray(
        candidate_points,
        dtype=np.float32,
    ).reshape(-1, 2)

    if len(existing) == 0:
        return candidates.reshape(-1, 1, 2)
    if len(candidates) == 0:
        return existing.reshape(-1, 1, 2)

    accepted = [point.copy() for point in existing if np.isfinite(point).all()]
    min_distance_squared = float(min_distance) ** 2
    bucket_size = max(float(min_distance), 1e-6)
    buckets: Dict[Tuple[int, int], list[int]] = {}

    def key(point: np.ndarray) -> Tuple[int, int]:
        return (
            int(np.floor(float(point[0]) / bucket_size)),
            int(np.floor(float(point[1]) / bucket_size)),
        )

    for index, point in enumerate(accepted):
        buckets.setdefault(key(point), []).append(index)

    for point in candidates:
        if not np.isfinite(point).all():
            continue
        bucket_x, bucket_y = key(point)
        too_close = False
        for delta_x in (-1, 0, 1):
            if too_close:
                break
            for delta_y in (-1, 0, 1):
                for other_index in buckets.get(
                    (bucket_x + delta_x, bucket_y + delta_y),
                    (),
                ):
                    difference = accepted[other_index] - point
                    if float(difference @ difference) < min_distance_squared:
                        too_close = True
                        break
                if too_close:
                    break
        if too_close:
            continue
        buckets.setdefault((bucket_x, bucket_y), []).append(len(accepted))
        accepted.append(point.copy())

    return np.asarray(accepted, dtype=np.float32).reshape(-1, 1, 2)


def replenish_points_grid(
    gray: np.ndarray,
    existing_points: np.ndarray,
    config: Mapping[str, object],
) -> np.ndarray:
    """Add new grid corners without discarding still-valid tracked points."""
    detected = detect_points_grid(gray, config)
    return _merge_spaced_points(
        existing_points,
        detected,
        min_distance=float(config["min_distance"]),
    )


def track_points_forward_backward_relaxed(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
    config: Mapping[str, object],
) -> Optional[dict]:
    """
    Stateful LK variant that returns all valid tracks, even below the model-fit
    threshold, so the pipeline can replenish them on the current frame.
    """
    if previous_gray.shape != current_gray.shape:
        raise ValueError("previous_gray and current_gray must have identical shapes")
    if previous_gray.ndim != 2 or previous_gray.dtype != np.uint8:
        raise ValueError("frames must be uint8 grayscale images")

    p0 = np.asarray(points, dtype=np.float32)
    if p0.ndim == 2 and p0.shape[1:] == (2,):
        p0 = p0.reshape(-1, 1, 2)
    if p0.ndim != 3 or p0.shape[1:] != (1, 2):
        raise ValueError("points must have shape N×2 or N×1×2")

    finite_input = np.isfinite(p0.reshape(-1, 2)).all(axis=1)
    p0 = p0[finite_input]
    if len(p0) < 4:
        return None

    lk_parameters = {
        "winSize": (int(config["lk_win_size"]), int(config["lk_win_size"])),
        "maxLevel": int(config["lk_max_level"]),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        "minEigThreshold": float(config["min_eig_threshold"]),
    }

    p1, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        p0,
        None,
        **lk_parameters,
    )
    if p1 is None or forward_status is None:
        return None

    p1 = np.asarray(p1, dtype=np.float32).reshape(-1, 1, 2)
    forward_status = np.asarray(forward_status, dtype=np.uint8).reshape(-1)
    if forward_error is None:
        forward_error = np.full(len(p0), np.nan, dtype=np.float32)
    else:
        forward_error = np.asarray(forward_error, dtype=np.float32).reshape(-1)

    p1_flat = p1.reshape(-1, 2)
    height, width = current_gray.shape
    p1_inside = (
        (p1_flat[:, 0] >= 0)
        & (p1_flat[:, 0] < width)
        & (p1_flat[:, 1] >= 0)
        & (p1_flat[:, 1] < height)
    )
    forward_eligible = (
        (forward_status == 1)
        & np.isfinite(p1_flat).all(axis=1)
        & p1_inside
    )
    eligible_indices = np.flatnonzero(forward_eligible)
    if len(eligible_indices) < 4:
        return None

    p0_back = np.full_like(p0, np.nan)
    backward_status = np.zeros(len(p0), dtype=np.uint8)
    p0_back_subset, backward_status_subset, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        p1[eligible_indices],
        None,
        **lk_parameters,
    )
    if p0_back_subset is None or backward_status_subset is None:
        return None

    p0_back[eligible_indices] = np.asarray(
        p0_back_subset,
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    backward_status[eligible_indices] = np.asarray(
        backward_status_subset,
        dtype=np.uint8,
    ).reshape(-1)

    p0_flat = p0.reshape(-1, 2)
    p0_back_flat = p0_back.reshape(-1, 2)
    finite_coordinates = (
        np.isfinite(p0_flat).all(axis=1)
        & np.isfinite(p1_flat).all(axis=1)
        & np.isfinite(p0_back_flat).all(axis=1)
    )
    fb_error = np.full(len(p0), np.inf, dtype=np.float32)
    comparable = forward_eligible & (backward_status == 1) & finite_coordinates
    fb_error[comparable] = np.linalg.norm(
        p0_flat[comparable] - p0_back_flat[comparable],
        axis=1,
    )
    valid_mask = comparable & (fb_error <= float(config["fb_threshold"]))
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < 4:
        return None

    return {
        "p0": p0,
        "p1": p1,
        "p0_back": p0_back,
        "forward_error": forward_error,
        "fb_error": fb_error,
        "valid_mask": valid_mask,
        "valid_p0": p0[valid_mask],
        "valid_p1": p1[valid_mask],
        "valid_count": valid_count,
    }


def track_points_forward_backward(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
    config: Mapping[str, object],
) -> Optional[dict]:
    """Run p0 → p1 → p0_back and keep forward/backward-consistent tracks."""
    if previous_gray.shape != current_gray.shape:
        raise ValueError("previous_gray and current_gray must have identical shapes")
    if previous_gray.ndim != 2 or previous_gray.dtype != np.uint8:
        raise ValueError("frames must be uint8 grayscale images")

    min_correspondences = int(config["min_correspondences"])

    p0 = np.asarray(points, dtype=np.float32)
    if p0.ndim == 2 and p0.shape[1:] == (2,):
        p0 = p0.reshape(-1, 1, 2)
    if p0.ndim != 3 or p0.shape[1:] != (1, 2):
        raise ValueError("points must have shape N×2 or N×1×2")

    finite_input = np.isfinite(p0.reshape(-1, 2)).all(axis=1)
    p0 = p0[finite_input]
    if len(p0) < min_correspondences:
        return None

    lk_parameters = {
        "winSize": (int(config["lk_win_size"]), int(config["lk_win_size"])),
        "maxLevel": int(config["lk_max_level"]),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        "minEigThreshold": float(config["min_eig_threshold"]),
    }

    p1, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        p0,
        None,
        **lk_parameters,
    )
    if p1 is None or forward_status is None:
        return None

    p1 = np.asarray(p1, dtype=np.float32).reshape(-1, 1, 2)
    forward_status = np.asarray(forward_status, dtype=np.uint8).reshape(-1)
    if forward_error is None:
        forward_error = np.full(len(p0), np.nan, dtype=np.float32)
    else:
        forward_error = np.asarray(forward_error, dtype=np.float32).reshape(-1)

    p1_flat = p1.reshape(-1, 2)
    height, width = current_gray.shape
    p1_inside = (
        (p1_flat[:, 0] >= 0)
        & (p1_flat[:, 0] < width)
        & (p1_flat[:, 1] >= 0)
        & (p1_flat[:, 1] < height)
    )
    forward_eligible = (
        (forward_status == 1)
        & np.isfinite(p1_flat).all(axis=1)
        & p1_inside
    )
    eligible_indices = np.flatnonzero(forward_eligible)
    if len(eligible_indices) < min_correspondences:
        return None

    p0_back = np.full_like(p0, np.nan)
    backward_status = np.zeros(len(p0), dtype=np.uint8)
    p0_back_subset, backward_status_subset, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        p1[eligible_indices],
        None,
        **lk_parameters,
    )
    if p0_back_subset is None or backward_status_subset is None:
        return None

    p0_back[eligible_indices] = np.asarray(
        p0_back_subset,
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    backward_status[eligible_indices] = np.asarray(
        backward_status_subset,
        dtype=np.uint8,
    ).reshape(-1)

    p0_flat = p0.reshape(-1, 2)
    p0_back_flat = p0_back.reshape(-1, 2)
    finite_coordinates = (
        np.isfinite(p0_flat).all(axis=1)
        & np.isfinite(p1_flat).all(axis=1)
        & np.isfinite(p0_back_flat).all(axis=1)
    )
    fb_error = np.full(len(p0), np.inf, dtype=np.float32)
    comparable = forward_eligible & (backward_status == 1) & finite_coordinates
    fb_error[comparable] = np.linalg.norm(
        p0_flat[comparable] - p0_back_flat[comparable],
        axis=1,
    )
    valid_mask = comparable & (fb_error <= float(config["fb_threshold"]))

    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < min_correspondences:
        return None

    return {
        "p0": p0,
        "p1": p1,
        "p0_back": p0_back,
        "forward_error": forward_error,
        "fb_error": fb_error,
        "valid_mask": valid_mask,
        "valid_p0": p0[valid_mask],
        "valid_p1": p1[valid_mask],
        "valid_count": valid_count,
    }


def _invalid_motion_result(
    motion_model: str,
    failure_reason: str,
    source_points: Optional[np.ndarray] = None,
    target_points: Optional[np.ndarray] = None,
    matrix: Optional[np.ndarray] = None,
    inlier_mask: Optional[np.ndarray] = None,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    model_parameters: Optional[Mapping[str, float]] = None,
    model_confidence: str = "invalid",
) -> dict:
    source_points = (
        np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
        if source_points is not None
        else np.empty((0, 2), dtype=np.float32)
    )
    target_points = (
        np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
        if target_points is not None
        else np.empty((0, 2), dtype=np.float32)
    )
    if inlier_mask is None:
        inlier_mask = np.zeros(len(source_points), dtype=bool)
    else:
        inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)

    parameters = dict(model_parameters or {})
    return {
        "valid": False,
        "motion_model": str(motion_model),
        "matrix": None if matrix is None else np.asarray(matrix, dtype=np.float32),
        "inlier_mask": inlier_mask,
        "inlier_count": int(inlier_count),
        "inlier_ratio": float(inlier_ratio),
        "model_dx": float(parameters.get("dx", np.nan)),
        "model_dy": float(parameters.get("dy", np.nan)),
        "model_rotation_deg": float(parameters.get("rotation_deg", np.nan)),
        "model_scale": float(parameters.get("scale", np.nan)),
        "model_confidence": str(model_confidence),
        "failure_reason": str(failure_reason),
        "source_points": source_points,
        "target_points": target_points,
    }


def _motion_model_parameters(matrix: np.ndarray) -> dict:
    """Return approximate dx, dy, rotation, scale and 2×2 determinant."""
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape == (2, 3):
        normalized = matrix
    elif matrix.shape == (3, 3):
        if abs(matrix[2, 2]) < 1e-12:
            raise ValueError("homography has near-zero H[2,2]")
        normalized = matrix / matrix[2, 2]
    else:
        raise ValueError("motion matrix must have shape 2×3 or 3×3")

    linear = normalized[:2, :2]
    determinant = float(np.linalg.det(linear))
    scale = float(np.sqrt(abs(determinant)))
    rotation_deg = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    dx = float(normalized[0, 2])
    dy = float(normalized[1, 2])

    return {
        "dx": dx,
        "dy": dy,
        "rotation_deg": rotation_deg,
        "scale": scale,
        "determinant": determinant,
    }


def estimate_camera_motion(
    p0: np.ndarray,
    p1: np.ndarray,
    config: Mapping[str, object],
    frame_shape: Tuple[int, ...],
) -> dict:
    """Estimate dominant previous → current motion with RANSAC."""
    source = np.asarray(p0, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    motion_model = str(config["motion_model"])
    min_correspondences = int(config["min_correspondences"])

    if source.shape != target.shape:
        raise ValueError("p0 and p1 must have identical shapes")

    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]

    if len(source) < min_correspondences:
        return _invalid_motion_result(
            motion_model,
            f"not enough finite correspondences: {len(source)} < {min_correspondences}",
            source,
            target,
        )

    ransac_threshold = float(config["ransac_threshold"])
    if motion_model == "affine_partial":
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
    elif motion_model == "affine_full":
        matrix, inlier_mask = cv2.estimateAffine2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
    elif motion_model == "homography":
        matrix, inlier_mask = cv2.findHomography(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=2000,
            confidence=0.99,
        )
    else:
        raise ValueError(
            "motion_model must be affine_partial, affine_full or homography"
        )

    if matrix is None or inlier_mask is None:
        return _invalid_motion_result(
            motion_model,
            f"{motion_model} RANSAC could not estimate a model",
            source,
            target,
        )

    matrix = np.asarray(matrix, dtype=np.float64)
    expected_shapes = {
        "affine_partial": (2, 3),
        "affine_full": (2, 3),
        "homography": (3, 3),
    }
    if matrix.shape != expected_shapes[motion_model]:
        return _invalid_motion_result(
            motion_model,
            f"invalid matrix shape: {matrix.shape}",
            source,
            target,
            matrix=matrix,
        )
    if not np.isfinite(matrix).all():
        return _invalid_motion_result(
            motion_model,
            "model matrix contains NaN or Inf",
            source,
            target,
            matrix=matrix,
        )

    if motion_model == "homography":
        if abs(matrix[2, 2]) < 1e-12:
            return _invalid_motion_result(
                motion_model,
                "degenerate: homography H[2,2] is near zero",
                source,
                target,
                matrix=matrix,
            )
        matrix = matrix / matrix[2, 2]

    inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(inlier_mask) != len(source):
        return _invalid_motion_result(
            motion_model,
            "RANSAC inlier mask has the wrong length",
            source,
            target,
            matrix=matrix,
        )

    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(np.mean(inlier_mask))

    try:
        parameters = _motion_model_parameters(matrix)
    except Exception as error:
        return _invalid_motion_result(
            motion_model,
            f"degenerate: {error}",
            source,
            target,
            matrix=matrix,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )

    preliminary_confidence = (
        "low"
        if inlier_ratio < float(config["low_confidence_inlier_ratio"])
        else "high"
    )

    def invalid_with_diagnostics(reason: str) -> dict:
        return _invalid_motion_result(
            motion_model,
            reason,
            source,
            target,
            matrix=matrix,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            model_parameters=parameters,
            model_confidence=preliminary_confidence,
        )

    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])

    if inlier_count < int(config["min_ransac_inliers"]):
        return invalid_with_diagnostics(
            f"not enough RANSAC inliers: {inlier_count} < {config['min_ransac_inliers']}"
        )
    if parameters["determinant"] <= 0:
        return invalid_with_diagnostics("degenerate: reflection")
    if not (
        float(config["min_model_scale"])
        <= parameters["scale"]
        <= float(config["max_model_scale"])
    ):
        return invalid_with_diagnostics(
            f"degenerate: scale={parameters['scale']:.2f}"
        )
    if abs(parameters["dx"]) > 0.5 * frame_width:
        return invalid_with_diagnostics("degenerate: dx too large")
    if abs(parameters["dy"]) > 0.5 * frame_height:
        return invalid_with_diagnostics("degenerate: dy too large")
    if inlier_ratio < float(config["min_inlier_ratio"]):
        return invalid_with_diagnostics(f"low inlier ratio {inlier_ratio:.2f}")

    model_confidence = (
        "low"
        if inlier_ratio < float(config["low_confidence_inlier_ratio"])
        else "high"
    )
    return {
        "valid": True,
        "motion_model": motion_model,
        "matrix": matrix.astype(np.float32),
        "inlier_mask": inlier_mask,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "model_dx": parameters["dx"],
        "model_dy": parameters["dy"],
        "model_rotation_deg": parameters["rotation_deg"],
        "model_scale": parameters["scale"],
        "model_confidence": model_confidence,
        "failure_reason": None,
        "source_points": source,
        "target_points": target,
    }



def _warped_validity_polygon(
    image_shape: Tuple[int, int],
    motion_matrix: np.ndarray,
) -> np.ndarray:
    """
    Build the valid warp region from transformed source corners.

    This replaces a second full-frame warp of an all-255 mask. The historical
    image still uses the same INTER_LINEAR warp; only validity construction is
    reduced to perspectiveTransform/transform + fillConvexPoly.
    """
    height, width = map(int, image_shape)
    corners = np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )

    matrix = np.asarray(motion_matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        transformed = cv2.transform(corners.reshape(1, -1, 2), matrix)[0]
    elif matrix.shape == (3, 3):
        transformed = cv2.perspectiveTransform(
            corners.reshape(1, -1, 2),
            matrix,
        )[0]
    else:
        raise ValueError("motion_matrix must have shape 2×3 or 3×3")

    if not np.isfinite(transformed).all():
        raise RuntimeError("Warped validity polygon contains NaN or Inf")

    polygon = np.rint(transformed).astype(np.int32)
    validity_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(validity_u8, polygon, 255, lineType=cv2.LINE_8)
    return validity_u8 > 127


def warp_grayscale_fast(
    previous_gray: np.ndarray,
    motion_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp one uint8 grayscale image and derive validity without mask warping."""
    if not isinstance(previous_gray, np.ndarray):
        raise ValueError("previous_gray must be a numpy array")
    if previous_gray.ndim != 2 or previous_gray.dtype != np.uint8:
        raise ValueError("previous_gray must be a uint8 grayscale image")
    if previous_gray.size == 0:
        raise ValueError("previous_gray must not be empty")

    motion_matrix = np.asarray(motion_matrix, dtype=np.float32)
    if motion_matrix.shape not in {(2, 3), (3, 3)}:
        raise ValueError("motion_matrix must have shape 2×3 or 3×3")
    if not np.isfinite(motion_matrix).all():
        raise ValueError("motion_matrix must contain only finite values")

    height, width = previous_gray.shape
    output_size = (width, height)

    if motion_matrix.shape == (2, 3):
        warped_previous = cv2.warpAffine(
            previous_gray,
            motion_matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped_previous = cv2.warpPerspective(
            previous_gray,
            motion_matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    validity_mask = _warped_validity_polygon(
        previous_gray.shape,
        motion_matrix,
    )
    if not np.any(validity_mask):
        raise RuntimeError("Warping produced an empty validity mask")

    return warped_previous, validity_mask


def _as_homography(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        homography = np.eye(3, dtype=np.float64)
        homography[:2, :] = matrix
    elif matrix.shape == (3, 3):
        homography = matrix.copy()
    else:
        raise ValueError("motion matrix must have shape 2×3 or 3×3")

    if not np.isfinite(homography).all():
        raise ValueError("motion matrix contains NaN or Inf")
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("homography H[2,2] is near zero")
    return (homography / homography[2, 2]).astype(np.float32)


def _natural_sort_key(path: Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )



class CompensationPipeline:
    """
    Stateful fixed-step compensation pipeline.

    Phase 3 changes are enabled for step=1, which is the production setting:
      - reuse LK tracks between frames;
      - full GFTT refresh every redetect_interval frames;
      - emergency replenishment when valid tracks drop below
        min_correspondences;
      - compose H(t-2->t) from consecutive pairwise models;
      - warp each historical image once;
      - reuse a preallocated (4,H,W) float32 tensor buffer.

    For step>1 the original fresh-detection path remains available. The
    default is step=1 because the released checkpoints were trained on
    [Warp(I_{t-2}), Warp(I_{t-1}), I_t], i.e. temporal_step=1.
    """

    def __init__(
        self,
        config: Mapping[str, object] = DEFAULT_COMPENSATION_CONFIG,
        step: int = 1,
    ) -> None:
        if int(step) <= 0:
            raise ValueError("step must be a positive integer")

        self.config = _validate_config(config)
        self.step = int(step)
        self.frames: Deque[Tuple[int, np.ndarray]] = deque(
            maxlen=2 * self.step + 1
        )
        self.matrices: Dict[int, np.ndarray] = {}
        self.stats = {}
        self.last_failure_reason: Optional[str] = None
        self._motion_cache: Dict[int, _MotionCacheEntry] = {}
        self._next_frame_index = 0
        self._motion_estimation_calls = 0
        self._processing_scale: Optional[float] = None
        self._frame_shape: Optional[Tuple[int, int]] = None

        self._tracked_points = np.empty((0, 1, 2), dtype=np.float32)
        self._frames_since_redetect = 0
        self._point_detection_calls = 0
        self._periodic_redetections = 0
        self._threshold_replenishments = 0
        self._tensor_buffer: Optional[np.ndarray] = None
        self._decoder_target_size: Optional[Tuple[int, int]] = None
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "emitted": 0,
            "warmup": 0,
            "invalid_compensation": 0,
            "point_detection_calls": 0,
            "periodic_redetections": 0,
            "threshold_replenishments": 0,
            "decoder_scaled_frames": 0,
        }

    @property
    def motion_estimation_calls(self) -> int:
        return self._motion_estimation_calls

    @property
    def point_detection_calls(self) -> int:
        return self._point_detection_calls

    def reset(self) -> None:
        self.frames.clear()
        self.matrices.clear()
        self._motion_cache.clear()
        self._reset_stats()
        self.last_failure_reason = None
        self._next_frame_index = 0
        self._motion_estimation_calls = 0
        self._processing_scale = None
        self._frame_shape = None
        self._tracked_points = np.empty((0, 1, 2), dtype=np.float32)
        self._frames_since_redetect = 0
        self._point_detection_calls = 0
        self._periodic_redetections = 0
        self._threshold_replenishments = 0
        self._tensor_buffer = None
        self._decoder_target_size = None

    def _frame_at(self, frame_index: int) -> np.ndarray:
        for cached_index, gray in self.frames:
            if cached_index == frame_index:
                return gray
        raise KeyError(f"frame {frame_index} is not present in the rolling buffer")

    def _detect_points(self, gray: np.ndarray) -> np.ndarray:
        self._point_detection_calls += 1
        self.stats["point_detection_calls"] = self._point_detection_calls
        return detect_points_grid(gray, self.config)

    def _replenish_points(
        self,
        gray: np.ndarray,
        existing_points: np.ndarray,
    ) -> np.ndarray:
        self._point_detection_calls += 1
        self.stats["point_detection_calls"] = self._point_detection_calls
        return replenish_points_grid(
            gray,
            existing_points,
            self.config,
        )

    def _estimate_pair_motion_fresh(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> _MotionCacheEntry:
        """Original per-pair detector path, retained for step>1."""
        self._motion_estimation_calls += 1

        if previous_gray.shape != current_gray.shape:
            return _MotionCacheEntry(
                valid=False,
                matrix=None,
                inlier_ratio=0.0,
                failure_reason="frame sizes do not match after preprocessing",
            )

        try:
            points = self._detect_points(previous_gray)
            if len(points) < int(self.config["min_correspondences"]):
                return _MotionCacheEntry(
                    valid=False,
                    matrix=None,
                    inlier_ratio=0.0,
                    failure_reason=(
                        f"insufficient points: {len(points)} < "
                        f"{self.config['min_correspondences']}"
                    ),
                )

            tracking = track_points_forward_backward(
                previous_gray,
                current_gray,
                points,
                self.config,
            )
            if tracking is None:
                return _MotionCacheEntry(
                    valid=False,
                    matrix=None,
                    inlier_ratio=0.0,
                    failure_reason="insufficient valid forward/backward LK tracks",
                )

            motion = estimate_camera_motion(
                tracking["valid_p0"],
                tracking["valid_p1"],
                self.config,
                current_gray.shape,
            )
            if not motion["valid"]:
                return _MotionCacheEntry(
                    valid=False,
                    matrix=None,
                    inlier_ratio=float(motion["inlier_ratio"]),
                    failure_reason=str(motion["failure_reason"]),
                )

            return _MotionCacheEntry(
                valid=True,
                matrix=_as_homography(motion["matrix"]),
                inlier_ratio=float(motion["inlier_ratio"]),
                failure_reason=None,
            )
        except (cv2.error, RuntimeError, ValueError) as error:
            return _MotionCacheEntry(
                valid=False,
                matrix=None,
                inlier_ratio=0.0,
                failure_reason=f"compensation failed: {error}",
            )

    def _select_current_tracks(
        self,
        tracking: Optional[dict],
        motion: Optional[dict],
    ) -> np.ndarray:
        if tracking is None:
            return np.empty((0, 1, 2), dtype=np.float32)

        current_points = np.asarray(
            tracking["valid_p1"],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        if motion is None:
            return current_points

        inlier_mask = np.asarray(
            motion.get("inlier_mask", []),
            dtype=bool,
        ).reshape(-1)
        if len(inlier_mask) == len(current_points) and np.any(inlier_mask):
            return current_points[inlier_mask]
        return current_points

    def _refresh_tracks_for_next_frame(
        self,
        current_gray: np.ndarray,
        current_tracks: np.ndarray,
    ) -> bool:
        """Return True when GFTT was invoked on the current frame."""
        self._frames_since_redetect += 1
        redetect_interval = int(self.config["redetect_interval"])
        valid_count = int(len(current_tracks))

        if self._frames_since_redetect >= redetect_interval:
            self._tracked_points = self._detect_points(current_gray)
            self._frames_since_redetect = 0
            self._periodic_redetections += 1
            self.stats["periodic_redetections"] = self._periodic_redetections
            return True

        if valid_count < int(self.config["min_correspondences"]):
            self._tracked_points = self._replenish_points(
                current_gray,
                current_tracks,
            )
            self._frames_since_redetect = 0
            self._threshold_replenishments += 1
            self.stats["threshold_replenishments"] = (
                self._threshold_replenishments
            )
            return True

        self._tracked_points = current_tracks
        return False

    def _estimate_pair_motion_stateful(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> _MotionCacheEntry:
        """Reuse existing tracks and detect new points only when required."""
        self._motion_estimation_calls += 1

        if previous_gray.shape != current_gray.shape:
            self._tracked_points = self._detect_points(current_gray)
            self._frames_since_redetect = 0
            return _MotionCacheEntry(
                valid=False,
                matrix=None,
                inlier_ratio=0.0,
                failure_reason="frame sizes do not match after preprocessing",
            )

        try:
            if len(self._tracked_points) < 4:
                self._tracked_points = self._detect_points(previous_gray)
                self._frames_since_redetect = 0

            tracking = track_points_forward_backward_relaxed(
                previous_gray,
                current_gray,
                self._tracked_points,
                self.config,
            )

            valid_count = (
                int(tracking["valid_count"])
                if tracking is not None
                else 0
            )
            motion = None
            if (
                tracking is not None
                and valid_count >= int(self.config["min_correspondences"])
            ):
                motion = estimate_camera_motion(
                    tracking["valid_p0"],
                    tracking["valid_p1"],
                    self.config,
                    current_gray.shape,
                )

            current_tracks = self._select_current_tracks(
                tracking,
                motion,
            )
            self._refresh_tracks_for_next_frame(
                current_gray,
                current_tracks,
            )

            if tracking is None:
                return _MotionCacheEntry(
                    valid=False,
                    matrix=None,
                    inlier_ratio=0.0,
                    failure_reason="forward/backward LK produced fewer than 4 tracks",
                )

            if valid_count < int(self.config["min_correspondences"]):
                return _MotionCacheEntry(
                    valid=False,
                    matrix=None,
                    inlier_ratio=0.0,
                    failure_reason=(
                        f"valid LK correspondences dropped to {valid_count} < "
                        f"{self.config['min_correspondences']}; points replenished"
                    ),
                )

            assert motion is not None
            if not motion["valid"]:
                return _MotionCacheEntry(
                    valid=False,
                    matrix=None,
                    inlier_ratio=float(motion["inlier_ratio"]),
                    failure_reason=str(motion["failure_reason"]),
                )

            return _MotionCacheEntry(
                valid=True,
                matrix=_as_homography(motion["matrix"]),
                inlier_ratio=float(motion["inlier_ratio"]),
                failure_reason=None,
            )
        except (cv2.error, RuntimeError, ValueError) as error:
            self._tracked_points = self._detect_points(current_gray)
            self._frames_since_redetect = 0
            return _MotionCacheEntry(
                valid=False,
                matrix=None,
                inlier_ratio=0.0,
                failure_reason=f"compensation failed: {error}",
            )

    def _estimate_pair_motion(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> _MotionCacheEntry:
        if self.step == 1:
            return self._estimate_pair_motion_stateful(
                previous_gray,
                current_gray,
            )
        return self._estimate_pair_motion_fresh(
            previous_gray,
            current_gray,
        )

    def _prune_cache(self, current_index: int) -> None:
        oldest_allowed = current_index - 2 * self.step
        for key in list(self._motion_cache):
            if key < oldest_allowed:
                del self._motion_cache[key]
                self.matrices.pop(key, None)

    def _invalid_sample(
        self,
        frame_index: int,
        skip_reason: str,
        inlier_ratio: float = 0.0,
        failure_reason: Optional[str] = None,
    ) -> CompensatedSample:
        self.stats[skip_reason] += 1
        self.last_failure_reason = failure_reason
        return CompensatedSample(
            tensor=_EMPTY_TENSOR.copy(),
            frame_index=frame_index,
            valid=False,
            inlier_ratio=float(inlier_ratio),
            skip_reason=skip_reason,
        )

    def _build_tensor(
        self,
        warped_oldest: np.ndarray,
        warped_middle: np.ndarray,
        current_gray: np.ndarray,
        validity: np.ndarray,
    ) -> np.ndarray:
        height, width = current_gray.shape
        shape = (4, height, width)
        if self._tensor_buffer is None or self._tensor_buffer.shape != shape:
            self._tensor_buffer = np.empty(shape, dtype=np.float32)

        scale = np.float32(1.0 / 255.0)
        np.multiply(
            warped_oldest,
            scale,
            out=self._tensor_buffer[0],
            casting="unsafe",
        )
        np.multiply(
            warped_middle,
            scale,
            out=self._tensor_buffer[1],
            casting="unsafe",
        )
        np.multiply(
            current_gray,
            scale,
            out=self._tensor_buffer[2],
            casting="unsafe",
        )
        np.copyto(
            self._tensor_buffer[3],
            validity,
            casting="unsafe",
        )
        return self._tensor_buffer

    def process_frame(self, frame: np.ndarray) -> CompensatedSample:
        """Consume one BGR frame and return one index-aligned sample."""
        frame_index = self._next_frame_index
        self._next_frame_index += 1

        gray, scale = preprocess_frame(
            frame,
            int(self.config["max_width"]),
            gray_first=bool(self.config["preprocess_gray_first"]),
        )
        if self._frame_shape is None:
            self._frame_shape = gray.shape
            self._processing_scale = scale
        self.frames.append((frame_index, gray))

        if frame_index == 0 and self.step == 1:
            self._tracked_points = self._detect_points(gray)
            self._frames_since_redetect = 0

        current_motion: Optional[_MotionCacheEntry] = None
        if frame_index >= self.step:
            previous_gray = self._frame_at(frame_index - self.step)
            current_motion = self._estimate_pair_motion(
                previous_gray,
                gray,
            )
            self._motion_cache[frame_index] = current_motion
            if current_motion.valid and current_motion.matrix is not None:
                self.matrices[frame_index] = current_motion.matrix

        self._prune_cache(frame_index)

        if frame_index < 2 * self.step:
            return self._invalid_sample(
                frame_index=frame_index,
                skip_reason="warmup",
                failure_reason=None,
            )

        previous_target_index = frame_index - self.step
        previous_motion = self._motion_cache.get(previous_target_index)
        if current_motion is None:
            current_motion = self._motion_cache.get(frame_index)

        entries = (previous_motion, current_motion)
        available_ratios = [
            entry.inlier_ratio for entry in entries if entry is not None
        ]
        weakest_ratio = min(available_ratios, default=0.0)

        if previous_motion is None or current_motion is None:
            return self._invalid_sample(
                frame_index=frame_index,
                skip_reason="invalid_compensation",
                inlier_ratio=weakest_ratio,
                failure_reason="required cached motion is missing",
            )

        invalid_entry = next(
            (
                entry
                for entry in entries
                if not entry.valid or entry.matrix is None
            ),
            None,
        )
        if invalid_entry is not None:
            return self._invalid_sample(
                frame_index=frame_index,
                skip_reason="invalid_compensation",
                inlier_ratio=weakest_ratio,
                failure_reason=invalid_entry.failure_reason,
            )

        assert previous_motion.matrix is not None
        assert current_motion.matrix is not None

        try:
            # This composition was already mathematically correct in the source
            # module. Phase 3 keeps it and avoids any image-to-image double warp.
            composed_matrix = current_motion.matrix @ previous_motion.matrix
            composed_matrix = _as_homography(composed_matrix)

            oldest_gray = self._frame_at(frame_index - 2 * self.step)
            middle_gray = self._frame_at(frame_index - self.step)

            warped_oldest, validity_oldest = warp_grayscale_fast(
                oldest_gray,
                composed_matrix,
            )
            warped_middle, validity_middle = warp_grayscale_fast(
                middle_gray,
                current_motion.matrix,
            )
            validity = np.logical_and(validity_oldest, validity_middle)
            if not np.any(validity):
                raise RuntimeError("intersection of validity masks is empty")

            tensor = self._build_tensor(
                warped_oldest,
                warped_middle,
                gray,
                validity,
            )
        except (cv2.error, RuntimeError, ValueError) as error:
            return self._invalid_sample(
                frame_index=frame_index,
                skip_reason="invalid_compensation",
                inlier_ratio=weakest_ratio,
                failure_reason=f"tensor construction failed: {error}",
            )

        self.stats["emitted"] += 1
        self.last_failure_reason = None
        return CompensatedSample(
            tensor=tensor,
            frame_index=frame_index,
            valid=True,
            inlier_ratio=float(weakest_ratio),
            skip_reason=None,
        )

    def stream_video(self, video_path: str | Path) -> Iterator[CompensatedSample]:
        """Reset and stream samples, requesting decoder downscale when supported."""
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reset()

        if bool(self.config["request_decoder_scaling"]):
            self._decoder_target_size = _request_decoder_scaling(
                capture,
                int(self.config["max_width"]),
            )

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if (
                    self._decoder_target_size is not None
                    and source_width > int(self.config["max_width"])
                    and tuple(frame.shape[1::-1]) == tuple(self._decoder_target_size)
                ):
                    self.stats["decoder_scaled_frames"] += 1

                yield self.process_frame(frame)
        finally:
            capture.release()

    def stream_frames(self, folder: str | Path) -> Iterator[CompensatedSample]:
        """Reset the pipeline and stream naturally sorted image files."""
        root = Path(folder)
        if not root.is_dir():
            raise NotADirectoryError(f"Frame folder does not exist: {root}")

        paths = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
            ),
            key=_natural_sort_key,
        )
        if not paths:
            raise FileNotFoundError(f"No supported image files found in: {root}")

        self.reset()
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"OpenCV could not read frame: {path}")
            yield self.process_frame(frame)


__all__ = [
    "CompensatedSample",
    "CompensationPipeline",
    "DEFAULT_COMPENSATION_CONFIG",
    "detect_points_grid",
    "replenish_points_grid",
    "estimate_camera_motion",
    "preprocess_frame",
    "track_points_forward_backward",
    "track_points_forward_backward_relaxed",
    "warp_grayscale_fast",
]
