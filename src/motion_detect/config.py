"""YAML configuration loading.

All paths and tunables come from a YAML config (see configs/default.yaml and
configs/paths.example.yaml) or CLI arguments. Defaults below reproduce the
research-notebook values exactly; loading a config only overrides what the
file sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .compensation import DEFAULT_COMPENSATION_CONFIG
from .dataset import (
    DEFAULT_MAX_WIDTH,
    IGNORE_BAND_MIN_SIDE,
    IGNORE_BAND_PX,
    TRAIN_CROP_SIZE,
    TRAIN_GROUP_RATIOS,
)
from .hard_negatives import (
    HARD_NEGATIVE_FRACTION,
    HARD_NEGATIVE_INLIER_MARGIN_SCALE,
    HARD_NEGATIVE_INLIER_PRIORITY_BOOST,
    HARD_NEGATIVE_REMINE_EVERY,
    HARD_NEGATIVE_REVIEW_COUNT,
)
from .train import (
    ALLOW_TRAIN_WITHOUT_EMPTY,
    ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES,
    LOSS_POS_WEIGHT,
    LOSS_POS_WEIGHT_NO_EMPTY,
    MIN_EMPTY_FRACTION_WARNING,
)

# Detection/post-processing parameters fixed for the production candidate
# (best_15.ipynb, PRODUCTION_DETECTION_PARAMS).
DEFAULT_DETECTION_PARAMS: dict[str, Any] = {
    "diff_threshold": 24,  # legacy absdiff fallback only
    "area_min": 12,
    "area_max": 856,
    "bbox_padding": 4,
    "max_age": 19,
    "min_hits": 5,
    "iou_thresh": 0.128,
    "max_dist": 52,
    "max_corners": 343,  # legacy absdiff fallback only
    "lk_winsize": 17,  # legacy absdiff fallback only
    "inlier_thresh": 0.578,
    "apply_platform_suppression": False,
}

# Evaluation clip sets used in the notebooks.
VAL_CLIPS = [3, 28, 38, 48, 50]
TEST_CLIPS = [2, 6, 7, 8, 9, 15, 16, 18, 35, 41]

# Calibration grid (best_15.ipynb).
CALIBRATION_UNET_THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98]
CALIBRATION_AREA_MIN_VALUES = [8, 12, 16]
CALIBRATION_AREA_MAX_VALUES = [640, 856, 1100]
CALIBRATION_BBOX_PADDING_VALUES = [2, 4, 6]
CALIBRATION_PRECISION_FLOOR = 0.30


@dataclass(frozen=True)
class AppConfig:
    """Typed view over the YAML config. All fields have notebook defaults."""

    # Paths (filled from paths.yaml / CLI).
    videos_dir: Optional[Path] = None
    annotations_dir: Optional[Path] = None
    dataset_root: Optional[Path] = None
    checkpoints_dir: Optional[Path] = None
    results_dir: Optional[Path] = None

    # Processing / compensation.
    max_width: int = DEFAULT_MAX_WIDTH
    compensation_step: int = 1
    return_size: tuple[int, int] = (1280, 720)
    compensation: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_COMPENSATION_CONFIG)
    )

    # Detection / tracker / thresholds.
    detection_params: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_DETECTION_PARAMS)
    )
    unet_threshold: float = 0.95
    bbox_iou_threshold: float = 0.20
    pixel_threshold: float = 0.5
    apply_validity_mask: bool = True
    apply_platform_suppression: bool = False

    # Training.
    epochs: int = 25
    batch_size: int = 4
    learning_rate: float = 1e-3
    seed: int = 42
    train_crop_size: int = TRAIN_CROP_SIZE
    group_ratios: Mapping[str, float] = field(
        default_factory=lambda: dict(TRAIN_GROUP_RATIOS)
    )
    pos_weight: float = LOSS_POS_WEIGHT
    pos_weight_no_empty: float = LOSS_POS_WEIGHT_NO_EMPTY
    allow_train_without_empty: bool = ALLOW_TRAIN_WITHOUT_EMPTY
    allow_train_without_hard_negatives: bool = ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES
    min_empty_fraction_warning: float = MIN_EMPTY_FRACTION_WARNING
    ignore_band_px: int = IGNORE_BAND_PX
    ignore_band_min_side: int = IGNORE_BAND_MIN_SIDE

    # Dataset generation / split.
    samples_per_clip: int = 60
    stride: int = 5
    split_seed: int = 42
    train_clips_count: int = 35
    val_clips_count: int = 5
    num_clips: int = 50

    # Hard-negative mining.
    hard_negative_fraction: float = HARD_NEGATIVE_FRACTION
    hard_negative_review_count: int = HARD_NEGATIVE_REVIEW_COUNT
    hard_negative_remine_every: int = HARD_NEGATIVE_REMINE_EVERY
    inlier_priority_boost: float = HARD_NEGATIVE_INLIER_PRIORITY_BOOST
    inlier_margin_scale: float = HARD_NEGATIVE_INLIER_MARGIN_SCALE

    # Calibration.
    calibration_clip_ids: tuple[int, ...] = tuple(TEST_CLIPS)
    calibration_unet_thresholds: tuple[float, ...] = tuple(CALIBRATION_UNET_THRESHOLDS)
    calibration_area_min_values: tuple[float, ...] = tuple(CALIBRATION_AREA_MIN_VALUES)
    calibration_area_max_values: tuple[float, ...] = tuple(CALIBRATION_AREA_MAX_VALUES)
    calibration_bbox_padding_values: tuple[int, ...] = tuple(CALIBRATION_BBOX_PADDING_VALUES)
    calibration_precision_floor: float = CALIBRATION_PRECISION_FLOOR

    # Evaluation clip sets.
    val_clips: tuple[int, ...] = tuple(VAL_CLIPS)
    test_clips: tuple[int, ...] = tuple(TEST_CLIPS)


def _as_path(value: Any) -> Optional[Path]:
    return None if value in (None, "") else Path(str(value))


def _deep_merge(base: dict, overlay: Mapping) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None,
    defaults_path: str | Path | None = None,
) -> AppConfig:
    """Load a YAML config file; None returns the pure defaults.

    When defaults_path is given, it is loaded first and the user file only
    overrides the keys it sets (deep-merged). Typical usage:
    load_config("configs/paths.yaml", defaults_path="configs/default.yaml").
    """
    if path is None and defaults_path is None:
        return AppConfig()
    raw: dict[str, Any] = {}
    if defaults_path is not None:
        loaded = yaml.safe_load(Path(defaults_path).read_text(encoding="utf-8"))
        if loaded is not None:
            raw = _deep_merge(raw, loaded)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, Mapping):
                raise ValueError(f"Config root must be a mapping: {path}")
            raw = _deep_merge(raw, loaded)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config root must be a mapping: {path}")

    kwargs: dict[str, Any] = {}
    paths = raw.get("paths", {})
    for key in ("videos_dir", "annotations_dir", "dataset_root", "checkpoints_dir", "results_dir"):
        if key in paths:
            kwargs[key] = _as_path(paths[key])

    processing = raw.get("processing", {})
    if "max_width" in processing:
        kwargs["max_width"] = int(processing["max_width"])
    if "compensation_step" in processing:
        kwargs["compensation_step"] = int(processing["compensation_step"])
    if "return_size" in processing:
        w, h = processing["return_size"]
        kwargs["return_size"] = (int(w), int(h))
    if "compensation" in processing:
        merged = dict(DEFAULT_COMPENSATION_CONFIG)
        merged.update(dict(processing["compensation"]))
        kwargs["compensation"] = merged

    det = raw.get("detection", {})
    if "params" in det:
        merged = dict(DEFAULT_DETECTION_PARAMS)
        merged.update(dict(det["params"]))
        kwargs["detection_params"] = merged
    for key in ("unet_threshold", "bbox_iou_threshold", "pixel_threshold"):
        if key in det:
            kwargs[key] = float(det[key])
    if "apply_validity_mask" in det:
        kwargs["apply_validity_mask"] = bool(det["apply_validity_mask"])
    if "apply_platform_suppression" in det:
        kwargs["apply_platform_suppression"] = bool(det["apply_platform_suppression"])

    tr = raw.get("training", {})
    for key in (
        "epochs", "batch_size", "seed", "train_crop_size", "samples_per_clip",
        "stride", "split_seed", "train_clips_count", "val_clips_count", "num_clips",
        "ignore_band_px", "ignore_band_min_side",
    ):
        if key in tr:
            kwargs[key] = int(tr[key])
    for key in ("learning_rate", "pos_weight", "pos_weight_no_empty", "min_empty_fraction_warning"):
        if key in tr:
            kwargs[key] = float(tr[key])
    for key in ("allow_train_without_empty", "allow_train_without_hard_negatives"):
        if key in tr:
            kwargs[key] = bool(tr[key])
    if "group_ratios" in tr:
        kwargs["group_ratios"] = {k: float(v) for k, v in dict(tr["group_ratios"]).items()}

    hn = raw.get("hard_negatives", {})
    for key in ("hard_negative_review_count", "hard_negative_remine_every"):
        if key in hn:
            kwargs[key] = int(hn[key])
    for key in ("hard_negative_fraction", "inlier_priority_boost", "inlier_margin_scale"):
        if key in hn:
            kwargs[key] = float(hn[key])

    cal = raw.get("calibration", {})
    if "clip_ids" in cal:
        kwargs["calibration_clip_ids"] = tuple(int(v) for v in cal["clip_ids"])
    for key in (
        "unet_thresholds", "area_min_values", "area_max_values",
        "bbox_padding_values",
    ):
        if key in cal:
            kwargs["calibration_" + key] = tuple(float(v) for v in cal[key])
    if "precision_floor" in cal:
        kwargs["calibration_precision_floor"] = float(cal["precision_floor"])

    ev = raw.get("evaluation", {})
    if "val_clips" in ev:
        kwargs["val_clips"] = tuple(int(v) for v in ev["val_clips"])
    if "test_clips" in ev:
        kwargs["test_clips"] = tuple(int(v) for v in ev["test_clips"])

    return AppConfig(**kwargs)
