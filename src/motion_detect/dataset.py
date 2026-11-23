"""Training dataset: GT-bbox-derived masks and dataset generation.

Masks are rasterized from ground-truth bounding boxes (never pseudo-labels).
GT annotation files store boxes as (y1, x1, y2, x2); internally everything is
(x1, y1, x2, y2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .compensation import CompensationPipeline, DEFAULT_COMPENSATION_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_MAX_WIDTH = 640
TRAIN_CROP_SIZE = 256
TRAIN_GROUP_RATIOS = {
    "positive": 0.60,
    "hard_negative": 0.25,
    "random_negative": 0.15,
}
IGNORE_BAND_PX = 2
IGNORE_BAND_MIN_SIDE = 6

class MotionDataset(Dataset):
    """
    Dataset with three explicit sampling groups:

      positive        — has GT bbox;
      hard_negative   — empty target, but the sample is marked in groups/*.txt;
      random_negative — ordinary frame without GT.

    Layout:
      data_dir/
        frames/       <sid>_t2.png, <sid>_t1.png, <sid>_t0.png
        masks/        <sid>_mask.png
        weights/      <sid>_weight.png
        bboxes/       <sid>.npy
        bboxes_src/   <sid>.npy
        groups/       <sid>.txt  # only for hard_negative

    WeightedRandomSampler receives the total 60/25/15 probability mass
    instead of mixing mined false positives into ordinary empty frames.
    """
    CROP = TRAIN_CROP_SIZE
    VALID_GROUPS = ("positive", "hard_negative", "random_negative")

    def __init__(
        self,
        data_dir,
        augment=True,
        group_ratios=TRAIN_GROUP_RATIOS,
    ):
        self.data_dir = Path(data_dir)
        self.augment = bool(augment)
        self.group_ratios = (
            None if group_ratios is None else dict(group_ratios)
        )

        masks_dir = self.data_dir / "masks"
        self.ids = sorted([
            f.stem.replace("_mask", "")
            for f in masks_dir.glob("*_mask.png")
        ])

        self.sample_group_by_id = {}
        self.ids_with_gt = []
        self.ids_hard_negative = []
        self.ids_random_negative = []

        for sid in self.ids:
            group = self._infer_group(sid)
            self.sample_group_by_id[sid] = group
            if group == "positive":
                self.ids_with_gt.append(sid)
            elif group == "hard_negative":
                self.ids_hard_negative.append(sid)
            else:
                self.ids_random_negative.append(sid)

        # Backward-compatible alias: ids_empty now means only random negatives.
        self.ids_empty = self.ids_random_negative

        logger.info(
            f"Dataset '{data_dir}': {len(self.ids)} samples | "
            f"positive={len(self.ids_with_gt)} | "
            f"hard_negative={len(self.ids_hard_negative)} | "
            f"random_negative={len(self.ids_random_negative)}"
        )

    def __len__(self):
        return len(self.ids)

    def _has_boxes(self, sid):
        path = self.data_dir / "bboxes" / f"{sid}.npy"
        if not path.exists():
            return False
        boxes = np.load(str(path), allow_pickle=False)
        return boxes.size > 0

    def _explicit_group(self, sid):
        path = self.data_dir / "groups" / f"{sid}.txt"
        if not path.exists():
            return None
        group = path.read_text(encoding="utf-8").strip().lower()
        if group not in self.VALID_GROUPS:
            raise ValueError(
                f"{path}: unknown sample group '{group}'. "
                f"Expected one of {self.VALID_GROUPS}."
            )
        return group

    def _infer_group(self, sid):
        explicit = self._explicit_group(sid)
        if explicit is not None:
            if explicit == "hard_negative" and self._has_boxes(sid):
                raise ValueError(
                    f"{sid}: hard_negative must have an empty bbox array"
                )
            return explicit
        return "positive" if self._has_boxes(sid) else "random_negative"

    def group_counts(self):
        return {
            "positive": len(self.ids_with_gt),
            "hard_negative": len(self.ids_hard_negative),
            "random_negative": len(self.ids_random_negative),
        }

    def sampling_weights(
        self,
        allow_single_class=False,
        allow_missing_hard_negative=False,
    ):
        """
        Per-sample weights whose total probability mass follows group_ratios.

        Missing groups are errors by default, so the requested 60/25/15 split
        cannot silently collapse back to 70/30.
        """
        if self.group_ratios is None:
            return np.ones(len(self.ids), dtype=np.float64)

        counts = self.group_counts()
        if counts["positive"] == 0:
            raise RuntimeError(
                f"Dataset '{self.data_dir}' contains no positive samples."
            )
        if counts["random_negative"] == 0 and not allow_single_class:
            raise RuntimeError(
                f"Dataset '{self.data_dir}' contains no random negatives."
            )
        if (
            counts["hard_negative"] == 0
            and not allow_missing_hard_negative
        ):
            raise RuntimeError(
                f"Dataset '{self.data_dir}' contains no approved hard negatives. "
                "Run mining -> review -> approval first, or explicitly set "
                "ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES=True for a diagnostic run."
            )

        requested = {
            group: float(self.group_ratios.get(group, 0.0))
            for group in self.VALID_GROUPS
        }
        if any(value < 0 for value in requested.values()):
            raise ValueError(f"Negative group ratio: {requested}")

        active = {
            group: ratio
            for group, ratio in requested.items()
            if counts[group] > 0 and ratio > 0
        }
        total = sum(active.values())
        if total <= 0:
            raise RuntimeError("No active sampler groups")
        active = {group: ratio / total for group, ratio in active.items()}

        weights_by_group = {
            group: (
                active.get(group, 0.0) / counts[group]
                if counts[group] > 0 else 0.0
            )
            for group in self.VALID_GROUPS
        }

        return np.asarray(
            [
                weights_by_group[self.sample_group_by_id[sid]]
                for sid in self.ids
            ],
            dtype=np.float64,
        )

    def _load(self, sid):
        def read_gray(path):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            return image

        t2 = read_gray(self.data_dir / "frames" / f"{sid}_t2.png")
        t1 = read_gray(self.data_dir / "frames" / f"{sid}_t1.png")
        t0 = read_gray(self.data_dir / "frames" / f"{sid}_t0.png")
        mask = read_gray(self.data_dir / "masks" / f"{sid}_mask.png")

        bbox_path = self.data_dir / "bboxes" / f"{sid}.npy"
        if bbox_path.exists():
            boxes = np.asarray(
                np.load(str(bbox_path), allow_pickle=False),
                dtype=np.float32,
            ).reshape(-1, 4)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        weight_path = self.data_dir / "weights" / f"{sid}_weight.png"
        if weight_path.exists():
            weight = read_gray(weight_path)
        else:
            # Backward compatibility for the pre-ignore-band dataset.
            _, weight = gt_boxes_to_mask(
                boxes,
                src_shape=mask.shape,
                dst_shape=mask.shape,
                ignore_px=IGNORE_BAND_PX,
                min_side_for_band=IGNORE_BAND_MIN_SIDE,
            )

        if weight.shape != mask.shape:
            raise ValueError(
                f"{sid}: weight shape {weight.shape} != mask shape {mask.shape}"
            )

        weight = (weight > 0).astype(np.uint8)
        weight[mask > 127] = 1
        return t2, t1, t0, mask, weight, boxes

    def _pad(self, arr, height, width):
        crop = self.CROP
        if height < crop or width < crop:
            arr = np.pad(
                arr,
                (
                    (0, max(0, crop - height)),
                    (0, max(0, crop - width)),
                ),
                mode="constant",
            )
        return arr

    def _crop(self, t2, t1, t0, mask, weight, height, width, boxes):
        crop = self.CROP

        if np.random.rand() < 0.5 and len(boxes) > 0:
            box = boxes[np.random.randint(len(boxes))]
            cx = int(round((box[0] + box[2]) / 2))
            cy = int(round((box[1] + box[3]) / 2))
            cx += np.random.randint(-32, 33)
            cy += np.random.randint(-32, 33)

            x0 = int(np.clip(cx - crop // 2, 0, max(0, width - crop)))
            y0 = int(np.clip(cy - crop // 2, 0, max(0, height - crop)))
        else:
            x0 = np.random.randint(0, max(1, width - crop + 1))
            y0 = np.random.randint(0, max(1, height - crop + 1))

        region = np.s_[y0:y0 + crop, x0:x0 + crop]
        return (
            t2[region],
            t1[region],
            t0[region],
            mask[region],
            weight[region],
        )

    def __getitem__(self, idx):
        sid = self.ids[idx]
        t2, t1, t0, mask, weight, boxes = self._load(sid)

        height, width = t0.shape
        t2 = self._pad(t2, height, width)
        t1 = self._pad(t1, height, width)
        t0 = self._pad(t0, height, width)
        mask = self._pad(mask, height, width)
        # Padding is outside the real frame and is ignored by BCE.
        weight = self._pad(weight, height, width)

        height, width = t0.shape
        t2, t1, t0, mask, weight = self._crop(
            t2, t1, t0, mask, weight, height, width, boxes
        )

        if self.augment and np.random.rand() > 0.5:
            t2 = t2[:, ::-1].copy()
            t1 = t1[:, ::-1].copy()
            t0 = t0[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            weight = weight[:, ::-1].copy()

        if self.augment:
            brightness = np.random.uniform(0.8, 1.2)
            t0 = np.clip(
                t0.astype(np.float32) * brightness,
                0,
                255,
            ).astype(np.uint8)

        x = np.stack([t2, t1, t0], axis=0).astype(np.float32) / 255.0
        y = (mask[None] > 127).astype(np.float32)
        w = (weight[None] > 0).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(w),
        )

# ================================================================

# ================================================================
# ================================================================

def _video_source_shape(video_path):
    """Return (H, W) of the source video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Could not determine video size: {video_path}"
        )
    return height, width

def scale_gt_boxes(boxes_src, src_shape, dst_shape):
    """
    GT bbox in (x1,y1,x2,y2) format with source-video coordinates
    -> bbox in processed-frame coordinates.
    """
    src_h, src_w = map(int, src_shape)
    dst_h, dst_w = map(int, dst_shape)

    boxes = np.asarray(boxes_src, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    scaled = boxes.copy()
    scaled[:, [0, 2]] *= float(dst_w) / float(src_w)
    scaled[:, [1, 3]] *= float(dst_h) / float(src_h)

    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, dst_w - 1)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, dst_h - 1)

    valid = (
        (scaled[:, 2] > scaled[:, 0])
        & (scaled[:, 3] > scaled[:, 1])
    )
    return np.ascontiguousarray(scaled[valid], dtype=np.float32)

def gt_boxes_to_mask(
    boxes_src,
    src_shape,
    dst_shape,
    ignore_px=IGNORE_BAND_PX,
    min_side_for_band=IGNORE_BAND_MIN_SIDE,
):
    """
    GT bbox in source coordinates -> (mask, weight)
    in processing coordinates.

    mask:   255 inside GT bbox, 0 on background.
    weight: 1 = pixel participates in BCE, 0 = ignore-band.

    No ignore-band is created for small boxes, so the positive signal of
    target objects a few pixels in size is not weakened.
    The inside of a bbox always has weight=1.
    """
    dst_h, dst_w = map(int, dst_shape)
    boxes_dst = scale_gt_boxes(boxes_src, src_shape, dst_shape)

    mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    weight = np.ones((dst_h, dst_w), dtype=np.uint8)

    ignore_px = max(0, int(ignore_px))
    min_side_for_band = float(min_side_for_band)

    for x1, y1, x2, y2 in boxes_dst:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        cv2.rectangle(mask, p1, p2, 255, thickness=-1)

        if (
            ignore_px > 0
            and min(x2 - x1, y2 - y1) >= min_side_for_band
        ):
            cv2.rectangle(
                weight,
                p1,
                p2,
                0,
                thickness=ignore_px * 2,
            )

    # The positive interior is never ignored.
    weight[mask > 0] = 1
    return mask, weight

def generate_dataset(
    video_path,
    gt_dict,
    output_dir,
    clip_id=0,
    stride=5,
    max_samples=1000,
    max_width: int = DEFAULT_MAX_WIDTH,
):
    """
    Generates train/val samples with exactly the same CompensationPipeline
    that is used at inference time.

    Network input:
      t2 = Warp(I_{t-2} -> I_t)
      t1 = Warp(I_{t-1} -> I_t)
      t0 = I_t

    Target:
      bbox-mask rasterized from the real GT annotation of frame I_t.
      weight-mask: 1 for ordinary pixels, 0 for the outer ignore-band.

    bboxes/*.npy are stored in processing coordinates for the positive crop.
    bboxes_src/*.npy are stored in source coordinates for auditing.
    """
    out = Path(output_dir)
    for directory in ["frames", "masks", "weights", "bboxes", "bboxes_src"]:
        (out / directory).mkdir(parents=True, exist_ok=True)

    source_shape = _video_source_shape(video_path)

    config = dict(DEFAULT_COMPENSATION_CONFIG)
    config["max_width"] = int(max_width)
    pipeline = CompensationPipeline(config, step=1)

    saved = 0
    saved_positive = 0
    saved_empty = 0
    skipped_invalid = 0

    for sample in pipeline.stream_video(video_path):
        # Warmup and failed compensation never become training samples.
        if not sample.valid:
            skipped_invalid += 1
            continue

        if sample.frame_index % int(stride) != 0:
            continue

        if saved >= int(max_samples):
            break

        tensor = np.asarray(sample.tensor)
        if tensor.ndim != 3 or tensor.shape[0] < 3:
            raise RuntimeError(
                f"Unexpected sample.tensor shape: {tensor.shape}"
            )

        t2 = np.clip(tensor[0] * 255.0, 0, 255).astype(np.uint8)
        t1 = np.clip(tensor[1] * 255.0, 0, 255).astype(np.uint8)
        t0 = np.clip(tensor[2] * 255.0, 0, 255).astype(np.uint8)

        # sample.frame_index is 0-based; time_layer in GT is 1-based.
        annotation_frame_id = int(sample.frame_index) + 1
        boxes_src = np.asarray(
            gt_dict.get(annotation_frame_id, []),
            dtype=np.float32,
        ).reshape(-1, 4)

        mask, weight = gt_boxes_to_mask(
            boxes_src,
            src_shape=source_shape,
            dst_shape=t0.shape,
            ignore_px=IGNORE_BAND_PX,
            min_side_for_band=IGNORE_BAND_MIN_SIDE,
        )
        boxes_processing = scale_gt_boxes(
            boxes_src,
            src_shape=source_shape,
            dst_shape=t0.shape,
        )

        sid = f"clip{clip_id:02d}_{saved:05d}"

        writes_ok = all([
            cv2.imwrite(str(out / "frames" / f"{sid}_t2.png"), t2),
            cv2.imwrite(str(out / "frames" / f"{sid}_t1.png"), t1),
            cv2.imwrite(str(out / "frames" / f"{sid}_t0.png"), t0),
            cv2.imwrite(str(out / "masks" / f"{sid}_mask.png"), mask),
            cv2.imwrite(str(out / "weights" / f"{sid}_weight.png"), weight),
        ])
        if not writes_ok:
            raise IOError(f"Could not save images for {sid}")

        np.save(
            str(out / "bboxes" / f"{sid}.npy"),
            boxes_processing,
            allow_pickle=False,
        )
        np.save(
            str(out / "bboxes_src" / f"{sid}.npy"),
            boxes_src,
            allow_pickle=False,
        )

        saved += 1
        if len(boxes_processing) > 0:
            saved_positive += 1
        else:
            saved_empty += 1

        if saved % 10 == 0:
            logger.info(
                f"Clip {clip_id}: {saved}/{max_samples} | "
                f"positive={saved_positive}, empty={saved_empty}"
            )

    logger.info(
        f"Clip {clip_id}: saved {saved}, "
        f"positive={saved_positive}, empty={saved_empty}, "
        f"invalid/warmup={skipped_invalid}"
    )
    return saved

# ================================================================

# ================================================================
# ================================================================