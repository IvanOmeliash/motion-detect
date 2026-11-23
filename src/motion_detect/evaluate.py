"""Object-level evaluation: Hungarian bbox matching with IoU, far/mid/near size buckets."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .inference import predict_video_bboxes
from .tracker import iou
from .utils import (
    extract_archives,
    index_annotation_files,
    index_video_files,
    parse_gt_file,
)

logger = logging.getLogger(__name__)

SIZE_BUCKETS_1280X720 = {
    "far": (0.0, 16.0),
    "mid": (16.0, 64.0),
    "near": (64.0, float("inf")),
}


def _bbox_area(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _size_bucket_from_area(area):
    area = float(area)
    if area < 16.0:
        return "far"
    if area <= 64.0:
        return "mid"
    return "near"

def _counts_to_metrics(counts):
    tp = int(counts["TP"])
    fp = int(counts["FP"])
    fn = int(counts["FN"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def evaluate_metrics(
    gt_dict,
    pred_dict,
    iou_thresh=0.3,
    return_details=False,
):
    """
    Hungarian bbox matching plus size-bucket metrics in 1280x720 coordinates.

    TP/FN are assigned by matched/unmatched GT area.
    An unmatched prediction has no GT size, so its FP is assigned by predicted
    bbox area. This makes per-bucket precision well-defined and transparent.
    """
    global_counts = {"TP": 0, "FP": 0, "FN": 0}
    bucket_counts = {
        name: {"TP": 0, "FP": 0, "FN": 0}
        for name in SIZE_BUCKETS_1280X720
    }

    for frame_id in sorted(set(gt_dict) | set(pred_dict)):
        gts = list(gt_dict.get(frame_id, []))
        preds = list(pred_dict.get(frame_id, []))

        if not gts:
            for pred in preds:
                bucket = _size_bucket_from_area(_bbox_area(pred))
                bucket_counts[bucket]["FP"] += 1
                global_counts["FP"] += 1
            continue

        if not preds:
            for gt in gts:
                bucket = _size_bucket_from_area(_bbox_area(gt))
                bucket_counts[bucket]["FN"] += 1
                global_counts["FN"] += 1
            continue

        matrix = np.zeros((len(preds), len(gts)), dtype=np.float32)
        for pred_index, pred_box in enumerate(preds):
            for gt_index, gt_box in enumerate(gts):
                matrix[pred_index, gt_index] = iou(pred_box, gt_box)

        row_indices, col_indices = linear_sum_assignment(-matrix)
        matched_predictions = set()
        matched_gt = set()

        for pred_index, gt_index in zip(row_indices, col_indices):
            if matrix[pred_index, gt_index] < float(iou_thresh):
                continue
            matched_predictions.add(int(pred_index))
            matched_gt.add(int(gt_index))
            bucket = _size_bucket_from_area(_bbox_area(gts[gt_index]))
            bucket_counts[bucket]["TP"] += 1
            global_counts["TP"] += 1

        for gt_index, gt_box in enumerate(gts):
            if gt_index in matched_gt:
                continue
            bucket = _size_bucket_from_area(_bbox_area(gt_box))
            bucket_counts[bucket]["FN"] += 1
            global_counts["FN"] += 1

        for pred_index, pred_box in enumerate(preds):
            if pred_index in matched_predictions:
                continue
            bucket = _size_bucket_from_area(_bbox_area(pred_box))
            bucket_counts[bucket]["FP"] += 1
            global_counts["FP"] += 1

    if not return_details:
        return (
            global_counts["TP"],
            global_counts["FP"],
            global_counts["FN"],
        )

    return {
        "global": _counts_to_metrics(global_counts),
        "buckets": {
            bucket: _counts_to_metrics(counts)
            for bucket, counts in bucket_counts.items()
        },
        "bucket_definition_1280x720": {
            "far": "area < 16 px^2",
            "mid": "16 <= area <= 64 px^2",
            "near": "area > 64 px^2",
        },
        "fp_bucket_basis": "predicted bbox area for unmatched predictions",
        "iou_threshold": float(iou_thresh),
    }

def _merge_metric_counts(accumulator, details):
    for key in ("TP", "FP", "FN"):
        accumulator["global"][key] += int(details["global"][key])
    for bucket in SIZE_BUCKETS_1280X720:
        for key in ("TP", "FP", "FN"):
            accumulator["buckets"][bucket][key] += int(
                details["buckets"][bucket][key]
            )

def _finalize_metric_accumulator(accumulator, iou_thresh):
    return {
        "global": _counts_to_metrics(accumulator["global"]),
        "buckets": {
            bucket: _counts_to_metrics(counts)
            for bucket, counts in accumulator["buckets"].items()
        },
        "bucket_definition_1280x720": {
            "far": "area < 16 px^2",
            "mid": "16 <= area <= 64 px^2",
            "near": "area > 64 px^2",
        },
        "fp_bucket_basis": "predicted bbox area for unmatched predictions",
        "iou_threshold": float(iou_thresh),
    }

def print_bucket_metrics(metrics):
    logger.info("\nSize buckets in 1280x720 coordinates")
    logger.info(
        f"{'bucket':<8} {'TP':>6} {'FP':>6} {'FN':>6} "
        f"{'precision':>11} {'recall':>9} {'F1':>9}"
    )
    for bucket in ("far", "mid", "near"):
        row = metrics["buckets"][bucket]
        logger.info(
            f"{bucket:<8} {row['TP']:>6} {row['FP']:>6} {row['FN']:>6} "
            f"{row['precision']:>11.4f} {row['recall']:>9.4f} "
            f"{row['f1']:>9.4f}"
        )

def quality_acceptance_gate(
    baseline_metrics,
    candidate_metrics,
    tolerance=1e-12,
):
    """
    Accept when global precision increases and far recall does not decrease.
    """
    baseline_precision = baseline_metrics["global"]["precision"]
    candidate_precision = candidate_metrics["global"]["precision"]
    baseline_far_recall = baseline_metrics["buckets"]["far"]["recall"]
    candidate_far_recall = candidate_metrics["buckets"]["far"]["recall"]

    precision_improved = (
        candidate_precision > baseline_precision + float(tolerance)
    )
    far_recall_preserved = (
        candidate_far_recall + float(tolerance) >= baseline_far_recall
    )
    return {
        "accepted": bool(precision_improved and far_recall_preserved),
        "precision_improved": bool(precision_improved),
        "far_recall_preserved": bool(far_recall_preserved),
        "baseline_precision": float(baseline_precision),
        "candidate_precision": float(candidate_precision),
        "baseline_far_recall": float(baseline_far_recall),
        "candidate_far_recall": float(candidate_far_recall),
    }

def run_evaluation_with_viz(
    videos_zip,
    annotations_zip,
    params,
    iou_thresh=0.3,
    clip_indices=None,
    num_samples=10,
    unet_model=None,
    unet_threshold=0.5,
    return_bucket_metrics=False,
):
    """
    Evaluate global bbox metrics and far/mid/near metrics.
    """
    videos_dir, annotations_dir = extract_archives(
        videos_zip,
        annotations_zip,
    )
    pred_dict_per_clip = {}

    if clip_indices is None:
        sampled_indices = sorted(
            random.sample(range(1, 51), num_samples)
        )
    else:
        sampled_indices = sorted(clip_indices)

    logger.info(f"Evaluation on {len(sampled_indices)} clips: {sampled_indices}")

    video_index = index_video_files(videos_dir)
    annotation_index = index_annotation_files(annotations_dir)

    accumulator = {
        "global": {"TP": 0, "FP": 0, "FN": 0},
        "buckets": {
            name: {"TP": 0, "FP": 0, "FN": 0}
            for name in SIZE_BUCKETS_1280X720
        },
    }

    for clip_id in tqdm(sampled_indices, desc="Evaluation"):
        video_path = video_index.get(clip_id)
        gt_path = annotation_index.get(clip_id)
        if video_path is None or gt_path is None:
            logger.info(f"Skip Clip_{clip_id}: missing video or annotation")
            continue

        gt_dict = parse_gt_file(
            str(gt_path),
            video_path=str(video_path),
            target_w=1280,
            target_h=720,
        )
        pred_dict = predict_video_bboxes(
            str(video_path),
            params,
            unet_model=unet_model,
            unet_threshold=unet_threshold,
            checkpoint_blur=False,
        )
        pred_dict_per_clip[clip_id] = pred_dict

        clip_metrics = evaluate_metrics(
            gt_dict,
            pred_dict,
            iou_thresh=iou_thresh,
            return_details=True,
        )
        _merge_metric_counts(accumulator, clip_metrics)

    metrics = _finalize_metric_accumulator(
        accumulator,
        iou_thresh=iou_thresh,
    )
    global_metrics = metrics["global"]

    logger.info(
        f"\n--- Results ({'UNet' if unet_model else 'Classic'}) ---"
    )
    logger.info(f"Precision: {global_metrics['precision']:.4f}")
    logger.info(f"Recall:    {global_metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {global_metrics['f1']:.4f}")
    print_bucket_metrics(metrics)

    result = (
        global_metrics["precision"],
        global_metrics["recall"],
        global_metrics["f1"],
        sampled_indices,
        pred_dict_per_clip,
    )
    if return_bucket_metrics:
        return result + (metrics,)
    return result

