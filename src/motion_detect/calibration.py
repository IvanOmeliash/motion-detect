"""Post-processing calibration: probability cache + grid search over
(unet_threshold, area_min, area_max, bbox_padding)."""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .compensation import CompensationPipeline, DEFAULT_COMPENSATION_CONFIG
from .evaluate import SIZE_BUCKETS_1280X720, _finalize_metric_accumulator, _merge_metric_counts, evaluate_metrics
from .postprocess import LEGACY_PLATFORM_SUPPRESSION_ZONE_1280, _probability_to_blobs, _scaled_detection_geometry
from .tracker import SortTracker
from .utils import parse_gt_file

logger = logging.getLogger(__name__)

DEFAULT_MAX_WIDTH = 640

def cache_validation_probabilities(
    video_path,
    cache_path,
    backend,
    compensation_config=None,
    compensation_step=1,
    max_frames=None,
    max_width: int = DEFAULT_MAX_WIDTH,
):
    """
    Run classical + network once and store float16 probability and uint8 validity maps.

    Only valid compensated frames are stored. Frame-index gaps are retained and
    converted into empty tracker updates during calibration.
    """
    cache_path = Path(cache_path)
    config = dict(DEFAULT_COMPENSATION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = int(max_width)

    pipeline = CompensationPipeline(
        config,
        step=int(compensation_step),
    )

    frame_indices = []
    probabilities = []
    validities = []
    inlier_ratios = []
    seen = 0

    for sample in pipeline.stream_video(video_path):
        seen += 1
        if max_frames is not None and seen > int(max_frames):
            break
        if not sample.valid:
            continue

        tensor = np.asarray(sample.tensor)
        result = backend.infer(
            tensor[:3],
            threshold=0.5,
            return_probability=True,
        )
        probability = np.asarray(
            result["probability"],
            dtype=np.float32,
        )
        probability_f16 = np.asarray(
            probability,
            dtype=np.float16,
        )

        frame_indices.append(int(sample.frame_index) + 1)
        probabilities.append(probability_f16)
        validities.append(
            (np.asarray(tensor[3]) > 0.5).astype(np.uint8)
        )
        inlier_ratios.append(float(sample.inlier_ratio))

    if not probabilities:
        raise RuntimeError(
            f"No valid probability maps produced for {video_path}"
        )

    probability_array = np.stack(probabilities)
    validity_array = np.stack(validities)

    metadata = {
        "video_path": str(video_path),
        "backend": getattr(backend, "name", type(backend).__name__),
        "processing_shape": list(probability_array.shape[1:]),
        "frames_seen": seen,
        "valid_frames_cached": len(frame_indices),
        "probability_encoding": "float16 probability",
        "classical_stats": dict(pipeline.stats),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        probabilities=probability_array,
        validities=validity_array,
        inlier_ratios=np.asarray(
            inlier_ratios,
            dtype=np.float32,
        ),
        metadata=np.asarray(
            json.dumps(metadata, ensure_ascii=False)
        ),
    )
    logger.info("Probability cache:", cache_path)
    return metadata

def load_probability_cache(cache_path):
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "frame_indices": data["frame_indices"].copy(),
            "probabilities": data["probabilities"].copy(),
            "validities": data["validities"].copy(),
            "inlier_ratios": data["inlier_ratios"].copy(),
            "metadata": json.loads(str(data["metadata"].item())),
        }

def predictions_from_probability_cache(
    cache,
    params,
    unet_threshold,
    return_size=(1280, 720),
    apply_validity_mask=True,
    apply_platform_suppression=None,
    platform_suppression_zone=LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
):
    probability_maps = cache["probabilities"]
    validities = cache["validities"]
    frame_indices = cache["frame_indices"]
    inlier_ratios = cache["inlier_ratios"]

    _, height, width = probability_maps.shape
    geometry = _scaled_detection_geometry(
        height,
        width,
        params,
        return_size=return_size,
    )
    if apply_platform_suppression is None:
        apply_platform_suppression = bool(
            params.get("apply_platform_suppression", True)
        )
    active_platform_zone = (
        platform_suppression_zone
        if apply_platform_suppression
        else None
    )
    tracker = SortTracker(
        max_age=params["max_age"],
        min_hits=params["min_hits"],
        iou_threshold=params["iou_thresh"],
        max_dist=max(
            1.0,
            float(params["max_dist"]) * geometry["scale_x"],
        ),
    )

    predictions = {}
    tensor_stub = np.empty(
        (4, height, width),
        dtype=np.float32,
    )
    tensor_stub[:3].fill(0.0)

    last_frame_index = 0
    for record_index, frame_index in enumerate(frame_indices):
        frame_index = int(frame_index)

        # Invalid/missing compensation frames age the tracker.
        gap = max(0, frame_index - last_frame_index - 1)
        for _ in range(gap):
            tracker.update([])
        last_frame_index = frame_index

        min_ratio = params.get("inlier_thresh")
        if (
            min_ratio is not None
            and float(inlier_ratios[record_index])
            < float(min_ratio)
        ):
            tracker.update([])
            continue

        probability_map = probability_maps[record_index]
        binary_mask = (
            probability_map > float(unet_threshold)
        ).astype(np.uint8) * np.uint8(255)

        np.copyto(
            tensor_stub[3],
            validities[record_index],
            casting="unsafe",
        )

        _, blobs = _probability_to_blobs(
            prob=None,
            binary_mask=binary_mask,
            tensor=tensor_stub,
            threshold=unet_threshold,
            area_min=geometry["area_min"],
            area_max=geometry["area_max"],
            bbox_padding=geometry["bbox_padding"],
            apply_validity_mask=apply_validity_mask,
            platform_suppression_zone=active_platform_zone,
        )

        tracks = tracker.update([
            blob["bbox"] for blob in blobs
        ])
        frame_predictions = []

        for track in tracks:
            if (
                track.hits < params["min_hits"]
                or track.time_since_update != 0
            ):
                continue

            x1, y1, x2, y2 = track.bbox
            frame_predictions.append([
                int(round(
                    np.clip(x1, 0, width)
                    * geometry["output_scale_x"]
                )),
                int(round(
                    np.clip(y1, 0, height)
                    * geometry["output_scale_y"]
                )),
                int(round(
                    np.clip(x2, 0, width)
                    * geometry["output_scale_x"]
                )),
                int(round(
                    np.clip(y2, 0, height)
                    * geometry["output_scale_y"]
                )),
            ])

        if frame_predictions:
            predictions[frame_index] = frame_predictions

    return predictions

def calibrate_postprocessing_grid(
    validation_items,
    base_params,
    backend,
    cache_dir,
    output_json,
    unet_threshold_values,
    area_min_values,
    area_max_values,
    bbox_padding_values,
    precision_floor,
    iou_threshold=0.3,
    rebuild_cache=False,
    max_frames_per_clip=None,
    apply_platform_suppression=None,
):
    """
    Maximize far recall subject to global precision >= precision_floor.

    validation_items:
      [{"video_path": ..., "gt_path": ..., "clip_id": ...}, ...]
    """
    cache_dir = Path(cache_dir)
    output_json = Path(output_json)
    cache_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for threshold, area_min, area_max, padding in itertools.product(
        unet_threshold_values,
        area_min_values,
        area_max_values,
        bbox_padding_values,
    ):
        if float(area_max) <= float(area_min):
            continue

        params = dict(base_params)
        params.update({
            "area_min": float(area_min),
            "area_max": float(area_max),
            "bbox_padding": int(padding),
        })
        candidates.append({
            "unet_threshold": float(threshold),
            "params": params,
            "accumulator": {
                "global": {"TP": 0, "FP": 0, "FN": 0},
                "buckets": {
                    name: {"TP": 0, "FP": 0, "FN": 0}
                    for name in SIZE_BUCKETS_1280X720
                },
            },
        })

    if not candidates:
        raise ValueError("The parameter grid produced no valid candidates")

    logger.info("Calibration candidates:", len(candidates))

    for item_index, item in enumerate(validation_items):
        video_path = Path(item["video_path"])
        gt_path = Path(item["gt_path"])
        clip_id = item.get("clip_id", item_index)

        cache_path = cache_dir / f"clip_{clip_id}_probabilities.npz"
        if rebuild_cache or not cache_path.exists():
            cache_validation_probabilities(
                video_path=video_path,
                cache_path=cache_path,
                backend=backend,
                max_frames=max_frames_per_clip,
            )

        cache = load_probability_cache(cache_path)
        gt_dict = parse_gt_file(
            str(gt_path),
            video_path=str(video_path),
            target_w=1280,
            target_h=720,
        )

        for candidate in candidates:
            pred_dict = predictions_from_probability_cache(
                cache=cache,
                params=candidate["params"],
                unet_threshold=candidate["unet_threshold"],
                return_size=(1280, 720),
                apply_platform_suppression=apply_platform_suppression,
            )
            clip_metrics = evaluate_metrics(
                gt_dict,
                pred_dict,
                iou_thresh=iou_threshold,
                return_details=True,
            )
            _merge_metric_counts(
                candidate["accumulator"],
                clip_metrics,
            )

        logger.info(
            f"Calibrated clip {clip_id}: "
            f"{len(cache['frame_indices'])} cached frames"
        )

    results = []
    for candidate in candidates:
        metrics = _finalize_metric_accumulator(
            candidate["accumulator"],
            iou_thresh=iou_threshold,
        )
        global_metrics = metrics["global"]
        far_metrics = metrics["buckets"]["far"]

        results.append({
            "unet_threshold": candidate["unet_threshold"],
            "area_min": candidate["params"]["area_min"],
            "area_max": candidate["params"]["area_max"],
            "bbox_padding": candidate["params"]["bbox_padding"],
            "global_precision": global_metrics["precision"],
            "global_recall": global_metrics["recall"],
            "global_f1": global_metrics["f1"],
            "far_precision": far_metrics["precision"],
            "far_recall": far_metrics["recall"],
            "far_f1": far_metrics["f1"],
            "metrics": metrics,
        })

    eligible = [
        result for result in results
        if result["global_precision"] >= float(precision_floor)
    ]

    if eligible:
        ranking_pool = eligible
        precision_floor_met = True
        ranking_pool.sort(
            key=lambda result: (
                result["far_recall"],
                result["global_precision"],
                result["far_f1"],
                result["global_f1"],
            ),
            reverse=True,
        )
    else:
        ranking_pool = results
        precision_floor_met = False
        ranking_pool.sort(
            key=lambda result: (
                result["global_precision"],
                result["far_recall"],
                result["global_f1"],
            ),
            reverse=True,
        )

    best = ranking_pool[0]
    report = {
        "objective": (
            "maximize far_recall subject to "
            "global_precision >= precision_floor"
        ),
        "precision_floor": float(precision_floor),
        "precision_floor_met": precision_floor_met,
        "iou_threshold": float(iou_threshold),
        "validation_clips": len(validation_items),
        "candidate_count": len(results),
        "best": best,
        "top_20": ranking_pool[:20],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Best calibrated configuration:")
    logger.info(json.dumps(best, indent=2, ensure_ascii=False))
    logger.info("Calibration report:", output_json)
    return report