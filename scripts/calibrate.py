#!/usr/bin/env python3
"""Calibrate post-processing on cached probability maps.

Grid-searches (unet_threshold, area_min, area_max, bbox_padding) maximizing
far recall subject to global precision >= precision_floor. Probability maps
are cached per clip as float16 npz and reused unless --rebuild-cache.

Example:
    python scripts/calibrate.py \
        --config configs/paths.yaml --defaults configs/default.yaml \
        --checkpoint checkpoints/best_unet.pth \
        --cache-dir results/calibration/probability_cache \
        --report results/calibration/postprocess_calibration.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from motion_detect.calibration import calibrate_postprocessing_grid
from motion_detect.config import load_config
from motion_detect.inference import load_frozen_pytorch_backend
from motion_detect.model import default_device
from motion_detect.utils import index_annotation_files, index_video_files

logger = logging.getLogger("calibrate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--defaults", default="configs/default.yaml")
    parser.add_argument("--videos", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clips", default=None, help="comma-separated clip ids (default: config calibration.clips)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-frames-per-clip", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config, defaults_path=args.defaults)

    videos_dir = args.videos or cfg.videos_dir
    annotations_dir = args.annotations or cfg.annotations_dir
    if videos_dir is None or annotations_dir is None:
        raise SystemExit("Provide --videos/--annotations or paths in the config file")
    video_index = index_video_files(videos_dir)
    annotation_index = index_annotation_files(annotations_dir)

    clip_ids = [int(c) for c in args.clips.split(",")] if args.clips else list(cfg.calibration_clip_ids)
    items = []
    for clip_id in clip_ids:
        video_path = video_index.get(clip_id)
        gt_path = annotation_index.get(clip_id)
        if video_path is None or gt_path is None:
            raise RuntimeError(f"Missing video/annotation for clip {clip_id}")
        items.append({"clip_id": clip_id, "video_path": str(video_path), "gt_path": str(gt_path)})

    checkpoint = args.checkpoint or (cfg.checkpoints_dir / "best_unet.pth" if cfg.checkpoints_dir else Path("checkpoints/best_unet.pth"))
    device = args.device or default_device()
    backend = load_frozen_pytorch_backend(str(checkpoint), device=device)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (cfg.results_dir or Path("results")) / "calibration" / "probability_cache"
    report_path = Path(args.report) if args.report else (cfg.results_dir or Path("results")) / "calibration" / "postprocess_calibration.json"

    report = calibrate_postprocessing_grid(
        validation_items=items,
        base_params=dict(cfg.detection_params),
        backend=backend,
        cache_dir=cache_dir,
        output_json=report_path,
        unet_threshold_values=list(cfg.calibration_unet_thresholds),
        area_min_values=list(cfg.calibration_area_min_values),
        area_max_values=list(cfg.calibration_area_max_values),
        bbox_padding_values=list(cfg.calibration_bbox_padding_values),
        precision_floor=cfg.calibration_precision_floor,
        iou_threshold=cfg.bbox_iou_threshold,
        rebuild_cache=args.rebuild_cache,
        max_frames_per_clip=args.max_frames_per_clip,
        apply_platform_suppression=cfg.apply_platform_suppression,
    )
    logger.info("Best: %s", report["best"])


if __name__ == "__main__":
    main()
