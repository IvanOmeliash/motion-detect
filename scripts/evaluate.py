#!/usr/bin/env python3
"""Object-level evaluation: bbox P/R/F1 with IoU matching + far/mid/near buckets,
optionally swept over UNet thresholds. Writes a JSON report.

Example:
    python scripts/evaluate.py \
        --config configs/paths.yaml --defaults configs/default.yaml \
        --checkpoint checkpoints/best_unet.pth --clips 3,28,38,48,50 \
        --thresholds 0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
        --report results/eval_val.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from motion_detect.config import load_config
from motion_detect.evaluate import (
    _finalize_metric_accumulator,
    _merge_metric_counts,
    evaluate_metrics,
)
from motion_detect.inference import load_frozen_pytorch_backend, predict_video_bboxes
from motion_detect.model import default_device
from motion_detect.utils import (
    index_annotation_files,
    index_video_files,
    parse_gt_file,
)

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--defaults", default="configs/default.yaml")
    parser.add_argument("--videos", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clips", default=None, help="comma-separated clip ids (default: config val_clips)")
    parser.add_argument("--thresholds", default=None, help="comma-separated UNet thresholds (default: single config value)")
    parser.add_argument("--iou", type=float, default=None, help="bbox IoU threshold (default: config bbox_iou_threshold)")
    parser.add_argument("--report", default=None, help="JSON report path")
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

    clips = [int(c) for c in args.clips.split(",")] if args.clips else list(cfg.val_clips)
    iou_thresh = args.iou if args.iou is not None else cfg.bbox_iou_threshold
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    else:
        thresholds = [cfg.unet_threshold]

    checkpoint = args.checkpoint or (cfg.checkpoints_dir / "best_unet.pth" if cfg.checkpoints_dir else Path("checkpoints/best_unet.pth"))
    device = args.device or default_device()
    backend = load_frozen_pytorch_backend(str(checkpoint), device=device)
    logger.info("Checkpoint %s on %s", checkpoint, device)

    report = {
        "checkpoint": str(checkpoint),
        "clips": clips,
        "iou_threshold": iou_thresh,
        "results": [],
    }
    for threshold in thresholds:
        accumulator = {
            "global": {"TP": 0, "FP": 0, "FN": 0},
            "buckets": {name: {"TP": 0, "FP": 0, "FN": 0} for name in ("far", "mid", "near")},
        }
        for clip_id in clips:
            video_path = video_index.get(clip_id)
            gt_path = annotation_index.get(clip_id)
            if video_path is None or gt_path is None:
                logger.warning("Skip Clip_%s: missing video or annotation", clip_id)
                continue
            gt_dict = parse_gt_file(str(gt_path), video_path=str(video_path), target_w=1280, target_h=720)
            pred_dict = predict_video_bboxes(
                str(video_path),
                params=dict(cfg.detection_params),
                unet_model=backend.model,
                unet_threshold=threshold,
                compensation_step=cfg.compensation_step,
                compensation_config=dict(cfg.compensation),
                return_size=cfg.return_size,
                apply_validity_mask=cfg.apply_validity_mask,
                apply_platform_suppression=cfg.apply_platform_suppression,
                max_width=cfg.max_width,
                device=device,
            )
            clip_metrics = evaluate_metrics(gt_dict, pred_dict, iou_thresh=iou_thresh, return_details=True)
            _merge_metric_counts(accumulator, clip_metrics)
        metrics = _finalize_metric_accumulator(accumulator, iou_thresh)
        g = metrics["global"]
        logger.info(
            "threshold=%.2f | precision=%.4f | recall=%.4f | F1=%.4f",
            threshold, g["precision"], g["recall"], g["f1"],
        )
        for bucket in ("far", "mid", "near"):
            b = metrics["buckets"][bucket]
            logger.info(
                "  %s: TP=%d FP=%d FN=%d P=%.4f R=%.4f F1=%.4f",
                bucket, b["TP"], b["FP"], b["FN"], b["precision"], b["recall"], b["f1"],
            )
        report["results"].append({"threshold": threshold, "metrics": metrics})

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
