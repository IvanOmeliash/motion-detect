#!/usr/bin/env python3
"""Run inference on one clip and write an annotated mp4 + a JSON of bboxes.

Example:
    python scripts/infer_video.py \
        --config configs/paths.yaml --defaults configs/default.yaml \
        --checkpoint checkpoints/best_unet.pth --clip 41 \
        --output results/render/Clip_41_annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from motion_detect.config import load_config
from motion_detect.inference import load_frozen_pytorch_backend, predict_video_bboxes
from motion_detect.model import default_device
from motion_detect.postprocess import render_predictions_video
from motion_detect.utils import index_annotation_files, index_video_files

logger = logging.getLogger("infer_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--defaults", default="configs/default.yaml")
    parser.add_argument("--videos", default=None)
    parser.add_argument("--input", default=None, help="direct video path (overrides --clip)")
    parser.add_argument("--clip", type=int, default=41, help="clip id to look up in videos dir")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="UNet threshold override")
    parser.add_argument("--output", default=None, help="annotated mp4 path")
    parser.add_argument("--json", default=None, help="bbox predictions JSON path")
    parser.add_argument("--pipelined", action="store_true", help="use the 2-thread pipeline")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config, defaults_path=args.defaults)

    if args.input:
        video_path = Path(args.input)
    else:
        videos_dir = args.videos or cfg.videos_dir
        if videos_dir is None:
            raise SystemExit("Provide --videos or --input")
        video_path = index_video_files(videos_dir)[args.clip]

    checkpoint = args.checkpoint or (cfg.checkpoints_dir / "best_unet.pth" if cfg.checkpoints_dir else Path("checkpoints/best_unet.pth"))
    device = args.device or default_device()
    threshold = args.threshold if args.threshold is not None else cfg.unet_threshold

    backend = load_frozen_pytorch_backend(str(checkpoint), device=device)
    params = dict(cfg.detection_params)

    if args.pipelined:
        from motion_detect.inference import predict_video_bboxes_pipelined

        predictions, report = predict_video_bboxes_pipelined(
            str(video_path),
            params=params,
            backend=backend,
            unet_threshold=threshold,
            compensation_step=cfg.compensation_step,
            compensation_config=dict(cfg.compensation),
            return_size=cfg.return_size,
            apply_validity_mask=cfg.apply_validity_mask,
            apply_platform_suppression=cfg.apply_platform_suppression,
            max_width=cfg.max_width,
        )
        logger.info("Pipelined report: %s", json.dumps({k: v for k, v in report.items() if k in ("throughput_fps", "frames_consumed", "invalid_samples", "skipped_low_confidence")}))
    else:
        predictions = predict_video_bboxes(
            str(video_path),
            params=params,
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

    frames_with_boxes = sum(1 for v in predictions.values() if v)
    total_boxes = sum(len(v) for v in predictions.values())
    logger.info("Frames with boxes: %d | total rendered boxes: %d", frames_with_boxes, total_boxes)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({str(k): v for k, v in predictions.items()}, indent=2), encoding="utf-8")
        logger.info("Predictions written to %s", json_path)

    output = args.output or str((cfg.results_dir or Path("results")) / "render" / (video_path.stem + "_annotated.mp4"))
    render_predictions_video(str(video_path), output, predictions, output_size=cfg.return_size)


if __name__ == "__main__":
    main()
