#!/usr/bin/env python3
"""Mine hard negatives, render review grids, and approve from review.csv.

Workflow:
  1. mine:   python scripts/mine_hard_negatives.py mine --round 1 ...
  2. review: open results/hard_negative_staging/round_01/review_grids/*.jpg and
     fill decision=approve/reject in results/hard_negative_staging/round_01/review.csv
     (schema: rank,sample_id,decision,notes,score,rank_score,prob_max,area,
      inlier_ratio,clip_id,frame_index,rotor_zone_hit,rotor_zone_overlap_area_1280)
  3. approve: python scripts/mine_hard_negatives.py approve --round 1 ...

Example:
    python scripts/mine_hard_negatives.py mine \
        --config configs/paths.yaml --defaults configs/default.yaml \
        --checkpoint checkpoints/best_unet.pth --round 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from motion_detect.config import load_config
from motion_detect.hard_negatives import (
    approve_from_review_csv,
    create_hard_negative_review_grids,
    mine_hard_negatives,
)
from motion_detect.inference import load_frozen_pytorch_backend
from motion_detect.model import default_device

logger = logging.getLogger("mine_hard_negatives")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("mine", "grids", "approve"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None)
        p.add_argument("--defaults", default="configs/default.yaml")
        p.add_argument("--videos", default=None)
        p.add_argument("--annotations", default=None)
        p.add_argument("--train-dir", default=None)
        p.add_argument("--staging-root", default=None)
        p.add_argument("--checkpoint", default=None)
        p.add_argument("--round", type=int, default=1)
        p.add_argument("--clips", default=None, help="comma-separated clip ids (default: config test clips)")
        p.add_argument("--top-fraction", type=float, default=None)
        p.add_argument("--review-count", type=int, default=None)
        p.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config, defaults_path=args.defaults)

    from motion_detect.utils import extract_archives, index_annotation_files, index_video_files

    videos_source = args.videos or cfg.videos_dir
    annotations_source = args.annotations or cfg.annotations_dir
    if videos_source is None or annotations_source is None:
        raise SystemExit("Provide --videos/--annotations or paths in the config file")
    videos_dir, annotations_dir = extract_archives(videos_source, annotations_source)
    dataset_root = cfg.dataset_root or Path("data/dataset_gt")
    train_dir = Path(args.train_dir) if args.train_dir else dataset_root / "train"
    staging_root = Path(args.staging_root) if args.staging_root else (cfg.results_dir or Path("results")) / "hard_negative_staging"
    round_dir = staging_root / f"round_{args.round:02d}"

    clips = (
        [int(c) for c in args.clips.split(",")] if args.clips else list(cfg.test_clips)
    )

    if args.command == "mine":
        checkpoint = args.checkpoint or (cfg.checkpoints_dir / "best_unet.pth" if cfg.checkpoints_dir else Path("checkpoints/best_unet.pth"))
        device = args.device or default_device()
        backend_model = load_frozen_pytorch_backend(str(checkpoint), device=device).model
        metadata = mine_hard_negatives(
            videos_source=videos_source,
            annotations_source=annotations_source,
            train_dir=train_dir,
            output_dir=round_dir,
            params=dict(cfg.detection_params),
            unet_model=backend_model,
            clip_indices=clips,
            unet_threshold=cfg.unet_threshold,
            top_fraction=args.top_fraction if args.top_fraction is not None else cfg.hard_negative_fraction,
            round_id=args.round,
        )
        logger.info("Mined %d candidates into %s", len(metadata), round_dir)
        create_hard_negative_review_grids(
            round_dir,
            count=args.review_count if args.review_count is not None else cfg.hard_negative_review_count,
            display_pages=False,
        )
    elif args.command == "grids":
        pages = create_hard_negative_review_grids(
            round_dir,
            count=args.review_count if args.review_count is not None else cfg.hard_negative_review_count,
            display_pages=False,
        )
        logger.info("Review grids: %s", pages)
    elif args.command == "approve":
        summary = approve_from_review_csv(
            staging_dir=round_dir,
            train_dir=train_dir,
            min_rank=1,
            max_rank=args.review_count if args.review_count is not None else cfg.hard_negative_review_count,
            replace_existing=True,
        )
        logger.info("Approval summary: %s", summary)


if __name__ == "__main__":
    main()
