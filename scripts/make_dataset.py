#!/usr/bin/env python3
"""Generate the train/val dataset from videos + GT annotations.

Runs the exact production CompensationPipeline per clip and rasterizes
GT bounding boxes (NOT pseudo-labels) into masks/weights.

Example:
    python scripts/make_dataset.py \
        --config configs/paths.yaml --defaults configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from motion_detect.config import load_config
from motion_detect.dataset import _video_source_shape, generate_dataset
from motion_detect.utils import (
    extract_archives,
    index_annotation_files,
    index_video_files,
    parse_gt_file,
    validate_gt_coordinate_ranges,
)

logger = logging.getLogger("make_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="paths YAML (see configs/paths.example.yaml)")
    parser.add_argument("--defaults", default="configs/default.yaml", help="base YAML with defaults")
    parser.add_argument("--videos", default=None, help="override videos dir/ZIP")
    parser.add_argument("--annotations", default=None, help="override annotations dir/ZIP")
    parser.add_argument("--out", default=None, help="override dataset root")
    parser.add_argument("--extract-dir", default="data", help="where ZIP archives are extracted")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config, defaults_path=args.defaults)

    videos_source = args.videos or (cfg.videos_dir if cfg.videos_dir else None)
    annotations_source = args.annotations or (cfg.annotations_dir if cfg.annotations_dir else None)
    dataset_root = args.out or (cfg.dataset_root if cfg.dataset_root else Path("data/dataset_gt"))
    if videos_source is None or annotations_source is None:
        raise SystemExit("Provide --videos/--annotations or paths in the config file")

    videos_dir, annotations_dir = extract_archives(
        videos_source, annotations_source, extract_dir=args.extract_dir
    )
    video_index = index_video_files(videos_dir)
    annotation_index = index_annotation_files(annotations_dir)

    random.seed(cfg.split_seed)
    all_clips = list(range(1, cfg.num_clips + 1))
    random.shuffle(all_clips)
    train_clips = sorted(all_clips[: cfg.train_clips_count])
    val_clips = sorted(
        all_clips[cfg.train_clips_count : cfg.train_clips_count + cfg.val_clips_count]
    )
    test_clips = sorted(all_clips[cfg.train_clips_count + cfg.val_clips_count :])
    logger.info("Split: train=%s val=%s test=%s", train_clips, val_clips, test_clips)

    required = sorted(set(train_clips + val_clips + test_clips))
    missing = [c for c in required if c not in video_index or c not in annotation_index]
    if missing:
        raise RuntimeError(f"Missing video/annotation for clips: {missing}")

    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    for split_name, clips, out_dir in (("train", train_clips, train_dir), ("val", val_clips, val_dir)):
        saved_total = 0
        for clip_id in clips:
            video_path = video_index[clip_id]
            gt = parse_gt_file(str(annotation_index[clip_id]), video_path=str(video_path))
            validate_gt_coordinate_ranges(gt, _video_source_shape(str(video_path)))
            saved = generate_dataset(
                str(video_path),
                gt,
                out_dir,
                clip_id=clip_id,
                stride=cfg.stride,
                max_samples=cfg.samples_per_clip,
                max_width=cfg.max_width,
            )
            saved_total += int(saved)
        logger.info("%s: %d samples saved to %s", split_name, saved_total, out_dir)
        if saved_total == 0:
            raise RuntimeError(f"Generator saved no {split_name} samples")

    logger.info("Dataset ready at %s (test clips %s stay untouched)", dataset_root, test_clips)


if __name__ == "__main__":
    main()
