#!/usr/bin/env python3
"""Train TinyUNet with the 60/25/15 sampler (positive / hard_negative / random_negative).

Checkpoint format is identical to the research notebooks.

Example:
    python scripts/train.py \
        --config configs/paths.yaml --defaults configs/default.yaml \
        --checkpoint checkpoints/best_unet.pth
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from motion_detect.config import load_config
from motion_detect.model import TinyUNet, default_device
from motion_detect.train import train_unet

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="paths YAML")
    parser.add_argument("--defaults", default="configs/default.yaml", help="base YAML")
    parser.add_argument("--train-dir", default=None, help="override train dataset dir")
    parser.add_argument("--val-dir", default=None, help="override val dataset dir")
    parser.add_argument("--save-dir", default=None, help="override checkpoints dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(args.config, defaults_path=args.defaults)

    seed = args.seed if args.seed is not None else cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_root = cfg.dataset_root or Path("data/dataset_gt")
    train_dir = args.train_dir or str(dataset_root / "train")
    val_dir = args.val_dir or str(dataset_root / "val")
    save_dir = args.save_dir or str(cfg.checkpoints_dir or Path("checkpoints"))

    device = args.device or default_device()
    logger.info("Device: %s | seed=%d", device, seed)

    model = TinyUNet()
    logger.info("Trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    best_path = train_unet(
        model,
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=args.epochs if args.epochs is not None else cfg.epochs,
        batch_size=args.batch_size if args.batch_size is not None else cfg.batch_size,
        lr=args.lr if args.lr is not None else cfg.learning_rate,
        save_dir=save_dir,
        device=device,
        group_ratios=cfg.group_ratios,
        pos_weight=cfg.pos_weight,
        pos_weight_no_empty=cfg.pos_weight_no_empty,
        allow_train_without_empty=cfg.allow_train_without_empty,
        allow_train_without_hard_negatives=cfg.allow_train_without_hard_negatives,
        min_empty_fraction_warning=cfg.min_empty_fraction_warning,
        processing_max_width=cfg.max_width,
        crop_size=cfg.train_crop_size,
        hard_negative_remine_every=cfg.hard_negative_remine_every,
    )
    checkpoint = torch.load(best_path, map_location="cpu")
    logger.info("Best checkpoint: %s (epoch=%s val_f1=%.4f)", best_path, checkpoint.get("epoch"), checkpoint.get("val_f1", float("nan")))


if __name__ == "__main__":
    main()
