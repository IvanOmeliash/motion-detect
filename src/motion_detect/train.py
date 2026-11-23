"""Training loop for TinyUNet: BCE+Dice loss, pixel metrics, sampler groups.

Checkpoint format is frozen and matches the original notebooks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .dataset import (
    DEFAULT_MAX_WIDTH,
    MotionDataset,
    TRAIN_CROP_SIZE,
    TRAIN_GROUP_RATIOS,
)
from .model import default_device

logger = logging.getLogger(__name__)

LOSS_POS_WEIGHT = 20.0
LOSS_POS_WEIGHT_NO_EMPTY = 15.0
ALLOW_TRAIN_WITHOUT_EMPTY = False
ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES = False
MIN_EMPTY_FRACTION_WARNING = 0.05
HARD_NEGATIVE_REMINE_EVERY = 3

class BCEDiceLoss(nn.Module):
    """
    Weighted BCE(logits, target) + plain Dice.

    weight == 1: the pixel participates in BCE.
    weight == 0: the pixel is in the ignore-band and does not participate in BCE.
    Dice is intentionally computed on the binary target without the ignore-band.
    """
    def __init__(self, smooth=1e-6, pos_weight=LOSS_POS_WEIGHT):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor([float(pos_weight)], dtype=torch.float32),
        )
        self.smooth = float(smooth)

    def forward(self, logits, target, weight=None):
        if weight is None:
            # Backward compatibility for old external loss calls.
            weight = torch.ones_like(target)
        if weight.shape != target.shape:
            raise ValueError(
                f"weight shape {tuple(weight.shape)} != "
                f"target shape {tuple(target.shape)}"
            )

        weight = weight.to(device=logits.device, dtype=logits.dtype)
        weight = weight.clamp_(0.0, 1.0)

        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        bce = (
            (bce_per_pixel * weight).sum()
            / weight.sum().clamp(min=1.0)
        )

        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + self.smooth) / (union + self.smooth)

        total = bce + dice.mean()
        return total, bce.item(), dice.mean().item()

class PixelMetrics:
    """Accumulates pixel TP/FP/FN per epoch, computes IoU and F1."""
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, logits, target):
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).long()
            tgt  = target.long()
            self.tp += (pred * tgt).sum().item()
            self.fp += (pred * (1 - tgt)).sum().item()
            self.fn += ((1 - pred) * tgt).sum().item()

    @property
    def iou(self):
        d = self.tp + self.fp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self):
        d = 2*self.tp + self.fp + self.fn
        return 2*self.tp / d if d > 0 else 0.0

    def reset(self):
        self.tp = self.fp = self.fn = 0

# ================================================================


def resolve_training_pos_weight(
    train_ds,
    group_ratios=TRAIN_GROUP_RATIOS,
    pos_weight=LOSS_POS_WEIGHT,
    pos_weight_no_empty=LOSS_POS_WEIGHT_NO_EMPTY,
    allow_train_without_empty=ALLOW_TRAIN_WITHOUT_EMPTY,
    allow_train_without_hard_negatives=ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES,
    min_empty_fraction_warning=MIN_EMPTY_FRACTION_WARNING,
):
    """
    Validate the three sampler groups and return the lower pos_weight used
    together with hard negatives.
    """
    counts = train_ds.group_counts()
    total_count = max(1, len(train_ds))
    random_fraction = counts["random_negative"] / total_count

    logger.info(
        "Train frame balance: "
        f"positive={counts['positive']}, "
        f"hard_negative={counts['hard_negative']}, "
        f"random_negative={counts['random_negative']}"
    )

    if counts["positive"] == 0:
        raise RuntimeError(
            "Train dataset contains no positive samples."
        )

    if counts["random_negative"] == 0:
        if not allow_train_without_empty:
            raise RuntimeError(
                "Train dataset contains no random-negative frames. "
                "Regenerate the dataset or explicitly set "
                "ALLOW_TRAIN_WITHOUT_EMPTY=True."
            )
        logger.info(
            "WARNING: no random negatives; "
            f"pos_weight={pos_weight_no_empty:.1f}"
        )
        return float(pos_weight_no_empty)

    if counts["hard_negative"] == 0:
        if not allow_train_without_hard_negatives:
            raise RuntimeError(
                "No approved hard negatives. Run Phase 1.2 mining, review "
                "the first ~200 crops, then approve them before training."
            )
        logger.info(
            "WARNING: ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES=True. "
            "The 60/25/15 target will be renormalized over available groups."
        )

    if random_fraction < float(min_empty_fraction_warning):
        logger.info(
            "WARNING: random-negative frames are below "
            f"{min_empty_fraction_warning:.0%} of stored samples."
        )

    logger.info(
        "WeightedRandomSampler target: "
        + ", ".join(
            f"{group}={ratio:.0%}"
            for group, ratio in group_ratios.items()
        )
        + f"; pos_weight={pos_weight:.1f}"
    )
    return float(pos_weight)

def _build_training_loader(
    train_dir,
    batch_size,
    group_ratios=TRAIN_GROUP_RATIOS,
    allow_single_class=ALLOW_TRAIN_WITHOUT_EMPTY,
    allow_missing_hard_negative=ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES,
    device=None,
):
    train_ds = MotionDataset(
        train_dir,
        augment=True,
        group_ratios=group_ratios,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"Empty train dataset: {train_dir}")

    train_weights = torch.as_tensor(
        train_ds.sampling_weights(
            allow_single_class=allow_single_class,
            allow_missing_hard_negative=allow_missing_hard_negative,
        ),
        dtype=torch.double,
    )
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    if device is None:
        device = default_device()
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    return train_ds, train_loader

def train_unet(
    model,
    train_dir,
    val_dir,
    epochs=15,
    batch_size=4,
    lr=1e-3,
    save_dir="checkpoints",
    hard_negative_refresh_callback=None,
    hard_negative_remine_every=HARD_NEGATIVE_REMINE_EVERY,
    device=None,
    group_ratios=TRAIN_GROUP_RATIOS,
    allow_train_without_empty=ALLOW_TRAIN_WITHOUT_EMPTY,
    allow_train_without_hard_negatives=ALLOW_TRAIN_WITHOUT_HARD_NEGATIVES,
    pos_weight=LOSS_POS_WEIGHT,
    pos_weight_no_empty=LOSS_POS_WEIGHT_NO_EMPTY,
    min_empty_fraction_warning=MIN_EMPTY_FRACTION_WARNING,
    processing_max_width=DEFAULT_MAX_WIDTH,
    crop_size=TRAIN_CROP_SIZE,
):
    """
    Train with three sampler groups.

    hard_negative_refresh_callback(model, epoch) may mine/approve a new round.
    If it returns True, the train dataset and sampler are rebuilt before the
    next epoch. The callback is intentionally optional because the first round
    must be manually reviewed.
    """
    if device is None:
        device = default_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    train_ds, train_loader = _build_training_loader(
        train_dir,
        batch_size=batch_size,
        group_ratios=group_ratios,
        allow_single_class=allow_train_without_empty,
        allow_missing_hard_negative=allow_train_without_hard_negatives,
        device=device,
    )
    val_ds = MotionDataset(
        val_dir,
        augment=False,
        group_ratios=None,
    )

    if len(val_ds) == 0:
        raise RuntimeError(f"Empty validation dataset: {val_dir}")

    effective_pos_weight = resolve_training_pos_weight(
        train_ds,
        group_ratios=group_ratios,
        pos_weight=pos_weight,
        pos_weight_no_empty=pos_weight_no_empty,
        allow_train_without_empty=allow_train_without_empty,
        allow_train_without_hard_negatives=allow_train_without_hard_negatives,
        min_empty_fraction_warning=min_empty_fraction_warning,
    )

    pin_memory = device == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    criterion = BCEDiceLoss(
        pos_weight=effective_pos_weight,
    ).to(device)

    best_f1 = -1.0
    best_path = os.path.join(save_dir, "best_unet.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_metrics = PixelMetrics()
        train_loss = 0.0

        for x, y, weight in tqdm(train_loader, desc=f"Ep{epoch:02d} train"):
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            weight = weight.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, _, _ = criterion(logits, y, weight)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_metrics.update(logits, y)

        model.eval()
        val_metrics = PixelMetrics()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, weight in tqdm(val_loader, desc=f"Ep{epoch:02d} val"):
                x = x.to(device, non_blocking=pin_memory)
                y = y.to(device, non_blocking=pin_memory)
                weight = weight.to(device, non_blocking=pin_memory)

                logits = model(x)
                loss, _, _ = criterion(logits, y, weight)
                val_loss += loss.item()
                val_metrics.update(logits, y)

        scheduler.step()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        logger.info(
            f"Ep{epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"tr_iou={train_metrics.iou:.4f} "
            f"tr_f1={train_metrics.f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_iou={val_metrics.iou:.4f} "
            f"val_f1={val_metrics.f1:.4f}"
        )

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            counts = train_ds.group_counts()
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_f1": best_f1,
                    "processing_max_width": processing_max_width,
                    "crop_size": crop_size,
                    "pos_weight": effective_pos_weight,
                    "train_group_counts": counts,
                    "train_group_ratios_target": dict(group_ratios),
                    "train_positive_frames": counts["positive"],
                    "train_hard_negative_frames": counts["hard_negative"],
                    "train_random_negative_frames": counts["random_negative"],
                    "target_source": "gt_bbox_mask_with_ignore_band",
                    "compensation_pipeline": "classical_compensation.CompensationPipeline",
                    "temporal_step": 1,
                },
                best_path,
            )
            logger.info(f"  best saved: val_f1={best_f1:.4f}")

        should_refresh = (
            hard_negative_refresh_callback is not None
            and int(hard_negative_remine_every) > 0
            and epoch < epochs
            and epoch % int(hard_negative_remine_every) == 0
        )
        if should_refresh:
            logger.info(
                f"Hard-negative refresh after epoch {epoch} "
                f"(cadence={hard_negative_remine_every})"
            )
            dataset_changed = bool(
                hard_negative_refresh_callback(model, epoch)
            )
            if dataset_changed:
                train_ds, train_loader = _build_training_loader(
                    train_dir,
                    batch_size=batch_size,
                    group_ratios=group_ratios,
                    allow_single_class=allow_train_without_empty,
                    allow_missing_hard_negative=allow_train_without_hard_negatives,
                    device=device,
                )
                # Validate the new group composition immediately.
                resolve_training_pos_weight(
                    train_ds,
                    group_ratios=group_ratios,
                    pos_weight=pos_weight,
                    pos_weight_no_empty=pos_weight_no_empty,
                    allow_train_without_empty=allow_train_without_empty,
                    allow_train_without_hard_negatives=allow_train_without_hard_negatives,
                    min_empty_fraction_warning=min_empty_fraction_warning,
                )
                logger.info(
                    "Train sampler rebuilt with:",
                    train_ds.group_counts(),
                )
            else:
                logger.info(
                    "Refresh callback did not approve new samples; "
                    "the current sampler remains active."
                )

    return best_path

# ================================================================

# ================================================================
