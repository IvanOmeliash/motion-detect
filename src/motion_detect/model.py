"""TinyUNet model definition and inference-only Conv+BN fusion.

Architecture and checkpoint layout are frozen: training checkpoints store the
unfused state dict; fusion is applied only to a separately built eval model.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSE_BATCHNORM_FOR_INFERENCE = True

logger = logging.getLogger(__name__)

def default_device() -> str:
    """Pick the best available torch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class DoubleConv(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class TinyUNet(nn.Module):
    """
    Input:  (B, 3, H, W) — three aligned grayscale frames as channels:
             channel 0 = Warp(I_{t-2}) / 255
             channel 1 = Warp(I_{t-1}) / 255
             channel 2 =    I_t        / 255
    Output: (B, 1, H, W) — LOGITS (sigmoid is applied separately at inference)
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3,  16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 48)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bott = DoubleConv(48, 64)
        # Decoder
        self.dec3 = DoubleConv(64 + 48, 48)
        self.dec2 = DoubleConv(48 + 32, 32)
        self.dec1 = DoubleConv(32 + 16, 16)
        # Head — logits, no activation
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                                   # (B,16,H,W)
        e2 = self.enc2(self.pool(e1))                       # (B,32,H/2,W/2)
        e3 = self.enc3(self.pool(e2))                       # (B,48,H/4,W/4)
        b  = self.bott(self.pool(e3))                       # (B,64,H/8,W/8)

        u3 = F.interpolate(b,  size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))          # (B,48,H/4,W/4)

        u2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))          # (B,32,H/2,W/2)

        u1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))          # (B,16,H,W)

        return self.head(d1)                                  # (B,1,H,W)

# ================================================================
# Conv + BatchNorm fusion for inference only
# ================================================================

def fuse_tiny_unet_batchnorm(model, inplace=False):
    """
    Fold every Conv2d + BatchNorm2d pair in TinyUNet into one Conv2d.

    Important:
      * call only after loading checkpoint weights;
      * model must be in eval mode;
      * use the returned model for inference only;
      * the training model/checkpoint layout is not modified.
    """
    fused_model = model if inplace else copy.deepcopy(model)
    fused_model.eval()

    fused_pairs = 0
    for module in fused_model.modules():
        if not isinstance(module, DoubleConv):
            continue

        block = module.block
        for conv_index, bn_index in ((0, 1), (3, 4)):
            conv = block[conv_index]
            bn = block[bn_index]

            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                block[conv_index] = torch.nn.utils.fusion.fuse_conv_bn_eval(
                    conv,
                    bn,
                )
                block[bn_index] = nn.Identity()
                fused_pairs += 1

    remaining_bn = sum(
        isinstance(module, nn.BatchNorm2d)
        for module in fused_model.modules()
    )
    if remaining_bn != 0:
        raise RuntimeError(
            f"BatchNorm fusion incomplete: {remaining_bn} BN layers remain"
        )

    fused_model._fused_batchnorm_pairs = int(fused_pairs)
    return fused_model

def build_inference_model(
    state_dict,
    device: Optional[str] = None,
    fuse_batchnorm: bool = FUSE_BATCHNORM_FOR_INFERENCE,
):
    """Build an eval-only TinyUNet, fuse BN on CPU, then move to device."""
    if device is None:
        device = default_device()
    inference_model = TinyUNet()
    inference_model.load_state_dict(state_dict)
    inference_model.eval()

    if fuse_batchnorm:
        inference_model = fuse_tiny_unet_batchnorm(
            inference_model,
            inplace=True,
        )
        logger.info(
            "Fused Conv+BN pairs: %s",
            inference_model._fused_batchnorm_pairs,
        )

    return inference_model.to(device).eval()

def phase2_fusion_equivalence_check(
    state_dict,
    input_shape=(1, 3, 256, 256),
    atol=1e-4,
    rtol=1e-4,
):
    """CPU correctness check: unfused and fused logits must stay equivalent."""
    reference = build_inference_model(
        state_dict,
        device="cpu",
        fuse_batchnorm=False,
    )
    fused = build_inference_model(
        state_dict,
        device="cpu",
        fuse_batchnorm=True,
    )

    generator = torch.Generator(device="cpu").manual_seed(12345)
    sample = torch.rand(input_shape, generator=generator)

    with torch.inference_mode():
        reference_logits = reference(sample)
        fused_logits = fused(sample)

    max_abs_error = float(
        (reference_logits - fused_logits).abs().max().item()
    )
    is_close = bool(torch.allclose(
        reference_logits,
        fused_logits,
        atol=float(atol),
        rtol=float(rtol),
    ))
    if not is_close:
        raise AssertionError(
            "Conv+BN fusion changed logits beyond tolerance: "
            f"max_abs_error={max_abs_error:.8g}"
        )

    return {
        "fused_pairs": int(fused._fused_batchnorm_pairs),
        "remaining_batchnorm": sum(
            isinstance(module, nn.BatchNorm2d)
            for module in fused.modules()
        ),
        "max_abs_error": max_abs_error,
        "allclose": is_close,
    }

# ================================================================
