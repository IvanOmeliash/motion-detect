"""Fused vs unfused TinyUNet must produce matching logits on CPU."""

from __future__ import annotations

import torch

from motion_detect.model import (
    TinyUNet,
    build_inference_model,
    phase2_fusion_equivalence_check,
)


def _random_state_dict(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    model = TinyUNet()
    for param in model.parameters():
        param.data = torch.randn(param.shape, generator=generator) * 0.1
    return model.state_dict()


def test_fused_matches_unfused_logits() -> None:
    state_dict = _random_state_dict()
    result = phase2_fusion_equivalence_check(state_dict, input_shape=(1, 3, 256, 256))
    assert result["allclose"], result
    assert result["fused_pairs"] == 14  # 7 DoubleConv blocks x 2 Conv+BN pairs
    assert result["remaining_batchnorm"] == 0
    assert result["max_abs_error"] < 1e-4


def test_build_inference_model_eval_mode() -> None:
    state_dict = _random_state_dict(seed=1)
    model = build_inference_model(state_dict, device="cpu", fuse_batchnorm=True)
    assert not model.training
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_unfused_state_dict_still_loads() -> None:
    # Checkpoints store the UNFUSED layout; loading must keep working.
    state_dict = _random_state_dict(seed=2)
    model = build_inference_model(state_dict, device="cpu", fuse_batchnorm=False)
    model.load_state_dict(state_dict)  # strict
