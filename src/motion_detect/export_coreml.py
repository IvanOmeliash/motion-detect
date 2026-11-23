"""Optional Core ML export (FP16 ML Program) and Core ML probability backend.

coremltools is imported lazily and only when export/prediction is requested;
Core ML prediction itself requires macOS.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .compensation import (
    DEFAULT_COMPENSATION_CONFIG,
    CompensationPipeline,
    preprocess_frame,
)
from .inference import load_frozen_pytorch_backend
from .model import build_inference_model, default_device

logger = logging.getLogger(__name__)

DEFAULT_MAX_WIDTH = 640

def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def resolve_coreml_input_shape(video_path, max_width: int = DEFAULT_MAX_WIDTH):
    """Read one frame and derive the fixed CHW shape used by CompensationPipeline."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode first frame: {video_path}")

    gray, _ = preprocess_frame(
        frame,
        int(max_width),
        gray_first=True,
    )
    height, width = map(int, gray.shape)
    if height % 8 != 0 or width % 8 != 0:
        raise RuntimeError(
            f"TinyUNet requires H/W divisible by 8, got {(height, width)}"
        )
    return (1, 3, height, width)

class TinyUNetProbabilityWrapper(nn.Module):
    """Core ML export wrapper: fixed-shape input -> probability map."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, frames):
        return torch.sigmoid(self.model(frames))

def export_tinyunet_coreml_fp16(
    checkpoint_path,
    output_path,
    input_shape,
    architecture_frozen,
    traced_output_path=None,
):
    """
    Export the frozen TinyUNet to an FP16 Core ML ML Program.

    The converter is allowed to use every available compute unit. The package
    retains a float32 MLMultiArray boundary for broad compatibility while
    weights/intermediate execution use FP16.
    """
    if not bool(architecture_frozen):
        raise RuntimeError(
            "Core ML export blocked: set PHASE4_ARCHITECTURE_FROZEN=True "
            "only after Phase 1 training and architecture selection are final."
        )

    ct = install_coremltools_if_needed()
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.suffix != ".mlpackage":
        raise ValueError("Core ML ML Program must be saved as .mlpackage")

    shape = tuple(map(int, input_shape))
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError(f"Expected input shape (1,3,H,W), got {shape}")
    if shape[2] % 8 or shape[3] % 8:
        raise ValueError("Core ML input H and W must be divisible by 8")

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
    )
    state_dict = checkpoint.get("model", checkpoint)

    pytorch_model = build_inference_model(
        state_dict,
        device=torch.device("cpu"),
        fuse_batchnorm=True,
    )
    wrapper = TinyUNetProbabilityWrapper(pytorch_model).eval()

    example_input = torch.zeros(shape, dtype=torch.float32)
    with torch.inference_mode():
        traced_model = torch.jit.trace(
            wrapper,
            example_input,
            strict=True,
        )
        traced_model = torch.jit.freeze(traced_model.eval())

    if traced_output_path is not None:
        traced_output_path = Path(traced_output_path)
        traced_output_path.parent.mkdir(parents=True, exist_ok=True)
        traced_model.save(str(traced_output_path))

    skip_load = platform.system() != "Darwin"
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="frames",
                shape=shape,
                dtype=np.float32,
            )
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=skip_load,
    )

    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name

    mlmodel.author = "CV_school_our1"
    mlmodel.short_description = (
        "TinyUNet motion probability map; FP16 ML Program"
    )
    mlmodel.version = str(checkpoint.get("epoch", "frozen"))
    mlmodel.user_defined_metadata.update({
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": str(checkpoint.get("epoch")),
        "state_dict_sha256": _state_dict_sha256(state_dict),
        "input_shape": json.dumps(shape),
        "output_semantics": "sigmoid probability map",
        "compute_precision": "FLOAT16",
        "requested_compute_units": "ALL",
    })

    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    manifest = {
        "coremltools_version": str(ct.__version__),
        "package_path": str(output_path),
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": list(shape),
        "compute_precision": "FLOAT16",
        "compute_units": "ALL",
        "minimum_deployment_target": "macOS13",
        "state_dict_sha256": _state_dict_sha256(state_dict),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "conversion_host": platform.platform(),
        "model_loaded_during_conversion": not skip_load,
    }
    manifest_path = output_path.parent / (
        output_path.stem + "_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Core ML package:", output_path)
    logger.info("Manifest:", manifest_path)
    logger.info("Input:", input_name, shape)
    logger.info("Output:", output_name)
    return mlmodel, manifest

class CoreMLProbabilityBackend:
    """
    Core ML Python backend. Prediction is available only on macOS.

    Important: Core ML's Python predict() API does not expose ANE/GPU transfer
    time separately from graph execution. Stage 8 is therefore intentionally
    labelled as an opaque Core ML predict block, not pure compute.
    """
    name = "coreml"
    timing_semantics = {
        "7_h2d_or_input_prepare_ms": (
            "CPU float32 batch materialization before Core ML predict"
        ),
        "8_forward_or_coreml_predict_ms": (
            "opaque MLModel.predict call: scheduling + transfers + compute"
        ),
        "9_sigmoid_threshold_ms": (
            "CPU threshold only; sigmoid is already inside the Core ML graph"
        ),
        "10_d2h_or_output_materialize_ms": (
            "Core ML output MLMultiArray/array materialization to NumPy"
        ),
    }

    def __init__(
        self,
        package_path,
        compute_units="ALL",
        input_name=None,
        output_name=None,
    ):
        if platform.system() != "Darwin":
            raise RuntimeError(
                "Core ML model prediction requires macOS. "
                "Export may be prepared elsewhere, but verification and "
                "Core ML inference must run on the target Mac."
            )

        import coremltools as ct

        unit_name = str(compute_units).upper()
        if not hasattr(ct.ComputeUnit, unit_name):
            raise ValueError(f"Unknown Core ML compute unit: {compute_units}")
        unit = getattr(ct.ComputeUnit, unit_name)

        self.model = ct.models.MLModel(
            str(package_path),
            compute_units=unit,
        )
        spec = self.model.get_spec()
        self.input_name = input_name or spec.description.input[0].name
        self.output_name = output_name or spec.description.output[0].name
        self.compute_units = unit_name

    def warmup(self, input_chw, iterations=30, threshold=0.5):
        iterations = max(0, int(iterations))
        for _ in range(iterations):
            self.infer(
                input_chw,
                threshold=threshold,
                return_probability=False,
                profile_stages=False,
            )
        return {
            "iterations": iterations,
            "backend": self.name,
            "compute_units": self.compute_units,
        }

    def infer(
        self,
        input_chw,
        threshold,
        return_probability=False,
        profile_stages=False,
    ):
        timings = _empty_backend_timings()
        total_started = time.perf_counter()

        stage_started = time.perf_counter()
        batch = np.ascontiguousarray(
            np.asarray(input_chw, dtype=np.float32)[None],
        )
        timings["7_h2d_or_input_prepare_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        prediction = self.model.predict({
            self.input_name: batch,
        })
        timings["8_forward_or_coreml_predict_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        probability = np.asarray(
            prediction[self.output_name],
            dtype=np.float32,
        ).squeeze()
        timings["10_d2h_or_output_materialize_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        if probability.ndim != 2:
            raise RuntimeError(
                f"Unexpected Core ML output shape: "
                f"{np.asarray(prediction[self.output_name]).shape}"
            )

        stage_started = time.perf_counter()
        binary = (
            probability > float(threshold)
        ).astype(np.uint8) * np.uint8(255)
        timings["9_sigmoid_threshold_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0
        timings["inference_total_ms"] = (
            time.perf_counter() - total_started
        ) * 1000.0

        return {
            "binary_mask": binary,
            "probability": probability if return_probability else None,
            "timings_ms": timings,
            "timing_semantics": dict(self.timing_semantics),
            "profile_stages": bool(profile_stages),
        }

def verify_coreml_against_pytorch(
    video_path,
    checkpoint_path,
    coreml_package_path,
    frame_count=100,
    threshold=0.5,
    compensation_config=None,
    report_path=None,
    device: Optional[str] = None,
    max_width: int = DEFAULT_MAX_WIDTH,
):
    """
    Compare FP16 Core ML probabilities with the frozen PyTorch model.
    Must run on macOS because coremltools predict() calls the Core ML framework.
    """
    if platform.system() != "Darwin":
        raise RuntimeError(
            "Core ML MAE verification must run on macOS."
        )
    if device is None:
        device = default_device()

    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    torch_backend = load_frozen_pytorch_backend(
        checkpoint_path,
        device=device,
    )
    coreml_backend = CoreMLProbabilityBackend(
        coreml_package_path,
        compute_units="ALL",
    )

    config = dict(DEFAULT_COMPENSATION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = int(max_width)
    pipeline = CompensationPipeline(config, step=1)

    total_abs_error = 0.0
    total_pixels = 0
    max_abs_error = 0.0
    frame_mae = []
    threshold_disagreement_pixels = 0

    compared = 0
    with torch.inference_mode():
        for sample in pipeline.stream_video(video_path):
            if not sample.valid:
                continue

            input_chw = np.asarray(
                sample.tensor[:3],
                dtype=np.float32,
            )
            torch_result = torch_backend.infer(
                input_chw,
                threshold=threshold,
                return_probability=True,
            )
            coreml_result = coreml_backend.infer(
                input_chw,
                threshold=threshold,
                return_probability=True,
            )

            torch_probability = torch_result["probability"]
            coreml_probability = coreml_result["probability"]
            if torch_probability.shape != coreml_probability.shape:
                raise RuntimeError(
                    f"Output shape mismatch: "
                    f"{torch_probability.shape} vs "
                    f"{coreml_probability.shape}"
                )

            absolute_error = np.abs(
                torch_probability - coreml_probability
            )
            current_mae = float(absolute_error.mean())
            frame_mae.append(current_mae)
            total_abs_error += float(absolute_error.sum())
            total_pixels += int(absolute_error.size)
            max_abs_error = max(
                max_abs_error,
                float(absolute_error.max()),
            )

            threshold_disagreement_pixels += int(
                np.count_nonzero(
                    (torch_probability > float(threshold))
                    !=
                    (coreml_probability > float(threshold))
                )
            )

            compared += 1
            if compared >= frame_count:
                break

    if compared == 0:
        raise RuntimeError("No valid compensated frames were available")

    report = {
        "frames_compared": compared,
        "global_mae": float(
            total_abs_error / max(1, total_pixels)
        ),
        "mean_frame_mae": float(np.mean(frame_mae)),
        "median_frame_mae": float(np.median(frame_mae)),
        "p95_frame_mae": float(np.percentile(frame_mae, 95)),
        "max_abs_error": float(max_abs_error),
        "threshold": float(threshold),
        "threshold_disagreement_rate": float(
            threshold_disagreement_pixels
            / max(1, total_pixels)
        ),
        "coreml_compute_units": "ALL",
        "device": str(device),
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info(json.dumps(report, indent=2, ensure_ascii=False))
    return report