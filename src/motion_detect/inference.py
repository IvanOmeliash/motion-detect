"""Production inference: video -> compensated 3-frame tensor -> TinyUNet -> blobs -> SORT.

Includes the bounded two-thread producer/consumer pipeline and the PyTorch
probability backend used by calibration.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch

from .compensation import CompensationPipeline, DEFAULT_COMPENSATION_CONFIG
from .model import build_inference_model, default_device
from .postprocess import (
    LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
    _prepare_unet_input,
    _probability_to_blobs,
    _scaled_detection_geometry,
)
from .tracker import SortTracker

logger = logging.getLogger(__name__)

DEFAULT_MAX_WIDTH = 640
USE_PREALLOCATED_UNET_INPUT = True

def predict_video_bboxes(
    input_path,
    params,
    unet_model,
    unet_threshold=0.5,
    compensation_step=1,
    compensation_config=None,
    return_size=(1280, 720),
    apply_validity_mask=True,
    apply_platform_suppression=None,
    platform_suppression_zone=LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
    checkpoint_blur=False,
    diagnostics_callback=None,
    max_width: int = DEFAULT_MAX_WIDTH,
    device: Optional[str] = None,
):
    """
    Production inference path:

      video -> CompensationPipeline -> first 3 channels -> TinyUNet
            -> probability -> shared contour filter -> SORT -> bbox dictionary

    diagnostics_callback is optional and receives one dictionary per valid frame
    with probability map, accepted blobs and inlier_ratio. This keeps mining and
    inference on exactly the same post-processing path without storing all maps.
    """
    if unet_model is None:
        raise ValueError("unet_model is required for the production path")
    if device is None:
        device = default_device()
    if int(compensation_step) <= 0:
        raise ValueError("compensation_step must be positive")

    config = dict(DEFAULT_COMPENSATION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = int(max_width)

    if apply_platform_suppression is None:
        apply_platform_suppression = bool(
            params.get("apply_platform_suppression", True)
        )
    active_platform_zone = (
        platform_suppression_zone
        if apply_platform_suppression
        else None
    )

    pipeline = CompensationPipeline(config, step=int(compensation_step))
    unet_model.eval()

    tracker = None
    predictions = {}
    processing_shape = None
    geometry = None

    skipped_low_confidence = 0
    invalid_samples = 0
    input_buffer = None

    with torch.inference_mode():
        for sample in pipeline.stream_video(input_path):
            frame_idx = int(sample.frame_index) + 1

            if not sample.valid:
                invalid_samples += 1
                if tracker is not None:
                    tracker.update([])
                continue

            tensor = np.asarray(sample.tensor)
            if tensor.ndim != 3 or tensor.shape[0] != 4:
                raise RuntimeError(
                    f"Unexpected classical tensor shape: {tensor.shape}"
                )

            _, height, width = tensor.shape
            if height % 8 != 0 or width % 8 != 0:
                raise RuntimeError(
                    f"TinyUNet requires H and W divisible by 8, got {(height, width)}"
                )

            if tracker is None:
                processing_shape = (height, width)
                geometry = _scaled_detection_geometry(
                    height,
                    width,
                    params,
                    return_size=return_size,
                )
                tracker = SortTracker(
                    max_age=params["max_age"],
                    min_hits=params["min_hits"],
                    iou_threshold=params["iou_thresh"],
                    max_dist=max(
                        1.0,
                        float(params["max_dist"]) * geometry["scale_x"],
                    ),
                )
            elif processing_shape != (height, width):
                raise RuntimeError(
                    f"Video changed processing size from "
                    f"{processing_shape} to {(height, width)}"
                )

            min_ratio = params.get("inlier_thresh")
            if (
                min_ratio is not None
                and float(sample.inlier_ratio) < float(min_ratio)
            ):
                skipped_low_confidence += 1
                tracker.update([])
                continue

            source_channels = tensor[:3]
            needs_copy_buffer = (
                checkpoint_blur
                or source_channels.dtype != np.float32
                or not source_channels.flags.c_contiguous
            )
            if (
                USE_PREALLOCATED_UNET_INPUT
                and needs_copy_buffer
                and input_buffer is None
            ):
                input_buffer = np.empty(
                    (3, height, width),
                    dtype=np.float32,
                )

            x_np = _prepare_unet_input(
                tensor,
                checkpoint_blur=checkpoint_blur,
                out=(
                    input_buffer
                    if USE_PREALLOCATED_UNET_INPUT and needs_copy_buffer
                    else None
                ),
            )
            x_t = torch.from_numpy(x_np).unsqueeze(0).to(
                device,
                non_blocking=(device == "cuda"),
            )

            logits = unet_model(x_t)
            probability_t = torch.sigmoid(logits)[0, 0]

            # Phase 2.1: threshold on CUDA/MPS and copy only uint8 to CPU.
            # HxW float32 -> HxW uint8 reduces transfer by 4x.
            prob_u8_t = (
                probability_t > float(unet_threshold)
            ).to(torch.uint8).mul_(255)
            thresh = prob_u8_t.cpu().numpy()

            # Full float probability is required only by offline diagnostics /
            # hard-negative mining for prob_max ranking.
            prob = None
            if diagnostics_callback is not None:
                prob = probability_t.detach().cpu().numpy()

            thresh, blobs = _probability_to_blobs(
                prob=prob,
                binary_mask=thresh,
                tensor=tensor,
                threshold=unet_threshold,
                area_min=geometry["area_min"],
                area_max=geometry["area_max"],
                bbox_padding=geometry["bbox_padding"],
                apply_validity_mask=apply_validity_mask,
                platform_suppression_zone=active_platform_zone,
            )

            if diagnostics_callback is not None:
                diagnostics_callback({
                    "frame_index": frame_idx,
                    "prob": prob,
                    "binary_mask": thresh,
                    "blobs": blobs,
                    "tensor": tensor,
                    "inlier_ratio": float(sample.inlier_ratio),
                    "processing_shape": (height, width),
                })

            tracks = tracker.update([blob["bbox"] for blob in blobs])
            frame_predictions = []
            for track in tracks:
                if (
                    track.hits < params["min_hits"]
                    or track.time_since_update != 0
                ):
                    continue

                x1, y1, x2, y2 = track.bbox
                frame_predictions.append([
                    int(round(
                        np.clip(x1, 0, width)
                        * geometry["output_scale_x"]
                    )),
                    int(round(
                        np.clip(y1, 0, height)
                        * geometry["output_scale_y"]
                    )),
                    int(round(
                        np.clip(x2, 0, width)
                        * geometry["output_scale_x"]
                    )),
                    int(round(
                        np.clip(y2, 0, height)
                        * geometry["output_scale_y"]
                    )),
                ])

            if frame_predictions:
                predictions[frame_idx] = frame_predictions

    logger.info("Classical pipeline stats:", pipeline.stats)
    logger.info("Motion-estimation calls:", pipeline.motion_estimation_calls)
    logger.info("Invalid/warmup samples:", invalid_samples)
    logger.info("Skipped by inlier threshold:", skipped_low_confidence)
    return predictions

# ================================================================

# ================================================================

# ================================================================
# Two-stage CPU -> GPU/ANE pipeline
# ================================================================

_PIPELINE_SENTINEL = object()

def _put_until_stopped(target_queue, item, stop_event):
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False

def _get_until_stopped(source_queue, stop_event):
    while not stop_event.is_set():
        try:
            return source_queue.get(timeout=0.1)
        except queue.Empty:
            continue
    raise RuntimeError("Pipeline stopped")

def _summarize_ms(values):
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

def _warmup_backend_from_video(
    backend,
    input_path,
    threshold,
    iterations,
    compensation_step,
    compensation_config,
    max_width: int = DEFAULT_MAX_WIDTH,
):
    """
    Warm the exact backend/input shape before any benchmark timer starts.

    A separate short CompensationPipeline pass is used, then the measured
    pipeline is opened from frame zero. Warmup frames never enter predictions
    or timing accumulators.
    """
    iterations = max(0, int(iterations))
    if iterations == 0:
        return {
            "iterations": 0,
            "valid_sample_found": False,
        }

    config = dict(DEFAULT_COMPENSATION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = int(max_width)

    warmup_pipeline = CompensationPipeline(
        config,
        step=int(compensation_step),
    )
    for sample in warmup_pipeline.stream_video(input_path):
        if not sample.valid:
            continue
        report = backend.warmup(
            np.asarray(sample.tensor[:3]),
            iterations=iterations,
            threshold=threshold,
        )
        report.update({
            "valid_sample_found": True,
            "input_shape": list(np.asarray(sample.tensor[:3]).shape),
            "excluded_from_measurement": True,
        })
        return report

    raise RuntimeError(
        "Could not find a valid compensated frame for backend warmup"
    )

def predict_video_bboxes_pipelined(
    input_path,
    params,
    backend,
    unet_threshold=0.5,
    compensation_step=1,
    compensation_config=None,
    return_size=(1280, 720),
    apply_validity_mask=True,
    apply_platform_suppression=None,
    platform_suppression_zone=LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
    queue_size=3,
    diagnostics_callback=None,
    backend_warmup_iterations=30,
    profile_backend_stages=True,
    opencv_threads=None,
    max_width: int = DEFAULT_MAX_WIDTH,
):
    """
    Bounded two-stage inference with excluded backend warmup.

      producer: decode + classical compensation on CPU
      consumer: backend inference + postprocess + SORT

    For PyTorch, profile_backend_stages=True synchronizes CUDA/MPS around stages
    7–10. This is diagnostic mode and can slightly reduce throughput. Run a
    second pass with False when measuring deployment throughput only.
    """
    queue_size = int(queue_size)
    if queue_size not in (2, 3):
        raise ValueError("queue_size must be 2 or 3")
    if backend is None:
        raise ValueError("backend is required")

    previous_opencv_threads = int(cv2.getNumThreads())
    requested_opencv_threads = (
        None if opencv_threads is None else int(opencv_threads)
    )
    if requested_opencv_threads is not None:
        cv2.setNumThreads(requested_opencv_threads)
    actual_opencv_threads = int(cv2.getNumThreads())

    if apply_platform_suppression is None:
        apply_platform_suppression = bool(
            params.get("apply_platform_suppression", True)
        )
    active_platform_zone = (
        platform_suppression_zone
        if apply_platform_suppression
        else None
    )

    try:
        warmup_report = _warmup_backend_from_video(
            backend=backend,
            input_path=input_path,
            threshold=unet_threshold,
            iterations=backend_warmup_iterations,
            compensation_step=compensation_step,
            compensation_config=compensation_config,
            max_width=max_width,
        )
    except BaseException:
        if requested_opencv_threads is not None:
            cv2.setNumThreads(previous_opencv_threads)
        raise

    config = dict(DEFAULT_COMPENSATION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = int(max_width)
    pipeline = CompensationPipeline(
        config,
        step=int(compensation_step),
    )

    ready_queue = queue.Queue(maxsize=queue_size)
    free_slots = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()
    producer_errors = []

    producer_blocked_seconds = 0.0
    produced_valid = 0

    def producer():
        nonlocal producer_blocked_seconds, produced_valid
        slot_shape = None
        try:
            for sample in pipeline.stream_video(input_path):
                if stop_event.is_set():
                    break

                frame_index = int(sample.frame_index) + 1
                if not sample.valid:
                    packet = {
                        "frame_index": frame_index,
                        "valid": False,
                        "inlier_ratio": float(sample.inlier_ratio),
                        "skip_reason": sample.skip_reason,
                        "slot": None,
                        "ready_at": time.perf_counter(),
                    }
                    if not _put_until_stopped(
                        ready_queue,
                        packet,
                        stop_event,
                    ):
                        return
                    continue

                tensor = np.asarray(sample.tensor)
                if tensor.ndim != 3 or tensor.shape[0] != 4:
                    raise RuntimeError(
                        f"Unexpected classical tensor: {tensor.shape}"
                    )

                if slot_shape is None:
                    slot_shape = tuple(tensor.shape)
                    for _ in range(queue_size):
                        free_slots.put(
                            np.empty(slot_shape, dtype=np.float32)
                        )
                elif tuple(tensor.shape) != slot_shape:
                    raise RuntimeError(
                        f"Processing shape changed: "
                        f"{slot_shape} -> {tensor.shape}"
                    )

                wait_started = time.perf_counter()
                slot = _get_until_stopped(
                    free_slots,
                    stop_event,
                )
                producer_blocked_seconds += (
                    time.perf_counter() - wait_started
                )

                np.copyto(slot, tensor, casting="unsafe")
                produced_valid += 1
                packet = {
                    "frame_index": frame_index,
                    "valid": True,
                    "inlier_ratio": float(sample.inlier_ratio),
                    "skip_reason": None,
                    "slot": slot,
                    "ready_at": time.perf_counter(),
                }
                if not _put_until_stopped(
                    ready_queue,
                    packet,
                    stop_event,
                ):
                    free_slots.put(slot)
                    return

        except BaseException as error:
            producer_errors.append(error)
        finally:
            if not stop_event.is_set():
                _put_until_stopped(
                    ready_queue,
                    _PIPELINE_SENTINEL,
                    stop_event,
                )

    producer_thread = threading.Thread(
        target=producer,
        name="classical-producer",
        daemon=True,
    )

    tracker = None
    geometry = None
    processing_shape = None
    predictions = {}

    invalid_samples = 0
    skipped_low_confidence = 0
    consumed_frames = 0
    queue_wait_ms = []
    postprocess_ms = []
    end_to_end_latency_ms = []
    backend_stage_ms = {
        key: []
        for key in _empty_backend_timings()
    }
    timing_semantics = None

    total_started = time.perf_counter()
    producer_thread.start()
    consumer_error = None

    try:
        while True:
            packet = ready_queue.get()
            if packet is _PIPELINE_SENTINEL:
                break

            consumed_frames += 1
            consumer_started = time.perf_counter()
            queue_wait_ms.append(
                (consumer_started - packet["ready_at"]) * 1000.0
            )

            if not packet["valid"]:
                invalid_samples += 1
                if tracker is not None:
                    tracker.update([])
                continue

            tensor = packet["slot"]
            try:
                _, height, width = tensor.shape
                if height % 8 or width % 8:
                    raise RuntimeError(
                        f"TinyUNet requires divisible-by-8 shape, "
                        f"got {(height, width)}"
                    )

                if tracker is None:
                    processing_shape = (height, width)
                    geometry = _scaled_detection_geometry(
                        height,
                        width,
                        params,
                        return_size=return_size,
                    )
                    tracker = SortTracker(
                        max_age=params["max_age"],
                        min_hits=params["min_hits"],
                        iou_threshold=params["iou_thresh"],
                        max_dist=max(
                            1.0,
                            float(params["max_dist"])
                            * geometry["scale_x"],
                        ),
                    )
                elif processing_shape != (height, width):
                    raise RuntimeError(
                        f"Shape changed: {processing_shape} "
                        f"-> {(height, width)}"
                    )

                min_ratio = params.get("inlier_thresh")
                if (
                    min_ratio is not None
                    and packet["inlier_ratio"] < float(min_ratio)
                ):
                    skipped_low_confidence += 1
                    tracker.update([])
                    continue

                backend_output = backend.infer(
                    tensor[:3],
                    threshold=unet_threshold,
                    return_probability=(
                        diagnostics_callback is not None
                    ),
                    profile_stages=profile_backend_stages,
                )
                current_timings = backend_output.get(
                    "timings_ms",
                    {},
                )
                for key in backend_stage_ms:
                    if key in current_timings:
                        backend_stage_ms[key].append(
                            float(current_timings[key])
                        )
                if timing_semantics is None:
                    timing_semantics = backend_output.get(
                        "timing_semantics"
                    )

                post_started = time.perf_counter()
                probability = backend_output["probability"]
                binary_mask = backend_output["binary_mask"]

                thresholded, blobs = _probability_to_blobs(
                    prob=probability,
                    binary_mask=binary_mask,
                    tensor=tensor,
                    threshold=unet_threshold,
                    area_min=geometry["area_min"],
                    area_max=geometry["area_max"],
                    bbox_padding=geometry["bbox_padding"],
                    apply_validity_mask=apply_validity_mask,
                    platform_suppression_zone=active_platform_zone,
                )

                if diagnostics_callback is not None:
                    diagnostics_callback({
                        "frame_index": packet["frame_index"],
                        "prob": probability,
                        "binary_mask": thresholded,
                        "blobs": blobs,
                        "tensor": tensor,
                        "inlier_ratio": packet["inlier_ratio"],
                        "processing_shape": (height, width),
                    })

                tracks = tracker.update([
                    blob["bbox"] for blob in blobs
                ])
                frame_predictions = []
                for track in tracks:
                    if (
                        track.hits < params["min_hits"]
                        or track.time_since_update != 0
                    ):
                        continue

                    x1, y1, x2, y2 = track.bbox
                    frame_predictions.append([
                        int(round(
                            np.clip(x1, 0, width)
                            * geometry["output_scale_x"]
                        )),
                        int(round(
                            np.clip(y1, 0, height)
                            * geometry["output_scale_y"]
                        )),
                        int(round(
                            np.clip(x2, 0, width)
                            * geometry["output_scale_x"]
                        )),
                        int(round(
                            np.clip(y2, 0, height)
                            * geometry["output_scale_y"]
                        )),
                    ])

                if frame_predictions:
                    predictions[packet["frame_index"]] = frame_predictions

                postprocess_ms.append(
                    (time.perf_counter() - post_started) * 1000.0
                )
                end_to_end_latency_ms.append(
                    (time.perf_counter() - packet["ready_at"])
                    * 1000.0
                )
            finally:
                free_slots.put(tensor)

    except BaseException as error:
        consumer_error = error
        stop_event.set()
    finally:
        if consumer_error is not None:
            while producer_thread.is_alive():
                try:
                    item = ready_queue.get(timeout=0.05)
                    if (
                        item is not _PIPELINE_SENTINEL
                        and isinstance(item, dict)
                        and item.get("slot") is not None
                    ):
                        free_slots.put(item["slot"])
                except queue.Empty:
                    pass
        producer_thread.join(timeout=5.0)
        stop_event.set()
        if requested_opencv_threads is not None:
            cv2.setNumThreads(previous_opencv_threads)

    if consumer_error is not None:
        raise consumer_error
    if producer_thread.is_alive():
        raise RuntimeError("Producer thread did not terminate")
    if producer_errors:
        raise RuntimeError(
            "Classical producer failed"
        ) from producer_errors[0]

    total_seconds = time.perf_counter() - total_started
    stage_summary = {
        key: _summarize_ms(values)
        for key, values in backend_stage_ms.items()
    }

    report = {
        "backend": getattr(backend, "name", type(backend).__name__),
        "backend_warmup": warmup_report,
        "warmup_included_in_measurements": False,
        "profile_backend_stages": bool(profile_backend_stages),
        "profiling_note": (
            "PyTorch stage profiling synchronizes CUDA/MPS around stages 7-10 "
            "and is diagnostic. Use profile_backend_stages=False for pure "
            "deployment throughput."
        ),
        "backend_timing_semantics": timing_semantics,
        "opencv_threads_requested": requested_opencv_threads,
        "opencv_threads_actual": actual_opencv_threads,
        "opencv_threads_restored_to": (
            previous_opencv_threads
            if requested_opencv_threads is not None
            else actual_opencv_threads
        ),
        "queue_size": queue_size,
        "platform_suppression_enabled": bool(
            apply_platform_suppression
        ),
        "platform_suppression_zone_1280": (
            list(platform_suppression_zone)
            if apply_platform_suppression
            else None
        ),
        "frames_consumed": consumed_frames,
        "valid_frames_produced": produced_valid,
        "total_seconds": float(total_seconds),
        "throughput_fps": float(
            consumed_frames / max(total_seconds, 1e-9)
        ),
        "producer_blocked_ms_total": float(
            producer_blocked_seconds * 1000.0
        ),
        "queue_wait_ms": _summarize_ms(queue_wait_ms),
        "backend_stages_ms": stage_summary,
        "inference_ms": stage_summary["inference_total_ms"],
        "postprocess_ms": _summarize_ms(postprocess_ms),
        "ready_to_result_latency_ms": _summarize_ms(
            end_to_end_latency_ms
        ),
        "invalid_samples": invalid_samples,
        "skipped_low_confidence": skipped_low_confidence,
        "classical_stats": dict(pipeline.stats),
        "motion_estimation_calls": int(
            pipeline.motion_estimation_calls
        ),
        "point_detection_calls": int(
            pipeline.point_detection_calls
        ),
    }

    logger.info(json.dumps(report, indent=2, ensure_ascii=False))
    return predictions, report

def _synchronize_torch_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def _empty_backend_timings():
    return {
        "7_h2d_or_input_prepare_ms": 0.0,
        "8_forward_or_coreml_predict_ms": 0.0,
        "9_sigmoid_threshold_ms": 0.0,
        "10_d2h_or_output_materialize_ms": 0.0,
        "inference_total_ms": 0.0,
    }

class TorchProbabilityBackend:
    """Single-consumer PyTorch backend for CUDA, MPS, or CPU."""
    name = "pytorch"
    timing_semantics = {
        "7_h2d_or_input_prepare_ms": (
            "NumPy contiguous/input wrapping plus CPU->device transfer"
        ),
        "8_forward_or_coreml_predict_ms": (
            "TinyUNet forward only; synchronized in profiling mode"
        ),
        "9_sigmoid_threshold_ms": (
            "sigmoid plus threshold and uint8 conversion on device"
        ),
        "10_d2h_or_output_materialize_ms": (
            "uint8 mask device->CPU; float probability too when requested"
        ),
    }

    def __init__(self, model, device: Optional[str] = None):
        if device is None:
            device = default_device()
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def warmup(self, input_chw, iterations=30, threshold=0.5):
        iterations = max(0, int(iterations))
        for _ in range(iterations):
            self.infer(
                input_chw,
                threshold=threshold,
                return_probability=False,
                profile_stages=False,
            )
        _synchronize_torch_device(self.device)
        return {
            "iterations": iterations,
            "backend": self.name,
            "device": str(self.device),
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

        if profile_stages:
            _synchronize_torch_device(self.device)
        stage_started = time.perf_counter()
        source = np.ascontiguousarray(input_chw, dtype=np.float32)
        tensor_cpu = torch.from_numpy(source).unsqueeze(0)
        tensor = tensor_cpu.to(
            self.device,
            non_blocking=(self.device.type == "cuda"),
        )
        if profile_stages:
            _synchronize_torch_device(self.device)
        timings["7_h2d_or_input_prepare_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        if profile_stages:
            _synchronize_torch_device(self.device)
        stage_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
        if profile_stages:
            _synchronize_torch_device(self.device)
        timings["8_forward_or_coreml_predict_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        if profile_stages:
            _synchronize_torch_device(self.device)
        stage_started = time.perf_counter()
        with torch.inference_mode():
            probability_t = torch.sigmoid(logits)[0, 0]
            binary_t = (
                probability_t > float(threshold)
            ).to(torch.uint8).mul_(255)
        if profile_stages:
            _synchronize_torch_device(self.device)
        timings["9_sigmoid_threshold_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        binary = binary_t.cpu().numpy()
        probability = None
        if return_probability:
            probability = (
                probability_t.detach().float().cpu().numpy()
            )
        # .cpu() is a synchronization boundary on CUDA/MPS.
        timings["10_d2h_or_output_materialize_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0
        timings["inference_total_ms"] = (
            time.perf_counter() - total_started
        ) * 1000.0

        return {
            "binary_mask": binary,
            "probability": probability,
            "timings_ms": timings,
            "timing_semantics": dict(self.timing_semantics),
            "profile_stages": bool(profile_stages),
        }

def load_frozen_pytorch_backend(checkpoint_path, device: Optional[str] = None):
    if device is None:
        device = default_device()
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
    )
    state_dict = checkpoint.get("model", checkpoint)
    model = build_inference_model(
        state_dict,
        device=device,
        fuse_batchnorm=True,
    )
    return TorchProbabilityBackend(model, device=device)