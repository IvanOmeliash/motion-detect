"""Post-processing: probability map -> blobs -> bboxes, plus video rendering."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def _prepare_unet_input(tensor, checkpoint_blur=False, out=None):
    """
    Copy the first three CompensationPipeline channels into a reusable
    contiguous float32 CHW buffer. This avoids np.stack/ascontiguousarray and
    a new large allocation on every frame.
    """
    source = np.asarray(tensor)[:3]
    expected_shape = tuple(source.shape)

    # Fastest case: CompensationPipeline already produced contiguous float32.
    # Return a zero-copy view instead of forcing an unnecessary memcpy.
    if (
        out is None
        and not checkpoint_blur
        and source.dtype == np.float32
        and source.flags.c_contiguous
    ):
        return source

    if out is None:
        out = np.empty(expected_shape, dtype=np.float32)
    else:
        if out.shape != expected_shape:
            raise ValueError(
                f"Input buffer shape {out.shape} != expected {expected_shape}"
            )
        if out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError(
                "Input buffer must be C-contiguous float32"
            )

    if checkpoint_blur:
        for channel_index in range(3):
            blurred = cv2.GaussianBlur(
                source[channel_index],
                (3, 3),
                0,
            )
            np.copyto(
                out[channel_index],
                blurred,
                casting="unsafe",
            )
    else:
        np.copyto(out, source, casting="unsafe")

    return out

def _scaled_detection_geometry(height, width, params, return_size=(1280, 720)):
    """Scale 1280x720 contour/tracker parameters to the processing frame."""
    reference_w, reference_h = 1280.0, 720.0
    scale_x = float(width) / reference_w
    scale_y = float(height) / reference_h
    area_scale = scale_x * scale_y

    area_min = max(1.0, float(params["area_min"]) * area_scale)
    area_max = max(area_min + 1.0, float(params["area_max"]) * area_scale)
    bbox_padding = max(
        0,
        int(round(float(params.get("bbox_padding", 4)) * scale_x)),
    )

    return_w, return_h = map(int, return_size)
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "area_min": area_min,
        "area_max": area_max,
        "bbox_padding": bbox_padding,
        "output_scale_x": float(return_w) / float(width),
        "output_scale_y": float(return_h) / float(height),
    }

LEGACY_PLATFORM_SUPPRESSION_ZONE_1280 = (0, 400, 175, 490)
LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE = (1280, 720)

def _scale_reference_zone_to_mask(
    zone_xyxy,
    mask_shape,
    reference_size=LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE,
):
    """Scale an xyxy zone from reference coordinates to an HxW mask."""
    height, width = map(int, mask_shape)
    reference_width, reference_height = map(int, reference_size)
    x1, y1, x2, y2 = map(float, zone_xyxy)

    scaled = [
        int(np.floor(x1 * width / max(1, reference_width))),
        int(np.floor(y1 * height / max(1, reference_height))),
        int(np.ceil(x2 * width / max(1, reference_width))),
        int(np.ceil(y2 * height / max(1, reference_height))),
    ]
    scaled[0] = int(np.clip(scaled[0], 0, width))
    scaled[2] = int(np.clip(scaled[2], 0, width))
    scaled[1] = int(np.clip(scaled[1], 0, height))
    scaled[3] = int(np.clip(scaled[3], 0, height))
    return tuple(scaled)

def _apply_platform_suppression_zone(
    binary_mask,
    zone_xyxy=LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
    reference_size=LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE,
):
    """
    Temporary safety net until hard negatives replace the old static mask.

    The zone is disabled explicitly only after:
      1) rotor/wing candidates are present in the first mining review;
      2) the model is retrained on approved hard negatives;
      3) precision improves and far-recall passes the acceptance gate.
    """
    mask = np.asarray(binary_mask, dtype=np.uint8)
    x1, y1, x2, y2 = _scale_reference_zone_to_mask(
        zone_xyxy,
        mask.shape,
        reference_size=reference_size,
    )
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 0
    return mask

def _probability_to_blobs(
    prob,
    tensor,
    threshold,
    area_min,
    area_max,
    bbox_padding,
    apply_validity_mask=True,
    binary_mask=None,
    platform_suppression_zone=None,
    platform_reference_size=LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE,
):
    """
    Shared neural post-processing used by production inference and mining.

    Fast production path passes binary_mask already thresholded on CUDA/MPS,
    so only a uint8 HxW map is copied to CPU. Hard-negative mining additionally
    passes the float probability map because it needs prob_max for ranking.
    """
    prob_array = None
    if prob is not None:
        prob_array = np.asarray(prob, dtype=np.float32)

    if binary_mask is None:
        if prob_array is None:
            raise ValueError("Either prob or binary_mask must be provided")
        thresh = (prob_array > float(threshold)).astype(np.uint8) * 255
    else:
        thresh = np.asarray(binary_mask, dtype=np.uint8)
        if thresh.ndim != 2:
            raise ValueError(
                f"binary_mask must be 2-D, got {thresh.shape}"
            )
        # The CPU tensor is no longer needed after this point, so its writable
        # NumPy view can be reused directly for validity masking/morphology.
        thresh = np.ascontiguousarray(thresh)
        if thresh.size and int(thresh.max()) <= 1:
            thresh *= np.uint8(255)

    if apply_validity_mask:
        validity = np.asarray(tensor[3]) > 0.5
        if validity.shape != thresh.shape:
            raise ValueError(
                f"Validity shape {validity.shape} != mask shape {thresh.shape}"
            )
        thresh[~validity] = 0

    if platform_suppression_zone is not None:
        _apply_platform_suppression_zone(
            thresh,
            zone_xyxy=platform_suppression_zone,
            reference_size=platform_reference_size,
        )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = thresh.shape
    blobs = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (float(area_min) < area < float(area_max)):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or (w / h) > 4.0:
            continue

        raw_bbox = [
            int(x),
            int(y),
            int(x + w),
            int(y + h),
        ]
        padded_bbox = [
            max(0, int(x - bbox_padding)),
            max(0, int(y - bbox_padding)),
            min(width, int(x + w + bbox_padding)),
            min(height, int(y + h + bbox_padding)),
        ]

        if prob_array is None:
            prob_max = None
        else:
            region = prob_array[y:y + h, x:x + w]
            prob_max = float(region.max()) if region.size else 0.0

        blobs.append({
            "raw_bbox": raw_bbox,
            "bbox": padded_bbox,
            "area": area,
            "prob_max": prob_max,
        })

    return thresh, blobs

def render_predictions_video(
    input_path,
    output_path,
    predictions,
    output_size=(1280, 720),
    box_color=(0, 255, 0),
    box_thickness=2,
):
    """Write an MP4 with the bboxes returned by predict_video_bboxes()."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {input_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = map(int, output_size)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not create video: {output_path}')

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            boxes = predictions.get(frame_index, [])
            for x1, y1, x2, y2 in boxes:
                x1 = int(np.clip(x1, 0, out_w - 1))
                y1 = int(np.clip(y1, 0, out_h - 1))
                x2 = int(np.clip(x2, 0, out_w - 1))
                y2 = int(np.clip(y2, 0, out_h - 1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            cv2.putText(
                frame,
                f'frame={frame_index} detections={len(boxes)}',
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    logger.info(f'Video saved: {output_path}')
    return str(output_path)

# ================================================================

# ================================================================
