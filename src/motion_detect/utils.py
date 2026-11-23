"""Video/annotation indexing, clip-id parsing, GT file parsing, archive extraction."""
from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {
    ".mov",
    ".mp4",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm",
}

# ---------------------------------------------------------------
# CLIP ID PARSING
# ---------------------------------------------------------------

def extract_clip_id(path, annotation=False):
    path = Path(path)
    stem = path.stem

    for pattern in (
        r"(?i)clip[ _-]*0*(\d+)",
        r"(?i)(?:video|vid)[ _-]*0*(\d+)",
    ):
        match = re.search(pattern, stem)

        if match:
            return int(match.group(1))

    if annotation:
        match = re.search(
            r"(?i)0*(\d+).*gt",
            stem,
        )

        if match:
            return int(match.group(1))

    return None

# ---------------------------------------------------------------
# RECURSIVE VIDEO INDEX
# ---------------------------------------------------------------

def index_video_files(root):
    root = Path(root)

    if not root.is_dir():
        raise NotADirectoryError(
            f"Video folder not found:\n{root}"
        )

    result = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.casefold() not in VIDEO_SUFFIXES:
            continue

        clip_id = extract_clip_id(path)

        if clip_id is not None and clip_id not in result:
            result[clip_id] = path

    return result

# ---------------------------------------------------------------
# RECURSIVE ANNOTATION INDEX
# ---------------------------------------------------------------

def index_annotation_files(root):
    root = Path(root)

    if not root.is_dir():
        raise NotADirectoryError(
            f"Annotation folder not found:\n{root}"
        )

    result = {}

    for path in root.rglob("*.txt"):
        clip_id = extract_clip_id(
            path,
            annotation=True,
        )

        if clip_id is not None and clip_id not in result:
            result[clip_id] = path

    return result

# ---------------------------------------------------------------
# VERIFY TEST FILES

def extract_archives(videos_source, annotations_source, extract_dir="./data"):
    """Return data folders. Sources may be folders or ZIP archives."""
    videos_source = Path(videos_source)
    annotations_source = Path(annotations_source)

    # Typical case: plain unpacked folders (no archive).
    if videos_source.is_dir() and annotations_source.is_dir():
        logger.info(f"Using video folder:    {videos_source}")
        logger.info(f"Using annotation folder: {annotations_source}")
        return str(videos_source), str(annotations_source)

    # Do not allow one folder and one ZIP at the same time: almost always a path mistake.
    if videos_source.is_dir() != annotations_source.is_dir():
        raise ValueError(
            "Videos and annotations must be either both folders or both ZIP files.\n"
            f"videos_source={videos_source}\n"
            f"annotations_source={annotations_source}"
        )

    if not videos_source.is_file():
        raise FileNotFoundError(f"Video source not found: {videos_source}")
    if not annotations_source.is_file():
        raise FileNotFoundError(f"Annotation source not found: {annotations_source}")
    if not zipfile.is_zipfile(videos_source):
        raise ValueError(f"Video file is not a ZIP archive: {videos_source}")
    if not zipfile.is_zipfile(annotations_source):
        raise ValueError(f"Annotation file is not a ZIP archive: {annotations_source}")

    os.makedirs(extract_dir, exist_ok=True)
    videos_dir = Path(extract_dir) / "Videos"
    annotations_dir = Path(extract_dir) / "Video_Annotation"

    if not videos_dir.exists():
        logger.info(f"Extracting {videos_source}...")
        with zipfile.ZipFile(videos_source, "r") as archive:
            archive.extractall(extract_dir)
    if not annotations_dir.exists():
        logger.info(f"Extracting {annotations_source}...")
        with zipfile.ZipFile(annotations_source, "r") as archive:
            archive.extractall(extract_dir)

    return str(videos_dir), str(annotations_dir)

def parse_gt_file(
    txt_path,
    video_path=None,
    target_w=None,
    target_h=None,
):
    """
    Parse the raw annotation format (y1,x1,y2,x2) and return
    bbox in the internal format (x1,y1,x2,y2).

    By default coordinates stay in the original resolution.
    If target_w and target_h are given, bbox are scaled to the target.
    """
    orig_w, orig_h = 1920, 1080

    if video_path and os.path.exists(video_path):
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if width > 0 and height > 0:
                orig_w, orig_h = width, height
        capture.release()

    if (target_w is None) != (target_h is None):
        raise ValueError("target_w and target_h must be given together")

    if target_w is None:
        scale_x = scale_y = 1.0
    else:
        scale_x = float(target_w) / float(orig_w)
        scale_y = float(target_h) / float(orig_h)

    gt_data = {}
    with open(txt_path, "r", encoding="utf-8") as file:
        for line in file:
            frame_match = re.search(r"time_layer:\s*(\d+)", line)
            if not frame_match:
                continue

            frame_id = int(frame_match.group(1))
            boxes = []

            for coordinate_string in re.findall(r"\(([^)]+)\)", line):
                coordinates = [
                    float(value.strip())
                    for value in coordinate_string.split(",")
                ]
                if len(coordinates) != 4:
                    continue

                # In the file: (y1, x1, y2, x2).
                y1, x1, y2, x2 = coordinates
                boxes.append([
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                ])

            # An empty list is also stored: it is a valid negative frame.
            gt_data[frame_id] = boxes

    return gt_data

# ================================================================

# ================================================================
# ================================================================

SIZE_BUCKETS_1280X720 = {
    "far": (0.0, 16.0),
    "mid": (16.0, 64.0),
    "near": (64.0, float("inf")),
}

def validate_gt_coordinate_ranges(gt_dict, source_shape):
    """Check that bbox (x1,y1,x2,y2) lie within the source frame."""
    source_h, source_w = source_shape
    arrays = [
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for boxes in gt_dict.values()
        if len(boxes) > 0
    ]
    if not arrays:
        return

    boxes = np.concatenate(arrays, axis=0)
    if boxes[:, [0, 2]].min() < 0 or boxes[:, [0, 2]].max() > source_w:
        raise AssertionError(
            f"x outside frame: min={boxes[:, [0, 2]].min()}, "
            f"max={boxes[:, [0, 2]].max()}, width={source_w}"
        )
    if boxes[:, [1, 3]].min() < 0 or boxes[:, [1, 3]].max() > source_h:
        raise AssertionError(
            f"y outside frame: min={boxes[:, [1, 3]].min()}, "
            f"max={boxes[:, [1, 3]].max()}, height={source_h}"
        )

