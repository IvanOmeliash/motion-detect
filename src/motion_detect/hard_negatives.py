"""Hard-negative mining, review-grid generation, and approval from review.csv.

Mined candidates are blobs that passed the production filters but have IoU == 0
with every GT box on their frame. Approval is data-driven: a review.csv with
decision=approve/reject per rank (schema written by _save_hard_negative_staging).
"""

from __future__ import annotations

import csv
from collections import Counter
import heapq
import json
import logging
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from .dataset import TRAIN_CROP_SIZE, _video_source_shape, scale_gt_boxes
from .inference import predict_video_bboxes
from .postprocess import LEGACY_PLATFORM_SUPPRESSION_ZONE_1280
from .tracker import iou
from .utils import (
    extract_archives,
    index_annotation_files,
    index_video_files,
    parse_gt_file,
)

logger = logging.getLogger(__name__)

HARD_NEGATIVE_FRACTION = 0.20
HARD_NEGATIVE_REVIEW_COUNT = 200
HARD_NEGATIVE_REMINE_EVERY = 3
HARD_NEGATIVE_INLIER_PRIORITY_BOOST = 1.0
HARD_NEGATIVE_INLIER_MARGIN_SCALE = 0.05
HARD_NEGATIVE_ROTOR_ZONE_1280 = LEGACY_PLATFORM_SUPPRESSION_ZONE_1280

def _bbox_overlap_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return float(width * height)

def _scale_bbox_to_reference(
    bbox,
    processing_shape,
    reference_size=(1280, 720),
):
    height, width = map(int, processing_shape)
    reference_width, reference_height = map(int, reference_size)
    x1, y1, x2, y2 = map(float, bbox)
    return [
        float(x1 * reference_width / max(1, width)),
        float(y1 * reference_height / max(1, height)),
        float(x2 * reference_width / max(1, width)),
        float(y2 * reference_height / max(1, height)),
    ]

def _max_iou_with_boxes(box, boxes):
    if len(boxes) == 0:
        return 0.0
    return max(float(iou(box, gt_box)) for gt_box in boxes)

def _center_crop_2d(array, center_x, center_y, crop_size=TRAIN_CROP_SIZE):
    """
    Fixed-size crop with zero padding. Returns (crop, x0, y0), where x0/y0
    are coordinates of the crop's upper-left corner in the processing frame.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {array.shape}")

    crop_size = int(crop_size)
    height, width = array.shape
    x0 = int(round(float(center_x) - crop_size / 2))
    y0 = int(round(float(center_y) - crop_size / 2))
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(width, x1)
    source_y1 = min(height, y1)

    crop = np.zeros((crop_size, crop_size), dtype=array.dtype)
    destination_x0 = source_x0 - x0
    destination_y0 = source_y0 - y0
    destination_x1 = destination_x0 + max(0, source_x1 - source_x0)
    destination_y1 = destination_y0 + max(0, source_y1 - source_y0)

    if source_x1 > source_x0 and source_y1 > source_y0:
        crop[
            destination_y0:destination_y1,
            destination_x0:destination_x1,
        ] = array[source_y0:source_y1, source_x0:source_x1]

    return crop, x0, y0

def _bbox_to_crop_coordinates(box, crop_x0, crop_y0, crop_size):
    x1, y1, x2, y2 = map(float, box)
    return [
        int(np.clip(round(x1 - crop_x0), 0, crop_size - 1)),
        int(np.clip(round(y1 - crop_y0), 0, crop_size - 1)),
        int(np.clip(round(x2 - crop_x0), 0, crop_size - 1)),
        int(np.clip(round(y2 - crop_y0), 0, crop_size - 1)),
    ]

def recommended_hard_negative_count(
    train_dir,
    fraction=HARD_NEGATIVE_FRACTION,
):
    """
    N ≈ 15–20% of the original train set, excluding already approved
    hard-negative samples.
    """
    train_dir = Path(train_dir)
    groups_dir = train_dir / "groups"
    base_count = 0
    for mask_path in (train_dir / "masks").glob("*_mask.png"):
        sid = mask_path.stem.replace("_mask", "")
        group_path = groups_dir / f"{sid}.txt"
        group = (
            group_path.read_text(encoding="utf-8").strip()
            if group_path.exists() else ""
        )
        if group != "hard_negative":
            base_count += 1

    return max(1, int(round(base_count * float(fraction))))

def _push_top_candidate(heap, candidate, top_n, serial):
    entry = (
        float(candidate["rank_score"]),
        int(serial),
        candidate,
    )
    if len(heap) < int(top_n):
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)

def _save_hard_negative_staging(candidates, output_dir, round_id):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for subdir in (
        "frames",
        "masks",
        "weights",
        "bboxes",
        "bboxes_src",
        "groups",
        "previews",
    ):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    metadata = []
    for rank, candidate in enumerate(candidates, start=1):
        sid = (
            f"hn_r{int(round_id):02d}_"
            f"clip{int(candidate['clip_id']):02d}_"
            f"f{int(candidate['frame_index']):06d}_"
            f"b{int(candidate['blob_index']):02d}"
        )

        t2, t1, t0 = candidate.pop("_crops")
        writes_ok = all([
            cv2.imwrite(
                str(output_dir / "frames" / f"{sid}_t2.png"),
                t2,
            ),
            cv2.imwrite(
                str(output_dir / "frames" / f"{sid}_t1.png"),
                t1,
            ),
            cv2.imwrite(
                str(output_dir / "frames" / f"{sid}_t0.png"),
                t0,
            ),
            cv2.imwrite(
                str(output_dir / "masks" / f"{sid}_mask.png"),
                np.zeros_like(t0, dtype=np.uint8),
            ),
            cv2.imwrite(
                str(output_dir / "weights" / f"{sid}_weight.png"),
                np.full_like(t0, 255, dtype=np.uint8),
            ),
        ])
        if not writes_ok:
            raise IOError(f"Failed to save hard-negative crop {sid}")

        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        np.save(
            str(output_dir / "bboxes" / f"{sid}.npy"),
            empty_boxes,
            allow_pickle=False,
        )
        np.save(
            str(output_dir / "bboxes_src" / f"{sid}.npy"),
            empty_boxes,
            allow_pickle=False,
        )
        (output_dir / "groups" / f"{sid}.txt").write_text(
            "hard_negative",
            encoding="utf-8",
        )

        preview = cv2.cvtColor(t0, cv2.COLOR_GRAY2BGR)
        px1, py1, px2, py2 = candidate["blob_bbox_crop"]
        cv2.rectangle(
            preview,
            (px1, py1),
            (px2, py2),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            preview,
            f"#{rank} {sid}",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            (
                f"p={candidate['prob_max']:.3f} "
                f"A={candidate['area']:.1f} "
                f"inl={candidate['inlier_ratio']:.3f}"
            ),
            (5, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(
            str(output_dir / "previews" / f"{sid}.jpg"),
            preview,
        )

        public_candidate = {
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        }
        public_candidate.update({
            "rank": rank,
            "sample_id": sid,
            "review_status": "pending",
        })
        metadata.append(public_candidate)

    metadata_path = output_dir / "candidates.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    review_csv = output_dir / "review.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "sample_id",
                "decision",
                "notes",
                "score",
                "rank_score",
                "prob_max",
                "area",
                "inlier_ratio",
                "clip_id",
                "frame_index",
                "rotor_zone_hit",
                "rotor_zone_overlap_area_1280",
            ],
        )
        writer.writeheader()
        for item in metadata:
            writer.writerow({
                "rank": item["rank"],
                "sample_id": item["sample_id"],
                "decision": "",
                "notes": "",
                "score": item["score"],
                "rank_score": item["rank_score"],
                "prob_max": item["prob_max"],
                "area": item["area"],
                "inlier_ratio": item["inlier_ratio"],
                "clip_id": item["clip_id"],
                "frame_index": item["frame_index"],
                "rotor_zone_hit": item.get(
                    "rotor_zone_hit",
                    False,
                ),
                "rotor_zone_overlap_area_1280": item.get(
                    "rotor_zone_overlap_area_1280",
                    0.0,
                ),
            })

    logger.info(f"Saved {len(metadata)} pending hard negatives to: {output_dir}")
    logger.info(f"Manual review table: {review_csv}")
    return metadata

def mine_hard_negatives(
    videos_source,
    annotations_source,
    train_dir,
    output_dir,
    params,
    unet_model,
    clip_indices,
    unet_threshold=0.95,
    top_fraction=HARD_NEGATIVE_FRACTION,
    top_n=None,
    round_id=1,
    apply_validity_mask=True,
):
    """
    Mine blobs that passed the exact production area/aspect filters but have
    IoU == 0 with every GT box on that frame.

    Raw candidate score:
        score = max(probability inside blob) * contour_area

    Ranking additionally boosts frames whose inlier_ratio is only slightly
    above inlier_thresh, because they tend to contain the hardest warps.
    """
    videos_dir, annotations_dir = extract_archives(
        videos_source,
        annotations_source,
    )
    video_index = index_video_files(videos_dir)
    annotation_index = index_annotation_files(annotations_dir)

    if top_n is None:
        top_n = recommended_hard_negative_count(
            train_dir,
            fraction=top_fraction,
        )
    top_n = max(1, int(top_n))

    logger.info(
        f"Mining top {top_n} hard negatives "
        f"({float(top_fraction):.0%} of base train when auto-sized)"
    )

    candidate_heap = []
    serial = 0
    total_zero_iou_blobs = 0

    for clip_id in tqdm(sorted(clip_indices), desc="Hard-negative clips"):
        video_path = video_index.get(int(clip_id))
        annotation_path = annotation_index.get(int(clip_id))
        if video_path is None or annotation_path is None:
            logger.info(f"Skip Clip_{clip_id}: missing video or annotation")
            continue

        source_shape = _video_source_shape(str(video_path))
        gt_source = parse_gt_file(
            str(annotation_path),
            video_path=str(video_path),
        )

        def collect_frame(frame_data, current_clip_id=int(clip_id)):
            nonlocal serial, total_zero_iou_blobs

            tensor = np.asarray(frame_data["tensor"])
            height, width = frame_data["processing_shape"]
            frame_id = int(frame_data["frame_index"])
            gt_processing = scale_gt_boxes(
                gt_source.get(frame_id, []),
                src_shape=source_shape,
                dst_shape=(height, width),
            )

            inlier_ratio = float(frame_data["inlier_ratio"])
            inlier_threshold = params.get("inlier_thresh")
            if inlier_threshold is None:
                inlier_margin = float("inf")
                inlier_factor = 1.0
            else:
                inlier_margin = max(
                    0.0,
                    inlier_ratio - float(inlier_threshold),
                )
                inlier_factor = (
                    1.0
                    + float(HARD_NEGATIVE_INLIER_PRIORITY_BOOST)
                    * np.exp(
                        -inlier_margin
                        / max(
                            1e-6,
                            float(HARD_NEGATIVE_INLIER_MARGIN_SCALE),
                        )
                    )
                )

            for blob_index, blob in enumerate(frame_data["blobs"]):
                raw_bbox = blob["raw_bbox"]
                max_iou = _max_iou_with_boxes(
                    raw_bbox,
                    gt_processing,
                )
                if max_iou > 0.0:
                    continue

                total_zero_iou_blobs += 1
                x1, y1, x2, y2 = raw_bbox
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                crops = []
                crop_x0 = crop_y0 = None
                for channel in tensor[:3]:
                    channel_u8 = np.clip(
                        channel * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                    crop, current_x0, current_y0 = _center_crop_2d(
                        channel_u8,
                        center_x,
                        center_y,
                        crop_size=TRAIN_CROP_SIZE,
                    )
                    crops.append(crop)
                    if crop_x0 is None:
                        crop_x0, crop_y0 = current_x0, current_y0

                score = float(blob["prob_max"]) * float(blob["area"])
                rank_score = score * float(inlier_factor)
                bbox_1280 = _scale_bbox_to_reference(
                    raw_bbox,
                    processing_shape=(height, width),
                    reference_size=(1280, 720),
                )
                rotor_overlap_area = _bbox_overlap_area(
                    bbox_1280,
                    HARD_NEGATIVE_ROTOR_ZONE_1280,
                )

                candidate = {
                    "clip_id": int(current_clip_id),
                    "frame_index": frame_id,
                    "blob_index": int(blob_index),
                    "blob_bbox_processing": [
                        int(value) for value in raw_bbox
                    ],
                    "blob_bbox_1280x720": [
                        float(value) for value in bbox_1280
                    ],
                    "rotor_zone_overlap_area_1280": float(
                        rotor_overlap_area
                    ),
                    "rotor_zone_hit": bool(rotor_overlap_area > 0.0),
                    "blob_bbox_crop": _bbox_to_crop_coordinates(
                        raw_bbox,
                        crop_x0,
                        crop_y0,
                        TRAIN_CROP_SIZE,
                    ),
                    "score": float(score),
                    "rank_score": float(rank_score),
                    "prob_max": float(blob["prob_max"]),
                    "area": float(blob["area"]),
                    "max_gt_iou": float(max_iou),
                    "inlier_ratio": float(inlier_ratio),
                    "inlier_margin": (
                        None
                        if not np.isfinite(inlier_margin)
                        else float(inlier_margin)
                    ),
                    "_crops": tuple(crops),
                }
                serial += 1
                _push_top_candidate(
                    candidate_heap,
                    candidate,
                    top_n=top_n,
                    serial=serial,
                )

        # diagnostics_callback receives probability + production-filtered blobs.
        predict_video_bboxes(
            input_path=str(video_path),
            params=params,
            unet_model=unet_model,
            unet_threshold=unet_threshold,
            compensation_step=1,
            return_size=(1280, 720),
            apply_validity_mask=apply_validity_mask,
            checkpoint_blur=False,
            apply_platform_suppression=False,
            diagnostics_callback=collect_frame,
        )

    selected = [
        entry[2]
        for entry in sorted(
            candidate_heap,
            key=lambda entry: (entry[0], entry[1]),
            reverse=True,
        )
    ]

    logger.info("Zero-IoU blobs before top-N ranking:", total_zero_iou_blobs)
    return _save_hard_negative_staging(
        selected,
        output_dir=output_dir,
        round_id=round_id,
    )

def create_hard_negative_review_grids(
    staging_dir,
    count=HARD_NEGATIVE_REVIEW_COUNT,
    per_page=20,
    columns=5,
    display_pages=False,
):
    """
    Build overlay contact sheets for manual review. The red rectangle is the
    mined blob; reject birds, people, vehicles or any other genuine motion.
    """
    staging_dir = Path(staging_dir)
    metadata = json.loads(
        (staging_dir / "candidates.json").read_text(encoding="utf-8")
    )
    metadata = metadata[: min(int(count), len(metadata))]

    output_dir = staging_dir / "review_grids"
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = max(1, int(columns))
    per_page = max(columns, int(per_page))
    rows = int(np.ceil(per_page / columns))
    tile_size = TRAIN_CROP_SIZE
    page_paths = []

    for page_index, start in enumerate(
        range(0, len(metadata), per_page),
        start=1,
    ):
        page_items = metadata[start:start + per_page]
        canvas = np.zeros(
            (rows * tile_size, columns * tile_size, 3),
            dtype=np.uint8,
        )

        for local_index, item in enumerate(page_items):
            preview_path = (
                staging_dir / "previews" / f"{item['sample_id']}.jpg"
            )
            preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
            if preview is None:
                raise FileNotFoundError(preview_path)
            preview = cv2.resize(
                preview,
                (tile_size, tile_size),
                interpolation=cv2.INTER_AREA,
            )

            row = local_index // columns
            column = local_index % columns
            y0 = row * tile_size
            x0 = column * tile_size
            canvas[y0:y0 + tile_size, x0:x0 + tile_size] = preview

        page_path = output_dir / f"review_page_{page_index:02d}.jpg"
        cv2.imwrite(str(page_path), canvas)
        page_paths.append(page_path)

    logger.info(
        f"Created {len(page_paths)} review pages for "
        f"{len(metadata)} candidates: {output_dir}"
    )
    if display_pages:
        logger.warning(
            "display_pages=True is a notebook-only option; pages are on disk at %s",
            output_dir,
        )

    return page_paths

def _hard_negative_sample_ids(train_dir):
    train_dir = Path(train_dir)
    result = []
    for group_path in (train_dir / "groups").glob("*.txt"):
        if group_path.read_text(encoding="utf-8").strip() == "hard_negative":
            result.append(group_path.stem)
    return sorted(result)

def remove_approved_hard_negatives(train_dir):
    """Remove the previous approved hard-negative group before replacement."""
    train_dir = Path(train_dir)
    sample_ids = _hard_negative_sample_ids(train_dir)
    for sid in sample_ids:
        paths = [
            train_dir / "frames" / f"{sid}_t2.png",
            train_dir / "frames" / f"{sid}_t1.png",
            train_dir / "frames" / f"{sid}_t0.png",
            train_dir / "masks" / f"{sid}_mask.png",
            train_dir / "weights" / f"{sid}_weight.png",
            train_dir / "bboxes" / f"{sid}.npy",
            train_dir / "bboxes_src" / f"{sid}.npy",
            train_dir / "groups" / f"{sid}.txt",
        ]
        for path in paths:
            path.unlink(missing_ok=True)
    logger.info(f"Removed {len(sample_ids)} previously approved hard negatives")
    return len(sample_ids)

def approve_hard_negatives(
    staging_dir,
    train_dir,
    approved_ids=None,
    rejected_ids=None,
    approve_remaining=False,
    replace_existing=False,
):
    """
    Copy reviewed candidates into train and mark them as hard_negative.

    Safe defaults:
      - explicit approved_ids are accepted;
      - or set approve_remaining=True and provide rejected_ids;
      - with neither choice, the function raises instead of silently approving.
    """
    staging_dir = Path(staging_dir)
    train_dir = Path(train_dir)
    metadata_path = staging_dir / "candidates.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    all_ids = {item["sample_id"] for item in metadata}

    approved_ids = set(approved_ids or [])
    rejected_ids = set(rejected_ids or [])

    unknown = (approved_ids | rejected_ids) - all_ids
    if unknown:
        raise KeyError(f"Unknown candidate IDs: {sorted(unknown)}")

    if approve_remaining:
        selected = all_ids - rejected_ids
    elif approved_ids:
        selected = approved_ids
    else:
        raise ValueError(
            "Provide approved_ids, or set approve_remaining=True after "
            "listing rejected_ids from manual review."
        )

    if replace_existing:
        remove_approved_hard_negatives(train_dir)

    for subdir in (
        "frames",
        "masks",
        "weights",
        "bboxes",
        "bboxes_src",
        "groups",
    ):
        (train_dir / subdir).mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in metadata:
        sid = item["sample_id"]
        if sid not in selected:
            item["review_status"] = "rejected"
            continue

        file_pairs = [
            (
                staging_dir / "frames" / f"{sid}_t2.png",
                train_dir / "frames" / f"{sid}_t2.png",
            ),
            (
                staging_dir / "frames" / f"{sid}_t1.png",
                train_dir / "frames" / f"{sid}_t1.png",
            ),
            (
                staging_dir / "frames" / f"{sid}_t0.png",
                train_dir / "frames" / f"{sid}_t0.png",
            ),
            (
                staging_dir / "masks" / f"{sid}_mask.png",
                train_dir / "masks" / f"{sid}_mask.png",
            ),
            (
                staging_dir / "weights" / f"{sid}_weight.png",
                train_dir / "weights" / f"{sid}_weight.png",
            ),
            (
                staging_dir / "bboxes" / f"{sid}.npy",
                train_dir / "bboxes" / f"{sid}.npy",
            ),
            (
                staging_dir / "bboxes_src" / f"{sid}.npy",
                train_dir / "bboxes_src" / f"{sid}.npy",
            ),
            (
                staging_dir / "groups" / f"{sid}.txt",
                train_dir / "groups" / f"{sid}.txt",
            ),
        ]
        for source, destination in file_pairs:
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination)

        item["review_status"] = "approved"
        copied += 1

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"Approved {copied} hard negatives; "
        f"rejected/pending={len(metadata) - copied}"
    )
    return copied

def make_hard_negative_refresh_callback(
    videos_source,
    annotations_source,
    train_dir,
    staging_root,
    params,
    clip_indices,
    unet_threshold=0.95,
    auto_approve=False,
):
    """
    Optional epoch callback for re-mining every 2–3 epochs.

    Keep auto_approve=False until the first round has been manually reviewed.
    Later rounds can be auto-approved only after verifying that mined blobs are
    genuinely false motion responses for this dataset.
    """
    staging_root = Path(staging_root)

    def callback(model, epoch):
        round_dir = staging_root / f"epoch_{int(epoch):03d}"
        metadata = mine_hard_negatives(
            videos_source=videos_source,
            annotations_source=annotations_source,
            train_dir=train_dir,
            output_dir=round_dir,
            params=params,
            unet_model=model,
            clip_indices=clip_indices,
            unet_threshold=unet_threshold,
            round_id=epoch,
        )

        if not auto_approve:
            create_hard_negative_review_grids(
                round_dir,
                count=HARD_NEGATIVE_REVIEW_COUNT,
                display_pages=False,
            )
            logger.info(
                "New round is pending manual review; sampler was not changed."
            )
            return False

        copied = approve_hard_negatives(
            staging_dir=round_dir,
            train_dir=train_dir,
            rejected_ids=[],
            approve_remaining=True,
            replace_existing=True,
        )
        return copied > 0

    return callback

# Columns written by _save_hard_negative_staging into review.csv.
REVIEW_CSV_FIELDS = [
    "rank",
    "sample_id",
    "decision",
    "notes",
    "score",
    "rank_score",
    "prob_max",
    "area",
    "inlier_ratio",
    "clip_id",
    "frame_index",
    "rotor_zone_hit",
    "rotor_zone_overlap_area_1280",
]


def read_review_decisions(
    review_csv_path,
    min_rank: int = 1,
    max_rank: int = HARD_NEGATIVE_REVIEW_COUNT,
) -> dict:
    """Read and strictly validate manual review decisions from review.csv.

    Every rank in [min_rank, max_rank] must appear exactly once and carry a
    decision of "approve" or "reject" (the empty pending rows outside the
    reviewed range are ignored). Returns approved/rejected sample ids and
    counts. This mirrors the notebook's strict review validation.
    """
    review_csv_path = Path(review_csv_path)
    if not review_csv_path.is_file():
        raise FileNotFoundError(review_csv_path)

    with review_csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    reviewed = []
    for row in rows:
        try:
            rank = int(float(row.get("rank", "")))
        except (TypeError, ValueError):
            continue
        if min_rank <= rank <= max_rank:
            decision = (row.get("decision") or "").strip().lower()
            row = dict(row)
            row["rank"] = rank
            row["decision"] = decision
            reviewed.append(row)

    reviewed.sort(key=lambda row: row["rank"])
    found_ranks = [row["rank"] for row in reviewed]
    expected_ranks = set(range(min_rank, max_rank + 1))
    missing_ranks = sorted(expected_ranks - set(found_ranks))
    duplicate_ranks = sorted(
        rank for rank, count in Counter(found_ranks).items() if count > 1
    )
    invalid_rows = [
        row for row in reviewed if row["decision"] not in ("approve", "reject")
    ]

    if len(reviewed) != (max_rank - min_rank + 1):
        raise RuntimeError(
            f"Expected {max_rank - min_rank + 1} reviewed rows, found {len(reviewed)}"
        )
    if missing_ranks:
        raise RuntimeError(f"Missing ranks: {missing_ranks}")
    if duplicate_ranks:
        raise RuntimeError(f"Duplicate ranks: {duplicate_ranks}")
    if invalid_rows:
        raise RuntimeError(
            "Reviewed rows must have decision approve/reject: "
            + ", ".join(str(row.get("sample_id")) for row in invalid_rows[:10])
        )

    approved_ids = [
        row["sample_id"] for row in reviewed if row["decision"] == "approve"
    ]
    rejected_ids = [
        row["sample_id"] for row in reviewed if row["decision"] == "reject"
    ]
    return {
        "reviewed": len(reviewed),
        "approved": len(approved_ids),
        "rejected": len(rejected_ids),
        "approved_ids": approved_ids,
        "rejected_ids": rejected_ids,
        "pending_outside_range": len(rows) - len(reviewed),
    }


def approve_from_review_csv(
    staging_dir,
    train_dir,
    review_csv_path=None,
    min_rank: int = 1,
    max_rank: int = HARD_NEGATIVE_REVIEW_COUNT,
    replace_existing: bool = True,
) -> dict:
    """Approve hard negatives from a reviewed review.csv and copy them into train.

    The CSV schema is data-free: rank, sample_id, decision, notes plus the
    mining metrics (see REVIEW_CSV_FIELDS). Round 01 of the research notebooks
    reviewed ranks 1-200 and approved 57 / rejected 143.
    """
    staging_dir = Path(staging_dir)
    if review_csv_path is None:
        review_csv_path = staging_dir / "review.csv"
    summary = read_review_decisions(
        review_csv_path,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    copied = approve_hard_negatives(
        staging_dir=staging_dir,
        train_dir=train_dir,
        approved_ids=summary["approved_ids"],
        rejected_ids=summary["rejected_ids"],
        replace_existing=replace_existing,
    )
    summary["copied_into_train"] = copied
    logger.info(
        "Hard-negative approval from %s: reviewed=%d approved=%d rejected=%d copied=%d",
        review_csv_path,
        summary["reviewed"],
        summary["approved"],
        summary["rejected"],
        copied,
    )
    return summary
