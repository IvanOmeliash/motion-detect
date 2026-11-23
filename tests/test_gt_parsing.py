"""GT annotation parsing: file format is (y1, x1, y2, x2), internal is (x1, y1, x2, y2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from motion_detect.utils import parse_gt_file

ANNOTATION = """\
# sample annotation
frame time_layer: 3 objects: (12.0, 34.0, 56.0, 78.0) (1.5, 2.5, 3.5, 4.5)
frame time_layer: 7 objects:
"""


def test_coordinate_order_conversion(tmp_path: Path) -> None:
    path = tmp_path / "Clip_01_gt.txt"
    path.write_text(ANNOTATION, encoding="utf-8")
    gt = parse_gt_file(str(path))  # no video -> default 1920x1080, no scaling
    assert set(gt) == {3, 7}
    # (y1, x1, y2, x2) = (12, 34, 56, 78) -> (x1, y1, x2, y2) = (34, 12, 78, 56)
    x1, y1, x2, y2 = gt[3][0]
    assert (x1, y1, x2, y2) == (34.0, 12.0, 78.0, 56.0)
    assert gt[3][1] == [2.5, 1.5, 4.5, 3.5]
    assert gt[7] == []  # empty list is a valid negative frame


def test_scaling_to_target(tmp_path: Path) -> None:
    path = tmp_path / "Clip_02_gt.txt"
    path.write_text(ANNOTATION, encoding="utf-8")
    gt = parse_gt_file(str(path), target_w=640, target_h=360)
    # 1920x1080 -> 640x360 is a factor 1/3 on both axes
    x1, y1, x2, y2 = gt[3][0]
    assert (x1, y1, x2, y2) == pytest.approx((34.0 / 3.0, 12.0 / 3.0, 78.0 / 3.0, 56.0 / 3.0))


def test_target_requires_both_dims(tmp_path: Path) -> None:
    path = tmp_path / "Clip_03_gt.txt"
    path.write_text(ANNOTATION, encoding="utf-8")
    try:
        parse_gt_file(str(path), target_w=640)
    except ValueError:
        return
    raise AssertionError("expected ValueError when only target_w is given")
