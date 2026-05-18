from edge.vision.geometry import (
    BBox,
    bbox_area_ratio,
    bbox_center_norm,
    crosswalk_orientation,
    horizontal_zone,
)


def test_bbox_center_norm_centered():
    box = BBox(280, 200, 360, 280)
    cx, cy = bbox_center_norm(box, 640, 480)
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6


def test_bbox_area_ratio():
    box = BBox(0, 0, 320, 240)
    assert abs(bbox_area_ratio(box, 640, 480) - 0.25) < 1e-6


def test_horizontal_zone_buckets():
    assert horizontal_zone(0.10) == "left"
    assert horizontal_zone(0.50) == "center"
    assert horizontal_zone(0.90) == "right"
    assert horizontal_zone(0.34) == "left"
    assert horizontal_zone(0.66) == "right"


def test_crosswalk_orientation_aligned():
    box = BBox(280, 300, 360, 460)
    assert crosswalk_orientation(box, 640, 480) == "aligned"


def test_crosswalk_orientation_drift_left():
    box = BBox(50, 300, 200, 460)
    assert crosswalk_orientation(box, 640, 480) == "drift_left"


def test_crosswalk_orientation_drift_right():
    box = BBox(450, 300, 600, 460)
    assert crosswalk_orientation(box, 640, 480) == "drift_right"
