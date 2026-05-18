from dataclasses import dataclass

from edge.config import (
    CROSSING_DRIFT_TOLERANCE,
    VOICE_SEARCH_CENTER_HI,
    VOICE_SEARCH_CENTER_LO,
)


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


def bbox_center_norm(bbox: BBox, frame_w: int, frame_h: int) -> tuple[float, float]:
    cx = (bbox.x1 + bbox.x2) / 2.0 / frame_w
    cy = (bbox.y1 + bbox.y2) / 2.0 / frame_h
    return cx, cy


def bbox_area_ratio(bbox: BBox, frame_w: int, frame_h: int) -> float:
    return (bbox.w * bbox.h) / float(frame_w * frame_h)


def horizontal_zone(cx: float) -> str:
    if cx < VOICE_SEARCH_CENTER_LO:
        return "left"
    if cx > VOICE_SEARCH_CENTER_HI:
        return "right"
    return "center"


def crosswalk_orientation(bbox: BBox, frame_w: int, frame_h: int) -> str:
    """Heuristic: a well-aligned crosswalk fills the bottom-center of view.

    Returns 'aligned' | 'drift_left' | 'drift_right'.
    drift_left means the crosswalk is to the LEFT of where the user is heading,
    so the user has drifted RIGHT and should correct LEFT.
    """
    cx, _ = bbox_center_norm(bbox, frame_w, frame_h)
    offset = cx - 0.5
    if abs(offset) <= CROSSING_DRIFT_TOLERANCE:
        return "aligned"
    return "drift_left" if offset < 0 else "drift_right"
