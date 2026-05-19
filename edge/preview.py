"""Optional preview window for the edge runtime.

When `python -m edge.main --preview` is used, this module opens a cv2
window showing the latest video frame received from the ESP32, with any
active detections drawn as bounding boxes.

The window is updated from an asyncio task at ~20 fps. Press 'q' (with
the window focused) to close it; the rest of the pipeline keeps running.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

    from edge.vision.detector import Detection

log = logging.getLogger(__name__)


@dataclass
class PreviewBus:
    """Shared blackboard: the currently-running mode writes here whenever
    it runs a detector, and the preview task reads."""

    frame: "np.ndarray | None" = None
    dets: list["Detection"] = field(default_factory=list)
    mode: str = "idle"

    def publish(self, frame: "np.ndarray", dets: list["Detection"],
                 mode: str = "idle") -> None:
        self.frame = frame
        self.dets = dets
        self.mode = mode


def _draw_overlay(frame, dets, mode: str):
    """Return a copy of `frame` with detection bboxes + class names overlaid."""
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(d.bbox.x1), int(d.bbox.y1),
                          int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.cls_name} {d.conf:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"mode: {mode}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


async def preview_loop(hub, bus: PreviewBus | None = None,
                        window: str = "AI_glasses preview",
                        fps: float = 20.0) -> None:
    log.info("preview window opened — focus the window and press 'q' to close")
    period = 1.0 / fps
    try:
        while True:
            # Prefer the (frame, dets) the mode published; fall back to the
            # raw most-recent frame from the hub when nothing's active.
            if bus is not None and bus.frame is not None:
                rendered = _draw_overlay(bus.frame, bus.dets, bus.mode)
            elif hub.latest_frame is not None:
                rendered = _draw_overlay(hub.latest_frame, [], "idle")
            else:
                rendered = None

            if rendered is not None:
                cv2.imshow(window, rendered)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    log.info("preview window closed by user")
                    break
            await asyncio.sleep(period)
    finally:
        # cv2.destroyWindow can raise on shutdown (e.g. if the headless build
        # was loaded), and we don't want that to mask the original error.
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
