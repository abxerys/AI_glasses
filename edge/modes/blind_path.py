"""Blind-path (盲道 / tactile paving) following mode.

Uses the shared segmentation model (yolo-seg.pt) but filters for tactile-paving
classes only. The user is guided to keep the detected paving strip centred in
the lower half of the frame; if the strip is lost, the user is told to search
for it.
"""

from __future__ import annotations

import asyncio
import logging

from edge.audio.speaker import Speaker
from edge.vision.detector import Detector, pick_largest
from edge.vision.geometry import crosswalk_orientation

log = logging.getLogger(__name__)


_BLIND_PATH_ALIASES = (
    "blind_path", "blindpath", "blind-path",
    "tactile", "tactile_paving", "tactile-paving",
    "paving", "guide_path", "guide-path", "guidepath",
    "盲道", "导盲", "導盲",
)

LOST_FRAMES_THRESHOLD = 12


def _resolve(detector: Detector, aliases: tuple[str, ...]) -> tuple[str, ...]:
    if detector is None:
        return ()
    available = {str(n).lower(): str(n) for n in detector.names.values()}
    return tuple(available[a.lower()] for a in aliases if a.lower() in available)


async def run(*, video_q: asyncio.Queue, speaker: Speaker,
              seg_detector: Detector | None,
              cancel: asyncio.Event) -> str:
    if seg_detector is None:
        speaker.say_key("no_blind_path_model", fallback="盲道模型尚未載入")
        return "config_error"

    bp_classes = _resolve(seg_detector, _BLIND_PATH_ALIASES)
    log.info("blind_path classes: %s", bp_classes)
    if not bp_classes:
        speaker.say_key("no_blind_path_class",
                        fallback="分割模型沒有盲道類別")
        return "config_error"

    speaker.say_key("blind_path_enter", fallback="進入盲道導航")

    lost_streak = 0
    last_hint: str | None = None

    while not cancel.is_set():
        try:
            frame = await asyncio.wait_for(video_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        h, w = frame.shape[:2]
        dets = await asyncio.to_thread(seg_detector.detect, frame)
        bp = pick_largest(dets, class_names=bp_classes)

        if bp is None:
            lost_streak += 1
            if lost_streak >= LOST_FRAMES_THRESHOLD and last_hint != "lost":
                speaker.say_key("blind_path_lost",
                                fallback="丟失路徑，請原地小幅轉動")
                last_hint = "lost"
            continue

        lost_streak = 0
        drift = crosswalk_orientation(bp.bbox, w, h)

        if drift == "aligned":
            if last_hint != "straight":
                speaker.say_key("blind_path_straight",
                                fallback="方向正確，請直行")
                last_hint = "straight"
        elif drift == "drift_left":
            # paving is to the LEFT of user heading → step LEFT to realign
            if last_hint != "left":
                speaker.say_key("blind_path_shift_left",
                                fallback="向左平移，對準盲道")
                last_hint = "left"
        else:
            if last_hint != "right":
                speaker.say_key("blind_path_shift_right",
                                fallback="向右平移，對準盲道")
                last_hint = "right"

    return "cancelled"
