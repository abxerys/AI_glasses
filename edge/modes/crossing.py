import asyncio
import logging

from edge.audio.speaker import Speaker
from edge.config import CROSSING_NO_CROSSWALK_FRAMES
from edge.vision.detector import Detector, pick_largest
from edge.vision.geometry import crosswalk_orientation

log = logging.getLogger(__name__)


# Map symbolic class -> aliases that may appear in the .pt's class names.
# Keep all lowercase; matching is case-insensitive against detector.names.
_LIGHT_ALIASES = {
    "red":   ("red", "red_light", "red light", "traffic_light_red",
              "stop", "countdown_stop",
              "紅燈", "红灯"),
    "green": ("green", "green_light", "green light", "traffic_light_green",
              "go", "countdown_go",
              "綠燈", "绿灯"),
}
_CROSSWALK_ALIASES = (
    "crosswalk", "zebra", "zebra_crossing", "zebracrossing",
    "road_crossing", "roadcrossing",
    "斑馬線", "斑马线",
)


def _resolve(detector: Detector, aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Pick which of detector.names match the given aliases (case-insensitive)."""
    if detector is None:
        return ()
    available = {str(n).lower(): str(n) for n in detector.names.values()}
    matches: list[str] = []
    for a in aliases:
        if a.lower() in available:
            matches.append(available[a.lower()])
    return tuple(matches)


async def run(*, video_q: asyncio.Queue, speaker: Speaker,
              traffic_light_detector: Detector | None,
              seg_detector: Detector | None,
              cancel: asyncio.Event,
              preview_bus=None) -> str:
    """Guide the user across a crosswalk.

    Two models cooperate:
      - traffic_light_detector: detection model with red / green classes.
      - seg_detector:           segmentation model whose 'crosswalk' class
                                gives us the strip the user must follow.
                                (Same model also serves the blind-path mode.)
    Either may be None; the mode degrades gracefully.
    """
    if traffic_light_detector is None and seg_detector is None:
        speaker.say_key("crossing_no_model", fallback="過街模型尚未載入")
        return "config_error"

    red_classes = _resolve(traffic_light_detector, _LIGHT_ALIASES["red"])
    green_classes = _resolve(traffic_light_detector, _LIGHT_ALIASES["green"])
    cw_classes = _resolve(seg_detector, _CROSSWALK_ALIASES)
    log.info("crossing classes: red=%s green=%s crosswalk=%s",
             red_classes, green_classes, cw_classes)

    speaker.say_key("crossing_enter", fallback="進入過街輔助模式")

    walk_state = "wait"   # wait | go
    no_cw_streak = 0

    while not cancel.is_set():
        try:
            frame = await asyncio.wait_for(video_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        h, w = frame.shape[:2]
        light_dets = traffic_light_detector.detect(frame) if traffic_light_detector else []
        cw_dets = seg_detector.detect(frame) if seg_detector else []
        if preview_bus is not None:
            preview_bus.publish(frame, light_dets + cw_dets, mode="crossing")

        red = pick_largest(light_dets, class_names=red_classes) if red_classes else None
        green = pick_largest(light_dets, class_names=green_classes) if green_classes else None
        cw = pick_largest(cw_dets, class_names=cw_classes) if cw_classes else None

        if red and (green is None or red.conf >= green.conf):
            speaker.say_key("light_red", fallback="紅燈，請等候")
            walk_state = "wait"
            no_cw_streak = 0
            continue

        if green is not None and walk_state != "go":
            speaker.say_key("light_green", fallback="綠燈，可以前進")
            walk_state = "go"

        if walk_state == "go":
            if cw is not None:
                no_cw_streak = 0
                drift = crosswalk_orientation(cw.bbox, w, h)
                if drift == "drift_left":
                    speaker.say_key("drift_correct_left",
                                    fallback="斑馬線在左側，請向左修正")
                elif drift == "drift_right":
                    speaker.say_key("drift_correct_right",
                                    fallback="斑馬線在右側，請向右修正")
                else:
                    speaker.say_key("keep_straight", fallback="保持直行")
            else:
                no_cw_streak += 1
                if no_cw_streak >= CROSSING_NO_CROSSWALK_FRAMES:
                    speaker.say_key("crossing_done", fallback="已通過斑馬線")
                    return "done"

    return "cancelled"
