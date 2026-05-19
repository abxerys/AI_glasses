"""Voice search mode.

Two phases:
  1) HEAD_GUIDE   — locate the target with the YOLO detector and tell the user
                    to turn left/right/center until the target is centered in
                    view AND close enough (bbox area >= threshold).
  2) HAND_ALIGN   — track the user's index-finger tip with MediaPipe; tell them
                    to move the hand left/right/up/down until the tip overlaps
                    the target bbox, then announce success.

Phase 2 is skipped if no HandTracker was provided (degrades gracefully).
"""

from __future__ import annotations

import asyncio
import logging
import time

from edge.audio.speaker import Speaker
from edge.config import (
    HAND_ALIGN_HOLD_FRAMES,
    HAND_ALIGN_TOLERANCE,
    VOICE_SEARCH_CENTER_HI,
    VOICE_SEARCH_CENTER_LO,
    VOICE_SEARCH_FOUND_FRAMES,
    VOICE_SEARCH_NEAR_AREA,
    VOICE_SEARCH_TIMEOUT_SEC,
)
from edge.vision.coco_classes import en_to_zh
from edge.vision.detector import Detector, pick_largest
from edge.vision.geometry import bbox_area_ratio, bbox_center_norm, horizontal_zone
from edge.vision.hand_tracker import HandTracker

log = logging.getLogger(__name__)


# Phase 1 — head guidance keys (asset preferred; fallback text in zh-TW)
_HEAD_KEYS = {
    "left":   ("target_left",   "目標在左邊"),
    "right":  ("target_right",  "目標在右邊"),
    "center": ("target_center", "目標在正前方，請慢慢靠近"),
}


async def run(*, target: str, video_q: asyncio.Queue,
              speaker: Speaker, detector: Detector,
              hand_tracker: HandTracker | None,
              cancel: asyncio.Event,
              preview_bus=None) -> str:
    zh = en_to_zh(target)
    speaker.say_key("search_start", fallback=f"開始尋找{zh}")

    deadline = time.monotonic() + VOICE_SEARCH_TIMEOUT_SEC

    # ── Phase 1: head guide ───────────────────────────────────────────
    centered_frames = 0
    last_head_hint: str | None = None

    while not cancel.is_set():
        if time.monotonic() > deadline:
            speaker.say_key("search_timeout", fallback=f"沒有找到{zh}")
            return "timeout"

        frame = await _next_frame(video_q)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        dets = await asyncio.to_thread(detector.detect, frame)
        if preview_bus is not None:
            preview_bus.publish(frame, dets, mode=f"voice_search:{target}")
        target_det = pick_largest(dets, class_name=target)

        if target_det is None:
            centered_frames = 0
            if last_head_hint != "lost":
                speaker.say_key_throttled(
                    "target_lost", fallback=f"沒看到{zh}，請慢慢轉動頭部")
                last_head_hint = "lost"
            continue

        cx, _ = bbox_center_norm(target_det.bbox, w, h)
        area = bbox_area_ratio(target_det.bbox, w, h)
        zone = horizontal_zone(cx)

        log.debug("phase=head target=%s zone=%s area=%.3f cx=%.3f",
                  target, zone, area, cx)

        if zone == "center" and area >= VOICE_SEARCH_NEAR_AREA:
            centered_frames += 1
            if centered_frames >= VOICE_SEARCH_FOUND_FRAMES:
                # close + centered → either announce found, or enter hand phase
                if hand_tracker is None:
                    speaker.say_key("search_found",
                                    fallback=f"找到了，{zh}就在你正前方")
                    return "found"
                speaker.say_key("hand_phase_enter",
                                fallback="請伸手對準物品")
                break
        else:
            centered_frames = 0
            key, fb = _HEAD_KEYS[zone]
            if key != last_head_hint:
                speaker.say_key(key, fallback=fb)
                last_head_hint = key

    if cancel.is_set():
        return "cancelled"

    # ── Phase 2: hand–target alignment ────────────────────────────────
    return await _hand_align_phase(
        target=target, zh=zh, video_q=video_q,
        speaker=speaker, detector=detector, hand_tracker=hand_tracker,
        cancel=cancel, deadline=deadline, preview_bus=preview_bus,
    )


async def _hand_align_phase(*, target: str, zh: str,
                             video_q: asyncio.Queue, speaker: Speaker,
                             detector: Detector, hand_tracker: HandTracker,
                             cancel: asyncio.Event, deadline: float,
                             preview_bus=None) -> str:
    aligned_frames = 0
    last_hand_hint: str | None = None
    no_hand_frames = 0

    while not cancel.is_set():
        if time.monotonic() > deadline:
            speaker.say_key("search_timeout", fallback=f"沒有找到{zh}")
            return "timeout"

        frame = await _next_frame(video_q)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        # run both vision tasks in worker threads so the asyncio loop stays free
        dets, hand = await asyncio.gather(
            asyncio.to_thread(detector.detect, frame),
            asyncio.to_thread(hand_tracker.track, frame),
        )
        if preview_bus is not None:
            preview_bus.publish(frame, dets, mode=f"voice_search:{target}+hand")
        target_det = pick_largest(dets, class_name=target)

        if target_det is None:
            aligned_frames = 0
            speaker.say_key_throttled(
                "target_lost_hand",
                fallback=f"{zh}看不到了，請維持姿勢")
            continue

        tx, ty = bbox_center_norm(target_det.bbox, w, h)

        if hand is None:
            no_hand_frames += 1
            if no_hand_frames >= 8:
                speaker.say_key_throttled(
                    "hand_not_seen", fallback="沒看到您的手，請把手伸入鏡頭")
                no_hand_frames = 0
            continue
        no_hand_frames = 0

        dx = tx - hand.cx   # +ve → target is to the right of the hand
        dy = ty - hand.cy   # +ve → target is below the hand

        log.debug("phase=hand dx=%.3f dy=%.3f handedness=%s",
                  dx, dy, hand.handedness)

        if abs(dx) <= HAND_ALIGN_TOLERANCE and abs(dy) <= HAND_ALIGN_TOLERANCE:
            aligned_frames += 1
            if aligned_frames >= HAND_ALIGN_HOLD_FRAMES:
                speaker.say_key("search_done",
                                fallback=f"對準了，{zh}就在您手邊，請抓取")
                return "found"
            continue

        aligned_frames = 0
        # pick the dominant axis to nudge first
        if abs(dx) >= abs(dy):
            hint = ("hand_right", "手請往右移") if dx > 0 else ("hand_left", "手請往左移")
        else:
            hint = ("hand_down", "手請往下移") if dy > 0 else ("hand_up", "手請往上移")
        if hint[0] != last_hand_hint:
            speaker.say_key(hint[0], fallback=hint[1])
            last_hand_hint = hint[0]

    return "cancelled"


async def _next_frame(video_q: asyncio.Queue):
    try:
        return await asyncio.wait_for(video_q.get(), timeout=1.0)
    except asyncio.TimeoutError:
        return None
