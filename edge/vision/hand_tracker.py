"""MediaPipe HandLandmarker wrapper.

We only ever care about a single hand's position in the frame, expressed in
normalised (cx, cy) ∈ [0, 1]. The reference landmark is the **index-finger tip**
(landmark 8), since users reaching for an object lead with their index finger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

INDEX_FINGER_TIP = 8


@dataclass(frozen=True)
class HandPos:
    cx: float
    cy: float
    handedness: str  # "Left" | "Right" | "Unknown"


class HandTracker:
    """Thin wrapper around mediapipe.tasks.vision.HandLandmarker.

    The wrapped object is **not thread-safe**; call `track()` from a single
    thread (we already use asyncio.to_thread to avoid blocking the loop).
    """

    def __init__(self, task_path: str | Path, num_hands: int = 1):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        task_path = str(task_path)
        if not Path(task_path).exists():
            raise FileNotFoundError(
                f"MediaPipe HandLandmarker task not found at {task_path}"
            )

        base_options = mp_python.BaseOptions(model_asset_path=task_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def track(self, frame_bgr: np.ndarray) -> HandPos | None:
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        tip = lm[INDEX_FINGER_TIP]
        handedness = "Unknown"
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        return HandPos(cx=float(tip.x), cy=float(tip.y), handedness=handedness)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
