from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edge.config import YOLO_CONF, YOLO_IMGSZ
from edge.vision.geometry import BBox


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    bbox: BBox
    # Segmentation models populate this with the polygon (Nx2 xy coords in pixel space).
    # Detection-only models leave it None.
    mask_xy: object = None


class Detector:
    def __init__(self, weights: str | Path, conf: float = YOLO_CONF):
        from ultralytics import YOLO

        weights = str(weights)
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights}. "
                "See models/README.md for how to obtain it."
            )
        self.model = YOLO(weights)
        self.conf = conf
        self.names: dict[int, str] = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        out: list[Detection] = []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        masks_xy = None
        if getattr(r, "masks", None) is not None and r.masks is not None:
            masks_xy = r.masks.xy

        for i, ((x1, y1, x2, y2), c, cid) in enumerate(zip(xyxy, confs, cls_ids)):
            mask_xy = masks_xy[i] if masks_xy is not None and i < len(masks_xy) else None
            out.append(
                Detection(
                    cls_id=int(cid),
                    cls_name=self.names.get(int(cid), str(cid)),
                    conf=float(c),
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                    mask_xy=mask_xy,
                )
            )
        return out


def pick_largest(dets: list[Detection], *, class_name: str | None = None,
                 class_names: tuple[str, ...] | None = None) -> Detection | None:
    candidates = dets
    if class_name is not None:
        candidates = [d for d in candidates if d.cls_name == class_name]
    if class_names is not None:
        candidates = [d for d in candidates if d.cls_name in class_names]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.bbox.w * d.bbox.h)
