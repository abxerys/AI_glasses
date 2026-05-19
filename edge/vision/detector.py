import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edge.config import YOLO_CONF, YOLO_IMGSZ
from edge.vision.geometry import BBox

log = logging.getLogger(__name__)

# Stock model names that ultralytics can download on demand. If the user
# hasn't placed the file in models/ yet, we let YOLO() resolve by name and
# auto-download to its own cache. Custom-trained .pt files must still be
# placed at the configured path.
_AUTO_DOWNLOAD_NAMES = {
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
}


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

        weights_str = str(weights)
        weights_path = Path(weights_str)
        if weights_path.exists():
            self.model = YOLO(weights_str)
        elif weights_path.name in _AUTO_DOWNLOAD_NAMES:
            log.info("Local %s not found; ultralytics will auto-download",
                     weights_path.name)
            self.model = YOLO(weights_path.name)
        else:
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_str}. "
                "See models/README.md for how to obtain it."
            )
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


class ItemDetectorSet:
    """A bundle of item-detection models (e.g. shopping + COCO) routed by
    class name. Earlier entries win on ties, so the custom shopping model is
    consulted before the generic COCO fallback.
    """

    def __init__(self, detectors: list["Detector"]):
        self.detectors: list["Detector"] = [d for d in detectors if d is not None]

    @property
    def empty(self) -> bool:
        return not self.detectors

    def for_class(self, name: str) -> "Detector | None":
        for d in self.detectors:
            if name in d.names.values():
                return d
        return None

    def all_classes(self) -> set[str]:
        out: set[str] = set()
        for d in self.detectors:
            out.update(d.names.values())
        return out
