"""Inspect a YOLO .pt file: print task type and class names.

Use to figure out which uploaded .pt is what:
    python tools/inspect_model.py models/trafficlight.pt
    python tools/inspect_model.py models/yolo-seg.pt
    python tools/inspect_model.py models/yoloe-11l-seg.pt

It also dumps a quick verdict: if classes contain crosswalk/zebra keywords,
it says "looks like crosswalk model"; if they contain red/green/light keywords,
"looks like traffic light model"; otherwise prints raw names so you can decide.
"""

import argparse
import sys
from pathlib import Path


CROSSWALK_KEYS = ("crosswalk", "zebra", "斑馬線", "斑马线", "crossing")
TRAFFIC_KEYS = ("red", "green", "yellow", "light", "紅燈", "綠燈", "红灯", "绿灯", "traffic")
SHOPPING_KEYS = ("bottle", "cup", "phone", "book", "key", "wallet")


def classify(names: list[str]) -> str:
    text = " ".join(names).lower()
    has_cw = any(k.lower() in text for k in CROSSWALK_KEYS)
    has_tl = any(k.lower() in text for k in TRAFFIC_KEYS)
    has_shop = sum(k in text for k in SHOPPING_KEYS) >= 2

    tags = []
    if has_cw:
        tags.append("CROSSWALK")
    if has_tl:
        tags.append("TRAFFIC_LIGHT")
    if has_shop:
        tags.append("SHOPPING/OBJECTS")
    if not tags:
        tags.append("UNKNOWN — inspect names manually")
    return ", ".join(tags)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("weights", help="Path to .pt file")
    args = p.parse_args()

    path = Path(args.weights)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(path))
    task = getattr(model, "task", "?")
    names = model.names

    if isinstance(names, dict):
        name_list = [v for _, v in sorted(names.items())]
    else:
        name_list = list(names)

    print(f"file:          {path}")
    print(f"task:          {task}")
    print(f"num_classes:   {len(name_list)}")
    print(f"classes:       {name_list}")
    print(f"verdict:       {classify(name_list)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
