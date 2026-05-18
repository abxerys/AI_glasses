from dataclasses import dataclass
from typing import Literal

from edge.vision.coco_classes import zh_to_coco

IntentName = Literal["voice_search", "crossing", "blind_path", "cancel", "unknown"]


@dataclass(frozen=True)
class Intent:
    name: IntentName
    target: str | None = None
    raw: str = ""


_CANCEL_WORDS = ("停", "取消", "結束", "離開", "退出")
_CROSSING_WORDS = ("過馬路", "斑馬線", "紅綠燈", "過街")
_BLIND_PATH_WORDS = ("盲道", "导盲", "導盲", "回到盲道", "盲道导航", "盲道導航")
_SEARCH_WORDS = ("找", "尋")


def parse(text: str) -> Intent:
    """Map a (Chinese) STT result to an intent.

    Priority: cancel > blind_path > crossing > voice_search > unknown.
    Blind-path is checked before crossing so that「回到盲道」doesn't get
    captured as a crossing command via a stray「過」-shaped word.
    """
    if not text:
        return Intent("unknown", raw=text)

    t = text.strip()

    if any(w in t for w in _CANCEL_WORDS):
        return Intent("cancel", raw=text)

    if any(w in t for w in _BLIND_PATH_WORDS):
        return Intent("blind_path", raw=text)

    if any(w in t for w in _CROSSING_WORDS):
        return Intent("crossing", raw=text)

    if any(w in t for w in _SEARCH_WORDS):
        target = zh_to_coco(t)
        if target is not None:
            return Intent("voice_search", target=target, raw=text)

    return Intent("unknown", raw=text)
