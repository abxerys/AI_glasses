import time

from edge.config import TTS_THROTTLE_SEC


class TTSThrottler:
    """Drops repeats of the same text within `interval` seconds.

    Distinct texts always pass through, so we don't suppress
    'green light' just because 'red light' was spoken recently.
    """

    def __init__(self, interval: float = TTS_THROTTLE_SEC):
        self.interval = interval
        self._last: dict[str, float] = {}

    def allow(self, text: str) -> bool:
        now = time.monotonic()
        last = self._last.get(text, 0.0)
        if now - last < self.interval:
            return False
        self._last[text] = now
        return True

    def reset(self) -> None:
        self._last.clear()
