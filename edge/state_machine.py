import asyncio
import logging
from typing import Any

from edge.audio.speaker import Speaker
from edge.intent import Intent
from edge.modes import blind_path, crossing, voice_search

log = logging.getLogger(__name__)


class StateMachine:
    """Owns the currently-active mode coroutine, if any.

    A 'mode' is an async function that takes (video_q, speaker, cancel_event, **kwargs)
    and returns a result string. The state machine ensures only one mode runs at a time.
    """

    def __init__(self, video_q: asyncio.Queue, speaker: Speaker, *,
                 item_detectors=None,
                 traffic_light_detector=None,
                 seg_detector=None,
                 hand_tracker=None):
        self.video_q = video_q
        self.speaker = speaker
        self.item_detectors = item_detectors
        self.traffic_light_detector = traffic_light_detector
        self.seg_detector = seg_detector
        self.hand_tracker = hand_tracker
        self._task: asyncio.Task | None = None
        self._cancel: asyncio.Event | None = None
        self._mode: str = "idle"

    @property
    def mode(self) -> str:
        return self._mode

    async def handle(self, intent: Intent) -> None:
        log.info("state=%s intent=%s", self._mode, intent)
        if intent.name == "cancel":
            await self._stop_active("cancelled", "已取消")
            return

        if intent.name == "voice_search":
            if self.item_detectors is None or self.item_detectors.empty:
                self.speaker.say_key("no_item_model", fallback="物品辨識模型尚未載入")
                return
            detector = self.item_detectors.for_class(intent.target)
            if detector is None:
                log.info("no detector knows class %r; available=%s",
                         intent.target, sorted(self.item_detectors.all_classes()))
                self.speaker.say_key("target_unsupported",
                                     fallback="目前模型沒有這個物品")
                return
            await self._stop_active(None, None)
            await self._start("voice_search", voice_search.run(
                target=intent.target,
                video_q=self.video_q,
                speaker=self.speaker,
                detector=detector,
                hand_tracker=self.hand_tracker,
                cancel=self._new_cancel(),
            ))
            return

        if intent.name == "crossing":
            if self.traffic_light_detector is None and self.seg_detector is None:
                self.speaker.say_key("no_crossing_model", fallback="過街模型尚未載入")
                return
            await self._stop_active(None, None)
            await self._start("crossing", crossing.run(
                video_q=self.video_q,
                speaker=self.speaker,
                traffic_light_detector=self.traffic_light_detector,
                seg_detector=self.seg_detector,
                cancel=self._new_cancel(),
            ))
            return

        if intent.name == "blind_path":
            if self.seg_detector is None:
                self.speaker.say_key("no_blind_path_model",
                                     fallback="盲道模型尚未載入")
                return
            await self._stop_active(None, None)
            await self._start("blind_path", blind_path.run(
                video_q=self.video_q,
                speaker=self.speaker,
                seg_detector=self.seg_detector,
                cancel=self._new_cancel(),
            ))
            return

        if intent.name == "unknown" and self._mode == "idle":
            self.speaker.say_key_throttled(
                "help_idle",
                fallback="請說找物品名稱、過馬路、或盲道導航")

    def _new_cancel(self) -> asyncio.Event:
        self._cancel = asyncio.Event()
        return self._cancel

    async def _start(self, name: str, coro) -> None:
        self._mode = name
        self.speaker.reset_throttle()
        self._task = asyncio.create_task(self._wrap(coro), name=f"mode-{name}")

    async def _wrap(self, coro) -> Any:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("mode %s crashed", self._mode)
            self.speaker.say_key("mode_error", fallback="發生錯誤，已返回待機")
        finally:
            self._mode = "idle"
            self._task = None

    async def _stop_active(self, key: str | None, fallback: str | None) -> None:
        if self._task is None or self._task.done():
            return
        if self._cancel is not None:
            self._cancel.set()
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        if key:
            self.speaker.say_key(key, fallback=fallback)
        self._mode = "idle"
        self._task = None
