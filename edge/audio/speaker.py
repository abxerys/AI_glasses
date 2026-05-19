import asyncio
import logging
from dataclasses import dataclass

from edge.audio.codec import to_pcm16_mono
from edge.audio.throttle import TTSThrottler
from edge.audio.tts import synthesize
from edge.audio.voice_assets import VoiceAssets
from edge.config import AUDIO_OUT_CHUNK_BYTES, AUDIO_OUT_SAMPLE_RATE

log = logging.getLogger(__name__)


@dataclass
class _Utterance:
    # Either `key` is set (look up pre-recorded asset, fall back to fallback text),
    # or only `text` is set (always synthesize via TTS).
    key: str | None
    text: str | None
    fallback: str | None


class Speaker:
    """High-level TTS / wav speaker.

    Three input methods:
      - `say(text)`                   — synthesize via edge-tts, always speak
      - `say_key(key, fallback)`      — play asset if present, else synthesize `fallback`
      - `say_throttled(text)` / `say_key_throttled(key, fallback)` — dedupe within window

    A background task drains the queue and plays one message at a time so
    utterances never overlap.

    The sink is an async callable that accepts the audio bytes (.wav or .mp3
    headers; downstream decoder figures it out).
    """

    def __init__(self, assets: VoiceAssets | None = None):
        self._queue: asyncio.Queue[_Utterance] = asyncio.Queue()
        self._throttler = TTSThrottler()
        self._sink = None
        self._task: asyncio.Task | None = None
        self._assets = assets

    def set_sink(self, sink) -> None:
        self._sink = sink

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="tts-speaker")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ---- text-only path ----
    def say(self, text: str) -> None:
        if text:
            self._queue.put_nowait(_Utterance(key=None, text=text, fallback=None))

    def say_throttled(self, text: str) -> None:
        if text and self._throttler.allow(f"text:{text}"):
            self._queue.put_nowait(_Utterance(key=None, text=text, fallback=None))

    # ---- keyed path (asset preferred, TTS fallback) ----
    def say_key(self, key: str, *, fallback: str | None = None) -> None:
        if key:
            self._queue.put_nowait(_Utterance(key=key, text=None, fallback=fallback))

    def say_key_throttled(self, key: str, *, fallback: str | None = None) -> None:
        if key and self._throttler.allow(f"key:{key}"):
            self._queue.put_nowait(_Utterance(key=key, text=None, fallback=fallback))

    def reset_throttle(self) -> None:
        self._throttler.reset()

    async def _run(self) -> None:
        while True:
            u = await self._queue.get()
            try:
                audio = await self._materialize(u)
                if not audio or self._sink is None:
                    continue
                # Chunk PCM frames so each WS binary message fits inside
                # the ESP32 WebSocketsClient buffer.
                for i in range(0, len(audio), AUDIO_OUT_CHUNK_BYTES):
                    await self._sink(audio[i : i + AUDIO_OUT_CHUNK_BYTES])
            except Exception:
                log.exception("speaker failed on %s", u)

    async def _materialize(self, u: _Utterance) -> bytes | None:
        if u.key is not None and self._assets is not None and self._assets.has(u.key):
            log.info("voice asset: %s", u.key)
            raw = self._assets.get_bytes(u.key)
        else:
            text = u.text if u.text is not None else u.fallback
            if not text:
                log.debug("no asset for key=%s and no fallback text", u.key)
                return None
            log.info("TTS: %s", text)
            raw = await synthesize(text)

        if not raw:
            return None
        # Decode + resample to a fixed PCM16 mono format so both fake_esp32.py
        # and the real ESP32 firmware can consume the stream identically.
        return await asyncio.to_thread(to_pcm16_mono, raw, AUDIO_OUT_SAMPLE_RATE)
