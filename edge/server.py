import asyncio
import logging
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


class StreamHub:
    """Routes WebSocket messages between ESP32-S3 and the rest of the app.

    Endpoints (all under the same ws server, dispatched by path):
      /ws/video      ESP32 -> PC   JPEG frames (binary)
      /ws/audio_in   ESP32 -> PC   PCM16 mono 16 kHz (binary chunks)
      /ws/audio_out  PC    -> ESP32  MP3 bytes (binary)

    The hub holds:
      - one outbound video queue (decoded BGR ndarrays)  -> consumed by state machine
      - a rolling PCM buffer feeding the STT loop
      - the latest /ws/audio_out client socket (only one ESP32 is expected)
    """

    def __init__(self, video_q: asyncio.Queue,
                 audio_in_buffer: "AudioRingBuffer"):
        self.video_q = video_q
        self.audio_in = audio_in_buffer
        self._audio_out_ws = None
        self._audio_out_lock = asyncio.Lock()

    async def handle(self, ws, path: str) -> None:
        log.info("ws connect path=%s peer=%s", path, getattr(ws, "remote_address", "?"))
        try:
            if path.endswith("/ws/video"):
                await self._handle_video(ws)
            elif path.endswith("/ws/audio_in"):
                await self._handle_audio_in(ws)
            elif path.endswith("/ws/audio_out"):
                await self._handle_audio_out(ws)
            else:
                log.warning("unknown ws path: %s", path)
                await ws.close(code=1008, reason="unknown path")
        finally:
            log.info("ws disconnect path=%s", path)

    async def _handle_video(self, ws) -> None:
        import cv2

        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            arr = np.frombuffer(msg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if self.video_q.full():
                try:
                    self.video_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self.video_q.put(frame)

    async def _handle_audio_in(self, ws) -> None:
        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_in.write(pcm)

    async def _handle_audio_out(self, ws) -> None:
        self._audio_out_ws = ws
        try:
            async for _ in ws:
                pass
        finally:
            if self._audio_out_ws is ws:
                self._audio_out_ws = None

    async def send_audio_out(self, mp3_bytes: bytes) -> None:
        ws = self._audio_out_ws
        if ws is None:
            log.debug("audio_out: no ESP32 connected, dropping %d bytes", len(mp3_bytes))
            return
        async with self._audio_out_lock:
            try:
                await ws.send(mp3_bytes)
            except Exception:
                log.exception("audio_out send failed")


class AudioRingBuffer:
    """Append-only ring of PCM float32 samples at a fixed sample rate.

    Consumers call `drain_seconds(n)` to atomically pull the most recent
    `n` seconds of audio for STT.
    """

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 8.0):
        self.sample_rate = sample_rate
        self._buf = deque(maxlen=int(sample_rate * max_seconds))

    def write(self, pcm: np.ndarray) -> None:
        self._buf.extend(pcm.tolist())

    def drain_seconds(self, seconds: float) -> np.ndarray:
        n = int(self.sample_rate * seconds)
        if len(self._buf) < n:
            return np.empty(0, dtype=np.float32)
        data = np.array(list(self._buf)[-n:], dtype=np.float32)
        self._buf.clear()
        return data

    def drain_all(self, max_seconds: float) -> np.ndarray:
        """Take every sample currently buffered, capped at the most recent
        `max_seconds`. Returns empty array if the buffer is empty.

        Used when we want to avoid losing speech that arrived while the
        previous STT call was still running, but also don't want to feed
        whisper an ever-growing chunk if it falls behind realtime.
        """
        n = len(self._buf)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        cap = int(self.sample_rate * max_seconds)
        if n > cap:
            data = np.array(list(self._buf)[-cap:], dtype=np.float32)
        else:
            data = np.array(list(self._buf), dtype=np.float32)
        self._buf.clear()
        return data

    def __len__(self) -> int:
        return len(self._buf)
