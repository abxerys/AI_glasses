"""Pretends to be an ESP32-S3 over WebSocket using the laptop's webcam + mic + speaker.

Usage:
    python tools/fake_esp32.py
    python tools/fake_esp32.py --host localhost --camera 0 --fps 10

Three concurrent WS connections are opened to the edge server:
    /ws/video      send JPEG frames from the webcam
    /ws/audio_in   send PCM16 mono 16kHz from the mic
    /ws/audio_out  receive PCM16 mono 16kHz from the server and play through speakers
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make `from edge.config import …` work when this is launched as
# `python tools/fake_esp32.py` from the repo root (no `-m`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import sounddevice as sd
import websockets

from edge.config import AUDIO_OUT_SAMPLE_RATE

log = logging.getLogger("fake_esp32")

SAMPLE_RATE = 16000
MIC_CHUNK_MS = 250
CONNECT_RETRY_DELAY = 2.0


async def _connect_with_retry(uri: str, label: str):
    """Keep retrying until the edge server accepts our connection.

    The edge server takes ~30–60 s on first run while faster-whisper downloads
    its model, so a single connect attempt will usually race past it. Retry
    forever — Ctrl+C is how the user stops the demo anyway.
    """
    while True:
        try:
            ws = await websockets.connect(uri, max_size=8 * 1024 * 1024)
            log.info("%s connected: %s", label, uri)
            return ws
        except (ConnectionRefusedError, OSError) as e:
            log.info("%s waiting for server (%s) …", label, e.__class__.__name__)
            await asyncio.sleep(CONNECT_RETRY_DELAY)


async def stream_video(uri: str, camera: int, fps: int, jpeg_quality: int,
                        preview: bool) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera}")
    period = 1.0 / fps
    win = "AI_glasses preview (fake_esp32)" if preview else None
    try:
        ws = await _connect_with_retry(uri, "video")
        async with ws:
            while True:
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.05)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if not ok:
                    continue
                await ws.send(buf.tobytes())
                if win is not None:
                    cv2.imshow(win, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
                await asyncio.sleep(period)
    finally:
        cap.release()
        if win is not None:
            cv2.destroyAllWindows()


async def stream_audio_in(uri: str) -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    chunk_frames = int(SAMPLE_RATE * MIC_CHUNK_MS / 1000)
    last_log = [0.0]

    def _cb(indata, _frames, _t, _status):
        rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        import time as _t
        now = _t.monotonic()
        # log mic level once a second so user can see if anything is being captured
        if now - last_log[0] >= 1.0:
            log.info("mic rms=%.4f %s", rms,
                     "(silent)" if rms < 0.005 else "(speaking)")
            last_log[0] = now
        pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(queue.put_nowait, pcm16)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=chunk_frames, callback=_cb,
    )
    stream.start()
    try:
        ws = await _connect_with_retry(uri, "audio_in")
        async with ws:
            while True:
                chunk = await queue.get()
                await ws.send(chunk)
    finally:
        stream.stop()
        stream.close()


async def receive_audio_out(uri: str, sample_rate: int) -> None:
    """Receive PCM16 mono frames from the server and play them.

    The server (edge/audio/speaker.py) decodes its source audio (MP3 or
    WAV) to PCM16 at AUDIO_OUT_SAMPLE_RATE before sending, so we don't
    need any decoder here — just feed sounddevice directly.
    """
    ws = await _connect_with_retry(uri, "audio_out")
    async with ws:
        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            try:
                pcm = np.frombuffer(msg, dtype="<i2").astype(np.float32) / 32768.0
                sd.play(pcm, sample_rate, blocking=False)
            except Exception:
                log.exception("playback failed")


async def amain(args) -> None:
    base = f"ws://{args.host}:{args.port}"
    tasks = []
    if not args.no_video:
        tasks.append(stream_video(f"{base}/ws/video", args.camera, args.fps,
                                  args.jpeg_quality, args.preview))
    else:
        log.info("video stream disabled (--no-video); expecting real ESP32 camera")
    if not args.no_audio_in:
        tasks.append(stream_audio_in(f"{base}/ws/audio_in"))
    if not args.no_audio_out:
        tasks.append(receive_audio_out(f"{base}/ws/audio_out", AUDIO_OUT_SAMPLE_RATE))
    if not tasks:
        log.error("nothing to do; remove at least one --no-* flag")
        return
    await asyncio.gather(*tasks)


def _list_audio_devices() -> None:
    print(sd.query_devices())
    print(f"\ndefault input  : {sd.default.device[0]}")
    print(f"default output : {sd.default.device[1]}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument("--preview", action="store_true",
                   help="show webcam frames in a window; press q to quit")
    p.add_argument("--list-mics", action="store_true",
                   help="print available audio devices and exit")
    p.add_argument("--mic", type=int, default=None,
                   help="sounddevice input device index (use --list-mics to find)")
    p.add_argument("--no-video", action="store_true",
                   help="don't stream laptop webcam (use when real ESP32 sends video)")
    p.add_argument("--no-audio-in", action="store_true",
                   help="don't capture laptop mic (use when real ESP32 sends audio)")
    p.add_argument("--no-audio-out", action="store_true",
                   help="don't play TTS through laptop speakers")
    args = p.parse_args()

    if args.list_mics:
        _list_audio_devices()
        return

    if args.mic is not None:
        sd.default.device = (args.mic, sd.default.device[1])

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
