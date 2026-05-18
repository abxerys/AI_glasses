"""Replay a video file as if it were the ESP32-S3 camera stream.

Useful for testing the crossing mode with pre-recorded crosswalk footage,
without needing a webcam or the real device.

Usage:
    python tools/video_replay.py path/to/crosswalk.mp4
    python tools/video_replay.py path/to/clip.mp4 --loop
"""

import argparse
import asyncio
import logging

import cv2
import websockets

log = logging.getLogger("video_replay")


async def replay(path: str, uri: str, fps: int, jpeg_quality: int, loop: bool) -> None:
    while True:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
        period = 1.0 / min(fps, src_fps)
        try:
            async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
                log.info("replaying %s -> %s at %.1f fps", path, uri, 1 / period)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    if not ok:
                        continue
                    await ws.send(buf.tobytes())
                    await asyncio.sleep(period)
        finally:
            cap.release()
        if not loop:
            return


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()
    uri = f"ws://{args.host}:{args.port}/ws/video"
    asyncio.run(replay(args.video, uri, args.fps, args.jpeg_quality, args.loop))


if __name__ == "__main__":
    main()
