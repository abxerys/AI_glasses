import asyncio
import logging
from pathlib import Path

import websockets

from edge.audio.speaker import Speaker
from edge.audio.stt import WhisperSTT
from edge.audio.voice_assets import VoiceAssets
from edge.config import (
    COCO_WEIGHTS,
    HAND_LANDMARKER_TASK,
    ROOT,
    SEGMENTATION_WEIGHTS,
    SHOPPING_WEIGHTS,
    STT_CHUNK_SECONDS,
    TRAFFIC_LIGHT_WEIGHTS,
    WS_HOST,
    WS_PORT,
)
from edge.intent import parse as parse_intent
from edge.server import AudioRingBuffer, StreamHub
from edge.state_machine import StateMachine
from edge.vision.detector import Detector, ItemDetectorSet

log = logging.getLogger(__name__)


def _load_detector(weights, label: str) -> Detector | None:
    try:
        d = Detector(weights)
        log.info("%s detector loaded from %s: %d classes",
                 label, weights, len(d.names))
        return d
    except FileNotFoundError as e:
        log.warning("%s detector unavailable: %s", label, e)
        return None
    except Exception:
        log.exception("%s detector failed to load", label)
        return None


def _load_hand_tracker():
    try:
        from edge.vision.hand_tracker import HandTracker

        h = HandTracker(HAND_LANDMARKER_TASK)
        log.info("hand tracker loaded from %s", HAND_LANDMARKER_TASK)
        return h
    except FileNotFoundError as e:
        log.warning("hand tracker unavailable: %s", e)
        return None
    except Exception:
        log.exception("hand tracker failed to load")
        return None


def _load_item_detectors() -> ItemDetectorSet:
    """Load BOTH the custom shopping model (if present) AND the COCO model.

    The shopping model is consulted first for items it knows about
    (AD_milk, Red_Bull). COCO covers the long tail of everyday objects
    (phone, bottle, chair, ...). ultralytics will auto-download yolov8s.pt
    on first run if the user hasn't placed it in models/.
    """
    detectors: list[Detector] = []
    if Path(SHOPPING_WEIGHTS).exists():
        d = _load_detector(SHOPPING_WEIGHTS, "item(shopping)")
        if d is not None:
            detectors.append(d)
    d = _load_detector(COCO_WEIGHTS, "item(COCO)")
    if d is not None:
        detectors.append(d)
    s = ItemDetectorSet(detectors)
    if not s.empty:
        log.info("item detectors total: %d, combined classes: %d",
                 len(s.detectors), len(s.all_classes()))
    return s


async def _stt_loop(stt: WhisperSTT, audio_buf: AudioRingBuffer,
                    state_machine: StateMachine) -> None:
    while True:
        await asyncio.sleep(STT_CHUNK_SECONDS)
        pcm = audio_buf.drain_seconds(STT_CHUNK_SECONDS)
        if pcm.size == 0:
            continue
        try:
            text = await asyncio.to_thread(stt.transcribe, pcm)
        except Exception:
            log.exception("stt failed")
            continue
        if not text:
            continue
        log.info("STT: %s", text)
        intent = parse_intent(text)
        await state_machine.handle(intent)


async def amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    video_q: asyncio.Queue = asyncio.Queue(maxsize=2)
    audio_buf = AudioRingBuffer(sample_rate=16000, max_seconds=8.0)
    hub = StreamHub(video_q=video_q, audio_in_buffer=audio_buf)

    voice_root = ROOT / "assets" / "voice"
    assets = VoiceAssets(voice_root)

    speaker = Speaker(assets=assets)
    speaker.set_sink(hub.send_audio_out)
    speaker.start()

    item_dets = _load_item_detectors()
    tl_det = _load_detector(TRAFFIC_LIGHT_WEIGHTS, "traffic_light")
    # One seg model covers both crosswalk and tactile-paving (盲道).
    # Each mode filters by class name to extract the slice it needs.
    seg_det = _load_detector(SEGMENTATION_WEIGHTS, "segmentation")
    hand = _load_hand_tracker()

    sm = StateMachine(
        video_q=video_q, speaker=speaker,
        item_detectors=item_dets,
        traffic_light_detector=tl_det,
        seg_detector=seg_det,
        hand_tracker=hand,
    )

    stt = WhisperSTT()
    asyncio.create_task(_stt_loop(stt, audio_buf, sm), name="stt-loop")

    async def _handler(ws):
        # websockets >= 13 moved `path` onto the request object.
        # Fall back to the legacy attribute for older releases.
        try:
            path = ws.request.path
        except AttributeError:
            path = getattr(ws, "path", "/")
        await hub.handle(ws, path)

    log.info("ws server listening on ws://%s:%d", WS_HOST, WS_PORT)
    async with websockets.serve(_handler, WS_HOST, WS_PORT, max_size=8 * 1024 * 1024):
        await asyncio.Future()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
