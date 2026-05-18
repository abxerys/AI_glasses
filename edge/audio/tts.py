import asyncio
import logging
import sys

from edge.config import TTS_RATE, TTS_VOICE

log = logging.getLogger(__name__)


async def synthesize(text: str, voice: str = TTS_VOICE, rate: str = TTS_RATE) -> bytes:
    """Return MP3 bytes from edge-tts for the given text."""
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    chunks: list[bytes] = []
    async for ev in communicate.stream():
        if ev.get("type") == "audio":
            chunks.append(ev["data"])
    return b"".join(chunks)


async def synthesize_to_file(text: str, path: str, voice: str = TTS_VOICE) -> None:
    data = await synthesize(text, voice=voice)
    with open(path, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "測試一二三"
    out = sys.argv[-1] if sys.argv[-1].endswith(".mp3") else "tts_out.mp3"
    asyncio.run(synthesize_to_file(text, out))
    print(f"wrote {out}")
