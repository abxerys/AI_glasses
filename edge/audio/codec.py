"""Audio codec helpers.

`to_pcm16_mono(audio_bytes, sample_rate)` decodes any audio bytes that the
soundfile library understands (MP3 from edge-tts, WAV from the user's
pre-recorded library) and re-emits them as little-endian PCM16 mono at the
requested sample rate. This is the format the ESP32 I2S TX expects, so the
firmware only needs `i2s_write()` — no MP3/Ogg decoder on the device.

For resampling we use a simple linear interp via numpy. Quality is fine for
voice prompts at 16 kHz; if we ever need broadcast-quality output we can
swap to scipy.signal.resample_poly.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


def to_pcm16_mono(audio_bytes: bytes, target_sr: int) -> bytes:
    if not audio_bytes:
        return b""
    try:
        data, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32",
                                always_2d=False)
    except Exception:
        log.exception("audio decode failed; passing through raw bytes")
        return audio_bytes

    if data.ndim == 2:
        data = data.mean(axis=1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1).mean(axis=1)

    if src_sr != target_sr:
        data = _resample_linear(data, src_sr, target_sr)

    np.clip(data, -1.0, 1.0, out=data)
    pcm16 = (data * 32767.0).astype("<i2", copy=False)
    return pcm16.tobytes()


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or x.size == 0:
        return x
    duration = x.size / float(src_sr)
    n_out = int(round(duration * dst_sr))
    if n_out <= 1:
        return x[:1].copy()
    src_t = np.linspace(0.0, duration, num=x.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)
