import logging
from pathlib import Path

import numpy as np

from edge.config import (
    STT_COMPUTE_TYPE,
    STT_CPU_THREADS,
    STT_DEVICE,
    STT_LANGUAGE,
    STT_MODEL,
)

log = logging.getLogger(__name__)


class WhisperSTT:
    """Thin wrapper over faster-whisper.

    Audio input is expected as mono float32 PCM at 16 kHz, range [-1.0, 1.0].
    """

    def __init__(
        self,
        model_size: str = STT_MODEL,
        device: str = STT_DEVICE,
        compute_type: str = STT_COMPUTE_TYPE,
        language: str = STT_LANGUAGE,
        cpu_threads: int = STT_CPU_THREADS,
    ):
        from faster_whisper import WhisperModel

        log.info("Loading faster-whisper model=%s device=%s threads=%s",
                 model_size, device, cpu_threads)
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self.language = language

    def transcribe(self, pcm_f32_16k: np.ndarray) -> str:
        if pcm_f32_16k.ndim != 1:
            pcm_f32_16k = pcm_f32_16k.reshape(-1)
        if pcm_f32_16k.dtype != np.float32:
            pcm_f32_16k = pcm_f32_16k.astype(np.float32)

        # Cheap silence gate: if the chunk is below ~-46 dBFS, skip Whisper
        # entirely. This is faster and more reliable than Silero VAD on the
        # 1.5 s chunks we feed in. We log the RMS so the user can tell
        # whether the mic is actually picking anything up.
        rms = float(np.sqrt(np.mean(pcm_f32_16k ** 2)))
        if rms < 0.005:
            log.debug("audio chunk silent (rms=%.4f), skip", rms)
            return ""
        log.info("STT input rms=%.4f duration=%.2fs", rms, len(pcm_f32_16k) / 16000)

        segments, _info = self.model.transcribe(
            pcm_f32_16k,
            language=self.language,
            beam_size=1,
            # vad_filter disabled — it was eating every short chunk. Our RMS
            # gate above already prevents transcription on pure silence.
        )
        return "".join(seg.text for seg in segments).strip()

    def transcribe_file(self, path: str | Path) -> str:
        segments, _info = self.model.transcribe(str(path), language=self.language, beam_size=1)
        return "".join(seg.text for seg in segments).strip()
