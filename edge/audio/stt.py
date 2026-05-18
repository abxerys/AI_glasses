import logging
from pathlib import Path

import numpy as np

from edge.config import STT_COMPUTE_TYPE, STT_DEVICE, STT_LANGUAGE, STT_MODEL

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
    ):
        from faster_whisper import WhisperModel

        log.info("Loading faster-whisper model=%s device=%s", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, pcm_f32_16k: np.ndarray) -> str:
        if pcm_f32_16k.ndim != 1:
            pcm_f32_16k = pcm_f32_16k.reshape(-1)
        if pcm_f32_16k.dtype != np.float32:
            pcm_f32_16k = pcm_f32_16k.astype(np.float32)
        segments, _info = self.model.transcribe(
            pcm_f32_16k,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return "".join(seg.text for seg in segments).strip()

    def transcribe_file(self, path: str | Path) -> str:
        segments, _info = self.model.transcribe(str(path), language=self.language, beam_size=1)
        return "".join(seg.text for seg in segments).strip()
