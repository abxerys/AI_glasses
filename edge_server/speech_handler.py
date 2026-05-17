"""Speech handler interfaces for STT/TTS processing."""


class SpeechHandler:
    def speech_to_text(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Convert input audio stream to text."""
        # TODO: Integrate STT provider/service.
        _ = (audio_bytes, sample_rate)
        return ""

    def text_to_speech(self, text: str, language: str = "zh-TW") -> bytes:
        """Convert text to speech audio bytes."""
        # TODO: Integrate TTS provider/service.
        _ = (text, language)
        return b""
