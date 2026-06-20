from __future__ import annotations

import os

from .base import STTEngine


class FasterWhisperEngine(STTEngine):
    key = "faster_whisper"
    default_model = "large-v3"
    requirements = "pip install faster-whisper"
    probe_module = "faster_whisper"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        from faster_whisper import WhisperModel

        device = os.getenv("FASTER_WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("FASTER_WHISPER_COMPUTE", "int8")
        self._model = WhisperModel(self.model, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        segments, _info = self._model.transcribe(audio_path, language=language)
        return " ".join(seg.text for seg in segments).strip()
