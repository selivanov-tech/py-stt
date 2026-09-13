from __future__ import annotations

import os

from .base import Segment, STTEngine, Transcript


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

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        raw_segments, info = self._model.transcribe(audio_path, language=language)
        segments = [
            Segment(start=round(seg.start, 2), end=round(seg.end, 2), text=seg.text.strip())
            for seg in raw_segments
        ]
        text = " ".join(seg.text for seg in segments).strip()
        return Transcript(text=text, language=info.language, segments=segments)
