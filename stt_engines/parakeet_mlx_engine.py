from __future__ import annotations

import os

from .base import Segment, STTEngine, Transcript

# Short names → MLX community model repos. Pass any HF repo as model to override.
_MODEL_MAP = {
    "tdt-0.6b-v3": "mlx-community/parakeet-tdt-0.6b-v3",
    "tdt-0.6b-v2": "mlx-community/parakeet-tdt-0.6b-v2",
}


class ParakeetMLXEngine(STTEngine):
    key = "parakeet_mlx"
    default_model = "tdt-0.6b-v3"
    languages = "v3: 25 EU incl. en, ru; v2: en"
    requirements = "uv sync --extra parakeet  (Apple Silicon)"
    probe_module = "parakeet_mlx"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(_MODEL_MAP.get(self.model, self.model))
        self._chunk_s = float(os.getenv("PARAKEET_CHUNK_SECONDS", "120"))
        self._overlap_s = float(os.getenv("PARAKEET_OVERLAP_SECONDS", "15"))

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        # Parakeet detects the language itself and takes no language hint.
        result = self._model.transcribe(
            audio_path, chunk_duration=self._chunk_s, overlap_duration=self._overlap_s
        )
        segments = [
            Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
            for s in result.sentences
        ]
        return Transcript(text=result.text.strip(), segments=segments)
