from __future__ import annotations

import os

from .base import Segment, STTEngine, Transcript

# Short names → MLX community model repos. Pass any HF repo as model to override.
_MODEL_MAP = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}


class MLXWhisperEngine(STTEngine):
    key = "mlx_whisper"
    default_model = "large-v3-turbo"
    requirements = "pip install mlx-whisper  (Apple Silicon only)"
    probe_module = "mlx_whisper"

    def _repo(self) -> str:
        return _MODEL_MAP.get(self.model, self.model)

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self._repo(),
            language=language,
            # false breaks repetition loops on long files, at some cost to consistency
            condition_on_previous_text=_env_flag("MLX_WHISPER_CONDITION_ON_PREVIOUS_TEXT", True),
        )
        segments = [
            Segment(start=round(s["start"], 2), end=round(s["end"], 2), text=s["text"].strip())
            for s in result.get("segments", [])
        ]
        return Transcript(
            text=result["text"].strip(), language=result.get("language"), segments=segments
        )


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes")
