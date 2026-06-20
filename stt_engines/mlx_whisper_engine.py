from __future__ import annotations

from .base import STTEngine

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

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=self._repo(), language=language)
        return result["text"].strip()
