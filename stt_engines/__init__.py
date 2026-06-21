"""Shared STT engine adapters, used by both the inference worker and the eval."""

from __future__ import annotations

from .base import STTEngine, Transcript
from .faster_whisper_engine import FasterWhisperEngine
from .mlx_whisper_engine import MLXWhisperEngine
from .mock import MockEngine
from .stubs import (
    AppleSpeechEngine,
    AssemblyAIEngine,
    DeepgramEngine,
    ElevenLabsEngine,
    ParakeetMLXEngine,
    WhisperCppEngine,
)

_ALL = [
    MockEngine,
    FasterWhisperEngine,
    MLXWhisperEngine,
    WhisperCppEngine,
    AppleSpeechEngine,
    ParakeetMLXEngine,
    DeepgramEngine,
    AssemblyAIEngine,
    ElevenLabsEngine,
]

# every registered engine is concrete (the shared ABC base trips mypy here)
ENGINES: dict[str, type[STTEngine]] = {cls.key: cls for cls in _ALL}  # type: ignore[type-abstract]

__all__ = ["ENGINES", "STTEngine", "Transcript", "build_engine"]


def build_engine(spec: str) -> STTEngine:
    """Build an engine from a spec like 'mock', 'faster_whisper', or
    'mlx_whisper:large-v3'."""
    key, _, model = spec.partition(":")
    if key not in ENGINES:
        known = ", ".join(ENGINES)
        raise KeyError(f"unknown engine {key!r}. Known: {known}")
    return ENGINES[key](model or None)
