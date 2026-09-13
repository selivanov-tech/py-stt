"""Shared STT engine adapters, used by both the inference worker and the eval."""

from __future__ import annotations

import importlib.util

from .base import Segment, STTEngine, Transcript
from .faster_whisper_engine import FasterWhisperEngine
from .mlx_whisper_engine import MLXWhisperEngine
from .mock import MockEngine
from .parakeet_mlx_engine import ParakeetMLXEngine
from .stubs import (
    AppleSpeechEngine,
    AssemblyAIEngine,
    DeepgramEngine,
    ElevenLabsEngine,
    WhisperCppEngine,
)

_ALL = [
    MockEngine,
    FasterWhisperEngine,
    MLXWhisperEngine,
    ParakeetMLXEngine,
    WhisperCppEngine,
    AppleSpeechEngine,
    DeepgramEngine,
    AssemblyAIEngine,
    ElevenLabsEngine,
]

# every registered engine is concrete (the shared ABC base trips mypy here)
ENGINES: dict[str, type[STTEngine]] = {cls.key: cls for cls in _ALL}  # type: ignore[type-abstract]

__all__ = [
    "ENGINES",
    "Segment",
    "STTEngine",
    "Transcript",
    "build_engine",
    "catalog",
    "engine_status",
]


def build_engine(spec: str) -> STTEngine:
    """Build an engine from a spec like 'mock', 'faster_whisper', or
    'mlx_whisper:large-v3'."""
    key, _, model = spec.partition(":")
    if key not in ENGINES:
        known = ", ".join(ENGINES)
        raise KeyError(f"unknown engine {key!r}. Known: {known}")
    return ENGINES[key](model or None)


def engine_status(cls: type[STTEngine]) -> str:
    if cls.is_stub:
        return "stub"
    if cls.probe_module is None:
        return "built-in"
    return "available" if importlib.util.find_spec(cls.probe_module) else "not installed"


def catalog() -> list[dict[str, str]]:
    return [
        {
            "key": key,
            "default_model": cls.default_model,
            "languages": cls.languages,
            "status": engine_status(cls),
            "requirements": cls.requirements,
        }
        for key, cls in ENGINES.items()
    ]
