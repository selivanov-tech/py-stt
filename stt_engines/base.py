from __future__ import annotations

from abc import ABC, abstractmethod


class STTEngine(ABC):
    """One speech-to-text model. Subclasses keep heavy imports inside
    __init__/transcribe so a missing library never breaks the whole tool."""

    key: str = ""
    default_model: str = ""
    requirements: str = ""  # how to install / enable this engine
    languages: str = "en, ru"
    probe_module: str | None = None  # import name used to check availability

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None) -> str: ...

    @property
    def label(self) -> str:
        return f"{self.key}:{self.model}" if self.model else self.key
