from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float  # seconds from the start of the audio
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    language: str | None = None  # language the engine detected (None if it can't)
    segments: list[Segment] = field(default_factory=list)


class STTEngine(ABC):
    """One speech-to-text model. Subclasses keep heavy imports inside
    __init__/transcribe so a missing library never breaks the whole tool."""

    key: str = ""
    default_model: str = ""
    requirements: str = ""  # how to install / enable this engine
    languages: str = "en, ru"
    probe_module: str | None = None  # import name used to check availability
    is_stub: bool = False

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript: ...

    @property
    def label(self) -> str:
        return f"{self.key}:{self.model}" if self.model else self.key
