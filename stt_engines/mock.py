from __future__ import annotations

from .base import STTEngine


class MockEngine(STTEngine):
    """Built-in fake engine. Loads instantly, needs no ML deps. Use it to smoke
    test the service end to end without downloading models."""

    key = "mock"
    default_model = "echo"
    requirements = "(built in — no install)"
    languages = "en, ru"
    probe_module = None  # always available

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        return f"[mock:{language or 'auto'}] transcript for {audio_path}"
