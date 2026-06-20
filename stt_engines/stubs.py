from __future__ import annotations

from .base import STTEngine


class _Stub(STTEngine):
    """Placeholder engine. Registered so it shows up everywhere; implement
    transcribe() when you want it for real."""

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        raise NotImplementedError(
            f"Engine '{self.key}' is a stub. Enable it with: {self.requirements}. "
            f"Then implement transcribe() in stt_engines/."
        )


class WhisperCppEngine(_Stub):
    key = "whisper_cpp"
    default_model = "ggml-large-v3"
    requirements = "build whisper.cpp + `pip install pywhispercpp` (or call its CLI)"


class AppleSpeechEngine(_Stub):
    key = "apple_speech"
    default_model = "system"
    requirements = (
        "macOS built-in: a small Swift helper using SpeechAnalyzer / "
        "SFSpeechRecognizer that takes a wav path and prints the transcript"
    )


class ParakeetMLXEngine(_Stub):
    key = "parakeet_mlx"
    default_model = "mlx-community/parakeet-tdt-0.6b-v2"
    languages = "en (weak ru)"
    requirements = "pip install parakeet-mlx  (Apple Silicon; English-focused)"


class DeepgramEngine(_Stub):
    key = "deepgram"
    default_model = "nova-2"
    requirements = "pip install deepgram-sdk + set DEEPGRAM_API_KEY"


class AssemblyAIEngine(_Stub):
    key = "assemblyai"
    default_model = "best"
    requirements = "pip install assemblyai + set ASSEMBLYAI_API_KEY"


class ElevenLabsEngine(_Stub):
    key = "elevenlabs"
    default_model = "scribe_v1"
    requirements = "pip install elevenlabs + set ELEVENLABS_API_KEY"
