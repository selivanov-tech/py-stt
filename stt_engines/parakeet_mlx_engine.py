from __future__ import annotations

import functools
import os

from .base import Segment, STTEngine, Transcript

# Short names → MLX community model repos. Pass any HF repo as model to override.
_MODEL_MAP = {
    "tdt-0.6b-v3": "mlx-community/parakeet-tdt-0.6b-v3",
    "tdt-0.6b-v2": "mlx-community/parakeet-tdt-0.6b-v2",
}

_UNKNOWN_TOKEN = "<unk>"


class ParakeetMLXEngine(STTEngine):
    key = "parakeet_mlx"
    default_model = "tdt-0.6b-v3"
    languages = "v3: 25 EU incl. en, ru; v2: en"
    requirements = "uv sync --extra parakeet  (Apple Silicon)"
    probe_module = "parakeet_mlx"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        from parakeet_mlx import from_pretrained

        _keep_stft_frames_in_bounds()
        self._model = from_pretrained(_MODEL_MAP.get(self.model, self.model))
        self._chunk_s = float(os.getenv("PARAKEET_CHUNK_SECONDS", "120"))
        self._overlap_s = float(os.getenv("PARAKEET_OVERLAP_SECONDS", "15"))

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        # Parakeet detects the language itself and takes no language hint.
        result = self._model.transcribe(
            audio_path, chunk_duration=self._chunk_s, overlap_duration=self._overlap_s
        )
        segments = [
            Segment(start=round(s.start, 2), end=round(s.end, 2), text=text)
            for s in result.sentences
            if (text := _without_unknown_tokens(s.text))
        ]
        return Transcript(text=_without_unknown_tokens(result.text), segments=segments)


def _without_unknown_tokens(text: str) -> str:
    return " ".join(text.replace(_UNKNOWN_TOKEN, "").split())


def _stft_overrun(n_samples: int, n_fft: int, win_length: int, hop_length: int) -> int:
    """Samples that the last frame of parakeet_mlx.audio.stft reads past its buffer."""
    padded = n_samples + n_fft  # reflect padding of n_fft // 2 on both sides
    frames = (padded - win_length + hop_length) // hop_length
    return max(0, (frames - 1) * hop_length + n_fft - padded)


def _keep_stft_frames_in_bounds() -> None:
    """parakeet_mlx.audio.stft cuts frames with mx.as_strided, and for most audio lengths
    the last frame reaches up to n_fft - win_length samples past the end of the buffer.
    Those samples are stale memory from MLX's buffer cache. The window is zero there, so
    finite junk does no harm, but an inf left by earlier work makes the frame NaN,
    per-feature normalisation spreads the NaN to every frame, and greedy TDT decoding then
    emits token 0 (<unk>) max_symbols times per frame. It depends on what the process ran
    before, so a long-lived worker hits it and a fresh process does not.

    Appending that many zeros (at most 7 ms at 16 kHz) keeps every frame inside the buffer
    and keeps the frame count."""
    from parakeet_mlx import parakeet

    get_logmel = parakeet.get_logmel
    if hasattr(get_logmel, "__wrapped__"):
        return

    @functools.wraps(get_logmel)
    def get_logmel_in_bounds(x, args):
        import mlx.core as mx

        n_samples = max(x.shape[-1], args.pad_to)
        overrun = _stft_overrun(n_samples, args.n_fft, args.win_length, args.hop_length)
        if overrun:
            tail = n_samples + overrun - x.shape[-1]
            x = mx.pad(x, [(0, tail)], constant_values=args.pad_value)
        return get_logmel(x, args)

    parakeet.get_logmel = get_logmel_in_bounds
