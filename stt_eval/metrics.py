from __future__ import annotations

from dataclasses import asdict, dataclass

import jiwer
from rapidfuzz import fuzz

from .normalize import normalize_text


@dataclass
class Score:
    wer: float
    cer: float
    word_accuracy_pct: float
    similarity_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def score(
    reference: str,
    hypothesis: str,
    language: str | None = None,
    normalize: bool = True,
) -> Score:
    ref = normalize_text(reference, language) if normalize else reference
    hyp = normalize_text(hypothesis, language) if normalize else hypothesis

    if not ref:
        # jiwer needs a non-empty reference; define a sensible fallback.
        wer = 0.0 if not hyp else 1.0
        cer = wer
    else:
        wer = jiwer.wer(ref, hyp)
        cer = jiwer.cer(ref, hyp)

    word_accuracy_pct = max(0.0, 1.0 - wer) * 100.0
    similarity_pct = float(fuzz.ratio(ref, hyp))  # char-level closeness, 0..100

    return Score(
        wer=round(wer, 4),
        cer=round(cer, 4),
        word_accuracy_pct=round(word_accuracy_pct, 2),
        similarity_pct=round(similarity_pct, 2),
    )
