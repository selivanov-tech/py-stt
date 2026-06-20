from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_text(text: str, language: str | None = None) -> str:
    """Lowercase, drop punctuation, collapse whitespace. For RU, fold ё → е."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    if language == "ru":
        text = text.replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    text = _SPACE.sub(" ", text)
    return text.strip()
