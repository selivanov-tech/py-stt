from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    id: str
    audio: str  # s3://bucket/key, https://..., or a local path
    reference: str  # human ground-truth text
    language: str | None = None  # "en" | "ru" | None (auto-detect)
    notes: str = ""


_REQUIRED = ("id", "audio", "reference")


def _to_sample(row: dict, idx: int) -> Sample:
    missing = [k for k in _REQUIRED if not row.get(k)]
    if missing:
        raise ValueError(f"manifest item {idx}: missing field(s): {', '.join(missing)}")
    return Sample(
        id=str(row["id"]),
        audio=str(row["audio"]),
        reference=str(row["reference"]),
        language=row.get("language") or None,
        notes=str(row.get("notes", "")),
    )


def load_manifest(path: str | Path) -> list[Sample]:
    """Load a dataset manifest. Supports .jsonl (one object per line) and a
    .json array."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if p.suffix == ".json" and text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    return [_to_sample(row, i) for i, row in enumerate(rows)]
