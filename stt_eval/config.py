from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    mlflow_uri: str = "./mlruns"
    experiment: str = "stt-eval"
    cache_dir: str = ".cache/audio"
    normalize: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
            experiment=os.getenv("MLFLOW_EXPERIMENT", "stt-eval"),
            cache_dir=os.getenv("AUDIO_CACHE_DIR", ".cache/audio"),
        )


def load_dotenv_if_present(path: str = ".env") -> None:
    """Load a .env file if one exists. Falls back to a tiny parser."""
    if not Path(path).exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
    except ImportError:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
