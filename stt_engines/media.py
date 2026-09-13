from __future__ import annotations

import shutil
import subprocess


def audio_duration_s(path: str) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return round(float(out.stdout.strip()), 2)
    except (subprocess.SubprocessError, ValueError):
        return None
