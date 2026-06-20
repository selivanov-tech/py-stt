from __future__ import annotations

import os
import tempfile
import time
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stt_engines import ENGINES, STTEngine, build_engine

app = FastAPI(title="stt model worker")

# Engines are cached by spec so a model loads once and is reused across requests.
_cache: dict[str, STTEngine] = {}


def _engine(spec: str) -> STTEngine:
    if spec not in _cache:
        _cache[spec] = build_engine(spec)
    return _cache[spec]


class TranscribeRequest(BaseModel):
    engine: str = "mock"
    model: str | None = None
    language: str | None = None
    audio_path: str | None = None
    audio_url: str | None = None
    audio_base64: str | None = None


class TranscribeResponse(BaseModel):
    text: str
    engine: str
    model: str
    latency_ms: int
    detected_language: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "engines": list(ENGINES), "loaded": list(_cache)}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    spec = req.engine + (f":{req.model}" if req.model else "")
    try:
        engine = _engine(spec)
    except Exception as exc:  # missing dep, unknown key, model load failure
        raise HTTPException(status_code=400, detail=f"engine '{spec}' load failed: {exc}") from exc

    audio = _resolve_audio(req)
    started = time.perf_counter()
    try:
        text = engine.transcribe(audio, req.language)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    return TranscribeResponse(
        text=text,
        engine=engine.key,
        model=engine.model,
        latency_ms=latency_ms,
        detected_language=req.language,
    )


def _resolve_audio(req: TranscribeRequest) -> str:
    if req.audio_path:
        return req.audio_path
    if req.audio_url:
        scheme = urlparse(req.audio_url).scheme.lower()
        if scheme in ("http", "https"):
            return _download(req.audio_url)
        # s3:// is handled in the eval (stt_eval.audio.resolve_audio), not here yet.
        raise HTTPException(status_code=400, detail=f"unsupported audio_url scheme: {scheme!r}")
    return "<no-audio>"  # fine for the mock engine


def _download(url: str) -> str:
    import requests

    suffix = os.path.splitext(urlparse(url).path)[1] or ".audio"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return path
