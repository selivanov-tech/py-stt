from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import socket
import tempfile
import threading
import time
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stt_engines import ENGINES, STTEngine, build_engine, catalog
from stt_engines.media import audio_duration_s

app = FastAPI(title="stt model worker")

_MAX_INLINE_AUDIO_BYTES = 32 << 20  # matches go-stt's request cap

# Engines are cached by spec so a model loads once and is reused across requests.
_cache: dict[str, STTEngine] = {}
# One lock for loading and inference: local models share one GPU, and MLX is not
# safe to drive from several threads at once. Concurrent requests wait here.
_model_lock = threading.Lock()


def _engine(spec: str) -> STTEngine:
    eng = _cache.get(spec)
    if eng is not None:
        return eng
    with _model_lock:  # double-checked: don't load the same model twice concurrently
        eng = _cache.get(spec)
        if eng is None:
            eng = build_engine(spec)
            _cache[spec] = eng
        return eng


class TranscribeRequest(BaseModel):
    engine: str = "mock"
    model: str | None = None
    language: str | None = None
    audio_path: str | None = None
    audio_url: str | None = None
    audio_base64: str | None = None


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    engine: str
    model: str
    latency_ms: int
    detected_language: str | None = None
    audio_duration_s: float | None = None
    segments: list[SegmentOut] = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engines": list(ENGINES),
        "loaded": list(_cache),
        "catalog": catalog(),
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    spec = req.engine + (f":{req.model}" if req.model else "")
    try:
        engine = _engine(spec)
    except Exception as exc:  # missing dep, unknown key, model load failure
        raise HTTPException(status_code=400, detail=f"engine '{spec}' load failed: {exc}") from exc

    audio, is_temp = _resolve_audio(req)
    duration_s = audio_duration_s(audio)
    try:
        with _model_lock:
            started = time.perf_counter()
            result = engine.transcribe(audio, req.language)
            latency_ms = int((time.perf_counter() - started) * 1000)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        if is_temp:
            _cleanup(audio)

    return TranscribeResponse(
        text=result.text,
        engine=engine.key,
        model=engine.model,
        latency_ms=latency_ms,
        detected_language=result.language,
        audio_duration_s=duration_s,
        segments=[SegmentOut(start=s.start, end=s.end, text=s.text) for s in result.segments],
    )


def _resolve_audio(req: TranscribeRequest) -> tuple[str, bool]:
    """Return (path, is_temp); is_temp paths are deleted after transcription."""
    if req.audio_path:
        return req.audio_path, False
    if req.audio_url:
        scheme = urlparse(req.audio_url).scheme.lower()
        if scheme in ("http", "https"):
            return _download(req.audio_url), True
        # s3:// is handled in the eval (stt_eval.audio.resolve_audio), not here yet.
        raise HTTPException(status_code=400, detail=f"unsupported audio_url scheme: {scheme!r}")
    if req.audio_base64:
        return _write_inline_audio(req.audio_base64), True
    return "<no-audio>", False  # fine for the mock engine


def _write_inline_audio(encoded: str) -> str:
    # base64 is 4 chars per 3 bytes: refuse oversized input before decoding it
    if len(encoded) > (_MAX_INLINE_AUDIO_BYTES // 3 + 1) * 4:
        raise HTTPException(status_code=413, detail="audio_base64 is larger than 32 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"audio_base64 is not valid: {exc}") from exc
    # no extension on purpose: ffmpeg detects the container from the content
    fd, path = tempfile.mkstemp(suffix=".audio")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _check_public_url(url: str) -> None:
    """Block SSRF: reject audio_url hosts that resolve to non-public addresses
    (loopback, private, link-local — e.g. cloud metadata at 169.254.169.254)."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="audio_url has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot resolve {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(
                status_code=400,
                detail=f"audio_url host {host!r} resolves to a non-public address",
            )


def _download(url: str) -> str:
    import requests

    _check_public_url(url)
    suffix = os.path.splitext(urlparse(url).path)[1] or ".audio"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except Exception:
        _cleanup(path)
        raise
    return path
