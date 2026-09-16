import base64
import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from stt_worker.app import _check_public_url, app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "mock" in body["engines"]
    statuses = {e["key"]: e["status"] for e in body["catalog"]}
    assert statuses["mock"] == "built-in"


def test_transcribe_mock():
    r = client.post(
        "/transcribe",
        json={"engine": "mock", "audio_path": "x.wav", "language": "ru"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "mock"
    assert "transcript" in body["text"]
    assert body["detected_language"] == "ru"
    assert body["segments"] == []
    assert body["audio_duration_s"] is None  # x.wav does not exist


def test_inline_audio_goes_to_a_temp_file_that_is_removed():
    payload = base64.b64encode(b"RIFF not really a wav").decode()
    r = client.post("/transcribe", json={"engine": "mock", "audio_base64": payload})
    assert r.status_code == 200
    # the mock engine echoes the path it was given
    path = r.json()["text"].rsplit(" ", 1)[-1]
    assert path.endswith(".audio")
    assert not os.path.exists(path)


def test_invalid_inline_audio_returns_400():
    r = client.post("/transcribe", json={"engine": "mock", "audio_base64": "not base64!"})
    assert r.status_code == 400


def test_oversized_inline_audio_returns_413():
    r = client.post("/transcribe", json={"engine": "mock", "audio_base64": "A" * (45 << 20)})
    assert r.status_code == 413


def test_stub_engine_returns_501():
    r = client.post("/transcribe", json={"engine": "deepgram", "audio_path": "x.wav"})
    assert r.status_code == 501


def test_ssrf_blocks_loopback():
    with pytest.raises(HTTPException):
        _check_public_url("http://127.0.0.1/clip.wav")


def test_ssrf_blocks_link_local_metadata():
    with pytest.raises(HTTPException):
        _check_public_url("http://169.254.169.254/latest/meta-data/")
