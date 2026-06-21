import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from stt_worker.app import _check_public_url, app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "mock" in r.json()["engines"]


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


def test_stub_engine_returns_501():
    r = client.post("/transcribe", json={"engine": "deepgram", "audio_path": "x.wav"})
    assert r.status_code == 501


def test_ssrf_blocks_loopback():
    with pytest.raises(HTTPException):
        _check_public_url("http://127.0.0.1/clip.wav")


def test_ssrf_blocks_link_local_metadata():
    with pytest.raises(HTTPException):
        _check_public_url("http://169.254.169.254/latest/meta-data/")
