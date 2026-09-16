import sys
import types

import pytest

from stt_engines import ENGINES, build_engine, catalog, engine_status


def test_mlx_whisper_maps_segments_and_passes_condition_flag(monkeypatch):
    calls: dict = {}

    def fake_transcribe(path, **kwargs):
        calls.update(kwargs)
        return {
            "text": " привет ",
            "language": "ru",
            "segments": [{"start": 0.0, "end": 1.234, "text": " привет "}],
        }

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe)
    )
    monkeypatch.setenv("MLX_WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false")

    out = build_engine("mlx_whisper:large-v3").transcribe("clip.wav", "ru")

    assert calls["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"
    assert calls["condition_on_previous_text"] is False
    assert out.text == "привет"
    assert out.segments[0].end == 1.23
    assert out.segments[0].text == "привет"


def test_mock_engine_transcribes():
    eng = build_engine("mock")
    out = eng.transcribe("clip.wav", "en")
    assert "transcript" in out.text
    assert out.language == "en"
    assert eng.key == "mock"


def test_unknown_engine_raises():
    with pytest.raises(KeyError):
        build_engine("nope")


def test_registry_has_core_engines():
    for key in ("mock", "faster_whisper", "mlx_whisper", "parakeet_mlx", "deepgram"):
        assert key in ENGINES


def test_parakeet_is_a_real_engine_not_a_stub():
    cls = ENGINES["parakeet_mlx"]
    assert not cls.is_stub
    assert engine_status(cls) in ("available", "not installed")


def test_catalog_marks_builtins_and_stubs():
    status = {e["key"]: e["status"] for e in catalog()}
    assert status["mock"] == "built-in"
    assert status["deepgram"] == "stub"
    assert status["apple_speech"] == "stub"
