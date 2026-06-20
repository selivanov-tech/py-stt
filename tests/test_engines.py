import pytest

from stt_engines import ENGINES, build_engine


def test_mock_engine_transcribes():
    eng = build_engine("mock")
    out = eng.transcribe("clip.wav", "en")
    assert "transcript" in out
    assert eng.key == "mock"


def test_unknown_engine_raises():
    with pytest.raises(KeyError):
        build_engine("nope")


def test_registry_has_core_engines():
    for key in ("mock", "faster_whisper", "mlx_whisper", "deepgram"):
        assert key in ENGINES
