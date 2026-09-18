import sys
import types

import pytest

from stt_engines import ENGINES, build_engine, catalog, engine_status
from stt_engines.parakeet_mlx_engine import _stft_overrun


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


def _fake_parakeet(monkeypatch, result):
    model = types.SimpleNamespace(transcribe=lambda path, **kwargs: result)
    parakeet = types.SimpleNamespace(get_logmel=lambda x, args: x)
    monkeypatch.setitem(
        sys.modules,
        "parakeet_mlx",
        types.SimpleNamespace(from_pretrained=lambda repo: model, parakeet=parakeet),
    )


def test_parakeet_drops_unknown_tokens(monkeypatch):
    # a clip cut mid-sentence came back as 430 <unk> tokens; a dash came back as one
    sentences = [
        types.SimpleNamespace(start=0.0, end=3.44, text="<unk>" * 430),
        types.SimpleNamespace(start=3.44, end=5.0, text=" Далее одну из задач <unk> это план."),
    ]
    _fake_parakeet(
        monkeypatch,
        types.SimpleNamespace(text="".join(s.text for s in sentences), sentences=sentences),
    )

    out = build_engine("parakeet_mlx").transcribe("clip.wav", "ru")

    assert out.text == "Далее одну из задач это план."
    assert [(s.start, s.end, s.text) for s in out.segments] == [
        (3.44, 5.0, "Далее одну из задач это план.")
    ]


def test_parakeet_all_unknown_tokens_give_an_empty_transcript(monkeypatch):
    sentence = types.SimpleNamespace(start=0.0, end=2.64, text="<unk>" * 330)
    _fake_parakeet(monkeypatch, types.SimpleNamespace(text=sentence.text, sentences=[sentence]))

    out = build_engine("parakeet_mlx").transcribe("clip.wav", "ru")

    assert out.text == ""
    assert out.segments == []


@pytest.mark.parametrize(
    ("n_samples", "overrun"),
    [(41536, 64), (54208, 32), (75520, 0), (142720, 0)],  # clips from the failed eval
)
def test_stft_overrun_of_parakeet_v3_frames(n_samples, overrun):
    assert _stft_overrun(n_samples, n_fft=512, win_length=400, hop_length=160) == overrun


def test_padding_by_the_overrun_keeps_frames_in_bounds_and_their_count():
    def frames(n):
        return (n + 512 - 400 + 160) // 160

    for n in range(16000, 16000 + 2 * 160):
        pad = _stft_overrun(n, 512, 400, 160)
        assert pad <= 512 - 400
        assert _stft_overrun(n + pad, 512, 400, 160) == 0
        assert frames(n + pad) == frames(n)


def test_parakeet_log_mel_stays_finite_when_freed_buffers_hold_inf(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("parakeet_mlx")
    from parakeet_mlx import parakeet
    from parakeet_mlx.audio import PreprocessArgs

    from stt_engines.parakeet_mlx_engine import _keep_stft_frames_in_bounds

    args = PreprocessArgs(  # parakeet-tdt-0.6b-v3
        sample_rate=16000,
        normalize="per_feature",
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        features=128,
        n_fft=512,
        dither=1e-05,
    )
    audio = mx.random.normal((41536,), key=mx.random.key(0)) * 0.05

    def log_mel_after_inf_was_freed(get_logmel):
        # the reflect-padded buffer is n + n_fft float32 samples; leave freed buffers of
        # about that size filled with inf so the allocator hands one back
        junk = [mx.full((41536 + 512 + 200,), float("inf")) for _ in range(8)]
        mx.eval(*junk)
        del junk
        mel = get_logmel(audio, args)
        mx.eval(mel)
        return mel

    monkeypatch.setattr(parakeet, "get_logmel", parakeet.get_logmel)
    if mx.isfinite(log_mel_after_inf_was_freed(parakeet.get_logmel)).all().item():
        pytest.skip("MLX did not reuse a poisoned buffer, the bug cannot show here")

    _keep_stft_frames_in_bounds()

    assert mx.isfinite(log_mel_after_inf_was_freed(parakeet.get_logmel)).all().item()


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
