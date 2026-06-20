import json

import pytest

from stt_eval.manifest import load_manifest


def test_load_jsonl(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        '{"id":"a","audio":"x.wav","reference":"hi","language":"en"}\n'
        '{"id":"b","audio":"y.wav","reference":"да","language":"ru"}\n',
        encoding="utf-8",
    )
    samples = load_manifest(p)
    assert len(samples) == 2
    assert samples[0].id == "a"
    assert samples[1].language == "ru"


def test_missing_field_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"a","audio":"x.wav"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(p)


def test_json_array(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps([{"id": "a", "audio": "x.wav", "reference": "hi"}]),
        encoding="utf-8",
    )
    samples = load_manifest(p)
    assert samples[0].id == "a"
    assert samples[0].language is None
