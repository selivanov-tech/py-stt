from stt_eval import compare


def test_clock_formats_minutes_and_seconds():
    assert compare._clock(0) == "00:00.0"
    assert compare._clock(303.24) == "05:03.2"


def test_compare_in_process_writes_transcripts_and_summary(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"not really audio")
    out = tmp_path / "out"

    code = compare.main(
        [
            str(audio),
            "--engines",
            "mock,mock:second,deepgram",
            "--out",
            str(out),
            "--language",
            "ru",
        ]
    )

    # deepgram is a stub, so the run reports a failure but still writes the rest
    assert code == 1
    assert (out / "mock.txt").read_text(encoding="utf-8").startswith("[mock:ru]")
    assert (out / "mock_second.txt").exists()
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| mock | mock:second | 0.000 |" in summary  # identical mock text → full agreement
    assert "- deepgram:" in summary
    assert (out / "results.json").exists()
