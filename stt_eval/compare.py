"""Run several engines on one recording and put their transcripts side by side.

There is no hand-written reference here, so the numbers show how close engines
are to each other, not how accurate they are: two engines can share a mistake.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stt_engines.media import audio_duration_s

from .metrics import score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stt_eval.compare",
        description="Transcribe one recording with several engines and compare the results",
    )
    parser.add_argument("audio", help="local audio file (any format ffmpeg reads)")
    parser.add_argument(
        "--engines", required=True, help="comma list, e.g. mlx_whisper,parakeet_mlx:tdt-0.6b-v3"
    )
    parser.add_argument("--language", default=None, help="language hint, e.g. ru")
    parser.add_argument(
        "--via", default=None, help="go-stt base URL; without it engines run in this process"
    )
    parser.add_argument("--out", default=None, help="output folder (default: runs/compare/...)")
    args = parser.parse_args(argv)

    audio = Path(args.audio).expanduser().resolve()
    if not audio.is_file():
        parser.error(f"audio not found: {audio}")
    specs = [s.strip() for s in args.engines.split(",") if s.strip()]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs/compare") / f"{_slug(audio.stem)}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for spec in specs:
        print(f"-> {spec}", flush=True)
        started = time.perf_counter()
        try:
            if args.via:
                res = _run_via_service(args.via.rstrip("/"), spec, str(audio), args.language)
            else:
                res = _run_in_process(spec, str(audio), args.language)
        except Exception as exc:
            print(f"   failed: {exc}", flush=True)
            results.append({"spec": spec, "error": str(exc)})
            continue
        res["spec"] = spec
        res["wall_ms"] = int((time.perf_counter() - started) * 1000)
        _write_transcript(out_dir, res)
        words = len(res["text"].split())
        print(f"   done in {res['wall_ms'] / 1000:.1f}s, {words} words", flush=True)
        results.append(res)

    summary = render_summary(audio, audio_duration_s(str(audio)), args.language, results)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{summary}\nsaved to {out_dir}")
    return 1 if any("error" in r for r in results) else 0


def render_summary(
    audio: Path, duration_s: float | None, language: str | None, results: list[dict]
) -> str:
    ok = [r for r in results if "error" not in r]
    lines = [
        f"# STT compare — {audio.name}",
        "",
        f"- audio: `{audio}`" + (f", {duration_s:.0f} s" if duration_s else ""),
        f"- language hint: {language or 'none'}",
        "- no reference transcript: agreement shows how close engines are, not accuracy",
        "",
        "| engine | model | words | inference s | wall s | x realtime | detected language |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        infer_s = r.get("latency_ms", 0) / 1000
        speed = f"{duration_s / infer_s:.1f}" if duration_s and infer_s else "?"
        lines.append(
            f"| {r['spec']} | {r.get('model') or ''} | {len(r['text'].split())} "
            f"| {infer_s:.1f} | {r['wall_ms'] / 1000:.1f} | {speed} "
            f"| {r.get('detected_language') or '?'} |"
        )

    if len(ok) > 1:
        lines += [
            "",
            "## Agreement",
            "",
            "| reference | compared | WER | CER | similarity % |",
            "|---|---|---|---|---|",
        ]
        for a, b in itertools.combinations(ok, 2):
            s = score(a["text"], b["text"], language)
            lines.append(
                f"| {a['spec']} | {b['spec']} "
                f"| {s.wer:.3f} | {s.cer:.3f} | {s.similarity_pct:.1f} |"
            )

    failed = [r for r in results if "error" in r]
    if failed:
        lines += ["", "## Failed", ""] + [f"- {r['spec']}: {r['error']}" for r in failed]
    return "\n".join(lines) + "\n"


def _run_in_process(spec: str, audio: str, language: str | None) -> dict:
    from stt_engines import build_engine

    engine = build_engine(spec)
    started = time.perf_counter()
    transcript = engine.transcribe(audio, language)
    return {
        "engine": engine.key,
        "model": engine.model,
        "text": transcript.text,
        "detected_language": transcript.language,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "segments": [asdict(s) for s in transcript.segments],
    }


def _run_via_service(base_url: str, spec: str, audio: str, language: str | None) -> dict:
    import requests

    key, _, model = spec.partition(":")
    body = {"audio_path": audio, "engine": f"python:{key}", "model": model, "language": language}
    resp = requests.post(
        f"{base_url}/v1/jobs", json={k: v for k, v in body.items() if v}, timeout=30
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    while True:
        time.sleep(1)
        job = requests.get(f"{base_url}/v1/jobs/{job_id}", timeout=30).json()
        if job["status"] == "done":
            return job
        if job["status"] == "failed":
            raise RuntimeError(job.get("error") or "job failed")


def _write_transcript(out_dir: Path, res: dict) -> None:
    name = _slug(res["spec"])
    (out_dir / f"{name}.txt").write_text(res["text"] + "\n", encoding="utf-8")
    segments = res.get("segments") or []
    if segments:
        lines = [f"[{_clock(s['start'])}] {s['text']}" for s in segments]
        (out_dir / f"{name}.timed.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("_") or "audio"


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


if __name__ == "__main__":
    sys.exit(main())
