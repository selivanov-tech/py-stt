# py-stt

> **Status: early draft (WIP).** Code lands incrementally — this README is the plan.

Python side of the **stt-service** project. Three packages, one **uv**-managed env:

- **`stt_engines/`** — STT model adapters (mock, faster-whisper, MLX Whisper, +
  stubs). Shared by the worker and the eval.
- **`stt_worker/`** — FastAPI model worker: `POST /transcribe` (audio → text).
  The Go service ([go-stt](https://github.com/selivanov-tech/go-stt)) calls this.
- **`stt_eval/`** — offline eval harness: score engines on your recordings, log
  to MLflow.

Python is pinned to **3.11** (`.python-version`) because the ML wheels (mlx,
numba, ctranslate2) have no 3.14 build yet. uv fetches 3.11 automatically.

## Setup

```bash
uv sync                # base + dev (worker, tests, mock engine)
uv sync --extra mlx    # + MLX Whisper (Apple Silicon, real transcription)
uv sync --extra eval   # + mlflow / pandas / boto3 (run the eval)
```

## Run the worker

```bash
make worker            # .venv/bin/uvicorn stt_worker.app:app --port 8090
curl localhost:8090/health
curl -X POST localhost:8090/transcribe \
  -H 'content-type: application/json' \
  -d '{"engine":"mock","audio_path":"x.wav","language":"en"}'
```

`mock` needs no models. Real engines (`faster_whisper`, `mlx_whisper`) load on
first use and stay cached.

## Run the eval

```bash
uv sync --extra eval
.venv/bin/python -m stt_eval --list-engines
.venv/bin/python -m stt_eval --manifest data/manifest.example.jsonl --engines faster_whisper,mlx_whisper
.venv/bin/mlflow ui
```

## Test

```bash
make test              # .venv/bin/pytest
make lint              # ruff check + format check
make typecheck         # mypy
```

## Project

Part of **stt-service**, split into two repos — clone them side by side:

| Repo | Role |
|---|---|
| [go-stt](https://github.com/selivanov-tech/go-stt) | Go service: HTTP API, queue, workers, job store, supervision |
| **py-stt** (this) | Python: STT engines, FastAPI model worker, eval harness |
