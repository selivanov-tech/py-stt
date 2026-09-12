# py-stt

The Python half of **stt-service**: speech-to-text model adapters behind one interface, a thin FastAPI model worker that the Go service ([go-stt](https://github.com/selivanov-tech/go-stt)) calls, and an offline eval harness that scores engines on your own recordings and logs to MLflow. Python owns the models; everything else (API, queue, job store, supervision) is Go.

[![CI](https://github.com/selivanov-tech/py-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/selivanov-tech/py-stt/actions/workflows/ci.yml)

Python **3.11** (pinned in `.python-version`: the ML wheels — mlx, numba, ctranslate2 — have no newer builds), managed with **uv**.

## Status

**Step 1 done: engines, worker and eval harness are built and tested.** The mock engine runs end to end with go-stt; faster-whisper and MLX Whisper are real adapters. The eval has not yet been run on real EN/RU clips — that is the next step. Six engines are registered as explicit stubs (they answer `501` from the worker and show as `stub` in `--list-engines`).

## Packages

```
stt_engines/   model adapters — shared by the worker and the eval
stt_worker/    FastAPI worker: POST /transcribe, GET /health
stt_eval/      offline eval: manifest → engines → WER/CER/similarity → MLflow
tests/         pytest, no models needed
```

### `stt_engines`

One abstract base, `STTEngine`, with `key`, `default_model`, `requirements`, `languages`, `probe_module` and `transcribe(audio_path, language) -> Transcript(text, language)`. Heavy imports stay inside `__init__` / `transcribe`, so a missing library never breaks the whole tool. `build_engine("mlx_whisper:large-v3")` parses `key[:model]`.

| key | type | status | notes |
|---|---|---|---|
| `mock` | built-in | ready | echoes a fake transcript; no models, used for wiring and tests |
| `faster_whisper` | local, CPU | ready | `FASTER_WHISPER_DEVICE` / `FASTER_WHISPER_COMPUTE` (default `cpu` / `int8`), default model `large-v3` |
| `mlx_whisper` | local, Apple Silicon | ready | short names map to `mlx-community/*` repos; default `large-v3-turbo`; any HF repo accepted |
| `whisper_cpp` | local | stub | |
| `apple_speech` | local, macOS | stub | needs a small Swift helper |
| `parakeet_mlx` | local, Apple Silicon | stub | English-focused |
| `deepgram` · `assemblyai` · `elevenlabs` | cloud | stub | |

The same engine code serves the worker and the eval, so the model that wins the eval is the one the worker runs.

### `stt_worker`

- `GET /health` → registered engines and which ones are loaded.
- `POST /transcribe` with `{engine, model?, language?, audio_path | audio_url}` → `{text, engine, model, latency_ms, detected_language}`.
- Engines are **cached by spec** under a double-checked lock: a model loads once and is reused; two concurrent first requests do not load it twice.
- `audio_url` (http/https) is downloaded to a temp file and deleted after the call. An **SSRF guard** resolves the host first and rejects loopback, private, link-local, reserved and multicast addresses (cloud metadata included).
- Error mapping: engine load failure → `400`, stub engine → `501`, transcription failure → `500`.
- `audio_base64` is accepted by the schema but not materialised yet; the Go service forwards it, the worker ignores it.

### `stt_eval`

```bash
uv sync --extra eval
.venv/bin/python -m stt_eval --list-engines
.venv/bin/python -m stt_eval --manifest data/manifest.example.jsonl --engines faster_whisper,mlx_whisper:large-v3
.venv/bin/mlflow ui
```

- **Manifest**: `.jsonl` or a `.json` array of `{id, audio, reference, language?, notes?}`; `audio` is a local path, `https://` URL or `s3://` key (S3-compatible endpoints via `S3_ENDPOINT_URL`). Remote clips are cached under `.cache/audio` by URI hash.
- **Scoring** per sample: WER and CER (jiwer), word accuracy, and character similarity (rapidfuzz), after normalisation — NFKC, casefold, punctuation stripped, and `ё → е` for Russian.
- **MLflow**: one parent run per eval, one nested run per engine; params (engine, model, manifest, sample count), mean metrics, **mean WER per language**, and the full per-sample table as an artifact. Engines that fail to load are recorded as `failed_to_load`, not crashed over.
- Settings from env / `.env`: `MLFLOW_TRACKING_URI` (default `./mlruns`), `MLFLOW_EXPERIMENT`, `AUDIO_CACHE_DIR`, S3 credentials.

## Setup and run

```bash
uv sync                # base + dev: worker, tests, mock engine
uv sync --extra mlx    # + MLX Whisper (Apple Silicon)
uv sync --extra faster-whisper
uv sync --extra eval   # + mlflow / pandas / boto3

make worker            # uvicorn stt_worker.app:app on :8090
curl localhost:8090/health
curl -X POST localhost:8090/transcribe -H 'content-type: application/json' \
  -d '{"engine":"mock","audio_path":"x.wav","language":"en"}'
```

With go-stt cloned next to this repo, `make dev` there starts this worker for you.

## Test and lint

```bash
make test        # pytest
make lint        # ruff check + format check
make typecheck   # mypy (check_untyped_defs)
```

Tests cover the engine registry and `build_engine`, the worker (health, mock transcribe, stub → 501, SSRF guard), manifest parsing, metrics and normalisation. CI ([`ci.yml`](.github/workflows/ci.yml)) runs ruff, mypy and pytest with `uv sync --frozen` on every push and pull request.

## Roadmap

1. Record ~15–30 EN/RU clips, transcribe by hand, run the eval, pick the default model.
2. Implement the stubs that the eval justifies (parakeet-mlx, whisper.cpp) and `audio_base64` in the worker.
3. Cloud engines move to Go (production profile); the Python worker stays the local, private path.

## Project

Part of **stt-service**, split into two repos cloned side by side:

| Repo | Role |
|---|---|
| [go-stt](https://github.com/selivanov-tech/go-stt) | Go service: HTTP API, queue, workers, job store, supervision |
| **py-stt** (this) | Python: STT engines, FastAPI model worker, eval harness |
