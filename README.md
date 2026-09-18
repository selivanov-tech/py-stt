# py-stt

The Python half of **stt-service**: speech-to-text model adapters behind one interface, a thin FastAPI model worker that the Go service ([go-stt](https://github.com/selivanov-tech/go-stt)) calls, and an offline eval harness that scores engines on your own recordings and logs to MLflow. Python owns the models; everything else (API, queue, job store, supervision) is Go.

[![CI](https://github.com/selivanov-tech/py-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/selivanov-tech/py-stt/actions/workflows/ci.yml)

Python **3.11** (pinned in `.python-version`: the ML wheels — mlx, numba, ctranslate2 — have no newer builds), managed with **uv**.

## Status

**Engines, worker and eval harness are built and tested; MLX Whisper and Parakeet run on real recordings.** faster-whisper, MLX Whisper and Parakeet (MLX) are real adapters, and all three return timed segments. `stt_eval.compare` runs several engines on one recording side by side. The scored eval on hand-transcribed EN/RU clips is still the next step. Five engines are registered as explicit stubs (they answer `501` from the worker and show as `stub` in `--list-engines`).

## Packages

```
stt_engines/   model adapters — shared by the worker and the eval
stt_worker/    FastAPI worker: POST /transcribe, GET /health
stt_eval/      offline eval: manifest → engines → WER/CER/similarity → MLflow; compare: one recording, many engines
tests/         pytest, no models needed
```

### `stt_engines`

One abstract base, `STTEngine`, with `key`, `default_model`, `requirements`, `languages`, `probe_module`, `is_stub` and `transcribe(audio_path, language) -> Transcript(text, language, segments)`; a `Segment` is `start`, `end` (seconds) and `text`. Heavy imports stay inside `__init__` / `transcribe`, so a missing library never breaks the whole tool. `build_engine("mlx_whisper:large-v3")` parses `key[:model]`; `catalog()` lists every engine with its status (`available`, `not installed`, `stub`, `built-in`) for the worker's `/health` and `--list-engines`.

| key | type | status | notes |
|---|---|---|---|
| `mock` | built-in | ready | echoes a fake transcript; no models, used for wiring and tests |
| `faster_whisper` | local, CPU | ready | `FASTER_WHISPER_DEVICE` / `FASTER_WHISPER_COMPUTE` (default `cpu` / `int8`), default model `large-v3` |
| `mlx_whisper` | local, Apple Silicon | ready | short names map to `mlx-community/*` repos; default `large-v3-turbo`; any HF repo accepted; one language per file (the hint, or detected from the first 30 s); `MLX_WHISPER_CONDITION_ON_PREVIOUS_TEXT=false` breaks repetition loops on long files |
| `parakeet_mlx` | local, Apple Silicon | ready | `tdt-0.6b-v3` (default; 25 European languages incl. English and Russian, detects the language itself) or `tdt-0.6b-v2` (English); files longer than `PARAKEET_CHUNK_SECONDS` (120) are chunked with `PARAKEET_OVERLAP_SECONDS` (15); `<unk>` tokens are removed from the text, and the audio gets up to 7 ms of trailing zeros so parakeet-mlx's STFT never reads past its buffer (stale `inf` there once turned whole clips into `<unk>` runs) |
| `whisper_cpp` | local | stub | |
| `apple_speech` | local, macOS | stub | needs a small Swift helper |
| `deepgram` · `assemblyai` · `elevenlabs` | cloud | stub | |

The same engine code serves the worker and the eval, so the model that wins the eval is the one the worker runs.

### `stt_worker`

- `GET /health` → registered engines, which ones are loaded, and `catalog` with each engine's status.
- `POST /transcribe` with `{engine, model?, language?, audio_path | audio_url}` → `{text, engine, model, latency_ms, detected_language, audio_duration_s, segments}`. `audio_duration_s` comes from `ffprobe` (`null` without it); `latency_ms` counts inference only, not the wait for the lock.
- Engines are **cached by spec**: a model loads once and is reused. **One lock covers loading and inference**: local models share one GPU and MLX is not safe to drive from several threads, so concurrent requests run one after another.
- `audio_url` (http/https) is downloaded to a temp file and deleted after the call. An **SSRF guard** resolves the host first and rejects loopback, private, link-local, reserved and multicast addresses (cloud metadata included).
- Error mapping: engine load failure → `400`, stub engine → `501`, transcription failure → `500`.
- `audio_base64` (inline audio, up to 32 MiB decoded, the same cap as go-stt's request) is written to a temp file and deleted after the call; ffmpeg detects the format from the content. Invalid base64 → `400`, too large → `413`. This is how a live voice bot (Pipecat) sends each phrase.

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

### `stt_eval.compare`

```bash
make compare AUDIO=~/Downloads/clip.m4a ENGINES=mlx_whisper,mlx_whisper:large-v3,parakeet_mlx STT_LANG=ru
.venv/bin/python -m stt_eval.compare clip.m4a --engines mlx_whisper,parakeet_mlx   # in process, no service
```

- Runs each engine on one recording: through go-stt's job queue with `--via <url>` (what `make compare` does, default `http://127.0.0.1:8080`), or in this process.
- Writes `runs/compare/<audio>-<time>/` (git-ignored: transcripts of private audio): `<engine>.txt`, `<engine>.timed.txt` with one `[mm:ss.s] text` line per segment, `results.json`, and `summary.md` — words, inference and wall time, speed as × realtime, detected language, and WER / CER / similarity for every pair of engines.
- There is no hand-written reference, so the pair numbers measure agreement between engines, not accuracy. Use them to find where engines disagree, then listen to those places.

## Setup and run

```bash
uv sync                # base + dev: worker, tests, mock engine
make sync-local        # + MLX Whisper and Parakeet (Apple Silicon)
uv sync --extra faster-whisper
uv sync --extra eval   # + mlflow / pandas / boto3

make worker            # worker on 127.0.0.1:8090 (WORKER_PORT=… to change)
curl localhost:8090/health
curl -X POST localhost:8090/transcribe -H 'content-type: application/json' \
  -d '{"engine":"mock","audio_path":"x.wav","language":"en"}'
```

`uv sync` keeps only the extras you name, so combine them in one call (`make sync-mlx` alone removes Parakeet). With go-stt cloned next to this repo, `make up` there runs this worker in the background, and `make dev` in the foreground.

## Parallel sessions

`.wt.conf` prepares worktrees made by `wt` ([agent-skills](https://github.com/selivanov-tech/agent-skills)): a new worktree gets its own `.venv` through `uv sync --frozen --extra mlx --extra parakeet` (the uv cache makes it fast) and a copy of `.env`. `.venv` is not cloned from the main checkout, because its scripts hold the main checkout's absolute paths. A worker from a worktree needs its own port: `WORKER_PORT=8091 make worker`, or go-stt's `PY_STT=<worktree> WORKER_PORT=8091 ADDR=127.0.0.1:8081 make up`.

## Test and lint

```bash
make test        # pytest
make lint        # ruff check + format check
make typecheck   # mypy (check_untyped_defs)
```

Tests cover the engine registry, `build_engine` and `catalog` statuses (Parakeet is real, cloud engines are stubs), Parakeet `<unk>` cleanup and the STFT tail guard (the MLX check runs only where MLX is installed), the worker (health with catalog, mock transcribe with segments and duration, stub → 501, SSRF guard), `compare` in process (transcript files, agreement table, a failing engine), manifest parsing, metrics and normalisation. CI ([`ci.yml`](.github/workflows/ci.yml)) runs ruff, mypy and pytest with `uv sync --frozen` on every push and pull request.

## Roadmap

1. Record ~15–30 EN/RU clips, transcribe by hand, run the eval, pick the default model.
2. Implement whisper.cpp if the eval justifies it.
3. Apple Speech (`SpeechAnalyzer`, macOS 26) through a small Swift helper, as the built-in local baseline.
4. Cloud engines move to Go (production profile); the Python worker stays the local, private path.

## Project

Part of **stt-service**, split into two repos cloned side by side:

| Repo | Role |
|---|---|
| [go-stt](https://github.com/selivanov-tech/go-stt) | Go service: HTTP API, queue, workers, job store, supervision |
| **py-stt** (this) | Python: STT engines, FastAPI model worker, eval harness |
