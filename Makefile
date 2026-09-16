.PHONY: help sync sync-mlx sync-local sync-eval worker test lint typecheck format eval compare lock clean

WORKER_PORT ?= 8090
STT_URL     ?= http://127.0.0.1:8080
ENGINES     ?= mlx_whisper,parakeet_mlx
STT_LANG    ?=

help:
	@echo "make sync       - uv sync (base + dev: worker, tests, mock engine)"
	@echo "make sync-mlx   - + MLX Whisper only (Apple Silicon)"
	@echo "make sync-local - + MLX Whisper and Parakeet: every local model on Apple Silicon"
	@echo "make sync-eval  - + mlflow / pandas / boto3 (run the eval)"
	@echo "make worker     - run the model worker on 127.0.0.1:$(WORKER_PORT)"
	@echo "make test       - run pytest"
	@echo "make lint       - ruff check + format check"
	@echo "make typecheck  - mypy"
	@echo "make format     - ruff format + autofix (writes)"
	@echo "make eval ARGS='--list-engines'"
	@echo "make compare AUDIO=<file> [ENGINES=a,b] [STT_LANG=ru] - engines side by side via go-stt"
	@echo ""
	@echo "uv sync keeps only the extras you name: 'make sync-mlx' removes Parakeet."

sync:
	uv sync

sync-mlx:
	uv sync --extra mlx

sync-local:
	uv sync --extra mlx --extra parakeet

sync-eval:
	uv sync --extra eval

worker:
	.venv/bin/python -m uvicorn stt_worker.app:app --host 127.0.0.1 --port $(WORKER_PORT)

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/python -m ruff check .
	.venv/bin/python -m ruff format --check .

typecheck:
	.venv/bin/python -m mypy stt_engines stt_worker stt_eval

format:
	.venv/bin/python -m ruff format .
	.venv/bin/python -m ruff check --fix .

eval:
	.venv/bin/python -m stt_eval $(ARGS)

compare:
	@test -n "$(AUDIO)" || { echo "usage: make compare AUDIO=<file> [ENGINES=a,b] [STT_LANG=ru]"; exit 2; }
	.venv/bin/python -m stt_eval.compare "$(AUDIO)" --engines "$(ENGINES)" --via $(STT_URL) \
		$(if $(STT_LANG),--language $(STT_LANG))

lock:
	uv lock

clean:
	rm -rf .venv *.egg-info .pytest_cache
