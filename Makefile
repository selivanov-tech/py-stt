.PHONY: help sync sync-mlx sync-eval worker test lint typecheck format eval lock clean

help:
	@echo "make sync       - uv sync (base + dev: worker, tests, mock engine)"
	@echo "make sync-mlx   - + MLX Whisper (Apple Silicon, real transcription)"
	@echo "make sync-eval  - + mlflow / pandas / boto3 (run the eval)"
	@echo "make worker     - run the model worker on :8090"
	@echo "make test       - run pytest"
	@echo "make lint       - ruff check + format check"
	@echo "make typecheck  - mypy"
	@echo "make format     - ruff format + autofix (writes)"
	@echo "make eval ARGS='--list-engines'"

sync:
	uv sync

sync-mlx:
	uv sync --extra mlx

sync-eval:
	uv sync --extra eval

worker:
	.venv/bin/uvicorn stt_worker.app:app --host 127.0.0.1 --port 8090

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

typecheck:
	.venv/bin/mypy stt_engines stt_worker stt_eval

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

eval:
	.venv/bin/python -m stt_eval $(ARGS)

lock:
	uv lock

clean:
	rm -rf .venv *.egg-info .pytest_cache
