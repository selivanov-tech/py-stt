from __future__ import annotations

import argparse
import importlib.util
import sys

from stt_engines import ENGINES

from .config import Settings, load_dotenv_if_present


def _list_engines() -> None:
    print("Engines  (key — languages — status — install):")
    for key, cls in ENGINES.items():
        probe = cls.probe_module
        if probe is None:
            status = "built-in" if key == "mock" else "stub"
        else:
            status = "available" if importlib.util.find_spec(probe) else "not installed"
        print(f"  {key:15} {cls.languages:14} {status:14} {cls.requirements}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stt_eval", description="STT evaluation harness (logs to MLflow)"
    )
    parser.add_argument("--manifest", help="dataset manifest path (.jsonl / .json)")
    parser.add_argument("--engines", help="comma list, e.g. faster_whisper,mlx_whisper:large-v3")
    parser.add_argument("--mlflow-uri", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv_if_present()

    if args.list_engines:
        _list_engines()
        return 0

    if not args.manifest or not args.engines:
        parser.error("--manifest and --engines are required (or use --list-engines)")

    settings = Settings.from_env()
    if args.mlflow_uri:
        settings.mlflow_uri = args.mlflow_uri
    if args.experiment:
        settings.experiment = args.experiment
    if args.cache_dir:
        settings.cache_dir = args.cache_dir
    if args.no_normalize:
        settings.normalize = False

    from .runner import run_eval  # lazy: pulls in mlflow only when actually running

    specs = [s.strip() for s in args.engines.split(",") if s.strip()]
    summary = run_eval(args.manifest, specs, settings)

    print("\nSummary:")
    for spec, m in summary.items():
        if "error" in m:
            print(f"  {spec}: ERROR — {m['error']}")
        else:
            print(f"  {spec}: similarity={m.get('mean_similarity_pct')}%  wer={m.get('mean_wer')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
