from __future__ import annotations

import time
from collections import defaultdict

import mlflow
import pandas as pd

from stt_engines import build_engine

from .audio import resolve_audio
from .config import Settings
from .manifest import Sample, load_manifest
from .metrics import score


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def run_eval(manifest_path: str, engine_specs: list[str], settings: Settings) -> dict:
    samples = load_manifest(manifest_path)
    if not samples:
        raise ValueError(f"manifest is empty: {manifest_path}")

    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.experiment)

    summary: dict[str, dict] = {}
    with mlflow.start_run(run_name="stt-eval"):
        mlflow.log_param("manifest", str(manifest_path))
        mlflow.log_param("samples", len(samples))
        mlflow.log_param("normalize", settings.normalize)
        for spec in engine_specs:
            summary[spec] = _run_engine(spec, samples, settings)
    return summary


def _run_engine(spec: str, samples: list[Sample], settings: Settings) -> dict:
    with mlflow.start_run(run_name=spec, nested=True):
        try:
            engine = build_engine(spec)
        except Exception as exc:  # missing dep, bad model name, etc.
            mlflow.set_tag("status", "failed_to_load")
            mlflow.log_param("error", str(exc)[:500])
            print(f"  ⚠️  {spec}: could not load engine: {exc}")
            return {"error": str(exc)}

        mlflow.log_param("engine", engine.key)
        mlflow.log_param("model", engine.model)

        rows: list[dict] = []
        wer_by_lang: dict[str, list] = defaultdict(list)
        for sample in samples:
            row = _run_sample(engine, sample, settings)
            rows.append(row)
            wer_by_lang[sample.language or "auto"].append(row.get("wer"))
            print(
                f"  {spec}  {sample.id} [{sample.language or 'auto'}]  "
                f"sim={row.get('similarity_pct')}%  wer={row.get('wer')}  "
                f"{row.get('latency_s')}s"
            )

        ok = [r for r in rows if "error" not in r]
        metrics = {
            "mean_wer": _mean([r["wer"] for r in ok]),
            "mean_cer": _mean([r["cer"] for r in ok]),
            "mean_word_accuracy_pct": _mean([r["word_accuracy_pct"] for r in ok]),
            "mean_similarity_pct": _mean([r["similarity_pct"] for r in ok]),
            "mean_latency_s": _mean([r["latency_s"] for r in ok]),
            "n_ok": len(ok),
            "n_error": len(rows) - len(ok),
        }
        for lang, wers in wer_by_lang.items():
            lang_mean = _mean(wers)
            if lang_mean is not None:
                metrics[f"mean_wer_{lang}"] = lang_mean
        metrics = {k: v for k, v in metrics.items() if v is not None}

        mlflow.log_metrics(metrics)
        mlflow.log_table(data=pd.DataFrame(rows), artifact_file="results.json")
        print(
            f"  → {spec}: similarity={metrics.get('mean_similarity_pct')}%  "
            f"wer={metrics.get('mean_wer')}  "
            f"({metrics['n_ok']} ok, {metrics['n_error']} err)"
        )
        return metrics


def _run_sample(engine, sample: Sample, settings: Settings) -> dict:
    try:
        audio_path = resolve_audio(sample.audio, settings.cache_dir)
        start = time.perf_counter()
        hypothesis = engine.transcribe(audio_path, sample.language)
        latency = round(time.perf_counter() - start, 2)
    except Exception as exc:
        return {
            "id": sample.id,
            "language": sample.language or "auto",
            "error": str(exc)[:300],
        }

    sc = score(sample.reference, hypothesis, sample.language, settings.normalize)
    return {
        "id": sample.id,
        "language": sample.language or "auto",
        "reference": sample.reference,
        "hypothesis": hypothesis,
        "latency_s": latency,
        **sc.as_dict(),
    }
