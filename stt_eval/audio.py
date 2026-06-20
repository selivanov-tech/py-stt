from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse


def _cache_path(uri: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    suffix = Path(urlparse(uri).path).suffix or ".audio"
    return cache_dir / f"{digest}{suffix}"


def resolve_audio(uri: str, cache_dir: str | Path = ".cache/audio") -> str:
    """Return a local path for an audio clip. Downloads + caches s3:// and
    http(s):// URIs; local paths are returned as-is."""
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        local = Path(parsed.path if scheme == "file" else uri)
        if not local.exists():
            raise FileNotFoundError(f"audio not found: {local}")
        return str(local)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = _cache_path(uri, cache_dir)
    if dest.exists():
        return str(dest)

    if scheme == "s3":
        _download_s3(parsed.netloc, parsed.path.lstrip("/"), dest)
    elif scheme in ("http", "https"):
        _download_http(uri, dest)
    else:
        raise ValueError(f"unsupported audio scheme: {scheme!r} ({uri})")
    return str(dest)


def _download_http(url: str, dest: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def _download_s3(bucket: str, key: str, dest: Path) -> None:
    import boto3

    endpoint = os.getenv("S3_ENDPOINT_URL") or None  # S3-compatible (R2/MinIO/...); unset = AWS
    region = os.getenv("AWS_REGION", "auto")
    client = boto3.client("s3", endpoint_url=endpoint, region_name=region)
    client.download_file(bucket, key, str(dest))
