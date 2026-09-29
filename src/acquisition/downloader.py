from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .downloader_common import (
    DEFAULT_USER_AGENT,
    build_remote_request,
    open_remote_request,
    resolve_api_headers,
    stream_download_to_file,
)
from .models import SourceResource


def download_resource(
    resource: SourceResource,
    destination: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    chunk_size: int = 1024 * 1024,
) -> str:
    request = build_remote_request(resource.url, headers={"User-Agent": user_agent})
    return stream_download_to_file(request, destination, chunk_size=chunk_size)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_api_headers(
    resource: SourceResource, *, user_agent: str
) -> dict[str, str]:
    return resolve_api_headers(resource, user_agent=user_agent)


def _extract_records_from_payload(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("results", "data", "items", "companies", "hits"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        return [payload]

    raise ValueError("API response payload is not a supported JSON object or array.")


def download_api_json_snapshot(
    resource: SourceResource,
    destination: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = _resolve_api_headers(resource, user_agent=user_agent)
    request = build_remote_request(resource.url, headers=headers)

    with open_remote_request(request) as response:
        raw_bytes = response.read()

    payload = json.loads(raw_bytes.decode("utf-8"))
    records = _extract_records_from_payload(payload)

    digest = hashlib.sha256()
    with destination.open("wb") as handle:
        for record in records:
            line = (
                json.dumps(record, ensure_ascii=True, sort_keys=True).encode("utf-8")
                + b"\n"
            )
            handle.write(line)
            digest.update(line)

    return digest.hexdigest()


__all__ = [
    "DEFAULT_USER_AGENT",
    "download_resource",
    "download_api_json_snapshot",
    "download_dbpedia_databus_latest",
    "download_gleif_latest_concatenated",
    "sha256_file",
]


# Backward-compatible export while GLEIF-specific logic lives in its own module.
from .downloader_dbpedia import download_dbpedia_databus_latest
from .downloader_gleif import download_gleif_latest_concatenated
