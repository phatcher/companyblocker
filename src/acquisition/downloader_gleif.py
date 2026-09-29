from __future__ import annotations

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


def _extract_gleif_download_url(item: dict[str, object]) -> str | None:
    for key in ("file", "download_url", "url"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value

    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        for key in ("file", "download_url", "url"):
            value = attributes.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value

    return None


def download_gleif_latest_concatenated(
    resource: SourceResource,
    destination: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    chunk_size: int = 1024 * 1024,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = resolve_api_headers(resource, user_agent=user_agent)
    request = build_remote_request(resource.url, headers=headers)

    with open_remote_request(request) as response:
        raw_bytes = response.read()

    payload = json.loads(raw_bytes.decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    entries: list[dict[str, object]] = []
    if isinstance(data, list):
        entries = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        entries = [data]

    if not entries:
        raise RuntimeError(
            "GLEIF concatenated metadata response did not include any data entries."
        )

    def _entry_sort_key(entry: dict[str, object]) -> tuple[int, str]:
        raw_id = entry.get("id")
        if isinstance(raw_id, (int, float, str)):
            try:
                numeric_id = int(raw_id)
            except (TypeError, ValueError):
                numeric_id = -1
        else:
            numeric_id = -1
        content_date = entry.get("content_date")
        return numeric_id, str(content_date) if content_date is not None else ""

    latest = max(entries, key=_entry_sort_key)
    download_url = _extract_gleif_download_url(latest)
    if not download_url:
        raise RuntimeError(
            "Unable to resolve GLEIF concatenated download URL from metadata response."
        )

    download_request = build_remote_request(
        download_url, headers={"User-Agent": user_agent}
    )
    return stream_download_to_file(download_request, destination, chunk_size=chunk_size)
