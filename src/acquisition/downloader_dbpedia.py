from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from .downloader_common import (
    DEFAULT_USER_AGENT,
    build_remote_request,
    open_remote_request,
    stream_download_to_file,
)
from .models import SourceResource


def _first_databus_download_url(csv_text: str) -> str:
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        candidate = (row.get("file") or row.get("downloadURL") or "").strip()
        if candidate.startswith(("http://", "https://")):
            return candidate
    raise RuntimeError(
        "Unable to resolve DBpedia Databus download URL from query response."
    )


def download_dbpedia_databus_latest(
    resource: SourceResource,
    destination: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    chunk_size: int = 1024 * 1024,
) -> str:
    query_request = build_remote_request(
        resource.url, headers={"User-Agent": user_agent}
    )
    with open_remote_request(query_request) as response:
        query_csv = response.read().decode("utf-8")

    download_url = _first_databus_download_url(query_csv)
    download_request = build_remote_request(
        download_url, headers={"User-Agent": user_agent}
    )
    return stream_download_to_file(download_request, destination, chunk_size=chunk_size)
