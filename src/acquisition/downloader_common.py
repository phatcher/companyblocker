from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .models import SourceResource

DEFAULT_USER_AGENT = "blocking-acquisition/0.1"


def validate_remote_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"Unsupported URL scheme '{parsed.scheme}' for remote download."
        )
    if not parsed.netloc:
        raise ValueError("Remote download URL must include a host.")
    return url


def build_remote_request(url: str, *, headers: dict[str, str]) -> Request:
    return Request(validate_remote_url(url), headers=headers)


def open_remote_request(request: Request, *, timeout_seconds: int = 60):
    # URL scheme and host are validated in build_remote_request/validate_remote_url.
    return urlopen(request, timeout=timeout_seconds)  # nosec B310


def stream_download_to_file(
    request: Request, destination: Path, *, chunk_size: int = 1024 * 1024
) -> str:
    """Stream `request`'s response body to `destination`, returning its sha256 hex digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open_remote_request(request) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def resolve_api_headers(resource: SourceResource, *, user_agent: str) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    for name, value in resource.request_headers:
        headers[name] = value

    for env_name in resource.required_env_vars:
        env_value = os.getenv(env_name)
        if not env_value:
            raise RuntimeError(
                f"Missing required environment variable '{env_name}' for source '{resource.name}'."
            )

    if resource.auth_header_name and resource.auth_header_env:
        token = os.getenv(resource.auth_header_env)
        if not token:
            raise RuntimeError(
                f"Missing required environment variable '{resource.auth_header_env}' for source '{resource.name}'."
            )
        if resource.auth_header_prefix:
            headers[resource.auth_header_name] = (
                f"{resource.auth_header_prefix} {token}"
            )
        else:
            headers[resource.auth_header_name] = token

    return headers
