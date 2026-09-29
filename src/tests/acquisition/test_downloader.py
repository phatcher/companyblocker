from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from acquisition import (
    downloader,
    downloader_common,
    downloader_dbpedia,
    downloader_gleif,
)
from acquisition.models import SourceResource


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _BytesResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._consumed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        if self._consumed:
            return b""
        self._consumed = True
        return self._payload


def _resource(**overrides: Any) -> SourceResource:
    values: dict[str, Any] = {
        "name": "sample",
        "url": "https://example.test/source",
        "file_format": "json",
    }
    values.update(overrides)
    return SourceResource(**values)


def test_download_resource_writes_chunks_and_returns_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "download" / "source.bin"
    resource = _resource(url="https://example.test/blob", file_format="zip")
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["user_agent"] = request.headers.get("User-agent")
        return _ChunkedResponse([b"abc", b"def", b""])

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    digest = downloader.download_resource(
        resource, destination, user_agent="agent/1.0", chunk_size=3
    )

    assert destination.read_bytes() == b"abcdef"
    assert digest == hashlib.sha256(b"abcdef").hexdigest()
    assert seen == {"url": "https://example.test/blob", "user_agent": "agent/1.0"}


def test_sha256_file_hashes_existing_file(tmp_path: Path):
    path = tmp_path / "payload.txt"
    path.write_bytes(b"alpha-beta")

    assert (
        downloader.sha256_file(path, chunk_size=4)
        == hashlib.sha256(b"alpha-beta").hexdigest()
    )


def test_resolve_api_headers_merges_headers_and_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEY", "secret-token")
    monkeypatch.setenv("REQUIRED_ENV", "present")
    resource = _resource(
        request_headers=(("X-Test", "1"),),
        required_env_vars=("REQUIRED_ENV",),
        auth_header_name="Authorization",
        auth_header_env="API_KEY",
        auth_header_prefix="Bearer",
    )

    headers = downloader._resolve_api_headers(resource, user_agent="custom-agent")

    assert headers == {
        "User-Agent": "custom-agent",
        "Accept": "application/json",
        "X-Test": "1",
        "Authorization": "Bearer secret-token",
    }


def test_resolve_api_headers_raises_for_missing_required_or_auth_env(
    monkeypatch: pytest.MonkeyPatch,
):
    resource_missing_required = _resource(required_env_vars=("MISSING_ENV",))
    with pytest.raises(
        RuntimeError, match="Missing required environment variable 'MISSING_ENV'"
    ):
        downloader._resolve_api_headers(resource_missing_required, user_agent="agent")

    resource_missing_auth = _resource(
        auth_header_name="X-Auth", auth_header_env="AUTH_TOKEN"
    )
    with pytest.raises(
        RuntimeError, match="Missing required environment variable 'AUTH_TOKEN'"
    ):
        downloader._resolve_api_headers(resource_missing_auth, user_agent="agent")


@pytest.mark.parametrize(
    "payload,expected",
    [
        ([{"id": 1}, "skip", {"id": 2}], [{"id": 1}, {"id": 2}]),
        ({"results": [{"id": 3}, 4]}, [{"id": 3}]),
        ({"companies": [{"id": 5}]}, [{"id": 5}]),
        ({"id": 9, "name": "single"}, [{"id": 9, "name": "single"}]),
    ],
)
def test_extract_records_from_payload_supported_shapes(payload, expected):
    assert downloader._extract_records_from_payload(payload) == expected


def test_extract_records_from_payload_rejects_unsupported_shape():
    with pytest.raises(ValueError, match="supported JSON object or array"):
        downloader._extract_records_from_payload("not-json-records")


def test_download_api_json_snapshot_writes_jsonl_and_returns_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "snapshots" / "records.jsonl"
    resource = _resource(
        url="https://example.test/api",
        request_headers=(("X-Trace", "yes"),),
    )
    observed: dict[str, object] = {}
    payload = {"results": [{"b": 2, "a": 1}, {"name": "Acme"}]}

    def fake_urlopen(request, timeout=None):
        observed["url"] = request.full_url
        observed["headers"] = dict(request.header_items())
        return _BytesResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    digest = downloader.download_api_json_snapshot(
        resource, destination, user_agent="api-agent"
    )

    expected_lines = b'{"a": 1, "b": 2}\n{"name": "Acme"}\n'
    assert destination.read_bytes() == expected_lines
    assert digest == hashlib.sha256(expected_lines).hexdigest()
    observed_headers = observed["headers"]
    assert isinstance(observed_headers, dict)
    assert observed["url"] == "https://example.test/api"
    assert observed_headers["User-agent"] == "api-agent"
    assert observed_headers["X-trace"] == "yes"


@pytest.mark.parametrize(
    "csv_text,expected",
    [
        (
            "file\nhttps://databus.dbpedia.org/a.ttl.bz2\n",
            "https://databus.dbpedia.org/a.ttl.bz2",
        ),
        (
            "downloadURL\nhttps://databus.dbpedia.org/b.ttl.bz2\n",
            "https://databus.dbpedia.org/b.ttl.bz2",
        ),
    ],
)
def test_first_databus_download_url(csv_text: str, expected: str):
    assert downloader_dbpedia._first_databus_download_url(csv_text) == expected


def test_first_databus_download_url_raises_on_missing_http_url():
    with pytest.raises(
        RuntimeError, match="Unable to resolve DBpedia Databus download URL"
    ):
        downloader_dbpedia._first_databus_download_url(
            "file\nftp://example.test/file.ttl\n"
        )


def test_download_dbpedia_databus_latest_resolves_query_then_downloads_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "dbpedia" / "instance-types.ttl.bz2"
    resource = _resource(
        url="https://databus.dbpedia.org/repo/sparql?format=text%2Fcsv&query=SELECT",
        file_format="ttl",
    )
    seen_urls: list[str] = []

    def fake_urlopen(request, timeout=None):
        seen_urls.append(request.full_url)
        if request.full_url.startswith("https://databus.dbpedia.org/repo/sparql"):
            return _BytesResponse(
                b"file\nhttps://databus.dbpedia.org/files/instance-types.ttl.bz2\n"
            )
        if (
            request.full_url
            == "https://databus.dbpedia.org/files/instance-types.ttl.bz2"
        ):
            return _ChunkedResponse([b"ttl", b"-data", b""])
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    digest = downloader_dbpedia.download_dbpedia_databus_latest(resource, destination)

    assert seen_urls == [
        "https://databus.dbpedia.org/repo/sparql?format=text%2Fcsv&query=SELECT",
        "https://databus.dbpedia.org/files/instance-types.ttl.bz2",
    ]
    assert destination.read_bytes() == b"ttl-data"
    assert digest == hashlib.sha256(b"ttl-data").hexdigest()


@pytest.mark.parametrize(
    "item,expected",
    [
        ({"file": "http://example.test/a.zip"}, "http://example.test/a.zip"),
        (
            {"attributes": {"download_url": "https://example.test/b.zip"}},
            "https://example.test/b.zip",
        ),
        ({"url": "ftp://example.test/not-http"}, None),
        ({"attributes": {"url": "mailto:test@example.com"}}, None),
    ],
)
def test_extract_gleif_download_url_cases(item, expected):
    assert downloader_gleif._extract_gleif_download_url(item) == expected


def test_download_gleif_latest_concatenated_downloads_latest_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "gleif" / "latest.zip"
    resource = _resource(url="https://example.test/gleif-meta")
    seen_urls: list[str] = []
    metadata_payload = {
        "data": [
            {
                "id": "7",
                "content_date": "2026-01-01",
                "attributes": {"download_url": "https://example.test/old.zip"},
            },
            {
                "id": "12",
                "content_date": "2026-06-01",
                "attributes": {"download_url": "https://example.test/new.zip"},
            },
            {
                "id": "bad",
                "content_date": "2026-07-01",
                "attributes": {"download_url": "https://example.test/ignored.zip"},
            },
        ]
    }

    def fake_urlopen(request, timeout=None):
        seen_urls.append(request.full_url)
        if request.full_url == "https://example.test/gleif-meta":
            return _BytesResponse(json.dumps(metadata_payload).encode("utf-8"))
        if request.full_url == "https://example.test/new.zip":
            return _ChunkedResponse([b"zip", b"-bytes", b""])
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    digest = downloader_gleif.download_gleif_latest_concatenated(
        resource,
        destination,
        user_agent="gleif-agent",
    )

    assert seen_urls == [
        "https://example.test/gleif-meta",
        "https://example.test/new.zip",
    ]
    assert destination.read_bytes() == b"zip-bytes"
    assert digest == hashlib.sha256(b"zip-bytes").hexdigest()


def test_download_gleif_latest_concatenated_raises_for_missing_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    resource = _resource(url="https://example.test/gleif-meta")

    def fake_urlopen(_request, timeout=None):
        return _BytesResponse(json.dumps({"data": []}).encode("utf-8"))

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="did not include any data entries"):
        downloader_gleif.download_gleif_latest_concatenated(
            resource, tmp_path / "latest.zip"
        )


def test_download_gleif_latest_concatenated_raises_when_download_url_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    resource = _resource(url="https://example.test/gleif-meta")

    def fake_urlopen(_request, timeout=None):
        return _BytesResponse(
            json.dumps(
                {"data": [{"id": "1", "attributes": {"url": "ftp://bad"}}]}
            ).encode("utf-8")
        )

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    with pytest.raises(
        RuntimeError, match="Unable to resolve GLEIF concatenated download URL"
    ):
        downloader_gleif.download_gleif_latest_concatenated(
            resource, tmp_path / "latest.zip"
        )
