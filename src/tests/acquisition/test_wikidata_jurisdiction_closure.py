from __future__ import annotations

import json
from urllib.parse import parse_qs
from urllib.request import Request

import pytest

from acquisition import downloader_common, wikidata_jurisdiction_closure


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


def _sparql_json_response(bindings: list[dict[str, str]]) -> bytes:
    payload = {
        "results": {
            "bindings": [
                {
                    "qid": {
                        "type": "uri",
                        "value": f"http://www.wikidata.org/entity/{qid}",
                    },
                    "resolvedCountry": {
                        "type": "uri",
                        "value": f"http://www.wikidata.org/entity/{country}",
                    },
                }
                for qid, country in ((row["qid"], row["country"]) for row in bindings)
            ]
        }
    }
    return json.dumps(payload).encode("utf-8")


def test_build_qid_country_closure_query_batches_all_qids_into_one_values_clause():
    query = wikidata_jurisdiction_closure.build_qid_country_closure_query(
        ["q30", "Q145"]
    )

    assert query.count("VALUES ?qid") == 1
    assert "wd:Q30" in query
    assert "wd:Q145" in query
    assert "wdt:P131*/wdt:P17" in query


def test_fetch_qid_country_closure_parses_response_and_uppercases_qids(
    monkeypatch: pytest.MonkeyPatch,
):
    seen_requests: list[Request] = []

    def fake_urlopen(request, timeout=None):
        seen_requests.append(request)
        return _BytesResponse(
            _sparql_json_response(
                [
                    {"qid": "Q99", "country": "Q30"},
                    {"qid": "Q100", "country": "Q145"},
                ]
            )
        )

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    resolved = wikidata_jurisdiction_closure.fetch_qid_country_closure(["q99", "Q100"])

    assert resolved == {"Q99": "Q30", "Q100": "Q145"}
    assert len(seen_requests) == 1
    assert (
        seen_requests[0].full_url
        == wikidata_jurisdiction_closure.WIKIDATA_QUERY_SERVICE_URL
    )
    assert seen_requests[0].data is not None


def test_fetch_qid_country_closure_returns_empty_mapping_for_no_input():
    assert wikidata_jurisdiction_closure.fetch_qid_country_closure([]) == {}


def test_fetch_qid_country_closure_omits_unresolvable_qids(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_urlopen(request, timeout=None):
        return _BytesResponse(_sparql_json_response([{"qid": "Q1", "country": "Q30"}]))

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    resolved = wikidata_jurisdiction_closure.fetch_qid_country_closure(["Q1", "Q2"])

    assert resolved == {"Q1": "Q30"}
    assert "Q2" not in resolved


def test_fetch_qid_country_closure_batches_large_qid_sets(
    monkeypatch: pytest.MonkeyPatch,
):
    requested_batches: list[int] = []

    def fake_urlopen(request, timeout=None):
        form = parse_qs(request.data.decode("utf-8"))
        query = form["query"][0]
        requested_batches.append(query.count("wd:Q"))
        return _BytesResponse(_sparql_json_response([]))

    monkeypatch.setattr(downloader_common, "urlopen", fake_urlopen)

    qids = [f"Q{i}" for i in range(1, 251)]
    wikidata_jurisdiction_closure.fetch_qid_country_closure(qids, batch_size=100)

    assert len(requested_batches) == 3
    assert sum(requested_batches) == 250
