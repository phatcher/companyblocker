"""Resolve a Wikidata QID to a country when the static QID-to-ISO2 table does not know it.

A QID the static table misses is resolved through a `?qid wdt:P131*/wdt:P17 ?country` walk against the Wikidata Query Service and cached in `resources/wikidata_qid_country_closure.json`, which `scripts/build_wikidata_jurisdiction_closure.py` refreshes offline from real cleansed output. The static table covers 72 sovereign states, so a country resolved outside it, Gabon for one, gives no jurisdiction rather than a wrong one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from urllib.parse import urlencode
from urllib.request import Request

from .downloader_common import (
    DEFAULT_USER_AGENT,
    open_remote_request,
    validate_remote_url,
)

WIKIDATA_QUERY_SERVICE_URL = "https://query.wikidata.org/sparql"

# WQS' own recommended ceiling for VALUES-clause batch size is generous, but
# keeping batches modest avoids tripping timeout/complexity limits on the
# shared public endpoint for a query this repo only ever runs offline, a
# handful of times, never in a hot path.
DEFAULT_BATCH_SIZE = 200


def _entity_uri_to_qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def build_qid_country_closure_query(qids: Iterable[str]) -> str:
    """Build a single SPARQL query resolving each QID in `qids` to a country
    QID via the Wikidata Query Service.

    Checks each QID's own P17 (country) claim, falling back to walking
    wdt:P131 ("located in the administrative territorial entity")
    transitively until something in the chain has a country. A single
    `wdt:P131*/wdt:P17` property path covers both cases in one query: `*`
    permits a zero-length hop, so a QID with its own direct P17 claim
    matches immediately without needing a separate branch.
    """
    values = " ".join(f"wd:{qid.upper()}" for qid in sorted(set(qids)))
    return (
        "PREFIX wd: <http://www.wikidata.org/entity/>\n"
        "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
        "SELECT ?qid (SAMPLE(?country) AS ?resolvedCountry) WHERE {\n"
        f"  VALUES ?qid {{ {values} }}\n"
        "  ?qid wdt:P131*/wdt:P17 ?country .\n"
        "}\n"
        "GROUP BY ?qid\n"
    )


def _parse_qid_country_closure_response(payload: dict) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for binding in payload.get("results", {}).get("bindings", []):
        qid_value = binding.get("qid", {}).get("value")
        country_value = binding.get("resolvedCountry", {}).get("value")
        if not qid_value or not country_value:
            continue
        qid = _entity_uri_to_qid(qid_value)
        country_qid = _entity_uri_to_qid(country_value)
        if qid and country_qid:
            resolved[qid] = country_qid
    return resolved


def _fetch_qid_country_closure_batch(
    qids: list[str],
    *,
    endpoint: str,
    user_agent: str,
    timeout_seconds: int,
) -> dict[str, str]:
    query = build_qid_country_closure_query(qids)
    body = urlencode({"query": query, "format": "json"}).encode("utf-8")
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    request = Request(validate_remote_url(endpoint), data=body, headers=headers)
    with open_remote_request(request, timeout_seconds=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _parse_qid_country_closure_response(payload)


def fetch_qid_country_closure(
    qids: Iterable[str],
    *,
    endpoint: str = WIKIDATA_QUERY_SERVICE_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_seconds: int = 120,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, str]:
    """Resolve every QID in `qids` to a country QID via the Wikidata Query
    Service, batching all of them into `VALUES`-clause queries (never one
    round-trip per QID).

    Offline, one-time fetch-and-cache pattern -- run once to build a static
    cache file, never a runtime dependency of the acquisition pipeline
    itself. QIDs with no resolvable country (a historical/defunct entity, a
    disputed territory, or simply no P131/P17 claim at all) are absent from
    the returned mapping; callers should treat "missing" as "unresolvable",
    not retry indefinitely.
    """
    unique_qids = sorted({qid.upper() for qid in qids if qid})
    if not unique_qids:
        return {}

    resolved: dict[str, str] = {}
    for start in range(0, len(unique_qids), batch_size):
        batch = unique_qids[start : start + batch_size]
        resolved.update(
            _fetch_qid_country_closure_batch(
                batch,
                endpoint=endpoint,
                user_agent=user_agent,
                timeout_seconds=timeout_seconds,
            )
        )
    return resolved
