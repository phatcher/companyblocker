#!/usr/bin/env python3
"""Build the QID -> country-QID closure cache for jurisdiction_code resolution.

The cache is consumed by acquisition.canonical_system_config.

Offline, one-time fetch-and-cache: discovers every unmapped country/
jurisdiction QID observed in real Wikidata cleansed output (a Hive
`jurisdiction_code=Q.../` partition not already covered by
_WIKIDATA_COUNTRY_QID_TO_ISO2), resolves each to a country QID via a single
batched SPARQL query against the Wikidata Query Service, and writes the
result to a static, checked-in resource file. Never run as part of the
regular acquisition pipeline -- re-run manually if/when new unmapped QIDs
show up in a later Wikidata cleansed regeneration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap

from acquisition.canonical_system_config import (
    _WIKIDATA_COUNTRY_QID_TO_ISO2,
)
from acquisition.wikidata_jurisdiction_closure import (
    DEFAULT_BATCH_SIZE,
    fetch_qid_country_closure,
)
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.layer_layout import partition_values
from workspace.roots import default_workspace_roots

DEFAULT_CLEANSED_DIR = system_layer_dir(
    default_workspace_roots(_bootstrap.REPO_ROOT), "wikidata", layer=CLEANSED_LAYER_NAME
)
DEFAULT_OUTPUT_PATH = (
    _bootstrap.REPO_ROOT
    / "src"
    / "acquisition"
    / "resources"
    / "wikidata_qid_country_closure.json"
)


def discover_offending_qids(cleansed_dir: Path) -> list[str]:
    """Scan a real Wikidata cleansed_dir's Hive jurisdiction_code=*/
    partition values (via `workspace.layer_layout.partition_values`, which
    resolves both the current `primary/`-family location and the
    pre-family-split top-level one) for QID-shaped values not already
    covered by the static ISO2 table -- these are exactly the values that
    leak through unresolved (or now resolve to None) today.
    """
    offending: set[str] = set()
    for value in partition_values(cleansed_dir):
        if not value or not value.startswith("q") or not value[1:].isdigit():
            continue
        qid = value.upper()
        if qid not in _WIKIDATA_COUNTRY_QID_TO_ISO2:
            offending.add(qid)
    return sorted(offending)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache Wikidata QID -> country-QID resolution "
        "for jurisdiction_code values not covered by the static ISO2 table."
    )
    parser.add_argument(
        "--cleansed-dir",
        type=Path,
        default=DEFAULT_CLEANSED_DIR,
        help="Wikidata cleansed_dir to scan for offending jurisdiction_code= partitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the resolved qid -> country-qid cache JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Max QIDs per VALUES-clause SPARQL query batch.",
    )
    parser.add_argument(
        "--qid",
        action="append",
        default=[],
        help="Explicit QID to resolve, in addition to any discovered from "
        "--cleansed-dir. May be repeated. Useful for extending the cache "
        "without a real cleansed_dir on hand.",
    )
    args = parser.parse_args()

    qids = set(discover_offending_qids(args.cleansed_dir))
    qids.update(qid.upper() for qid in args.qid)
    if not qids:
        print("No offending QIDs found -- nothing to fetch.")
        return

    print(f"Resolving {len(qids)} QID(s) via the Wikidata Query Service...")
    started = time.monotonic()
    resolved = fetch_qid_country_closure(sorted(qids), batch_size=args.batch_size)
    elapsed = time.monotonic() - started
    print(
        f"Resolved {len(resolved)}/{len(qids)} QID(s) to a country QID in {elapsed:.1f}s."
    )

    unresolved = sorted(qids - resolved.keys())
    if unresolved:
        print(
            f"{len(unresolved)} QID(s) remain unresolved (no P131*/P17 path found): "
            f"{', '.join(unresolved[:20])}" + (" ..." if len(unresolved) > 20 else "")
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(sorted(resolved.items())), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
