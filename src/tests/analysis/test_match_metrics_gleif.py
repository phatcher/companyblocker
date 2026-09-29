from __future__ import annotations

import polars as pl
import pytest

from analysis.match_metrics import run_match_analysis
from analysis.match_metrics_gleif import (
    compute_gleif_not_recoverable_by_entity_category,
)
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration

_LOOKUP_DEFAULTS: dict[str, object] = {
    "company_number": None,
    "lei": None,
    "vat": None,
    "tax_id": None,
    "alternative_names": None,
    "previous_names": None,
    "company_type": None,
    "current_status": None,
    "incorporation_date": None,
    "dissolution_date": None,
    "inactive": None,
    "branch": None,
    "branch_status": None,
    "registered_address_in_full": None,
    "registry_url": None,
    "opencorporates_url": None,
    "metadata_generated_utc": None,
    "short_name": None,
    "quoted_name": None,
    "acronym": None,
    "personal_owner": None,
    "company_type_source": None,
    "company_type_missing": None,
}


def _lookup_row(
    *,
    system_uri: str,
    jurisdiction_code: str,
    name: str,
    name_cleansed_basic: str,
    name_cleansed: str,
    lei: str | None = None,
    company_number: str | None = None,
) -> dict[str, object]:
    row = dict(_LOOKUP_DEFAULTS)
    row.update(
        {
            "system_uri": system_uri,
            "jurisdiction_code": jurisdiction_code,
            "name": name,
            "name_cleansed_basic": name_cleansed_basic,
            "name_cleansed": name_cleansed,
            "lei": lei,
            "company_number": company_number,
        }
    )
    return row


def _write_lookup(
    roots: WorkspaceRoots, system: str, country: str, rows: list[dict[str, object]]
) -> None:
    target = layer_partition_dir(
        system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME), value=country
    )
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / "part-0.parquet")


def _write_matched(
    roots: WorkspaceRoots,
    source_system: str,
    country: str,
    rows: list[dict[str, object]],
) -> None:
    target = layer_partition_dir(
        system_layer_dir(roots, source_system, layer=MATCHED_LAYER_NAME), value=country
    )
    target.mkdir(parents=True, exist_ok=True)
    schema = {"system_uri": pl.Utf8, "match_uri": pl.Utf8, "jurisdiction_code": pl.Utf8}
    pl.DataFrame(rows, schema=schema).write_parquet(target / "part-0.parquet")


def _write_gleif_source_shard(
    roots: WorkspaceRoots, run_date: str, rows: list[dict[str, object]]
) -> None:
    # No constant for the source layer in `workspace.data_layout` -- the four
    # it names are canonical/cleansed/matched/tokenized -- so the layer is
    # named here while the path itself still comes from the layout function.
    target = system_layer_dir(roots, "gleif", layer="source") / run_date
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / "gleif-001.parquet")


def test_compute_gleif_not_recoverable_by_entity_category(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_lookup(
        workspace_roots,
        "gleif",
        "gb",
        [
            _lookup_row(
                system_uri="gleif:1",
                jurisdiction_code="gb",
                name="Acme Fund Ltd",
                name_cleansed_basic="acme fund ltd",
                name_cleansed="acmefund",
                lei="LEI-FUND-1",
                company_number="123",
            ),
            _lookup_row(
                system_uri="gleif:2",
                jurisdiction_code="gb",
                name="Zzq Unmatchable Co",
                name_cleansed_basic="zzq unmatchable co",
                name_cleansed="zzqunmatchable",
                lei="LEI-GENERAL-1",
                company_number=None,
            ),
        ],
    )
    _write_lookup(
        workspace_roots,
        "gb",
        "gb",
        [
            _lookup_row(
                system_uri="gb:1",
                jurisdiction_code="gb",
                name="Other Ltd",
                name_cleansed_basic="other ltd",
                name_cleansed="other",
                company_number="999",
            ),
        ],
    )
    _write_matched(
        workspace_roots,
        "gleif",
        "gb",
        [
            {"system_uri": "gleif:1", "match_uri": None, "jurisdiction_code": "gb"},
            {"system_uri": "gleif:2", "match_uri": None, "jurisdiction_code": "gb"},
        ],
    )
    _write_gleif_source_shard(
        workspace_roots,
        "2026-06-21",
        [
            {"LEI": "LEI-FUND-1", "Entity_EntityCategory": "FUND"},
            {"LEI": "LEI-GENERAL-1", "Entity_EntityCategory": "GENERAL"},
        ],
    )

    run_match_analysis(
        workspace_roots,
        source_system="gleif",
        target_system="gb",
        run_date="2026-08-24",
    )

    breakdown_df, match_paths, breakdown_path, chart_path = (
        compute_gleif_not_recoverable_by_entity_category(
            workspace_roots,
            run_date="2026-08-24",
            target_system="gb",
        )
    )

    assert breakdown_path.exists()
    assert chart_path.exists()
    assert match_paths.scenario == "gleif_to_gb_gb"

    rows = {row["entity_category"]: row for row in breakdown_df.iter_rows(named=True)}
    assert rows["FUND"]["not_recoverable_rows"] == 1
    assert rows["GENERAL"]["not_recoverable_rows"] == 1
    assert rows["FUND"]["pct_of_not_recoverable_total"] == 50.0
