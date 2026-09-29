from __future__ import annotations

import polars as pl

from analysis.match_metrics import (
    resolve_match_analysis_paths,
    resolve_match_analysis_run_root,
    run_match_analysis,
    run_match_analysis_batch,
)
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots

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
    pl.DataFrame(rows).write_parquet(target / "part-0.parquet")


def _seed_scenario(
    roots: WorkspaceRoots, *, source_system: str, target_system: str, country: str
) -> None:
    _write_lookup(
        roots,
        source_system,
        country,
        [
            _lookup_row(
                system_uri="src:1",
                jurisdiction_code=country,
                name="Acme Ltd",
                name_cleansed_basic="acme ltd",
                name_cleansed="acme",
                company_number="123",
            ),
            _lookup_row(
                system_uri="src:2",
                jurisdiction_code=country,
                name="Beta Group",
                name_cleansed_basic="beta group",
                name_cleansed="beta",
                company_number=None,
            ),
        ],
    )
    _write_lookup(
        roots,
        target_system,
        country,
        [
            _lookup_row(
                system_uri="tgt:1",
                jurisdiction_code=country,
                name="Acme Ltd",
                name_cleansed_basic="acme ltd",
                name_cleansed="acme",
                company_number="123",
            ),
            _lookup_row(
                system_uri="tgt:2",
                jurisdiction_code=country,
                name="Beta Group PLC",
                name_cleansed_basic="beta group plc",
                name_cleansed="beta",
                company_number="456",
            ),
        ],
    )
    _write_matched(
        roots,
        source_system,
        country,
        [
            {"system_uri": "src:1", "match_uri": "tgt:1", "jurisdiction_code": country},
            {"system_uri": "src:2", "match_uri": None, "jurisdiction_code": country},
        ],
    )


def test_resolve_match_analysis_run_root_names_the_same_directory_the_paths_resolver_uses(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A caller reporting a dry-run plan wants this path with no side effect;
    it must agree with what `resolve_match_analysis_paths` (which does create
    the directories) actually resolves."""
    run_root = resolve_match_analysis_run_root(
        workspace_roots,
        run_date="2026-09-16",
        source_system="gleif",
        target_system="gb",
        country="gb",
    )

    assert not run_root.exists()

    paths = resolve_match_analysis_paths(
        workspace_roots,
        run_date="2026-09-16",
        source_system="gleif",
        target_system="gb",
        country="gb",
    )

    assert paths.run_root == run_root


def test_run_match_analysis_writes_artifacts(workspace_roots: WorkspaceRoots) -> None:
    _seed_scenario(
        workspace_roots, source_system="gleif", target_system="gb", country="gb"
    )

    paths = run_match_analysis(
        workspace_roots,
        source_system="gleif",
        target_system="gb",
        run_date="2026-08-24",
    )

    assert paths.summary_path.exists()
    assert paths.reasons_path.exists()
    assert paths.examples_path.exists()
    assert paths.match_outcome_chart_path.exists()
    assert paths.equality_rates_chart_path.exists()
    assert paths.equality_overlap_venn_path.exists()
    assert paths.matched_overlap_states_chart_path.exists()
    assert paths.no_match_reasons_chart_path.exists()
    assert paths.no_match_recoverability_chart_path.exists()
    assert paths.no_match_recoverability_detail_path.exists()

    detail_df = pl.read_parquet(paths.no_match_recoverability_detail_path)
    assert detail_df.height == 1
    assert detail_df.item(0, "system_uri") == "src:2"
    assert detail_df.item(0, "not_recoverable") is False

    summary_df = pl.read_parquet(paths.summary_path)
    values = dict(zip(summary_df["metric"].to_list(), summary_df["value"].to_list()))
    assert values["total_rows"] == 2.0
    assert values["matched_rows"] == 1.0
    assert values["unmatched_rows"] == 1.0
    assert values["match_rate_pct"] == 50.0

    reasons_df = pl.read_parquet(paths.reasons_path)
    assert reasons_df.height == 1
    assert reasons_df.item(0, "no_match_reason") == "MISSING_COMPANY_NUMBER"

    strategy_df = pl.read_parquet(paths.no_match_strategy_path)
    beta_recoverable = strategy_df.filter(pl.col("strategy") == "cleansed").item(
        0, "recoverable_rows"
    )
    assert beta_recoverable == 1


def test_run_match_analysis_reads_family_split_primary_directories(
    workspace_roots: WorkspaceRoots,
) -> None:
    """Real matched/ and cleansed/ output lives under a `primary/` family
    directory, not directly at the layer's own top level --
    `_load_country_frame` must resolve that shape rather than only the
    pre-split layout the other test in this file exercises.
    """
    country = "gb"
    lookup_rows = [
        _lookup_row(
            system_uri="src:1",
            jurisdiction_code=country,
            name="Acme Ltd",
            name_cleansed_basic="acme ltd",
            name_cleansed="acme",
            company_number="123",
        )
    ]
    for system in ("gleif", "gb"):
        _write_lookup(workspace_roots, system, country, lookup_rows)

    _write_matched(
        workspace_roots,
        "gleif",
        country,
        [{"system_uri": "src:1", "match_uri": "src:1", "jurisdiction_code": country}],
    )

    paths = run_match_analysis(
        workspace_roots,
        source_system="gleif",
        target_system="gb",
        run_date="2026-08-24",
    )

    summary_df = pl.read_parquet(paths.summary_path)
    values = dict(zip(summary_df["metric"].to_list(), summary_df["value"].to_list()))
    assert values["total_rows"] == 1.0
    assert values["matched_rows"] == 1.0


def test_run_match_analysis_batch_runs_multiple_scenarios(
    workspace_roots: WorkspaceRoots,
) -> None:
    _seed_scenario(
        workspace_roots, source_system="gleif", target_system="gb", country="gb"
    )
    _seed_scenario(
        workspace_roots, source_system="gleif", target_system="fr", country="fr"
    )

    results = run_match_analysis_batch(
        workspace_roots,
        [
            {"source_system": "gleif", "target_system": "gb", "country": "gb"},
            {"source_system": "gleif", "target_system": "fr", "country": "fr"},
        ],
        run_date="2026-08-24",
    )

    assert len(results) == 2
    scenarios = {paths.scenario for paths in results}
    assert scenarios == {"gleif_to_gb_gb", "gleif_to_fr_fr"}
    for paths in results:
        assert paths.summary_path.exists()
        assert paths.match_outcome_chart_path.exists()
