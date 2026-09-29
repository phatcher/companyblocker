import polars as pl

from validation.gleif_name_variant_collapse import (
    _STRATUM_EXCLUDED_KNOCKOUT,
    _STRATUM_EXCLUDED_SHARED_SUBSTRING,
    _STRATUM_TESTABLE,
    build_variant_pair_frame,
    compute_gleif_name_variant_collapse,
    compute_live_name_representations,
    load_gleif_name_variant_rows,
    resolve_gleif_cleanse_config,
    resolve_gleif_entity_identities,
    split_alias_and_primary_rows,
)
from workspace.data_layout import (
    MATCHED_LAYER_NAME,
    canonical_snapshot_dir,
    system_layer_dir,
)
from workspace.roots import WorkspaceRoots


def _write_gleif_name_rows(
    roots: WorkspaceRoots,
    rows: list[dict[str, object]],
    *,
    snapshot: str = "2026-01-01",
    file_name: str = "gleif-names-001.parquet",
) -> None:
    snapshot_dir = canonical_snapshot_dir(roots, system="gleif", run_date=snapshot)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(snapshot_dir / file_name)


def _write_gleif_matched_layer(
    roots: WorkspaceRoots, rows: list[dict[str, object]]
) -> None:
    matched_dir = system_layer_dir(roots, "gleif", layer=MATCHED_LAYER_NAME)
    matched_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(matched_dir / "gleif-001.parquet")


# Five synthetic entities exercising every stratum and the population-scoping
# rule, real GLEIF cleanse config, values confirmed by hand against
# resolve_gleif_cleanse_config()'s actual output. Each entity's primary row
# keeps GLEIF's own `gleif://` entity-scheme system_uri; each alias/variant
# row is a regenerated sidecar row with its own unique `name://` system_uri
# and a `source_uri` back-reference to its owner entity, matching the shape
# `workspace.match_resolution` resolves.
#   E1: "ACME CORPORATION LIMITED" / "ACME LTD"      -> testable, collapses (short_name "acme" both sides)
#   E2: "NORTHWIND HOLDINGS LIMITED" / "NORTHWIND TRADERS LIMITED" -> testable, does not collapse
#   E3: "TOTALLY DIFFERENT WIDGETS LLC" / "UNRELATED BRAND CO" -> zero raw-token overlap
#   E4: "FOOBAR SERVICES LIMITED" / "  Foobar   Services   Limited  " -> identical after cleanse
#   E5: "SINGLETON ONLY LIMITED" (primary only, no variant row) -> outside the self-join population
_FIXTURE_ROWS: list[dict[str, object]] = [
    {
        "system_uri": "gleif://E1",
        "source_uri": None,
        "name": "ACME CORPORATION LIMITED",
        "name_type": "primary",
    },
    {
        "system_uri": "name://gleif/E1/0000000000000001",
        "source_uri": "gleif://E1",
        "name": "ACME LTD",
        "name_type": "trading",
    },
    {
        "system_uri": "gleif://E2",
        "source_uri": None,
        "name": "NORTHWIND HOLDINGS LIMITED",
        "name_type": "primary",
    },
    {
        "system_uri": "name://gleif/E2/0000000000000002",
        "source_uri": "gleif://E2",
        "name": "NORTHWIND TRADERS LIMITED",
        "name_type": "trading",
    },
    {
        "system_uri": "gleif://E3",
        "source_uri": None,
        "name": "TOTALLY DIFFERENT WIDGETS LLC",
        "name_type": "primary",
    },
    {
        "system_uri": "name://gleif/E3/0000000000000003",
        "source_uri": "gleif://E3",
        "name": "UNRELATED BRAND CO",
        "name_type": "previous",
    },
    {
        "system_uri": "gleif://E4",
        "source_uri": None,
        "name": "FOOBAR SERVICES LIMITED",
        "name_type": "primary",
    },
    {
        "system_uri": "name://gleif/E4/0000000000000004",
        "source_uri": "gleif://E4",
        "name": "  Foobar   Services   Limited  ",
        "name_type": "trading",
    },
    {
        "system_uri": "gleif://E5",
        "source_uri": None,
        "name": "SINGLETON ONLY LIMITED",
        "name_type": "primary",
    },
]

# GLEIF's own matched/ layer, keyed by each entity's own gleif:// system_uri
# -- the terminal hop resolve_cross_system_match performs for every row here,
# alias or primary, once its walk reaches the entity's own scheme. Distinct
# match_uri per entity so different entities never collide on the resolved
# identity; E5 gets one too even though it never appears in an alias pairing.
_MATCHED_ROWS: list[dict[str, object]] = [
    {"system_uri": "gleif://E1", "match_uri": "gb://m1"},
    {"system_uri": "gleif://E2", "match_uri": "gb://m2"},
    {"system_uri": "gleif://E3", "match_uri": "gb://m3"},
    {"system_uri": "gleif://E4", "match_uri": "gb://m4"},
    {"system_uri": "gleif://E5", "match_uri": "gb://m5"},
]


def _write_fixture(roots: WorkspaceRoots) -> None:
    _write_gleif_name_rows(roots, _FIXTURE_ROWS)
    _write_gleif_matched_layer(roots, _MATCHED_ROWS)


def test_resolve_gleif_cleanse_config_uses_name_column() -> None:
    config = resolve_gleif_cleanse_config()
    assert config.company_col == "name"


def test_load_gleif_name_variant_rows_reads_latest_snapshot(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_gleif_name_rows(
        workspace_roots,
        [
            {
                "system_uri": "gleif://old",
                "source_uri": None,
                "name": "OLD ENTITY LIMITED",
                "name_type": "primary",
            }
        ],
        snapshot="2025-01-01",
    )
    _write_gleif_name_rows(
        workspace_roots,
        [
            {
                "system_uri": "gleif://new",
                "source_uri": None,
                "name": "NEW ENTITY LIMITED",
                "name_type": "primary",
            }
        ],
        snapshot="2026-01-01",
    )

    rows = load_gleif_name_variant_rows(roots=workspace_roots)

    assert rows["system_uri"].to_list() == ["gleif://new"]


def test_resolve_gleif_entity_identities_resolves_through_the_shared_walk(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_fixture(workspace_roots)
    name_rows = load_gleif_name_variant_rows(roots=workspace_roots)

    identities = resolve_gleif_entity_identities(
        name_rows.get_column("system_uri"), roots=workspace_roots
    )
    identity_by_uri = dict(
        zip(identities["system_uri"].to_list(), identities["entity_identity"].to_list())
    )

    # An alias's own system_uri and its primary's own system_uri resolve to
    # the same entity identity, even though they no longer share a raw
    # system_uri.
    assert (
        identity_by_uri["name://gleif/E1/0000000000000001"]
        == identity_by_uri["gleif://E1"]
    )
    assert identity_by_uri["gleif://E1"] == "gb://m1"


def test_split_alias_and_primary_rows_scopes_to_entities_with_variants(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_fixture(workspace_roots)
    name_rows = load_gleif_name_variant_rows(roots=workspace_roots)

    alias_rows, primary_rows = split_alias_and_primary_rows(
        name_rows, roots=workspace_roots
    )

    # E5 is primary-only (no variant row) and must not appear as a target.
    assert set(primary_rows["target_id"].to_list()) == {
        "gb://m1",
        "gb://m2",
        "gb://m3",
        "gb://m4",
    }
    assert alias_rows.height == 4
    # variant_id is the alias row's own real, already-unique system_uri.
    assert set(alias_rows["variant_id"].to_list()) == {
        "name://gleif/E1/0000000000000001",
        "name://gleif/E2/0000000000000002",
        "name://gleif/E3/0000000000000003",
        "name://gleif/E4/0000000000000004",
    }
    assert alias_rows["variant_id"].n_unique() == alias_rows.height


def test_split_alias_and_primary_rows_drops_entities_with_no_recorded_match(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_gleif_name_rows(
        workspace_roots,
        [
            {
                "system_uri": "gleif://E1",
                "source_uri": None,
                "name": "ACME CORPORATION LIMITED",
                "name_type": "primary",
            },
            {
                "system_uri": "name://gleif/E1/0000000000000001",
                "source_uri": "gleif://E1",
                "name": "ACME LTD",
                "name_type": "trading",
            },
        ],
    )
    _write_gleif_matched_layer(
        workspace_roots, [{"system_uri": "gleif://E1", "match_uri": None}]
    )
    name_rows = load_gleif_name_variant_rows(roots=workspace_roots)

    alias_rows, primary_rows = split_alias_and_primary_rows(
        name_rows, roots=workspace_roots
    )

    assert alias_rows.height == 0
    assert primary_rows.height == 0


def test_build_variant_pair_frame_stratifies_correctly(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_fixture(workspace_roots)
    name_rows = load_gleif_name_variant_rows(roots=workspace_roots)
    alias_rows, primary_rows = split_alias_and_primary_rows(
        name_rows, roots=workspace_roots
    )
    cleanse_config = resolve_gleif_cleanse_config()
    alias_reps, primary_reps = compute_live_name_representations(
        alias_rows=alias_rows, primary_rows=primary_rows, cleanse_config=cleanse_config
    )

    pairs = build_variant_pair_frame(alias_reps, primary_reps)
    stratum_by_primary = dict(
        zip(pairs["primary_system_uri"].to_list(), pairs["stratum"].to_list())
    )

    assert stratum_by_primary["gb://m1"] == _STRATUM_TESTABLE
    assert stratum_by_primary["gb://m2"] == _STRATUM_TESTABLE
    assert stratum_by_primary["gb://m3"] == _STRATUM_EXCLUDED_SHARED_SUBSTRING
    assert stratum_by_primary["gb://m4"] == _STRATUM_EXCLUDED_KNOCKOUT

    # Live-compute discipline: short_name/name_cleansed only exist because
    # this pass computed them itself -- no name_cleansed column was ever
    # written into the fixture's raw rows.
    assert "name_cleansed" not in _FIXTURE_ROWS[0]
    assert pairs["alias_short_name"].null_count() == 0
    assert pairs["primary_short_name"].null_count() == 0


def test_compute_gleif_name_variant_collapse_short_name_recall(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_fixture(workspace_roots)

    result = compute_gleif_name_variant_collapse(
        roots=workspace_roots, representation="short_name"
    )

    assert result.total_variant_pairs == 4
    assert result.excluded_shared_substring == 1
    assert result.excluded_knockout == 1
    assert result.testable_pairs == 2
    assert result.target_rows == 4

    eval_row = result.pair_truth_eval
    assert eval_row.height == 1
    assert eval_row["labelled_sources"][0] == 2
    assert eval_row["truth_pairs"][0] == 2
    assert eval_row["tp"][0] == 1
    assert eval_row["fn"][0] == 1
    assert eval_row["fp"][0] == 0

    # E1's variant ("ACME LTD" -> short_name "acme") collapses into its own
    # primary ("ACME CORPORATION LIMITED" -> short_name "acme"); E2's variant
    # ("NORTHWIND TRADERS LIMITED" -> short_name "northwind traders") does
    # not collapse into its primary's "northwind" -- collapse rate is 1/2.
    assert result.collapse_rate == 0.5


def test_compute_gleif_name_variant_collapse_name_cleansed_recall(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_fixture(workspace_roots)

    result = compute_gleif_name_variant_collapse(
        roots=workspace_roots, representation="name_cleansed"
    )

    # Neither testable pair (E1, E2) is byte-identical on name_cleansed
    # (that's exactly what the knockout stratum would have caught), so
    # name_cleansed as a "representation" collapses neither -- 0/2.
    assert result.collapse_rate == 0.0
