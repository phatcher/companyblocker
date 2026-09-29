import json
from pathlib import Path

import polars as pl
import pytest

from acquisition.match_ops import materialize_match_uri_artifact
from workspace.roots import WorkspaceRoots


@pytest.mark.integration
def test_materialize_match_uri_artifact_left_join_preserves_source_rows(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    source_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    source_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1", "ie:2", "ie:3"],
            "jurisdiction_code": ["ie", "ie", "ie"],
            "company_number": ["1", "2", None],
            "match_uri": [None, "legacy:2", None],
        }
    ).write_parquet(source_partition / "part-00001.parquet")
    pl.DataFrame(
        {
            "system_uri": ["ie:4"],
            "jurisdiction_code": ["ie"],
            "company_number": ["4"],
            "match_uri": ["legacy:4"],
        }
    ).write_parquet(source_partition / "part-00002.parquet")

    target_ie_partition = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    target_ie_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": ["gleif:entity:1"],
        }
    ).write_parquet(target_ie_partition / "part-00001.parquet")

    target_gb_partition = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=gb"
    )
    target_gb_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:gb:1"],
            "jurisdiction_code": ["gb"],
            "company_number": ["GB1"],
            "match_uri": ["gleif:entity:gb1"],
        }
    ).write_parquet(target_gb_partition / "part-00001.parquet")

    summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="gleif",
        target_system="ie",
    )

    output_partition = (
        layer_fixture_dir("gleif", layer="matched") / "primary" / "jurisdiction_code=ie"
    )
    output_files = sorted(output_partition.glob("part-*.parquet"))
    assert [path.name for path in output_files] == ["part-00001.parquet"]

    out = pl.read_parquet(output_files)
    assert out.height == 1
    assert out["system_uri"].to_list() == ["gleif:1"]
    assert out["match_uri"].to_list() == ["ie:1"]

    assert summary.source_system == "gleif"
    assert summary.source_country == "ie"
    assert summary.target_system == "ie"
    assert summary.output_file_count == 1
    assert summary.rows_written == 1
    assert summary.rows_matched == 1
    assert summary.source_rows_total == 1
    assert summary.source_valid_keys == 1
    assert summary.source_invalid_keys == 0
    assert summary.target_rows_total == 4
    assert summary.target_valid_keys == 3
    assert summary.target_invalid_keys == 1
    assert summary.elapsed_seconds >= 0.0


@pytest.mark.integration
def test_materialize_match_uri_writes_only_the_named_source(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    source_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    source_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": [None],
        }
    ).write_parquet(source_partition / "part-00001.parquet")

    gleif_ie = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    gleif_ie.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": ["gleif:1"],
        }
    ).write_parquet(gleif_ie / "part-00001.parquet")
    gleif_gb = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=gb"
    )
    gleif_gb.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:gb:1"],
            "jurisdiction_code": ["gb"],
            "company_number": ["GB1"],
            "match_uri": ["gleif:gb:1"],
        }
    ).write_parquet(gleif_gb / "part-00001.parquet")

    wikidata_ie = (
        layer_fixture_dir("wikidata", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    wikidata_ie.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["wikidata:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": ["wikidata:1"],
        }
    ).write_parquet(wikidata_ie / "part-00001.parquet")
    wikidata_gb = (
        layer_fixture_dir("wikidata", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=gb"
    )
    wikidata_gb.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["wikidata:gb:1"],
            "jurisdiction_code": ["gb"],
            "company_number": ["GB1"],
            "match_uri": ["wikidata:gb:1"],
        }
    ).write_parquet(wikidata_gb / "part-00001.parquet")

    # A match is the named source against the named target. Another system
    # on disk sharing the jurisdiction is not matched because it is there.
    summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="gleif",
        target_system="ie",
    )
    assert (summary.source_system, summary.target_system) == ("gleif", "ie")
    assert summary.source_country == "ie"
    assert layer_fixture_dir("gleif", layer="matched").is_dir()
    assert not layer_fixture_dir("wikidata", layer="matched").exists()
    assert not layer_fixture_dir("ie", layer="matched").exists()


@pytest.mark.integration
def test_materialize_match_uri_supports_reverse_direction_with_explicit_target(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    gleif_ie = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    gleif_ie.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:ie:1", "gleif:ie:2"],
            "jurisdiction_code": ["ie", "ie"],
            "company_number": ["1", "999"],
            "match_uri": [None, None],
        }
    ).write_parquet(gleif_ie / "part-00001.parquet")

    gleif_gb = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=gb"
    )
    gleif_gb.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:gb:1"],
            "jurisdiction_code": ["gb"],
            "company_number": ["GB1"],
            "match_uri": [None],
        }
    ).write_parquet(gleif_gb / "part-00001.parquet")

    ie_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    ie_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": ["ie:1"],
        }
    ).write_parquet(ie_partition / "part-00001.parquet")

    summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="gleif",
        target_system="ie",
    )

    out = pl.read_parquet(
        layer_fixture_dir("gleif", layer="matched")
        / "primary"
        / "jurisdiction_code=ie"
        / "part-00001.parquet"
    )
    assert out["system_uri"].to_list() == ["gleif:ie:1", "gleif:ie:2"]
    assert out["match_uri"].to_list() == ["ie:1", None]

    assert summary.source_system == "gleif"
    assert summary.source_country == "ie"
    assert summary.target_system == "ie"
    assert summary.rows_written == 2
    assert summary.rows_matched == 1


@pytest.mark.integration
def test_materialize_match_uri_excludes_ambiguous_target_keys_without_failing(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    source_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    source_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1", "ie:2"],
            "jurisdiction_code": ["ie", "ie"],
            "company_number": ["1", "2"],
            "match_uri": [None, None],
        }
    ).write_parquet(source_partition / "part-00001.parquet")

    target_partition = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    target_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            # Two target rows share company_number "1" -- an ambiguous key.
            # Company_number "2" resolves cleanly.
            "system_uri": ["gleif:1a", "gleif:1b", "gleif:2"],
            "jurisdiction_code": ["ie", "ie", "ie"],
            "company_number": ["1", "1", "2"],
            "match_uri": [None, None, None],
        }
    ).write_parquet(target_partition / "part-00001.parquet")

    summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="ie",
        target_system="gleif",
    )

    assert summary.target_duplicate_keys == 1

    out = pl.read_parquet(
        layer_fixture_dir("ie", layer="matched")
        / "primary"
        / "jurisdiction_code=ie"
        / "part-00001.parquet"
    )
    by_system_uri = dict(
        zip(out["system_uri"].to_list(), out["match_uri"].to_list(), strict=True)
    )
    assert by_system_uri["ie:1"] is None  # ambiguous key excluded, not guessed
    assert by_system_uri["ie:2"] == "gleif:2"

    duplicate_report = pl.read_parquet(
        layer_fixture_dir("ie", layer="matched") / "_duplicate_keys.parquet"
    )
    assert duplicate_report.height == 1
    row = duplicate_report.row(0, named=True)
    assert row["side"] == "target"
    assert row["jurisdiction_key"] == "ie"
    assert row["company_number_key"] == "1"
    assert row["row_count"] == 2


@pytest.mark.integration
def test_materialize_match_uri_writes_empty_duplicate_report_when_clean(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    source_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    source_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": [None],
        }
    ).write_parquet(source_partition / "part-00001.parquet")

    target_partition = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    target_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": [None],
        }
    ).write_parquet(target_partition / "part-00001.parquet")

    materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="ie",
        target_system="gleif",
    )

    duplicate_report = pl.read_parquet(
        layer_fixture_dir("ie", layer="matched") / "_duplicate_keys.parquet"
    )
    assert duplicate_report.height == 0


@pytest.mark.integration
def test_materialize_match_uri_metadata_merges_target_systems_across_countries(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    """Regression: a second country's match run must not erase the
    first country's `target_systems` entry in the shared `_match_metadata.json`.
    """
    gleif_ie = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    gleif_ie.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": [None],
        }
    ).write_parquet(gleif_ie / "part-00001.parquet")

    gleif_gb = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=gb"
    )
    gleif_gb.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:gb:1"],
            "jurisdiction_code": ["gb"],
            "company_number": ["GB1"],
            "match_uri": [None],
        }
    ).write_parquet(gleif_gb / "part-00001.parquet")

    ie_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    ie_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": ["ie:1"],
        }
    ).write_parquet(ie_partition / "part-00001.parquet")

    gb_partition = (
        layer_fixture_dir("gb", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=gb"
    )
    gb_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb:1"],
            "jurisdiction_code": ["gb"],
            "company_number": ["GB1"],
            "match_uri": ["gb:1"],
        }
    ).write_parquet(gb_partition / "part-00001.parquet")

    first_summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="gleif",
        target_system="ie",
    )
    assert first_summary.target_system == "ie"

    metadata_path = layer_fixture_dir("gleif", layer="matched") / "_match_metadata.json"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["target_systems"] == [
        "ie"
    ]

    second_summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="gleif",
        target_system="gb",
    )
    assert second_summary.target_system == "gb"

    merged_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert merged_metadata["target_systems"] == ["ie", "gb"]

    # Both real partitions on disk are still intact after the second run.
    assert (
        layer_fixture_dir("gleif", layer="matched")
        / "primary"
        / "jurisdiction_code=ie"
        / "part-00001.parquet"
    ).exists()
    assert (
        layer_fixture_dir("gleif", layer="matched")
        / "primary"
        / "jurisdiction_code=gb"
        / "part-00001.parquet"
    ).exists()


@pytest.mark.integration
def test_materialize_match_uri_only_replaces_owned_jurisdiction_partition(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    gleif_ie = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    gleif_ie.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": [None],
        }
    ).write_parquet(gleif_ie / "part-00001.parquet")

    gleif_gb = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=gb"
    )
    gleif_gb.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:gb:1"],
            "jurisdiction_code": ["gb"],
            "company_number": ["GB1"],
            "match_uri": [None],
        }
    ).write_parquet(gleif_gb / "part-00001.parquet")

    ie_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    ie_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["1"],
            "match_uri": ["ie:1"],
        }
    ).write_parquet(ie_partition / "part-00001.parquet")

    existing_fr_partition = (
        layer_fixture_dir("gleif", layer="matched") / "primary" / "jurisdiction_code=fr"
    )
    existing_fr_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:fr:1"],
            "jurisdiction_code": ["fr"],
            "company_number": ["FR1"],
            "match_uri": ["fr:1"],
        }
    ).write_parquet(existing_fr_partition / "part-00001.parquet")

    materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="gleif",
        target_system="ie",
    )

    assert (existing_fr_partition / "part-00001.parquet").exists()
    fr_out = pl.read_parquet(existing_fr_partition / "part-00001.parquet")
    assert fr_out.get_column("system_uri").to_list() == ["gleif:fr:1"]

    ie_out = pl.read_parquet(
        layer_fixture_dir("gleif", layer="matched")
        / "primary"
        / "jurisdiction_code=ie"
        / "part-00001.parquet"
    )
    assert ie_out.get_column("match_uri").to_list() == ["ie:1"]


@pytest.mark.integration
def test_materialize_match_uri_tolerates_leading_zero_padding(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    """A purely-numeric company_number matches across inconsistent
    zero-padding (e.g. '000123' vs '123'), for any jurisdiction.
    """
    source_partition = (
        layer_fixture_dir("ie", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    source_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["000123"],
            "match_uri": [None],
        }
    ).write_parquet(source_partition / "part-00001.parquet")

    target_partition = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=ie"
    )
    target_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1"],
            "jurisdiction_code": ["ie"],
            "company_number": ["123"],
            "match_uri": [None],
        }
    ).write_parquet(target_partition / "part-00001.parquet")

    summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="ie",
        target_system="gleif",
    )
    assert summary.rows_matched == 1

    out = pl.read_parquet(
        layer_fixture_dir("ie", layer="matched")
        / "primary"
        / "jurisdiction_code=ie"
        / "part-00001.parquet"
    )
    assert out["match_uri"].to_list() == ["gleif:1"]


@pytest.mark.integration
def test_materialize_match_uri_strips_german_trailing_court_suffix(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    """OffeneRegister and GLEIF disagree on a trailing court-specific
    suffix appended after the register number (e.g. 'HRB12826' vs
    'HRB12826HL'); the register-type prefix (HRA/HRB) still has to match.
    """
    source_partition = (
        layer_fixture_dir("offeneregister", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=de"
    )
    source_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["offeneregister:1", "offeneregister:2"],
            "jurisdiction_code": ["de", "de"],
            "company_number": ["RA000290|HRB12826HL", "RA000290|HRA100109"],
            "match_uri": [None, None],
        }
    ).write_parquet(source_partition / "part-00001.parquet")

    target_partition = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=de"
    )
    target_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1", "gleif:2"],
            "jurisdiction_code": ["de", "de"],
            # gleif:1 matches via suffix tolerance; gleif:2 shares the same
            # digits but a different register type (HRB, not HRA) and must
            # NOT be treated as a match.
            "company_number": ["RA000290|HRB12826", "RA000290|HRB100109"],
            "match_uri": [None, None],
        }
    ).write_parquet(target_partition / "part-00001.parquet")

    summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="offeneregister",
        target_system="gleif",
    )
    assert summary.rows_matched == 1

    out = pl.read_parquet(
        layer_fixture_dir("offeneregister", layer="matched")
        / "primary"
        / "jurisdiction_code=de"
        / "part-00001.parquet"
    )
    by_system_uri = dict(
        zip(out["system_uri"].to_list(), out["match_uri"].to_list(), strict=True)
    )
    assert by_system_uri["offeneregister:1"] == "gleif:1"
    assert by_system_uri["offeneregister:2"] is None  # HRA vs HRB: not merged


@pytest.mark.integration
def test_materialize_match_uri_reduces_french_siret_to_siren(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    """A 14-digit SIRET (SIREN + 5-digit establishment code) on the
    GLEIF side matches the target's 9-digit SIREN-only company_number.
    """
    source_partition = (
        layer_fixture_dir("fr", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=fr"
    )
    source_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr:1"],
            "jurisdiction_code": ["fr"],
            "company_number": ["931942544"],
            "match_uri": [None],
        }
    ).write_parquet(source_partition / "part-00001.parquet")

    target_partition = (
        layer_fixture_dir("gleif", layer="canonical")
        / "2026-01-01"
        / "primary"
        / "jurisdiction_code=fr"
    )
    target_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:1"],
            "jurisdiction_code": ["fr"],
            "company_number": ["93194254400012"],
            "match_uri": [None],
        }
    ).write_parquet(target_partition / "part-00001.parquet")

    summary = materialize_match_uri_artifact(
        roots=workspace_roots,
        source_system="fr",
        target_system="gleif",
    )
    assert summary.rows_matched == 1

    out = pl.read_parquet(
        layer_fixture_dir("fr", layer="matched")
        / "primary"
        / "jurisdiction_code=fr"
        / "part-00001.parquet"
    )
    assert out["match_uri"].to_list() == ["gleif:1"]
