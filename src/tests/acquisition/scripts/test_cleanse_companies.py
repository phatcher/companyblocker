from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from acquisition.cleansed_contracts import CLEANSED_MERGE_REQUIRED_COLUMNS
from scripts import process_companies
from workspace.data_file_naming import COMPANION_DUPLICATES, COMPANION_NAMES


def test_process_script_cleanse_applies_registry_source_company_type_mapping(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    canonical_dir = (
        layer_fixture_dir("gb", layer="canonical")
        / "2026-06-15"
        / "primary"
        / "jurisdiction_code=gb"
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gb:sample"],
            "jurisdiction_code": ["gb"],
            "name": ["EXAMPLE FUND"],
            "CompanyCategory": ["Investment Company with Variable Capital"],
        }
    )
    source.write_parquet(canonical_dir / "part-00001.parquet")

    monkeypatch.setattr(
        "sys.argv",
        [
            "process_companies.py",
            "--systems",
            "gb",
            "--processes",
            "cleanse",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
        ],
    )

    exit_code = process_companies.main()
    assert exit_code == 0

    output = pl.read_parquet(
        layer_fixture_dir("gb", layer="cleansed")
        / "primary"
        / "jurisdiction_code=gb"
        / "part-00001.parquet"
    )
    row = output.row(0, named=True)
    assert row["company_type"] == "investment company"
    assert all(
        column_name in output.columns for column_name in CLEANSED_MERGE_REQUIRED_COLUMNS
    )
    # Cleanse maps canonical partitions 1:1 and stages nothing.
    assert not (layer_fixture_dir("gb", layer="cleansed") / "chunks").exists()


def test_process_script_cleanse_clears_stale_cleansed_outputs(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    canonical_dir = (
        layer_fixture_dir("fr", layer="canonical")
        / "2026-06-15"
        / "primary"
        / "jurisdiction_code=fr"
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["fr:sample"],
            "jurisdiction_code": ["fr"],
            "name": ["EXEMPLE"],
            "CompanyCategory": ["6540"],
        }
    )
    source.write_parquet(canonical_dir / "part-00001.parquet")

    stale_file = layer_fixture_dir("fr", layer="cleansed") / "stale-999.parquet"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": [1]}).write_parquet(stale_file)

    monkeypatch.setattr(
        "sys.argv",
        [
            "process_companies.py",
            "--systems",
            "fr",
            "--processes",
            "cleanse",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
            "--force",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    assert not stale_file.exists()
    assert not (layer_fixture_dir("fr", layer="cleansed") / "chunks").exists()
    merged = pl.read_parquet(
        layer_fixture_dir("fr", layer="cleansed")
        / "primary"
        / "jurisdiction_code=fr"
        / "part-00001.parquet"
    )
    assert all(
        column_name in merged.columns for column_name in CLEANSED_MERGE_REQUIRED_COLUMNS
    )
    assert (
        layer_fixture_dir("fr", layer="cleansed")
        / "primary"
        / "jurisdiction_code=fr"
        / "part-00001.parquet"
    ).exists()


def test_process_script_cleanse_without_force_keeps_existing_cleansed_outputs(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    canonical_dir = (
        layer_fixture_dir("fr", layer="canonical")
        / "2026-06-15"
        / "primary"
        / "jurisdiction_code=fr"
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["fr:sample"],
            "jurisdiction_code": ["fr"],
            "name": ["EXEMPLE"],
            "CompanyCategory": ["6540"],
        }
    )
    source.write_parquet(canonical_dir / "part-00001.parquet")

    # Named to match the real merged-view naming convention
    # ({system_code}-*.parquet at cleansed/'s own top level) and written
    # after the canonical input, so the freshness check correctly treats
    # cleanse as already up-to-date and skips -- proving a non-force run
    # leaves existing output alone rather than reprocessing it.
    stale_file = layer_fixture_dir("fr", layer="cleansed") / "fr-001.parquet"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": [1]}).write_parquet(stale_file)

    monkeypatch.setattr(
        "sys.argv",
        [
            "process_companies.py",
            "--systems",
            "fr",
            "--processes",
            "cleanse",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    assert stale_file.exists()
    assert (layer_fixture_dir("fr", layer="cleansed") / "fr-001.parquet").exists()


def test_cleanse_refuses_a_flat_unpartitioned_canonical_snapshot(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    """A flat canonical snapshot for a partitioning system predates the
    partitioned canonical layout.

    Mapping it 1:1 would replace the partitioned cleansed view with a flat
    one and break every consumer of `jurisdiction_code=*/`, so it must fail
    loudly rather than produce differently-shaped output.
    """
    canonical_dir = layer_fixture_dir("fr", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr:1"],
            "jurisdiction_code": ["fr"],
            "name": ["EXEMPLE"],
            "CompanyCategory": ["6540"],
        }
    ).write_parquet(canonical_dir / "fr-001.parquet")

    monkeypatch.setattr(
        "sys.argv",
        [
            "process_companies.py",
            "--systems",
            "fr",
            "--processes",
            "cleanse",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
            "--force",
        ],
    )

    assert process_companies.main() == 1
    assert not any((layer_fixture_dir("fr", layer="cleansed")).glob("*.parquet"))


@pytest.mark.integration
def test_cleanse_ignores_the_canonical_companion_families(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    """The `duplicates` and `names` companions sit alongside the entity view
    (one at the snapshot's top level, one in its own subdirectory) and are
    each a different row shape keyed off `system_uri` rather than a company
    record. Cleansing either would put rows into the cleansed view that
    Canonical deliberately excluded or multiply it per name variant.

    Which file name belongs to which family is `workspace.data_file_naming`'s
    own contract, pinned directly and cheaply in `test_data_file_naming.py`;
    this is the one path proving the real script still wires that contract
    into an end-to-end cleanse run.
    """
    snapshot_dir = layer_fixture_dir("fr", layer="canonical") / "2026-06-15"
    partition_dir = snapshot_dir / "primary" / "jurisdiction_code=fr"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr:1"],
            "jurisdiction_code": ["fr"],
            "name": ["KEPT"],
            "CompanyCategory": ["6540"],
        }
    ).write_parquet(partition_dir / "part-00001.parquet")
    pl.DataFrame(
        {
            "system_uri": ["fr:1"],
            "jurisdiction_code": ["fr"],
            "name": ["DROPPED"],
            "CompanyCategory": ["6540"],
            "dropped_reason": ["duplicate_system_uri"],
        }
    ).write_parquet(snapshot_dir / f"fr-{COMPANION_DUPLICATES}-001.parquet")
    names_dir = snapshot_dir / "names"
    names_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr:1"],
            "jurisdiction_code": ["fr"],
            "name": ["A VARIANT"],
            "name_type": ["previous"],
        }
    ).write_parquet(names_dir / f"fr-{COMPANION_NAMES}-001.parquet")

    monkeypatch.setattr(
        "sys.argv",
        [
            "process_companies.py",
            "--systems",
            "fr",
            "--processes",
            "cleanse",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
            "--force",
        ],
    )

    assert process_companies.main() == 0

    cleansed_root = layer_fixture_dir("fr", layer="cleansed")
    cleansed = pl.read_parquet(sorted(cleansed_root.glob("**/*.parquet")))
    assert cleansed["name"].to_list() == ["KEPT"]


def test_merge_preserves_source_name_rather_than_the_cleansed_one(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    """`name` in the merged output is the source value, not the selected cleansed
    column rewritten over it.

    The existing contract assertions only check the column is *present*, so a change
    that populated `name` from `name_cleansed` would keep them green. This pins the
    value, and asserts the cleansed form genuinely differs so the check cannot pass
    by the cleanser having done nothing.
    """
    source_name = "ÉXEMPLE HOLDINGS S.A.R.L."
    canonical_dir = (
        layer_fixture_dir("fr", layer="canonical")
        / "2026-06-15"
        / "primary"
        / "jurisdiction_code=fr"
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr:preserve"],
            "jurisdiction_code": ["fr"],
            "name": [source_name],
            "CompanyCategory": ["6540"],
        }
    ).write_parquet(canonical_dir / "part-00001.parquet")

    monkeypatch.setattr(
        "sys.argv",
        [
            "process_companies.py",
            "--systems",
            "fr",
            "--processes",
            "cleanse",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
            "--force",
        ],
    )

    assert process_companies.main() == 0

    merged = pl.read_parquet(
        layer_fixture_dir("fr", layer="cleansed")
        / "primary"
        / "jurisdiction_code=fr"
        / "part-00001.parquet"
    )
    row = merged.row(0, named=True)
    assert row["name"] == source_name
    assert row["name_cleansed"] != source_name


def test_cleanse_mirrors_the_canonical_family_and_partition_layout(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    """Cleanse mirrors its input rather than deciding a shape of its own,
     so a canonical input at `primary/jurisdiction_code=fr/part-00001.parquet`
     lands at the identical relative path under `cleansed/`. The cleansed
     layer has a `primary/` because `canonical/` does -- no stage chooses it
    .
    """
    canonical_dir = (
        layer_fixture_dir("fr", layer="canonical")
        / "2026-06-15"
        / "primary"
        / "jurisdiction_code=fr"
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr:1"],
            "jurisdiction_code": ["fr"],
            "name": ["EXEMPLE"],
            "CompanyCategory": ["6540"],
        }
    ).write_parquet(canonical_dir / "part-00001.parquet")

    monkeypatch.setattr(
        "sys.argv",
        [
            "process_companies.py",
            "--systems",
            "fr",
            "--processes",
            "cleanse",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
            "--force",
        ],
    )

    assert process_companies.main() == 0

    cleansed_root = layer_fixture_dir("fr", layer="cleansed")
    assert (
        cleansed_root / "primary" / "jurisdiction_code=fr" / "part-00001.parquet"
    ).exists()
