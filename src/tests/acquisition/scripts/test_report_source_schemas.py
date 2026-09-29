from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from scripts import report_source_schemas


def test_a_shard_is_grouped_by_its_family_not_its_number() -> None:
    """The entity shards of a snapshot are one family and each companion its
    own, so a system's sidecar is not tabled as though it were its entities."""
    families = {
        report_source_schemas.family_of(Path(name), system="wikidata")
        for name in (
            "wikidata-001.parquet",
            "wikidata-002.parquet",
            "wikidata-names-001.parquet",
            "wikidata-sidecar-001.parquet",
        )
    }

    assert families == {
        "wikidata (entities)",
        "wikidata-names",
        "wikidata-sidecar",
    }


def test_a_column_carries_the_canonical_field_its_catalog_entry_maps_it_to(
    tmp_path: Path,
) -> None:
    """Which source column feeds which canonical field is the catalog's to
    say, identifier candidates included; an unmapped column has none."""
    catalog = tmp_path / report_source_schemas.CATALOG_DIR / "ie.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "canonical_source_column_aliases": {"company_num": "company_number"},
                "system_field_candidates": {"name": ["company_name"]},
                "system_uri_identifier_candidates": ["company_number"],
            }
        ),
        encoding="utf-8",
    )

    mapped = report_source_schemas.canonical_fields("ie", checkout=tmp_path)

    # The alias renames the column before anything reads it, so the
    # identifier candidate naming `company_number` is this same column.
    assert mapped["company_num"] == "company_number, system_uri"
    assert mapped["company_name"] == "name"
    # A canonical field nothing claims is fed by a column of its own name.
    assert mapped["current_status"] == "current_status"
    # One the catalog maps elsewhere is not.
    assert "name" not in mapped
    assert report_source_schemas.canonical_fields("absent", checkout=tmp_path) == {}


def test_a_family_reports_every_files_rows_and_the_samples_own_figures(
    tmp_path: Path,
) -> None:
    """`Rows` counts every file; the null, distinct and example figures are
    the sample's alone, so the document never implies a full scan."""
    first = tmp_path / "ie-001.parquet"
    second = tmp_path / "ie-002.parquet"
    pl.DataFrame({"name": ["a", "b", None], "n": [1, 2, 3]}).write_parquet(first)
    pl.DataFrame({"name": ["c"], "n": [4]}).write_parquet(second)

    facts = report_source_schemas.read_family(
        [first, second], name="ie (entities)", sample_rows=3, mapped={"name": "name"}
    )

    assert (facts.files, facts.rows, facts.sampled) == (2, 4, 3)
    name, number = facts.columns
    assert (name.name, name.dtype, name.canonical_field) == ("name", "String", "name")
    assert name.null_share == 1 / 3
    assert name.example == "a"
    assert (number.name, number.null_share, number.canonical_field) == ("n", 0.0, "")
