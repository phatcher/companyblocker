import json
from pathlib import Path

import polars as pl
import pytest

from blocking.loader import (
    load_country_frame,
    load_dataset_descriptor,
    load_name_variant_frame,
)
from blocking.truth import ColumnTruth
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    layer_directory,
    perturbed_dataset_dir,
)
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots


def _write_partition(
    layer_dir_for,
    *,
    system: str,
    layer: str,
    country: str,
    rows: list[dict[str, object]],
) -> Path:
    layer_dir = layer_dir_for(system, layer=layer)
    if layer == "canonical":
        # Canonical is dated: a load reads its latest snapshot.
        layer_dir = layer_dir / "2026-01-01"
    partition_dir = layer_partition_dir(layer_dir, value=country)
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(partition_dir / "part-00001.parquet")
    return layer_dir


def _write_match_metadata(
    layer_dir: Path, *, source_system: str, target_systems: list[str]
) -> None:
    metadata = {
        "source_system": source_system,
        "target_systems": target_systems,
    }
    (layer_dir / "_match_metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )


def test_load_dataset_descriptor_resolves_matched_by_default(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme", "match_uri": "gb:1"}],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])

    descriptor = load_dataset_descriptor(roots=workspace_roots, system="gleif")

    assert descriptor.layer == "matched"
    assert descriptor.has_ground_truth is True
    assert descriptor.matched_target_systems == ("gb",)
    assert descriptor.available_countries == ("gb",)
    assert descriptor.system_dir == matched_dir


def test_load_dataset_descriptor_requires_matched_by_default(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="cleansed",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme"}],
    )

    with pytest.raises(FileNotFoundError, match="gleif"):
        load_dataset_descriptor(roots=workspace_roots, system="gleif")


def test_load_dataset_descriptor_reads_canonical_when_truth_not_required(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    canonical_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme"}],
    )

    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gleif", require_ground_truth=False
    )

    assert descriptor.layer == "canonical"
    assert descriptor.has_ground_truth is False
    assert descriptor.matched_target_systems == ()
    assert descriptor.system_dir == canonical_dir


def test_load_dataset_descriptor_column_truth_takes_a_canonical_layer_carrying_it(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """Under a column rule the layer name says nothing about truth: a
    canonical snapshot whose rows carry the column supplies it, with no
    matched metadata to load."""
    canonical_dir = _write_partition(
        layer_fixture_dir,
        system="ie",
        layer="canonical",
        country="ie",
        rows=[{"system_uri": "name://a", "source_uri": "ie://1", "name": "acme"}],
    )

    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="ie", truth=ColumnTruth("source_uri")
    )

    assert descriptor.layer == "canonical"
    assert descriptor.has_ground_truth is True
    assert descriptor.matched_target_systems == ()
    assert descriptor.system_dir == canonical_dir


def test_load_dataset_descriptor_column_truth_refuses_a_layer_without_the_column(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """The refusal names both the missing truth column and the location it
    was missing from, not just the resolver's repr."""
    canonical_dir = _write_partition(
        layer_fixture_dir,
        system="ie",
        layer="canonical",
        country="ie",
        rows=[{"system_uri": "ie://1", "name": "acme"}],
    )

    with pytest.raises(ValueError) as excinfo:
        load_dataset_descriptor(
            roots=workspace_roots, system="ie", truth=ColumnTruth("source_uri")
        )

    message = str(excinfo.value)
    assert "'source_uri'" in message
    assert str(canonical_dir) in message


def _write_perturbed_dataset(
    roots: WorkspaceRoots, rows: list[dict[str, object]]
) -> Path:
    """One materialized perturbed dataset where validation's materializer
    writes it: its own directory, one cleansed layer, partitioned by country."""
    layer_dir = layer_directory(
        perturbed_dataset_dir(
            roots=roots, source_system="ie", profile_id="p1", version="v1", seed=1
        ),
        CLEANSED_LAYER_NAME,
    )
    partition_dir = layer_partition_dir(layer_dir, value="ie")
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(partition_dir / "part-00000.parquet")
    return layer_dir


def test_load_dataset_descriptor_resolves_a_perturbed_selector(
    workspace_roots: WorkspaceRoots,
) -> None:
    layer_dir = _write_perturbed_dataset(
        workspace_roots,
        [
            {
                "system_uri": "perturbed://ie/p1/1/a",
                "source_uri": "ie://1",
                "match_uri": None,
                "name": "acme limted",
                "jurisdiction_code": "ie",
            }
        ],
    )

    descriptor = load_dataset_descriptor(
        roots=workspace_roots,
        system="perturbed://ie/p1/v1/1",
        truth=ColumnTruth("source_uri"),
    )

    assert descriptor.system == "perturbed://ie/p1/v1/1"
    assert descriptor.system_dir == layer_dir
    assert descriptor.layer == "cleansed"
    assert descriptor.has_ground_truth is True
    assert descriptor.available_countries == ("ie",)


def test_load_dataset_descriptor_perturbed_selector_under_the_default_rule_says_why(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A perturbed set carries no `match_uri`, so the default rule cannot
    score it; the refusal names the column rule that can."""
    _write_perturbed_dataset(
        workspace_roots,
        [{"system_uri": "perturbed://ie/p1/1/a", "source_uri": "ie://1", "name": "a"}],
    )

    with pytest.raises(ValueError, match="--match-col source_uri"):
        load_dataset_descriptor(roots=workspace_roots, system="perturbed://ie/p1/v1/1")


def test_load_dataset_descriptor_never_reads_a_cleansed_layer(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A load without ground truth is a target: it reads its canonical
    snapshot even with a cleansed layer beside it, and a system holding only
    a cleansed layer does not load, since the run derives every name form
    itself and a carried one could only be picked up by mistake."""
    _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="cleansed",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme", "name_cleansed": "stale"}],
    )
    with pytest.raises(FileNotFoundError, match="canonical"):
        load_dataset_descriptor(
            roots=workspace_roots, system="gb", require_ground_truth=False
        )

    canonical_dir = _write_partition(
        layer_fixture_dir,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gb:1", "name": "acme"}],
    )
    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )

    assert descriptor.layer == "canonical"
    assert descriptor.system_dir == canonical_dir
    assert "name_cleansed" not in load_country_frame(descriptor, country="gb").columns


def test_load_dataset_descriptor_falls_back_to_the_latest_canonical_snapshot(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A target loads from its canonical layer, the run deriving its own name
    forms from the raw `name`; the latest dated snapshot is the one read."""
    canonical_dir = layer_fixture_dir("gb", layer=CANONICAL_LAYER_NAME)
    for date in ("2026-01-01", "2026-03-01"):
        partition_dir = layer_partition_dir(canonical_dir / date, value="gb")
        partition_dir.mkdir(parents=True)
        pl.DataFrame({"system_uri": ["gb:1"], "name": [f"acme {date}"]}).write_parquet(
            partition_dir / "part-00001.parquet"
        )

    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gb", require_ground_truth=False
    )

    assert descriptor.layer == "canonical"
    assert descriptor.system_dir == canonical_dir / "2026-03-01"
    assert descriptor.has_ground_truth is False
    assert descriptor.available_countries == ("gb",)


def test_load_dataset_descriptor_requires_match_metadata(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme", "match_uri": "gb:1"}],
    )

    with pytest.raises(ValueError, match="_match_metadata.json"):
        load_dataset_descriptor(roots=workspace_roots, system="gleif")


def _write_names_sidecar(
    layer_dir_for,
    *,
    system: str,
    date: str,
    filename: str,
    rows: list[dict[str, object]],
) -> None:
    sidecar_dir = layer_dir_for(system, layer=CANONICAL_LAYER_NAME) / date
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(sidecar_dir / filename)


def test_load_name_variant_frame_returns_none_without_canonical_dir(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    matched_dir = _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme", "match_uri": "gb:1"}],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    descriptor = load_dataset_descriptor(roots=workspace_roots, system="gleif")

    assert load_name_variant_frame(descriptor) is None


def test_load_name_variant_frame_returns_none_without_matching_names_files(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme"}],
    )
    _write_names_sidecar(
        layer_fixture_dir,
        system="gleif",
        date="2026-01-01",
        filename="gleif-successors-001.parquet",
        rows=[{"system_uri": "gleif:1", "name": "old acme", "name_type": "previous"}],
    )
    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gleif", require_ground_truth=False
    )

    assert load_name_variant_frame(descriptor) is None


def test_load_name_variant_frame_loads_and_concats_shards(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme"}],
    )
    _write_names_sidecar(
        layer_fixture_dir,
        system="gleif",
        date="2026-01-01",
        filename="gleif-names-001.parquet",
        rows=[{"system_uri": "gleif:1", "name": "old acme", "name_type": "previous"}],
    )
    _write_names_sidecar(
        layer_fixture_dir,
        system="gleif",
        date="2026-01-01",
        filename="gleif-names-002.parquet",
        rows=[{"system_uri": "gleif:2", "name": "beta prior", "name_type": "previous"}],
    )
    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gleif", require_ground_truth=False
    )

    variants = load_name_variant_frame(descriptor)

    assert variants is not None
    assert sorted(variants.get_column("name").to_list()) == ["beta prior", "old acme"]


def test_load_name_variant_frame_uses_lexicographically_latest_dated_dir(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_partition(
        layer_fixture_dir,
        system="gleif",
        layer="canonical",
        country="gb",
        rows=[{"system_uri": "gleif:1", "name": "acme"}],
    )
    _write_names_sidecar(
        layer_fixture_dir,
        system="gleif",
        date="2026-01-01",
        filename="gleif-names-001.parquet",
        rows=[{"system_uri": "gleif:1", "name": "stale row", "name_type": "previous"}],
    )
    _write_names_sidecar(
        layer_fixture_dir,
        system="gleif",
        date="2026-06-21",
        filename="gleif-names-001.parquet",
        rows=[{"system_uri": "gleif:1", "name": "fresh row", "name_type": "previous"}],
    )
    descriptor = load_dataset_descriptor(
        roots=workspace_roots, system="gleif", require_ground_truth=False
    )

    variants = load_name_variant_frame(descriptor)

    assert variants is not None
    assert variants.get_column("name").to_list() == ["fresh row"]
