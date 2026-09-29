import json

import polars as pl
import pytest

from acquisition.cleanser_orchestrate import materialize_cleansed_merge
from validation.config import ValidationRunConfig
from validation.prepared_dataset import (
    list_prepared_countries,
    load_prepared_country_frame,
    prepare_system_dataset,
)
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    data_root,
    system_layer_dir,
)
from workspace.roots import WorkspaceRoots


def _write_cleansed_chunk(
    roots: WorkspaceRoots, system: str, rows: list[dict[str, object]], file_name: str
) -> None:
    source_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME) / "chunks"
    source_dir.mkdir(parents=True, exist_ok=True)
    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        cleansed_value = str(enriched.get("name_cleansed") or "")
        enriched.setdefault("name", cleansed_value)
        enriched.setdefault("name_cleansed_basic", cleansed_value)
        enriched_rows.append(enriched)
    pl.DataFrame(enriched_rows).write_parquet(source_dir / file_name)


def _materialize_cleansed_merge(roots: WorkspaceRoots, system: str) -> None:
    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    source_dir = cleansed_dir / "chunks"
    materialize_cleansed_merge(
        source_dir=source_dir,
        output_dir=cleansed_dir,
        system=system,
        name_col="name_cleansed",
        rows_per_file=2,
        force_rebuild=True,
    )


def _build_config(roots: WorkspaceRoots, *, rows_per_file: int) -> ValidationRunConfig:
    return ValidationRunConfig(
        roots=roots,
        run_date=None,
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        source_name_col="name_cleansed",
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=data_root(roots) / "validation" / "outputs",
        prepared_base_dir=None,
        prepared_rows_per_file=rows_per_file,
        include_reverse_direction=False,
        exception_policy_path=None,
    )


@pytest.mark.integration
def test_prepare_system_dataset_writes_hive_partitions_chunked(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:gb:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gleif:gb:2",
                "name_cleansed": "beta holdings",
                "jurisdiction_code": "gb",
            },
        ],
        "gleif-001.parquet",
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:gb:3",
                "name_cleansed": "charlie services",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gleif:fr:1",
                "name_cleansed": "",
                "jurisdiction_code": "fr",
            },
        ],
        "gleif-002.parquet",
    )
    _materialize_cleansed_merge(workspace_roots, "gleif")

    config = _build_config(workspace_roots, rows_per_file=2)
    stats = prepare_system_dataset(config=config, system="gleif")

    base = system_layer_dir(workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME)
    assert base.exists()
    assert (base / "_prepared_metadata.json").exists()

    gb_partition = base / "primary" / "jurisdiction_code=gb"
    assert gb_partition.exists()
    assert len(list(gb_partition.glob("*.parquet"))) == 2

    countries = list_prepared_countries(stats)
    assert countries == {"fr", "gb"}

    gb_frame = load_prepared_country_frame(stats=stats, country="gb")
    assert gb_frame.height == 3
    assert "country" not in gb_frame.columns

    fr_frame = load_prepared_country_frame(stats=stats, country="fr")
    assert fr_frame.height == 1

    assert stats.total_rows_by_country["gb"] == 3
    assert stats.total_rows_by_country["fr"] == 1
    assert stats.filtered_out_rows_by_country["fr"] == 0
    assert stats.kept_rows_by_country["gb"] == 3

    metadata = json.loads(
        (base / "_prepared_metadata.json").read_text(encoding="utf-8")
    )
    assert list(metadata["total_rows_by_country"].keys()) == sorted(
        metadata["total_rows_by_country"].keys()
    )
    assert "files_written_by_country" not in metadata


@pytest.mark.integration
def test_prepare_system_dataset_materializes_all_jurisdictions(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:gb:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gleif:fr:1",
                "name_cleansed": "beta holdings",
                "jurisdiction_code": "fr",
            },
        ],
        "gleif-001.parquet",
    )
    _materialize_cleansed_merge(workspace_roots, "gleif")

    config = _build_config(workspace_roots, rows_per_file=10)
    stats = prepare_system_dataset(config=config, system="gleif")

    base = system_layer_dir(workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME)
    assert (base / "primary" / "jurisdiction_code=gb").exists()
    assert (base / "primary" / "jurisdiction_code=fr").exists()
    assert list_prepared_countries(stats) == {"gb", "fr"}


def test_prepare_system_dataset_reuses_existing_partitions_without_reshard(
    workspace_roots: WorkspaceRoots,
) -> None:
    base = system_layer_dir(workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME)
    partition = base / "primary" / "jurisdiction_code=gb"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:gb:1"],
            "name": ["acme limited"],
        }
    ).write_parquet(partition / "part-00001.parquet")
    (base / "_prepared_metadata.json").write_text(
        json.dumps(
            {
                "system": "gleif",
                "rows_per_file": 1000000,
                "total_rows_by_country": {"gb": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = _build_config(workspace_roots, rows_per_file=10)
    stats = prepare_system_dataset(config=config, system="gleif")

    assert stats.system_dir == base
    assert list_prepared_countries(stats) == {"gb"}
    assert stats.kept_rows_by_country["gb"] == 1


def test_prepare_system_dataset_requires_existing_partitions(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = _build_config(workspace_roots, rows_per_file=10)

    with pytest.raises(FileNotFoundError, match="No prepared parquet files found"):
        prepare_system_dataset(config=config, system="gleif")


@pytest.mark.integration
def test_prepare_system_dataset_respects_max_rows(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {"system_uri": "gleif:1", "name_cleansed": "a", "jurisdiction_code": "gb"},
            {"system_uri": "gleif:2", "name_cleansed": "b", "jurisdiction_code": "gb"},
            {"system_uri": "gleif:3", "name_cleansed": "c", "jurisdiction_code": "gb"},
        ],
        "gleif-001.parquet",
    )
    _materialize_cleansed_merge(workspace_roots, "gleif")

    config = _build_config(workspace_roots, rows_per_file=10)
    stats = prepare_system_dataset(config=config, system="gleif")

    frame = load_prepared_country_frame(stats=stats, country="gb")
    assert frame.height == 3
    assert stats.kept_rows_by_country["gb"] == 3


def test_prepare_system_dataset_prefers_matched_over_cleansed_merge(
    workspace_roots: WorkspaceRoots,
) -> None:
    matched_base = system_layer_dir(workspace_roots, "gleif", layer=MATCHED_LAYER_NAME)
    matched_partition = matched_base / "primary" / "jurisdiction_code=gb"
    matched_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:matched:1"],
            "name": ["matched row"],
            "jurisdiction_code": ["gb"],
        }
    ).write_parquet(matched_partition / "part-00001.parquet")

    cleansed_base = system_layer_dir(
        workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME
    )
    cleansed_partition = cleansed_base / "primary" / "jurisdiction_code=gb"
    cleansed_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif:cleanse:1"],
            "name": ["cleansed row"],
            "jurisdiction_code": ["gb"],
        }
    ).write_parquet(cleansed_partition / "part-00001.parquet")

    config = _build_config(workspace_roots, rows_per_file=10)
    stats = prepare_system_dataset(config=config, system="gleif")

    assert stats.system_dir == matched_base
