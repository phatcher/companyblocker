import polars as pl

from validation.config import ValidationRunConfig
from validation.loader import build_directional_frames, list_validation_input_files
from workspace.artifact_layout import validation_artifact_root
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.roots import WorkspaceRoots


def _write_cleansed_chunk(
    roots: WorkspaceRoots, system: str, rows: list[dict[str, object]]
) -> None:
    target = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / f"{system}-001.parquet")


def _write_matched_chunk(
    roots: WorkspaceRoots, system: str, rows: list[dict[str, object]]
) -> None:
    target = (
        system_layer_dir(roots, system, layer=MATCHED_LAYER_NAME)
        / "jurisdiction_code=gb"
    )
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / "part-00001.parquet")


def test_list_validation_input_files_excludes_name_variant_companion_files(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_cleansed_chunk(workspace_roots, "gleif", [{"name_cleansed": "alpha ltd"}])
    # A names-corpus companion file is one row per name-variant, not one row
    # per entity -- must never be picked up as a regular cleansed shard.
    names_dir = system_layer_dir(workspace_roots, "gleif", layer=CLEANSED_LAYER_NAME)
    pl.DataFrame({"name_cleansed": ["previous name"]}).write_parquet(
        names_dir / "gleif-names-001.parquet"
    )

    files = list_validation_input_files(
        roots=workspace_roots, system="gleif", run_date=None
    )

    assert [f.name for f in files] == ["gleif-001.parquet"]


def _base_config(
    roots: WorkspaceRoots, run_date: str, countries: tuple[str, ...] | None = None
) -> ValidationRunConfig:
    return ValidationRunConfig(
        roots=roots,
        run_date=run_date,
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=countries,
        source_name_col="name_cleansed",
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=validation_artifact_root(roots),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
    )


def test_build_directional_frames_filters_source_to_target_jurisdictions(
    workspace_roots: WorkspaceRoots,
) -> None:
    run_date = "2026-07-07"
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
                "name_cleansed": "acme france",
                "jurisdiction_code": "fr",
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
        ],
    )

    source_frame, target_frame, source_filter_stats = build_directional_frames(
        config=_base_config(workspace_roots, run_date),
        source_system="gleif",
        target_system="gb",
    )

    assert set(source_frame.get_column("country").to_list()) == {"gb"}
    assert set(target_frame.get_column("country").to_list()) == {"gb"}
    assert source_frame.height == 1
    assert source_filter_stats["gb"]["source_total_rows"] == 1
    assert source_filter_stats["gb"]["source_filtered_out_rows"] == 0


def test_build_directional_frames_still_applies_user_country_filter_before_target_gate(
    workspace_roots: WorkspaceRoots,
) -> None:
    run_date = "2026-07-07"
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
                "name_cleansed": "acme france",
                "jurisdiction_code": "fr",
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
        ],
    )

    source_frame, target_frame, source_filter_stats = build_directional_frames(
        config=_base_config(workspace_roots, run_date, countries=("gb", "fr")),
        source_system="gleif",
        target_system="gb",
    )

    assert set(target_frame.get_column("country").to_list()) == {"gb"}
    assert set(source_frame.get_column("country").to_list()) == {"gb"}
    assert source_filter_stats["gb"]["source_total_rows"] == 1
    assert source_filter_stats["gb"]["source_filtered_out_rows"] == 0


def test_build_directional_frames_prefers_matched_inputs(
    workspace_roots: WorkspaceRoots,
) -> None:
    run_date = "2026-07-07"
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:cleanse:1",
                "name_cleansed": "from cleanse",
                "jurisdiction_code": "gb",
            },
        ],
    )
    _write_matched_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:matched:1",
                "name_cleansed": "from matched",
                "jurisdiction_code": "gb",
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
        ],
    )

    source_frame, _, _ = build_directional_frames(
        config=_base_config(workspace_roots, run_date),
        source_system="gleif",
        target_system="gb",
    )

    assert source_frame.height == 1
    assert source_frame.get_column("system_uri").to_list() == ["gleif:matched:1"]
