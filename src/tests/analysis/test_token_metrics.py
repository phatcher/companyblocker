from __future__ import annotations

import polars as pl
import pytest

from analysis.run_manifest import load_run_manifest
from analysis.token_metrics import (
    _system_parquet_files,
    build_phase1_token_metrics,
    run_phase1_token_analysis,
)
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots


def _write_system_data(
    roots: WorkspaceRoots, system: str, rows: list[dict[str, object]]
) -> None:
    target = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / f"{system}-001.parquet")


def test_system_parquet_files_excludes_name_variant_companion_files(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(workspace_roots, "gleif", [{"name_cleansed": "alpha ltd"}])
    # A names-corpus companion file is one row per name-variant, not one row
    # per entity -- must never be picked up as a regular tokenized shard.
    names_dir = system_layer_dir(workspace_roots, "gleif", layer=TOKENIZED_LAYER_NAME)
    pl.DataFrame({"name_cleansed": ["previous name"]}).write_parquet(
        names_dir / "gleif-names-001.parquet"
    )

    files = _system_parquet_files(workspace_roots, "gleif")

    assert [f.name for f in files] == ["gleif-001.parquet"]


def test_system_parquet_files_finds_partitioned_output(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A real tokenized/ layer mirrors cleansed/'s own shape and is
    partitioned under jurisdiction_code=*/ for most systems -- the old flat
    f"{system}-*.parquet" top-level glob never matched it.
    """
    partition_dir = layer_partition_dir(
        system_layer_dir(workspace_roots, "gleif", layer=TOKENIZED_LAYER_NAME),
        value="gb",
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name_cleansed": ["alpha ltd"]}).write_parquet(
        partition_dir / "part-00001.parquet"
    )

    files = _system_parquet_files(workspace_roots, "gleif")

    assert [f.name for f in files] == ["part-00001.parquet"]


def test_build_phase1_token_metrics_computes_country_and_global_stats(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gb",
        [
            {"country_tokens": ["ACME", "LTD"], "global_tokens": ["ACME", "LTD"]},
            {"country_tokens": ["ACME", "GROUP"], "global_tokens": ["ACME", "GROUP"]},
        ],
    )
    _write_system_data(
        workspace_roots,
        "ie",
        [
            {"country_tokens": ["BETA", "LTD"], "global_tokens": ["BETA", "LTD"]},
        ],
    )

    country_df, global_df, summary_df = build_phase1_token_metrics(
        workspace_roots,
        systems=["gb", "ie"],
        country_token_column="country_tokens",
        global_token_column="global_tokens",
    )

    assert country_df.height > 0
    assert global_df.height > 0
    assert summary_df.height == 2

    gb_acme = country_df.filter(
        (pl.col("system") == "gb") & (pl.col("token") == "ACME")
    )
    assert gb_acme.height == 1
    assert gb_acme.item(0, "tf") == 2
    assert gb_acme.item(0, "df") == 2
    assert gb_acme.item(0, "tf_ratio_to_global") is not None
    assert gb_acme.item(0, "idf") is not None
    assert gb_acme.item(0, "tfidf") is not None

    ltd_global = global_df.filter(pl.col("token") == "LTD")
    assert ltd_global.height == 1
    assert ltd_global.item(0, "systems_present") == 2


@pytest.mark.graphics
def test_run_phase1_token_analysis_writes_artifacts(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "fr",
        [
            {
                "country_tokens": ["ALPHA", "SA"],
                "global_tokens": ["ALPHA", "SA"],
            },
            {
                "country_tokens": ["ALPHA", "HOLDING"],
                "global_tokens": ["ALPHA", "HOLDING"],
            },
        ],
    )

    paths = run_phase1_token_analysis(
        workspace_roots,
        run_date="2026-06-21",
        systems=["fr"],
        top_n=5,
    )

    assert paths.country_stats_path.exists()
    assert paths.global_stats_path.exists()
    assert paths.report_path.exists()
    assert paths.top_tokens_plot_path.exists()
    assert paths.top_tokens_ex_legal_form_plot_path.exists()
    assert paths.top_tfidf_plot_path.exists()
    assert paths.top_tfidf_ex_legal_form_plot_path.exists()
    assert paths.stoplist_candidates_path.exists()
    assert paths.stoplist_candidates_csv_path.exists()
    assert paths.overview_plot_path.exists()
    assert paths.run_manifest_path.exists()

    manifest = load_run_manifest(workspace_roots, "2026-06-21")
    phase_entry = manifest["phases"]["phase1_token_metrics"]
    assert phase_entry["run_date"] == "2026-06-21"
    assert phase_entry["systems"] == ["fr"]
    assert phase_entry["parameters"]["top_n"] == 5
    assert (
        phase_entry["outputs"]["country_stats"] == paths.country_stats_path.as_posix()
    )
    country_rows_written = pl.read_parquet(paths.country_stats_path).height
    assert phase_entry["row_counts"]["country_rows"] == country_rows_written

    report = paths.report_path.read_text(encoding="utf-8")
    assert "# Token Metrics Summary" in report
    assert "Top 5 Tokens By Country" in report
    assert "Excluding Legal Form Tokens" in report
    assert "Stoplist Candidates" in report
    assert "Top 5 TF-IDF Tokens By Country" in report
    assert "FR" in report

    stoplist_df = pl.read_parquet(paths.stoplist_candidates_path)
    assert "token" in stoplist_df.columns
    assert "coverage_ratio" in stoplist_df.columns
    assert "token_type" in stoplist_df.columns


@pytest.mark.graphics
def test_run_phase1_token_analysis_duckdb_handles_extra_columns(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gb",
        [
            {
                "country_tokens": ["ACME", "LTD"],
                "global_tokens": ["ACME", "LTD"],
                "extra_col": "x",
            },
            {
                "country_tokens": ["ACME", "GROUP"],
                "global_tokens": ["ACME", "GROUP"],
                "extra_col": "y",
            },
        ],
    )

    paths = run_phase1_token_analysis(
        workspace_roots,
        run_date="2026-06-22",
        systems=["gb"],
        engine="duckdb",
    )

    country_df = pl.read_parquet(paths.country_stats_path)
    global_df = pl.read_parquet(paths.global_stats_path)

    assert country_df.height > 0
    assert global_df.height > 0
    assert "idf" in country_df.columns
    assert "tfidf" in country_df.columns


def test_build_phase1_token_metrics_default_input_file_excludes_sample(
    workspace_roots: WorkspaceRoots,
) -> None:
    target = system_layer_dir(workspace_roots, "gb", layer=TOKENIZED_LAYER_NAME)
    target.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        [
            {"country_tokens": ["MAIN"], "global_tokens": ["MAIN"]},
        ]
    ).write_parquet(target / "gb-001.parquet")

    pl.DataFrame(
        [
            {"country_tokens": ["SAMPLE"], "global_tokens": ["SAMPLE"]},
        ]
    ).write_parquet(target / "sample.parquet")

    country_df, global_df, _ = build_phase1_token_metrics(
        workspace_roots,
        systems=["gb"],
        input_file=None,
    )

    assert country_df.filter(pl.col("token") == "MAIN").height == 1
    assert country_df.filter(pl.col("token") == "SAMPLE").height == 0
    assert global_df.filter(pl.col("token") == "SAMPLE").height == 0
