"""Per-system token metrics over tokenized output, with reports and pictures.

`run_phase1_token_analysis` writes metrics, reports and `viz/` under `artifacts/analysis/token_performance/runs/<run_date>/` and records the run as the token-metrics phase in `run_manifest`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import cast

import duckdb
import matplotlib
import polars as pl
from company_cleanse import get_company_type_rules, get_manual_noise_words

from analysis.data_loader import build_projected_parquet_sql
from analysis.report_layout import metrics_dir as _metrics_dir
from analysis.report_layout import reports_dir as _reports_dir
from analysis.report_layout import viz_dir as _viz_dir
from analysis.run_manifest import build_phase_entry, record_phase_run
from analysis.system_discovery import discover_systems_with_tokenized_parquet
from workspace.artifact_layout import analysis_report_run_dir
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.layer_layout import resolve_primary_files
from workspace.roots import WorkspaceRoots

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Phase1Paths:
    run_root: Path
    country_stats_path: Path
    global_stats_path: Path
    report_path: Path
    top_tokens_plot_path: Path
    top_tokens_ex_legal_form_plot_path: Path
    top_tfidf_plot_path: Path
    top_tfidf_ex_legal_form_plot_path: Path
    stoplist_candidates_path: Path
    stoplist_candidates_csv_path: Path
    overview_plot_path: Path
    run_manifest_path: Path


def resolve_phase1_paths(roots: WorkspaceRoots, run_date: str) -> Phase1Paths:
    run_root = analysis_report_run_dir(roots, "token_performance", run_date)
    metrics_dir = _metrics_dir(run_root)
    reports_dir = _reports_dir(run_root)
    viz_dir = _viz_dir(run_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    return Phase1Paths(
        run_root=run_root,
        country_stats_path=metrics_dir / "token_stats_country.parquet",
        global_stats_path=metrics_dir / "token_stats_global.parquet",
        report_path=reports_dir / "token_summary.md",
        top_tokens_plot_path=viz_dir / "token_top_tokens_by_country.png",
        top_tokens_ex_legal_form_plot_path=viz_dir
        / "token_top_tokens_by_country_ex_legal_form.png",
        top_tfidf_plot_path=viz_dir / "token_top_tfidf_by_country.png",
        top_tfidf_ex_legal_form_plot_path=viz_dir
        / "token_top_tfidf_by_country_ex_legal_form.png",
        stoplist_candidates_path=metrics_dir / "token_stoplist_candidates.parquet",
        stoplist_candidates_csv_path=metrics_dir / "token_stoplist_candidates.csv",
        overview_plot_path=viz_dir / "token_country_overview.png",
        run_manifest_path=run_root / "run_manifest.json",
    )


def _system_parquet_files(
    roots: WorkspaceRoots, system: str, input_file: str | None = None
) -> list[Path]:
    tokenized_dir = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    if not tokenized_dir.exists():
        return []
    if input_file:
        selected = tokenized_dir / input_file
        return [selected] if selected.exists() else []
    # The real tokenized/ layer mirrors cleansed/'s shape, which is
    # partitioned under jurisdiction_code=*/ for most systems -- the old
    # flat f"{system}-*.parquet" glob at tokenized_dir's own top level never
    # matched that shape. resolve_primary_files already knows both it and
    # the flat layout.
    return resolve_primary_files(tokenized_dir, system_code=system)


def _discover_systems_from_tokenized(
    roots: WorkspaceRoots, input_file: str | None = None
) -> list[str]:
    return discover_systems_with_tokenized_parquet(roots=roots, input_file=input_file)


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        print(f"[analysis] {message}")


def _resolve_system_files(
    roots: WorkspaceRoots,
    systems: Iterable[str],
    input_file: str | None,
) -> dict[str, list[Path]]:
    return {
        system: _system_parquet_files(roots, system, input_file) for system in systems
    }


def _tokenize_legal_form_value(value: str | None) -> list[str]:
    if value is None:
        return []
    normalized = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]


def _build_legal_form_token_table(
    *,
    systems: Iterable[str],
) -> pl.DataFrame:
    _regex, company_type_mapping = get_company_type_rules()
    canonical_company_types = sorted(
        {
            str(value).strip()
            for value in company_type_mapping.values()
            if str(value).strip()
        }
    )

    canonical_tokens = {
        token
        for company_type in canonical_company_types
        for token in _tokenize_legal_form_value(company_type)
    }

    if not canonical_tokens:
        return pl.DataFrame(
            {"system": [], "token": []}, schema={"system": pl.Utf8, "token": pl.Utf8}
        )

    token_rows: list[dict[str, str]] = []
    for system in systems:
        for token in canonical_tokens:
            token_rows.append({"system": system, "token": token})

    return pl.DataFrame(token_rows)


def _exclude_legal_form_tokens(
    country_df: pl.DataFrame, legal_form_tokens_df: pl.DataFrame
) -> pl.DataFrame:
    if country_df.height == 0 or legal_form_tokens_df.height == 0:
        return country_df
    return country_df.join(legal_form_tokens_df, on=["system", "token"], how="anti")


def _build_noise_word_table() -> pl.DataFrame:
    token_rows: list[dict[str, str]] = []
    scoped_tokens = get_manual_noise_words()
    for scope in ("suffix", "anywhere"):
        for token in scoped_tokens.get(scope, ()):
            normalized = _tokenize_legal_form_value(token)
            for part in normalized:
                token_rows.append({"token": part})

    if not token_rows:
        return pl.DataFrame(
            {"token": [], "token_type_noise": []},
            schema={"token": pl.Utf8, "token_type_noise": pl.Utf8},
        )

    return (
        pl.DataFrame(token_rows)
        .unique()
        .with_columns(pl.lit("noise").alias("token_type_noise"))
    )


def _derive_stoplist_candidates(
    *,
    country_df: pl.DataFrame,
    global_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    legal_form_tokens_df: pl.DataFrame,
    noise_words_df: pl.DataFrame,
    min_coverage_ratio: float,
    min_global_df_pct: float,
) -> pl.DataFrame:
    if country_df.height == 0 or global_df.height == 0:
        return pl.DataFrame(
            {
                "token": [],
                "systems_present": [],
                "coverage_ratio": [],
                "global_df_pct": [],
                "global_tf_pct": [],
                "mean_tfidf": [],
                "std_tfidf": [],
                "token_type": [],
            }
        )

    systems_count = (
        summary_df.height
        if summary_df.height > 0
        else country_df.select(pl.col("system").n_unique()).item()
    )
    systems_count = max(int(systems_count), 1)

    tfidf_stats = country_df.group_by("token").agg(
        pl.col("tfidf").mean().alias("mean_tfidf"),
        pl.col("tfidf").std().fill_null(0.0).alias("std_tfidf"),
    )

    legal_form_flags = (
        legal_form_tokens_df.select("token")
        .unique()
        .with_columns(pl.lit("company_type").alias("token_type_legal_form"))
        if legal_form_tokens_df.height > 0
        else pl.DataFrame(
            {"token": [], "token_type_legal_form": []},
            schema={"token": pl.Utf8, "token_type_legal_form": pl.Utf8},
        )
    )

    noise_flags = (
        noise_words_df.select("token", "token_type_noise").unique()
        if noise_words_df.height > 0
        else pl.DataFrame(
            {"token": [], "token_type_noise": []},
            schema={"token": pl.Utf8, "token_type_noise": pl.Utf8},
        )
    )

    candidates = (
        global_df.select(["token", "systems_present", "global_df_pct", "global_tf_pct"])
        .join(tfidf_stats, on="token", how="left")
        .join(legal_form_flags, on="token", how="left")
        .join(noise_flags, on="token", how="left")
        .with_columns(
            (
                pl.col("systems_present").cast(pl.Float64)
                / pl.lit(float(systems_count))
            ).alias("coverage_ratio"),
        )
        .with_columns(
            pl.when(
                pl.col("token_type_legal_form").is_not_null()
                & pl.col("token_type_noise").is_not_null()
            )
            .then(pl.lit("company_type+noise"))
            .when(pl.col("token_type_legal_form").is_not_null())
            .then(pl.lit("company_type"))
            .when(pl.col("token_type_noise").is_not_null())
            .then(pl.lit("noise"))
            .otherwise(pl.lit("other"))
            .alias("token_type")
        )
        .drop(["token_type_legal_form", "token_type_noise"])
        .filter(
            (pl.col("coverage_ratio") >= pl.lit(min_coverage_ratio))
            & (pl.col("global_df_pct") >= pl.lit(min_global_df_pct))
        )
        .sort(
            ["coverage_ratio", "global_df_pct", "global_tf_pct"],
            descending=[True, True, True],
        )
    )

    return candidates


def _input_inventory(system_files: dict[str, list[Path]]) -> tuple[int, int, int]:
    total_files = 0
    total_bytes = 0
    active_systems = 0
    for files in system_files.values():
        if files:
            active_systems += 1
        total_files += len(files)
        for path in files:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
    return active_systems, total_files, total_bytes


def _normalized_tokens_expr(token_column: str) -> pl.Expr:
    return (
        pl.col(token_column)
        .cast(pl.List(pl.Utf8), strict=False)
        .list.eval(pl.element().str.strip_chars().str.to_uppercase())
        .alias("_tokens")
    )


def _compute_token_stats(
    frame: pl.DataFrame, token_column: str
) -> tuple[pl.DataFrame, dict[str, float]]:
    if token_column not in frame.columns:
        return pl.DataFrame({"token": [], "tf": [], "df": []}), {
            "doc_count": float(frame.height),
            "total_tokens": 0.0,
            "unique_tokens": 0.0,
            "avg_tokens_per_doc": 0.0,
            "hapax_ratio": 0.0,
        }

    tokens = frame.select(_normalized_tokens_expr(token_column))

    tf_df = (
        tokens.select(pl.col("_tokens").explode().alias("token"))
        .drop_nulls("token")
        .filter(pl.col("token") != "")
        .group_by("token")
        .len()
        .rename({"len": "tf"})
    )

    df_df = (
        tokens.with_columns(pl.col("_tokens").list.unique().alias("_tokens"))
        .select(pl.col("_tokens").explode().alias("token"))
        .drop_nulls("token")
        .filter(pl.col("token") != "")
        .group_by("token")
        .len()
        .rename({"len": "df"})
    )

    stats = tf_df.join(df_df, on="token", how="full").fill_null(0)
    if stats.height == 0:
        return stats, {
            "doc_count": float(frame.height),
            "total_tokens": 0.0,
            "unique_tokens": 0.0,
            "avg_tokens_per_doc": 0.0,
            "hapax_ratio": 0.0,
        }

    total_tokens = float(stats.select(pl.col("tf").sum()).item())
    unique_tokens = float(stats.height)
    doc_count = float(frame.height)
    hapax_count = float(stats.filter(pl.col("tf") == 1).height)

    return stats, {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "avg_tokens_per_doc": (total_tokens / doc_count) if doc_count > 0 else 0.0,
        "hapax_ratio": (hapax_count / unique_tokens) if unique_tokens > 0 else 0.0,
    }


def _attach_tf_df_percentages(
    stats_df: pl.DataFrame,
    *,
    total_tokens: float,
    doc_count: float,
    tf_alias: str,
    df_alias: str,
) -> pl.DataFrame:
    return stats_df.with_columns(
        (pl.col("tf") / pl.lit(max(total_tokens, 1.0))).alias(tf_alias),
        (pl.col("df") / pl.lit(max(doc_count, 1.0))).alias(df_alias),
    )


def _attach_global_comparison_columns(
    country_df: pl.DataFrame, global_df: pl.DataFrame
) -> pl.DataFrame:
    if country_df.height == 0 or global_df.height == 0:
        return country_df

    compare_cols = global_df.select(
        [pl.col("token"), pl.col("global_tf_pct"), pl.col("global_df_pct")]
    )
    return country_df.join(compare_cols, on="token", how="left").with_columns(
        pl.when(pl.col("global_tf_pct") > 0)
        .then(pl.col("tf_pct") / pl.col("global_tf_pct"))
        .otherwise(None)
        .alias("tf_ratio_to_global"),
        (pl.col("tf_pct") - pl.col("global_tf_pct")).alias("tf_lift_to_global"),
        pl.when(pl.col("global_df_pct") > 0)
        .then(pl.col("df_pct") / pl.col("global_df_pct"))
        .otherwise(None)
        .alias("df_ratio_to_global"),
        (pl.col("df_pct") - pl.col("global_df_pct")).alias("df_lift_to_global"),
    )


def _attach_tfidf(country_df: pl.DataFrame) -> pl.DataFrame:
    if country_df.height == 0:
        return country_df
    return country_df.with_columns(
        (((pl.col("system_doc_count") + 1.0) / (pl.col("df") + 1.0)).log() + 1.0).alias(
            "idf"
        )
    ).with_columns((pl.col("tf").cast(pl.Float64) * pl.col("idf")).alias("tfidf"))


def _build_phase1_token_metrics_duckdb(
    roots: WorkspaceRoots,
    *,
    systems: Iterable[str],
    country_token_column: str,
    global_token_column: str,
    input_file: str | None = None,
    system_files: dict[str, list[Path]] | None = None,
    progress: Callable[[str], None] | None = None,
    verbose_progress: bool = False,
    threads: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    conn = duckdb.connect(database=":memory:")
    try:
        effective_threads = (
            threads if threads is not None and threads > 0 else (os.cpu_count() or 1)
        )
        conn.execute(f"PRAGMA threads={effective_threads}")

        def sql_str(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        def quote_ident(value: str) -> str:
            return '"' + str(value).replace('"', '""') + '"'

        required_loader_columns = [country_token_column, global_token_column]

        relation_sqls: list[str] = []
        present_systems: list[str] = []
        systems_list = list(systems)
        for index, system in enumerate(systems_list, start=1):
            files = (
                list(system_files.get(system, []))
                if system_files is not None
                else _system_parquet_files(roots, system, input_file)
            )
            if not files:
                if verbose_progress:
                    _emit_progress(
                        progress,
                        f"duckdb [{index}/{len(systems_list)}] system={system} files=0 (skip)",
                    )
                continue
            if verbose_progress:
                _emit_progress(
                    progress,
                    f"duckdb [{index}/{len(systems_list)}] system={system} files={len(files)}",
                )
            present_systems.append(system)
            projected_sql = build_projected_parquet_sql(
                input_paths=[str(file) for file in files],
                required_columns=required_loader_columns,
                filter_non_empty_column=None,
            )
            # nosec B608 - every interpolated fragment passes through sql_str /
            # quote_ident, which escapes/quotes the literal or identifier it wraps.
            relation_sql = (
                f"SELECT {sql_str(system)} AS system, "  # nosec B608
                f"COALESCE(src.{quote_ident(country_token_column)}, []) AS country_tokens, "
                f"COALESCE(src.{quote_ident(global_token_column)}, []) AS global_tokens "
                f"FROM ({projected_sql}) src"
            )
            relation_sqls.append(relation_sql)

        if not relation_sqls:
            return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

        union_sql = " UNION ALL ".join(relation_sqls)
        # nosec B608 - union_sql is built entirely from the already-quoted
        # relation_sqls above.
        phase1_base_sql = (
            f"CREATE TEMP TABLE phase1_base AS\n"  # nosec B608
            "SELECT\n"
            "    ROW_NUMBER() OVER () AS doc_id,\n"
            "    system,\n"
            "    CASE WHEN country_tokens IS NULL THEN []::VARCHAR[] ELSE country_tokens END AS country_tokens,\n"
            "    CASE WHEN global_tokens IS NULL THEN []::VARCHAR[] ELSE global_tokens END AS global_tokens\n"
            f"FROM ({union_sql}) src"
        )
        conn.execute(phase1_base_sql)

        summary_df = cast(
            pl.DataFrame,
            pl.from_arrow(
                conn.execute(
                    """
                WITH docs AS (
                    SELECT
                        system,
                        COUNT(*)::DOUBLE AS doc_count
                    FROM phase1_base
                    GROUP BY system
                ), unnested AS (
                    SELECT
                        system,
                        UNNEST(country_tokens) AS token
                    FROM phase1_base
                ), token_counts AS (
                    SELECT
                        system,
                        UPPER(TRIM(token)) AS token,
                        COUNT(*) AS tf
                    FROM unnested
                    WHERE token IS NOT NULL AND LENGTH(TRIM(token)) > 0
                    GROUP BY system, UPPER(TRIM(token))
                )
                SELECT
                    d.system,
                    d.doc_count,
                    COALESCE(SUM(tc.tf), 0)::DOUBLE AS total_tokens,
                    COUNT(tc.token)::DOUBLE AS unique_tokens,
                    CASE WHEN d.doc_count > 0 THEN COALESCE(SUM(tc.tf), 0)::DOUBLE / d.doc_count ELSE 0 END AS avg_tokens_per_doc,
                    CASE WHEN COUNT(tc.token) > 0 THEN SUM(CASE WHEN tc.tf = 1 THEN 1 ELSE 0 END)::DOUBLE / COUNT(tc.token)::DOUBLE ELSE 0 END AS hapax_ratio
                FROM docs d
                LEFT JOIN token_counts tc ON d.system = tc.system
                GROUP BY d.system, d.doc_count
                ORDER BY d.system
                """
                ).to_arrow_table()
            ),
        )

        country_df = cast(
            pl.DataFrame,
            pl.from_arrow(
                conn.execute(
                    """
                WITH unnested AS (
                    SELECT
                        system,
                        UPPER(TRIM(UNNEST(country_tokens))) AS token
                        ,doc_id
                    FROM phase1_base
                ),
                tf AS (
                    SELECT system, token, COUNT(*) AS tf
                    FROM unnested
                    WHERE token IS NOT NULL AND LENGTH(token) > 0
                    GROUP BY system, token
                ),
                df AS (
                    SELECT system, token, COUNT(*) AS df
                    FROM (
                        SELECT DISTINCT system, token, doc_id
                        FROM unnested
                        WHERE token IS NOT NULL AND LENGTH(token) > 0
                    )
                    GROUP BY system, token
                ),
                docs AS (
                    SELECT system, COUNT(*) AS system_doc_count
                    FROM phase1_base
                    GROUP BY system
                ),
                totals AS (
                    SELECT system, SUM(tf) AS system_total_tokens
                    FROM tf
                    GROUP BY system
                )
                SELECT
                    tf.system,
                    tf.token,
                    tf.tf,
                    df.df,
                    docs.system_doc_count,
                    totals.system_total_tokens,
                    CASE WHEN totals.system_total_tokens > 0 THEN tf.tf::DOUBLE / totals.system_total_tokens::DOUBLE ELSE 0 END AS tf_pct,
                    CASE WHEN docs.system_doc_count > 0 THEN df.df::DOUBLE / docs.system_doc_count::DOUBLE ELSE 0 END AS df_pct
                FROM tf
                JOIN df ON tf.system = df.system AND tf.token = df.token
                JOIN docs ON tf.system = docs.system
                JOIN totals ON tf.system = totals.system
                ORDER BY tf.system, tf.tf DESC
                """
                ).to_arrow_table()
            ),
        )

        global_df = cast(
            pl.DataFrame,
            pl.from_arrow(
                conn.execute(
                    """
                WITH unnested AS (
                    SELECT
                        system,
                        UPPER(TRIM(UNNEST(global_tokens))) AS token,
                        doc_id
                    FROM phase1_base
                ),
                tf AS (
                    SELECT token, COUNT(*) AS tf
                    FROM unnested
                    WHERE token IS NOT NULL AND LENGTH(token) > 0
                    GROUP BY token
                ),
                df AS (
                    SELECT token, COUNT(DISTINCT doc_id) AS df
                    FROM unnested
                    WHERE token IS NOT NULL AND LENGTH(token) > 0
                    GROUP BY token
                ),
                system_presence AS (
                    SELECT token, COUNT(DISTINCT system) AS systems_present
                    FROM unnested
                    WHERE token IS NOT NULL AND LENGTH(token) > 0
                    GROUP BY token
                ),
                docs AS (
                    SELECT COUNT(*) AS global_doc_count FROM phase1_base
                ),
                totals AS (
                    SELECT COALESCE(SUM(tf), 0) AS global_total_tokens FROM tf
                )
                SELECT
                    tf.token,
                    tf.tf,
                    df.df,
                    docs.global_doc_count,
                    totals.global_total_tokens,
                    CASE WHEN totals.global_total_tokens > 0 THEN tf.tf::DOUBLE / totals.global_total_tokens::DOUBLE ELSE 0 END AS global_tf_pct,
                    CASE WHEN docs.global_doc_count > 0 THEN df.df::DOUBLE / docs.global_doc_count::DOUBLE ELSE 0 END AS global_df_pct,
                    COALESCE(system_presence.systems_present, 0) AS systems_present
                FROM tf
                JOIN df ON tf.token = df.token
                CROSS JOIN docs
                CROSS JOIN totals
                LEFT JOIN system_presence ON tf.token = system_presence.token
                ORDER BY tf.tf DESC
                """
                ).to_arrow_table()
            ),
        )

        country_df = _attach_global_comparison_columns(country_df, global_df)
        country_df = _attach_tfidf(country_df)
        if country_df.height > 0:
            country_df = country_df.sort(["system", "tf"], descending=[False, True])
        if global_df.height > 0:
            global_df = global_df.sort("tf", descending=True)
        return country_df, global_df, summary_df
    finally:
        conn.close()


def _load_token_frame(
    files: list[Path], country_token_column: str, global_token_column: str
) -> pl.DataFrame:
    file_frames: list[pl.DataFrame] = []
    for parquet_file in files:
        schema = pl.read_parquet_schema(parquet_file)
        projected_columns = [
            column
            for column in (country_token_column, global_token_column)
            if column in schema
        ]

        if projected_columns:
            source = pl.read_parquet(parquet_file, columns=projected_columns)
            row_count = source.height
        else:
            # Keep doc counts accurate without loading the full table payload.
            row_count = int(
                pl.scan_parquet(parquet_file)
                .select(pl.len().alias("row_count"))
                .collect()
                .item(0, "row_count")
            )
            source = pl.DataFrame()

        country_series = (
            source.get_column(country_token_column)
            if country_token_column in source.columns
            else pl.Series(country_token_column, [None] * row_count)
        )
        global_series = (
            source.get_column(global_token_column)
            if global_token_column in source.columns
            else pl.Series(global_token_column, [None] * row_count)
        )
        file_frames.append(
            pl.DataFrame(
                {
                    country_token_column: country_series,
                    global_token_column: global_series,
                }
            )
        )

    if not file_frames:
        return pl.DataFrame({country_token_column: [], global_token_column: []})
    return pl.concat(file_frames, how="vertical_relaxed")


def build_phase1_token_metrics(
    roots: WorkspaceRoots,
    *,
    systems: Iterable[str] | None = None,
    country_token_column: str = "country_tokens",
    global_token_column: str = "global_tokens",
    input_file: str | None = None,
    system_files: dict[str, list[Path]] | None = None,
    progress: Callable[[str], None] | None = None,
    verbose_progress: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    effective_systems = (
        list(systems)
        if systems is not None
        else _discover_systems_from_tokenized(roots, input_file)
    )
    country_rows: list[pl.DataFrame] = []
    global_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []

    systems_list = list(effective_systems)
    for index, system in enumerate(systems_list, start=1):
        files = (
            list(system_files.get(system, []))
            if system_files is not None
            else _system_parquet_files(roots, system, input_file)
        )
        if not files:
            if verbose_progress:
                _emit_progress(
                    progress,
                    f"polars [{index}/{len(systems_list)}] system={system} files=0 (skip)",
                )
            continue
        if verbose_progress:
            _emit_progress(
                progress,
                f"polars [{index}/{len(systems_list)}] system={system} files={len(files)}",
            )

        frame = _load_token_frame(files, country_token_column, global_token_column)

        country_stats, country_summary = _compute_token_stats(
            frame, country_token_column
        )
        if country_stats.height > 0:
            country_stats = country_stats.with_columns(
                pl.lit(system).alias("system"),
                pl.lit(int(country_summary["doc_count"])).alias("system_doc_count"),
                pl.lit(int(country_summary["total_tokens"])).alias(
                    "system_total_tokens"
                ),
            )
            country_stats = _attach_tf_df_percentages(
                country_stats,
                total_tokens=float(country_summary["total_tokens"]),
                doc_count=float(country_summary["doc_count"]),
                tf_alias="tf_pct",
                df_alias="df_pct",
            )
            country_rows.append(country_stats)

        global_frames.append(frame.select(pl.col(global_token_column)))

        summary_rows.append(
            {
                "system": system,
                "doc_count": country_summary["doc_count"],
                "total_tokens": country_summary["total_tokens"],
                "unique_tokens": country_summary["unique_tokens"],
                "avg_tokens_per_doc": country_summary["avg_tokens_per_doc"],
                "hapax_ratio": country_summary["hapax_ratio"],
            }
        )

    country_df = (
        pl.concat(country_rows, how="diagonal_relaxed")
        if country_rows
        else pl.DataFrame()
    )
    summary_df = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()

    if global_frames:
        global_source = pl.concat(global_frames, how="vertical_relaxed")
    else:
        global_source = pl.DataFrame({global_token_column: []})

    global_stats, global_summary = _compute_token_stats(
        global_source, global_token_column
    )
    if global_stats.height > 0:
        systems_present = (
            country_df.group_by("token").agg(
                pl.col("system").n_unique().alias("systems_present")
            )
            if country_df.height > 0
            else pl.DataFrame({"token": [], "systems_present": []})
        )
        global_df = (
            _attach_tf_df_percentages(
                global_stats.with_columns(
                    pl.lit(int(global_summary["doc_count"])).alias("global_doc_count"),
                    pl.lit(int(global_summary["total_tokens"])).alias(
                        "global_total_tokens"
                    ),
                ),
                total_tokens=float(global_summary["total_tokens"]),
                doc_count=float(global_summary["doc_count"]),
                tf_alias="global_tf_pct",
                df_alias="global_df_pct",
            )
            .join(systems_present, on="token", how="left")
            .with_columns(pl.col("systems_present").fill_null(0))
            .sort("tf", descending=True)
        )
    else:
        global_df = pl.DataFrame()

    country_df = _attach_global_comparison_columns(country_df, global_df)
    country_df = _attach_tfidf(country_df)
    if country_df.height > 0:
        country_df = country_df.sort(["system", "tf"], descending=[False, True])
    return country_df, global_df, summary_df


def _render_country_top_metric(
    country_df: pl.DataFrame,
    output_path: Path,
    *,
    top_n: int,
    sort_column: str,
    value_column: str,
    x_label: str,
    title_suffix: str,
    color: str,
    value_cast: Callable[[object], float],
) -> None:
    systems = (
        sorted(country_df.get_column("system").unique().to_list())
        if country_df.height > 0
        else []
    )
    if not systems:
        return

    cols = 2
    rows = ceil(len(systems) / cols)
    panel_height = max(4.5, min(8.5, 2.5 + (0.18 * top_n)))
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(panel_height * rows, 5)))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, system in enumerate(systems):
        ax = axes_list[idx]
        top = (
            country_df.filter(pl.col("system") == system)
            .sort(sort_column, descending=True)
            .head(top_n)
            .to_dicts()
        )
        labels = [row["token"] for row in top][::-1]
        values = [value_cast(row[value_column]) for row in top][::-1]
        ax.barh(labels, values, color=color)
        ax.set_title(f"{system.upper()} top {top_n} {title_suffix}")
        ax.set_xlabel(x_label)
        ax.tick_params(axis="y", labelsize=8)

    for idx in range(len(systems), len(axes_list)):
        axes_list[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _render_country_top_tokens(
    country_df: pl.DataFrame, output_path: Path, *, top_n: int = 15
) -> None:
    _render_country_top_metric(
        country_df,
        output_path,
        top_n=top_n,
        sort_column="tf",
        value_column="tf",
        x_label="TF",
        title_suffix="tokens",
        color="#2E5EAA",
        value_cast=_coerce_int_as_float,
    )


def _render_country_top_tfidf(
    country_df: pl.DataFrame, output_path: Path, *, top_n: int = 15
) -> None:
    _render_country_top_metric(
        country_df,
        output_path,
        top_n=top_n,
        sort_column="tfidf",
        value_column="tfidf",
        x_label="TF-IDF",
        title_suffix="TF-IDF tokens",
        color="#6C8EAD",
        value_cast=_coerce_float,
    )


def _coerce_int_as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(int(value))
    raise TypeError(f"Unsupported metric value type: {type(value)!r}")


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Unsupported metric value type: {type(value)!r}")


def _render_system_overview(summary_df: pl.DataFrame, output_path: Path) -> None:
    if summary_df.height == 0:
        return

    ordered = summary_df.sort("system")
    systems = ordered.get_column("system").to_list()
    docs = ordered.get_column("doc_count").to_list()
    unique_tokens = ordered.get_column("unique_tokens").to_list()
    avg_tokens = ordered.get_column("avg_tokens_per_doc").to_list()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(systems, docs, color="#2A9D8F")
    axes[0].set_title("Documents per system")
    axes[0].set_ylabel("Documents")

    axes[1].bar(systems, unique_tokens, color="#E76F51", label="Unique tokens")
    axes[1].plot(
        systems, avg_tokens, marker="o", color="#264653", label="Avg tokens/doc"
    )
    axes[1].set_title("Token richness by system")
    axes[1].set_ylabel("Count / average")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _markdown_table(
    rows: list[dict[str, object]], columns: list[tuple[str, str]]
) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _append_country_top_section(
    lines: list[str],
    *,
    heading: str,
    source_df: pl.DataFrame,
    top_n: int,
    sort_column: str,
    columns: list[tuple[str, str]],
    empty_message: str,
) -> None:
    lines.extend(["", heading, ""])
    if source_df.height == 0:
        lines.append(empty_message)
        return

    for system in sorted(source_df.get_column("system").unique().to_list()):
        lines.extend([f"### {system.upper()}", ""])
        top_rows = (
            source_df.filter(pl.col("system") == system)
            .sort(sort_column, descending=True)
            .head(top_n)
            .to_dicts()
        )
        lines.extend(_markdown_table(top_rows, columns))
        lines.append("")


def _build_phase1_report(
    *,
    run_date: str,
    summary_df: pl.DataFrame,
    country_df: pl.DataFrame,
    country_df_ex_legal_form: pl.DataFrame,
    global_df: pl.DataFrame,
    stoplist_candidates_df: pl.DataFrame,
    paths: Phase1Paths,
    top_n_country: int,
) -> str:
    lines = [
        "# Token Metrics Summary",
        "",
        f"- Run date: {run_date}",
        f"- Country token rows: {country_df.height}",
        f"- Global token rows: {global_df.height}",
        f"- Plot: `{paths.top_tokens_plot_path.as_posix()}`",
        f"- Plot: `{paths.top_tokens_ex_legal_form_plot_path.as_posix()}`",
        f"- Plot: `{paths.top_tfidf_plot_path.as_posix()}`",
        f"- Plot: `{paths.top_tfidf_ex_legal_form_plot_path.as_posix()}`",
        f"- Plot: `{paths.overview_plot_path.as_posix()}`",
        f"- Stoplist candidates: `{paths.stoplist_candidates_path.as_posix()}`",
        f"- Stoplist candidates CSV: `{paths.stoplist_candidates_csv_path.as_posix()}`",
        "",
        "## Country Statistical Overview",
        "",
    ]

    if summary_df.height > 0:
        lines.extend(
            _markdown_table(
                summary_df.sort("system").to_dicts(),
                [
                    ("system", "System"),
                    ("doc_count", "Documents"),
                    ("total_tokens", "Total Tokens"),
                    ("unique_tokens", "Unique Tokens"),
                    ("avg_tokens_per_doc", "Avg Tokens/Doc"),
                    ("hapax_ratio", "Hapax Ratio"),
                ],
            )
        )
    else:
        lines.append("No country summary rows produced.")

    _append_country_top_section(
        lines,
        heading=f"## Top {top_n_country} Tokens By Country",
        source_df=country_df,
        top_n=top_n_country,
        sort_column="tf",
        columns=[
            ("token", "Token"),
            ("tf", "TF"),
            ("df", "DF"),
            ("tf_pct", "TF %"),
            ("df_pct", "DF %"),
            ("tf_ratio_to_global", "TF Ratio vs Global"),
        ],
        empty_message="No country token rows produced.",
    )
    _append_country_top_section(
        lines,
        heading=f"## Top {top_n_country} Tokens By Country (Excluding Legal Form Tokens)",
        source_df=country_df_ex_legal_form,
        top_n=top_n_country,
        sort_column="tf",
        columns=[
            ("token", "Token"),
            ("tf", "TF"),
            ("df", "DF"),
            ("tf_pct", "TF %"),
            ("df_pct", "DF %"),
        ],
        empty_message="No country token rows produced after legal-form filtering.",
    )
    _append_country_top_section(
        lines,
        heading=f"## Top {top_n_country} TF-IDF Tokens By Country",
        source_df=country_df,
        top_n=top_n_country,
        sort_column="tfidf",
        columns=[
            ("token", "Token"),
            ("tfidf", "TF-IDF"),
            ("tf", "TF"),
            ("df", "DF"),
            ("idf", "IDF"),
        ],
        empty_message="No country token rows produced.",
    )
    _append_country_top_section(
        lines,
        heading=f"## Top {top_n_country} TF-IDF Tokens By Country (Excluding Legal Form Tokens)",
        source_df=country_df_ex_legal_form,
        top_n=top_n_country,
        sort_column="tfidf",
        columns=[
            ("token", "Token"),
            ("tfidf", "TF-IDF"),
            ("tf", "TF"),
            ("df", "DF"),
            ("idf", "IDF"),
        ],
        empty_message="No TF-IDF rows produced after legal-form filtering.",
    )

    lines.extend(["", "## Stoplist Candidates (Likely Non-Discriminative)", ""])
    if stoplist_candidates_df.height > 0:
        lines.extend(
            _markdown_table(
                stoplist_candidates_df.head(top_n_country).to_dicts(),
                [
                    ("token", "Token"),
                    ("coverage_ratio", "Coverage Ratio"),
                    ("systems_present", "Systems Present"),
                    ("global_df_pct", "Global DF %"),
                    ("mean_tfidf", "Mean TF-IDF"),
                    ("std_tfidf", "Std TF-IDF"),
                    ("token_type", "Token Type"),
                ],
            )
        )
    else:
        lines.append("No stoplist candidates met thresholds.")

    lines.extend(["## Global Top Tokens", ""])
    if global_df.height > 0:
        lines.extend(
            _markdown_table(
                global_df.sort("tf", descending=True).head(20).to_dicts(),
                [
                    ("token", "Token"),
                    ("tf", "TF"),
                    ("df", "DF"),
                    ("global_tf_pct", "Global TF %"),
                    ("global_df_pct", "Global DF %"),
                    ("systems_present", "Systems Present"),
                ],
            )
        )
    else:
        lines.append("No global token rows produced.")

    return "\n".join(lines) + "\n"


def run_phase1_token_analysis(
    roots: WorkspaceRoots,
    *,
    run_date: str | None = None,
    systems: Iterable[str] | None = None,
    country_token_column: str = "country_tokens",
    global_token_column: str = "global_tokens",
    legal_form_column: str = "company_type",
    input_file: str | None = None,
    top_n: int = 50,
    top_n_plot: int | None = None,
    stoplist_min_coverage_ratio: float = 0.6,
    stoplist_min_global_df_pct: float = 0.01,
    engine: str = "polars",
    progress: Callable[[str], None] | None = None,
    verbose_progress: bool = False,
    threads: int | None = None,
) -> Phase1Paths:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if top_n_plot is not None and top_n_plot < 1:
        raise ValueError("top_n_plot must be >= 1 when provided")
    if not 0.0 <= stoplist_min_coverage_ratio <= 1.0:
        raise ValueError("stoplist_min_coverage_ratio must be within [0, 1]")
    if not 0.0 <= stoplist_min_global_df_pct <= 1.0:
        raise ValueError("stoplist_min_global_df_pct must be within [0, 1]")

    effective_top_n = top_n_plot if top_n_plot is not None else top_n

    t_total_start = perf_counter()
    effective_run_date = run_date or datetime.now(UTC).date().isoformat()
    paths = resolve_phase1_paths(roots, effective_run_date)
    systems_list = (
        list(systems)
        if systems is not None
        else _discover_systems_from_tokenized(roots, input_file)
    )
    system_files = _resolve_system_files(roots, systems_list, input_file)
    active_systems, total_files, total_bytes = _input_inventory(system_files)

    _emit_progress(
        progress,
        (
            f"phase1 preflight: engine={engine}, systems={len(systems_list)}, active_systems={active_systems}, "
            f"files={total_files}, bytes={total_bytes:,}, input_file={input_file or '<all>'}"
        ),
    )

    if verbose_progress:
        for system in systems_list:
            files = system_files.get(system, [])
            _emit_progress(progress, f"preflight system={system} files={len(files)}")

    _emit_progress(progress, "phase1 stage=aggregate start")
    t_aggregate_start = perf_counter()
    if engine == "duckdb":
        country_df, global_df, summary_df = _build_phase1_token_metrics_duckdb(
            roots,
            systems=systems_list,
            country_token_column=country_token_column,
            global_token_column=global_token_column,
            input_file=input_file,
            system_files=system_files,
            progress=progress,
            verbose_progress=verbose_progress,
            threads=threads,
        )
    else:
        country_df, global_df, summary_df = build_phase1_token_metrics(
            roots,
            systems=systems_list,
            country_token_column=country_token_column,
            global_token_column=global_token_column,
            input_file=input_file,
            system_files=system_files,
            progress=progress,
            verbose_progress=verbose_progress,
        )
    _emit_progress(
        progress,
        (
            f"phase1 stage=aggregate done in {perf_counter() - t_aggregate_start:.2f}s "
            f"(country_rows={country_df.height}, global_rows={global_df.height}, summary_rows={summary_df.height})"
        ),
    )

    legal_form_tokens_df = _build_legal_form_token_table(systems=systems_list)
    noise_words_df = _build_noise_word_table()
    country_df_ex_legal_form = _exclude_legal_form_tokens(
        country_df, legal_form_tokens_df
    )
    stoplist_candidates_df = _derive_stoplist_candidates(
        country_df=country_df,
        global_df=global_df,
        summary_df=summary_df,
        legal_form_tokens_df=legal_form_tokens_df,
        noise_words_df=noise_words_df,
        min_coverage_ratio=stoplist_min_coverage_ratio,
        min_global_df_pct=stoplist_min_global_df_pct,
    )

    _emit_progress(progress, "phase1 stage=write_metrics start")
    t_write_metrics_start = perf_counter()
    country_df.write_parquet(paths.country_stats_path)
    global_df.write_parquet(paths.global_stats_path)
    stoplist_candidates_df.write_parquet(paths.stoplist_candidates_path)
    stoplist_candidates_df.write_csv(paths.stoplist_candidates_csv_path)
    _emit_progress(
        progress,
        f"phase1 stage=write_metrics done in {perf_counter() - t_write_metrics_start:.2f}s",
    )

    _emit_progress(progress, "phase1 stage=render_plots start")
    t_render_start = perf_counter()
    _render_country_top_tokens(
        country_df, paths.top_tokens_plot_path, top_n=effective_top_n
    )
    _render_country_top_tokens(
        country_df_ex_legal_form,
        paths.top_tokens_ex_legal_form_plot_path,
        top_n=effective_top_n,
    )
    _render_country_top_tfidf(
        country_df, paths.top_tfidf_plot_path, top_n=effective_top_n
    )
    _render_country_top_tfidf(
        country_df_ex_legal_form,
        paths.top_tfidf_ex_legal_form_plot_path,
        top_n=effective_top_n,
    )
    _render_system_overview(summary_df, paths.overview_plot_path)
    _emit_progress(
        progress,
        f"phase1 stage=render_plots done in {perf_counter() - t_render_start:.2f}s",
    )

    _emit_progress(progress, "phase1 stage=write_report start")
    t_report_start = perf_counter()
    report_text = _build_phase1_report(
        run_date=effective_run_date,
        summary_df=summary_df,
        country_df=country_df,
        country_df_ex_legal_form=country_df_ex_legal_form,
        global_df=global_df,
        stoplist_candidates_df=stoplist_candidates_df,
        paths=paths,
        top_n_country=effective_top_n,
    )
    paths.report_path.write_text(report_text, encoding="utf-8")
    _emit_progress(
        progress,
        f"phase1 stage=write_report done in {perf_counter() - t_report_start:.2f}s",
    )

    total_runtime_seconds = perf_counter() - t_total_start
    manifest_entry = build_phase_entry(
        created_utc=datetime.now(UTC).isoformat(),
        phase="phase1_token_metrics",
        run_date=effective_run_date,
        systems=systems_list,
        active_systems=active_systems,
        parameters={
            "engine": engine,
            "country_token_column": country_token_column,
            "global_token_column": global_token_column,
            "legal_form_column": legal_form_column,
            "input_file": input_file,
            "top_n": top_n,
            "top_n_plot": effective_top_n,
            "stoplist_min_coverage_ratio": stoplist_min_coverage_ratio,
            "stoplist_min_global_df_pct": stoplist_min_global_df_pct,
        },
        outputs={
            "country_stats": paths.country_stats_path.as_posix(),
            "global_stats": paths.global_stats_path.as_posix(),
            "report": paths.report_path.as_posix(),
            "top_tokens_plot": paths.top_tokens_plot_path.as_posix(),
            "top_tokens_ex_legal_form_plot": paths.top_tokens_ex_legal_form_plot_path.as_posix(),
            "top_tfidf_plot": paths.top_tfidf_plot_path.as_posix(),
            "top_tfidf_ex_legal_form_plot": paths.top_tfidf_ex_legal_form_plot_path.as_posix(),
            "stoplist_candidates": paths.stoplist_candidates_path.as_posix(),
            "stoplist_candidates_csv": paths.stoplist_candidates_csv_path.as_posix(),
            "overview_plot": paths.overview_plot_path.as_posix(),
        },
        row_counts={
            "country_rows": country_df.height,
            "global_rows": global_df.height,
            "summary_rows": summary_df.height,
            "stoplist_candidate_rows": stoplist_candidates_df.height,
        },
        runtime_seconds=total_runtime_seconds,
    )
    record_phase_run(roots, effective_run_date, manifest_entry)

    _emit_progress(progress, f"phase1 complete in {total_runtime_seconds:.2f}s")
    return paths
