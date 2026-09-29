"""GLEIF's unrecoverable no-matches broken down by entity category, run through `scripts/analyze_gleif_entity_category.py`."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.match_metrics import MatchAnalysisPaths, resolve_match_analysis_paths
from analysis.report_layout import metrics_dir as _metrics_dir
from analysis.report_layout import viz_dir as _viz_dir
from workspace.data_layout import system_layer_dir
from workspace.roots import WorkspaceRoots

GLEIF_ENTITY_CATEGORY_UNKNOWN = "UNKNOWN"
# `workspace.data_layout` names constants for canonical/cleansed/matched/
# tokenized only; the acquire-time source snapshot layer has none, so the
# layer is named here while the path still comes from the layout function.
GLEIF_SOURCE_LAYER_NAME = "source"


def _resolve_gleif_source_dir(roots: WorkspaceRoots, run_date: str | None) -> Path:
    source_root = system_layer_dir(roots, "gleif", layer=GLEIF_SOURCE_LAYER_NAME)
    if run_date:
        candidate = source_root / run_date
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"No GLEIF source snapshot found for run_date={run_date}: {candidate}"
            )
        return candidate
    candidates = sorted(p for p in source_root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No GLEIF source snapshots found under {source_root}")
    return candidates[-1]


def _load_gleif_entity_category_lookup(
    roots: WorkspaceRoots, run_date: str | None
) -> pl.DataFrame:
    source_dir = _resolve_gleif_source_dir(roots, run_date)
    shard_files = sorted(
        p for p in source_dir.glob("*.parquet") if "sidecar" not in p.name
    )
    if not shard_files:
        raise FileNotFoundError(f"No GLEIF shard files found under {source_dir}")
    lf = pl.scan_parquet([str(p) for p in shard_files]).select(
        pl.col("LEI").alias("source_lei"),
        pl.col("Entity_EntityCategory").alias("entity_category"),
    )
    return lf.unique(subset=["source_lei"]).collect()


def _render_gleif_entity_category_chart(
    breakdown_df: pl.DataFrame, output_path: Path, *, context_label: str
) -> None:
    labels = breakdown_df["entity_category"].to_list()
    values = breakdown_df["not_recoverable_rows"].to_list()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values)
    ax.set_title(
        f"Not-Recoverable No-Matches by GLEIF Entity Category ({context_label})"
    )
    ax.set_xlabel("Rows")
    for i, v in enumerate(values):
        ax.text(v, i, str(v), va="center")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_gleif_not_recoverable_by_entity_category(
    roots: WorkspaceRoots,
    *,
    run_date: str,
    target_system: str,
    country: str | None = None,
    target_display: str = "",
    gleif_source_run_date: str | None = None,
) -> tuple[pl.DataFrame, MatchAnalysisPaths, Path, Path]:
    resolved_country = country or target_system
    match_paths = resolve_match_analysis_paths(
        roots,
        run_date=run_date,
        source_system="gleif",
        target_system=target_system,
        country=resolved_country,
        target_display=target_display,
    )
    detail_path = match_paths.no_match_recoverability_detail_path
    if not detail_path.exists():
        raise FileNotFoundError(
            f"No-match recoverability detail not found at {detail_path}. "
            "Run scripts/analyze_matches.py for this scenario first."
        )

    detail_df = pl.read_parquet(detail_path)
    category_lookup_df = _load_gleif_entity_category_lookup(
        roots, gleif_source_run_date
    )

    enriched_df = detail_df.join(
        category_lookup_df, on="source_lei", how="left"
    ).with_columns(pl.col("entity_category").fill_null(GLEIF_ENTITY_CATEGORY_UNKNOWN))

    breakdown_df = (
        enriched_df.group_by("entity_category")
        .agg(
            pl.len().alias("no_match_rows"),
            pl.col("not_recoverable").sum().alias("not_recoverable_rows"),
        )
        .with_columns(
            (
                pl.col("not_recoverable_rows")
                / pl.col("not_recoverable_rows").sum()
                * 100
            )
            .round(2)
            .alias("pct_of_not_recoverable_total"),
            (pl.col("not_recoverable_rows") / pl.col("no_match_rows") * 100)
            .round(2)
            .alias("not_recoverable_rate_within_category_pct"),
        )
        .sort("not_recoverable_rows", descending=True)
    )

    breakdown_path = (
        _metrics_dir(match_paths.run_root) / "gleif_entity_category_breakdown.parquet"
    )
    chart_path = (
        _viz_dir(match_paths.run_root) / "gleif_entity_category_not_recoverable.png"
    )
    breakdown_df.write_parquet(breakdown_path)
    _render_gleif_entity_category_chart(
        breakdown_df, chart_path, context_label=match_paths.context_label
    )

    return breakdown_df, match_paths, breakdown_path, chart_path
