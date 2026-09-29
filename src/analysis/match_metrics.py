"""Match-outcome analysis for one scenario: KPIs, no-match triage, recoverability and comparison charts.

`run_match_analysis` analyses one scenario and `run_match_analysis_batch` a list of them; `scripts/analyze_matches.py` and `notebooks/analyse_matches.ipynb` both call these, so both write the same tree under `artifacts/analysis/match_analysis/runs/<run_date>/<scenario>/`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import polars as pl

from analysis.report_layout import metrics_dir as _metrics_dir
from analysis.report_layout import viz_dir as _viz_dir
from workspace.artifact_layout import analysis_report_run_dir
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import layer_partition_dir, resolve_partition_dir
from workspace.roots import WorkspaceRoots

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REQUIRED_LOOKUP_COLUMNS = {
    "system_uri",
    "jurisdiction_code",
    "name",
    "name_cleansed_basic",
    "name_cleansed",
}
_REQUIRED_MATCH_COLUMNS = {"system_uri", "match_uri"}

# Shared projection blocks used across triage and no-match output tables.
COMMON_MATCH_PROJECTION = [
    "system_uri",
    "match_uri",
    "source_company_number",
    "source_lei",
    "source_vat",
    "source_tax_id",
    "source_jurisdiction_code",
    "source_company_type",
    "source_current_status",
    "source_incorporation_date",
    "source_dissolution_date",
    "source_inactive",
    "source_branch",
    "source_branch_status",
    "source_registered_address_in_full",
    "source_registry_url",
    "source_opencorporates_url",
    "source_metadata_generated_utc",
    "source_short_name",
    "source_quoted_name",
    "source_acronym",
    "source_personal_owner",
    "source_company_type_source",
    "source_company_type_missing",
]

SOURCE_COMPARISON_PROJECTION = [
    "source_name",
    "source_basic_name",
    "source_cleansed_name",
]

MATCH_RESULT_FLAGS_PROJECTION = [
    "name_equal",
    "basic_equal",
    "cleansed_equal",
]

UNMATCHED_AUDIT_PROJECTION = [
    "has_match_uri",
    *MATCH_RESULT_FLAGS_PROJECTION,
    "source_company_number_safe",
    "source_lei_safe",
    "source_vat_safe",
    "source_tax_id_safe",
]

MATCHED_RIGHT_ID_PROJECTION = [
    "target_company_number",
    "target_lei",
    "target_vat",
    "target_tax_id",
]

NO_MATCH_RECOVERABILITY_FLAGS = [
    "can_recover_by_name",
    "can_recover_by_basic",
    "can_recover_by_cleansed",
    "not_recoverable",
]

TARGET_MATCH_PROJECTION = [
    "target_name",
    "target_name_cleansed",
    "target_jurisdiction_code",
    "target_company_number",
    "target_company_type",
    "target_current_status",
    "target_incorporation_date",
    "target_dissolution_date",
    "target_inactive",
    "target_branch",
    "target_branch_status",
    "target_registered_address_in_full",
    "target_registry_url",
    "target_opencorporates_url",
    "target_lei",
    "target_vat",
    "target_tax_id",
    "target_metadata_generated_utc",
    "target_short_name",
    "target_quoted_name",
    "target_acronym",
    "target_name_cleansed_basic",
    "target_personal_owner",
    "target_company_type_source",
    "target_company_type_missing",
]

EXAMPLE_SELECT_OUTPUT_NAMES = [
    "category",
    *NO_MATCH_RECOVERABILITY_FLAGS,
    "source_name",
    "source_basic_name",
    "source_cleansed_name",
    "source_company_number",
    "source_lei",
    "source_vat",
    "source_tax_id",
    "source_jurisdiction_code",
    "source_company_type",
    "source_current_status",
    "source_incorporation_date",
    "source_dissolution_date",
    "source_inactive",
    "source_branch",
    "source_branch_status",
    "source_registered_address_in_full",
    "source_registry_url",
    "source_opencorporates_url",
    "source_metadata_generated_utc",
    "source_short_name",
    "source_quoted_name",
    "source_acronym",
    "source_personal_owner",
    "source_company_type_source",
    "source_company_type_missing",
    *TARGET_MATCH_PROJECTION,
]

EXAMPLE_SELECT_COLS = [
    "category",
    *NO_MATCH_RECOVERABILITY_FLAGS,
    *COMMON_MATCH_PROJECTION,
    *TARGET_MATCH_PROJECTION,
]


@dataclass(frozen=True)
class MatchAnalysisPaths:
    run_root: Path
    scenario: str
    context_label: str
    summary_path: Path
    reasons_path: Path
    examples_path: Path
    name_disagree_summary_path: Path
    name_disagree_examples_path: Path
    no_match_name_examples_path: Path
    no_match_basic_examples_path: Path
    no_match_cleansed_examples_path: Path
    no_match_strategy_path: Path
    no_match_recoverability_detail_path: Path
    match_outcome_chart_path: Path
    equality_rates_chart_path: Path
    equality_overlap_venn_path: Path
    matched_overlap_states_chart_path: Path
    no_match_reasons_chart_path: Path
    no_match_recoverability_chart_path: Path


def _context_label(
    source_system: str, target_system: str, country: str, target_display: str
) -> str:
    if target_display:
        target_context = target_display
    elif target_system == country:
        target_context = target_system.upper()
    else:
        target_context = f"{target_system.title()} ({country.upper()})"
    return f"{source_system.title()} -> {target_context}"


def resolve_match_analysis_run_root(
    roots: WorkspaceRoots,
    *,
    run_date: str,
    source_system: str,
    target_system: str,
    country: str,
) -> Path:
    """One scenario's run directory, with no side effect -- for a caller (a
    script's `--dry-run` report) that wants the path `resolve_match_analysis_paths`
    would resolve without that function's directory-creating side effect."""
    scenario = f"{source_system}_to_{target_system}_{country}"
    return analysis_report_run_dir(roots, "match_analysis", run_date) / scenario


def resolve_match_analysis_paths(
    roots: WorkspaceRoots,
    *,
    run_date: str,
    source_system: str,
    target_system: str,
    country: str,
    target_display: str = "",
) -> MatchAnalysisPaths:
    run_root = resolve_match_analysis_run_root(
        roots,
        run_date=run_date,
        source_system=source_system,
        target_system=target_system,
        country=country,
    )
    metrics_dir = _metrics_dir(run_root)
    viz_dir = _viz_dir(run_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    return MatchAnalysisPaths(
        run_root=run_root,
        scenario=run_root.name,
        context_label=_context_label(
            source_system, target_system, country, target_display
        ),
        summary_path=metrics_dir / "summary.parquet",
        reasons_path=metrics_dir / "no_match_reasons.parquet",
        examples_path=metrics_dir / "unmatched_examples.parquet",
        name_disagree_summary_path=metrics_dir
        / "matched_name_disagree_summary.parquet",
        name_disagree_examples_path=metrics_dir
        / "matched_name_disagree_examples.parquet",
        no_match_name_examples_path=metrics_dir / "no_match_name_examples.parquet",
        no_match_basic_examples_path=metrics_dir / "no_match_basic_examples.parquet",
        no_match_cleansed_examples_path=metrics_dir
        / "no_match_cleansed_examples.parquet",
        no_match_strategy_path=metrics_dir / "no_match_recoverability_strategy.parquet",
        no_match_recoverability_detail_path=metrics_dir
        / "no_match_recoverability_detail.parquet",
        match_outcome_chart_path=viz_dir / "match_outcome.png",
        equality_rates_chart_path=viz_dir / "equality_rates_by_metric.png",
        equality_overlap_venn_path=viz_dir / "equality_overlap_venn.png",
        matched_overlap_states_chart_path=viz_dir / "matched_overlap_states.png",
        no_match_reasons_chart_path=viz_dir / "no_match_reasons.png",
        no_match_recoverability_chart_path=viz_dir
        / "no_match_recoverability_by_strategy.png",
    )


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        print(f"[analysis] {message}")


def _load_country_frame(
    base_dir: Path, country: str, max_rows: int | None
) -> pl.DataFrame:
    # resolve_partition_dir prefers the current primary/-family location and
    # falls back to the pre-family-split top-level one; when neither exists
    # (a genuinely missing country) layer_partition_dir names the current
    # shape's location anyway, so a missing partition still fails the same
    # way it did before -- scan_parquet raising on an empty glob.
    partition_dir = resolve_partition_dir(base_dir, value=country) or (
        layer_partition_dir(base_dir, value=country)
    )
    pattern = str(partition_dir / "*.parquet")
    lf = pl.scan_parquet(pattern).filter(
        pl.col("jurisdiction_code").cast(pl.Utf8).str.to_lowercase() == country
    )
    if max_rows is not None:
        lf = lf.limit(max_rows)
    return lf.collect()


def _validate_match_schema(
    source_df: pl.DataFrame, target_df: pl.DataFrame, matched_df: pl.DataFrame
) -> None:
    source_missing = sorted(_REQUIRED_LOOKUP_COLUMNS - set(source_df.columns))
    target_missing = sorted(_REQUIRED_LOOKUP_COLUMNS - set(target_df.columns))
    matched_missing = sorted(_REQUIRED_MATCH_COLUMNS - set(matched_df.columns))
    if source_missing:
        raise ValueError(f"source_df missing required columns: {source_missing}")
    if target_missing:
        raise ValueError(f"target_df missing required columns: {target_missing}")
    if matched_missing:
        raise ValueError(f"matched_df missing required columns: {matched_missing}")


# Carried onto the analysis frame verbatim under a `<side>_` prefix, in this
# order. One list for both sides, so a column added to the source projection
# cannot go missing from the target one.
_SIDE_SUBSET_COLUMNS: tuple[str, ...] = (
    "jurisdiction_code",
    "company_number",
    "lei",
    "vat",
    "tax_id",
    "company_type",
    "current_status",
    "incorporation_date",
    "dissolution_date",
    "inactive",
    "branch",
    "branch_status",
    "registered_address_in_full",
    "registry_url",
    "opencorporates_url",
    "metadata_generated_utc",
    "short_name",
    "quoted_name",
    "acronym",
    "personal_owner",
    "company_type_source",
    "company_type_missing",
)

# The two cleansed-name columns are the only place the sides' output names do
# not follow the `<side>_<column>` rule: the source side reads
# `source_basic_name`/`source_cleansed_name` while the target side reads
# `target_name_cleansed_basic`/`target_name_cleansed`. Both spellings are
# consumed downstream as-is, so they are pinned here rather than derived.
_SIDE_CLEANSED_NAME_ALIASES: dict[str, tuple[str, str]] = {
    "source": ("source_basic_name", "source_cleansed_name"),
    "target": ("target_name_cleansed_basic", "target_name_cleansed"),
}


def _project_side_subset(
    frame: pl.DataFrame,
    *,
    side: str,
    name_col: str,
    basic_col: str,
    cleansed_col: str,
) -> pl.DataFrame:
    """Project one side of the match ("source" or "target") onto its prefixed
    analysis-frame columns, deduplicated on that side's `system_uri`.
    """
    basic_alias, cleansed_alias = _SIDE_CLEANSED_NAME_ALIASES[side]
    return frame.select(
        [
            pl.col("system_uri").alias(f"{side}_system_uri"),
            pl.col(name_col).alias(f"{side}_name"),
            pl.col(basic_col).alias(basic_alias),
            pl.col(cleansed_col).alias(cleansed_alias),
            *(
                pl.col(column).alias(f"{side}_{column}")
                for column in _SIDE_SUBSET_COLUMNS
            ),
        ]
    ).unique(subset=[f"{side}_system_uri"])


def _build_analysis_frame(
    matched_df: pl.DataFrame,
    source_subset: pl.DataFrame,
    target_subset: pl.DataFrame,
) -> pl.DataFrame:
    analysis_df = (
        matched_df.select(["system_uri", "match_uri"])
        .join(
            source_subset,
            left_on="system_uri",
            right_on="source_system_uri",
            how="left",
        )
        .join(
            target_subset,
            left_on="match_uri",
            right_on="target_system_uri",
            how="left",
        )
    )

    analysis_df = analysis_df.with_columns(
        [
            pl.col("system_uri").alias("source_system_uri"),
            pl.col("match_uri").alias("target_system_uri"),
        ]
    )

    analysis_df = analysis_df.with_columns(
        pl.when(
            pl.col("match_uri").is_null() | (pl.col("match_uri").fill_null("") == "")
        )
        .then(False)
        .otherwise(True)
        .alias("has_match_uri")
    )

    # Source-to-target equality: compare the two lookup frames, not the bridge row.
    analysis_df = analysis_df.with_columns(
        pl.when(
            pl.col("has_match_uri")
            & pl.col("source_name").is_not_null()
            & pl.col("target_name").is_not_null()
            & (pl.col("source_name") == pl.col("target_name"))
        )
        .then(True)
        .when(pl.col("has_match_uri"))
        .then(False)
        .otherwise(None)
        .alias("name_equal")
    )

    analysis_df = analysis_df.with_columns(
        pl.when(
            pl.col("has_match_uri")
            & pl.col("source_basic_name").is_not_null()
            & pl.col("target_name_cleansed_basic").is_not_null()
            & (pl.col("source_basic_name") == pl.col("target_name_cleansed_basic"))
        )
        .then(True)
        .when(pl.col("has_match_uri"))
        .then(False)
        .otherwise(None)
        .alias("basic_equal")
    )

    analysis_df = analysis_df.with_columns(
        pl.when(
            pl.col("has_match_uri")
            & pl.col("source_cleansed_name").is_not_null()
            & pl.col("target_name_cleansed").is_not_null()
            & (pl.col("source_cleansed_name") == pl.col("target_name_cleansed"))
        )
        .then(True)
        .when(pl.col("has_match_uri"))
        .then(False)
        .otherwise(None)
        .alias("cleansed_equal")
    )

    return analysis_df


def _compute_summary(analysis_df: pl.DataFrame) -> pl.DataFrame:
    total_rows = analysis_df.height
    matched_rows = analysis_df.filter(pl.col("has_match_uri")).height
    unmatched_rows = total_rows - matched_rows
    match_rate = (matched_rows / total_rows) if total_rows else 0.0

    return pl.DataFrame(
        {
            "metric": [
                "total_rows",
                "matched_rows",
                "unmatched_rows",
                "match_rate_pct",
            ],
            "value": [
                float(total_rows),
                float(matched_rows),
                float(unmatched_rows),
                float(round(match_rate * 100, 2)),
            ],
        },
        schema={"metric": pl.Utf8, "value": pl.Float64},
    )


def _equality_breakdown(matched_only_df: pl.DataFrame, column: str) -> pl.DataFrame:
    return (
        matched_only_df.group_by(column)
        .agg(pl.len().alias("rows"))
        .with_columns(
            (pl.col("rows") / pl.col("rows").sum() * 100)
            .round(2)
            .alias("pct_of_matched")
        )
        .sort(column)
    )


def _compute_equality_breakdowns(
    matched_only_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, int]:
    name_breakdown_df = _equality_breakdown(matched_only_df, "name_equal")
    basic_breakdown_df = _equality_breakdown(matched_only_df, "basic_equal")
    cleansed_breakdown_df = _equality_breakdown(matched_only_df, "cleansed_equal")

    equality_cross_df = (
        matched_only_df.group_by(["name_equal", "basic_equal", "cleansed_equal"])
        .agg(pl.len().alias("rows"))
        .with_columns(
            (pl.col("rows") / pl.col("rows").sum() * 100)
            .round(2)
            .alias("pct_of_matched")
        )
        .sort(["name_equal", "basic_equal", "cleansed_equal"])
    )

    disagree_count = matched_only_df.filter(
        (pl.col("name_equal") != pl.col("basic_equal"))
        | (pl.col("basic_equal") != pl.col("cleansed_equal"))
    ).height

    return (
        name_breakdown_df,
        basic_breakdown_df,
        cleansed_breakdown_df,
        equality_cross_df,
        disagree_count,
    )


def _render_match_outcome_chart(
    analysis_df: pl.DataFrame, output_path: Path, *, context_label: str
) -> None:
    status_counts = (
        analysis_df.with_columns(
            pl.when(pl.col("has_match_uri"))
            .then(pl.lit("Matched"))
            .otherwise(pl.lit("Unmatched"))
            .alias("status"),
        )
        .group_by("status")
        .agg(pl.len().alias("rows"))
        .sort("status")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(status_counts["status"].to_list(), status_counts["rows"].to_list())
    ax.set_title(f"Match Outcome ({context_label})")
    ax.set_ylabel("Rows")
    for i, v in enumerate(status_counts["rows"].to_list()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _pct_for_flag(df: pl.DataFrame, value: bool) -> float:
    match_df = df.filter(pl.col(df.columns[0]) == value)
    return float(match_df["pct_of_matched"][0]) if match_df.height else 0.0


def _render_equality_rates_chart(
    name_breakdown_df: pl.DataFrame,
    basic_breakdown_df: pl.DataFrame,
    cleansed_breakdown_df: pl.DataFrame,
    output_path: Path,
    *,
    context_label: str,
) -> None:
    grouped_metrics = [
        ("name", name_breakdown_df),
        ("basic", basic_breakdown_df),
        ("cleansed", cleansed_breakdown_df),
    ]
    metric_labels = [m for m, _ in grouped_metrics]
    false_pcts = [_pct_for_flag(df, False) for _, df in grouped_metrics]
    true_pcts = [_pct_for_flag(df, True) for _, df in grouped_metrics]
    x = list(range(len(metric_labels)))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - bar_w / 2 for i in x], false_pcts, width=bar_w, label="False")
    ax.bar([i + bar_w / 2 for i in x], true_pcts, width=bar_w, label="True")
    ax.set_title(f"Equality Rates by Metric ({context_label})")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Share of matched records (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    for i, v in enumerate(false_pcts):
        ax.text(i - bar_w / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(true_pcts):
        ax.text(i + bar_w / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _render_equality_overlap_chart(
    matched_only_df: pl.DataFrame, output_path: Path, *, context_label: str
) -> None:
    name_flag = pl.col("name_equal")
    basic_flag = pl.col("basic_equal")
    cleansed_flag = pl.col("cleansed_equal")

    venn_subsets = (
        matched_only_df.filter(name_flag & ~basic_flag & ~cleansed_flag).height,
        matched_only_df.filter(~name_flag & basic_flag & ~cleansed_flag).height,
        matched_only_df.filter(name_flag & basic_flag & ~cleansed_flag).height,
        matched_only_df.filter(~name_flag & ~basic_flag & cleansed_flag).height,
        matched_only_df.filter(name_flag & ~basic_flag & cleansed_flag).height,
        matched_only_df.filter(~name_flag & basic_flag & cleansed_flag).height,
        matched_only_df.filter(name_flag & basic_flag & cleansed_flag).height,
    )

    try:
        from matplotlib_venn import venn3

        fig, ax = plt.subplots(figsize=(8, 6))
        venn3(
            subsets=venn_subsets,
            set_labels=("name_equal", "basic_equal", "cleansed_equal"),
            ax=ax,
        )
        ax.set_title(f"Overlap of Equality Conditions ({context_label})")
    except Exception:  # noqa: BLE001
        # Dependency-free fallback: fixed-layout three-circle overlap with region counts.
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"Overlap of Equality Conditions ({context_label})")

        c1 = Circle((-0.75, 0.0), 1.25, alpha=0.25, color="#d62728", ec="black", lw=1.0)
        c2 = Circle((0.75, 0.0), 1.25, alpha=0.25, color="#1f77b4", ec="black", lw=1.0)
        c3 = Circle((0.0, 0.85), 1.25, alpha=0.25, color="#2ca02c", ec="black", lw=1.0)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.add_patch(c3)

        ax.text(-1.85, -1.25, "name_equal", fontsize=10)
        ax.text(1.05, -1.25, "basic_equal", fontsize=10)
        ax.text(-0.4, 2.05, "cleansed_equal", fontsize=10)

        ax.text(-1.65, -0.10, f"{venn_subsets[0]:,}", ha="center", va="center")
        ax.text(1.65, -0.10, f"{venn_subsets[1]:,}", ha="center", va="center")
        ax.text(0.00, -0.55, f"{venn_subsets[2]:,}", ha="center", va="center")
        ax.text(0.00, 1.95, f"{venn_subsets[3]:,}", ha="center", va="center")
        ax.text(-0.65, 0.90, f"{venn_subsets[4]:,}", ha="center", va="center")
        ax.text(0.65, 0.90, f"{venn_subsets[5]:,}", ha="center", va="center")
        ax.text(
            0.00,
            0.35,
            f"{venn_subsets[6]:,}",
            ha="center",
            va="center",
            fontweight="bold",
        )
        ax.text(
            0.0,
            -1.55,
            "Fallback Venn is layout-based (not area-proportional). Use bar/table below for exact proportions.",
            ha="center",
            fontsize=9,
        )

        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1.8, 2.4)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _compute_matched_overlap_states(
    equality_cross_df: pl.DataFrame, matched_total: int
) -> pl.DataFrame:
    state_order = [
        "All equal",
        "Name mismatch only",
        "Name + basic mismatch",
        "All mismatch",
        "Other pattern",
    ]
    state_counts = {state: 0 for state in state_order}

    for row in equality_cross_df.select(
        ["name_equal", "basic_equal", "cleansed_equal", "rows"]
    ).to_dicts():
        n = bool(row["name_equal"])
        b = bool(row["basic_equal"])
        c = bool(row["cleansed_equal"])
        rows = int(row["rows"])

        if n and b and c:
            state = "All equal"
        elif (not n) and b and c:
            state = "Name mismatch only"
        elif (not n) and (not b) and c:
            state = "Name + basic mismatch"
        elif (not n) and (not b) and (not c):
            state = "All mismatch"
        else:
            state = "Other pattern"

        state_counts[state] += rows

    state_rows = [state_counts[s] for s in state_order if state_counts[s] > 0]
    state_labels = [s for s in state_order if state_counts[s] > 0]
    state_pcts = [
        round((r / matched_total) * 100, 2) if matched_total else 0.0
        for r in state_rows
    ]

    return pl.DataFrame(
        {"state": state_labels, "rows": state_rows, "pct_of_matched": state_pcts}
    )


def _render_matched_overlap_states_chart(
    overlap_state_df: pl.DataFrame, output_path: Path, *, context_label: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    states = overlap_state_df["state"].to_list()
    pcts = overlap_state_df["pct_of_matched"].to_list()
    rows = overlap_state_df["rows"].to_list()
    ax.bar(states, pcts)
    ax.set_title(f"Matched Rows by Overlap State ({context_label})")
    ax.set_xlabel("State")
    ax.set_ylabel("Share of matched records (%)")
    ax.set_ylim(0, 100)
    for i, (pct, row_count) in enumerate(zip(pcts, rows)):
        ax.text(i, pct, f"{pct}% ({row_count:,})", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _classify_unmatched(analysis_df: pl.DataFrame) -> pl.DataFrame:
    unmatched_df = analysis_df.filter(~pl.col("has_match_uri")).with_columns(
        [
            pl.col("source_company_number")
            .fill_null("")
            .alias("source_company_number_safe"),
            pl.col("source_lei").fill_null("").alias("source_lei_safe"),
            pl.col("source_vat").fill_null("").alias("source_vat_safe"),
            pl.col("source_tax_id").fill_null("").alias("source_tax_id_safe"),
        ]
    )

    any_id_expr = (
        (pl.col("source_company_number_safe") != "")
        | (pl.col("source_lei_safe") != "")
        | (pl.col("source_vat_safe") != "")
        | (pl.col("source_tax_id_safe") != "")
    )

    reason_expr = (
        pl.when(pl.col("source_company_number_safe") == "")
        .then(pl.lit("MISSING_COMPANY_NUMBER"))
        .when(~any_id_expr)
        .then(pl.lit("NO_IDENTIFIERS_PRESENT"))
        .otherwise(pl.lit("NO_MATCHING_COMPANY"))
    )

    return unmatched_df.with_columns(reason_expr.alias("no_match_reason"))


def _compute_reason_counts(unmatched_df: pl.DataFrame) -> pl.DataFrame:
    reason_counts = (
        unmatched_df.group_by("no_match_reason")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
    )
    return reason_counts.with_columns(
        (pl.col("rows") / pl.col("rows").sum() * 100).round(2).alias("pct")
    )


def _render_no_match_reasons_chart(
    reason_counts: pl.DataFrame, output_path: Path, *, context_label: str
) -> None:
    labels = reason_counts["no_match_reason"].to_list()
    values = reason_counts["rows"].to_list()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values)
    ax.set_title(f"No-Match Reasons ({context_label})")
    ax.set_xlabel("Rows")
    for i, v in enumerate(values):
        ax.text(v, i, str(v), va="center")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_unmatched_examples(unmatched_df: pl.DataFrame) -> pl.DataFrame:
    return unmatched_df.select(
        [
            "no_match_reason",
            *COMMON_MATCH_PROJECTION,
            *SOURCE_COMPARISON_PROJECTION,
            *UNMATCHED_AUDIT_PROJECTION,
        ]
    ).sort("no_match_reason")


def _build_name_disagree(
    matched_only_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    matched_name_disagree_df = matched_only_df.filter(
        ~pl.col("name_equal")
    ).with_columns(
        [
            pl.col("source_name").fill_null("").alias("source_name_safe"),
            pl.col("target_name").fill_null("").alias("target_name_safe"),
        ]
    )

    name_disagree_reason_expr = (
        pl.when((pl.col("source_name_safe") == "") & (pl.col("target_name_safe") == ""))
        .then(pl.lit("BOTH_NAMES_MISSING"))
        .when(pl.col("source_name_safe") == "")
        .then(pl.lit("SOURCE_NAME_MISSING"))
        .when(pl.col("target_name_safe") == "")
        .then(pl.lit("TARGET_NAME_MISSING"))
        .otherwise(pl.lit("BOTH_PRESENT_DIFFERENT"))
    )

    matched_name_disagree_df = matched_name_disagree_df.with_columns(
        name_disagree_reason_expr.alias("name_disagree_reason")
    )

    reason_counts = (
        matched_name_disagree_df.group_by("name_disagree_reason")
        .agg(pl.len().alias("rows"))
        .with_columns(
            (pl.col("rows") / pl.col("rows").sum() * 100)
            .round(2)
            .alias("pct_of_name_disagree"),
        )
        .sort("rows", descending=True)
    )

    examples = matched_name_disagree_df.select(
        [
            "name_disagree_reason",
            *COMMON_MATCH_PROJECTION,
            *SOURCE_COMPARISON_PROJECTION,
            *MATCH_RESULT_FLAGS_PROJECTION,
            *MATCHED_RIGHT_ID_PROJECTION,
            "has_match_uri",
        ]
    ).sort("name_disagree_reason")

    examples_per_reason = examples.group_by(
        "name_disagree_reason", maintain_order=True
    ).head(30)

    return reason_counts, examples_per_reason


def _compute_no_match_recoverability(
    analysis_df: pl.DataFrame,
    target_df: pl.DataFrame,
    *,
    name_col: str,
    basic_col: str,
    cleansed_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    no_match_source_df = analysis_df.with_row_index("source_row_id").filter(
        ~pl.col("has_match_uri")
    )

    name_lookup_df = target_df.select(
        pl.col(name_col).drop_nulls().unique().alias("name_key")
    )
    basic_lookup_df = target_df.select(
        pl.col(basic_col).drop_nulls().unique().alias("basic_key")
    )
    cleansed_lookup_df = target_df.select(
        pl.col(cleansed_col).drop_nulls().unique().alias("cleansed_key")
    )

    name_hit_ids = (
        no_match_source_df.join(
            name_lookup_df, left_on="source_name", right_on="name_key", how="semi"
        )
        .select("source_row_id")
        .with_columns(pl.lit(True).alias("can_recover_by_name"))
    )
    basic_hit_ids = (
        no_match_source_df.join(
            basic_lookup_df,
            left_on="source_basic_name",
            right_on="basic_key",
            how="semi",
        )
        .select("source_row_id")
        .with_columns(pl.lit(True).alias("can_recover_by_basic"))
    )
    cleansed_hit_ids = (
        no_match_source_df.join(
            cleansed_lookup_df,
            left_on="source_cleansed_name",
            right_on="cleansed_key",
            how="semi",
        )
        .select("source_row_id")
        .with_columns(pl.lit(True).alias("can_recover_by_cleansed"))
    )

    no_match_recoverability_df = (
        no_match_source_df.join(name_hit_ids, on="source_row_id", how="left")
        .join(basic_hit_ids, on="source_row_id", how="left")
        .join(cleansed_hit_ids, on="source_row_id", how="left")
        .with_columns(
            [
                pl.col("can_recover_by_name").fill_null(False),
                pl.col("can_recover_by_basic").fill_null(False),
                pl.col("can_recover_by_cleansed").fill_null(False),
            ]
        )
    )
    no_match_recoverability_df = no_match_recoverability_df.with_columns(
        (
            ~(
                pl.col("can_recover_by_name")
                | pl.col("can_recover_by_basic")
                | pl.col("can_recover_by_cleansed")
            )
        ).alias("not_recoverable")
    )

    no_match_total = no_match_recoverability_df.height
    strategy_df = (
        pl.DataFrame(
            {
                "strategy": ["name", "basic", "cleansed"],
                "recoverable_rows": [
                    no_match_recoverability_df.filter(
                        pl.col("can_recover_by_name")
                    ).height,
                    no_match_recoverability_df.filter(
                        pl.col("can_recover_by_basic")
                    ).height,
                    no_match_recoverability_df.filter(
                        pl.col("can_recover_by_cleansed")
                    ).height,
                ],
            }
        )
        .with_columns(
            pl.when(pl.lit(no_match_total) > 0)
            .then((pl.col("recoverable_rows") / pl.lit(no_match_total) * 100).round(2))
            .otherwise(0.0)
            .alias("recoverable_pct")
        )
        .with_columns(
            (pl.lit(no_match_total) - pl.col("recoverable_rows")).alias(
                "not_recoverable_rows"
            )
        )
    )

    return no_match_recoverability_df, strategy_df


def _render_no_match_recoverability_chart(
    strategy_df: pl.DataFrame, output_path: Path, *, context_label: str
) -> None:
    strategy_labels = strategy_df["strategy"].to_list()
    recoverable_rows = strategy_df["recoverable_rows"].to_list()
    not_recoverable_rows = strategy_df["not_recoverable_rows"].to_list()
    x = list(range(len(strategy_labels)))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        [i - bar_w / 2 for i in x], recoverable_rows, width=bar_w, label="Recoverable"
    )
    ax.bar(
        [i + bar_w / 2 for i in x],
        not_recoverable_rows,
        width=bar_w,
        label="Not recoverable",
    )
    ax.set_title(f"No-Match Recoverability by Strategy ({context_label})")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Rows")
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels)
    ax.legend()
    for i, v in enumerate(recoverable_rows):
        ax.text(i - bar_w / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(not_recoverable_rows):
        ax.text(i + bar_w / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _project_target_by_key(
    target_df: pl.DataFrame, key_col: str, key_alias: str
) -> pl.DataFrame:
    target_prefixed_cols = [pl.col(c).alias(f"target_{c}") for c in target_df.columns]
    return target_df.with_row_index("target_row_id").select(
        pl.col(key_col).alias(key_alias),
        pl.col("target_row_id"),
        *target_prefixed_cols,
    )


def _project_examples(df: pl.DataFrame) -> pl.DataFrame:
    target_output_columns = [
        column_name
        for column_name in EXAMPLE_SELECT_OUTPUT_NAMES
        if column_name.startswith("target_")
    ]
    overwrite_exprs = [
        pl.col(f"{column_name}_right").alias(column_name)
        for column_name in target_output_columns
        if f"{column_name}_right" in df.columns
    ]
    if overwrite_exprs:
        df = df.with_columns(overwrite_exprs)
    missing_columns = [
        column_name
        for column_name in EXAMPLE_SELECT_OUTPUT_NAMES
        if column_name not in df.columns
    ]
    if missing_columns:
        df = df.with_columns(
            [pl.lit(None).alias(column_name) for column_name in missing_columns]
        )
    return df.select(EXAMPLE_SELECT_COLS)


def _build_no_match_recoverability_examples(
    no_match_recoverability_df: pl.DataFrame,
    target_df: pl.DataFrame,
    *,
    name_col: str,
    basic_col: str,
    cleansed_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    target_by_name_df = _project_target_by_key(target_df, name_col, "_name_key")
    target_by_basic_df = _project_target_by_key(target_df, basic_col, "_basic_key")
    target_by_cleansed_df = _project_target_by_key(
        target_df, cleansed_col, "_cleansed_key"
    )

    name_examples = (
        no_match_recoverability_df.filter(pl.col("can_recover_by_name"))
        .join(
            target_by_name_df, left_on="source_name", right_on="_name_key", how="inner"
        )
        .group_by("source_row_id", maintain_order=True)
        .head(1)
        .with_columns(pl.lit("can_recover_by_name").alias("category"))
    )
    basic_examples = (
        no_match_recoverability_df.filter(pl.col("can_recover_by_basic"))
        .join(
            target_by_basic_df,
            left_on="source_basic_name",
            right_on="_basic_key",
            how="inner",
        )
        .group_by("source_row_id", maintain_order=True)
        .head(1)
        .with_columns(pl.lit("can_recover_by_basic").alias("category"))
    )
    cleansed_examples = (
        no_match_recoverability_df.filter(pl.col("can_recover_by_cleansed"))
        .join(
            target_by_cleansed_df,
            left_on="source_cleansed_name",
            right_on="_cleansed_key",
            how="inner",
        )
        .group_by("source_row_id", maintain_order=True)
        .head(1)
        .with_columns(pl.lit("can_recover_by_cleansed").alias("category"))
    )

    return (
        _project_examples(name_examples),
        _project_examples(basic_examples),
        _project_examples(cleansed_examples),
    )


def run_match_analysis(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    target_system: str,
    country: str | None = None,
    target_display: str = "",
    max_rows: int | None = None,
    run_date: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> MatchAnalysisPaths:
    source_system = source_system.strip().lower()
    target_system = target_system.strip().lower()
    country = (country or target_system).strip().lower()
    target_display = target_display.strip()

    effective_run_date = run_date or datetime.now().astimezone().date().isoformat()
    paths = resolve_match_analysis_paths(
        roots,
        run_date=effective_run_date,
        source_system=source_system,
        target_system=target_system,
        country=country,
        target_display=target_display,
    )
    context_label = paths.context_label

    root_path = roots
    matched_dir = system_layer_dir(root_path, source_system, layer=MATCHED_LAYER_NAME)
    source_lookup_dir = system_layer_dir(
        root_path, source_system, layer=CLEANSED_LAYER_NAME
    )
    target_lookup_dir = system_layer_dir(
        root_path, target_system, layer=CLEANSED_LAYER_NAME
    )

    _emit_progress(progress, f"match_analysis scenario={paths.scenario} start")

    matched_df = _load_country_frame(matched_dir, country, max_rows)
    source_df = _load_country_frame(source_lookup_dir, country, max_rows)
    target_df = _load_country_frame(target_lookup_dir, country, max_rows)
    _emit_progress(
        progress,
        f"match_analysis scenario={paths.scenario} loaded "
        f"matched={matched_df.height} source={source_df.height} target={target_df.height}",
    )

    _validate_match_schema(source_df, target_df, matched_df)

    name_col = "name"
    basic_col = "name_cleansed_basic"
    cleansed_col = "name_cleansed"

    source_subset = _project_side_subset(
        source_df,
        side="source",
        name_col=name_col,
        basic_col=basic_col,
        cleansed_col=cleansed_col,
    )
    target_subset = _project_side_subset(
        target_df,
        side="target",
        name_col=name_col,
        basic_col=basic_col,
        cleansed_col=cleansed_col,
    )
    analysis_df = _build_analysis_frame(matched_df, source_subset, target_subset)

    summary_df = _compute_summary(analysis_df)
    matched_only_df = analysis_df.filter(pl.col("has_match_uri"))
    (
        name_breakdown_df,
        basic_breakdown_df,
        cleansed_breakdown_df,
        equality_cross_df,
        _disagree_count,
    ) = _compute_equality_breakdowns(matched_only_df)

    _emit_progress(progress, f"match_analysis scenario={paths.scenario} stage=charts")
    _render_match_outcome_chart(
        analysis_df, paths.match_outcome_chart_path, context_label=context_label
    )
    _render_equality_rates_chart(
        name_breakdown_df,
        basic_breakdown_df,
        cleansed_breakdown_df,
        paths.equality_rates_chart_path,
        context_label=context_label,
    )
    _render_equality_overlap_chart(
        matched_only_df, paths.equality_overlap_venn_path, context_label=context_label
    )
    overlap_state_df = _compute_matched_overlap_states(
        equality_cross_df, matched_only_df.height
    )
    _render_matched_overlap_states_chart(
        overlap_state_df,
        paths.matched_overlap_states_chart_path,
        context_label=context_label,
    )

    unmatched_df = _classify_unmatched(analysis_df)
    reason_counts = _compute_reason_counts(unmatched_df)
    _render_no_match_reasons_chart(
        reason_counts, paths.no_match_reasons_chart_path, context_label=context_label
    )
    examples_df = _build_unmatched_examples(unmatched_df)

    name_disagree_summary_df, name_disagree_examples_df = _build_name_disagree(
        matched_only_df
    )

    no_match_recoverability_df, no_match_strategy_df = _compute_no_match_recoverability(
        analysis_df,
        target_df,
        name_col=name_col,
        basic_col=basic_col,
        cleansed_col=cleansed_col,
    )
    _render_no_match_recoverability_chart(
        no_match_strategy_df,
        paths.no_match_recoverability_chart_path,
        context_label=context_label,
    )
    (
        no_match_name_examples_df,
        no_match_basic_examples_df,
        no_match_cleansed_examples_df,
    ) = _build_no_match_recoverability_examples(
        no_match_recoverability_df,
        target_df,
        name_col=name_col,
        basic_col=basic_col,
        cleansed_col=cleansed_col,
    )

    _emit_progress(
        progress, f"match_analysis scenario={paths.scenario} stage=write_artifacts"
    )
    summary_df.write_parquet(paths.summary_path)
    reason_counts.write_parquet(paths.reasons_path)
    examples_df.write_parquet(paths.examples_path)
    name_disagree_summary_df.write_parquet(paths.name_disagree_summary_path)
    name_disagree_examples_df.write_parquet(paths.name_disagree_examples_path)
    no_match_name_examples_df.write_parquet(paths.no_match_name_examples_path)
    no_match_basic_examples_df.write_parquet(paths.no_match_basic_examples_path)
    no_match_cleansed_examples_df.write_parquet(paths.no_match_cleansed_examples_path)
    no_match_strategy_df.write_parquet(paths.no_match_strategy_path)
    no_match_recoverability_df.select(
        [
            "system_uri",
            "source_jurisdiction_code",
            "source_company_number",
            "source_lei",
            "source_vat",
            "source_tax_id",
            *NO_MATCH_RECOVERABILITY_FLAGS,
        ]
    ).write_parquet(paths.no_match_recoverability_detail_path)

    _emit_progress(progress, f"match_analysis scenario={paths.scenario} complete")
    return paths


def run_match_analysis_batch(
    roots: WorkspaceRoots,
    scenarios: Iterable[dict[str, object]],
    *,
    run_date: str | None = None,
    max_rows: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[MatchAnalysisPaths]:
    effective_run_date = run_date or datetime.now().astimezone().date().isoformat()
    results: list[MatchAnalysisPaths] = []
    for scenario in scenarios:
        country = scenario.get("country")
        results.append(
            run_match_analysis(
                roots,
                source_system=str(scenario["source_system"]),
                target_system=str(scenario["target_system"]),
                country=str(country) if country else None,
                target_display=str(scenario.get("target_display", "")),
                max_rows=scenario.get("max_rows", max_rows),  # type: ignore[arg-type]
                run_date=effective_run_date,
                progress=progress,
            )
        )
    return results
