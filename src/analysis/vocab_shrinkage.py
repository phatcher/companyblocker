"""Forecast what a vocabulary pooled across corpora drops from each corpus.

For each system and a log-spaced grid of vocabulary budgets V, compares the corpus's own top-V raw-tier tokens with a pooled top-V, weighted by count (summed document frequency) or equally (summed per-corpus occurrence share, so a large corpus does not crowd a small one out), and reports the occurrence mass of the corpus's own top-V the pool drops. Draws the dropped mass against V per weighting and, at a chosen V, an overlap diagram of the per-corpus top-V sets split into in-language and out-of-vocabulary panels, where a token is in-language if `wordfreq` recognizes it in any compared corpus's language. Writes under `artifacts/analysis/vocab_shrinkage/runs/<run_date>/`. Run through `scripts/analyze_vocab_shrinkage.py`.

It is a forecast, not a measurement: a subword tokenizer fragments a dropped word rather than losing it, and fertility is what the tokenizer runs measure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import polars as pl
from wordfreq import zipf_frequency

from analysis.noise_layers import (
    RAW_TIER,
    load_zipf_tier_stats,
    project_cross_corpus_tokens,
    render_overlap_diagrams,
    summarize_layer_regions,
)
from analysis.report_layout import metrics_dir as _metrics_dir
from analysis.report_layout import viz_dir as _viz_dir
from analysis.token_rarity import resolve_system_language
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The two ways a shared vocabulary budget can be split across corpora before
# ranking which tokens win a slot: "count" sums each corpus's raw
# document_frequency, so a large corpus's tokens dominate the pool (`fr`'s
# 12.9M rows crowd `ie`'s 0.8M out of a count-ranked pool); "equal" sums each
# corpus's own document_frequency_pct instead, so every named corpus contributes the same
# total weight (1.0) regardless of its absolute size, and a token that is
# proportionally important to a small corpus can outrank one that is merely
# large in absolute terms.
POOL_WEIGHTS: tuple[str, ...] = ("count", "equal")

# This module's own default budget grid, used only when a caller (the CLI)
# doesn't pass an explicit one. Not tied to any corpus's real vocabulary
# size -- `run_vocab_shrinkage_analysis` widens `DEFAULT_V_MAX_FLOOR` to the
# largest named corpus's actual raw-tier vocab size when computing the
# effective grid.
DEFAULT_V_MIN = 100
DEFAULT_V_MAX_FLOOR = 100
DEFAULT_BUDGET_POINTS = 12

# The three overlap-diagram layers `build_top_v_overlap_membership` produces,
# reusing `noise_layers.render_overlap_diagrams`/`summarize_layer_regions`'s
# own `layer`-keyed membership-table contract rather than a separate drawing
# path: `UNSPLIT_LAYER` is every named corpus's own top-V set;
# `IN_LANGUAGE_LAYER`/`OOV_LAYER` split the corpora with a resolved reference
# language by wordfreq recognition.
UNSPLIT_LAYER = "all"
IN_LANGUAGE_LAYER = "in_language"
OOV_LAYER = "oov"


@dataclass(frozen=True)
class VocabShrinkagePaths:
    run_root: Path
    metrics_dir: Path
    viz_dir: Path


def resolve_vocab_shrinkage_paths(
    roots: WorkspaceRoots, run_date: str
) -> VocabShrinkagePaths:
    run_root = analysis_report_run_dir(roots, "vocab_shrinkage", run_date)
    metrics_dir = _metrics_dir(run_root)
    viz_dir = _viz_dir(run_root)
    for directory in (metrics_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return VocabShrinkagePaths(
        run_root=run_root, metrics_dir=metrics_dir, viz_dir=viz_dir
    )


def build_budget_grid(
    v_min: int, v_max: int, num_points: int = DEFAULT_BUDGET_POINTS
) -> list[int]:
    """A log-spaced integer grid of vocabulary budgets from `v_min` to
    `v_max` inclusive, deduplicated after rounding (so a narrow
    `[v_min, v_max]` range with many `num_points` doesn't produce repeated
    values)."""
    if v_min <= 0 or v_max <= 0:
        raise ValueError("v_min and v_max must be positive")
    if v_min > v_max:
        raise ValueError("v_min must be <= v_max")
    if num_points < 1:
        raise ValueError("num_points must be >= 1")
    if num_points == 1 or v_min == v_max:
        return [v_max]

    log_min, log_max = math.log10(v_min), math.log10(v_max)
    raw_values = [
        10 ** (log_min + (log_max - log_min) * step / (num_points - 1))
        for step in range(num_points)
    ]
    grid = sorted({max(1, round(value)) for value in raw_values})
    grid[0] = v_min
    grid[-1] = v_max
    return sorted(set(grid))


def _rank_by_document_frequency(stats: pl.DataFrame, *, rank_col: str) -> pl.DataFrame:
    """Sort `stats` descending by `document_frequency`, tie-broken by token
    ascending for determinism, and attach a 1-based `rank_col`."""
    ordered = stats.sort(["document_frequency", "token"], descending=[True, False])
    return ordered.with_row_index(rank_col, offset=1)


def build_pooled_ranking(
    projected: pl.DataFrame,
    systems: Sequence[str],
    *,
    weight: str,
    tier: str = RAW_TIER,
) -> pl.DataFrame:
    """The pooled token ranking across `systems`'s `tier` vocabulary: token,
    `pooled_score`, and a 1-based `rank` (ties broken by token ascending).

    `weight="count"` ranks by summed `document_frequency` (raw occurrence
    counts, so a large corpus's tokens dominate); `weight="equal"` ranks by
    summed `document_frequency_pct` (each corpus's own occurrence share, so
    every named corpus contributes the same total weight regardless of its
    absolute size). See `POOL_WEIGHTS`'s own docstring for why both exist.
    """
    if weight not in POOL_WEIGHTS:
        raise ValueError(f"weight must be one of {POOL_WEIGHTS}, got {weight!r}")

    score_col = "document_frequency" if weight == "count" else "document_frequency_pct"
    subset = projected.filter(
        (pl.col("tier") == tier) & pl.col("system").is_in(list(systems))
    ).select("token", score_col)
    if subset.height == 0:
        return pl.DataFrame(
            schema={"token": pl.Utf8, "pooled_score": pl.Float64, "rank": pl.UInt32}
        )

    pooled = subset.group_by("token").agg(
        pl.col(score_col).cast(pl.Float64).sum().alias("pooled_score")
    )
    ordered = pooled.sort(["pooled_score", "token"], descending=[True, False])
    return ordered.with_row_index("rank", offset=1)


def _empty_curve_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "system": pl.Utf8,
            "pool_weight": pl.Utf8,
            "v": pl.Int64,
            "corpus_top_v_size": pl.Int64,
            "dropped_token_count": pl.Int64,
            "dropped_mass": pl.Float64,
        }
    )


def compute_dropped_mass_curve(
    projected: pl.DataFrame,
    *,
    systems: Sequence[str],
    budgets: Sequence[int],
    tier: str = RAW_TIER,
) -> pl.DataFrame:
    """Per system, pool weight, and budget `v`: the occurrence mass (sum of
    that corpus's own `document_frequency_pct`) of its own top-`v` tokens
    that the `weight`-pooled top-`v` set drops, plus the corpus's own top-`v`
    token count and how many of those were dropped.

    A token is dropped at `v` when its pooled rank is missing (never ranked
    in the pool -- shouldn't happen for a token present in a named corpus's
    own vocabulary, since the pool is built from the union of the same
    corpora) or greater than `v`. One row per (system, pool_weight, v).
    """
    if not budgets or not systems:
        return _empty_curve_frame()

    pooled_rankings = {
        weight: build_pooled_ranking(projected, systems, weight=weight, tier=tier)
        for weight in POOL_WEIGHTS
    }

    rows: list[dict[str, object]] = []
    for system in systems:
        own = projected.filter(
            (pl.col("system") == system) & (pl.col("tier") == tier)
        ).select("token", "document_frequency", "document_frequency_pct")
        if own.height == 0:
            continue
        own_ranked = _rank_by_document_frequency(own, rank_col="own_rank")

        for weight in POOL_WEIGHTS:
            pool_lookup = pooled_rankings[weight].select(
                "token", pl.col("rank").alias("pooled_rank")
            )
            joined = own_ranked.join(pool_lookup, on="token", how="left")

            for v in budgets:
                subset = joined.filter(pl.col("own_rank") <= v)
                dropped = subset.filter(
                    pl.col("pooled_rank").is_null() | (pl.col("pooled_rank") > v)
                )
                dropped_mass = (
                    float(dropped.get_column("document_frequency_pct").sum())
                    if dropped.height
                    else 0.0
                )
                rows.append(
                    {
                        "system": system,
                        "pool_weight": weight,
                        "v": v,
                        "corpus_top_v_size": subset.height,
                        "dropped_token_count": dropped.height,
                        "dropped_mass": dropped_mass,
                    }
                )

    if not rows:
        return _empty_curve_frame()
    return pl.DataFrame(rows).sort(["system", "pool_weight", "v"])


def draw_dropped_mass_curve(
    ax: plt.Axes,
    curve: pl.DataFrame,
    *,
    pool_weight: str,
    title: str,
) -> None:
    """One line per system, dropped mass against budget V on a log x-axis,
    for a single pool weight."""
    subset = curve.filter(pl.col("pool_weight") == pool_weight)
    systems = list(dict.fromkeys(subset.get_column("system").to_list()))
    if not systems:
        ax.set_title(title)
        return

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#2E5EAA"])
    for index, system in enumerate(systems):
        system_rows = subset.filter(pl.col("system") == system).sort("v")
        ax.plot(
            system_rows.get_column("v").to_list(),
            system_rows.get_column("dropped_mass").to_list(),
            marker="o",
            color=color_cycle[index % len(color_cycle)],
            label=system,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Vocabulary budget V (log)")
    ax.set_ylabel("Dropped occurrence mass")
    ax.set_title(title)
    ax.legend(fontsize=8)


def render_dropped_mass_curves(
    curve: pl.DataFrame, output_dir: Path
) -> dict[str, Path]:
    """Render one dropped-mass-vs-V curve file per pool weight present in
    `curve` (skips a weight with no rows)."""
    outputs: dict[str, Path] = {}
    if curve.height == 0:
        return outputs
    for weight in POOL_WEIGHTS:
        if curve.filter(pl.col("pool_weight") == weight).height == 0:
            continue
        output_path = output_dir / f"_dropped_mass_{weight}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_dropped_mass_curve(
            ax,
            curve,
            pool_weight=weight,
            title=f"Dropped occurrence mass vs vocabulary budget -- {weight}-weighted pool",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs[weight] = output_path
    return outputs


def classify_tokens_in_language(
    tokens: Iterable[str], languages: Iterable[str]
) -> set[str]:
    """Tokens wordfreq recognizes (nonzero `zipf_frequency`) in at least one
    of `languages` -- the "any member language" rule for a token shared
    across corpora with different reference languages, so a legal-form/brand
    token recognized in one member language but not another still lands on
    the in-language side consistently everywhere it appears."""
    language_list = list(languages)
    if not language_list:
        return set()
    return {
        token
        for token in tokens
        if any(
            zipf_frequency(str(token).lower(), language) > 0.0
            for language in language_list
        )
    }


def _corpus_top_v_frame(
    projected: pl.DataFrame, system: str, v: int, *, tier: str
) -> pl.DataFrame:
    own = projected.filter(
        (pl.col("system") == system) & (pl.col("tier") == tier)
    ).select("token", "document_frequency", "document_frequency_pct")
    if own.height == 0:
        return own.with_columns(pl.lit(system, dtype=pl.Utf8).alias("system"))
    ranked = _rank_by_document_frequency(own, rank_col="own_rank")
    top = ranked.filter(pl.col("own_rank") <= v).drop("own_rank")
    return top.with_columns(pl.lit(system).alias("system"))


def _empty_top_v_membership_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "layer": pl.Utf8,
            "system": pl.Utf8,
            "token": pl.Utf8,
            "document_frequency": pl.Int64,
            "document_frequency_pct": pl.Float64,
            "region": pl.Utf8,
        }
    )


def _attach_region_column(frame: pl.DataFrame) -> pl.DataFrame:
    """Same region-derivation rule as `noise_layers.compute_layer_removals`:
    the `+`-joined, sorted set of systems sharing a token, grouped on
    `token` alone since each caller here already scopes `frame` to one
    layer."""
    if frame.height == 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("region"))
    region_map = (
        frame.group_by("token")
        .agg(pl.col("system").unique().sort().alias("_systems"))
        .with_columns(pl.col("_systems").list.join("+").alias("region"))
        .select("token", "region")
    )
    return frame.join(region_map, on="token", how="left")


_MEMBERSHIP_COLUMNS = (
    "layer",
    "system",
    "token",
    "document_frequency",
    "document_frequency_pct",
    "region",
)


def build_top_v_overlap_membership(
    projected: pl.DataFrame,
    *,
    systems: Sequence[str],
    v: int,
    tier: str = RAW_TIER,
) -> pl.DataFrame:
    """One row per (layer, system, token) for the top-V overlap diagram at a
    chosen budget `v`, in the same `layer`-keyed membership shape
    `noise_layers.render_overlap_diagrams`/`summarize_layer_regions` already
    draw from: `UNSPLIT_LAYER` covers every named corpus's own top-`v`
    tokens; `IN_LANGUAGE_LAYER`/`OOV_LAYER` cover only the corpora with a
    resolved reference language (`resolve_system_language`), split by
    `classify_tokens_in_language` against every member language at once --
    so a corpus with no mapped language contributes to `UNSPLIT_LAYER` only
    and never to the two split layers, and the split layers are omitted
    entirely when fewer than two named corpora have a mapped language (an
    overlap diagram needs at least two sets).
    """
    top_frames = [
        _corpus_top_v_frame(projected, system, v, tier=tier) for system in systems
    ]
    top_frames = [frame for frame in top_frames if frame.height > 0]
    if not top_frames:
        return _empty_top_v_membership_frame()

    combined = pl.concat(top_frames, how="vertical")
    frames = [
        _attach_region_column(combined).with_columns(
            pl.lit(UNSPLIT_LAYER).alias("layer")
        )
    ]

    member_systems = [system for system in systems if resolve_system_language(system)]
    if len(member_systems) >= 2:
        languages = sorted(
            {
                language
                for system in member_systems
                if (language := resolve_system_language(system)) is not None
            }
        )
        mapped = combined.filter(pl.col("system").is_in(member_systems))
        tokens = mapped.get_column("token").unique().to_list()
        in_language_tokens = classify_tokens_in_language(tokens, languages)

        in_language_frame = mapped.filter(
            pl.col("token").is_in(list(in_language_tokens))
        )
        if in_language_frame.height > 0:
            frames.append(
                _attach_region_column(in_language_frame).with_columns(
                    pl.lit(IN_LANGUAGE_LAYER).alias("layer")
                )
            )

        oov_frame = mapped.filter(~pl.col("token").is_in(list(in_language_tokens)))
        if oov_frame.height > 0:
            frames.append(
                _attach_region_column(oov_frame).with_columns(
                    pl.lit(OOV_LAYER).alias("layer")
                )
            )

    return pl.concat(
        [frame.select(*_MEMBERSHIP_COLUMNS) for frame in frames], how="vertical"
    )


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        print(f"[vocab_shrinkage] {message}")


def run_vocab_shrinkage_analysis(
    roots: WorkspaceRoots,
    *,
    systems: Sequence[str],
    run_date: str | None = None,
    zipf_run_date: str | None = None,
    budgets: Sequence[int] | None = None,
    v_min: int = DEFAULT_V_MIN,
    v_max: int | None = None,
    budget_points: int = DEFAULT_BUDGET_POINTS,
    chosen_v: int | None = None,
    tier: str = RAW_TIER,
    progress: Callable[[str], None] | None = None,
) -> VocabShrinkagePaths:
    """Run the full pooled-tokenizer vocabulary-shrinkage forecast for the
    named systems: project their `tier` token stats into one table (reusing
    `analysis.noise_layers.project_cross_corpus_tokens`), compute the
    dropped-occurrence-mass curve against a budget grid for both pool
    weights, and render the curves plus one chosen-V overlap diagram (Venn/
    UpSet, unsplit and in-language/OOV split) -- all under its own run tree
    beside `noise_layers`'s and `token_zipf`'s.

    `zipf_run_date` selects which existing `analysis.token_zipf` run to read
    per-tier stats from (defaults to `run_date`); it does not run
    `token_zipf` itself. `budgets` overrides the log-spaced grid the CLI
    otherwise builds from `v_min`/`v_max`/`budget_points` (`v_max` defaults
    to the largest named corpus's own `tier` vocab size). `chosen_v`
    defaults to the largest budget in the effective grid.
    """
    effective_run_date = run_date or datetime.now(UTC).date().isoformat()
    effective_zipf_run_date = zipf_run_date or effective_run_date
    paths = resolve_vocab_shrinkage_paths(roots, effective_run_date)

    tier_stats = load_zipf_tier_stats(
        roots, effective_zipf_run_date, systems, tiers=(tier,)
    )
    projected = project_cross_corpus_tokens(tier_stats)
    _emit_progress(
        progress,
        f"projected {projected.height:,} token rows across {len(systems)} systems",
    )

    if budgets is not None:
        effective_budgets = sorted(set(budgets))
    else:
        max_vocab = max(
            (entry.stats.height for entry in tier_stats if entry.tier == tier),
            default=0,
        )
        effective_v_max = (
            v_max if v_max is not None else max(max_vocab, v_min, DEFAULT_V_MAX_FLOOR)
        )
        effective_budgets = build_budget_grid(v_min, effective_v_max, budget_points)

    curve = compute_dropped_mass_curve(
        projected, systems=systems, budgets=effective_budgets, tier=tier
    )
    curve.write_parquet(paths.metrics_dir / "dropped_mass_curve.parquet")
    _emit_progress(
        progress,
        f"dropped-mass curve: {curve.height:,} rows across {len(effective_budgets)} budgets",
    )

    curve_paths = render_dropped_mass_curves(curve, paths.viz_dir)
    for weight, curve_path in curve_paths.items():
        _emit_progress(progress, f"dropped-mass curve weight={weight} -> {curve_path}")

    effective_chosen_v = chosen_v if chosen_v is not None else effective_budgets[-1]
    membership = build_top_v_overlap_membership(
        projected, systems=systems, v=effective_chosen_v, tier=tier
    )
    membership.write_parquet(paths.metrics_dir / "top_v_membership.parquet")
    _emit_progress(
        progress,
        f"top-V membership at v={effective_chosen_v}: {membership.height:,} rows",
    )

    region_frames = [
        summarize_layer_regions(membership, layer=layer).with_columns(
            pl.lit(layer).alias("layer")
        )
        for layer in sorted(membership.get_column("layer").unique().to_list())
    ]
    region_table = (
        pl.concat(region_frames, how="vertical")
        if region_frames
        else pl.DataFrame(
            schema={
                "region": pl.Utf8,
                "system": pl.Utf8,
                "token_count": pl.UInt32,
                "mass": pl.Float64,
                "layer": pl.Utf8,
            }
        )
    )
    region_table.write_parquet(paths.metrics_dir / "top_v_region_summary.parquet")

    corpus_totals = {
        entry.system: (entry.stats.height, entry.total_docs)
        for entry in tier_stats
        if entry.tier == tier
    }
    overlap_paths = render_overlap_diagrams(membership, corpus_totals, paths.viz_dir)
    for layer, overlap_path in overlap_paths.items():
        _emit_progress(
            progress,
            f"overlap diagram v={effective_chosen_v} layer={layer} -> {overlap_path}",
        )

    return paths
