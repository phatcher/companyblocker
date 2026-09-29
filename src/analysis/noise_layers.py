"""What each noise layer removes from the names, across corpora.

Projects an existing `token_zipf` run (`--zipf-run-date` picks it; this module does not run it) and each system's tokenizer `token_tfidf_stats.parquet` into one long table of token, system, tier, language, document frequency and IDF. Against the raw tier it measures what each layer removes: the cleanse stage, the packaged default noise-word list, each per-corpus profile, and any list named with `--extra-noise-words LABEL=PATH`, by document-frequency mass beside unique-token count. Draws a bar chart per corpus and an overlap diagram per layer (Venn for two or three corpora, UpSet for four or more), and writes `membership.parquet` with region and Jaccard summaries under `artifacts/analysis/noise_layers/runs/<run_date>/`. Run through `scripts/analyze_noise_layers.py`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import matplotlib
import polars as pl
from company_tokenize.paths import scope_directory_files
from company_tokenize.tfidf import resolve_noise_words

from analysis.report_layout import metrics_dir as _metrics_dir
from analysis.report_layout import viz_dir as _viz_dir
from analysis.token_rarity import resolve_system_language
from analysis.token_zipf import NAME_TIER_COLUMNS
from workspace.artifact_layout import analysis_report_run_dir, tokenizer_scope_dir
from workspace.roots import WorkspaceRoots

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import UpSet, from_contents

# The tier every layer's removed-set/mass arithmetic is measured against
# (raw carries no noise reduction, so it's the one consistent baseline
# every layer -- cleanse-stage or noise-word -- can be compared to). Reuses `token_zipf.NAME_TIER_COLUMNS`'s own key rather than
# redefining the tier name.
RAW_TIER = "raw"

# Column name -> tier reverse lookup, so a tokenizer's `metadata.json`
# `name_col` (e.g. "name_cleansed") can be resolved back to the tier name
# (e.g. "cleansed") the projection table uses.
_COLUMN_TO_TIER: dict[str, str] = {col: tier for tier, col in NAME_TIER_COLUMNS.items()}


@dataclass(frozen=True)
class NoiseLayerPaths:
    run_root: Path
    metrics_dir: Path
    viz_dir: Path


def resolve_noise_layer_paths(roots: WorkspaceRoots, run_date: str) -> NoiseLayerPaths:
    run_root = analysis_report_run_dir(roots, "noise_layers", run_date)
    metrics_dir = _metrics_dir(run_root)
    viz_dir = _viz_dir(run_root)
    for directory in (metrics_dir, viz_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return NoiseLayerPaths(run_root=run_root, metrics_dir=metrics_dir, viz_dir=viz_dir)


@dataclass(frozen=True)
class CorpusTierStats:
    """One (system, tier) vocabulary snapshot: token-level document-frequency
    stats plus the corpus's total document count, as persisted by
    `analysis.token_zipf`'s per-tier run output (`<system>_<tier>_token_stats.parquet`
    plus its `total_docs` from `token_zipf_summary.parquet`)."""

    system: str
    tier: str
    stats: pl.DataFrame  # token, document_frequency, document_frequency_pct
    total_docs: int


@dataclass(frozen=True)
class TokenizerIdfStats:
    """One system's tokenizer-stage TF-IDF stats (`token_tfidf_stats.parquet`),
    tagged with the name-tier its tokenizer `metadata.json`'s `name_col` resolves
    to."""

    system: str
    tier: str
    stats: pl.DataFrame  # token, idf


@dataclass(frozen=True)
class NoiseLayerSpec:
    """One noise-word-based noise layer to measure: a resolved token set plus
    the name it appears under everywhere downstream (bars, membership rows,
    overlap diagrams). The cleanse-stage layer is not represented here -- it
    is a tier diff, not a noise-word set, and is handled directly by
    `compute_layer_removals`."""

    name: str
    noise_words: frozenset[str]


CLEANSE_LAYER = "cleanse"


def _empty_projection_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "token": pl.Utf8,
            "system": pl.Utf8,
            "tier": pl.Utf8,
            "language": pl.Utf8,
            "document_frequency": pl.Int64,
            "document_frequency_pct": pl.Float64,
            "idf": pl.Float64,
        }
    )


def project_cross_corpus_tokens(
    tier_stats: Sequence[CorpusTierStats],
    idf_stats: Sequence[TokenizerIdfStats] = (),
) -> pl.DataFrame:
    """Stack every (system, tier) token-stats snapshot into one long-form
    table: token, system, tier, language, document_frequency,
    document_frequency_pct, idf.

    `tier_stats` is the Zipf run's per-tier parquets (one entry per
    system/tier); `idf_stats` is the tokenizer directories'
    `token_tfidf_stats.parquet`, each tagged with the tier its `name_col`
    resolves to. IDF is attached by a left join on (system, tier, token) --
    null wherever no tokenizer stats file covers that system/tier, exactly
    as for a system with no mapped language (`language` is also null there,
    via `resolve_system_language`).
    """
    if not tier_stats:
        return _empty_projection_frame()

    stacked_frames = [
        entry.stats.select(
            "token",
            pl.col("document_frequency").cast(pl.Int64),
            pl.col("document_frequency_pct").cast(pl.Float64),
        ).with_columns(
            pl.lit(entry.system).alias("system"),
            pl.lit(entry.tier).alias("tier"),
            pl.lit(resolve_system_language(entry.system), dtype=pl.Utf8).alias(
                "language"
            ),
        )
        for entry in tier_stats
    ]
    stacked = pl.concat(stacked_frames, how="vertical")

    if idf_stats:
        idf_frames = [
            entry.stats.select("token", pl.col("idf").cast(pl.Float64)).with_columns(
                pl.lit(entry.system).alias("system"),
                pl.lit(entry.tier).alias("tier"),
            )
            for entry in idf_stats
        ]
        idf_table = pl.concat(idf_frames, how="vertical").unique(
            subset=["system", "tier", "token"], keep="first"
        )
        stacked = stacked.join(idf_table, on=["system", "tier", "token"], how="left")
    else:
        stacked = stacked.with_columns(pl.lit(None, dtype=pl.Float64).alias("idf"))

    return stacked.select(
        "token",
        "system",
        "tier",
        "language",
        "document_frequency",
        "document_frequency_pct",
        "idf",
    )


def load_zipf_tier_stats(
    roots: WorkspaceRoots,
    run_date: str,
    systems: Iterable[str],
    *,
    tiers: Iterable[str] = (RAW_TIER,),
) -> list[CorpusTierStats]:
    """Load the persisted per-tier token stats and `total_docs` for each
    named system/tier from an existing `analysis.token_zipf` run
    (`artifacts/analysis/token_zipf/runs/<run_date>/`). Silently skips a
    system/tier with no persisted file (a system whose cleansed schema lacks
    that tier's column, per `token_zipf`'s own per-system skip)."""
    zipf_root = analysis_report_run_dir(roots, "token_zipf", run_date)
    stats_dir = _metrics_dir(zipf_root)
    summary_path = stats_dir / "token_zipf_summary.parquet"
    total_docs_by_system_tier: dict[tuple[str, str], int] = {}
    if summary_path.exists():
        summary = pl.read_parquet(summary_path)
        for row in summary.select("system", "tier", "total_docs").iter_rows(named=True):
            total_docs_by_system_tier[(row["system"], row["tier"])] = int(
                row["total_docs"]
            )

    entries: list[CorpusTierStats] = []
    for system in systems:
        for tier in tiers:
            stats_path = stats_dir / f"{system}_{tier}_token_stats.parquet"
            if not stats_path.exists():
                continue
            stats = pl.read_parquet(stats_path)
            total_docs = total_docs_by_system_tier.get((system, tier), 0)
            entries.append(
                CorpusTierStats(
                    system=system, tier=tier, stats=stats, total_docs=total_docs
                )
            )
    return entries


def load_tokenizer_idf_stats(
    roots: WorkspaceRoots, systems: Iterable[str]
) -> list[TokenizerIdfStats]:
    """Load each named system's tokenizer-stage `token_tfidf_stats.parquet`,
    tagging it with the tier its `metadata.json`'s `name_col` resolves to via
    `NAME_TIER_COLUMNS`. Skips a system with no stats file, no metadata, or a
    `name_col` outside the standard raw/basic/cleansed contract."""
    entries: list[TokenizerIdfStats] = []
    for system in systems:
        scope_files = scope_directory_files(tokenizer_scope_dir(roots, system=system))
        stats_path = scope_files.tfidf_stats
        metadata_path = scope_files.metadata
        if not stats_path.exists() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        name_col = metadata.get("name_col")
        tier = _COLUMN_TO_TIER.get(name_col)
        if tier is None:
            continue
        stats = pl.read_parquet(stats_path)
        if "idf" not in stats.columns:
            continue
        entries.append(TokenizerIdfStats(system=system, tier=tier, stats=stats))
    return entries


def resolve_noise_layers(
    system: str,
    roots: WorkspaceRoots,
    *,
    extra_noise_word_paths: Sequence[tuple[str, Path]] = (),
) -> list[NoiseLayerSpec]:
    """The noise-word-based layer set for one system, in a stable order:
    `packaged_default`, then `per_corpus_<profile>` for each profile if this
    system has a real generated `noise_words.json`, then `extra_<label>` for
    each `extra_noise_word_paths` entry (how a pooled noise-word candidate
    file generated by a separate corpus-driven pass, or any other noise-word
    file passed by path, is compared as one more layer without replacing
    anything).

    Not `token_zipf.build_trim_settings`: that includes a `none` baseline
    setting (irrelevant here, there's nothing to remove) and its own naming
    doesn't distinguish an `extra` layer from a per-corpus profile.
    """
    layers = [
        NoiseLayerSpec(
            name="packaged_default",
            noise_words=frozenset(
                resolve_noise_words(noise_words_profile="aggressive")
            ),
        )
    ]

    per_corpus_path = scope_directory_files(
        tokenizer_scope_dir(roots, system=system)
    ).noise_words
    if per_corpus_path.exists():
        for profile in ("strict", "balanced", "aggressive"):
            layers.append(
                NoiseLayerSpec(
                    name=f"per_corpus_{profile}",
                    noise_words=frozenset(
                        resolve_noise_words(
                            noise_words_path=per_corpus_path,
                            noise_words_profile=profile,
                        )
                    ),
                )
            )

    for label, path in extra_noise_word_paths:
        layers.append(
            NoiseLayerSpec(
                name=f"extra_{label}",
                noise_words=frozenset(resolve_noise_words(noise_words_path=path)),
            )
        )
    return layers


def _empty_membership_frame() -> pl.DataFrame:
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


def compute_layer_removals(
    projected: pl.DataFrame,
    *,
    systems: Sequence[str],
    cleanse_tier: str,
    layers_by_system: Mapping[str, Sequence[NoiseLayerSpec]],
) -> pl.DataFrame:
    """Per system and layer, the raw-tier tokens the layer removes, with that
    corpus's raw-tier document frequency/pct for each -- one row per (layer,
    system, token), with `region` (the `+`-joined, sorted set of systems
    sharing this token under this layer) attached last so it can be grouped
    on directly.

    Every layer's removed set is computed against the same raw-tier
    baseline (see `RAW_TIER`'s docstring): the cleanse-stage layer is
    `raw_vocab - cleanse_tier_vocab` (a real transformation, not a token-set
    subtraction); every noise-word layer is `raw_vocab & layer.noise_words`
    (what that noise-word set would strip out of the untouched baseline,
    not the production-chained cleansed-then-trimmed result) -- this is
    what makes the layers comparable on one scale instead of measuring each
    against a different upstream tier.
    """
    frames: list[pl.DataFrame] = []
    for system in systems:
        raw_frame = projected.filter(
            (pl.col("system") == system) & (pl.col("tier") == RAW_TIER)
        ).select("token", "document_frequency", "document_frequency_pct")
        if raw_frame.height == 0:
            continue
        raw_vocab = set(raw_frame.get_column("token").to_list())

        tier_frame = projected.filter(
            (pl.col("system") == system) & (pl.col("tier") == cleanse_tier)
        )
        tier_vocab = set(tier_frame.get_column("token").to_list())
        cleanse_removed = raw_vocab - tier_vocab
        frames.append(
            _layer_removal_frame(
                raw_frame, cleanse_removed, layer=CLEANSE_LAYER, system=system
            )
        )

        for layer in layers_by_system.get(system, ()):
            removed = raw_vocab & layer.noise_words
            frames.append(
                _layer_removal_frame(
                    raw_frame, removed, layer=layer.name, system=system
                )
            )

    frames = [frame for frame in frames if frame.height > 0]
    if not frames:
        return _empty_membership_frame()

    removals = pl.concat(frames, how="vertical")
    region_map = (
        removals.group_by(["layer", "token"])
        .agg(pl.col("system").unique().sort().alias("_systems"))
        .with_columns(pl.col("_systems").list.join("+").alias("region"))
        .select("layer", "token", "region")
    )
    return removals.join(region_map, on=["layer", "token"], how="left").select(
        "layer",
        "system",
        "token",
        "document_frequency",
        "document_frequency_pct",
        "region",
    )


def _layer_removal_frame(
    raw_frame: pl.DataFrame, removed_tokens: set[str], *, layer: str, system: str
) -> pl.DataFrame:
    if not removed_tokens:
        return raw_frame.clear()
    return raw_frame.filter(pl.col("token").is_in(list(removed_tokens))).with_columns(
        pl.lit(layer).alias("layer"), pl.lit(system).alias("system")
    )


def compute_corpus_layer_metrics(
    projected: pl.DataFrame,
    membership: pl.DataFrame,
    *,
    layer_names_by_system: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """Per system and layer: raw-tier vocab size, unique tokens removed and
    remaining, and removed document-frequency mass (the sum of raw-tier
    `document_frequency_pct` over the removed tokens) -- the table both bar
    charts are drawn from. `layer_names_by_system` (including `CLEANSE_LAYER`)
    is passed explicitly rather than inferred from `membership` so a layer
    that removed nothing still gets a zero-height bar instead of vanishing.
    """
    raw_vocab_sizes = (
        projected.filter(pl.col("tier") == RAW_TIER)
        .group_by("system")
        .agg(pl.col("token").n_unique().alias("raw_vocab_size"))
    )

    pairs = pl.DataFrame(
        [
            {"system": system, "layer": layer}
            for system, layers in layer_names_by_system.items()
            for layer in layers
        ],
        schema={"system": pl.Utf8, "layer": pl.Utf8},
    )

    if membership.height > 0:
        removed = membership.group_by(["system", "layer"]).agg(
            pl.col("token").n_unique().alias("removed_count"),
            pl.col("document_frequency_pct").sum().alias("removed_mass"),
        )
    else:
        removed = pl.DataFrame(
            schema={
                "system": pl.Utf8,
                "layer": pl.Utf8,
                "removed_count": pl.UInt32,
                "removed_mass": pl.Float64,
            }
        )

    combined = (
        pairs.join(removed, on=["system", "layer"], how="left")
        .with_columns(
            pl.col("removed_count").fill_null(0),
            pl.col("removed_mass").fill_null(0.0),
        )
        .join(raw_vocab_sizes, on="system", how="left")
        .with_columns(
            (pl.col("raw_vocab_size") - pl.col("removed_count")).alias(
                "remaining_count"
            )
        )
    )
    return combined.select(
        "system",
        "layer",
        "raw_vocab_size",
        "removed_count",
        "remaining_count",
        "removed_mass",
    ).sort(["system", "layer"])


def summarize_layer_regions(membership: pl.DataFrame, *, layer: str) -> pl.DataFrame:
    """One row per (region, system) for a single layer: the region's exact
    token count (same across every system row for that region, since a
    region is defined as membership in exactly that set of corpora) and this
    corpus's document-frequency mass within it."""
    subset = membership.filter(pl.col("layer") == layer)
    if subset.height == 0:
        return pl.DataFrame(
            schema={
                "region": pl.Utf8,
                "system": pl.Utf8,
                "token_count": pl.UInt32,
                "mass": pl.Float64,
            }
        )
    token_counts = subset.group_by("region").agg(
        pl.col("token").n_unique().alias("token_count")
    )
    mass = subset.group_by(["region", "system"]).agg(
        pl.col("document_frequency_pct").sum().alias("mass")
    )
    return (
        mass.join(token_counts, on="region", how="left")
        .select("region", "system", "token_count", "mass")
        .sort(["region", "system"])
    )


def compute_pairwise_jaccard(membership: pl.DataFrame, *, layer: str) -> pl.DataFrame:
    """Per-pair Jaccard similarity of the removed token sets across every
    named corpus, for one layer."""
    subset = membership.filter(pl.col("layer") == layer)
    removed_sets = {
        # `group_by` yields a tuple key even for a single grouping column.
        system: frozenset(group.get_column("token").to_list())
        for (system,), group in subset.group_by("system")
    }
    systems = sorted(removed_sets)
    rows = []
    for system_a, system_b in combinations(systems, 2):
        set_a, set_b = removed_sets[system_a], removed_sets[system_b]
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union) if union else 0.0
        rows.append({"system_a": system_a, "system_b": system_b, "jaccard": jaccard})
    if not rows:
        return pl.DataFrame(
            schema={"system_a": pl.Utf8, "system_b": pl.Utf8, "jaccard": pl.Float64}
        )
    return pl.DataFrame(rows)


def _removed_sets_by_system(
    membership: pl.DataFrame, *, layer: str
) -> dict[str, frozenset[str]]:
    subset = membership.filter(pl.col("layer") == layer)
    return {
        # `group_by` yields a tuple key even for a single grouping column.
        system: frozenset(group.get_column("token").to_list())
        for (system,), group in subset.group_by("system")
    }


def draw_layer_bar_chart(
    ax: plt.Axes,
    metrics: pl.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
) -> None:
    """Grouped bar chart with one system per x-position and one bar per
    layer (systems, then layers, taken in `metrics`'s own row order so a
    caller controls presentation order by how it sorts the frame first)."""
    systems = list(dict.fromkeys(metrics.get_column("system").to_list()))
    layers = list(dict.fromkeys(metrics.get_column("layer").to_list()))
    if not systems or not layers:
        ax.set_title(title)
        return

    by_system_layer = {
        (row["system"], row["layer"]): row[value_col]
        for row in metrics.iter_rows(named=True)
    }

    x = list(range(len(systems)))
    bar_width = 0.8 / len(layers)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#2E5EAA"])

    for layer_index, layer in enumerate(layers):
        offsets = [pos + (layer_index - (len(layers) - 1) / 2) * bar_width for pos in x]
        heights = [by_system_layer.get((system, layer), 0) for system in systems]
        color = color_cycle[layer_index % len(color_cycle)]
        ax.bar(offsets, heights, width=bar_width, color=color, label=layer)

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def render_layer_bar_charts(metrics: pl.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Render both required bar charts (remaining unique tokens, removed
    mass) to standalone files under `output_dir`."""
    outputs: dict[str, Path] = {}
    specs = [
        (
            "remaining_count",
            "Unique tokens remaining",
            "Unique tokens remaining per corpus, by layer",
            "_remaining_tokens_by_layer.png",
        ),
        (
            "removed_mass",
            "Removed document-frequency mass",
            "Removed document-frequency mass per corpus, by layer",
            "_removed_mass_by_layer.png",
        ),
    ]
    for value_col, ylabel, title, filename in specs:
        output_path = output_dir / filename
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * metrics["system"].n_unique()), 5))
        draw_layer_bar_chart(
            ax, metrics, value_col=value_col, ylabel=ylabel, title=title
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs[value_col] = output_path
    return outputs


def draw_overlap_diagram(
    fig: plt.Figure,
    removed_sets: Mapping[str, frozenset[str]],
    corpus_totals: Mapping[str, tuple[int, int]],
    *,
    title: str,
) -> None:
    """Draw a removed-set overlap diagram onto `fig`: a Venn for two or three
    named corpora, an UpSet plot beyond. `corpus_totals` maps system ->
    (raw-tier unique token count, raw-tier name count), shown in the Venn's
    set labels or an UpSet caption -- reference context the diagram itself
    (which only ever shows the smaller *removed* sets) doesn't otherwise
    carry.

    Takes a `Figure`, not an `Axes`: an UpSet plot is a matrix-plus-bars grid
    that needs the whole figure's subplot space, so a single shared-axis
    signature (as `token_zipf.draw_zipf_curve` uses) can't cover both cases.
    A notebook wanting this inline supplies its own `fig, _ = plt.subplots()`.
    """
    systems = sorted(removed_sets)
    if len(systems) < 2:
        raise ValueError(
            f"An overlap diagram needs at least two named corpora, got {systems!r}."
        )

    def _label(system: str) -> str:
        unique_tokens, name_count = corpus_totals.get(system, (0, 0))
        return f"{system} ({unique_tokens:,} tokens, {name_count:,} names)"

    if len(systems) in (2, 3):
        ax = fig.add_subplot(111)
        sets = [set(removed_sets[system]) for system in systems]
        labels = tuple(_label(system) for system in systems)
        if len(systems) == 2:
            venn2(sets, set_labels=labels, ax=ax)
        else:
            venn3(sets, set_labels=labels, ax=ax)
    else:
        # upsetplot 0.9.0's `show_counts=True` text-label path crashes under
        # matplotlib 3.11 (`TypeError: only 0-dimensional arrays can be
        # converted to Python scalars`, found running this for real) -- a
        # known version incompatibility, not this module's bug. Draw counts
        # ourselves via `ax.bar_label` on the returned intersections axis
        # instead of the library's own broken label path. The FutureWarnings
        # upsetplot's own pandas `inplace=True` fillna calls emit are
        # unrelated noise from the same version mismatch, suppressed here.
        original_size = fig.get_size_inches()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module="upsetplot"
            )
            upset_data = from_contents(
                {system: set(removed_sets[system]) for system in systems}
            )
            axes = UpSet(upset_data, subset_size="count").plot(fig=fig)
        for container in axes["intersections"].containers:
            axes["intersections"].bar_label(container, fmt="%d")
        # `UpSet.plot()` unconditionally resizes `fig` to its own preferred
        # dimensions (found running this for real: an 8.8x6.5 figure came
        # back 4.4x4.4), discarding whatever size the caller set up before
        # this call -- restoring it afterward is safe since matplotlib's
        # gridspec-placed axes are stored as figure-fraction positions, not
        # absolute ones, so a post-hoc resize doesn't disturb the layout.
        fig.set_size_inches(*original_size)
        caption = "; ".join(_label(system) for system in systems)
        fig.text(0.01, 0.01, caption, fontsize=7, wrap=True)

    fig.suptitle(title)


def render_overlap_diagrams(
    membership: pl.DataFrame,
    corpus_totals: Mapping[str, tuple[int, int]],
    output_dir: Path,
) -> dict[str, Path]:
    """One overlap-diagram file per layer present in `membership`, named
    `_overlap_<layer>.png`. A layer with fewer than two named corpora
    carrying a removed token is skipped (nothing to overlap)."""
    outputs: dict[str, Path] = {}
    for layer in sorted(membership.get_column("layer").unique().to_list()):
        removed_sets = _removed_sets_by_system(membership, layer=layer)
        if len(removed_sets) < 2:
            continue
        output_path = output_dir / f"_overlap_{layer}.png"
        fig = plt.figure(figsize=(max(8, 2.2 * len(removed_sets)), 6.5))
        draw_overlap_diagram(
            fig, removed_sets, corpus_totals, title=f"Removed-set overlap -- {layer}"
        )
        fig.subplots_adjust(top=0.88, bottom=0.2)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs[layer] = output_path
    return outputs


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        print(f"[noise_layers] {message}")


def run_noise_layer_analysis(
    roots: WorkspaceRoots,
    *,
    systems: Sequence[str],
    run_date: str | None = None,
    zipf_run_date: str | None = None,
    cleanse_tier: str = "cleansed",
    extra_noise_word_paths: Sequence[tuple[str, Path]] = (),
    progress: Callable[[str], None] | None = None,
) -> NoiseLayerPaths:
    """Run the full cross-corpus token-projection and noise-layer pipeline
    for the named systems: project their raw/`cleanse_tier` token stats plus
    tokenizer IDF into one table, compute
    every layer's raw-tier removed set, and render both bar charts plus one
    overlap diagram per layer -- all under one run tree beside
    `analysis.token_zipf`'s own.

    `zipf_run_date` selects which existing `analysis.token_zipf` run to read
    per-tier stats from (defaults to `run_date`, the common case of running
    both the same day); it does not itself run `token_zipf` -- that's a
    separate, already-dispatched pipeline stage.
    """
    effective_run_date = run_date or datetime.now(UTC).date().isoformat()
    effective_zipf_run_date = zipf_run_date or effective_run_date
    paths = resolve_noise_layer_paths(roots, effective_run_date)

    tier_stats = load_zipf_tier_stats(
        roots,
        effective_zipf_run_date,
        systems,
        tiers=(RAW_TIER, cleanse_tier),
    )
    idf_stats = load_tokenizer_idf_stats(roots, systems)
    projected = project_cross_corpus_tokens(tier_stats, idf_stats)
    projected.write_parquet(paths.metrics_dir / "projected_tokens.parquet")
    _emit_progress(
        progress,
        f"projected {projected.height:,} token rows across {len(systems)} systems",
    )

    layers_by_system: dict[str, list[NoiseLayerSpec]] = {
        system: resolve_noise_layers(
            system, roots, extra_noise_word_paths=extra_noise_word_paths
        )
        for system in systems
    }
    layer_names_by_system = {
        system: [CLEANSE_LAYER, *(layer.name for layer in layers)]
        for system, layers in layers_by_system.items()
    }

    membership = compute_layer_removals(
        projected,
        systems=systems,
        cleanse_tier=cleanse_tier,
        layers_by_system=layers_by_system,
    )
    membership.write_parquet(paths.metrics_dir / "membership.parquet")
    _emit_progress(progress, f"membership: {membership.height:,} removed-token rows")

    metrics = compute_corpus_layer_metrics(
        projected, membership, layer_names_by_system=layer_names_by_system
    )
    metrics.write_parquet(paths.metrics_dir / "layer_metrics.parquet")

    region_frames = []
    jaccard_frames = []
    for layer in sorted(membership.get_column("layer").unique().to_list()):
        region_frames.append(
            summarize_layer_regions(membership, layer=layer).with_columns(
                pl.lit(layer).alias("layer")
            )
        )
        jaccard_frames.append(
            compute_pairwise_jaccard(membership, layer=layer).with_columns(
                pl.lit(layer).alias("layer")
            )
        )
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
    jaccard_table = (
        pl.concat(jaccard_frames, how="vertical")
        if jaccard_frames
        else pl.DataFrame(
            schema={
                "system_a": pl.Utf8,
                "system_b": pl.Utf8,
                "jaccard": pl.Float64,
                "layer": pl.Utf8,
            }
        )
    )
    region_table.write_parquet(paths.metrics_dir / "region_summary.parquet")
    jaccard_table.write_parquet(paths.metrics_dir / "region_jaccard.parquet")

    bar_paths = render_layer_bar_charts(metrics, paths.viz_dir)
    for value_col, bar_path in bar_paths.items():
        _emit_progress(progress, f"bar chart {value_col} -> {bar_path}")

    corpus_totals = {
        entry.system: (entry.stats.height, entry.total_docs)
        for entry in tier_stats
        if entry.tier == RAW_TIER
    }
    overlap_paths = render_overlap_diagrams(membership, corpus_totals, paths.viz_dir)
    for layer, overlap_path in overlap_paths.items():
        _emit_progress(progress, f"overlap diagram layer={layer} -> {overlap_path}")

    return paths
