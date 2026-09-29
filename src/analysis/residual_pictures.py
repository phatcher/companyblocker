"""Four pictures over the true blocking residual.

`blocking.inspection`'s `residual_pairs`/`pair_truth_eval` helpers report
scalar recall/precision numbers and a bar chart; they say *that* the blocker
missed pairs, not *where* they sit or *why*. This module draws four pictures
over the same `never`-level truth-pair population `residual_pairs` already
carries (pairs still unequal after cleansing, `blocking.inspection
.filter_residual_pairs`'s default): a missed pair has no
similarity or rank at all, so every axis here is computed from the two
cleansed names and each source row's cluster size, never from a score the
matcher only produces for a pair it proposed.

One function per picture, each taking a `matplotlib.axes.Axes` in the shape
`analysis.token_zipf.draw_zipf_curve` uses -- public and axis-level so a
notebook can supply its own axis for one picture inline, while
`render_residual_pictures` wraps all four into one 2x2 figure for the script.
Every picture accepts a `series`: a list of `(label, frame)` pairs, so more
than one run (representation, or country when only one representation
exists) overlays on the same axis -- except the hex picture, which draws one
run only, since overlaid hexbins do not compose.

`build_residual_picture_frame` is the one place that reads a run's residual
per-pair artefact (either `pair_truth_eval_detail.parquet` or its pre-rename
name `residual_pairs.parquet` -- this accepts either, preferring the new
name) and `clusters.parquet` and returns the shared per-pair frame both the
notebook and `scripts/analyze_residual_pairs.py` draw from, so the two never
compute distance statistics two different ways.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jellyfish
import matplotlib
import numpy as np
import polars as pl

from validation.runner import NAME_EQUALITY_NEVER

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# polars accepts a dtype class (e.g. pl.Utf8) or an instance (e.g. pl.Utf8())
# interchangeably in schema dicts; pl.DataType alone only covers the latter.
# Restated here rather than imported: this is a typing idiom, not a contract
# anyone owns -- `validation` itself declares it four times over
# (`contracts.py`, `input_contract.py`, `runner.py`, and as `_PolarsDType` in
# `perturbation_materializer.py`), so there is no single definition to drift
# from and no reason for an area boundary to be crossed for it.
PolarsDType = type[pl.DataType] | pl.DataType

# A blocking run's per-pair truth artefact was renamed from
# `residual_pairs.parquet` to `pair_truth_eval_detail.parquet`. Runs written
# before that rename still carry the old name on disk, so the new name is
# preferred and the old one is the fallback -- never the reverse, so a run
# written after the rename is read under its real name.
RESIDUAL_PAIRS_ARTIFACT_NAMES: tuple[str, ...] = (
    "pair_truth_eval_detail.parquet",
    "residual_pairs.parquet",
)
CLUSTERS_ARTIFACT_NAME = "clusters.parquet"

# Palette follows `token_zipf.py`'s convention: distinct hues per series
# (run/representation), reserving line style for the found/missed split
# within a series so the same colour always means the same run across all
# four pictures.
_SERIES_COLORS: tuple[str, ...] = (
    "#2E5EAA",
    "#3F9142",
    "#C0392B",
    "#8E44AD",
    "#D68910",
)
_OUTCOME_LINESTYLES: dict[str, str] = {"found": "-", "missed": "--"}

RESIDUAL_PICTURE_FRAME_SCHEMA: dict[str, PolarsDType] = {
    "source_id": pl.Utf8,
    "target_id": pl.Utf8,
    "found": pl.Boolean,
    "similarity": pl.Float64,
    "rank": pl.Int64,
    "token_jaccard": pl.Float64,
    "levenshtein_distance": pl.Int64,
    "length_diff": pl.Int64,
    "cluster_size": pl.Int64,
}


def resolve_residual_pairs_path(run_dir: Path) -> Path:
    """The per-pair truth artefact for a blocking run's output directory,
    preferring the renamed `pair_truth_eval_detail.parquet` over the
    pre-rename `residual_pairs.parquet` when both exist -- see this module's
    docstring. Raises `FileNotFoundError` when neither is present, naming
    both candidates so the message stays useful either side of the rename.
    """
    for name in RESIDUAL_PAIRS_ARTIFACT_NAMES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {' or '.join(RESIDUAL_PAIRS_ARTIFACT_NAMES)} under {run_dir}"
    )


def _tokenize(name: str | None) -> tuple[str, ...]:
    """Plain whitespace tokenization, matching
    `scripts/measure_initialism_recall.py`'s `_tokenize` convention (not
    imported from there -- `src/analysis` does not depend on `scripts/`)."""
    if not name:
        return ()
    return tuple(token for token in name.strip().lower().split() if token)


def _token_jaccard(source_name: str | None, target_name: str | None) -> float | None:
    source_tokens = set(_tokenize(source_name))
    target_tokens = set(_tokenize(target_name))
    union = source_tokens | target_tokens
    if not union:
        return None
    return len(source_tokens & target_tokens) / len(union)


def _source_cluster_sizes(clusters: pl.DataFrame) -> pl.DataFrame:
    """One row per source node id with the total node count (source plus
    target) of the cluster it belongs to. A source id absent from `clusters`
    (never grouped with anything) is simply absent here -- callers fill that
    in as a singleton (`cluster_size=1`), since that is what "clustered with
    nothing else" means, not a missing value.
    """
    if clusters.height == 0:
        return pl.DataFrame(schema={"node_id": pl.Utf8, "cluster_size": pl.Int64})
    sizes = clusters.group_by("cluster_id").len(name="cluster_size")
    source_nodes = clusters.filter(pl.col("node_role") == "source").select(
        "cluster_id", "node_id"
    )
    return source_nodes.join(sizes, on="cluster_id", how="left").select(
        "node_id", "cluster_size"
    )


def build_residual_picture_frame(
    run_dir: Path, *, country: str | None = None
) -> pl.DataFrame:
    """The shared per-pair frame every picture in this module draws from:
    `never`-level truth pairs (still unequal after cleansing)
    from `run_dir`'s residual-pairs artefact (either name -- see
    `resolve_residual_pairs_path`), joined to `run_dir/clusters.parquet` for
    each source row's cluster size.

    Both difficulty measures are computed here rather than in each picture,
    so the axis stays pluggable across pictures without recomputing it:
    `token_jaccard` (whitespace token-set overlap between the
    two cleansed names) and `levenshtein_distance` (character-edit distance
    between them) -- a missed pair still carries a target, since a truth
    pair's target is known whether or not the blocker proposed it, so both
    measures are always populated for every row this returns.
    `length_diff` is `abs(len(source_name_cleansed) - len(target_name_cleansed))`,
    the hex picture's other axis.

    `country` optionally restricts to one country's rows (the notebook's
    convention); `None` (default) returns every country in the run.

    Returns `RESIDUAL_PICTURE_FRAME_SCHEMA`'s columns, 0 rows when the run
    has no `never`-level truth pairs (or no residual-pairs artefact
    at all raises `FileNotFoundError` instead -- see
    `resolve_residual_pairs_path`).
    """
    residual_pairs_path = resolve_residual_pairs_path(run_dir)
    residual_pairs = pl.read_parquet(residual_pairs_path)
    if country is not None:
        residual_pairs = residual_pairs.filter(pl.col("country") == country)

    truth = residual_pairs.filter(
        (pl.col("name_equality") == NAME_EQUALITY_NEVER) & pl.col("is_truth_pair")
    )
    if truth.height == 0:
        return pl.DataFrame(schema=RESIDUAL_PICTURE_FRAME_SCHEMA)

    clusters_path = run_dir / CLUSTERS_ARTIFACT_NAME
    clusters = (
        pl.read_parquet(clusters_path)
        if clusters_path.exists()
        else pl.DataFrame(
            schema={"cluster_id": pl.Utf8, "node_id": pl.Utf8, "node_role": pl.Utf8}
        )
    )
    cluster_sizes = _source_cluster_sizes(clusters)

    source_names = truth.get_column("source_name_cleansed").to_list()
    target_names = truth.get_column("target_name_cleansed").to_list()
    levenshtein = [
        jellyfish.levenshtein_distance(source, target)
        if source is not None and target is not None
        else None
        for source, target in zip(source_names, target_names)
    ]
    jaccard = [
        _token_jaccard(source, target)
        for source, target in zip(source_names, target_names)
    ]

    return (
        truth.with_columns(
            pl.Series("token_jaccard", jaccard, dtype=pl.Float64),
            pl.Series("levenshtein_distance", levenshtein, dtype=pl.Int64),
            (
                pl.col("source_name_cleansed").str.len_chars().cast(pl.Int64)
                - pl.col("target_name_cleansed").str.len_chars().cast(pl.Int64)
            )
            .abs()
            .alias("length_diff"),
        )
        .join(cluster_sizes, left_on="source_id", right_on="node_id", how="left")
        .with_columns(pl.col("cluster_size").fill_null(1))
        .select(list(RESIDUAL_PICTURE_FRAME_SCHEMA.keys()))
        .cast(pl.Schema(RESIDUAL_PICTURE_FRAME_SCHEMA))
    )


def draw_axis_histogram(
    ax: plt.Axes,
    series: Sequence[tuple[str, pl.DataFrame]],
    *,
    title: str,
    axis_column: str = "token_jaccard",
    bins: int = 20,
) -> dict[str, dict[str, int]]:
    """Histogram of `axis_column` (`token_jaccard` by default, or
    `levenshtein_distance` -- the difficulty axis is pluggable) over the
    residual truth-pair population, found against
    missed as two step outlines per series/run so a picture with more than
    one run shows a wider window's effect on where the misses sit without
    the bars themselves overlapping.

    Returns each series label's found/missed row counts. An empty `series`
    or a series whose frame has no rows for one outcome draws nothing for
    that outcome rather than raising.
    """
    result: dict[str, dict[str, int]] = {}
    if not series:
        ax.set_title(title)
        return result

    all_values = [
        value
        for _, frame in series
        for value in frame.get_column(axis_column).to_list()
        if value is not None
    ]
    axis_min = min(all_values) if all_values else 0.0
    axis_max = max(all_values) if all_values else 1.0
    if axis_min == axis_max:
        axis_max = axis_min + 1.0
    bin_edges = [
        axis_min + (axis_max - axis_min) * step / bins for step in range(bins + 1)
    ]

    for series_index, (label, frame) in enumerate(series):
        color = _SERIES_COLORS[series_index % len(_SERIES_COLORS)]
        counts: dict[str, int] = {}
        for outcome, is_found in (("found", True), ("missed", False)):
            outcome_values = [
                value
                for value in frame.filter(pl.col("found") == is_found)
                .get_column(axis_column)
                .to_list()
                if value is not None
            ]
            counts[outcome] = len(outcome_values)
            if not outcome_values:
                continue
            ax.hist(
                outcome_values,
                bins=bin_edges,
                histtype="step",
                linewidth=1.5,
                linestyle=_OUTCOME_LINESTYLES[outcome],
                color=color,
                label=f"{label} {outcome} (n={len(outcome_values):,})",
            )
        result[label] = counts

    ax.set_xlabel(axis_column.replace("_", " "))
    ax.set_ylabel("residual truth pairs")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return result


def draw_recall_by_rank(
    ax: plt.Axes,
    series: Sequence[tuple[str, pl.DataFrame]],
    *,
    title: str,
    max_rank: int | None = None,
) -> dict[str, float]:
    """Cumulative share of residual truth pairs found at rank <= k, as k
    sweeps upward from 1, one line per series/run. The found share of the
    population is the ceiling every curve approaches (drawn as a dotted
    horizontal reference line) since a missed pair has no rank to climb
    toward at all -- this picture answers whether a wider candidate window
    (a larger k) would recover more of the residual, not whether recall as
    scored today is high.

    `max_rank` bounds every series' x-axis; defaults to the highest rank any
    series' found pairs actually reached. Returns each series label's found
    share (the ceiling value).
    """
    result: dict[str, float] = {}
    for series_index, (label, frame) in enumerate(series):
        color = _SERIES_COLORS[series_index % len(_SERIES_COLORS)]
        total = frame.height
        if total == 0:
            continue
        ranks = sorted(
            rank
            for rank in frame.filter(pl.col("found")).get_column("rank").to_list()
            if rank is not None
        )
        found_share = len(ranks) / total
        limit = max_rank or (ranks[-1] if ranks else 1)
        xs = list(range(1, limit + 1))
        cumulative: list[float] = []
        idx = 0
        n = len(ranks)
        for k in xs:
            while idx < n and ranks[idx] <= k:
                idx += 1
            cumulative.append(idx / total)
        ax.plot(
            xs,
            cumulative,
            color=color,
            linewidth=1.5,
            label=f"{label} (ceiling={found_share:.1%})",
        )
        ax.axhline(found_share, color=color, linestyle=":", linewidth=1.0)
        result[label] = found_share

    ax.set_xlabel("rank k")
    ax.set_ylabel("cumulative share of residual truth pairs found by rank k")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return result


def draw_cluster_size_cumulative(
    ax: plt.Axes,
    series: Sequence[tuple[str, pl.DataFrame]],
    *,
    title: str,
) -> dict[str, dict[str, int]]:
    """Cumulative share of found and missed residual truth pairs by source
    cluster size (a singleton source counts as size 1), on a log x-axis, one
    found line and one missed line per series/run. Puts whether misses
    concentrate among sources that never clustered with anything, or among
    sources sitting in an already-populous cluster, on one picture.

    Returns each series label's found/missed row counts.
    """
    result: dict[str, dict[str, int]] = {}
    for series_index, (label, frame) in enumerate(series):
        color = _SERIES_COLORS[series_index % len(_SERIES_COLORS)]
        counts_for_label: dict[str, int] = {}
        for outcome, is_found in (("found", True), ("missed", False)):
            outcome_frame = frame.filter(pl.col("found") == is_found)
            total = outcome_frame.height
            counts_for_label[outcome] = total
            if total == 0:
                continue
            sizes = sorted(outcome_frame.get_column("cluster_size").to_list())
            xs = sorted(set(sizes))
            cumulative: list[float] = []
            idx = 0
            n = len(sizes)
            for x in xs:
                while idx < n and sizes[idx] <= x:
                    idx += 1
                cumulative.append(idx / total)
            ax.plot(
                xs,
                cumulative,
                color=color,
                linestyle=_OUTCOME_LINESTYLES[outcome],
                linewidth=1.5,
                label=f"{label} {outcome} (n={total:,})",
            )
        result[label] = counts_for_label

    ax.set_xscale("log")
    ax.set_xlabel("source cluster size (log)")
    ax.set_ylabel("cumulative share")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return result


def draw_missed_fraction_hex(
    ax: plt.Axes,
    frame: pl.DataFrame,
    *,
    title: str,
    axis_column: str = "token_jaccard",
    y_column: str = "length_diff",
    gridsize: int = 15,
    size_scale: float = 40.0,
) -> int:
    """One hex layer over `axis_column` against `y_column`, coloured by
    missed fraction with cell size following pair count -- deliberately one
    layer, not two overlaid, since overlaid hexbins do not compose. Draws a
    single run's frame, unlike the other three
    pictures' `series` -- a caller comparing runs picks one to draw here, or
    calls this once per run into separate axes.

    Bins via `Axes.hexbin` twice on the same grid (once for per-cell count,
    once with `C=` the missed indicator and `reduce_C_function=mean` for
    per-cell missed fraction) rather than a hand-rolled hex grid, then
    re-draws the same cell centres as a scatter sized by count so a sparse
    cell reads as a small dot next to a dense cell's large one -- plain
    `hexbin` cells are all the same size regardless of count.

    Returns the number of non-empty hex cells drawn (0 for an empty
    `frame`, drawing only the title).
    """
    if frame.height == 0:
        ax.set_title(title)
        return 0

    x = frame.get_column(axis_column).to_numpy()
    y = frame.get_column(y_column).to_numpy()
    missed = (~frame.get_column("found").to_numpy()).astype(float)

    count_hex = ax.hexbin(x, y, gridsize=gridsize, mincnt=1)
    offsets = np.asarray(count_hex.get_offsets())
    counts = np.asarray(count_hex.get_array())
    frac_hex = ax.hexbin(
        x, y, C=missed, gridsize=gridsize, mincnt=1, reduce_C_function=np.mean
    )
    fractions = np.asarray(frac_hex.get_array())
    count_hex.remove()
    frac_hex.remove()

    sizes = size_scale * np.sqrt(counts)
    scatter = ax.scatter(
        offsets[:, 0],
        offsets[:, 1],
        s=sizes,
        c=fractions,
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        edgecolors="#33322c",
        linewidths=0.3,
    )
    figure = ax.get_figure()
    if figure is not None:
        figure.colorbar(scatter, ax=ax, label="missed fraction")
    ax.set_xlabel(axis_column.replace("_", " "))
    ax.set_ylabel(y_column.replace("_", " "))
    ax.set_title(title)
    return len(offsets)


def render_residual_pictures(
    series: Sequence[tuple[str, pl.DataFrame]],
    output_path: Path,
    *,
    suptitle: str,
    hex_label: str | None = None,
) -> Path:
    """Render all four pictures into one 2x2 figure: `token_jaccard`
    histogram, cumulative recall by rank, cumulative share by source cluster
    size, and the missed-fraction hex (drawn for `hex_label`'s series, or
    the first series when `hex_label` is `None` -- see
    `draw_missed_fraction_hex`'s single-run note).

    `series` empty writes a figure with four empty-titled panels rather than
    raising -- callers checking "is there anything to draw" should check
    `series` themselves first.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    draw_axis_histogram(axes[0][0], series, title="Token-Jaccard: found vs missed")
    draw_recall_by_rank(axes[0][1], series, title="Cumulative share found by rank")
    draw_cluster_size_cumulative(
        axes[1][0], series, title="Cumulative share by source cluster size"
    )
    hex_frame = pl.DataFrame(schema=RESIDUAL_PICTURE_FRAME_SCHEMA)
    hex_title = "Missed fraction by Jaccard x length difference"
    if series:
        if hex_label is not None:
            matching = [frame for label, frame in series if label == hex_label]
            hex_frame = matching[0] if matching else series[0][1]
        else:
            hex_label, hex_frame = series[0]
        hex_title = f"{hex_title} ({hex_label})"
    draw_missed_fraction_hex(axes[1][1], hex_frame, title=hex_title)

    fig.suptitle(suptitle)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
