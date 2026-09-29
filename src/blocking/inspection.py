"""Plot-ready frames from a blocking run, for notebooks.

Every function here is a pure reshape of a run's frames, live or reloaded from disk,
with no file access and no rendering: the similarity histogram, the cluster-size
distribution, the cluster metrics in one long frame, and `filter_residual_pairs()`,
which narrows a per-pair evaluation to the `never` level by default.
"""

from __future__ import annotations

import polars as pl

from validation.contracts import PolarsDType
from validation.runner import (
    NAME_EQUALITY_LEVELS,
    NAME_EQUALITY_NEVER,
    NAME_EQUALITY_UNKNOWN,
)

from .contracts import BlockingRunResult

NAME_EQUALITY_COLUMN = "name_equality"

# Cascade order, with `unknown` last: a pair whose equality could not be
# judged is its own row rather than folded in with the trivial levels, since
# folding an unknown into the trivial half is how a miss gets reported as a
# success.
_NAME_EQUALITY_ORDER: tuple[str, ...] = (*NAME_EQUALITY_LEVELS, NAME_EQUALITY_UNKNOWN)
_RECALL_SCHEMA: dict[str, PolarsDType] = {
    "found": pl.Int64,
    "missed": pl.Int64,
    "total": pl.Int64,
    "recall": pl.Float64,
}
_PRECISION_SCHEMA: dict[str, PolarsDType] = {
    "true_pair": pl.Int64,
    "false_pair": pl.Int64,
    "total": pl.Int64,
    "precision": pl.Float64,
}

# `population` is `pair_truth_eval`'s own key, so its figures stay
# distinguishable once melted; the frames without one carry it null.
_CLUSTER_METRICS_ID_COLUMNS: tuple[str, ...] = (
    "country",
    "source_system",
    "target_system",
    "population",
)
_CLUSTER_METRICS_SCHEMA: dict[str, PolarsDType] = {
    "source": pl.Utf8,
    "country": pl.Utf8,
    "source_system": pl.Utf8,
    "target_system": pl.Utf8,
    "population": pl.Utf8,
    "metric": pl.Utf8,
    "value": pl.Float64,
}
_SIMILARITY_DISTRIBUTION_SCHEMA: dict[str, PolarsDType] = {
    "bucket_start": pl.Float64,
    "bucket_end": pl.Float64,
    "edge_count": pl.Int64,
    "bucket_midpoint": pl.Float64,
    "bucket_label": pl.Utf8,
    "edge_share": pl.Float64,
    "cumulative_edge_share": pl.Float64,
}
_CLUSTER_SIZE_DISTRIBUTION_SCHEMA: dict[str, PolarsDType] = {
    "cluster_size": pl.Int64,
    "cluster_count": pl.Int64,
    "node_count": pl.Int64,
    "cluster_share": pl.Float64,
    "node_share": pl.Float64,
}


def summarize_similarity_distribution(
    similarity_distribution: pl.DataFrame,
) -> pl.DataFrame:
    """Reshape a `similarity_distribution` frame (`bucket_start`/`bucket_end`/
    `edge_count`, `workflow.py`'s `_compute_similarity_distribution`) into a
    notebook-plot-ready frame: adds a bucket midpoint (for an x-axis), a
    human-readable label (mirroring `notebooks/analyse_tfidf.ipynb`'s
    `bucket_label` convention), and normalised/cumulative edge shares (mirroring
    that notebook's `cumulative_unique_share` pattern).

    Pure reshape -- does not change bucket count or edge counts. Safe on the
    zero-filled distribution an empty run emits (all buckets present,
    `edge_count` 0): shares are all 0 rather than dividing by zero. Safe on a genuinely empty
    (0-row) frame too, returning an empty frame with the same added columns.
    """

    if similarity_distribution.height == 0:
        return pl.DataFrame(schema=_SIMILARITY_DISTRIBUTION_SCHEMA)

    total_edges = similarity_distribution.get_column("edge_count").sum()
    denominator = total_edges if total_edges > 0 else 1

    return similarity_distribution.with_columns(
        ((pl.col("bucket_start") + pl.col("bucket_end")) / 2).alias("bucket_midpoint"),
        pl.format(
            "[{}, {})",
            pl.col("bucket_start").round(4).cast(pl.Utf8),
            pl.col("bucket_end").round(4).cast(pl.Utf8),
        ).alias("bucket_label"),
        (pl.col("edge_count") / denominator).alias("edge_share"),
    ).with_columns(pl.col("edge_share").cum_sum().alias("cumulative_edge_share"))


def compute_cluster_size_distribution(clusters: pl.DataFrame) -> pl.DataFrame:
    """Distribution of cluster (block) sizes derived from `clusters`
    (`cluster_id, node_id, node_name, node_role`): for each
    distinct node-count-per-cluster value, how many clusters have that size and
    what share of clusters/nodes they represent.

    This is the block-size *distribution* companion to
    `company_vectorize.clustering_metrics.compute_cluster_shape_metrics`'s
    single-row aggregate stats (mean/p95/max/singleton_ratio) that
    `BlockingRunResult.cluster_shape` already carries -- that function summarizes
    the same per-cluster sizes this one bucket-counts, so a notebook can pair
    this histogram with those aggregate stats as reference lines.

    Always returns the same five columns
    (`cluster_size, cluster_count, node_count, cluster_share, node_share`),
    empty (0 rows) when `clusters` is empty. Unlike `similarity_distribution`,
    there is no fixed bucket domain to zero-fill against here -- cluster sizes
    are unbounded, so an empty distribution is genuinely empty rather than
    zero-filled across a known range.
    """

    if clusters.height == 0:
        return pl.DataFrame(schema=_CLUSTER_SIZE_DISTRIBUTION_SCHEMA)

    sizes = clusters.group_by("cluster_id").len(name="cluster_size")
    total_clusters = sizes.height
    total_nodes = int(sizes.get_column("cluster_size").sum())

    return (
        sizes.group_by("cluster_size")
        .len(name="cluster_count")
        .with_columns(
            (pl.col("cluster_size") * pl.col("cluster_count")).alias("node_count")
        )
        .with_columns(
            (pl.col("cluster_count") / total_clusters).alias("cluster_share"),
            (pl.col("node_count") / total_nodes).alias("node_share"),
        )
        .sort("cluster_size")
        .select(list(_CLUSTER_SIZE_DISTRIBUTION_SCHEMA.keys()))
        .cast(pl.Schema(_CLUSTER_SIZE_DISTRIBUTION_SCHEMA))
    )


def _melt_metrics(frame: pl.DataFrame | None, *, source: str) -> pl.DataFrame:
    if frame is None or frame.height == 0:
        return pl.DataFrame(schema=_CLUSTER_METRICS_SCHEMA)

    present_id_columns = [c for c in _CLUSTER_METRICS_ID_COLUMNS if c in frame.columns]
    value_columns = [c for c in frame.columns if c not in present_id_columns]

    melted = frame.unpivot(
        index=present_id_columns,
        on=value_columns,
        variable_name="metric",
        value_name="value",
    ).with_columns(pl.col("value").cast(pl.Float64), pl.lit(source).alias("source"))

    for column in _CLUSTER_METRICS_ID_COLUMNS:
        if column not in melted.columns:
            melted = melted.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))

    return melted.select(list(_CLUSTER_METRICS_SCHEMA.keys()))


def summarize_cluster_metrics(
    *,
    cluster_shape: pl.DataFrame,
    directional_coverage: pl.DataFrame,
    pair_truth_eval: pl.DataFrame | None = None,
    raw_pair_truth_eval: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Flatten a run's cluster/coverage/truth-eval summaries into one long
    `source, country, source_system, target_system, metric, value` frame --
    the run-level counterpart to `compute_cluster_size_distribution`'s
    per-cluster-size histogram, covering aggregate cluster shape,
    directional coverage, and (when ground truth is available) pruned/raw
    pair-truth-eval precision/recall/F1/confusion counts.

    `source` distinguishes which input frame each row came from
    (`cluster_shape`, `directional_coverage`, `pair_truth_eval`,
    `raw_pair_truth_eval`) so a notebook can filter/facet without parsing
    `metric` strings. `pair_truth_eval`/`raw_pair_truth_eval` rows are one per
    population per scored country (plus the `__all__` cross-country rollup,
    `_build_pair_truth_eval_output`'s convention) via the `population` and
    `country` columns, which are null on the other two sources;
    `cluster_shape`/`directional_coverage` carry a single `__all__` row
    (`workflow.py`'s tagging) since blocking clusters once
    across all countries rather than per-country.

    `pair_truth_eval`/`raw_pair_truth_eval` are optional and independently
    omittable (both `None` when the source dataset lacks ground truth,
    matching `BlockingRunResult.pair_truth_eval`) -- their rows are simply
    absent from the result rather than raising.
    """

    frames = [
        _melt_metrics(cluster_shape, source="cluster_shape"),
        _melt_metrics(directional_coverage, source="directional_coverage"),
        _melt_metrics(pair_truth_eval, source="pair_truth_eval"),
        _melt_metrics(raw_pair_truth_eval, source="raw_pair_truth_eval"),
    ]
    return pl.concat(frames, how="vertical")


def filter_residual_pairs(
    pair_truth_eval_detail: pl.DataFrame,
    *,
    bucket: str | None = NAME_EQUALITY_NEVER,
    is_truth_pair: bool | None = None,
    found: bool | None = None,
) -> pl.DataFrame:
    """Filter a `pair_truth_eval_detail` frame
    (`reporting.write_blocking_report()`'s per-pair artefact -- every truth
    pair plus every predicted-but-untrue pair, across every name-equality
    level) down to the population worth inspecting, so a notebook filters
    through this one call rather than reshaping the frame in a cell. This is
    the function name to reach for the actual residual: the frame itself
    carries every level, not just the still-unequal one.

    Defaults to `NAME_EQUALITY_NEVER` -- the pairs still unequal after
    cleansing that the matching algorithm actually has to earn, the
    non-trivial residual -- since that is what a reader almost always wants
    first. Pass `bucket=None` to see every level.

    `is_truth_pair`/`found` narrow further, independently of `bucket`:
    `is_truth_pair=False` isolates candidate-set pollution (predicted pairs
    ground truth says are wrong); `is_truth_pair=True, found=False` isolates
    misses (false negatives) -- the population this artefact exists to make
    inspectable, since a miss leaves no trace in `matched_edges` at all.
    Both default to `None` (no filter on that column).
    """
    filtered = pair_truth_eval_detail
    if bucket is not None:
        filtered = filtered.filter(pl.col("name_equality") == bucket)
    if is_truth_pair is not None:
        filtered = filtered.filter(pl.col("is_truth_pair") == is_truth_pair)
    if found is not None:
        filtered = filtered.filter(pl.col("found") == found)
    return filtered


def _finish_outcome_matrix(
    matrix: pl.DataFrame,
    *,
    by: str,
    positive: str,
    negative: str,
    rate: str,
    order: tuple[str, ...],
) -> pl.DataFrame:
    total = pl.col(positive) + pl.col(negative)
    out = matrix.with_columns(
        total.alias("total"),
        pl.when(total > 0).then(pl.col(positive) / total).otherwise(None).alias(rate),
    )
    present = set(out.get_column(by).to_list())
    categories = [value for value in order if value in present]
    categories += sorted(present - set(categories))
    return (
        out.with_columns(pl.col(by).cast(pl.Enum(categories)))
        .sort(by)
        .with_columns(pl.col(by).cast(pl.Utf8))
        .select(by, positive, negative, "total", rate)
    )


def summarize_pair_outcomes(
    pair_truth_eval_detail: pl.DataFrame, *, by: str = NAME_EQUALITY_COLUMN
) -> dict[str, pl.DataFrame]:
    """Cross-tabulate a `pair_truth_eval_detail` frame into the two outcome
    matrices a reader actually wants: `"recall"` (truth pairs, found vs
    missed) and `"precision"` (predicted pairs, true vs false), broken down by
    `name_equality` -- `raw`, `basic`, `cleansed`, `never`.

    There is no coarse easy/difficult axis any more. It existed because only
    the still-unequal level carried figures of its own, so a reader needed the
    rest folded into one comparison row; every level now reports its own
    confusion matrix in `pair_truth_eval`, and folding them here would only
    hide which level a pair sat at.

    Two populations, deliberately not one table. A truth pair carries `found`;
    a predicted-but-untrue pair has `found=None`, because "found" has no
    meaning for a pair that was never true. So `"recall"` counts truth pairs
    only and `"precision"` counts the predicted set -- every
    `is_truth_pair=False` row plus the truth pairs that were found. A single
    2x2 confusion matrix is not available here at all: true negatives are
    every unproposed pair in the cross product, which this artefact does not
    (and could not) enumerate.

    Rates are fractions, matching `compute_pair_truth_eval()`'s convention,
    and are null for a row with no pairs rather than zero.
    """

    if by != NAME_EQUALITY_COLUMN:
        raise ValueError(f"by must be {NAME_EQUALITY_COLUMN!r}, got {by!r}")

    order = _NAME_EQUALITY_ORDER
    if pair_truth_eval_detail.height == 0:
        return {
            "recall": pl.DataFrame(schema={by: pl.Utf8, **_RECALL_SCHEMA}),
            "precision": pl.DataFrame(schema={by: pl.Utf8, **_PRECISION_SCHEMA}),
        }

    frame = pair_truth_eval_detail

    truth_pairs = frame.filter(pl.col("is_truth_pair").eq(True))
    recall = truth_pairs.group_by(by).agg(
        pl.col("found").eq(True).sum().cast(pl.Int64).alias("found"),
        pl.col("found").eq(False).sum().cast(pl.Int64).alias("missed"),
    )

    predicted_pairs = frame.filter(
        pl.col("is_truth_pair").eq(False) | pl.col("found").eq(True)
    )
    precision = predicted_pairs.group_by(by).agg(
        pl.col("is_truth_pair").eq(True).sum().cast(pl.Int64).alias("true_pair"),
        pl.col("is_truth_pair").eq(False).sum().cast(pl.Int64).alias("false_pair"),
    )

    return {
        "recall": _finish_outcome_matrix(
            recall,
            by=by,
            positive="found",
            negative="missed",
            rate="recall",
            order=order,
        ),
        "precision": _finish_outcome_matrix(
            precision,
            by=by,
            positive="true_pair",
            negative="false_pair",
            rate="precision",
            order=order,
        ),
    }


def build_inspection_summary(result: BlockingRunResult) -> dict[str, pl.DataFrame]:
    """One-call convenience wrapper: builds all three notebook-inspection
    structures (`similarity_distribution`, `cluster_size_distribution`,
    `cluster_metrics`) from a single `BlockingRunResult`, so a notebook user
    does not need to know which constituent frames each helper needs.

    Equivalent to calling `summarize_similarity_distribution`,
    `compute_cluster_size_distribution`, and `summarize_cluster_metrics`
    individually with `result`'s matching fields -- use those directly when
    working from artefacts loaded back from disk (`reporting.py`'s
    `write_blocking_report()` output) rather than a live `BlockingRunResult`.
    """

    return {
        "similarity_distribution": summarize_similarity_distribution(
            result.similarity_distribution
        ),
        "cluster_size_distribution": compute_cluster_size_distribution(result.clusters),
        "cluster_metrics": summarize_cluster_metrics(
            cluster_shape=result.cluster_shape,
            directional_coverage=result.directional_coverage,
            pair_truth_eval=result.pair_truth_eval,
            raw_pair_truth_eval=result.raw_pair_truth_eval,
        ),
    }
