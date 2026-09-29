"""One picture over a blocking run's target-neighbour union: how the
cluster-size distribution changes once target-to-target edges are unioned
into a run's clustering.

`blocking.inspection.compute_cluster_size_distribution` already produces the
per-cluster-size histogram (`cluster_size`, `cluster_count`, `node_count`,
`cluster_share`, `node_share`) this module draws from -- called once for each
side of the union (a run's `clusters_before_union` and `clusters`) -- rather
than a second computation of the same thing. Follows
`analysis.residual_pictures`'s shape: one axis-level draw function, public
and pluggable so a notebook can supply its own axis, plus a render wrapper
for a script.

Cluster count and node volume share one axis (log-log: cluster size spans
orders of magnitude and so do the counts at each size), each series drawn as
two lines distinguished by linestyle -- the same
line-per-outcome-within-a-series idiom
`residual_pictures.draw_cluster_size_cumulative` already uses for
found/missed, applied here to `cluster_count`/`node_count` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# See `analysis.residual_pictures`'s own module docstring for why this is
# restated here rather than imported: a typing idiom several modules in this
# area already carry their own copy of, not a contract any one of them owns.
PolarsDType = type[pl.DataType] | pl.DataType

CLUSTER_SIZE_DISTRIBUTION_SCHEMA: dict[str, PolarsDType] = {
    "cluster_size": pl.Int64,
    "cluster_count": pl.Int64,
    "node_count": pl.Int64,
    "cluster_share": pl.Float64,
    "node_share": pl.Float64,
}

# Palette follows `residual_pictures.py`'s own convention: one colour per
# series (a run, or a before/after label), linestyle carrying the second
# axis within a series.
_SERIES_COLORS: tuple[str, ...] = (
    "#2E5EAA",
    "#3F9142",
    "#C0392B",
    "#8E44AD",
    "#D68910",
)
_VALUE_LINESTYLES: dict[str, str] = {"cluster_count": "-", "node_count": "--"}


def draw_cluster_size_line(
    ax: plt.Axes,
    series: Sequence[tuple[str, pl.DataFrame]],
    *,
    title: str,
) -> dict[str, dict[str, int]]:
    """Cluster count and node volume against cluster size, two lines per
    series/run, on a log-log axis.

    `series` is a list of `(label, frame)` pairs, `frame` shaped like
    `blocking.inspection.compute_cluster_size_distribution`'s output --
    typically `[("before union", ...), ("after union", ...)]` for one run's
    target-neighbour-union comparison, or one label per representation/
    backend for several runs side by side. Returns each series label's
    total cluster and node counts (summed across every cluster size, not the
    largest bucket alone). An empty `series`, or a series whose frame has no
    rows, draws nothing for that series rather than raising.
    """
    result: dict[str, dict[str, int]] = {}
    if not series:
        ax.set_title(title)
        return result

    for series_index, (label, frame) in enumerate(series):
        color = _SERIES_COLORS[series_index % len(_SERIES_COLORS)]
        counts = {
            "cluster_count": (
                int(frame.get_column("cluster_count").sum()) if frame.height else 0
            ),
            "node_count": (
                int(frame.get_column("node_count").sum()) if frame.height else 0
            ),
        }
        result[label] = counts
        if frame.height == 0:
            continue
        ordered = frame.sort("cluster_size")
        sizes = ordered.get_column("cluster_size").to_list()
        for value_column, linestyle in _VALUE_LINESTYLES.items():
            values = ordered.get_column(value_column).to_list()
            ax.plot(
                sizes,
                values,
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
                label=f"{label} {value_column} (total={counts[value_column]:,})",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cluster size (log)")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return result


def render_cluster_shape_pictures(
    series: Sequence[tuple[str, pl.DataFrame]],
    output_path: Path,
    *,
    suptitle: str,
) -> Path:
    """Render `draw_cluster_size_line`'s one picture to `output_path`.

    A one-axis figure, unlike `analysis.residual_pictures.render_residual_pictures`'s
    2x2 grid: cluster count and node volume share one axis here rather than
    a grid of separate panels.

    `series` empty writes a figure with one empty-titled panel rather than
    raising -- callers checking "is there anything to draw" should check
    `series` themselves first, matching `render_residual_pictures`'s own
    convention.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_cluster_size_line(ax, series, title=suptitle)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
