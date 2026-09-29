from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis.cluster_shape_pictures import (
    CLUSTER_SIZE_DISTRIBUTION_SCHEMA,
    draw_cluster_size_line,
    render_cluster_shape_pictures,
)


def _distribution(sizes: list[int], counts: list[int]) -> pl.DataFrame:
    """A `compute_cluster_size_distribution`-shaped frame -- `cluster_share`/
    `node_share` are never read by this module, so they are left at 0 rather
    than computed."""
    node_counts = [size * count for size, count in zip(sizes, counts, strict=True)]
    return pl.DataFrame(
        {
            "cluster_size": sizes,
            "cluster_count": counts,
            "node_count": node_counts,
            "cluster_share": [0.0] * len(sizes),
            "node_share": [0.0] * len(sizes),
        },
        schema=CLUSTER_SIZE_DISTRIBUTION_SCHEMA,
    )


def test_draw_cluster_size_line_reports_totals_and_draws_two_lines_per_series():
    before = _distribution([1, 2], [3, 1])  # 3 singletons, 1 pair -> 5 nodes
    after = _distribution([1, 3], [1, 1])  # 1 singleton, 1 triple -> 4 nodes

    fig, ax = plt.subplots()
    try:
        result = draw_cluster_size_line(
            ax, [("before union", before), ("after union", after)], title="test"
        )
    finally:
        plt.close(fig)

    assert result == {
        "before union": {"cluster_count": 4, "node_count": 5},
        "after union": {"cluster_count": 2, "node_count": 4},
    }
    # Two lines per series (cluster_count, node_count): four in total.
    assert len(ax.get_lines()) == 4


def test_draw_cluster_size_line_empty_series_draws_nothing():
    fig, ax = plt.subplots()
    try:
        result = draw_cluster_size_line(ax, [], title="test")
    finally:
        plt.close(fig)

    assert result == {}
    assert ax.get_lines() == []


def test_draw_cluster_size_line_series_with_no_rows_is_a_zero_total_not_an_error():
    empty = pl.DataFrame(schema=CLUSTER_SIZE_DISTRIBUTION_SCHEMA)

    fig, ax = plt.subplots()
    try:
        result = draw_cluster_size_line(ax, [("empty run", empty)], title="test")
    finally:
        plt.close(fig)

    assert result == {"empty run": {"cluster_count": 0, "node_count": 0}}
    assert ax.get_lines() == []


@pytest.mark.graphics
def test_render_cluster_shape_pictures_writes_a_png(tmp_path: Path):
    before = _distribution([1, 2], [3, 1])
    after = _distribution([1, 3], [1, 1])
    output_path = tmp_path / "viz" / "cluster_shape.png"

    result = render_cluster_shape_pictures(
        [("before union", before), ("after union", after)],
        output_path,
        suptitle="test",
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
