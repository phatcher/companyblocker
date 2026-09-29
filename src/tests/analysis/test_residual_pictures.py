from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis.residual_pictures import (
    RESIDUAL_PICTURE_FRAME_SCHEMA,
    build_residual_picture_frame,
    draw_axis_histogram,
    draw_cluster_size_cumulative,
    draw_missed_fraction_hex,
    draw_recall_by_rank,
    render_residual_pictures,
    resolve_residual_pairs_path,
)

_RESIDUAL_PAIRS_COLUMNS = [
    "source_system",
    "target_system",
    "country",
    "source_id",
    "target_id",
    "source_name",
    "source_name_cleansed",
    "target_name",
    "target_name_cleansed",
    "name_equality",
    "is_truth_pair",
    "found",
    "similarity",
    "rank",
]


def _residual_pairs_frame() -> pl.DataFrame:
    """A small, hand-built `residual_pairs`-shaped frame: three
    `never`-level truth pairs (two found, one missed), one
    `raw`-level truth pair (excluded by the level filter) and one
    predicted-but-untrue row (excluded by `is_truth_pair`)."""
    rows = [
        # cleansed_different, found, rank 1 -- source clusters with 1 other node.
        (
            "gleif",
            "gb",
            "gb",
            "s1",
            "t1",
            "Acme Systems",
            "acme systems",
            "Acme Sys",
            "acme sys",
            "never",
            True,
            True,
            0.9,
            1,
        ),
        # cleansed_different, found, rank 3 -- source not in clusters.parquet at all (singleton).
        (
            "gleif",
            "gb",
            "gb",
            "s2",
            "t2",
            "Beta Holdings",
            "beta holdings",
            "Beta Hold",
            "beta hold",
            "never",
            True,
            True,
            0.7,
            3,
        ),
        # cleansed_different, missed -- no similarity/rank at all; source clusters with 3 others.
        (
            "gleif",
            "gb",
            "gb",
            "s3",
            "t3",
            "Gamma Ltd",
            "gamma",
            "Completely Different",
            "completely different",
            "never",
            True,
            False,
            None,
            None,
        ),
        # raw_identical truth pair -- excluded by the difficult-bucket filter.
        (
            "gleif",
            "gb",
            "gb",
            "s4",
            "t4",
            "Delta",
            "delta",
            "Delta",
            "delta",
            "raw",
            True,
            True,
            1.0,
            1,
        ),
        # predicted-but-untrue row -- excluded by is_truth_pair.
        (
            "gleif",
            "gb",
            "gb",
            "s5",
            "t5",
            "Epsilon",
            "epsilon",
            "Not Epsilon",
            "not epsilon",
            "never",
            False,
            None,
            0.4,
            2,
        ),
    ]
    return pl.DataFrame(rows, schema=_RESIDUAL_PAIRS_COLUMNS, orient="row")


def _clusters_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cluster_id": ["c1", "c1", "c2", "c2", "c2", "c2"],
            "node_id": ["s1", "t1", "s3", "t3", "other-a", "other-b"],
            "node_name": [
                "Acme Systems",
                "Acme Sys",
                "Gamma Ltd",
                "Completely Different",
                "X",
                "Y",
            ],
            "node_role": ["source", "target", "source", "target", "source", "target"],
        }
    )


def _write_run_dir(
    tmp_path: Path, *, residual_pairs_name: str = "residual_pairs.parquet"
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _residual_pairs_frame().write_parquet(run_dir / residual_pairs_name)
    _clusters_frame().write_parquet(run_dir / "clusters.parquet")
    return run_dir


def test_resolve_residual_pairs_path_prefers_the_renamed_artifact(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "residual_pairs.parquet").write_bytes(b"old")
    (run_dir / "pair_truth_eval_detail.parquet").write_bytes(b"new")

    assert resolve_residual_pairs_path(run_dir).name == "pair_truth_eval_detail.parquet"


def test_resolve_residual_pairs_path_falls_back_to_the_old_name(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "residual_pairs.parquet").write_bytes(b"old")

    assert resolve_residual_pairs_path(run_dir).name == "residual_pairs.parquet"


def test_resolve_residual_pairs_path_raises_when_neither_exists(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_residual_pairs_path(run_dir)


def test_build_residual_picture_frame_keeps_only_difficult_truth_pairs(tmp_path: Path):
    run_dir = _write_run_dir(tmp_path)

    frame = build_residual_picture_frame(run_dir)

    # s1, s2 (found), s3 (missed) -- s4 (raw_identical) and s5 (not a truth
    # pair) are excluded.
    assert frame.height == 3
    assert set(frame.get_column("source_id").to_list()) == {"s1", "s2", "s3"}
    assert set(frame.schema.keys()) == set(RESIDUAL_PICTURE_FRAME_SCHEMA.keys())


def test_build_residual_picture_frame_computes_axes_and_cluster_size(tmp_path: Path):
    run_dir = _write_run_dir(tmp_path)

    frame = build_residual_picture_frame(run_dir).sort("source_id")
    by_source = {row["source_id"]: row for row in frame.iter_rows(named=True)}

    # s1: "acme systems" vs "acme sys" -- one shared token ("acme") of three
    # distinct tokens union {"acme", "systems", "sys"}.
    assert by_source["s1"]["token_jaccard"] == pytest.approx(1 / 3)
    assert by_source["s1"]["cluster_size"] == 2  # c1 has 2 nodes (s1, t1)

    # s2's source id never appears in clusters.parquet -- a singleton, not null.
    assert by_source["s2"]["cluster_size"] == 1

    # s3: "gamma" vs "completely different" share no tokens at all.
    assert by_source["s3"]["token_jaccard"] == 0.0
    assert by_source["s3"]["cluster_size"] == 4  # c2 has 4 nodes
    assert by_source["s3"]["found"] is False
    assert by_source["s3"]["similarity"] is None
    assert by_source["s3"]["rank"] is None

    assert by_source["s1"]["length_diff"] == abs(len("acme systems") - len("acme sys"))


def test_build_residual_picture_frame_empty_when_no_difficult_truth_pairs(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    only_easy = _residual_pairs_frame().filter(pl.col("name_equality") != "never")
    only_easy.write_parquet(run_dir / "residual_pairs.parquet")
    _clusters_frame().write_parquet(run_dir / "clusters.parquet")

    frame = build_residual_picture_frame(run_dir)

    assert frame.height == 0
    assert set(frame.schema.keys()) == set(RESIDUAL_PICTURE_FRAME_SCHEMA.keys())


def test_draw_axis_histogram_reports_found_and_missed_counts():
    frame = pl.DataFrame(
        {
            "found": [True, True, False],
            "token_jaccard": [0.9, 0.8, 0.1],
        }
    )
    fig, ax = plt.subplots()
    try:
        result = draw_axis_histogram(ax, [("run-a", frame)], title="test")
    finally:
        plt.close(fig)

    assert result == {"run-a": {"found": 2, "missed": 1}}


def test_draw_axis_histogram_empty_series_draws_nothing():
    fig, ax = plt.subplots()
    try:
        result = draw_axis_histogram(ax, [], title="test")
    finally:
        plt.close(fig)

    assert result == {}


def test_draw_recall_by_rank_ceiling_matches_found_share():
    frame = pl.DataFrame(
        {
            "found": [True, True, False, False],
            "rank": [1, 3, None, None],
        }
    )
    fig, ax = plt.subplots()
    try:
        result = draw_recall_by_rank(ax, [("run-a", frame)], title="test")
    finally:
        plt.close(fig)

    assert result["run-a"] == pytest.approx(0.5)


def test_draw_cluster_size_cumulative_reaches_one_at_the_largest_size():
    frame = pl.DataFrame(
        {
            "found": [True, True, False],
            "cluster_size": [1, 5, 2],
        }
    )
    fig, ax = plt.subplots()
    try:
        result = draw_cluster_size_cumulative(ax, [("run-a", frame)], title="test")
    finally:
        plt.close(fig)

    assert result == {"run-a": {"found": 2, "missed": 1}}
    lines = {line.get_label(): line for line in ax.get_lines()}
    found_line = lines["run-a found (n=2)"]
    assert found_line.get_ydata()[-1] == pytest.approx(1.0)


def test_draw_missed_fraction_hex_on_empty_frame_draws_nothing():
    empty = pl.DataFrame(schema=RESIDUAL_PICTURE_FRAME_SCHEMA)
    fig, ax = plt.subplots()
    try:
        cell_count = draw_missed_fraction_hex(ax, empty, title="test")
    finally:
        plt.close(fig)

    assert cell_count == 0


@pytest.mark.graphics
def test_render_residual_pictures_writes_a_png(tmp_path: Path):
    run_dir = _write_run_dir(tmp_path)
    frame = build_residual_picture_frame(run_dir)
    output_path = tmp_path / "viz" / "residual_pictures.png"

    result = render_residual_pictures([("tfidf", frame)], output_path, suptitle="test")

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
