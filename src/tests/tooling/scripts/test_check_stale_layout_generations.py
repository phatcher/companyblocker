from __future__ import annotations

from pathlib import Path

from scripts import check_stale_layout_generations as check


def _make_partition(base: Path, *, relative: str) -> None:
    directory = base / relative
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "part-00000.parquet").write_bytes(b"")


def test_discovers_layer_with_both_top_level_and_primary_generation(
    tmp_path: Path,
) -> None:
    layer = tmp_path / "gleif" / "matched"
    _make_partition(layer, relative="jurisdiction_code=gb")
    _make_partition(layer, relative="primary/jurisdiction_code=gb")

    stale = check.discover_stale_generations(tmp_path)

    assert stale == [check.StaleGeneration(layer_dir="gleif/matched", values=("gb",))]


def test_ignores_layer_with_only_primary_generation(tmp_path: Path) -> None:
    layer = tmp_path / "gb" / "canonical" / "2026-06-01"
    _make_partition(layer, relative="primary/jurisdiction_code=gb")

    assert check.discover_stale_generations(tmp_path) == []


def test_ignores_layer_with_only_top_level_generation(tmp_path: Path) -> None:
    layer = tmp_path / "gb" / "canonical" / "2026-06-01"
    _make_partition(layer, relative="jurisdiction_code=gb")

    assert check.discover_stale_generations(tmp_path) == []


def test_reports_only_the_values_present_in_both_generations(tmp_path: Path) -> None:
    layer = tmp_path / "gleif" / "matched"
    _make_partition(layer, relative="jurisdiction_code=gb")
    _make_partition(layer, relative="primary/jurisdiction_code=gb")
    _make_partition(layer, relative="primary/jurisdiction_code=ie")

    stale = check.discover_stale_generations(tmp_path)

    assert stale == [check.StaleGeneration(layer_dir="gleif/matched", values=("gb",))]


def test_covers_a_layer_the_checker_was_not_told_about(tmp_path: Path) -> None:
    """No hand-listed set of layers -- an arbitrary system/layer path is found
    by walking, matching this item's `by query, not by listing` requirement."""
    layer = tmp_path / "some_new_system" / "some_new_layer" / "2026-09-01"
    _make_partition(layer, relative="jurisdiction_code=fr")
    _make_partition(layer, relative="primary/jurisdiction_code=fr")

    stale = check.discover_stale_generations(tmp_path)

    assert stale == [
        check.StaleGeneration(
            layer_dir="some_new_system/some_new_layer/2026-09-01",
            values=("fr",),
        )
    ]


def test_measure_returns_counts_keyed_by_layer(tmp_path: Path) -> None:
    layer = tmp_path / "gleif" / "matched"
    _make_partition(layer, relative="jurisdiction_code=gb")
    _make_partition(layer, relative="primary/jurisdiction_code=gb")
    _make_partition(layer, relative="jurisdiction_code=ie")
    _make_partition(layer, relative="primary/jurisdiction_code=ie")

    assert check.measure(tmp_path) == {"gleif/matched": 2}
