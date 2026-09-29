"""Direct tests for `validation.public_benchmark_materializer`.

A handful of rows is enough: the module's contract is the id/name wiring for
each table, the truth column it composes for the queried side, and the
multi-match disambiguation, none of which needs a real benchmark download.
The end-to-end run over this same shape lives in
`tests/blocking/test_workflow.py` beside the perturbed-dataset test it
follows.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from validation.public_benchmark_materializer import (
    BenchmarkFormatError,
    PublicBenchmarkMaterializationConfig,
    load_benchmark_matches,
    materialize_public_benchmark,
    read_benchmark_table,
)
from workspace.roots import default_workspace_roots


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _write_benchmark(
    root: Path,
    *,
    table_a: list[dict[str, object]],
    table_b: list[dict[str, object]],
    matches: list[dict[str, object]] | None = None,
    splits: dict[str, list[dict[str, object]]] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "tableA.csv", table_a)
    _write_csv(root / "tableB.csv", table_b)
    if matches is not None:
        _write_csv(root / "matches.csv", matches)
    if splits is not None:
        for split_name, rows in splits.items():
            _write_csv(root / f"{split_name}.csv", rows)
    return root


# -- read_benchmark_table ---------------------------------------------------


def test_read_benchmark_table_concatenates_text_columns_in_order(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "tableA.csv"
    _write_csv(
        csv_path,
        [
            {
                "id": 0,
                "name": "Acme Widget",
                "description": "A fine widget",
                "price": 10,
            },
            {"id": 1, "name": "Zenith Gadget", "description": "A gadget", "price": 20},
        ],
    )

    table = read_benchmark_table(csv_path)

    assert table.columns == ["id", "name"]
    assert table.get_column("id").to_list() == ["0", "1"]
    assert table.get_column("name").to_list() == [
        "Acme Widget A fine widget",
        "Zenith Gadget A gadget",
    ]


def test_read_benchmark_table_requires_id_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "tableA.csv"
    _write_csv(csv_path, [{"name": "Acme Widget"}])

    with pytest.raises(BenchmarkFormatError, match="id"):
        read_benchmark_table(csv_path)


def test_read_benchmark_table_requires_a_textual_attribute(tmp_path: Path) -> None:
    csv_path = tmp_path / "tableA.csv"
    _write_csv(csv_path, [{"id": 0, "price": 10}])

    with pytest.raises(BenchmarkFormatError, match="textual attribute"):
        read_benchmark_table(csv_path)


# -- load_benchmark_matches --------------------------------------------------


def test_load_benchmark_matches_reads_matches_csv(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "matches.csv",
        [{"ltable_id": 0, "rtable_id": 1}, {"ltable_id": 2, "rtable_id": 3}],
    )

    matches = load_benchmark_matches(tmp_path)

    assert sorted(matches.iter_rows()) == [("0", "1"), ("2", "3")]


def test_load_benchmark_matches_derives_from_labelled_splits(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "train.csv",
        [
            {"ltable_id": 0, "rtable_id": 1, "label": 1},
            {"ltable_id": 0, "rtable_id": 2, "label": 0},
        ],
    )
    _write_csv(
        tmp_path / "test.csv",
        [{"ltable_id": 3, "rtable_id": 4, "label": 1}],
    )

    matches = load_benchmark_matches(tmp_path)

    assert sorted(matches.iter_rows()) == [("0", "1"), ("3", "4")]


def test_load_benchmark_matches_requires_matches_or_splits(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkFormatError, match="matches.csv"):
        load_benchmark_matches(tmp_path)


# -- materialize_public_benchmark -------------------------------------------


def _fixture_root(tmp_path: Path) -> Path:
    return _write_benchmark(
        tmp_path / "benchmark",
        table_a=[
            {
                "id": 0,
                "name": "Acme Widget",
                "description": "A fine widget",
                "price": 10,
            },
            {"id": 1, "name": "Zenith Gadget", "description": "A gadget", "price": 20},
        ],
        table_b=[
            {
                "id": 0,
                "name": "Acme Widgets Inc",
                "description": "widget maker",
                "price": 11,
            },
            {"id": 1, "name": "Other Corp", "description": "unrelated", "price": 5},
            {
                "id": 2,
                "name": "Zenith Gadgets",
                "description": "gadget maker",
                "price": 21,
            },
        ],
        matches=[
            {"ltable_id": 0, "rtable_id": 0},
            {"ltable_id": 1, "rtable_id": 2},
            {"ltable_id": 0, "rtable_id": 2},
        ],
    )


@pytest.mark.integration
def test_materialize_public_benchmark_writes_system_name_and_truth_wiring(
    tmp_path: Path,
) -> None:
    """Table A is the indexed system with no truth column; table B is the
    queried system, one row per record except the doubly-matched one, which
    is one row per true pair -- the perturbed-dataset pattern this module
    reuses (`source_uri`, not `match_uri`)."""
    roots = default_workspace_roots(tmp_path / "repo")
    config = PublicBenchmarkMaterializationConfig(
        roots=roots, benchmark_root=_fixture_root(tmp_path), benchmark_name="testbench"
    )

    result = materialize_public_benchmark(config)

    assert result.indexed_system == "testbench-a"
    assert result.queried_system == "testbench-b"
    assert result.indexed_rows == 2
    assert result.queried_rows == 4  # 3 records + one extra row for the double match
    assert result.truth_pairs == 3
    assert result.multi_match_source_count == 1

    indexed = pl.read_parquet(list(result.indexed_dir.rglob("*.parquet")))
    assert set(indexed.columns) == {"system_uri", "name", "jurisdiction_code"}
    assert sorted(indexed.get_column("system_uri").to_list()) == [
        "testbench-a://0",
        "testbench-a://1",
    ]

    queried = pl.read_parquet(list(result.queried_dir.rglob("*.parquet")))
    assert set(queried.columns) == {
        "system_uri",
        "name",
        "jurisdiction_code",
        "source_uri",
    }
    by_system_uri = dict(
        zip(
            queried.get_column("system_uri").to_list(),
            queried.get_column("source_uri").to_list(),
            strict=True,
        )
    )
    assert by_system_uri["testbench-b://0"] == "testbench-a://0"
    assert by_system_uri["testbench-b://1"] is None
    assert by_system_uri["testbench-b://2~0"] == "testbench-a://0"
    assert by_system_uri["testbench-b://2~1"] == "testbench-a://1"


@pytest.mark.integration
def test_materialize_public_benchmark_rejects_matches_naming_unknown_ids(
    tmp_path: Path,
) -> None:
    root = _write_benchmark(
        tmp_path / "benchmark",
        table_a=[{"id": 0, "name": "Acme"}],
        table_b=[{"id": 0, "name": "Acme Inc"}],
        matches=[{"ltable_id": 99, "rtable_id": 0}],
    )
    config = PublicBenchmarkMaterializationConfig(
        roots=default_workspace_roots(tmp_path / "repo"),
        benchmark_root=root,
        benchmark_name="testbench",
    )

    with pytest.raises(BenchmarkFormatError, match="ltable_id"):
        materialize_public_benchmark(config)
