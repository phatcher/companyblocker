from __future__ import annotations

from pathlib import Path

import polars as pl

from workspace.identity import (
    RunKeys,
    combine_part_keys,
    current_commit,
    digest_file,
    digest_settings,
    index_key,
    neighbour_edges_key,
    rows_digest,
)

_PINNED_DIGEST = "205b035715aaa2a03df153aeb17f88db"


def _digest(ids: list[str], values: list[str | None]) -> str:
    return rows_digest(
        pl.DataFrame(
            {"id": ids, "value": values}, schema={"id": pl.Utf8, "value": pl.Utf8}
        ),
        id_col="id",
        value_col="value",
    )


def test_rows_digest_tells_a_missing_value_from_an_empty_one() -> None:
    assert _digest(["a"], [None]) != _digest(["a"], [""])


def test_rows_digest_keeps_values_apart_when_their_bytes_run_together() -> None:
    assert _digest(["a", "b"], ["ab", "c"]) != _digest(["a", "b"], ["a", "bc"])
    assert _digest(["ab", "c"], ["x", "y"]) != _digest(["a", "bc"], ["x", "y"])


def test_rows_digest_counts_the_rows() -> None:
    assert _digest([], []) != _digest([""], [""])


def test_rows_digest_is_independent_of_how_the_frame_is_chunked() -> None:
    whole = pl.DataFrame({"id": ["a", "b", "c"], "value": ["x", None, "z"]})
    chunked = pl.concat([whole.head(1), whole.tail(2)], rechunk=False)

    assert rows_digest(whole, id_col="id", value_col="value") == rows_digest(
        chunked, id_col="id", value_col="value"
    )


def test_rows_digest_is_pinned_to_a_stored_value() -> None:
    """A key is compared against keys earlier runs wrote, so the digest of a
    fixed frame must not move with a library upgrade or a refactor."""
    assert _digest(["b", "a"], ["Beta", None]) == _PINNED_DIGEST


def test_rows_digest_ignores_row_order_and_moves_with_a_value() -> None:
    forward = pl.DataFrame({"system_uri": ["a", "b"], "name": ["Acme", "Beta"]})
    reversed_rows = pl.DataFrame({"system_uri": ["b", "a"], "name": ["Beta", "Acme"]})
    changed = pl.DataFrame({"system_uri": ["a", "b"], "name": ["Acme", "Gamma"]})

    key = rows_digest(forward, id_col="system_uri", value_col="name")

    assert key == rows_digest(reversed_rows, id_col="system_uri", value_col="name")
    assert key != rows_digest(changed, id_col="system_uri", value_col="name")


def test_rows_digest_reads_only_the_two_named_columns() -> None:
    base = pl.DataFrame({"system_uri": ["a"], "name": ["Acme"]})
    widened = base.with_columns(pl.lit("unrelated").alias("extra"))

    assert rows_digest(base, id_col="system_uri", value_col="name") == rows_digest(
        widened, id_col="system_uri", value_col="name"
    )


def test_digest_settings_ignores_key_order() -> None:
    assert digest_settings({"a": 1, "b": 2}) == digest_settings({"b": 2, "a": 1})
    assert digest_settings({"a": 1}) != digest_settings({"a": 2})


def test_every_digest_is_the_one_width() -> None:
    assert len(_digest(["a"], ["x"])) == 32
    assert len(digest_settings({"a": 1})) == 32


def test_digest_file_reads_bytes_not_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text("vocab", encoding="utf-8")
    first = digest_file(path)

    path.write_text("vocab", encoding="utf-8")
    assert digest_file(path) == first

    path.write_text("vocab-optimized", encoding="utf-8")
    assert digest_file(path) != first


def test_combine_part_keys_names_each_part() -> None:
    assert combine_part_keys({"gb": "x", "ie": "y"}) == combine_part_keys(
        {"ie": "y", "gb": "x"}
    )
    assert combine_part_keys({"gb": "x", "ie": "y"}) != combine_part_keys(
        {"gb": "y", "ie": "x"}
    )


def test_index_key_moves_with_population_settings_and_tokenizer() -> None:
    base = index_key(
        population_key="p", build_settings={"ngram": 2}, tokenizer_digest=None
    )

    assert base != index_key(
        population_key="q", build_settings={"ngram": 2}, tokenizer_digest=None
    )
    assert base != index_key(
        population_key="p", build_settings={"ngram": 3}, tokenizer_digest=None
    )
    assert base != index_key(
        population_key="p", build_settings={"ngram": 2}, tokenizer_digest="t"
    )


def test_neighbour_edges_key_is_the_index_key_plus_edge_settings() -> None:
    base = neighbour_edges_key(index_key="i", edge_settings={"cap": 5})

    assert base != neighbour_edges_key(index_key="j", edge_settings={"cap": 5})
    assert base != neighbour_edges_key(index_key="i", edge_settings={"cap": 6})


def test_run_keys_round_trip_through_their_recorded_identity() -> None:
    keys = RunKeys(
        settings="s",
        source_population="sp",
        target_population="tp",
        truth=None,
        index="i",
    )

    assert RunKeys.from_identity(keys.as_identity()) == keys


def test_current_commit_is_none_outside_a_checkout(tmp_path: Path) -> None:
    assert current_commit(tmp_path) is None
