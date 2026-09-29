"""Direct tests for `workspace.name_forms_store`.

What is pinned is what two areas scoring the same names rely on: the forms are
derived once per population and pair of profiles, every later call reads the
same columns back without deriving, and anything that changes what would be
derived gives a new entry.
"""

from __future__ import annotations

import polars as pl
import pytest

from workspace.name_forms_store import resolve_name_forms
from workspace.roots import WorkspaceRoots

_COLUMNS = ("name_cleansed", "name_preprocessed")


def _frame(names: dict[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"system_uri": list(names), "name": list(names.values())},
        schema={"system_uri": pl.Utf8, "name": pl.Utf8},
    )


def _derive(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("name").str.to_lowercase().alias("name_cleansed"),
        pl.col("name")
        .str.to_lowercase()
        .str.replace(" ltd$", "")
        .alias("name_preprocessed"),
    )


def _never(frame: pl.DataFrame) -> pl.DataFrame:
    raise AssertionError("the store held these forms; nothing should be derived")


def _resolve(
    roots: WorkspaceRoots, frame: pl.DataFrame, derive=_derive, **overrides: str
):
    settings = {
        "cleanse_profile": "default",
        "preprocess_profile": "default",
        "scored_col": "name_cleansed",
        "deriver_version": "1",
        **overrides,
    }
    return resolve_name_forms(
        roots,
        frame,
        system="ie",
        country="ie",
        columns=_COLUMNS,
        derive=derive,
        **settings,
    )


def test_forms_are_derived_once_and_read_back_the_same(workspace_roots: WorkspaceRoots):
    frame = _frame({"ie:1": "Acme Ltd", "ie:2": "Murphy & Sons Ltd", "ie:3": None})

    first, first_was_stored = _resolve(workspace_roots, frame)
    again, again_was_stored = _resolve(workspace_roots, frame, derive=_never)

    assert (first_was_stored, again_was_stored) == (False, True)
    assert first.equals(again)
    assert first.get_column("name_preprocessed").to_list() == [
        "acme",
        "murphy & sons",
        None,
    ]


def test_the_same_rows_in_another_order_or_with_other_columns_reuse_the_entry(
    workspace_roots: WorkspaceRoots,
):
    _resolve(workspace_roots, _frame({"ie:1": "Acme Ltd", "ie:2": "Beta Ltd"}))
    rewritten = _frame({"ie:2": "Beta Ltd", "ie:1": "Acme Ltd"}).with_columns(
        pl.lit("stale").alias("name_cleansed"), pl.lit("ie").alias("jurisdiction_code")
    )

    out, was_stored = _resolve(workspace_roots, rewritten, derive=_never)

    assert was_stored
    assert out.get_column("system_uri").to_list() == ["ie:2", "ie:1"]
    assert out.get_column("name_cleansed").to_list() == ["beta ltd", "acme ltd"]
    assert out.get_column("jurisdiction_code").to_list() == ["ie", "ie"]


@pytest.mark.parametrize(
    "change",
    [
        {"cleanse_profile": "default|-lowercase"},
        {"preprocess_profile": "default|+noise_words:balanced"},
        {"scored_col": "name"},
        {"deriver_version": "2"},
    ],
)
def test_a_changed_profile_or_ruleset_is_derived_again(
    workspace_roots: WorkspaceRoots, change: dict[str, str]
):
    frame = _frame({"ie:1": "Acme Ltd"})
    _resolve(workspace_roots, frame)

    _, was_stored = _resolve(workspace_roots, frame, **change)

    assert not was_stored


def test_a_changed_name_is_derived_again(workspace_roots: WorkspaceRoots):
    _resolve(workspace_roots, _frame({"ie:1": "Acme Ltd"}))

    out, was_stored = _resolve(workspace_roots, _frame({"ie:1": "Acme Holdings Ltd"}))

    assert not was_stored
    assert out.get_column("name_preprocessed").to_list() == ["acme holdings"]


def test_a_derivation_that_drops_a_column_is_refused_and_nothing_is_stored(
    workspace_roots: WorkspaceRoots,
):
    frame = _frame({"ie:1": "Acme Ltd"})

    with pytest.raises(ValueError, match="name_preprocessed"):
        _resolve(
            workspace_roots,
            frame,
            derive=lambda f: _derive(f).drop("name_preprocessed"),
        )

    _, was_stored = _resolve(workspace_roots, frame)
    assert not was_stored
