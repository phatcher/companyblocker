"""Direct tests for `validation.name_forms`.

What is pinned is what a stored entry relies on: the forms are derived from
the raw `name` and never read from a column the frame carried, the declared
column list is exactly what the derivation adds, and the ruleset digest is
stable within a process.
"""

from __future__ import annotations

import polars as pl
import pytest

from validation.name_forms import (
    NAME_FORM_COLUMNS,
    RAW_NAME_COLUMN,
    derive_name_forms,
    name_forms_ruleset_digest,
    name_forms_workers,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "system_uri": ["ie:1", "ie:2"],
            RAW_NAME_COLUMN: ["Acme Holdings Limited", "Murphy & Sons Teoranta"],
            "name_cleansed": ["stale", "stale"],
            "jurisdiction_code": ["ie", "ie"],
        }
    )


def test_forms_are_derived_from_the_raw_name_and_replace_what_the_frame_carried() -> (
    None
):
    out = derive_name_forms(_frame(), profile="default")

    assert out.get_column("name_cleansed").to_list() == [
        "acme holdings ltd",
        "murphy & sons ltd",
    ]
    assert (
        out.get_column(RAW_NAME_COLUMN).to_list()
        == _frame().get_column("name").to_list()
    )


def test_the_declared_columns_are_exactly_what_the_derivation_adds() -> None:
    bare = _frame().drop("name_cleansed")

    out = derive_name_forms(bare, profile="default")

    assert set(out.columns) - set(bare.columns) == set(NAME_FORM_COLUMNS)


def test_a_frame_with_no_raw_name_is_refused_and_an_empty_one_passes_through() -> None:
    with pytest.raises(ValueError, match="derived from the 'name' column"):
        derive_name_forms(_frame().drop(RAW_NAME_COLUMN), profile="default")

    empty = _frame().head(0)
    assert derive_name_forms(empty, profile="default").equals(empty)


def test_the_ruleset_digest_is_a_stable_short_hex_string() -> None:
    digest = name_forms_ruleset_digest()

    assert digest == name_forms_ruleset_digest()
    assert len(digest) == 16 and int(digest, 16) >= 0


def test_deriving_across_processes_gives_the_one_pass_frame() -> None:
    """A frame split across workers comes back as the single pass derived
    it, in the same row order: a row's forms are its own."""
    frame = pl.DataFrame(
        {
            "system_uri": [f"ie:{index}" for index in range(8)],
            "name": [
                "ACME HOLDINGS LIMITED",
                'THE "BLUE" CAFE LTD',
                "P. ROONEY & SONS",
                "Zeta Trading PLC",
                "OMEGA GROUP DAC",
                "J SMITH T/A SMITH TOOLS",
                "NORTHERN LIGHTS CO. LTD",
                "AB INITIO LIMITED",
            ],
        }
    )

    one_pass = derive_name_forms(frame, profile="default", workers=1)
    across_workers = derive_name_forms(
        pl.concat([frame] * 2), profile="default", workers=3
    )

    assert across_workers.equals(pl.concat([one_pass] * 2))


def test_the_worker_count_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`NAME_FORMS_WORKERS` sets how many processes a large frame is split
    across, and anything unreadable leaves the default standing."""
    monkeypatch.setenv("NAME_FORMS_WORKERS", "7")
    assert name_forms_workers() == 7

    monkeypatch.setenv("NAME_FORMS_WORKERS", "not a number")
    assert name_forms_workers() == 4

    monkeypatch.delenv("NAME_FORMS_WORKERS")
    assert name_forms_workers() == 4
