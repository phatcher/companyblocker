from __future__ import annotations

import polars as pl
import pytest
from company_cleanse.rules import COMPANY_TYPE_MAPPING
from company_cleanse.step_engines import (
    stepengine_noop_ops,
    stepengine_polars_ops,
    stepengine_udf_ops,
)
from company_cleanse.step_engines.stepengine_common_ops import with_company_type_column
from polars.testing import assert_frame_equal

# Every step-engine ops module -- noop, polars and udf -- implements the same five
# free functions, called with the same arguments by `DelegatingCleanseStepEngine`.
# A behaviour genuinely shared by all three belongs here, parametrized once, so a
# fourth implementation inherits the same bar by existing rather than by someone
# remembering to add a case for it.
_OPS_IMPLEMENTATIONS = (stepengine_noop_ops, stepengine_polars_ops, stepengine_udf_ops)


def _ops_id(ops) -> str:
    return ops.__name__.rsplit(".", 1)[-1]


@pytest.mark.parametrize("ops", _OPS_IMPLEMENTATIONS, ids=_ops_id)
def test_extract_canonical_company_type_defaults_unresolved_to_private(ops):
    source = pl.DataFrame({"_effective_company_type": [None]})

    actual = ops.extract_canonical_company_type(source.lazy(), {}).collect()

    assert actual["company_type"][0] == "private"


@pytest.mark.parametrize("ops", _OPS_IMPLEMENTATIONS, ids=_ops_id)
def test_ensure_non_acronym_short_name_is_noop_without_a_short_name(ops):
    source = pl.DataFrame(
        {
            "short_name": pl.Series([None], dtype=pl.Utf8),
            "_short_name_norm": [""],
            "acronym": pl.Series([None], dtype=pl.Utf8),
            "name_cleansed": ["acme corp"],
            "_name_cleansed_norm": ["acme corp"],
        }
    )

    actual = ops.ensure_non_acronym_short_name(source.lazy()).collect()

    assert actual["name_cleansed"][0] == "acme corp"


@pytest.mark.parametrize("ops", _OPS_IMPLEMENTATIONS, ids=_ops_id)
def test_ensure_quoted_name_in_cleansed_is_noop_without_a_quoted_name(ops):
    source = pl.DataFrame(
        {
            "quoted_name": pl.Series([None], dtype=pl.Utf8),
            "_quoted_name_norm": [""],
            "acronym": pl.Series([None], dtype=pl.Utf8),
            "name_cleansed": ["acme corp"],
            "_name_cleansed_norm": ["acme corp"],
        }
    )

    actual = ops.ensure_quoted_name_in_cleansed(source.lazy()).collect()

    assert actual["name_cleansed"][0] == "acme corp"


# `stepengine_polars_ops` and `stepengine_udf_ops` are two implementation strategies
# -- vectorized expressions versus row-wise UDFs -- for the exact same contract, so
# the real assertion worth making is that they agree, not that either one alone
# produces some expected value. One parametrized runner across the five methods
# replaces what would otherwise be six unrelated per-method tests.
_EQUIVALENCE_CASES = [
    (
        "extract_canonical_company_type",
        {
            "_effective_company_type": [
                "LIMITED",
                "LLC",
                "PRIVATE",
                None,
                "GMBH",
                "UNKNOWNTYPE",
            ]
        },
        {"company_type_mapping": COMPANY_TYPE_MAPPING},
    ),
    (
        "generate_cleansed_company_name",
        {
            "_normalized_name_body": [
                "ACME LIMITED",
                "SMITH & JONES LIMITED",
                "EXAMPLE PRIVATE",
                "OMEGA GMBH",
            ],
            "_matched_company_type": ["LIMITED", "LIMITED", None, "GMBH"],
            "company_type": ["ltd", "ltd", "private", "gmbh"],
        },
        {"and_tokens": ("AND",), "char_whitelist": r"[^a-z0-9\s!&]"},
    ),
    (
        "derive_acronym_field",
        {
            "_special_short_name": [None, "IBM", None],
            "short_name": ["ACME", "INTERNATIONAL", "XYZ"],
            "name_cleansed": [
                "acme holdings ltd",
                "international business machines ltd",
                "xyz consulting",
            ],
            "company_type": ["ltd", "ltd", "private"],
            "_name_cleansed_norm": [
                "acme holdings ltd",
                "international business machines ltd",
                "xyz consulting",
            ],
            "_company_type_norm": ["ltd", "ltd", "private"],
            "_short_name_norm": ["acme", "international", "xyz"],
        },
        {},
    ),
    (
        "ensure_non_acronym_short_name",
        {
            "short_name": ["LTD", "ALPHA", "", None],
            "acronym": [None, None, None, "LLC"],
            "name_cleansed": [
                "acme holdings",
                "alpha services",
                "delta ltd",
                "gamma llc",
            ],
            "_short_name_norm": ["ltd", "alpha", "", ""],
            "_name_cleansed_norm": [
                "acme holdings",
                "alpha services",
                "delta ltd",
                "gamma llc",
            ],
        },
        {},
    ),
    (
        "ensure_quoted_name_in_cleansed",
        {
            "quoted_name": ["IBIS", None, "ACME"],
            "acronym": [None, None, "acme"],
            "name_cleansed": ["corporation ltd", "beta ltd", ""],
            "_quoted_name_norm": ["ibis", "", "acme"],
            "_name_cleansed_norm": ["corporation ltd", "beta ltd", ""],
        },
        {},
    ),
]


@pytest.mark.parametrize(
    "method_name, frame_data, method_kwargs",
    _EQUIVALENCE_CASES,
    ids=[case[0] for case in _EQUIVALENCE_CASES],
)
def test_polars_and_udf_ops_agree_on_the_shared_contract(
    method_name, frame_data, method_kwargs
):
    lf = pl.DataFrame(frame_data).lazy()
    polars_fn = getattr(stepengine_polars_ops, method_name)
    udf_fn = getattr(stepengine_udf_ops, method_name)

    polars_actual = polars_fn(lf, **method_kwargs).collect()
    udf_actual = udf_fn(lf, **method_kwargs).collect()

    assert_frame_equal(polars_actual, udf_actual)


def test_with_company_type_column_defaults_null_to_private():
    source = pl.DataFrame({"raw": [None, "gmbh"]})

    actual = with_company_type_column(
        source.lazy(), company_type_expr=pl.col("raw")
    ).collect()

    assert actual["company_type"].to_list() == ["private", "gmbh"]


def test_with_company_type_column_preserves_other_columns():
    source = pl.DataFrame({"raw": ["ltd"], "keep": ["value"]})

    actual = with_company_type_column(
        source.lazy(), company_type_expr=pl.col("raw")
    ).collect()

    assert actual["keep"].to_list() == ["value"]


def test_noop_ops_ignores_its_arguments_and_returns_fixed_defaults():
    source = pl.DataFrame(
        {
            "_effective_company_type": ["LIMITED"],
            "_normalized_name_body": ["acme limited"],
            "short_name": ["acme"],
            "name_cleansed": ["acme limited"],
            "quoted_name": [None],
        }
    )

    canonical = stepengine_noop_ops.extract_canonical_company_type(
        source.lazy(), {"LIMITED": "Ltd"}
    ).collect()
    assert canonical["company_type"].to_list() == ["LIMITED"]

    cleansed = stepengine_noop_ops.generate_cleansed_company_name(
        source.lazy(), ("AND",), r"[^a-z]", ("lowercase",)
    ).collect()
    assert cleansed["name_cleansed"].to_list() == ["acme limited"]

    acronym = stepengine_noop_ops.derive_acronym_field(source.lazy()).collect()
    assert acronym["acronym"].to_list() == [None]

    assert_frame_equal(
        stepengine_noop_ops.ensure_non_acronym_short_name(source.lazy()).collect(),
        source,
    )
    assert_frame_equal(
        stepengine_noop_ops.ensure_quoted_name_in_cleansed(source.lazy()).collect(),
        source,
    )
