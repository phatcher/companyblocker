from __future__ import annotations

import polars as pl
from company_cleanse.rules import COMPANY_TYPE_MAPPING
from company_cleanse.step_engines import stepengine_polars_ops, stepengine_udf_ops
from company_cleanse.step_engines.stepengine_polars import PolarsCleanseStepEngine
from company_cleanse.step_engines.stepengine_udf import UdfCleanseStepEngine
from polars.testing import assert_frame_equal


class _RecordingOps:
    """Records which contract methods a `DelegatingCleanseStepEngine` calls on it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract_canonical_company_type(self, lf, company_type_mapping):
        del company_type_mapping
        self.calls.append("extract_canonical_company_type")
        return lf

    def generate_cleansed_company_name(
        self, lf, and_tokens, char_whitelist, normalization_operations=()
    ):
        del and_tokens, char_whitelist, normalization_operations
        self.calls.append("generate_cleansed_company_name")
        return lf

    def derive_acronym_field(self, lf):
        self.calls.append("derive_acronym_field")
        return lf

    def ensure_non_acronym_short_name(self, lf):
        self.calls.append("ensure_non_acronym_short_name")
        return lf

    def ensure_quoted_name_in_cleansed(self, lf):
        self.calls.append("ensure_quoted_name_in_cleansed")
        return lf


def test_polars_engine_defaults_to_the_polars_ops_module():
    engine = PolarsCleanseStepEngine()
    lf = pl.DataFrame({"_effective_company_type": ["LIMITED"]}).lazy()

    engine_actual = engine.extract_canonical_company_type(
        lf, COMPANY_TYPE_MAPPING
    ).collect()
    ops_actual = stepengine_polars_ops.extract_canonical_company_type(
        lf, COMPANY_TYPE_MAPPING
    ).collect()

    assert_frame_equal(engine_actual, ops_actual)


def test_udf_engine_defaults_to_the_udf_ops_module():
    engine = UdfCleanseStepEngine()
    lf = pl.DataFrame({"_effective_company_type": ["LIMITED"]}).lazy()

    engine_actual = engine.extract_canonical_company_type(
        lf, COMPANY_TYPE_MAPPING
    ).collect()
    ops_actual = stepengine_udf_ops.extract_canonical_company_type(
        lf, COMPANY_TYPE_MAPPING
    ).collect()

    assert_frame_equal(engine_actual, ops_actual)


def test_polars_engine_accepts_an_ops_override_and_delegates_all_five_methods():
    ops = _RecordingOps()
    engine = PolarsCleanseStepEngine(ops=ops)
    lf = pl.DataFrame({"x": [1]}).lazy()

    engine.extract_canonical_company_type(lf, {})
    engine.generate_cleansed_company_name(lf, None, "")
    engine.derive_acronym_field(lf)
    engine.ensure_non_acronym_short_name(lf)
    engine.ensure_quoted_name_in_cleansed(lf)

    assert ops.calls == [
        "extract_canonical_company_type",
        "generate_cleansed_company_name",
        "derive_acronym_field",
        "ensure_non_acronym_short_name",
        "ensure_quoted_name_in_cleansed",
    ]


def test_udf_engine_accepts_an_ops_override_and_delegates_all_five_methods():
    ops = _RecordingOps()
    engine = UdfCleanseStepEngine(ops=ops)
    lf = pl.DataFrame({"x": [1]}).lazy()

    engine.extract_canonical_company_type(lf, {})
    engine.generate_cleansed_company_name(lf, None, "")
    engine.derive_acronym_field(lf)
    engine.ensure_non_acronym_short_name(lf)
    engine.ensure_quoted_name_in_cleansed(lf)

    assert ops.calls == [
        "extract_canonical_company_type",
        "generate_cleansed_company_name",
        "derive_acronym_field",
        "ensure_non_acronym_short_name",
        "ensure_quoted_name_in_cleansed",
    ]
