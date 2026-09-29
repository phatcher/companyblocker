from __future__ import annotations

import polars as pl
from company_cleanse.step_engines.stepengine_base import NoopCleanseStepEngine
from polars.testing import assert_frame_equal


def test_noop_step_engine_methods_are_noop_or_defaulting():
    engine = NoopCleanseStepEngine()
    source = pl.DataFrame(
        {
            "_effective_company_type": [None, "LIMITED"],
            "_normalized_name_body": ["acme limited", "beta llc"],
            "short_name": ["acme", "beta"],
            "name_cleansed": ["acme limited", "beta llc"],
            "quoted_name": [None, "ibis"],
        }
    )

    canonical = engine.extract_canonical_company_type(
        source.lazy(), {"LIMITED": "Ltd"}
    ).collect()
    assert canonical["company_type"].to_list() == ["private", "LIMITED"]

    cleansed = engine.generate_cleansed_company_name(
        source.lazy(), ("AND",), r"[^a-z]"
    ).collect()
    assert cleansed["name_cleansed"].to_list() == ["acme limited", "beta llc"]

    acronym = engine.derive_acronym_field(source.lazy()).collect()
    assert acronym["acronym"].to_list() == [None, None]

    assert_frame_equal(
        engine.ensure_non_acronym_short_name(source.lazy()).collect(), source
    )
    assert_frame_equal(
        engine.ensure_quoted_name_in_cleansed(source.lazy()).collect(), source
    )
