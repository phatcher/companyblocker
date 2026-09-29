from __future__ import annotations

import os
import random
import re
import time

import polars as pl
import pytest
from company_cleanse.pipeline import (
    STEP_ENGINE_POLARS,
    STEP_ENGINE_UDF,
    _step_extract_canonical_company_type,
    _step_resolve_and_unpack_company_type_decision,
)
from company_cleanse.rules import (
    COMPANY_TYPE_MAPPING,
    _build_company_type_suffix_trie,
    get_company_type_rules,
)
from polars.testing import assert_frame_equal

pytestmark = pytest.mark.performance


@pytest.mark.skipif(
    os.environ.get("PERF_BENCHMARK", "0") != "1",
    reason="Set PERF_BENCHMARK=1 to run performance benchmarks.",
)
def test_step_extract_canonical_company_type_udf_vs_polars() -> None:
    rows = int(os.environ.get("PERF_ROWS", "250000"))
    seed = int(os.environ.get("PERF_SEED", "20260630"))

    rng = random.Random(seed)
    variants = sorted(COMPANY_TYPE_MAPPING.keys())
    # Include null and unknown values to exercise fallback behavior.
    variants_with_fallback = [*variants, None, "UNKNOWN TYPE"]
    values = [rng.choice(variants_with_fallback) for _ in range(rows)]

    source = pl.DataFrame({"_effective_company_type": values})

    t0 = time.perf_counter()
    udf_actual = (
        _step_extract_canonical_company_type(
            source.lazy(),
            company_type_mapping=COMPANY_TYPE_MAPPING,
            engine=STEP_ENGINE_UDF,
        )
        .select(["company_type"])
        .collect()
    )
    udf_elapsed = max(time.perf_counter() - t0, 1e-9)

    t1 = time.perf_counter()
    polars_actual = (
        _step_extract_canonical_company_type(
            source.lazy(),
            company_type_mapping=COMPANY_TYPE_MAPPING,
            engine=STEP_ENGINE_POLARS,
        )
        .select(["company_type"])
        .collect()
    )
    polars_elapsed = max(time.perf_counter() - t1, 1e-9)

    assert_frame_equal(polars_actual, udf_actual)

    udf_rps = rows / udf_elapsed
    polars_rps = rows / polars_elapsed
    print(
        "canonical_company_type step benchmark: "
        f"rows={rows:,}, seed={seed}, "
        f"udf={udf_rps:,.0f} rows/s ({udf_elapsed:.3f}s), "
        f"polars={polars_rps:,.0f} rows/s ({polars_elapsed:.3f}s), "
        f"speedup={(polars_rps / udf_rps):.3f}x"
    )


@pytest.mark.skipif(
    os.environ.get("PERF_BENCHMARK", "0") != "1",
    reason="Set PERF_BENCHMARK=1 to run performance benchmarks.",
)
def test_step_resolve_company_type_decision_regex_vs_trie() -> None:
    rows = int(os.environ.get("PERF_ROWS", "250000"))
    seed = int(os.environ.get("PERF_SEED", "20260630"))
    rng = random.Random(seed)

    company_type_regex, company_type_mapping = get_company_type_rules()
    compiled_regex = re.compile(company_type_regex)
    suffix_trie, suffix_trie_max_tokens = _build_company_type_suffix_trie(
        company_type_mapping
    )

    variants = sorted(company_type_mapping.keys())
    surfaces = [
        f"{rng.choice(['ALPHA', 'BETA', 'GAMMA', 'DELTA'])} {rng.choice(variants)}"
        for _ in range(rows)
    ]
    source = pl.DataFrame(
        {
            "_special_company_type": [None] * rows,
            "_normalized_name_body": surfaces,
            "_name_body_reordered": surfaces,
            "_source_company_type_raw": [None] * rows,
        }
    )

    t0 = time.perf_counter()
    regex_actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="regex",
            company_type_regex=compiled_regex,
            suffix_trie=None,
            suffix_trie_max_tokens=0,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            ["_matched_company_type", "_effective_company_type", "company_type_source"]
        )
        .collect()
    )
    regex_elapsed = max(time.perf_counter() - t0, 1e-9)

    t1 = time.perf_counter()
    trie_actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="trie",
            company_type_regex=None,
            suffix_trie=suffix_trie,
            suffix_trie_max_tokens=suffix_trie_max_tokens,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            ["_matched_company_type", "_effective_company_type", "company_type_source"]
        )
        .collect()
    )
    trie_elapsed = max(time.perf_counter() - t1, 1e-9)

    assert_frame_equal(trie_actual, regex_actual)

    regex_rps = rows / regex_elapsed
    trie_rps = rows / trie_elapsed
    print(
        "company_type_resolution step benchmark: "
        f"rows={rows:,}, seed={seed}, "
        f"regex={regex_rps:,.0f} rows/s ({regex_elapsed:.3f}s), "
        f"trie={trie_rps:,.0f} rows/s ({trie_elapsed:.3f}s), "
        f"speedup={(trie_rps / regex_rps):.3f}x"
    )
