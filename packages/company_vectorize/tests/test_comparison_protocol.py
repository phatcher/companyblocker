"""Unit coverage for the fixed comparison protocol's constants.

Everything here is a pure data check -- the protocol module carries no
behaviour of its own, only the values `scripts/compare_blocking_strategies.py`
enforces.
"""

from __future__ import annotations

import polars as pl
import pytest
from company_vectorize.comparison_protocol import (
    CELL_STATUSES,
    DENSE_VOCABULARY_MAX_ROWS,
    EXCLUDED_SLICES,
    FIXED_SLICES,
    GATED_REFERENCE_CELLS,
    MANDATORY_SUITE,
    REFERENCE_SUITE,
    RUNTIME_BUDGET_SECONDS,
    BaselineSuiteCell,
    BaselineSuiteIncompleteError,
    BaselineSuiteStatus,
    missing_baseline_suite_cells,
    require_baseline_suite_complete,
)
from company_vectorize.dense_vocabulary_gate import DEFAULT_DENSE_VOCABULARY_MAX_ROWS


def test_fixed_slices_are_gb_and_ie_only():
    assert FIXED_SLICES == (("gleif", "gb"), ("gleif", "ie"))


def test_fr_is_excluded_not_merely_absent():
    assert ("gleif", "fr") in EXCLUDED_SLICES
    assert ("gleif", "fr") not in FIXED_SLICES


def test_fixed_and_excluded_slices_are_disjoint():
    assert set(FIXED_SLICES).isdisjoint(EXCLUDED_SLICES)


def test_dense_vocabulary_max_rows_matches_the_gate_default():
    assert DENSE_VOCABULARY_MAX_ROWS == DEFAULT_DENSE_VOCABULARY_MAX_ROWS


def test_runtime_budget_has_headroom_above_the_known_good_run_and_margin_below_the_pathological_one():
    # gb's largest known-good accelerated run measured 203s; the scaling table's
    # unaccelerated pathological case measured 10,986s. The budget sits
    # strictly between the two.
    assert 203 < RUNTIME_BUDGET_SECONDS < 10_986


def test_cell_statuses_cover_completed_timeout_and_error():
    assert CELL_STATUSES == ("completed", "timeout", "error")


def test_mandatory_suite_pairs_each_classical_representation_with_kmeans_and_lsh():
    # kmeans is the only partition backend that accepts a sparse
    # matrix -- hdbscan requires a dense one and so never applies to the
    # three classical (sparse) representations.
    for representation in ("tfidf", "wordpiece", "sentencepiece"):
        assert BaselineSuiteCell(representation, "kmeans") in MANDATORY_SUITE
        assert BaselineSuiteCell(representation, "lsh") in MANDATORY_SUITE
        assert BaselineSuiteCell(representation, "hdbscan") not in MANDATORY_SUITE


def test_mandatory_suite_pairs_sbert_with_hnsw_only():
    sbert_cells = [cell for cell in MANDATORY_SUITE if cell.representation == "sbert"]
    assert sbert_cells == [BaselineSuiteCell("sbert", "hnsw")]


def test_reference_suite_is_one_exact_cell_per_representation():
    assert REFERENCE_SUITE == (
        BaselineSuiteCell("tfidf", "sklearn"),
        BaselineSuiteCell("wordpiece", "sklearn"),
        BaselineSuiteCell("sentencepiece", "sklearn"),
        BaselineSuiteCell("sbert", "dense_brute"),
    )


def test_mandatory_and_reference_suites_are_disjoint():
    assert set(MANDATORY_SUITE).isdisjoint(REFERENCE_SUITE)


def test_gated_reference_cells_are_wordpiece_and_sentencepiece_on_sklearn():
    assert GATED_REFERENCE_CELLS == {
        BaselineSuiteCell("wordpiece", "sklearn"),
        BaselineSuiteCell("sentencepiece", "sklearn"),
    }
    # tfidf's sparse vocabulary is exactly what the gate does not apply to
    # (dense_vocabulary_gate.py), and sbert's reference cell is dense_brute,
    # not one of the gate's exhaustive sparse backends.
    assert BaselineSuiteCell("tfidf", "sklearn") not in GATED_REFERENCE_CELLS
    assert BaselineSuiteCell("sbert", "dense_brute") not in GATED_REFERENCE_CELLS


def _comparison_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "representation": pl.Utf8,
        "similarity_backend": pl.Utf8,
        "country": pl.Utf8,
        "stage": pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _completed_row(representation: str, similarity_backend: str, country: str) -> dict:
    return {
        "representation": representation,
        "similarity_backend": similarity_backend,
        "country": country,
        "stage": "pruned",
    }


def _complete_comparison(countries: list[str]) -> pl.DataFrame:
    rows = [
        _completed_row(cell.representation, cell.similarity_backend, country)
        for cell in (*MANDATORY_SUITE, *REFERENCE_SUITE)
        for country in countries
    ]
    return _comparison_frame(rows)


def test_missing_baseline_suite_cells_is_empty_on_both_sets_when_every_cell_completed():
    comparison = _complete_comparison(["gb", "ie"])

    status = missing_baseline_suite_cells(comparison, countries=["gb", "ie"])

    assert status == BaselineSuiteStatus(
        missing_mandatory_cells=(),
        missing_reference_cells=(),
        refused_reference_cells=(),
    )


def test_missing_baseline_suite_cells_reports_the_uncompleted_mandatory_combination():
    rows = [
        _completed_row("tfidf", "kmeans", "gb"),
        _completed_row("wordpiece", "kmeans", "gb"),
        # tfidf/lsh, wordpiece/lsh, sentencepiece/{kmeans,lsh} and sbert/hnsw
        # never completed.
    ]
    comparison = _comparison_frame(rows)

    status = missing_baseline_suite_cells(comparison, countries=["gb"])

    assert ("tfidf", "lsh", "gb") in status.missing_mandatory_cells
    assert ("tfidf", "kmeans", "gb") not in status.missing_mandatory_cells
    assert ("sbert", "hnsw", "gb") in status.missing_mandatory_cells


def test_missing_baseline_suite_cells_ignores_a_raw_stage_only_row():
    # A "raw" row (the unpruned scored candidates) is not a
    # completed cell for baseline-suite purposes -- only "pruned" counts,
    # matching build_strategy_comparison()'s own pruned/raw pairing.
    comparison = _comparison_frame(
        [
            {
                "representation": "tfidf",
                "similarity_backend": "kmeans",
                "country": "gb",
                "stage": "raw",
            }
        ]
    )

    status = missing_baseline_suite_cells(comparison, countries=["gb"])

    assert ("tfidf", "kmeans", "gb") in status.missing_mandatory_cells


def test_missing_baseline_suite_cells_on_an_empty_frame_reports_every_cell():
    comparison = _comparison_frame([])

    status = missing_baseline_suite_cells(comparison, countries=["gb"])

    assert len(status.missing_mandatory_cells) == len(MANDATORY_SUITE)
    assert len(status.missing_reference_cells) == len(REFERENCE_SUITE)


def test_missing_baseline_suite_cells_reports_reference_cells_separately_from_mandatory():
    # The mandatory suite completes in full; the reference column never ran
    # at all -- a slice can be complete on its mandatory cells with the
    # reference column entirely missing (module docstring).
    rows = [
        _completed_row(cell.representation, cell.similarity_backend, "gb")
        for cell in MANDATORY_SUITE
    ]
    comparison = _comparison_frame(rows)

    status = missing_baseline_suite_cells(comparison, countries=["gb"])

    assert status.missing_mandatory_cells == ()
    assert len(status.missing_reference_cells) == len(REFERENCE_SUITE)


def test_a_refused_reference_cell_is_reported_as_refused_not_missing():
    # Every mandatory cell completes; the wordpiece reference cell is known
    # (by the caller, from the dense-vocabulary gate) to have been refused
    # at gb's scale rather than simply not run.
    rows = [
        _completed_row(cell.representation, cell.similarity_backend, "gb")
        for cell in MANDATORY_SUITE
    ] + [
        _completed_row("tfidf", "sklearn", "gb"),
        _completed_row("sentencepiece", "sklearn", "gb"),
        _completed_row("sbert", "dense_brute", "gb"),
        # wordpiece/sklearn is refused, not completed.
    ]
    comparison = _comparison_frame(rows)

    status = missing_baseline_suite_cells(
        comparison,
        countries=["gb"],
        refused_reference_cells=[("wordpiece", "sklearn", "gb")],
    )

    assert status.missing_mandatory_cells == ()
    assert status.missing_reference_cells == ()
    assert status.refused_reference_cells == (("wordpiece", "sklearn", "gb"),)


def test_a_refused_triple_outside_the_reference_suite_is_ignored():
    # A caller-supplied triple that is not actually one of REFERENCE_SUITE's
    # own (representation, backend) pairs for a swept country is dropped
    # rather than fabricating a refusal the suite doesn't define.
    comparison = _comparison_frame([])

    status = missing_baseline_suite_cells(
        comparison,
        countries=["gb"],
        refused_reference_cells=[
            ("tfidf", "kmeans", "gb"),
            ("wordpiece", "sklearn", "fr"),
        ],
    )

    assert status.refused_reference_cells == ()


def test_require_baseline_suite_complete_raises_on_a_missing_mandatory_cell():
    comparison = _comparison_frame([_completed_row("tfidf", "kmeans", "gb")])

    with pytest.raises(BaselineSuiteIncompleteError) as excinfo:
        require_baseline_suite_complete(comparison, countries=["gb"])

    assert ("tfidf", "lsh", "gb") in excinfo.value.missing_cells
    assert ("sbert", "hnsw", "gb") in excinfo.value.missing_cells


def test_require_baseline_suite_complete_is_silent_when_mandatory_cells_complete():
    rows = [
        _completed_row(cell.representation, cell.similarity_backend, "gb")
        for cell in MANDATORY_SUITE
    ]
    comparison = _comparison_frame(rows)

    require_baseline_suite_complete(comparison, countries=["gb"])


def test_require_baseline_suite_complete_is_silent_even_with_the_reference_column_entirely_missing():
    # The reference column being unrun (or refused) never gates a promotion
    # decision -- only the mandatory suite does (module docstring).
    rows = [
        _completed_row(cell.representation, cell.similarity_backend, "gb")
        for cell in MANDATORY_SUITE
    ]
    comparison = _comparison_frame(rows)

    require_baseline_suite_complete(
        comparison,
        countries=["gb"],
        refused_reference_cells=[("wordpiece", "sklearn", "gb")],
    )
