"""Direct tests for `validation.recall_curve`: the
recall-against-comparisons-spent curve and its normalised area, read from a
completed run's `matched_edges`/`pair_truth_eval_detail` without re-running
the run."""

import polars as pl

from validation.recall_curve import (
    RECALL_CURVE_COLUMNS,
    compute_recall_curve,
    rank_by_recall_area,
    recall_curve_area,
    summarize_recall_curve,
)
from validation.runner import PAIR_TRUTH_EVAL_DETAIL_COLUMNS


def _matched_edges(rows: list[tuple[str, str, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_id": [row[0] for row in rows],
            "target_id": [row[1] for row in rows],
            "similarity": [row[2] for row in rows],
            "rank": [row[3] for row in rows],
        },
        schema={
            "source_id": pl.Utf8,
            "target_id": pl.Utf8,
            "similarity": pl.Float64,
            "rank": pl.Int64,
        },
    )


def _truth_detail_row(
    *,
    source_id: str,
    target_id: str,
    name_equality: str,
    found: bool | None,
    similarity: float | None,
    rank: int | None,
    is_truth_pair: bool = True,
) -> dict[str, object]:
    return {
        "source_system": "gleif",
        "target_system": "ie",
        "country": "ie",
        "source_id": source_id,
        "target_id": target_id,
        "source_name": source_id,
        "source_name_cleansed": source_id,
        "target_name": target_id,
        "target_name_cleansed": target_id,
        "name_equality": name_equality,
        "is_truth_pair": is_truth_pair,
        "found": found,
        "similarity": similarity,
        "rank": rank,
    }


def _pair_truth_eval_detail(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS)
    return pl.DataFrame(rows, schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS)


def _curve_row(curve: pl.DataFrame, *, population: str, budget_step: int) -> dict:
    matching = curve.filter(
        (pl.col("population") == population) & (pl.col("budget_step") == budget_step)
    )
    assert matching.height == 1, (population, budget_step, curve)
    return matching.row(0, named=True)


class TestComputeRecallCurve:
    """s1 is found for free (raw), s2 is found at a cost (cleansed), s3 is
    never found (never, with candidate-set noise), s4 is found at a cost
    (never) -- covers every attribution rule `compute_recall_curve`'s
    docstring states: a wrong candidate spends its own source's budget, a
    free match counts from budget_step 0, and a source's level comes from
    its own truth pair, not a candidate row's own name comparison."""

    def _build(self) -> pl.DataFrame:
        matched_edges = _matched_edges(
            [
                ("s1", "t1", 1.0, 1),
                ("s2", "tw2", 0.95, 1),
                ("s2", "t2", 0.9, 2),
                ("s3", "tw3a", 0.8, 1),
                ("s3", "tw3b", 0.7, 2),
                ("s4", "t4", 0.99, 1),
            ]
        )
        detail = _pair_truth_eval_detail(
            [
                _truth_detail_row(
                    source_id="s1",
                    target_id="t1",
                    name_equality="raw",
                    found=True,
                    similarity=1.0,
                    rank=1,
                ),
                _truth_detail_row(
                    source_id="s2",
                    target_id="t2",
                    name_equality="cleansed",
                    found=True,
                    similarity=0.9,
                    rank=2,
                ),
                _truth_detail_row(
                    source_id="s2",
                    target_id="tw2",
                    name_equality="never",
                    found=None,
                    similarity=0.95,
                    rank=1,
                    is_truth_pair=False,
                ),
                _truth_detail_row(
                    source_id="s3",
                    target_id="t3",
                    name_equality="never",
                    found=False,
                    similarity=None,
                    rank=None,
                ),
                _truth_detail_row(
                    source_id="s4",
                    target_id="t4",
                    name_equality="never",
                    found=True,
                    similarity=0.99,
                    rank=1,
                ),
            ]
        )
        return compute_recall_curve(
            matched_edges=matched_edges, pair_truth_eval_detail=detail
        )

    def test_schema(self) -> None:
        curve = self._build()
        assert curve.schema == pl.Schema(RECALL_CURVE_COLUMNS)

    def test_raw_level_is_free_from_budget_zero(self) -> None:
        curve = self._build()
        row = _curve_row(curve, population="raw", budget_step=0)
        assert row["comparisons_spent"] == 0
        assert row["truth_pairs"] == 1
        assert row["pairs_found"] == 1
        assert row["recall"] == 1.0
        # No costed candidate for this population at all: a single row.
        assert curve.filter(pl.col("population") == "raw").height == 1

    def test_basic_level_has_no_sources(self) -> None:
        curve = self._build()
        row = _curve_row(curve, population="basic", budget_step=0)
        assert row["truth_pairs"] == 0
        assert row["pairs_found"] == 0
        assert row["recall"] is None

    def test_cleansed_level_climbs_once_its_own_cost_is_paid(self) -> None:
        curve = self._build()
        base = _curve_row(curve, population="cleansed", budget_step=0)
        assert (base["comparisons_spent"], base["recall"]) == (0, 0.0)
        step1 = _curve_row(curve, population="cleansed", budget_step=1)
        assert (step1["comparisons_spent"], step1["recall"]) == (1, 0.0)
        step2 = _curve_row(curve, population="cleansed", budget_step=2)
        assert (step2["comparisons_spent"], step2["recall"]) == (2, 1.0)

    def test_never_level_counts_every_source_wrong_candidates_too(self) -> None:
        curve = self._build()
        # s3's two wrong candidates and s4's one right one all spend the
        # `never` population's budget together.
        base = _curve_row(curve, population="never", budget_step=0)
        assert (base["truth_pairs"], base["comparisons_spent"], base["recall"]) == (
            2,
            0,
            0.0,
        )
        step1 = _curve_row(curve, population="never", budget_step=1)
        assert (step1["comparisons_spent"], step1["recall"]) == (2, 0.5)
        step2 = _curve_row(curve, population="never", budget_step=2)
        assert (step2["comparisons_spent"], step2["recall"]) == (3, 0.5)

    def test_universe_rolls_up_every_level(self) -> None:
        curve = self._build()
        base = _curve_row(curve, population="universe", budget_step=0)
        assert (base["truth_pairs"], base["pairs_found"], base["recall"]) == (
            4,
            1,
            0.25,
        )
        step1 = _curve_row(curve, population="universe", budget_step=1)
        assert (step1["comparisons_spent"], step1["pairs_found"]) == (3, 2)
        step2 = _curve_row(curve, population="universe", budget_step=2)
        assert (step2["comparisons_spent"], step2["pairs_found"]) == (5, 3)


class TestDegenerateCases:
    def test_empty_truth_population(self) -> None:
        matched_edges = _matched_edges([])
        detail = _pair_truth_eval_detail([])
        curve = compute_recall_curve(
            matched_edges=matched_edges, pair_truth_eval_detail=detail
        )
        for population in ("universe", "raw", "basic", "cleansed", "never"):
            row = _curve_row(curve, population=population, budget_step=0)
            assert row["truth_pairs"] == 0
            assert row["recall"] is None
            assert recall_curve_area(curve, population=population) is None

    def test_every_truth_pair_retrieved_before_any_non_truth_pair(self) -> None:
        # Both sources are resolved before a single wrong candidate is ever
        # scored: s1 for free, s2 at its own first (and only) candidate.
        matched_edges = _matched_edges(
            [
                ("s1", "t1", 1.0, 1),
                ("s2", "t2", 0.88, 1),
            ]
        )
        detail = _pair_truth_eval_detail(
            [
                _truth_detail_row(
                    source_id="s1",
                    target_id="t1",
                    name_equality="never",
                    found=True,
                    similarity=1.0,
                    rank=1,
                ),
                _truth_detail_row(
                    source_id="s2",
                    target_id="t2",
                    name_equality="never",
                    found=True,
                    similarity=0.88,
                    rank=1,
                ),
            ]
        )
        curve = compute_recall_curve(
            matched_edges=matched_edges, pair_truth_eval_detail=detail
        )
        never = curve.filter(pl.col("population") == "never").sort("comparisons_spent")
        assert never["recall"].to_list() == [0.5, 1.0]
        assert never["comparisons_spent"].to_list() == [0, 1]
        # The area is still well-defined rather than raising or dividing by
        # zero: over its own reached budget of 1, recall held at 0.5 (the
        # free find alone) for the whole unit spent reaching the second one.
        assert recall_curve_area(curve, population="never") == 0.5


class TestRecallCurveArea:
    def _curve(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "population": ["never", "never", "never"],
                "budget_step": [0, 1, 2],
                "comparisons_spent": [0, 2, 3],
                "truth_pairs": [2, 2, 2],
                "pairs_found": [0, 1, 1],
                "recall": [0.0, 0.5, 0.5],
            },
            schema=RECALL_CURVE_COLUMNS,
        )

    def test_full_budget(self) -> None:
        # area = 0.0*2 + 0.5*1 = 0.5, over its own full budget of 3.
        assert recall_curve_area(self._curve(), population="never") == 0.5 / 3

    def test_truncated_budget(self) -> None:
        # Read only as far as comparisons_spent == 2: recall never leaves
        # 0.0 within that window.
        assert (
            recall_curve_area(self._curve(), population="never", max_comparisons=2)
            == 0.0
        )

    def test_unknown_population_is_none(self) -> None:
        assert recall_curve_area(self._curve(), population="raw") is None


class TestSummarizeRecallCurve:
    def test_headline_is_never(self) -> None:
        curve = pl.DataFrame(
            {
                "population": ["raw", "never"],
                "budget_step": [0, 0],
                "comparisons_spent": [0, 0],
                "truth_pairs": [1, 1],
                "pairs_found": [1, 1],
                "recall": [1.0, 1.0],
            },
            schema=RECALL_CURVE_COLUMNS,
        )
        summary = summarize_recall_curve(
            curve,
            similarity_backend="sklearn",
            is_exact_backend=True,
            target_rows=1000,
            min_similarity=0.75,
            top_k=5,
            max_candidates_per_source=50,
        )
        row = summary.row(0, named=True)
        assert row["recall_area_never"] == 1.0
        assert row["recall_area_raw"] == 1.0
        assert row["recall_area_basic"] is None
        assert row["similarity_backend"] == "sklearn"
        assert row["is_exact_backend"] is True
        assert row["target_rows"] == 1000
        assert row["min_similarity"] == 0.75
        assert row["top_k"] == 5
        assert row["max_candidates_per_source"] == 50

    def test_scale_fields_are_optional(self) -> None:
        curve = pl.DataFrame(
            {
                "population": ["never"],
                "budget_step": [0],
                "comparisons_spent": [0],
                "truth_pairs": [1],
                "pairs_found": [1],
                "recall": [1.0],
            },
            schema=RECALL_CURVE_COLUMNS,
        )
        summary = summarize_recall_curve(
            curve,
            similarity_backend="sklearn",
            is_exact_backend=True,
            target_rows=None,
        )
        row = summary.row(0, named=True)
        assert row["min_similarity"] is None
        assert row["top_k"] is None
        assert row["max_candidates_per_source"] is None


class TestRankByRecallArea:
    """Two runs of one pairing under different representations, ranked on
    `never` at the budget both reached; the ordering must survive a
    smaller-top_k re-read, the run re-read at fewer budget steps than it
    reached the first time."""

    def _full_curves(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        tfidf = pl.DataFrame(
            {
                "population": ["never"] * 3,
                "budget_step": [0, 1, 2],
                "comparisons_spent": [0, 3, 5],
                "truth_pairs": [2, 2, 2],
                "pairs_found": [1, 2, 2],
                "recall": [0.5, 1.0, 1.0],
            },
            schema=RECALL_CURVE_COLUMNS,
        )
        sentencepiece = pl.DataFrame(
            {
                "population": ["never"] * 3,
                "budget_step": [0, 1, 2],
                "comparisons_spent": [0, 3, 6],
                "truth_pairs": [2, 2, 2],
                "pairs_found": [0, 1, 2],
                "recall": [0.0, 0.5, 1.0],
            },
            schema=RECALL_CURVE_COLUMNS,
        )
        return tfidf, sentencepiece

    def test_ranking_at_full_budget(self) -> None:
        tfidf, sentencepiece = self._full_curves()
        ranking = rank_by_recall_area(
            [("tfidf", "sklearn", tfidf), ("sentencepiece", "sklearn", sentencepiece)],
            population="never",
        )
        assert ranking["label"].to_list() == ["tfidf", "sentencepiece"]
        assert ranking["shared_budget"].to_list() == [5, 5]
        area_row = ranking.row(0, named=True)
        assert area_row["recall_area"] == 3.5 / 5

    def test_ranking_unchanged_when_reread_at_smaller_top_k(self) -> None:
        tfidf, sentencepiece = self._full_curves()
        smaller_tfidf = tfidf.filter(pl.col("budget_step") <= 1)
        smaller_sentencepiece = sentencepiece.filter(pl.col("budget_step") <= 1)

        ranking = rank_by_recall_area(
            [
                ("tfidf", "sklearn", smaller_tfidf),
                ("sentencepiece", "sklearn", smaller_sentencepiece),
            ],
            population="never",
        )
        assert ranking["label"].to_list() == ["tfidf", "sentencepiece"]
        assert ranking["shared_budget"].to_list() == [3, 3]
        first, second = ranking.rows(named=True)
        assert first["recall_area"] == 0.5
        assert second["recall_area"] == 0.0
        assert first["recall_area"] > second["recall_area"]
