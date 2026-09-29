from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from blocking.comparison import (
    StrategyRunEntry,
    build_pair_outcome_candidates,
    build_pair_outcome_recall_curves,
    build_pair_outcome_runs,
    build_pair_outcome_source_candidates,
    build_pair_outcomes,
)
from blocking.contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingRunResult,
    BlockingStrategyConfig,
    validate_blocking_artifact_schema,
)
from validation.runner import PAIR_TRUTH_EVAL_DETAIL_COLUMNS
from workspace.roots import default_workspace_roots

_NO_FRAME = pl.DataFrame()

_ROOTS = default_workspace_roots(Path("/root"))


def _descriptor(system: str) -> BlockingDatasetDescriptor:
    return BlockingDatasetDescriptor(
        system=system,
        system_dir=Path(system),
        layer="matched" if system == "gleif" else "cleansed",
        has_ground_truth=system == "gleif",
        matched_target_systems=("gb",) if system == "gleif" else (),
        available_countries=("gb",),
    )


def _entry(
    label: str,
    detail_rows: list[dict[str, object]] | None,
    *,
    representation: str = "tfidf",
    backend_options: dict[str, object] | None = None,
    candidates: list[tuple[float, int]] | None = None,
    sourced_candidates: list[tuple[str, float, int]] | None = None,
) -> StrategyRunEntry:
    """A completed `gleif -> gb` run whose only content is its per-pair
    detail, the one frame the outcomes read, and its candidates, as
    `(similarity, rank)` or, per source, `(source_id, similarity, rank)`."""
    config = BlockingRunConfig(
        roots=_ROOTS,
        prepared_base_dir=None,
        source=_descriptor("gleif"),
        target=_descriptor("gb"),
        countries=("gb",),
        strategy=BlockingStrategyConfig(
            representation=representation,
            top_k=2,
            min_similarity=0.2,
            max_candidates_per_source=None,
            similarity_backend="kmeans",
            backend_options=backend_options,
        ),
    )
    detail = (
        None
        if detail_rows is None
        else pl.DataFrame(detail_rows, schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS)
    )
    edges = (
        _NO_FRAME
        if candidates is None
        else pl.DataFrame(
            candidates,
            schema={"similarity": pl.Float64, "rank": pl.Int64},
            orient="row",
        )
    )
    if sourced_candidates is not None:
        edges = pl.DataFrame(
            sourced_candidates,
            schema={"source_id": pl.Utf8, "similarity": pl.Float64, "rank": pl.Int64},
            orient="row",
        ).with_columns(pl.lit("gb").alias("country"))
    result = BlockingRunResult(
        matched_edges=edges,
        raw_matched_edges=_NO_FRAME,
        pair_truth_eval=None,
        raw_pair_truth_eval=None,
        pair_truth_eval_detail=detail,
        pruning_summary=_NO_FRAME,
        exact_match_summary=_NO_FRAME,
        clusters=_NO_FRAME,
        cluster_shape=_NO_FRAME,
        directional_coverage=_NO_FRAME,
        similarity_distribution=_NO_FRAME,
        candidate_pair_count=edges.height,
    )
    return StrategyRunEntry(label=label, config=config, result=result)


def _pair(
    number: int,
    *,
    level: str = "never",
    found: bool | None,
    rank: int | None = None,
    is_truth_pair: bool = True,
) -> dict[str, object]:
    return {
        "source_system": "gleif",
        "target_system": "gb",
        "country": "gb",
        "source_id": f"gleif:{number}",
        "target_id": f"gb:{number}",
        "source_name": f"source {number}",
        "source_name_cleansed": f"source {number}",
        "target_name": f"target {number}",
        "target_name_cleansed": f"target {number}",
        "name_equality": level,
        "is_truth_pair": is_truth_pair,
        "found": found,
        "similarity": None if rank is None else 0.5,
        "rank": rank,
    }


def test_each_run_gives_every_truth_pair_one_row_with_its_own_verdict() -> None:
    entries = [
        _entry("tfidf/a", [_pair(1, found=True, rank=1), _pair(2, found=False)]),
        _entry("wordpiece/b", [_pair(1, found=False), _pair(2, found=True, rank=3)]),
    ]

    outcomes = build_pair_outcomes(entries)

    validate_blocking_artifact_schema(outcomes, artifact_name="pair_outcomes")
    assert outcomes.select("label", "source_id", "found", "rank").rows() == [
        ("tfidf/a", "gleif:1", True, 1),
        ("tfidf/a", "gleif:2", False, None),
        ("wordpiece/b", "gleif:1", False, None),
        ("wordpiece/b", "gleif:2", True, 3),
    ]


def test_every_level_is_kept_and_a_predicted_untrue_pair_is_left_out() -> None:
    entries = [
        _entry(
            "tfidf/a",
            [
                _pair(1, level="raw", found=True, rank=1),
                _pair(2, level="never", found=False),
                _pair(3, level="never", found=None, is_truth_pair=False),
            ],
        )
    ]

    outcomes = build_pair_outcomes(entries)

    assert outcomes.select("source_id", "name_equality").rows() == [
        ("gleif:1", "raw"),
        ("gleif:2", "never"),
    ]


def test_the_runs_table_holds_one_row_per_run_with_its_settings() -> None:
    entries = [
        _entry(
            "wordpiece/b",
            [_pair(1, found=True, rank=1)],
            representation="wordpiece",
            backend_options={"kmeans_clusters": 300},
        ),
        _entry("tfidf/a", [_pair(1, found=False), _pair(2, found=False)]),
    ]

    runs = build_pair_outcome_runs(entries)

    validate_blocking_artifact_schema(runs, artifact_name="pair_outcome_runs")
    assert runs.select(
        "label", "representation", "accelerator_settings", "top_k", "truth_pairs"
    ).rows() == [
        ("tfidf/a", "tfidf", "exact_name_filter=true", 2, 2),
        (
            "wordpiece/b",
            "wordpiece",
            "exact_name_filter=true;kmeans_clusters=300",
            2,
            1,
        ),
    ]


def test_a_run_without_a_detail_frame_is_listed_and_has_no_outcomes() -> None:
    entries = [_entry("tfidf/a", None), _entry("tfidf/b", [_pair(1, found=True)])]

    runs = build_pair_outcome_runs(entries)
    outcomes = build_pair_outcomes(entries)

    assert runs.select("label", "truth_pairs").rows() == [
        ("tfidf/a", None),
        ("tfidf/b", 1),
    ]
    assert outcomes.get_column("label").unique().to_list() == ["tfidf/b"]


def test_no_run_with_a_detail_frame_gives_an_empty_outcomes_table() -> None:
    outcomes = build_pair_outcomes([_entry("tfidf/a", None)])

    validate_blocking_artifact_schema(outcomes, artifact_name="pair_outcomes")
    assert outcomes.height == 0


def test_two_runs_under_one_label_are_refused() -> None:
    entries = [
        _entry("tfidf/a", [_pair(1, found=True)]),
        _entry("tfidf/a", [_pair(1, found=False)]),
    ]

    with pytest.raises(ValueError, match="tfidf/a"):
        build_pair_outcomes(entries)
    with pytest.raises(ValueError, match="tfidf/a"):
        build_pair_outcome_runs(entries)


def test_the_runs_table_carries_when_each_run_finished() -> None:
    earlier = replace(
        _entry("tfidf/a", [_pair(1, found=True)]),
        finished_at="2026-09-19T10:00:00+00:00",
    )
    made_here = _entry("tfidf/b", [_pair(1, found=True)])

    runs = build_pair_outcome_runs([earlier, made_here])

    assert runs.select("label", "finished_at").rows() == [
        ("tfidf/a", "2026-09-19T10:00:00+00:00"),
        ("tfidf/b", None),
    ]


def test_candidates_are_counted_by_the_thousandth_and_rank_they_sit_at() -> None:
    entries = [
        _entry(
            "tfidf/a",
            None,
            candidates=[(0.7504, 2), (0.7509, 2), (0.9, 1), (1.0, 1), (1.0, 1)],
        ),
        _entry("tfidf/none", None),
    ]

    candidates = build_pair_outcome_candidates(entries)

    validate_blocking_artifact_schema(
        candidates, artifact_name="pair_outcome_candidates"
    )
    assert candidates.rows() == [
        ("tfidf/a", 0.75, 2, 2),
        ("tfidf/a", 0.9, 1, 1),
        ("tfidf/a", 1.0, 1, 2),
    ]
    runs = build_pair_outcome_runs(entries)
    assert runs.select("label", "candidate_pairs").rows() == [
        ("tfidf/a", 5),
        ("tfidf/none", 0),
    ]


def test_source_candidates_keep_only_sources_holding_a_truth_pair() -> None:
    entries = [
        _entry(
            "tfidf/a",
            [_pair(1, found=True, rank=1), _pair(2, found=False, is_truth_pair=False)],
            sourced_candidates=[
                ("gleif:1", 0.9, 1),
                ("gleif:1", 0.9004, 2),
                ("gleif:1", 0.8, 3),
                ("gleif:2", 0.95, 1),
                ("gleif:9", 0.99, 1),
            ],
        ),
        _entry("tfidf/none", None),
    ]

    source_candidates = build_pair_outcome_source_candidates(entries)

    validate_blocking_artifact_schema(
        source_candidates, artifact_name="pair_outcome_source_candidates"
    )
    assert source_candidates.rows() == [
        ("tfidf/a", "gb", "gleif:1", 0.8, 3, 1),
        ("tfidf/a", "gb", "gleif:1", 0.9, 1, 1),
        ("tfidf/a", "gb", "gleif:1", 0.9, 2, 1),
    ]


def test_a_run_records_the_target_rows_it_scanned_per_country_summed() -> None:
    entry = _entry("tfidf/a", [_pair(1, found=True, rank=1)])
    evaluation = pl.DataFrame(
        {"country": ["gb", "gb", "ie"], "target_rows": [100, 100, 40]}
    )
    entry = replace(entry, result=replace(entry.result, pair_truth_eval=evaluation))

    runs = build_pair_outcome_runs([entry, _entry("tfidf/none", None)])

    assert runs.select("label", "target_rows").rows() == [
        ("tfidf/a", 140),
        ("tfidf/none", None),
    ]


def test_an_outcome_carries_the_cleansed_names_its_level_was_read_from() -> None:
    outcomes = build_pair_outcomes([_entry("tfidf/a", [_pair(1, found=True)])])

    assert outcomes.select("source_name_cleansed", "target_name_cleansed").rows() == [
        ("source 1", "target 1")
    ]


def test_each_runs_recall_curve_is_stacked_under_its_label() -> None:
    entries = [
        _entry(
            "tfidf/a",
            [_pair(1, found=True, rank=1), _pair(2, found=False)],
            sourced_candidates=[
                ("gleif:1", 0.5, 1),
                ("gleif:1", 0.4, 2),
                ("gleif:2", 0.3, 1),
            ],
        ),
        _entry("tfidf/none", None),
    ]

    curves = build_pair_outcome_recall_curves(entries)

    validate_blocking_artifact_schema(
        curves, artifact_name="pair_outcome_recall_curves"
    )
    assert curves.get_column("label").unique().to_list() == ["tfidf/a"]
    assert curves.filter(pl.col("population") == "never").select(
        "budget_step", "comparisons_spent", "pairs_found"
    ).rows() == [(0, 0, 0), (1, 2, 1), (2, 3, 1)]
