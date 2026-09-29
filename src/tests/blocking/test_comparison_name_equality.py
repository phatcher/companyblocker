from pathlib import Path

import polars as pl

from blocking.comparison import (
    NEVER_RECOVERED_BY_CLEANSE,
    NEVER_SOLVED_BY_ALGORITHM,
    StrategyRunEntry,
    diff_name_equality_levels,
    tally_name_equality_transitions,
)
from blocking.contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingRunResult,
    BlockingStrategyConfig,
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
    cleanse_profile: str, detail_rows: list[dict[str, object]] | None
) -> StrategyRunEntry:
    """A completed tfidf `gleif -> gb` run under `cleanse_profile` whose only
    content is its per-pair detail, the one frame the attribution reads."""
    config = BlockingRunConfig(
        roots=_ROOTS,
        prepared_base_dir=None,
        source=_descriptor("gleif"),
        target=_descriptor("gb"),
        countries=("gb",),
        strategy=BlockingStrategyConfig(
            representation="tfidf",
            top_k=2,
            min_similarity=0.2,
            max_candidates_per_source=None,
            cleanse_profile=cleanse_profile,
        ),
    )
    detail = (
        None
        if detail_rows is None
        else pl.DataFrame(detail_rows, schema=PAIR_TRUTH_EVAL_DETAIL_COLUMNS)
    )
    result = BlockingRunResult(
        matched_edges=_NO_FRAME,
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
        candidate_pair_count=0,
    )
    return StrategyRunEntry(label="tfidf", config=config, result=result)


def _pair(
    number: int, *, level: str, found: bool | None, is_truth_pair: bool = True
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
        "similarity": None,
        "rank": None,
    }


def test_a_pair_left_in_never_that_its_run_found_is_solved_by_the_algorithm() -> None:
    baseline = _entry("default", [_pair(1, level="cleansed", found=True)])
    variant = _entry("default|-lowercase", [_pair(1, level="never", found=True)])

    attribution = diff_name_equality_levels(baseline, variant)

    assert attribution.select(
        "source_id",
        "baseline_name_transform",
        "variant_name_transform",
        "baseline_name_equality",
        "variant_name_equality",
        "variant_found",
        "never_attribution",
    ).rows() == [
        (
            "gleif:1",
            "cleanse:default;preprocess:default",
            "cleanse:default|-lowercase;preprocess:default",
            "cleansed",
            "never",
            True,
            NEVER_SOLVED_BY_ALGORITHM,
        )
    ]


def test_a_pair_left_in_never_that_its_run_missed_is_recovered_by_cleanse() -> None:
    baseline = _entry("default|-lowercase", [_pair(1, level="never", found=False)])
    variant = _entry("default", [_pair(1, level="cleansed", found=True)])

    attribution = diff_name_equality_levels(baseline, variant)

    assert attribution.select(
        "baseline_name_equality", "baseline_found", "never_attribution"
    ).rows() == [("never", False, NEVER_RECOVERED_BY_CLEANSE)]


def test_only_truth_pairs_whose_level_moved_are_attributed() -> None:
    baseline = _entry(
        "default",
        [
            _pair(1, level="raw", found=True),
            _pair(2, level="basic", found=True),
            _pair(3, level="cleansed", found=None, is_truth_pair=False),
        ],
    )
    variant = _entry(
        "default|-lowercase",
        [
            _pair(1, level="raw", found=True),
            _pair(2, level="cleansed", found=True),
            _pair(3, level="never", found=None, is_truth_pair=False),
        ],
    )

    attribution = diff_name_equality_levels(baseline, variant)

    # A move between levels other than `never` carries no never attribution.
    assert attribution.select("source_id", "never_attribution").rows() == [
        ("gleif:2", None)
    ]


def test_a_run_without_a_detail_frame_attributes_nothing() -> None:
    baseline = _entry("default", None)
    variant = _entry("default|-lowercase", [_pair(1, level="never", found=True)])

    assert diff_name_equality_levels(baseline, variant).height == 0


def test_the_tally_counts_pairs_by_transition_and_never_attribution() -> None:
    baseline = _entry(
        "default",
        [
            _pair(1, level="cleansed", found=True),
            _pair(2, level="cleansed", found=True),
            _pair(3, level="cleansed", found=True),
        ],
    )
    variant = _entry(
        "default|-lowercase",
        [
            _pair(1, level="never", found=True),
            _pair(2, level="never", found=True),
            _pair(3, level="never", found=False),
        ],
    )

    tally = tally_name_equality_transitions(
        diff_name_equality_levels(baseline, variant)
    )

    assert tally.select(
        "country",
        "baseline_name_equality",
        "variant_name_equality",
        "never_attribution",
        "truth_pairs",
    ).rows() == [
        ("gb", "cleansed", "never", NEVER_RECOVERED_BY_CLEANSE, 1),
        ("gb", "cleansed", "never", NEVER_SOLVED_BY_ALGORITHM, 2),
    ]
