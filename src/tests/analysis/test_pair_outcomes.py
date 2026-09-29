import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis.pair_outcomes import (
    candidates_at,
    char_ngram_jaccard,
    cluster_sizes,
    count_by_reach,
    count_found_by,
    count_found_by_k,
    cutoffs_for_candidates,
    draw_cluster_sizes,
    draw_name_similarity,
    draw_run_overlap,
    drop_unanimous_pairs,
    latest_runs,
    level_metrics,
    pair_found_matrix,
    pairs_found_by,
    random_pair_level,
    run_columns,
    run_disagreement,
    run_overlap,
    select_runs,
    with_name_similarity,
)


def _runs(*rows: tuple[str, str, str | None]) -> pl.DataFrame:
    """A runs table of `(label, representation, finished_at)` rows."""
    return pl.DataFrame(
        [
            {
                "label": label,
                "representation": representation,
                "tokenizer": "wordpiece:promoted",
                "similarity_backend": "kmeans",
                "finished_at": finished_at,
            }
            for label, representation, finished_at in rows
        ],
        schema={
            "label": pl.Utf8,
            "representation": pl.Utf8,
            "tokenizer": pl.Utf8,
            "similarity_backend": pl.Utf8,
            "finished_at": pl.Utf8,
        },
    )


def _outcome(
    label: str,
    number: int,
    *,
    found: bool,
    level: str = "never",
    country: str = "ie",
    similarity: float | None = None,
) -> dict[str, object]:
    return {
        "label": label,
        "country": country,
        "source_id": f"gleif:{number}",
        "target_id": f"{country}:{number}",
        "source_name": f"source {number}",
        "target_name": f"target {number}",
        "name_equality": level,
        "found": found,
        "similarity": similarity,
    }


_RUNS = _runs(("tfidf/a", "tfidf", None), ("wordpiece/b", "wordpiece", None))

# Pair 1 only tfidf found, pair 2 both, pair 3 neither, pair 4 is not `never`.
_OUTCOMES = pl.DataFrame(
    [
        _outcome("tfidf/a", 1, found=True),
        _outcome("wordpiece/b", 1, found=False),
        _outcome("tfidf/a", 2, found=True),
        _outcome("wordpiece/b", 2, found=True),
        _outcome("tfidf/a", 3, found=False),
        _outcome("wordpiece/b", 3, found=False),
        _outcome("tfidf/a", 4, found=True, level="raw"),
        _outcome("wordpiece/b", 4, found=True, level="raw"),
    ]
)


def test_the_latest_run_of_each_representation_is_picked_by_finish_time() -> None:
    runs = _runs(
        ("tfidf/old", "tfidf", "2026-09-19T10:00:00+00:00"),
        ("tfidf/new", "tfidf", "2026-09-20T10:00:00+00:00"),
        ("tfidf/made_here", "tfidf", None),
        ("wordpiece/only", "wordpiece", "2026-09-18T10:00:00+00:00"),
    )

    assert latest_runs(runs).get_column("label").to_list() == [
        "tfidf/new",
        "wordpiece/only",
    ]


def test_runs_are_selected_by_every_filter_given() -> None:
    assert select_runs(_RUNS, representation=["wordpiece"]).get_column(
        "label"
    ).to_list() == ["wordpiece/b"]
    assert select_runs(_RUNS).height == 2


def test_a_label_no_run_carries_is_refused() -> None:
    with pytest.raises(ValueError, match="tfidf/missing"):
        select_runs(_RUNS, labels=["tfidf/missing"])


def test_the_matrix_holds_one_row_per_never_pair_with_each_runs_verdict() -> None:
    matrix = pair_found_matrix(_OUTCOMES, _RUNS)

    assert run_columns(matrix) == ["tfidf/a", "wordpiece/b"]
    assert matrix.select("source_id", "tfidf/a", "wordpiece/b").rows() == [
        ("gleif:1", True, False),
        ("gleif:2", True, True),
        ("gleif:3", False, False),
    ]


def test_run_columns_take_the_names_given_and_a_shared_name_is_refused() -> None:
    names = {"tfidf/a": "tfidf", "wordpiece/b": "wordpiece"}

    assert run_columns(pair_found_matrix(_OUTCOMES, _RUNS, run_names=names)) == [
        "tfidf",
        "wordpiece",
    ]
    with pytest.raises(ValueError, match="share a column name"):
        pair_found_matrix(
            _OUTCOMES, _RUNS, run_names={"tfidf/a": "x", "wordpiece/b": "x"}
        )


def test_a_pair_one_run_places_at_the_level_carries_every_runs_verdict() -> None:
    outcomes = pl.DataFrame(
        [
            _outcome("tfidf/a", 1, found=False, level="never"),
            _outcome("wordpiece/b", 1, found=True, level="cleansed"),
        ]
    )

    matrix = pair_found_matrix(outcomes, _RUNS)

    assert matrix.select("tfidf/a", "wordpiece/b").rows() == [(False, True)]


def test_the_matrix_keeps_one_country_and_every_level_when_asked() -> None:
    outcomes = pl.concat(
        [_OUTCOMES, pl.DataFrame([_outcome("tfidf/a", 9, found=True, country="gb")])]
    )

    assert pair_found_matrix(outcomes, _RUNS, country="gb").height == 1
    assert pair_found_matrix(_OUTCOMES, _RUNS, name_equality=None).height == 4


def test_dropping_unanimous_pairs_leaves_what_the_runs_disagree_on() -> None:
    matrix = pair_found_matrix(_OUTCOMES, _RUNS)

    assert drop_unanimous_pairs(matrix).get_column("source_id").to_list() == [
        "gleif:1",
        "gleif:2",
    ]
    assert drop_unanimous_pairs(matrix, found_by_all=True).get_column(
        "source_id"
    ).to_list() == ["gleif:1"]


def test_pairs_are_counted_by_the_runs_that_found_them() -> None:
    counts = count_found_by(pair_found_matrix(_OUTCOMES, _RUNS))

    assert counts.select("found_by", "pairs").rows() == [
        ("(none)", 1),
        ("tfidf/a", 1),
        ("tfidf/a + wordpiece/b", 1),
    ]
    assert counts.get_column("share").sum() == pytest.approx(1.0)


def test_the_pairs_behind_one_combination_are_named() -> None:
    matrix = pair_found_matrix(_OUTCOMES, _RUNS)

    assert pairs_found_by(matrix, ["tfidf/a"]).get_column("source_name").to_list() == [
        "source 1"
    ]
    assert pairs_found_by(matrix, []).get_column("source_id").to_list() == ["gleif:3"]
    with pytest.raises(ValueError, match="sbert"):
        pairs_found_by(matrix, ["sbert"])


def test_a_run_cut_above_a_pairs_similarity_no_longer_finds_it() -> None:
    outcomes = pl.DataFrame(
        [
            _outcome("tfidf/a", 1, found=True, similarity=0.80),
            _outcome("tfidf/a", 2, found=True, similarity=0.95),
            _outcome("tfidf/a", 3, found=True, similarity=None),
            _outcome("wordpiece/b", 1, found=True, similarity=0.80),
            _outcome("wordpiece/b", 2, found=False),
            _outcome("wordpiece/b", 3, found=False),
        ],
        schema_overrides={"similarity": pl.Float64},
    )

    matrix = pair_found_matrix(outcomes, _RUNS, min_similarity={"tfidf/a": 0.9})

    # Only tfidf is cut; a found pair with no similarity has nothing to be cut by.
    assert matrix.select("source_id", "tfidf/a", "wordpiece/b").rows() == [
        ("gleif:1", False, True),
        ("gleif:2", True, False),
        ("gleif:3", True, False),
    ]
    with pytest.raises(ValueError, match="sbert/x"):
        pair_found_matrix(outcomes, _RUNS, min_similarity={"sbert/x": 0.9})


# Cells of (cutoff, rank): tfidf keeps 100 pairs as made, 40 from 0.85 and 10
# at 1.0, a fifth of each at rank one; wordpiece keeps 30, 25 of them at 1.0.
_CANDIDATES = pl.DataFrame(
    [
        {"label": "tfidf/a", "min_similarity": 0.75, "rank": 1, "candidates": 12},
        {"label": "tfidf/a", "min_similarity": 0.75, "rank": 2, "candidates": 48},
        {"label": "tfidf/a", "min_similarity": 0.85, "rank": 1, "candidates": 6},
        {"label": "tfidf/a", "min_similarity": 0.85, "rank": 2, "candidates": 24},
        {"label": "tfidf/a", "min_similarity": 1.0, "rank": 1, "candidates": 2},
        {"label": "tfidf/a", "min_similarity": 1.0, "rank": 2, "candidates": 8},
        {"label": "wordpiece/b", "min_similarity": 0.75, "rank": 1, "candidates": 5},
        {"label": "wordpiece/b", "min_similarity": 1.0, "rank": 1, "candidates": 25},
    ]
)


def test_a_budget_gives_each_run_the_lowest_cutoff_that_fits_it() -> None:
    assert cutoffs_for_candidates(_CANDIDATES, _RUNS, budget=40) == {
        "tfidf/a": 0.85,
        "wordpiece/b": 0.75,
    }
    # No cutoff brings either run under five, so each takes its highest.
    assert cutoffs_for_candidates(_CANDIDATES, _RUNS, budget=5) == {
        "tfidf/a": 1.0,
        "wordpiece/b": 1.0,
    }
    with pytest.raises(ValueError, match="at least one"):
        cutoffs_for_candidates(_CANDIDATES, _RUNS, budget=0)


def test_candidates_kept_are_read_at_the_step_at_or_above_a_cutoff() -> None:
    assert candidates_at(_CANDIDATES, {"tfidf/a": 0.8, "wordpiece/b": 0.75}) == {
        "tfidf/a": 40,
        "wordpiece/b": 30,
    }


def test_a_cap_per_source_is_part_of_what_a_run_keeps_and_of_its_budget() -> None:
    cutoffs = {"tfidf/a": 0.75, "wordpiece/b": 0.75}

    assert candidates_at(_CANDIDATES, cutoffs, max_rank=1) == {
        "tfidf/a": 20,
        "wordpiece/b": 30,
    }
    # Within rank one tfidf already fits twenty, so it is not cut at all.
    assert cutoffs_for_candidates(_CANDIDATES, _RUNS, budget=20, max_rank=1) == {
        "tfidf/a": 0.75,
        "wordpiece/b": 1.0,
    }


def test_a_pair_found_beyond_the_cap_per_source_no_longer_counts() -> None:
    outcomes = pl.DataFrame(
        [
            {**_outcome("tfidf/a", 1, found=True), "rank": 1},
            {**_outcome("tfidf/a", 2, found=True), "rank": 4},
            {**_outcome("tfidf/a", 3, found=True), "rank": None},
            {**_outcome("wordpiece/b", 1, found=False), "rank": None},
            {**_outcome("wordpiece/b", 2, found=True), "rank": 3},
            {**_outcome("wordpiece/b", 3, found=False), "rank": None},
        ],
        schema_overrides={"rank": pl.Int64, "similarity": pl.Float64},
    )

    matrix = pair_found_matrix(outcomes, _RUNS, max_rank=3)

    assert matrix.select("source_id", "tfidf/a", "wordpiece/b").rows() == [
        ("gleif:1", True, False),
        ("gleif:2", False, True),
        ("gleif:3", True, False),
    ]
    with pytest.raises(ValueError, match="at least one"):
        pair_found_matrix(outcomes, _RUNS, max_rank=0)


def test_the_matrix_takes_the_first_selected_runs_cleansed_names() -> None:
    outcomes = pl.DataFrame(
        [
            {
                **_outcome("wordpiece/b", 1, found=True),
                "source_name_cleansed": "acme",
                "target_name_cleansed": "acme ltd",
            },
            {
                **_outcome("tfidf/a", 1, found=True),
                "source_name_cleansed": "acme co",
                "target_name_cleansed": "acme ltd",
            },
        ]
    )

    matrix = pair_found_matrix(outcomes, _RUNS)

    assert matrix.select("source_name_cleansed", "target_name_cleansed").rows() == [
        ("acme co", "acme ltd")
    ]
    assert run_columns(matrix) == ["tfidf/a", "wordpiece/b"]


def test_character_trigram_jaccard_is_the_shared_share_of_padded_trigrams() -> None:
    assert char_ngram_jaccard("abc", "abc") == 1.0
    assert char_ngram_jaccard("abc", "xyz") == 0.0
    # " ab", "abc", "bc " against " ab", "abd", "bd ": one shared of five.
    assert char_ngram_jaccard("abc", "abd") == pytest.approx(1 / 5)
    assert char_ngram_jaccard("", "") == 0.0


def _named_matrix() -> pl.DataFrame:
    names = [
        ("daly foods", "daly foods ireland"),
        ("blue tree systems", "orbcomm ireland"),
        ("roap partnership", "elite drain service"),
        ("cargotec engineering", "hiab"),
    ]
    return pl.DataFrame(
        [
            {
                "country": "ie",
                "source_id": f"gleif:{number}",
                "target_id": f"ie:{number}",
                "source_name": source,
                "target_name": target,
                "tfidf/a": number == 0,
            }
            for number, (source, target) in enumerate(names)
        ]
    )


def test_the_random_pair_level_is_repeatable_and_marks_pairs_above_it() -> None:
    matrix = _named_matrix()

    level = random_pair_level(matrix, draws=5, seed=3)

    assert level == random_pair_level(matrix, draws=5, seed=3)
    assert 0.0 <= level < 1.0
    marked = with_name_similarity(matrix, random_level=0.3)
    assert marked.select("source_id", "above_random").rows() == [
        ("gleif:0", True),
        ("gleif:1", False),
        ("gleif:2", False),
        ("gleif:3", False),
    ]
    assert run_columns(marked) == ["tfidf/a"]
    with pytest.raises(ValueError, match="at least two"):
        random_pair_level(matrix.head(1))


def test_the_random_pair_level_raises_when_no_draw_scored_anything() -> None:
    # Seed 0's one shuffle of two pairs leaves each name with its own target.
    with pytest.raises(ValueError, match="no draw scored"):
        random_pair_level(_named_matrix().head(2), draws=1, seed=0)


def test_pairs_are_counted_by_how_many_runs_found_them() -> None:
    counts = count_found_by_k(pair_found_matrix(_OUTCOMES, _RUNS))

    assert counts.rows() == [(0, 1), (1, 1), (2, 1)]


def test_every_two_runs_are_read_against_each_other() -> None:
    disagreement = run_disagreement(pair_found_matrix(_OUTCOMES, _RUNS))

    # tfidf alone found pair 1; both found pair 2; neither found pair 3.
    assert disagreement.rows() == [("tfidf/a", "wordpiece/b", 1, 0, 1)]


def test_run_overlap_reads_every_run_against_every_run_as_jaccard_and_containment() -> (
    None
):
    overlap = run_overlap(pair_found_matrix(_OUTCOMES, _RUNS))

    # tfidf found pairs 1 and 2, wordpiece pair 2 alone.
    assert overlap.select(
        "run", "other_run", "both", "jaccard", "containment"
    ).rows() == [
        ("tfidf/a", "tfidf/a", 2, 1.0, 1.0),
        ("tfidf/a", "wordpiece/b", 1, 0.5, 0.5),
        ("wordpiece/b", "tfidf/a", 1, 0.5, 1.0),
        ("wordpiece/b", "wordpiece/b", 1, 1.0, 1.0),
    ]


def test_pairs_are_counted_by_whether_found_and_whether_above_random() -> None:
    marked = with_name_similarity(pair_found_matrix(_OUTCOMES, _RUNS), random_level=1.0)

    assert count_by_reach(marked).rows() == [(True, False, 2), (False, False, 1)]


def _source_candidates(*rows: tuple[str, str, float, int, int]) -> pl.DataFrame:
    """A per-source candidates table of `(label, source_id, min_similarity,
    rank, candidates)` rows, every source in `ie`."""
    return pl.DataFrame(
        [
            {
                "label": label,
                "country": "ie",
                "source_id": source_id,
                "min_similarity": min_similarity,
                "rank": rank,
                "candidates": candidates,
            }
            for label, source_id, min_similarity, rank, candidates in rows
        ]
    )


def test_level_metrics_read_precision_recall_rr_and_f2_on_the_levels_own_sources() -> (
    None
):
    runs = _RUNS.with_columns(pl.lit(100).alias("target_rows"))
    # Source gleif:1 also holds a raw pair tfidf found, which its candidates serve.
    outcomes = pl.concat(
        [
            _OUTCOMES,
            pl.DataFrame(
                [
                    {
                        **_outcome("tfidf/a", 1, found=True, level="raw"),
                        "target_id": "ie:99",
                    },
                    {
                        **_outcome("wordpiece/b", 1, found=False, level="raw"),
                        "target_id": "ie:99",
                    },
                ]
            ),
        ]
    )
    source_candidates = _source_candidates(
        ("tfidf/a", "gleif:1", 0.5, 1, 1),
        ("tfidf/a", "gleif:1", 0.5, 2, 1),
        ("tfidf/a", "gleif:2", 0.9, 1, 1),
        ("tfidf/a", "gleif:3", 0.6, 1, 1),
        ("tfidf/a", "gleif:4", 1.0, 1, 5),
        ("wordpiece/b", "gleif:1", 0.7, 1, 2),
        ("wordpiece/b", "gleif:2", 0.8, 1, 1),
        ("wordpiece/b", "gleif:3", 0.4, 1, 1),
    )

    metrics = level_metrics(outcomes, runs, source_candidates)

    # never holds pairs 1-3 from sources 1-3; source 4 is raw alone.
    assert metrics.select(
        "run", "sources", "pairs", "found", "candidates", "truth_found"
    ).rows() == [
        ("tfidf/a", 3, 3, 2, 4, 3),
        ("wordpiece/b", 3, 3, 1, 4, 1),
    ]
    tfidf = metrics.row(0, named=True)
    assert tfidf["recall"] == pytest.approx(2 / 3)
    assert tfidf["precision"] == pytest.approx(3 / 4)
    assert tfidf["reduction_ratio"] == pytest.approx(1 - 4 / 300)
    assert tfidf["f2"] == pytest.approx(5 * 0.75 * (2 / 3) / (4 * 0.75 + 2 / 3))

    cut = level_metrics(
        outcomes,
        runs,
        source_candidates,
        min_similarity={"tfidf/a": 0.55},
        random_level=-1.0,
    )
    assert cut.row(0, named=True)["candidates"] == 2
    assert (
        cut.get_column("recall_above_random").to_list()
        == cut.get_column("recall").to_list()
    )


def test_the_overlap_heatmap_takes_jaccard_or_containment_only() -> None:
    fig, ax = plt.subplots()
    overlap = run_overlap(pair_found_matrix(_OUTCOMES, _RUNS))

    draw_run_overlap(ax, overlap, value="jaccard", title="overlap")

    assert [text.get_text() for text in ax.texts] == ["1.00", "0.50", "0.50", "1.00"]
    with pytest.raises(ValueError, match="jaccard or containment"):
        draw_run_overlap(ax, overlap, value="both", title="overlap")
    plt.close(fig)


def test_the_name_similarity_histogram_splits_found_from_missed_at_the_random_level() -> (
    None
):
    fig, ax = plt.subplots()
    marked = with_name_similarity(pair_found_matrix(_OUTCOMES, _RUNS), random_level=0.2)

    draw_name_similarity(ax, marked, random_level=0.2, title="reach")

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == [
        "found by some run (2)",
        "found by no run (1)",
        "random-pair level",
    ]
    plt.close(fig)


def test_level_metrics_can_set_aside_the_pairs_no_name_method_could_find() -> None:
    runs = _RUNS.with_columns(pl.lit(100).alias("target_rows"))
    source_candidates = _source_candidates(
        ("tfidf/a", "gleif:1", 0.5, 1, 1), ("wordpiece/b", "gleif:2", 0.5, 1, 1)
    )

    everything = level_metrics(_OUTCOMES, runs, source_candidates)
    all_above = level_metrics(
        _OUTCOMES, runs, source_candidates, random_level=-1.0, above_random_only=True
    )
    none_above = level_metrics(
        _OUTCOMES, runs, source_candidates, random_level=1.0, above_random_only=True
    )

    assert all_above.drop(
        "recall_above_random", "missed_findable", "missed_random"
    ).equals(everything)
    assert none_above.select("sources", "pairs", "candidates").rows() == [
        (0, 0, 0),
        (0, 0, 0),
    ]
    with pytest.raises(ValueError, match="random_level"):
        level_metrics(_OUTCOMES, runs, source_candidates, above_random_only=True)


def test_level_metrics_split_each_runs_misses_into_findable_and_random() -> None:
    runs = _RUNS.with_columns(pl.lit(100).alias("target_rows"))
    source_candidates = _source_candidates(("tfidf/a", "gleif:1", 0.5, 1, 1))

    all_above = level_metrics(_OUTCOMES, runs, source_candidates, random_level=-1.0)
    none_above = level_metrics(_OUTCOMES, runs, source_candidates, random_level=1.0)

    # tfidf missed pair 3, wordpiece pairs 1 and 3.
    assert all_above.select("run", "missed_findable", "missed_random").rows() == [
        ("tfidf/a", 1, 0),
        ("wordpiece/b", 2, 0),
    ]
    assert none_above.select("run", "missed_findable", "missed_random").rows() == [
        ("tfidf/a", 0, 1),
        ("wordpiece/b", 0, 2),
    ]


def test_level_metrics_read_each_runs_own_recall_area_and_its_audit() -> None:
    runs = _RUNS.with_columns(pl.lit(100).alias("target_rows"))
    source_candidates = _source_candidates(("tfidf/a", "gleif:1", 0.5, 1, 1))
    recall_curves = pl.DataFrame(
        [
            ("tfidf/a", "never", 0, 0.5),
            ("tfidf/a", "never", 10, 1.0),
            ("wordpiece/b", "never", 0, 0.0),
            ("wordpiece/b", "never", 4, 1.0),
        ],
        schema={
            "label": pl.Utf8,
            "population": pl.Utf8,
            "comparisons_spent": pl.Int64,
            "recall": pl.Float64,
        },
        orient="row",
    )
    # Pair 4 is not `never`, so it is not read.
    audits = pl.DataFrame(
        [
            ("tfidf/a", "ie", "gleif:1", "ie:1", True, "found"),
            ("tfidf/a", "ie", "gleif:3", "ie:3", True, "lost_to_backend"),
            ("tfidf/a", "ie", "gleif:4", "ie:4", False, "lost_to_backend"),
        ],
        schema={
            "label": pl.Utf8,
            "country": pl.Utf8,
            "source_id": pl.Utf8,
            "target_id": pl.Utf8,
            "exact_kept": pl.Boolean,
            "verdict": pl.Utf8,
        },
        orient="row",
    )

    metrics = level_metrics(
        _OUTCOMES, runs, source_candidates, recall_curves=recall_curves, audits=audits
    )

    # tfidf holds 0.5 over its own 10 comparisons, wordpiece 0 over its own 4.
    assert metrics.select(
        "run", "recall_area", "lost_to_backend", "exact_scan_recall"
    ).rows() == [
        ("tfidf/a", 0.5, 1, 1.0),
        ("wordpiece/b", 0.0, None, None),
    ]


def test_cluster_sizes_count_each_runs_pairs_by_their_sources_candidates() -> None:
    source_candidates = _source_candidates(
        ("tfidf/a", "gleif:1", 0.5, 1, 1),
        ("tfidf/a", "gleif:2", 0.5, 1, 1),
        ("tfidf/a", "gleif:2", 0.9, 2, 1),
        ("tfidf/a", "gleif:4", 0.9, 1, 7),
    )
    matrix = pair_found_matrix(_OUTCOMES, _RUNS)

    sizes = cluster_sizes(matrix, source_candidates, _RUNS.head(1))
    cut = cluster_sizes(
        matrix, source_candidates, _RUNS.head(1), min_similarity={"tfidf/a": 0.6}
    )

    # never holds pairs 1-3, one per source; pair 4 is raw and not counted.
    totals = sizes.group_by("cluster_size", maintain_order=True).agg(
        pl.col("pairs").sum()
    )
    assert dict(totals.iter_rows()) == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 0,
        "4-5": 0,
        "6-10": 0,
        "11-19": 0,
        "20+": 0,
    }
    # tfidf found pairs 1 and 2, whose sources were given 1 and 2 candidates.
    found = sizes.filter((pl.col("outcome") == "found") & (pl.col("pairs") > 0))
    assert found.get_column("cluster_size").to_list() == ["1", "2"]
    assert cut.filter(pl.col("cluster_size") == "0").get_column("pairs").sum() == 2
    fig, ax = plt.subplots()
    draw_cluster_sizes(ax, sizes, title="sizes")
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "found / findable"
    assert [text.get_text() for text in legend.get_texts()] == ["tfidf/a"]
    plt.close(fig)

    # Every name is below a level of 1.0, so the one missed pair moves over.
    split = cluster_sizes(matrix, source_candidates, _RUNS.head(1), random_level=1.0)
    missed = split.filter(pl.col("pairs") > 0, pl.col("outcome") != "found")
    assert missed.select("cluster_size", "outcome").rows() == [("0", "random")]
    fig, ax = plt.subplots()
    draw_cluster_sizes(ax, split, title="sizes")
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "found / findable / random"
    plt.close(fig)
