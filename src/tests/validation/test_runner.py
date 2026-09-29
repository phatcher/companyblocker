from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
import pytest
from company_vectorize.sbert_strategy import SbertClusteringStrategy

from acquisition.cleanser_orchestrate import materialize_cleansed_merge
from validation.config import ValidationRunConfig, validate_run_config
from validation.runner import (
    PAIR_TRUTH_EVAL_DETAIL_COLUMNS,
    _build_pair_truth_eval_output,
    _build_run_metrics_output,
    _resolve_source_truth_column,
    _result_schemas,
    build_source_perturbation_map,
    build_source_truth_map,
    compute_pair_truth_eval,
    compute_pair_truth_eval_detail,
    compute_robustness_eval,
    compute_run_metrics,
    rollup_robustness_eval,
    run_validation_matrix,
)
from workspace.artifact_layout import artifact_store_root, validation_artifact_root
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots
from workspace.telemetry import TELEMETRY_SCHEMA


def _write_cleansed_chunk(
    roots: WorkspaceRoots, system: str, rows: list[dict[str, object]]
) -> None:
    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    source_dir = cleansed_dir / "chunks"
    source_dir.mkdir(parents=True, exist_ok=True)
    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        cleansed_value = str(enriched.get("name_cleansed") or "")
        enriched.setdefault("name", cleansed_value)
        enriched.setdefault("name_cleansed_basic", cleansed_value)
        enriched_rows.append(enriched)
    pl.DataFrame(enriched_rows).write_parquet(source_dir / f"{system}-001.parquet")
    materialize_cleansed_merge(
        source_dir=source_dir,
        output_dir=cleansed_dir,
        system=system,
        name_col="name_cleansed",
        rows_per_file=1_000_000,
        force_rebuild=True,
    )


def _by_population(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Each `pair_truth_eval` row keyed by its population, read by name rather
    than position; a population appearing twice fails here rather than being
    silently overwritten."""
    populations = frame.get_column("population").to_list()
    assert len(populations) == len(set(populations)), populations
    return {row["population"]: row for row in frame.iter_rows(named=True)}


def test_compute_pair_truth_eval_reports_the_matrix_and_derived_rates() -> None:
    source_truth_map = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4", "s5"],
            "source_match_uri": ["t1", "t2", "t3", "t4", "t5"],
        }
    )
    matched_edges = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4"],
            "target_id": ["t1", "t2", "t3", "tX"],
        }
    )

    default_result = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        target_rows=5,
    )
    row = _by_population(default_result)["universe"]
    assert row["tp"] == 3
    assert row["fp"] == 1
    assert row["fn"] == 2
    assert row["precision"] == 0.75
    assert row["recall"] == 0.6
    # 5 labelled sources against 5 target rows. Ground truth lists matching
    # pairs only, so an unlisted pair is unknown rather than a known
    # non-match and there is no tn to count; with the three cells present
    # any F score is a division at read time, so none is stored.
    assert row["pair_universe"] == 25
    assert "tn" not in row
    assert "f1" not in row
    assert "f_beta" not in row

    # Without name forms no source can be given a level, so only the universe
    # is written: a level row of zeros would read as "computed as empty".
    assert default_result.get_column("population").to_list() == ["universe"]

    # candidate_pair_count is always populated (it costs nothing
    # beyond matched_edges itself), but candidate_set_size_ratio/recall_at_k
    # stay null when source_rows/top_k are omitted.
    assert row["candidate_pair_count"] == 4
    assert row["source_rows"] is None
    assert row["candidate_set_size_ratio"] is None
    assert row["recall_at_k"] is None


def test_compute_pair_truth_eval_candidate_set_size_ratio_and_recall_at_k() -> None:
    source_truth_map = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4", "s5"],
            "source_match_uri": ["t1", "t2", "t3", "t4", "t5"],
        }
    )
    # s1: correct target ranked 1st -- within any k. s2: correct target
    # ranked 2nd -- within k=2 but not k=1. s3: predicted target is wrong
    # (fp), s3's true target t3 never appears in matched_edges at all (fn).
    # s4/s5 have no predicted edge at all (fn).
    matched_edges = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s2", "s3"],
            "target_id": ["t1", "tY", "t2", "tX"],
            "rank": [1, 1, 2, 1],
        }
    )

    result = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        target_rows=10,
        source_rows=20,
        top_k=2,
    )
    row = _by_population(result)["universe"]

    assert row["candidate_pair_count"] == 4
    assert row["source_rows"] == 20
    # 4 candidate edges against a 20 (source) x 10 (target) exhaustive space.
    assert row["candidate_set_size_ratio"] == pytest.approx(4 / 200)
    # s1 (rank 1) and s2 (rank 2) both fall within k=2 -- 2 of 5 truth pairs.
    assert row["recall_at_k"] == pytest.approx(2 / 5)

    top1_result = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        target_rows=10,
        top_k=1,
    )
    top1_row = _by_population(top1_result)["universe"]
    # Only s1's rank-1 hit survives a k=1 window -- s2's correct target is
    # ranked 2nd, so it drops out even though it is present in matched_edges.
    assert top1_row["recall_at_k"] == pytest.approx(1 / 5)
    # source_rows was not passed here -- the ratio stays null independently
    # of top_k/recall_at_k, matching the two parameters' independence.
    assert top1_row["candidate_set_size_ratio"] is None


def test_compute_pair_truth_eval_recall_at_k_null_without_rank_column() -> None:
    source_truth_map = pl.DataFrame({"source_id": ["s1"], "source_match_uri": ["t1"]})
    matched_edges = pl.DataFrame({"source_id": ["s1"], "target_id": ["t1"]})

    result = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        target_rows=1,
        top_k=1,
    )
    assert _by_population(result)["universe"]["recall_at_k"] is None


def _three_bucket_name_forms() -> tuple[pl.DataFrame, pl.DataFrame]:
    # One pair per level, laid over the same 5 truth pairs / 4 predicted
    # pairs as test_compute_pair_truth_eval_reports_the_matrix_and_derived_rates
    # (tp={s1,s2,s3}, fp={s4-tX}, fn={s4-t4, s5-t5}):
    #   s1-t1  raw identical            -> raw
    #   s5-t5  raw identical            -> raw
    #   s2-t2  raw differs, cleansed equal -> cleansed
    #   s3-t3  cleansed differs         -> never
    #   s4-t4  cleansed differs         -> never
    #   s4-tX  an fp, counted at its source s4's level -> never
    source_name_forms = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4", "s5"],
            "name": [
                "acme limited",
                "beta holdings",
                "gamma incorporated",
                "delta corporation",
                "epsilon llc",
            ],
            "name_cleansed": [
                "acme ltd",
                "beta holdings",
                "gamma inc",
                "delta corp",
                "epsilon llc",
            ],
        }
    )
    target_name_forms = pl.DataFrame(
        {
            "target_id": ["t1", "t2", "t3", "t4", "t5", "tX"],
            "name": [
                "acme limited",
                "beta hldgs",
                "gamma group",
                "delta holdings",
                "epsilon llc",
                "unrelated name",
            ],
            "name_cleansed": [
                "acme ltd",
                "beta holdings",
                "gamma group",
                "delta hldgs",
                "epsilon llc",
                "unrelated name",
            ],
        }
    )
    return source_name_forms, target_name_forms


def _preprocessed_name_forms(*, with_column: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two pairs that differ as cleansed: s1's differs only by company type, so
    its preprocessed names are one string, and s2's differs in the name itself."""
    source = {
        "source_id": ["s1", "s2"],
        "name": ["Acme Ltd", "Bolt Ltd"],
        "name_cleansed": ["acme limited", "bolt limited"],
    }
    target = {
        "target_id": ["t1", "t2"],
        "name": ["Acme DAC", "Boult DAC"],
        "name_cleansed": ["acme designated activity company", "boult dac"],
    }
    if with_column:
        source["name_preprocessed"] = ["acme", "bolt"]
        target["name_preprocessed"] = ["acme", "boult"]
    return pl.DataFrame(source), pl.DataFrame(target)


def _levels_of_two_found_pairs(*, with_column: bool) -> dict[str, int]:
    source_name_forms, target_name_forms = _preprocessed_name_forms(
        with_column=with_column
    )
    result = compute_pair_truth_eval(
        source_truth_map=pl.DataFrame(
            {"source_id": ["s1", "s2"], "source_match_uri": ["t1", "t2"]}
        ),
        matched_edges=pl.DataFrame(
            {"source_id": ["s1", "s2"], "target_id": ["t1", "t2"]}
        ),
        source_system="src",
        target_system="tgt",
        country="ie",
        target_rows=2,
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
    )
    return {
        population: row["truth_pairs"]
        for population, row in _by_population(result).items()
    }


def test_a_pair_equal_only_once_preprocessed_is_kept_out_of_never() -> None:
    """The scan was handed `acme` twice, so finding that pair is not the
    algorithm's work: it is counted at `preprocessed`, and `never` holds only
    the pair whose names still differ."""
    truth_pairs = _levels_of_two_found_pairs(with_column=True)

    assert truth_pairs["preprocessed"] == 1
    assert truth_pairs["never"] == 1
    assert truth_pairs["cleansed"] == 0


def test_without_a_preprocessed_name_the_level_is_empty_and_never_keeps_the_pair() -> (
    None
):
    truth_pairs = _levels_of_two_found_pairs(with_column=False)

    assert truth_pairs["preprocessed"] == 0
    assert truth_pairs["never"] == 2


def test_compute_pair_truth_eval_gives_every_level_its_own_matrix() -> None:
    """Each level is the sources whose truth pair reaches it, with the whole
    matrix over those sources, so the easy and the hard are measured apart
    instead of blended."""
    source_truth_map = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4", "s5"],
            "source_match_uri": ["t1", "t2", "t3", "t4", "t5"],
        }
    )
    matched_edges = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4"],
            "target_id": ["t1", "t2", "t3", "tX"],
        }
    )
    source_name_forms, target_name_forms = _three_bucket_name_forms()

    result = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        target_rows=5,
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
    )
    rows = _by_population(result)

    assert list(rows) == [
        "universe",
        "raw",
        "basic",
        "cleansed",
        "preprocessed",
        "never",
        "unknown",
    ]

    # The universe is exactly what it was before the levels were added.
    universe = rows["universe"]
    assert (universe["tp"], universe["fp"], universe["fn"]) == (3, 1, 2)
    assert universe["precision"] == pytest.approx(0.75)
    assert universe["recall"] == pytest.approx(0.6)

    # never holds s3 (found) and s4 (missed). s4's wrong candidate tX is an
    # fp produced by a never source, so it is counted here, and the hard
    # names get a precision of their own.
    never = rows["never"]
    assert (never["labelled_sources"], never["truth_pairs"]) == (2, 2)
    assert (never["tp"], never["fp"], never["fn"]) == (1, 1, 1)
    assert never["precision"] == pytest.approx(0.5)
    assert never["recall"] == pytest.approx(0.5)
    assert never["pair_universe"] == 2 * 5

    # raw holds s1 (found) and s5 (missed), with no wrong candidate.
    raw = rows["raw"]
    assert (raw["tp"], raw["fp"], raw["fn"]) == (1, 0, 1)
    assert raw["precision"] == pytest.approx(1.0)
    assert raw["recall"] == pytest.approx(0.5)

    assert (rows["cleansed"]["tp"], rows["cleansed"]["fn"]) == (1, 0)
    assert rows["basic"]["labelled_sources"] == 0
    assert rows["basic"]["precision"] is None

    # The levels partition the universe exactly over its labelled-scoped cells.
    levels = [rows[name] for name in ("raw", "basic", "cleansed", "never", "unknown")]
    for cell in (
        "labelled_sources",
        "pair_universe",
        "truth_pairs",
        "predicted_pairs",
        "tp",
        "fp",
        "fn",
    ):
        assert sum(level[cell] for level in levels) == universe[cell], cell


def test_compute_pair_truth_eval_splits_basic_cleansing_from_full_cleansing() -> None:
    """A pair the basic tier already resolves is not the same difficulty as
    one needing the full cleanse, and reporting them as one lump hides which
    level of cleansing closed the gap."""
    source_name_forms = pl.DataFrame(
        {
            "source_id": ["s1"],
            "name": ["Acme Limited"],
            "name_cleansed_basic": ["acme limited"],
            "name_cleansed": ["acme ltd"],
        }
    )
    target_name_forms = pl.DataFrame(
        {
            "target_id": ["t1"],
            "name": ["ACME LIMITED"],
            "name_cleansed_basic": ["acme limited"],
            "name_cleansed": ["acme ltd"],
        }
    )

    rows = _by_population(
        compute_pair_truth_eval(
            source_truth_map=pl.DataFrame(
                {"source_id": ["s1"], "source_match_uri": ["t1"]}
            ),
            matched_edges=pl.DataFrame({"source_id": ["s1"], "target_id": ["t1"]}),
            source_system="src",
            target_system="tgt",
            country="gb",
            target_rows=1,
            source_name_forms=source_name_forms,
            target_name_forms=target_name_forms,
        )
    )

    # Raw names differ only in case, so basic cleansing closes it -- not the
    # full cleanse, which would have claimed it before the split existed.
    assert rows["basic"]["truth_pairs"] == 1
    assert rows["cleansed"]["truth_pairs"] == 0
    assert rows["raw"]["truth_pairs"] == 0
    assert rows["never"]["truth_pairs"] == 0


def test_compute_pair_truth_eval_without_basic_column_keeps_the_old_split() -> None:
    """An input carrying no `name_cleansed_basic` collapses the level rather
    than failing: its pairs stay in `cleanse_absorbed`, exactly as they were
    reported before the basic tier was measured at all."""
    source_name_forms = pl.DataFrame(
        {"source_id": ["s1"], "name": ["Acme Limited"], "name_cleansed": ["acme ltd"]}
    )
    target_name_forms = pl.DataFrame(
        {"target_id": ["t1"], "name": ["ACME LIMITED"], "name_cleansed": ["acme ltd"]}
    )

    rows = _by_population(
        compute_pair_truth_eval(
            source_truth_map=pl.DataFrame(
                {"source_id": ["s1"], "source_match_uri": ["t1"]}
            ),
            matched_edges=pl.DataFrame({"source_id": ["s1"], "target_id": ["t1"]}),
            source_system="src",
            target_system="tgt",
            country="gb",
            target_rows=1,
            source_name_forms=source_name_forms,
            target_name_forms=target_name_forms,
        )
    )

    assert rows["basic"]["truth_pairs"] == 0
    assert rows["cleansed"]["truth_pairs"] == 1


def test_compute_pair_truth_eval_detail_matches_pair_truth_eval_bucket_split() -> None:
    # Same 5 truth pairs / 4 predicted pairs as
    # test_compute_pair_truth_eval_three_bucket_split (tp={s1,s2,s3},
    # fp={s4-tX}, fn={s4-t4, s5-t5}), so the per-pair rows here must sum to
    # that test's aggregate counts exactly.
    source_truth_map = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4", "s5"],
            "source_match_uri": ["t1", "t2", "t3", "t4", "t5"],
        }
    )
    matched_edges = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4"],
            "target_id": ["t1", "t2", "t3", "tX"],
            "similarity": [0.99, 0.95, 0.80, 0.60],
            "rank": [1, 1, 1, 2],
            "country": ["gb", "gb", "gb", "gb"],
        }
    )
    source_name_forms, target_name_forms = _three_bucket_name_forms()

    detail = compute_pair_truth_eval_detail(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
    )

    assert detail.columns == list(PAIR_TRUTH_EVAL_DETAIL_COLUMNS.keys())
    assert detail.height == 6  # 5 truth pairs + 1 predicted-but-untrue pair
    assert set(detail["source_system"].unique().to_list()) == {"src"}
    assert set(detail["target_system"].unique().to_list()) == {"tgt"}
    assert set(detail["country"].unique().to_list()) == {"gb"}

    truth_rows = detail.filter(pl.col("is_truth_pair"))
    assert truth_rows.height == 5
    found_by_pair = {
        (row["source_id"], row["target_id"]): row["found"]
        for row in truth_rows.iter_rows(named=True)
    }
    assert found_by_pair == {
        ("s1", "t1"): True,
        ("s2", "t2"): True,
        ("s3", "t3"): True,
        ("s4", "t4"): False,
        ("s5", "t5"): False,
    }
    # A truth pair that was never predicted at all has no similarity/rank to
    # report -- there is no edge to read them from.
    missed_row = truth_rows.filter(pl.col("source_id") == "s5").row(0, named=True)
    assert missed_row["similarity"] is None
    assert missed_row["rank"] is None

    fp_rows = detail.filter(~pl.col("is_truth_pair"))
    assert fp_rows.height == 1
    fp_row = fp_rows.row(0, named=True)
    assert (fp_row["source_id"], fp_row["target_id"]) == ("s4", "tX")
    assert fp_row["found"] is None
    assert fp_row["similarity"] == pytest.approx(0.60)
    assert fp_row["rank"] == 2

    bucket_by_pair = {
        (row["source_id"], row["target_id"]): row["name_equality"]
        for row in detail.iter_rows(named=True)
    }
    assert bucket_by_pair[("s1", "t1")] == "raw"
    assert bucket_by_pair[("s5", "t5")] == "raw"
    assert bucket_by_pair[("s2", "t2")] == "cleansed"
    assert bucket_by_pair[("s3", "t3")] == "never"
    assert bucket_by_pair[("s4", "t4")] == "never"
    assert bucket_by_pair[("s4", "tX")] == "never"

    # The detail frame judges each pair by its own two names, so the fp s4-tX
    # is `never` because s4 and tX differ. `pair_truth_eval` counts an fp at
    # its source's truth-pair level instead, s4-t4 here; the two coincide in
    # this fixture but need not, so regrouping this frame by `name_equality`
    # does not reproduce the summary's per-level fp counts.
    cleansed_different = detail.filter(pl.col("name_equality") == "never")
    assert (
        cleansed_different.filter(pl.col("is_truth_pair") & pl.col("found")).height == 1
    )
    assert (
        cleansed_different.filter(pl.col("is_truth_pair") & ~pl.col("found")).height
        == 1
    )
    assert cleansed_different.filter(~pl.col("is_truth_pair")).height == 1

    # Names are carried on every row, not just the residual bucket.
    s1_row = detail.filter(
        (pl.col("source_id") == "s1") & (pl.col("target_id") == "t1")
    ).row(0, named=True)
    assert s1_row["source_name"] == "acme limited"
    assert s1_row["source_name_cleansed"] == "acme ltd"
    assert s1_row["target_name"] == "acme limited"
    assert s1_row["target_name_cleansed"] == "acme ltd"


def test_compute_pair_truth_eval_counts_missing_side_as_unclassified() -> None:
    # s9's truth target t9 is absent from the target population, so no
    # cleansed-vs-cleansed verdict is possible. It must be counted as
    # unclassified, never silently treated as a difference and dragged into
    # the residual the primary figures are computed over -- that conflation
    # is the residual-scoped defect this replaces.
    source_name_forms, target_name_forms = _three_bucket_name_forms()
    source_name_forms = pl.concat(
        [
            source_name_forms,
            pl.DataFrame(
                {
                    "source_id": ["s9"],
                    "name": ["zeta group"],
                    "name_cleansed": ["zeta grp"],
                }
            ),
        ]
    )

    rows = _by_population(
        compute_pair_truth_eval(
            source_truth_map=pl.DataFrame(
                {
                    "source_id": ["s3", "s9"],
                    "source_match_uri": ["t3", "t9"],
                }
            ),
            matched_edges=pl.DataFrame({"source_id": ["s3"], "target_id": ["t3"]}),
            source_system="src",
            target_system="tgt",
            country="gb",
            target_rows=6,
            source_name_forms=source_name_forms,
            target_name_forms=target_name_forms,
        )
    )

    assert rows["universe"]["truth_pairs"] == 2
    assert rows["unknown"]["truth_pairs"] == 1
    assert rows["never"]["truth_pairs"] == 1
    # s9-t9 is an unmatched truth pair, but it is not a residual fn: it
    # never had a verdict, so it counts against unknown and not never.
    assert rows["universe"]["fn"] == 1
    assert rows["unknown"]["fn"] == 1
    assert rows["never"]["fn"] == 0
    assert rows["never"]["recall"] == pytest.approx(1.0)


def test_build_pair_truth_eval_output_rolls_up_each_population_apart() -> None:
    schemas = _result_schemas()
    source_name_forms = pl.DataFrame(
        {
            "source_id": ["s1", "s2"],
            "name": ["acme limited", "beta holdings"],
            "name_cleansed": ["acme ltd", "beta holdings"],
        }
    )
    target_name_forms = pl.DataFrame(
        {
            "target_id": ["t1", "t2"],
            "name": ["acme limited", "beta group"],
            "name_cleansed": ["acme ltd", "beta group"],
        }
    )
    gb_eval = compute_pair_truth_eval(
        source_truth_map=pl.DataFrame(
            {"source_id": ["s1", "s2"], "source_match_uri": ["t1", "t2"]}
        ),
        matched_edges=pl.DataFrame(
            {"source_id": ["s1", "s2"], "target_id": ["t1", "t2"]}
        ),
        source_system="src",
        target_system="tgt",
        country="gb",
        target_rows=2,
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
    )
    # ie has no name-form lookups, so it contributes a universe row only.
    ie_eval = compute_pair_truth_eval(
        source_truth_map=pl.DataFrame(
            {"source_id": ["s3"], "source_match_uri": ["t3"]}
        ),
        matched_edges=pl.DataFrame({"source_id": ["s3"], "target_id": ["t3"]}),
        source_system="src",
        target_system="tgt",
        country="ie",
        target_rows=1,
    )

    output = _build_pair_truth_eval_output(
        pair_truth_eval_parts=[gb_eval, ie_eval], schemas=schemas
    )
    rolled = _by_population(output.filter(pl.col("country") == "__all__"))

    # Summed within a population and never across them: the universe covers
    # gb's two sources and ie's one, and is not inflated by gb's level rows.
    assert rolled["universe"]["labelled_sources"] == 3
    assert rolled["universe"]["tp"] == 3
    # Only gb has levels: s1-t1 is raw-equal, s2-t2 never-equal, both found.
    assert rolled["raw"]["tp"] == 1
    assert rolled["never"]["tp"] == 1
    assert rolled["never"]["fn"] == 0
    assert rolled["never"]["recall"] == pytest.approx(1.0)


# -- self-validation truth resolution --------------------------------------------------


def test_resolve_source_truth_column_uses_source_uri_for_self_validation() -> None:
    source_slice = pl.DataFrame(
        {
            "system_uri": ["perturbed://fp1/robustness-v1/light-typo"],
            "source_uri": ["gb://1"],
            "match_uri": [None],
        }
    )
    assert (
        _resolve_source_truth_column(source_slice, target_system="gb") == "source_uri"
    )


def test_resolve_source_truth_column_falls_back_to_match_uri_for_cross_system() -> None:
    # A perturbed row materialized from gleif, validated against gb: source_uri points at
    # gleif, not gb, so this is cross-system mode and match_uri (real ground truth,
    # passed through unchanged by perturbation_materializer.py) is what must be read.
    source_slice = pl.DataFrame(
        {
            "system_uri": ["perturbed://fp1/robustness-v1/light-typo"],
            "source_uri": ["gleif://LEI-1"],
            "match_uri": ["gb://1"],
        }
    )
    assert _resolve_source_truth_column(source_slice, target_system="gb") == "match_uri"


def test_resolve_source_truth_column_falls_back_for_plain_data() -> None:
    source_slice = pl.DataFrame({"system_uri": ["gleif:1"], "match_uri": ["gb:1"]})
    assert _resolve_source_truth_column(source_slice, target_system="gb") == "match_uri"


def test_resolve_source_truth_column_falls_back_when_source_uri_all_null() -> None:
    source_slice = pl.DataFrame(
        {"system_uri": ["gb://1"], "source_uri": [None], "match_uri": ["gleif://LEI-1"]}
    )
    assert _resolve_source_truth_column(source_slice, target_system="gb") == "match_uri"


def test_resolve_source_truth_column_falls_back_when_source_uri_absent() -> None:
    source_slice = pl.DataFrame({"system_uri": ["gb:1"], "match_uri": [None]})
    assert _resolve_source_truth_column(source_slice, target_system="gb") == "match_uri"


def test_build_source_truth_map_self_validation() -> None:
    source_slice = pl.DataFrame(
        {
            "system_uri": ["perturbed://fp1/robustness-v1/light-typo"],
            "source_uri": ["gb://1"],
            "match_uri": [None],
        }
    )
    truth_map = build_source_truth_map(source_slice, target_system="gb")
    row = truth_map.row(0, named=True)
    assert row["source_id"] == "perturbed://fp1/robustness-v1/light-typo"
    assert row["source_match_uri"] == "gb://1"


def test_build_source_truth_map_cross_system_unchanged() -> None:
    source_slice = pl.DataFrame({"system_uri": ["gleif:1"], "match_uri": ["gb:1"]})
    truth_map = build_source_truth_map(source_slice, target_system="gb")
    row = truth_map.row(0, named=True)
    assert row["source_id"] == "gleif:1"
    assert row["source_match_uri"] == "gb:1"


def test_build_source_truth_map_no_match_uri_column() -> None:
    source_slice = pl.DataFrame({"system_uri": ["gleif:1"]})
    truth_map = build_source_truth_map(source_slice, target_system="gb")
    row = truth_map.row(0, named=True)
    assert row["source_match_uri"] is None


# -- v1 robustness metric contract ------------------------------------------------------


def test_build_source_perturbation_map_returns_none_without_perturbation_columns() -> (
    None
):
    frame = pl.DataFrame({"system_uri": ["gb:1"]})
    assert build_source_perturbation_map(frame) is None


def test_build_source_perturbation_map_extracts_lookup() -> None:
    frame = pl.DataFrame(
        {
            "system_uri": ["perturbed://fp1/robustness-v1/light-typo"],
            "profile_id": ["robustness-v1"],
            "profile_version": ["1.0.0"],
            "scenario_id": ["light-typo"],
            "intensity": [0.5],
        }
    )
    lookup = build_source_perturbation_map(frame)
    assert lookup is not None
    row = lookup.row(0, named=True)
    assert row["source_id"] == "perturbed://fp1/robustness-v1/light-typo"
    assert row["profile_id"] == "robustness-v1"
    assert row["intensity"] == 0.5


def test_build_source_perturbation_map_drops_null_profile_id_rows() -> None:
    frame = pl.DataFrame(
        {
            "system_uri": ["gb:1", "perturbed://fp1/robustness-v1/light-typo"],
            "profile_id": [None, "robustness-v1"],
            "profile_version": [None, "1.0.0"],
            "scenario_id": [None, "light-typo"],
            "intensity": [None, 0.5],
        }
    )
    lookup = build_source_perturbation_map(frame)
    assert lookup is not None
    assert lookup.height == 1
    assert lookup.row(0, named=True)["source_id"] == (
        "perturbed://fp1/robustness-v1/light-typo"
    )


def _robustness_fixture(
    *, intensities: tuple[float, ...] = (0.25, 0.75)
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """A tiny two-scenario, two-intensity fixture: `light` scenario has one
    true positive at each intensity, `heavy` degrades from a true positive
    at the lower intensity to a false negative at the higher one -- so the
    two scenarios' `recall_retention_ratio` trend differently, exercising
    the per-group split rather than one pooled number.
    """
    source_perturbation_map = pl.DataFrame(
        {
            "source_id": ["s-light-lo", "s-light-hi", "s-heavy-lo", "s-heavy-hi"],
            "profile_id": ["robustness-v1"] * 4,
            "profile_version": ["1.0.0"] * 4,
            "scenario_id": ["light", "light", "heavy", "heavy"],
            "intensity": [
                intensities[0],
                intensities[1],
                intensities[0],
                intensities[1],
            ],
        }
    )
    source_truth_map = pl.DataFrame(
        {
            "source_id": ["s-light-lo", "s-light-hi", "s-heavy-lo", "s-heavy-hi"],
            "source_match_uri": ["t1", "t2", "t3", "t4"],
        }
    )
    matched_edges = pl.DataFrame(
        {
            "source_id": ["s-light-lo", "s-light-hi", "s-heavy-lo"],
            "target_id": ["t1", "t2", "t3"],
        }
    )
    return source_perturbation_map, source_truth_map, matched_edges


def test_compute_robustness_eval_one_row_per_group_with_correct_ratio() -> None:
    source_perturbation_map, source_truth_map, matched_edges = _robustness_fixture()

    result = compute_robustness_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_perturbation_map=source_perturbation_map,
        source_system="perturbed:robustness-v1",
        target_system="gb",
        country="gb",
        baseline_recall=1.0,
    )

    assert result.height == 4
    assert set(zip(result["scenario_id"], result["intensity"], strict=True)) == {
        ("light", 0.25),
        ("light", 0.75),
        ("heavy", 0.25),
        ("heavy", 0.75),
    }

    heavy_hi = result.filter(
        (pl.col("scenario_id") == "heavy") & (pl.col("intensity") == 0.75)
    ).row(0, named=True)
    assert heavy_hi["recall"] == pytest.approx(0.0)
    assert heavy_hi["recall_retention_ratio"] == pytest.approx(0.0)

    light_lo = result.filter(
        (pl.col("scenario_id") == "light") & (pl.col("intensity") == 0.25)
    ).row(0, named=True)
    assert light_lo["recall"] == pytest.approx(1.0)
    assert light_lo["recall_retention_ratio"] == pytest.approx(1.0)


def test_compute_robustness_eval_ratio_is_null_when_baseline_missing_or_zero() -> None:
    source_perturbation_map, source_truth_map, matched_edges = _robustness_fixture()

    for baseline_recall in (None, 0.0):
        result = compute_robustness_eval(
            source_truth_map=source_truth_map,
            matched_edges=matched_edges,
            source_perturbation_map=source_perturbation_map,
            source_system="perturbed:robustness-v1",
            target_system="gb",
            country="gb",
            baseline_recall=baseline_recall,
        )
        assert result.get_column("recall_retention_ratio").is_null().all()
        assert result.get_column("baseline_recall").to_list() == (
            [baseline_recall] * result.height
        )


def test_compute_robustness_eval_empty_perturbation_map_returns_empty_frame() -> None:
    empty_map = pl.DataFrame(
        schema={
            "source_id": pl.Utf8,
            "profile_id": pl.Utf8,
            "profile_version": pl.Utf8,
            "scenario_id": pl.Utf8,
            "intensity": pl.Float64,
        }
    )
    result = compute_robustness_eval(
        source_truth_map=pl.DataFrame(
            schema={"source_id": pl.Utf8, "source_match_uri": pl.Utf8}
        ),
        matched_edges=pl.DataFrame(schema={"source_id": pl.Utf8, "target_id": pl.Utf8}),
        source_perturbation_map=empty_map,
        source_system="perturbed:robustness-v1",
        target_system="gb",
        country="gb",
        baseline_recall=1.0,
    )
    assert result.height == 0
    from validation.contracts import ARTIFACT_SCHEMAS

    assert set(result.columns) == set(ARTIFACT_SCHEMAS["robustness_eval"])


def test_compute_robustness_eval_carries_dataset_snapshot_manifest_path_pointer() -> (
    None
):
    """`dataset_snapshot_manifest_path` is a repo-relative pointer carried onto every
    output row, defaulting to null when the caller has no provenance to offer."""
    source_perturbation_map, source_truth_map, matched_edges = _robustness_fixture()
    relative = "data/perturbed/robustness-v1/_materialization_manifest_gb.json"

    with_pointer = compute_robustness_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_perturbation_map=source_perturbation_map,
        source_system="perturbed:robustness-v1",
        target_system="gb",
        country="gb",
        baseline_recall=1.0,
        dataset_snapshot_manifest_path=relative,
    )
    assert with_pointer.height == 4
    assert (
        with_pointer.get_column("dataset_snapshot_manifest_path").to_list()
        == [relative] * 4
    )

    without_pointer = compute_robustness_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_perturbation_map=source_perturbation_map,
        source_system="perturbed:robustness-v1",
        target_system="gb",
        country="gb",
        baseline_recall=1.0,
    )
    assert without_pointer.get_column("dataset_snapshot_manifest_path").is_null().all()


@pytest.mark.parametrize(
    "absolute",
    [
        "/data/perturbed/robustness-v1/_materialization_manifest_gb.json",
        r"C:\Devel\Brunel\blocking\data\perturbed\robustness-v1\m.json",
        r"\\share\blocking\data\perturbed\robustness-v1\m.json",
    ],
)
def test_compute_robustness_eval_rejects_an_absolute_manifest_pointer(
    absolute: str,
) -> None:
    """An absolute path names a root no other checkout has: every worktree reaches the
    one shared `data/` through its own root, so persisting one would make the pointer
    unresolvable everywhere but the machine that wrote it. Both path flavours are
    rejected on either platform, since a POSIX-absolute path carries no drive letter
    and would otherwise read as merely drive-relative on Windows."""
    source_perturbation_map, source_truth_map, matched_edges = _robustness_fixture()

    with pytest.raises(ValueError, match="must be repo-relative"):
        compute_robustness_eval(
            source_truth_map=source_truth_map,
            matched_edges=matched_edges,
            source_perturbation_map=source_perturbation_map,
            source_system="perturbed:robustness-v1",
            target_system="gb",
            country="gb",
            baseline_recall=1.0,
            dataset_snapshot_manifest_path=absolute,
        )


def test_compute_robustness_eval_normalises_a_windows_relative_pointer() -> None:
    """A relative path written with backslashes is stored POSIX-separated, so an
    artifact written on Windows resolves on Linux."""
    source_perturbation_map, source_truth_map, matched_edges = _robustness_fixture()

    frame = compute_robustness_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_perturbation_map=source_perturbation_map,
        source_system="perturbed:robustness-v1",
        target_system="gb",
        country="gb",
        baseline_recall=1.0,
        dataset_snapshot_manifest_path=r"data\perturbed\robustness-v1\m.json",
    )
    assert (
        frame.get_column("dataset_snapshot_manifest_path").to_list()
        == ["data/perturbed/robustness-v1/m.json"] * 4
    )


def test_rollup_robustness_eval_reports_mean_and_std_across_countries() -> None:
    frame = pl.DataFrame(
        {
            "profile_id": ["robustness-v1"] * 3,
            "profile_version": ["1.0.0"] * 3,
            "scenario_id": ["light"] * 3,
            "intensity": [0.5, 0.5, 0.5],
            "recall_retention_ratio": [1.0, 0.5, 0.75],
        },
        schema_overrides={"intensity": pl.Float64},
    )

    rollup = rollup_robustness_eval(frame)

    assert rollup.height == 1
    row = rollup.row(0, named=True)
    assert row["sample_count"] == 3
    assert row["recall_retention_ratio_mean"] == pytest.approx(0.75)
    assert row["recall_retention_ratio_std"] is not None


def test_rollup_robustness_eval_std_is_null_for_a_single_sample() -> None:
    frame = pl.DataFrame(
        {
            "profile_id": ["robustness-v1"],
            "profile_version": ["1.0.0"],
            "scenario_id": ["light"],
            "intensity": [0.5],
            "recall_retention_ratio": [1.0],
        },
        schema_overrides={"intensity": pl.Float64},
    )

    rollup = rollup_robustness_eval(frame)

    row = rollup.row(0, named=True)
    assert row["sample_count"] == 1
    assert row["recall_retention_ratio_std"] is None


def test_rollup_robustness_eval_empty_frame_returns_empty_schema() -> None:
    from validation.contracts import ARTIFACT_SCHEMAS

    empty = pl.DataFrame(schema=ARTIFACT_SCHEMAS["robustness_eval"])
    rollup = rollup_robustness_eval(empty)
    assert rollup.height == 0
    assert set(rollup.columns) == set(ARTIFACT_SCHEMAS["robustness_eval_rollup"])


def test_compute_run_metrics_latency_and_duplication() -> None:
    # 4 candidate edges from 4 distinct source rows, but only 2 distinct
    # targets -> half of the candidate edges are "duplicates" of a target
    # already claimed by another source row's candidate set.
    candidate_edges = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3", "s4"],
            "target_id": ["t1", "t1", "t2", "t2"],
        }
    )

    result = compute_run_metrics(
        candidate_edges=candidate_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        source_rows=200,
        elapsed_seconds=4.0,
    )
    row = result.row(0, named=True)

    assert row["source_rows"] == 200
    assert row["elapsed_seconds"] == pytest.approx(4.0)
    # 4.0s for 200 rows -> 20.0s per 1k rows.
    assert row["latency_seconds_per_1k_rows"] == pytest.approx(20.0)
    assert row["candidate_edges"] == 4
    assert row["candidate_distinct_targets"] == 2
    assert row["candidate_duplication_ratio"] == pytest.approx(0.5)


def test_compute_run_metrics_handles_no_candidates_and_no_rows() -> None:
    empty_edges = pl.DataFrame(
        {"source_id": [], "target_id": []},
        schema={"source_id": pl.Utf8, "target_id": pl.Utf8},
    )

    result = compute_run_metrics(
        candidate_edges=empty_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        source_rows=0,
        elapsed_seconds=1.5,
    )
    row = result.row(0, named=True)

    assert row["source_rows"] == 0
    assert row["latency_seconds_per_1k_rows"] is None
    assert row["candidate_edges"] == 0
    assert row["candidate_distinct_targets"] == 0
    assert row["candidate_duplication_ratio"] is None


def test_compute_run_metrics_no_duplication_when_targets_all_unique() -> None:
    candidate_edges = pl.DataFrame(
        {
            "source_id": ["s1", "s2", "s3"],
            "target_id": ["t1", "t2", "t3"],
        }
    )

    result = compute_run_metrics(
        candidate_edges=candidate_edges,
        source_system="src",
        target_system="tgt",
        country="gb",
        source_rows=3,
        elapsed_seconds=0.3,
    )
    row = result.row(0, named=True)

    assert row["candidate_duplication_ratio"] == pytest.approx(0.0)


def test_build_run_metrics_output_rolls_up_across_countries() -> None:
    schemas = _result_schemas()
    gb_metrics = compute_run_metrics(
        candidate_edges=pl.DataFrame(
            {"source_id": ["s1", "s2"], "target_id": ["t1", "t1"]}
        ),
        source_system="src",
        target_system="tgt",
        country="gb",
        source_rows=100,
        elapsed_seconds=2.0,
    ).select(list(schemas["run_metrics"].keys()))
    ie_metrics = compute_run_metrics(
        candidate_edges=pl.DataFrame({"source_id": ["s3"], "target_id": ["t2"]}),
        source_system="src",
        target_system="tgt",
        country="ie",
        source_rows=50,
        elapsed_seconds=1.0,
    ).select(list(schemas["run_metrics"].keys()))

    output = _build_run_metrics_output(
        run_metrics_parts=[gb_metrics, ie_metrics],
        schemas=schemas,
    )

    assert output.height == 3
    global_row = output.filter(pl.col("country") == "__all__").row(0, named=True)
    assert global_row["source_rows"] == 150
    assert global_row["elapsed_seconds"] == pytest.approx(3.0)
    # (2.0s + 1.0s) / 150 rows * 1000 = 20.0s per 1k rows.
    assert global_row["latency_seconds_per_1k_rows"] == pytest.approx(20.0)
    assert global_row["candidate_edges"] == 3
    assert global_row["candidate_distinct_targets"] == 2
    assert global_row["candidate_duplication_ratio"] == pytest.approx(1.0 / 3.0)


def test_build_run_metrics_output_empty_parts_returns_empty_frame() -> None:
    schemas = _result_schemas()
    output = _build_run_metrics_output(run_metrics_parts=[], schemas=schemas)
    assert output.height == 0
    assert set(output.columns) == set(schemas["run_metrics"].keys())


@pytest.mark.integration
def test_run_validation_matrix_with_synthetic_data(
    workspace_roots: WorkspaceRoots,
) -> None:
    run_date = "2026-07-07"
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
            {
                "system_uri": "gleif:2",
                "name_cleansed": "beta holdings",
                "jurisdiction_code": "gb",
                "match_uri": None,
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:2",
                "name_cleansed": "omega plc",
                "jurisdiction_code": "gb",
            },
        ],
    )

    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date=run_date,
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=("gb",),
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=validation_artifact_root(workspace_roots),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )
    validate_run_config(config)

    results = run_validation_matrix(config)

    assert set(results.keys()) == {
        "source_outcomes",
        "clusters",
        "directional_coverage",
        "cluster_shape",
        "pair_truth_eval",
        "exceptions",
        "run_metrics",
        "timings",
    }
    assert results["source_outcomes"].height >= 1
    assert results["clusters"].height >= 2
    # One record per phase, labelled with the unit it ran over.
    timings = results["timings"]
    assert timings.schema == TELEMETRY_SCHEMA
    routines = timings.get_column("routine").to_list()
    assert "prepare_system" in routines
    assert "prepare_country" in routines
    assert "scoring" in routines
    scoring = timings.filter(pl.col("routine") == "scoring").row(0, named=True)
    assert scoring["label"] == "gleif->gb:gb"
    assert scoring["rows_in"] >= 1

    source_outcomes = results["source_outcomes"]
    pair_eval = results["pair_truth_eval"]
    assert "source_match_uri" in source_outcomes.columns
    assert "is_true_match" in source_outcomes.columns
    assert "true_match_status" in source_outcomes.columns
    assert "reduction_ratio" in pair_eval.columns
    true_match_rows = source_outcomes.filter(
        (pl.col("source_id") == "gleif:1")
        & (pl.col("target_id") == "gb:1")
        & pl.col("is_true_match")
        & (pl.col("true_match_status") == "true_match")
    )
    assert true_match_rows.height >= 1

    assert pair_eval.height >= 1
    gb_rows = _by_population(pair_eval.filter(pl.col("country") == "gb"))
    gb_eval_row = gb_rows["universe"]
    assert gb_eval_row["tp"] >= 1
    assert gb_eval_row["reduction_ratio"] is not None
    # gleif:1 and gb:1 differ on both raw name and name_cleansed
    # ("acme limited" vs "acme ltd"), so this real end-to-end TP lands in
    # the never level -- exercised through the full run_validation_matrix()
    # pipeline, not just compute_pair_truth_eval() called directly.
    assert gb_rows["never"]["tp"] == gb_eval_row["tp"]
    assert gb_rows["raw"]["truth_pairs"] == 0
    assert gb_rows["cleansed"]["truth_pairs"] == 0
    assert gb_rows["unknown"]["truth_pairs"] == 0
    assert gb_rows["never"]["recall"] is not None

    coverage = results["directional_coverage"].row(0, named=True)
    assert coverage["source_records"] == 2
    assert 0.0 <= coverage["directional_coverage_ratio"] <= 1.0

    shape = results["cluster_shape"].row(0, named=True)
    assert shape["total_clusters"] >= 1
    assert shape["max_cluster_size"] >= 1

    run_metrics = results["run_metrics"]
    assert run_metrics.height >= 1
    gb_metrics = run_metrics.filter(pl.col("country") == "gb").row(0, named=True)
    assert gb_metrics["source_rows"] == 2
    assert gb_metrics["elapsed_seconds"] >= 0.0
    assert gb_metrics["latency_seconds_per_1k_rows"] is not None
    assert gb_metrics["latency_seconds_per_1k_rows"] >= 0.0
    assert gb_metrics["candidate_edges"] >= 1
    assert gb_metrics["candidate_duplication_ratio"] is not None
    assert 0.0 <= gb_metrics["candidate_duplication_ratio"] <= 1.0

    global_metrics_frame = run_metrics.filter(pl.col("country") == "__all__")
    assert global_metrics_frame.height == 1
    global_metrics = global_metrics_frame.row(0, named=True)
    assert global_metrics["source_rows"] == gb_metrics["source_rows"]
    assert global_metrics["candidate_edges"] == gb_metrics["candidate_edges"]


@pytest.mark.integration
def test_run_validation_matrix_self_validation_scores_against_source_uri(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A perturbed row's ground truth for self-validation is its own
    `source_uri` (its pre-perturbation identity in the target system it was
    materialized from), not `match_uri` -- which stays null here exactly as
    `perturbation_materializer.py` leaves it on a row with no pre-existing
    cross-system truth. Materializes the perturbed row directly via a
    cleansed-chunk fixture (mirroring `_write_cleansed_chunk`), keeping this
    test decoupled from `company_perturbation`'s real operators.
    """
    run_date = "2026-07-07"
    _write_cleansed_chunk(
        workspace_roots,
        "synth",
        [
            {
                "system_uri": "perturbed://fp1/robustness-v1/light-typo",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
                "source_uri": "gb://1",
                "match_uri": None,
                "profile_id": "robustness-v1",
                "profile_version": "1.0.0",
                "scenario_id": "light-typo",
                "intensity": 0.5,
                "seed": "12345",
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb://1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb://2",
                "name_cleansed": "omega plc",
                "jurisdiction_code": "gb",
            },
        ],
    )

    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date=run_date,
        source_systems=("synth",),
        target_systems=("gb",),
        countries=("gb",),
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=validation_artifact_root(workspace_roots),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )
    validate_run_config(config)

    results = run_validation_matrix(config)

    source_outcomes = results["source_outcomes"]
    true_match_rows = source_outcomes.filter(
        (pl.col("source_id") == "perturbed://fp1/robustness-v1/light-typo")
        & (pl.col("target_id") == "gb://1")
        & pl.col("is_true_match")
        & (pl.col("true_match_status") == "true_match")
    )
    assert true_match_rows.height >= 1

    row = _by_population(results["pair_truth_eval"].filter(pl.col("country") == "gb"))[
        "universe"
    ]
    assert row["tp"] == 1
    assert row["recall"] == pytest.approx(1.0)


class _RecordingFixedVectorEncoder:
    """Deterministic stand-in for a real sentence-transformer encoder, keyed on
    the exact text it is handed, and recording every text it saw.

    `sentence-transformers` is an optional dependency that is not installed
    here, so the encoder is injected through
    `SbertClusteringStrategy(encoder_factory=...)` rather than monkeypatched.
    Recording the texts is what makes the text-view assertion possible: a
    `"tokens"` view would arrive as re-joined subword pieces (or fail the
    `Utf8` cast outright), never as the original name.
    """

    _VECTORS: ClassVar[dict[str, list[float]]] = {
        "acme limited": [0.98, 0.02],
        "acme ltd": [1.0, 0.0],
        "beta holdings": [0.05, 0.999],
        "omega plc": [0.0, 1.0],
    }

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def fit(self, texts: list[str]) -> "_RecordingFixedVectorEncoder":
        _ = texts
        return self

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        self.seen_texts.extend(texts)
        return np.array([self._VECTORS[text] for text in texts], dtype=np.float64)


@pytest.mark.integration
def test_run_validation_matrix_sbert_scores_on_the_name_text_view(
    workspace_roots: WorkspaceRoots, mocker
) -> None:
    """`sbert` encodes whole names, not the re-joined `cluster_tokens` view
    it was once forced onto.

    Covers the `run_validation_matrix` call site of
    `resolve_text_view_for_representation()`; `src/tests/blocking/
    test_workflow.py` covers the `execute_blocking_run` one.
    """
    run_date = "2026-07-07"
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
            {
                "system_uri": "gleif:2",
                "name_cleansed": "beta holdings",
                "jurisdiction_code": "gb",
                "match_uri": None,
            },
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            },
            {
                "system_uri": "gb:2",
                "name_cleansed": "omega plc",
                "jurisdiction_code": "gb",
            },
        ],
    )

    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date=run_date,
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=("gb",),
        name_col="name_cleansed",
        representation="sbert",
        similarity="cosine",
        clustering="knn_cc",
        text_view=None,
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=validation_artifact_root(workspace_roots),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
    )
    validate_run_config(config)

    encoder = _RecordingFixedVectorEncoder()
    mocker.patch(
        "validation.runner.resolve_clustering_strategy",
        return_value=SbertClusteringStrategy(encoder_factory=lambda _: encoder),
    )

    results = run_validation_matrix(config)

    # Whole names, not re-joined subword tokens: the encoder never saw
    # anything but the values written into the cleansed fixture.
    assert set(encoder.seen_texts) == {
        "acme limited",
        "beta holdings",
        "acme ltd",
        "omega plc",
    }
    outcomes = results["source_outcomes"]
    assert (
        outcomes.filter(
            (pl.col("source_id") == "gleif:1") & (pl.col("target_id") == "gb:1")
        ).height
        >= 1
    )


@pytest.mark.integration
def test_run_validation_matrix_sbert_second_run_encodes_nothing_on_the_target(
    workspace_roots: WorkspaceRoots, mocker
) -> None:
    """A second S-BERT run against the same target under the same model
    encodes nothing on the target side.

    `sentence-transformers` is optional and not installed here (importing it
    inside pytest crashes the process on this machine), so the encoder is injected the same way
    `test_run_validation_matrix_sbert_scores_on_the_name_text_view` injects
    it.
    """
    run_date = "2026-07-07"
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
            }
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            }
        ],
    )

    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date=run_date,
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=("gb",),
        name_col="name_cleansed",
        representation="sbert",
        similarity="cosine",
        clustering="knn_cc",
        text_view=None,
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=validation_artifact_root(workspace_roots),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
    )
    validate_run_config(config)

    encoder = _RecordingFixedVectorEncoder()
    mocker.patch(
        "validation.runner.resolve_clustering_strategy",
        return_value=SbertClusteringStrategy(encoder_factory=lambda _: encoder),
    )

    run_validation_matrix(config)
    assert set(encoder.seen_texts) == {"acme limited", "acme ltd"}

    encoder.seen_texts.clear()
    second_results = run_validation_matrix(config)

    # The source side is scored fresh every run ("acme limited" reappears);
    # the target side -- "acme ltd", the one row `gb`'s target index holds --
    # is the one this item's cache covers, and is never re-encoded.
    assert "acme ltd" not in encoder.seen_texts
    assert second_results["source_outcomes"].height >= 1


def test_run_validation_matrix_rejects_sbert_with_tokens_text_view(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=("gb",),
        name_col="name_cleansed",
        representation="sbert",
        similarity="cosine",
        clustering="knn_cc",
        text_view="tokens",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
    )

    with pytest.raises(ValueError, match="requires text_view='name'"):
        validate_run_config(config)


@pytest.mark.integration
def test_run_validation_matrix_reuses_target_index_cache(
    workspace_roots: WorkspaceRoots, mocker
) -> None:
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
            }
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            }
        ],
    )

    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-08",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=("gb",),
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=validation_artifact_root(workspace_roots),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )
    validate_run_config(config)

    import validation.runner as runner_module

    original_strategy = runner_module.resolve_clustering_strategy("tfidf")
    build_target_index_spy = mocker.spy(original_strategy, "build_target_index")
    resolve_clustering_strategy = mocker.patch.object(
        runner_module,
        "resolve_clustering_strategy",
        return_value=original_strategy,
    )

    first_results = run_validation_matrix(config)
    assert first_results["source_outcomes"].height >= 1
    resolve_clustering_strategy.assert_called_with("tfidf")
    assert build_target_index_spy.call_count == 1

    # The cache lives under `artifacts/store`, not `data/`'s
    # cleansed layer -- a worktree must never write beneath `data/`.
    cache_root = (
        artifact_store_root(workspace_roots) / "target-index" / "tfidf" / "gb" / "gb"
    )
    assert cache_root.exists()
    cache_dirs = [child for child in cache_root.iterdir() if child.is_dir()]
    assert len(cache_dirs) == 1

    build_target_index_spy.reset_mock()
    second_results = run_validation_matrix(config)
    assert second_results["source_outcomes"].height >= 1
    assert build_target_index_spy.call_count == 0


@pytest.mark.integration
def test_run_validation_matrix_tokens_view_generates_tokens_in_stage(
    workspace_roots: WorkspaceRoots, mocker
) -> None:
    _write_cleansed_chunk(
        workspace_roots,
        "gleif",
        [
            {
                "system_uri": "gleif:1",
                "name_cleansed": "acme limited",
                "jurisdiction_code": "gb",
            }
        ],
    )
    _write_cleansed_chunk(
        workspace_roots,
        "gb",
        [
            {
                "system_uri": "gb:1",
                "name_cleansed": "acme ltd",
                "jurisdiction_code": "gb",
            }
        ],
    )

    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-08",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=("gb",),
        name_col="name_cleansed",
        representation="wordpiece",
        similarity="cosine",
        clustering="knn_cc",
        text_view="tokens",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        output_dir=validation_artifact_root(workspace_roots),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )
    validate_run_config(config)

    import validation.runner as runner_module

    def _fake_build_validation_text_frame(
        *, config, frame, stats, system, target_system, country
    ):
        tokenized = frame.with_columns(
            pl.col("name")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.split(" ")
            .alias("validation_tokens")
        )
        return tokenized, "fake-token-cache", False

    build_validation_text_frame = mocker.patch.object(
        runner_module,
        "_build_validation_text_frame",
        side_effect=_fake_build_validation_text_frame,
    )

    results = run_validation_matrix(config)

    assert results["source_outcomes"].height >= 1
    assert results["clusters"].height >= 2
    assert build_validation_text_frame.call_count >= 2
    # Both sides read the target's tokenizer: `gleif` tokens split by
    # `gleif`'s own vocabulary are not comparable with `gb`'s.
    calls = build_validation_text_frame.call_args_list
    assert {call.kwargs["system"] for call in calls} == {"gleif", "gb"}
    assert {call.kwargs["target_system"] for call in calls} == {"gb"}
