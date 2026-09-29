import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from company_tokenize import TrainerOptions
from company_tokenize.contracts import OptimizeCandidatePolicy
from company_tokenize.optimize import (
    build_optimize_canary_payload,
    build_optimize_summary_payload,
    classify_candidate_rejection,
    compute_candidate_selection_score,
    compute_optimize_grid_hash,
    compute_standard_error_of_difference,
    derive_refine_step_vocab,
    evaluate_refined_points_against_fit,
    evaluate_tokenizer_metrics,
    fertility_slope_at,
    interpolate_fertility,
    merge_run_logs,
    predict_fertility_crossing,
    recompute_selection_scores,
    select_best_pair,
    summarize_cell_train_fertility_distance,
    write_seed_split_artifacts,
)


def _selection_policy(**overrides: float) -> OptimizeCandidatePolicy:
    defaults: dict[str, Any] = {
        "fertility_target": 1.25,
        "unk_rate_threshold": 0.005,
        "fertility_tolerance": 0.1,
        "fertility_min": 1.05,
        "fertility_max": 1.6,
        "token_count_median_max": 4.0,
        "token_count_p95_max": 6.0,
        "eligibility_pass_rate": 0.8,
        "diagnostic_top_n": 1,
        "delete_rejected_models": False,
    }
    defaults.update(overrides)
    return OptimizeCandidatePolicy(**defaults)


def test_classify_candidate_rejection_returns_multiple_reasons_in_order():
    metrics = {
        "unk_rate": 0.01,
        "fertility": 1.8,
        "fertility_distance": 0.2,
        "token_count_median": 4.5,
        "token_count_p95": 6.5,
    }

    rejected, reason = classify_candidate_rejection(
        metrics=metrics,
        unk_rate_threshold=0.005,
        fertility_tolerance=0.1,
        fertility_min=1.05,
        fertility_max=1.6,
        token_count_median_max=4.0,
        token_count_p95_max=6.0,
    )

    assert rejected is True
    assert reason == "unk_rate;fertility_too_high"


def test_classify_candidate_rejection_does_not_reject_on_token_counts():
    # token_count_p95 and token_count_median are recorded observables, not
    # gates. A candidate whose only breaches are the two token counts (well
    # above their caps) must pass, even though both are still reported and
    # still accepted as parameters for call-site compatibility.
    metrics = {
        "unk_rate": 0.001,
        "fertility": 1.25,
        "fertility_distance": 0.05,
        "token_count_median": 5.0,
        "token_count_p95": 50.0,
    }

    rejected, reason = classify_candidate_rejection(
        metrics=metrics,
        unk_rate_threshold=0.005,
        fertility_tolerance=0.1,
        fertility_min=1.05,
        fertility_max=1.6,
        token_count_median_max=4.0,
        token_count_p95_max=6.0,
    )

    assert rejected is False
    assert reason == ""


def test_classify_candidate_rejection_accepts_clean_metrics():
    metrics = {
        "unk_rate": 0.001,
        "fertility": 1.25,
        "fertility_distance": 0.05,
        "token_count_median": 3.0,
        "token_count_p95": 5.0,
    }

    rejected, reason = classify_candidate_rejection(
        metrics=metrics,
        unk_rate_threshold=0.005,
        fertility_tolerance=0.1,
        fertility_min=1.05,
        fertility_max=1.6,
        token_count_median_max=4.0,
        token_count_p95_max=6.0,
    )

    assert rejected is False
    assert reason == ""


def test_compute_candidate_selection_score_ignores_token_counts():
    # Neither token count contributes a term to the selection score. Two
    # otherwise-identical metric sets that differ only in token_count_median
    # and token_count_p95 must score identically.
    base_kwargs: dict[str, Any] = {
        "fertility_distance_for_selection": 0.05,
        "fertility_tolerance": 0.1,
        "unk_rate_threshold": 0.005,
        "token_count_median_max": 4.0,
        "token_count_p95_max": 6.0,
        "fertility_generalization_delta": 0.0,
        "unk_rate_generalization_delta": 0.0,
    }

    low_p95_score = compute_candidate_selection_score(
        metrics={
            "unk_rate": 0.001,
            "token_count_median": 3.0,
            "token_count_p95": 5.0,
        },
        **base_kwargs,
    )
    high_p95_score = compute_candidate_selection_score(
        metrics={
            "unk_rate": 0.001,
            "token_count_median": 6.0,
            "token_count_p95": 50.0,
        },
        **base_kwargs,
    )

    assert low_p95_score == high_p95_score


def test_select_best_pair_prefers_lowest_selection_score():
    run_log_df = pl.DataFrame(
        {
            "vocab_size_requested": [50000, 50000, 100000, 100000],
            "vocab_size_resolved": [50000, 50000, 100000, 100000],
            "min_frequency": [2, 2, 2, 2],
            "rejected": [False, False, False, False],
            "selection_score": [2.10, 2.00, 2.35, 2.40],
            "fertility_distance": [0.051, 0.049, 0.041, 0.039],
            "unk_rate": [0.0002, 0.0002, 0.0001, 0.0001],
        }
    )

    best = select_best_pair(
        run_log_df=run_log_df,
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0.01,
    )

    assert best is not None
    assert int(best["vocab_size_requested"]) == 50000


def test_select_best_pair_uses_score_tolerance_then_smallest_vocab():
    run_log_df = pl.DataFrame(
        {
            "vocab_size_requested": [50000, 50000, 100000, 100000],
            "vocab_size_resolved": [50000, 50000, 100000, 100000],
            "min_frequency": [2, 2, 2, 2],
            "rejected": [False, False, False, False],
            "selection_score": [2.04, 2.06, 2.00, 2.02],
            "fertility_distance": [0.051, 0.049, 0.041, 0.039],
            "unk_rate": [0.0002, 0.0002, 0.0001, 0.0001],
        }
    )

    best = select_best_pair(
        run_log_df=run_log_df,
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0.05,
    )

    assert best is not None
    assert int(best["vocab_size_requested"]) == 50000


def test_select_best_pair_recomputes_stale_selection_score_from_raw_metrics():
    # Regression test for the reuse-staleness gap (2026-08-26): the optimize
    # canary/reuse system can skip retraining a candidate whose config hasn't
    # changed, which means a stored `selection_score` computed under an old
    # scoring formula can survive unchanged into a later run. This asserts
    # select_best_pair, given a `policy`, recomputes selection_score from
    # each row's raw metrics using the *current* formula rather than trusting
    # the stale stored scalar -- so a code-only scoring fix takes effect
    # immediately, without needing a grid/config change to invalidate history.
    run_log_df = pl.DataFrame(
        {
            "vocab_size_requested": [50000, 100000],
            "vocab_size_resolved": [50000, 100000],
            "min_frequency": [2, 2],
            "rejected": [False, False],
            # Stale scores from an old run: vocab=50000 looks like the clear
            # winner by its persisted selection_score, and by validation-split
            # fertility_distance -- but its train-split fertility_distance
            # (what the current formula actually uses) is far worse.
            "selection_score": [1.00, 5.00],
            "fertility_distance": [0.01, 0.20],
            "train_fertility_distance": [0.20, 0.01],
            "unk_rate": [0.0001, 0.0001],
            "token_count_median": [3.0, 3.0],
            "token_count_p95": [5.0, 5.0],
            "fertility_generalization_delta": [0.0, 0.0],
            "unk_rate_generalization_delta": [0.0, 0.0],
        }
    )

    best = select_best_pair(
        run_log_df=run_log_df,
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0.0,
        policy=_selection_policy(),
    )

    assert best is not None
    assert int(best["vocab_size_requested"]) == 100000


def test_select_best_pair_reclassifies_stale_rejection_from_raw_metrics():
    # Companion to test_select_best_pair_recomputes_stale_selection_score_...:
    # a candidate's `rejected` flag is baked in at training time under
    # whatever safety-gate thresholds were active then. Since the canary
    # excludes pure gate/scoring policy fields (see
    # build_optimize_canary_payload), a threshold-only rerun reuses this
    # cached row untouched -- so `rejected` must be recomputed from the row's
    # own stored raw metrics under the *current* policy, or a candidate a
    # since-loosened threshold would now accept stays excluded forever.
    run_log_df = pl.DataFrame(
        {
            "vocab_size_requested": [50000, 100000],
            "vocab_size_resolved": [50000, 100000],
            "min_frequency": [2, 2],
            # vocab=100000 was rejected under the old, stricter
            # fertility_max=1.2 gate (fertility=1.25 here). Its raw
            # metrics are otherwise clearly better than vocab=50000's.
            "rejected": [False, True],
            "rejection_reason": ["", "fertility_too_high"],
            "selection_score": [5.00, 1.00],
            "fertility_distance": [0.20, 0.01],
            "train_fertility_distance": [0.20, 0.01],
            "unk_rate": [0.0001, 0.0001],
            "fertility": [1.40, 1.25],
            "token_count_median": [3.0, 4.5],
            "token_count_p95": [5.0, 7.0],
            "fertility_generalization_delta": [0.0, 0.0],
            "unk_rate_generalization_delta": [0.0, 0.0],
        }
    )

    # Old policy (what the row was rejected under) would keep vocab=100000
    # excluded; the *current*, loosened policy passed here should reclassify
    # it as accepted and let it win purely from cached history.
    best = select_best_pair(
        run_log_df=run_log_df,
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0.0,
        policy=_selection_policy(fertility_max=1.6),
    )

    assert best is not None
    assert int(best["vocab_size_requested"]) == 100000


def test_build_optimize_canary_payload_excludes_gate_and_scoring_policy_fields(
    tmp_path: Path,
):
    # Pure safety-gate/scoring fields (fertility_target/tolerance/min/max,
    # unk_rate_threshold, token_count_median_max, token_count_p95_max,
    # eligibility_pass_rate) don't change what gets *trained* for a given
    # (corpus, vocab, min_frequency, seed) -- only how an already-trained
    # candidate's stored metrics are judged. They must stay out of the
    # canary so a threshold-only change doesn't purge and force a full
    # re-sweep; select_best_pair's reclassify/recompute keeps ranking correct
    # off cached history instead. vocab_size_distance_tolerance is likewise a
    # pure tie-break threshold with no bearing on training.
    corpus_path = (
        tmp_path / "artifacts" / "tokenizers" / "fr" / "training_corpus.parquet"
    )

    def build(**overrides: Any) -> dict:
        kwargs = {
            "target_scope": "country",
            "systems": ["fr"],
            "trainer": "wordpiece",
            "trainer_options": TrainerOptions(),
            "profile": "default",
            "vocab_schedule": [10000],
            "min_frequencies": [1],
            "seeds": [42],
            "elbow_extra_steps": 0,
            "validation_fraction": 0.2,
            "policy": _selection_policy(),
            "vocab_size_distance_tolerance": 0.01,
            "pilot_seed_count": 1,
            "vocab_frontier_top_k": 1,
            "vocab_frontier_distance_tolerance": 0.0,
            "corpus_path": corpus_path,
            "corpus_content_hash": "abc123",
            "project_root": tmp_path,
        }
        kwargs.update(overrides)
        return build_optimize_canary_payload(**kwargs)

    baseline = build()
    threshold_changed = build(
        policy=_selection_policy(token_count_p95_max=7.0),
        vocab_size_distance_tolerance=0.05,
    )
    grid_changed = build(vocab_schedule=[20000])

    assert baseline == threshold_changed
    assert baseline != grid_changed


def _optimize_canary_payload(tmp_path: Path, **overrides: Any) -> dict:
    kwargs: dict[str, Any] = {
        "target_scope": "country",
        "systems": ["fr"],
        "trainer": "wordpiece",
        "trainer_options": TrainerOptions(),
        "profile": "default",
        "vocab_schedule": [10000],
        "min_frequencies": [1],
        "seeds": [42],
        "elbow_extra_steps": 0,
        "validation_fraction": 0.2,
        "policy": _selection_policy(),
        "vocab_size_distance_tolerance": 0.01,
        "pilot_seed_count": 1,
        "vocab_frontier_top_k": 1,
        "vocab_frontier_distance_tolerance": 0.0,
        "corpus_path": tmp_path / "artifacts" / "tokenizers" / "fr" / "corpus.parquet",
        "corpus_content_hash": "abc123",
        "project_root": tmp_path,
    }
    kwargs.update(overrides)
    return build_optimize_canary_payload(**kwargs)


def test_compute_optimize_grid_hash_is_a_short_stable_hex_digest(tmp_path: Path):
    # This is the tokenizer archive's one key derivation: pinning its shape
    # (deterministic, short, hex) guards the boundary decision that keeps it
    # a single, package-owned implementation rather than two that could
    # silently drift apart.
    payload = _optimize_canary_payload(tmp_path)

    first = compute_optimize_grid_hash(payload)
    second = compute_optimize_grid_hash(payload)

    assert first == second
    assert len(first) == 8
    assert all(char in "0123456789abcdef" for char in first)


def test_compute_optimize_grid_hash_ignores_non_grid_fields(tmp_path: Path):
    # corpus/trainer identity are already handled by the directory levels
    # above the grid hash (resolve_optimize_sweep_paths), so a payload that
    # only differs in those fields must still produce the same grid hash.
    baseline = _optimize_canary_payload(tmp_path)
    different_corpus_and_trainer = _optimize_canary_payload(
        tmp_path,
        systems=["ie"],
        corpus_content_hash="def456",
        corpus_path=tmp_path / "artifacts" / "tokenizers" / "ie" / "corpus.parquet",
    )

    assert compute_optimize_grid_hash(baseline) == compute_optimize_grid_hash(
        different_corpus_and_trainer
    )


def test_compute_optimize_grid_hash_changes_with_a_grid_field(tmp_path: Path):
    baseline = _optimize_canary_payload(tmp_path)
    wider_grid = _optimize_canary_payload(tmp_path, vocab_schedule=[10000, 20000])

    assert compute_optimize_grid_hash(baseline) != compute_optimize_grid_hash(
        wider_grid
    )


def test_compute_optimize_grid_hash_carries_sentencepiece_options_only(tmp_path: Path):
    # Nothing else in a sweep's path says which character coverage or byte
    # fallback a SentencePiece sweep trained with, and WordPiece reads neither.
    covered = TrainerOptions(sp_character_coverage=0.9995, sp_byte_fallback=True)

    sentencepiece = _optimize_canary_payload(tmp_path, trainer="sentencepiece")
    sentencepiece_covered = _optimize_canary_payload(
        tmp_path, trainer="sentencepiece", trainer_options=covered
    )
    wordpiece = _optimize_canary_payload(tmp_path)
    wordpiece_covered = _optimize_canary_payload(tmp_path, trainer_options=covered)

    assert compute_optimize_grid_hash(sentencepiece) != compute_optimize_grid_hash(
        sentencepiece_covered
    )
    assert compute_optimize_grid_hash(wordpiece) == compute_optimize_grid_hash(
        wordpiece_covered
    )


def test_recompute_selection_scores_falls_back_when_train_distance_missing():
    # A row from before `train_fertility_distance` existed shouldn't error --
    # it degrades to its own validation-split fertility_distance.
    run_log_df = pl.DataFrame(
        {
            "fertility_distance": [0.03],
            "unk_rate": [0.0002],
            "token_count_median": [3.0],
            "token_count_p95": [5.0],
            "fertility_generalization_delta": [0.0],
            "unk_rate_generalization_delta": [0.0],
        }
    )

    result = recompute_selection_scores(
        run_log_df=run_log_df, policy=_selection_policy()
    )

    assert result["selection_score"][0] > 0


def _refining_run_log_row(
    *, vocab: int, min_frequency: int, train_distance: float, rejected: bool = False
) -> dict[str, Any]:
    return {
        "vocab_size_requested": vocab,
        "min_frequency": min_frequency,
        "train_fertility_distance": train_distance,
        "fertility_distance": train_distance,
        "rejected": rejected,
    }


def test_summarize_cell_train_fertility_distance_reports_median_and_stdev():
    run_log_df = pl.DataFrame(
        [
            _refining_run_log_row(vocab=25000, min_frequency=1, train_distance=0.0068),
            _refining_run_log_row(vocab=25000, min_frequency=1, train_distance=0.0072),
            _refining_run_log_row(vocab=25000, min_frequency=1, train_distance=0.0070),
        ]
    )

    summary = summarize_cell_train_fertility_distance(
        run_log_df=run_log_df, min_frequency=1, vocab_requested=25000
    )

    assert summary is not None
    assert summary.median == pytest.approx(0.0070)
    assert summary.seed_count == 3
    assert summary.stdev == pytest.approx(0.0002, abs=1e-4)


def test_summarize_cell_train_fertility_distance_falls_back_to_all_rows_when_none_pass():
    run_log_df = pl.DataFrame(
        [
            _refining_run_log_row(
                vocab=25000, min_frequency=1, train_distance=0.05, rejected=True
            ),
            _refining_run_log_row(
                vocab=25000, min_frequency=1, train_distance=0.06, rejected=True
            ),
        ]
    )

    summary = summarize_cell_train_fertility_distance(
        run_log_df=run_log_df, min_frequency=1, vocab_requested=25000
    )

    assert summary is not None
    assert summary.median == pytest.approx(0.055)


def test_summarize_cell_train_fertility_distance_returns_none_for_empty_cell():
    run_log_df = pl.DataFrame(
        [_refining_run_log_row(vocab=25000, min_frequency=1, train_distance=0.01)]
    )

    summary = summarize_cell_train_fertility_distance(
        run_log_df=run_log_df, min_frequency=1, vocab_requested=30000
    )

    assert summary is None


def test_summarize_cell_train_fertility_distance_single_seed_has_no_stdev():
    run_log_df = pl.DataFrame(
        [_refining_run_log_row(vocab=25000, min_frequency=1, train_distance=0.01)]
    )

    summary = summarize_cell_train_fertility_distance(
        run_log_df=run_log_df, min_frequency=1, vocab_requested=25000
    )

    assert summary is not None
    assert summary.stdev is None
    assert summary.seed_count == 1


def test_summarize_cell_train_fertility_distance_returns_none_for_empty_run_log():
    summary = summarize_cell_train_fertility_distance(
        run_log_df=pl.DataFrame(), min_frequency=1, vocab_requested=25000
    )

    assert summary is None


def test_summarize_cell_train_fertility_distance_returns_none_when_all_distances_null():
    run_log_df = pl.DataFrame(
        {
            "vocab_size_requested": [25000],
            "min_frequency": [1],
            "train_fertility_distance": pl.Series([None], dtype=pl.Float64),
            "rejected": [False],
        }
    )

    summary = summarize_cell_train_fertility_distance(
        run_log_df=run_log_df, min_frequency=1, vocab_requested=25000
    )

    assert summary is None


def test_compute_standard_error_of_difference_rejects_non_positive_seed_count():
    with pytest.raises(ValueError, match="seed_count must be positive"):
        compute_standard_error_of_difference(sigma=0.001, seed_count=0)


def test_compute_standard_error_of_difference_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        compute_standard_error_of_difference(sigma=-0.001, seed_count=5)


def test_derive_refine_step_vocab_rejects_negative_se_diff():
    with pytest.raises(ValueError, match="se_diff must be non-negative"):
        derive_refine_step_vocab(se_diff=-0.001, slope=-1e-6)


def test_derive_refine_step_vocab_rejects_non_positive_minimum_step():
    with pytest.raises(ValueError, match="minimum_step must be positive"):
        derive_refine_step_vocab(se_diff=0.001, slope=-1e-6, minimum_step=0)


def test_interpolate_fertility_rejects_a_single_point():
    with pytest.raises(ValueError, match="at least two"):
        interpolate_fertility(vocab_points=[1000], fertilities=[1.2], vocab=1000)


def test_compute_standard_error_of_difference_matches_methodology_doc_example():
    # optimize_search_methodology.md Part 6: sigma=0.000339, 2*SE_diff=0.00043.
    se_diff = compute_standard_error_of_difference(sigma=0.000339, seed_count=5)

    assert 2 * se_diff == pytest.approx(0.00043, abs=1e-5)


def test_derive_refine_step_vocab_is_noise_over_slope():
    # A V of slope 1e-6 fertility per vocab: noise 0.0005 is 500 vocab wide.
    step = derive_refine_step_vocab(se_diff=0.0005, slope=-1e-6, minimum_step=100)

    assert step == 500


def test_derive_refine_step_vocab_falls_back_to_floor_on_a_flat_slope():
    step = derive_refine_step_vocab(se_diff=0.0002, slope=0.0, minimum_step=100)

    assert step == 100


def test_derive_refine_step_vocab_never_goes_below_the_floor():
    # A very steep slope drives the formula's raw step near zero.
    step = derive_refine_step_vocab(se_diff=0.0002, slope=-1.0, minimum_step=250)

    assert step == 250


def test_interpolate_fertility_is_linear_in_log_vocab_between_bracket_points():
    # Midway in log vocab between 10,000 and 40,000 is 20,000.
    fertility = interpolate_fertility(
        vocab_points=[10000, 40000, 160000], fertilities=[1.4, 1.2, 1.1], vocab=20000
    )

    assert fertility == pytest.approx(1.3)


def test_fertility_slope_at_is_per_unit_vocab():
    # 0.2 of fertility over ln(4) of log vocab, read at vocab 20,000.
    slope = fertility_slope_at(
        vocab_points=[10000, 40000, 160000], fertilities=[1.4, 1.2, 1.1], vocab=20000
    )

    assert slope == pytest.approx(-0.2 / math.log(4) / 20000)


def test_predict_fertility_crossing_solves_in_log_vocab():
    # 1.4 -> 1.2 between 10,000 and 40,000 crosses 1.3 midway in log vocab.
    crossing = predict_fertility_crossing(
        vocab_points=[10000, 40000, 160000],
        fertilities=[1.4, 1.2, 1.1],
        fertility_target=1.3,
    )

    assert crossing == pytest.approx(20000)


def test_predict_fertility_crossing_is_none_when_fertility_stays_above_target():
    crossing = predict_fertility_crossing(
        vocab_points=[10000, 40000, 160000],
        fertilities=[1.4, 1.35, 1.3],
        fertility_target=1.25,
    )

    assert crossing is None


def test_interpolate_fertility_rejects_duplicate_vocab_points():
    with pytest.raises(ValueError, match="distinct"):
        interpolate_fertility(
            vocab_points=[10, 10, 20], fertilities=[1.0, 2.0, 3.0], vocab=15
        )


def test_evaluate_refined_points_against_fit_predicts_a_v_around_the_crossing():
    # Fertility 1.4 -> 1.2 -> 1.1 crosses the 1.25 target between 10,000 and
    # 40,000, at 20,000 exactly 1.3, so distance 0.05 is on the V and 0.2 is not.
    matches = evaluate_refined_points_against_fit(
        fit_vocab_points=[10000, 40000, 160000],
        fit_fertilities=[1.4, 1.2, 1.1],
        fertility_target=1.25,
        refined_points=[(20000, 0.05), (20000, 0.2)],
        noise_threshold=0.01,
    )

    on_v_vocab, on_v_residual, on_v_matched = matches[0]
    _off_v_vocab, _off_v_residual, off_v_matched = matches[1]

    assert on_v_vocab == 20000
    assert on_v_residual == pytest.approx(0.0, abs=1e-9)
    assert on_v_matched is True
    assert off_v_matched is False


def test_evaluate_tokenizer_metrics_records_single_char_token_pct(
    tmp_path, monkeypatch
):
    corpus_path = tmp_path / "validation.parquet"
    pl.DataFrame(
        {
            "system_uri": ["ie://a", "ie://b"],
            "name": ["alpha", "beta"],
        }
    ).write_parquet(corpus_path)

    def fake_load_tokenizer_encoder(*, tokenizer_path, trainer):
        def encode_tokens(name: str):
            if name == "alpha":
                return ["a", "##b", "cat"]
            return ["▁z", "bb"]

        return encode_tokens, "[UNK]"

    monkeypatch.setattr(
        "company_tokenize.optimize.load_tokenizer_encoder", fake_load_tokenizer_encoder
    )

    metrics, _, _, _ = evaluate_tokenizer_metrics(
        corpus_path=corpus_path,
        tokenizer_path=tmp_path / "tokenizer.json",
        trainer="wordpiece",
        fertility_target=1.25,
        diagnostic_top_n=5,
    )

    # Single-char tokens: a, ##b, ▁z => 3 of 5 total tokens => 60%
    assert metrics["single_char_token_pct"] == 60.0


def test_write_seed_split_artifacts_gives_independent_splits_across_seeds(
    tmp_path: Path,
):
    # Regression test for a real bug (2026-08-26): the split assignment used
    # to be a hand-rolled linear formula (classic weak ANSI-C rand() LCG
    # constants) with the seed folded in as a pure additive offset. Nearby
    # seeds produced nearly-duplicate splits (measured 76% validation-set
    # overlap between seeds 42/43 on a real corpus, vs. an expected ~20%
    # under independence), silently undermining multi-seed robustness
    # estimates. This asserts the fix (hash-based split) actually behaves
    # like independent sampling, not just that it runs without error.
    row_count = 20_000
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "system_uri": [f"ie://{i}" for i in range(row_count)],
            "name": [f"name{i}" for i in range(row_count)],
        }
    ).write_parquet(corpus_path)

    validation_fraction = 0.20
    validation_sets = {}
    for seed in (42, 43, 46, 142):
        optimize_dir = tmp_path / f"seed_{seed}"
        optimize_dir.mkdir()
        _, val_path, _, val_rows = write_seed_split_artifacts(
            corpus_path=corpus_path,
            optimize_dir=optimize_dir,
            seed=seed,
            validation_fraction=validation_fraction,
        )
        # Each seed's own validation share should be close to the requested
        # fraction (loose tolerance -- this is a sample, not an exact cut).
        assert abs(val_rows / row_count - validation_fraction) < 0.02
        validation_sets[seed] = set(pl.read_parquet(val_path)["system_uri"].to_list())

    # Overlap between any two seeds' validation sets should sit near the
    # independence baseline (fraction^2 of total rows), not near 0% or near
    # 100% -- the failure modes the old linear-offset formula produced.
    expected_overlap_fraction = validation_fraction
    for seed_a, seed_b in [(42, 43), (42, 46), (42, 142)]:
        overlap = len(validation_sets[seed_a] & validation_sets[seed_b])
        overlap_fraction = overlap / len(validation_sets[seed_a])
        assert abs(overlap_fraction - expected_overlap_fraction) < 0.03, (
            f"seeds {seed_a}/{seed_b}: overlap_fraction={overlap_fraction:.3f}, "
            f"expected ~{expected_overlap_fraction:.3f}"
        )


def test_merge_run_logs_handles_schema_evolution_across_shards(tmp_path: Path):
    # Regression test: merge_run_logs must survive a shard written before the
    # phase/expansion_round/strategy columns existed alongside a shard written
    # after -- vertical_relaxed only relaxes dtypes, not column sets, and
    # would raise here. diagonal_relaxed fills the missing columns with nulls
    # instead.
    run_log_dir = tmp_path / "run_logs"
    run_log_dir.mkdir()

    old_schema_shard = pl.DataFrame(
        {
            "session_id": ["old-session"],
            "seed": [1],
            "vocab_size_requested": [10000],
            "min_frequency": [1],
            "rejected": [False],
            "selection_score": [0.5],
        }
    )
    old_schema_shard.write_parquet(run_log_dir / "old.parquet")

    new_schema_shard = pl.DataFrame(
        {
            "session_id": ["new-session"],
            "seed": [2],
            "vocab_size_requested": [20000],
            "min_frequency": [1],
            "rejected": [False],
            "selection_score": [0.4],
            "phase": ["expansion"],
            "expansion_round": [1],
            "strategy": ["heuristic"],
        }
    )
    new_schema_shard.write_parquet(run_log_dir / "new.parquet")

    merged = merge_run_logs(
        run_log_dir=run_log_dir,
        merged_run_log_path=tmp_path / "run_log.parquet",
    )

    assert merged.height == 2
    assert set(merged.columns) >= {
        "session_id",
        "phase",
        "expansion_round",
        "strategy",
    }
    old_row = merged.filter(pl.col("session_id") == "old-session").row(0, named=True)
    assert old_row["phase"] is None
    new_row = merged.filter(pl.col("session_id") == "new-session").row(0, named=True)
    assert new_row["phase"] == "expansion"


def test_build_optimize_summary_payload_includes_job_totals_and_strategy(
    tmp_path: Path,
):
    best_pair = {
        "vocab_size_requested": 10000,
        "min_frequency": 1,
        "median_selection_score": 0.5,
        "median_fertility_distance": 0.01,
        "median_train_fertility_distance": 0.01,
        "pass_rate": 1.0,
        "accepted_runs": 5,
        "total_runs": 5,
    }

    summary = build_optimize_summary_payload(
        trainer="wordpiece",
        target_scope="country",
        systems=["fr"],
        session_id="sess-1",
        wall_seconds=1.0,
        effective_elbow_extra_steps=0,
        elbow_min_points=3,
        elbow_min_improvement=0.002,
        best_pair=best_pair,
        final_resolved_vocab=10000,
        best_vocab_requested=10000,
        best_min_frequency=1,
        final_tokenizer_path=tmp_path / "wordpiece.json",
        easy_result_path=None,
        final_metrics={},
        retrain_elapsed_seconds=1.0,
        per_system_rows=[],
        high_fertility_rows=[],
        unk_rows=[],
        run_log_path=tmp_path / "run_log.parquet",
        summary_path=tmp_path / "summary.json",
        trainer_options=TrainerOptions(),
        candidate_stats={},
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0.01,
        unk_rate_threshold=0.005,
        delete_rejected_models=True,
        fertility_target=1.25,
        fertility_tolerance=0.1,
        fertility_min=1.05,
        fertility_max=1.6,
        token_count_median_max=4.0,
        token_count_p95_max=6.0,
        completed_jobs=7,
        submitted_jobs=9,
        expansion_strategy="heuristic",
    )

    assert summary["expansion_strategy"] == "heuristic"
    assert summary["job_totals"] == {"completed_jobs": 7, "submitted_jobs": 9}


def test_build_optimize_summary_payload_writes_portable_paths_under_project_root(
    tmp_path: Path,
):
    best_pair = {
        "vocab_size_requested": 10000,
        "min_frequency": 1,
        "median_selection_score": 0.5,
        "median_fertility_distance": 0.01,
        "median_train_fertility_distance": 0.01,
        "pass_rate": 1.0,
        "accepted_runs": 5,
        "total_runs": 5,
    }
    tokenizer_dir = tmp_path / "artifacts" / "tokenizers" / "fr"
    tokenizer_dir.mkdir(parents=True)

    summary = build_optimize_summary_payload(
        trainer="wordpiece",
        target_scope="country",
        systems=["fr"],
        session_id="sess-1",
        wall_seconds=1.0,
        effective_elbow_extra_steps=0,
        elbow_min_points=3,
        elbow_min_improvement=0.002,
        best_pair=best_pair,
        final_resolved_vocab=10000,
        best_vocab_requested=10000,
        best_min_frequency=1,
        final_tokenizer_path=tokenizer_dir / "wordpiece.json",
        easy_result_path=tokenizer_dir / "wordpiece.json",
        final_metrics={},
        retrain_elapsed_seconds=1.0,
        per_system_rows=[],
        high_fertility_rows=[],
        unk_rows=[],
        run_log_path=tokenizer_dir / "optimize" / "run_log.parquet",
        summary_path=tokenizer_dir / "optimize" / "summary.json",
        trainer_options=TrainerOptions(),
        candidate_stats={},
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0.01,
        unk_rate_threshold=0.005,
        delete_rejected_models=True,
        fertility_target=1.25,
        fertility_tolerance=0.1,
        fertility_min=1.05,
        fertility_max=1.6,
        token_count_median_max=4.0,
        token_count_p95_max=6.0,
        completed_jobs=7,
        submitted_jobs=9,
        expansion_strategy="heuristic",
        project_root=tmp_path,
    )

    assert (
        summary["winner_output"]["tokenizer_path"]
        == "artifacts/tokenizers/fr/wordpiece.json"
    )
    assert (
        summary["winner_output"]["easy_result_path"]
        == "artifacts/tokenizers/fr/wordpiece.json"
    )
    assert summary["artifacts"] == {
        "run_log_parquet": "artifacts/tokenizers/fr/optimize/run_log.parquet",
        "summary_path": "artifacts/tokenizers/fr/optimize/summary.json",
        "tokenizer_path": "artifacts/tokenizers/fr/wordpiece.json",
    }


def test_build_optimize_canary_payload_writes_portable_corpus_path_under_project_root(
    tmp_path: Path,
):
    corpus_path = (
        tmp_path / "artifacts" / "tokenizers" / "fr" / "training_corpus.parquet"
    )

    payload = build_optimize_canary_payload(
        target_scope="country",
        systems=["fr"],
        trainer="wordpiece",
        trainer_options=TrainerOptions(),
        profile="default",
        vocab_schedule=[10000],
        min_frequencies=[1],
        seeds=[42],
        elbow_extra_steps=0,
        validation_fraction=0.2,
        policy=_selection_policy(),
        vocab_size_distance_tolerance=0.01,
        pilot_seed_count=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        corpus_path=corpus_path,
        corpus_content_hash="abc123",
        project_root=tmp_path,
    )

    assert payload["corpus_path"] == "artifacts/tokenizers/fr/training_corpus.parquet"


def test_default_metrics_count_every_word_of_the_name(tmp_path: Path):
    from company_tokenize.training import load_tokenizer_encoder, train_wordpiece

    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["ie:1", "ie:2"], "name": ["alpha ltd", "beta ltd"]}
    ).write_parquet(corpus_path)
    tokenizer_path = tmp_path / "model.json"
    train_wordpiece(corpus_path, tokenizer_path, vocab_size=50, show_progress=False)
    encode, _ = load_tokenizer_encoder(
        tokenizer_path=tokenizer_path, trainer="wordpiece"
    )

    metrics, _, _, _ = evaluate_tokenizer_metrics(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        trainer="wordpiece",
        fertility_target=1.25,
        diagnostic_top_n=1,
    )

    counts = [len(encode("alpha ltd")), len(encode("beta ltd"))]
    assert metrics["token_count_mean"] == sum(counts) / len(counts)
