"""The schema of every artefact a validation run writes, and the check a frame is held to.

`ARTIFACT_SCHEMAS` names each artefact's columns and types, and `validate_artifact_schema` refuses a frame that does not match. The population names `pair_truth_eval` rows carry are declared here too.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

# polars accepts a dtype class (e.g. pl.Utf8) or an instance (e.g. pl.Utf8())
# interchangeably in schema dicts; pl.DataType alone only covers the latter.
PolarsDType = type[pl.DataType] | pl.DataType

# `pair_truth_eval` holds one row per population per country. `universe` is
# every labelled source; each other population is the sources whose one truth
# pair reaches a given `name_equality` level -- the first name form at which the
# two sides are the same string, `raw`, `basic`, `cleansed` or `never` -- with
# `unknown` for a pair where a value is missing on one side and no verdict is
# possible. Every pair is judged over the *full* population on each side, so no
# pair's level depends on which rows survive an upstream split.
#
# A level is a population of sources, and so carries the whole matrix. Every
# evaluated source has ground truth and exactly one truth target, and every
# predicted pair is restricted to labelled sources, so partitioning the sources
# partitions every cell together: an `fp` belongs to the source that produced
# it. The table is the stored summary, so a reader never re-joins the per-pair
# frame for a confusion matrix at any cut, and two runs under different cleanse
# profiles show segment by segment what moved.
#
# The populations partition the universe exactly over its labelled-scoped
# cells: `labelled_sources`, `pair_universe`, `truth_pairs`, `predicted_pairs`,
# `tp`, `fp` and `fn` on the universe row are the sum of the other rows'.
# `candidate_pair_count` and `source_rows` on the universe row count every
# scored source, labelled or not, so the levels do not sum to them. A run given
# no name forms has no levels to classify and writes the universe row alone.
#
# Nothing here supports a `tn`. Ground truth names matching pairs and asserts
# nothing about any other pair, so an unlisted pair is unknown rather than a
# known non-match, and there is no population of true negatives to count.
POPULATION_UNIVERSE = "universe"

NAME_EQUALITY_POPULATIONS: tuple[str, ...] = (
    "raw",
    "basic",
    "cleansed",
    "preprocessed",
    "never",
    "unknown",
)

PAIR_TRUTH_EVAL_POPULATIONS: tuple[str, ...] = (
    POPULATION_UNIVERSE,
    *NAME_EQUALITY_POPULATIONS,
)

ARTIFACT_SCHEMAS: dict[str, dict[str, PolarsDType]] = {
    "source_outcomes": {
        "source_id": pl.Utf8,
        "source_name": pl.Utf8,
        "source_match_uri": pl.Utf8,
        "target_id": pl.Utf8,
        "target_name": pl.Utf8,
        "similarity": pl.Float64,
        "rank": pl.Int32,
        "is_true_match": pl.Boolean,
        "true_match_status": pl.Utf8,
        "match_status": pl.Utf8,
        "reason_code": pl.Utf8,
    },
    "clusters": {
        "cluster_id": pl.Utf8,
        "node_id": pl.Utf8,
        "node_name": pl.Utf8,
        "node_role": pl.Utf8,
    },
    "directional_coverage": {
        "country": pl.Utf8,
        "source_records_total": pl.Int64,
        "source_records_filtered_out": pl.Int64,
        "source_records": pl.Int64,
        "source_records_with_cluster": pl.Int64,
        "source_records_clustered_with_target": pl.Int64,
        "directional_coverage_ratio": pl.Float64,
    },
    "cluster_shape": {
        "country": pl.Utf8,
        "total_clusters": pl.Int64,
        "singleton_clusters": pl.Int64,
        "singleton_ratio": pl.Float64,
        "mean_cluster_size": pl.Float64,
        "p95_cluster_size": pl.Float64,
        "max_cluster_size": pl.Int64,
    },
    "pair_truth_eval": {
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        # One of `PAIR_TRUTH_EVAL_POPULATIONS`; see the note above them.
        "population": pl.Utf8,
        "labelled_sources": pl.Int64,
        "target_rows": pl.Int64,
        "pair_universe": pl.Int64,
        "truth_pairs": pl.Int64,
        "predicted_pairs": pl.Int64,
        "tp": pl.Int64,
        # A predicted pair absent from ground truth: false only under a
        # closed-world reading of the labelled sources, since ground truth
        # records matches and never records a non-match.
        "fp": pl.Int64,
        "fn": pl.Int64,
        "precision": pl.Float64,
        "recall": pl.Float64,
        # No stored F score. With tp, fp and fn present any beta is one
        # division at read time, so storing one is storing a choice -- and a
        # stored `f_beta` carried no beta, which made two runs weighted
        # differently compare as though they measured the same thing.
        "reduction_ratio": pl.Float64,
        # candidate_pair_count/source_rows/candidate_set_size_ratio/
        # recall_at_k -- see runner.compute_pair_truth_eval()'s own
        # docstring for what each measures and why they are independent of
        # (not a restatement of) reduction_ratio/recall above.
        "candidate_pair_count": pl.Int64,
        "source_rows": pl.Int64,
        "candidate_set_size_ratio": pl.Float64,
        "recall_at_k": pl.Float64,
    },
    "exceptions": {
        "run_id": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "rule_id": pl.Utf8,
        "node_id": pl.Utf8,
        "included_in_exception": pl.Boolean,
        "reason": pl.Utf8,
    },
    "run_metrics": {
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "source_rows": pl.Int64,
        "elapsed_seconds": pl.Float64,
        "latency_seconds_per_1k_rows": pl.Float64,
        "candidate_edges": pl.Int64,
        "candidate_distinct_targets": pl.Int64,
        "candidate_duplication_ratio": pl.Float64,
    },
    "generation_report": {
        "profile_id": pl.Utf8,
        "scenario_id": pl.Utf8,
        "country": pl.Utf8,
        "system": pl.Utf8,
        "records_evaluated": pl.Int64,
        "records_emitted": pl.Int64,
        "records_no_op": pl.Int64,
        "records_excluded": pl.Int64,
    },
    # The v1 robustness metric contract: one row per (profile_id,
    # profile_version, scenario_id, intensity, country), produced by
    # `runner.compute_robustness_eval()` from a perturbed source slice's own
    # `pair_truth_eval` recall against `baseline_recall` (recall of the same
    # target under a genuinely unperturbed run, supplied by the caller --
    # this contract makes no claim about how that baseline is established).
    # `recall_retention_ratio = recall / baseline_recall`: higher is better,
    # `1.0` means no degradation, and it is null whenever `baseline_recall`
    # is null or <= 0.0 rather than raising or silently reporting `0.0`. The
    # pass/fail threshold that turns this ratio into a promotion decision is
    # deliberately not part of this contract -- it belongs to whatever makes
    # the promotion decision.
    #
    # `dataset_snapshot_manifest_path` (the contract's provenance closure): a
    # pointer, not a copy -- the repo-relative, POSIX-separated path to the
    # `_materialization_manifest_<source_system>.json` file
    # `perturbation_materializer.
    # write_materialization_manifest` already writes at materialization time,
    # composing `company_perturbation`'s `RunManifest` (the resolved profile
    # that generated this row's `profile_id`/`profile_version`/`scenario_id`)
    # with `source_dataset_snapshot_fingerprint_by_country` (which canonical
    # dataset snapshot fed the run, per country). Null when the caller didn't
    # supply one (`compute_robustness_eval`'s own default), same
    # "not computed" convention the rest of this contract uses. Read it back
    # with `perturbation_materializer.load_materialization_manifest` rather
    # than re-deriving the path by hand.
    "robustness_eval": {
        "profile_id": pl.Utf8,
        "profile_version": pl.Utf8,
        "scenario_id": pl.Utf8,
        "intensity": pl.Float64,
        "country": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "labelled_sources": pl.Int64,
        "truth_pairs": pl.Int64,
        "predicted_pairs": pl.Int64,
        "tp": pl.Int64,
        "fp": pl.Int64,
        "fn": pl.Int64,
        "precision": pl.Float64,
        "recall": pl.Float64,
        "baseline_recall": pl.Float64,
        "recall_retention_ratio": pl.Float64,
        "dataset_snapshot_manifest_path": pl.Utf8,
    },
    # `recall_curve.compute_recall_curve()`'s own artifact -- one row
    # per population (the universe and each `NAME_EQUALITY_LEVELS` entry) per
    # budget step, a recall-against-comparisons-spent curve read from a
    # completed run's own `matched_edges`/`pair_truth_eval_detail`, not a
    # re-run. See `recall_curve.RECALL_CURVE_COLUMNS` for the column-by-column
    # documentation this reuses rather than restating.
    "recall_curve": {
        "population": pl.Utf8,
        "budget_step": pl.Int64,
        "comparisons_spent": pl.Int64,
        "truth_pairs": pl.Int64,
        "pairs_found": pl.Int64,
        "recall": pl.Float64,
    },
    # `recall_curve.summarize_recall_curve()`'s one-row-per-run companion:
    # the normalised area (mean recall over comparisons spent) per
    # name-equality level, `recall_area_never` the headline, beside the
    # backend a reader needs to place the figures against.
    "recall_curve_summary": {
        "similarity_backend": pl.Utf8,
        "is_exact_backend": pl.Boolean,
        "target_rows": pl.Int64,
        "min_similarity": pl.Float64,
        "top_k": pl.Int64,
        "max_candidates_per_source": pl.Int64,
        "recall_area_raw": pl.Float64,
        "recall_area_basic": pl.Float64,
        "recall_area_cleansed": pl.Float64,
        "recall_area_preprocessed": pl.Float64,
        "recall_area_never": pl.Float64,
    },
    # `runner.rollup_robustness_eval()`'s aggregate over "robustness_eval":
    # one row per (profile_id, profile_version, scenario_id, intensity),
    # reporting mean/std of `recall_retention_ratio` across whatever
    # finer-grained rows feed the group -- countries today, and draws once
    # the perturbed `system_uri` disambiguates them.
    # `recall_retention_ratio_std` is null (not `0.0`) when only one row
    # feeds a group -- spread is undefined for a single sample.
    "robustness_eval_rollup": {
        "profile_id": pl.Utf8,
        "profile_version": pl.Utf8,
        "scenario_id": pl.Utf8,
        "intensity": pl.Float64,
        "sample_count": pl.Int64,
        "recall_retention_ratio_mean": pl.Float64,
        "recall_retention_ratio_std": pl.Float64,
    },
}


def validate_schema_against_registry(
    frame: pl.DataFrame,
    *,
    artifact_name: str,
    schemas: Mapping[str, Mapping[str, PolarsDType]],
    require_exact_columns: bool = False,
) -> None:
    """Check `frame` against `schemas[artifact_name]`, raising on any mismatch.

    Registry-agnostic so each package keeps its own artifact registry and its
    own named entry point (`validate_artifact_schema` here,
    `blocking.contracts.validate_blocking_artifact_schema`) while the checks
    and their error wording stay in one place.
    """
    if artifact_name not in schemas:
        allowed = ", ".join(sorted(schemas))
        raise ValueError(
            f"Unknown artifact_name '{artifact_name}'. Expected one of: {allowed}"
        )

    expected = schemas[artifact_name]
    present = set(frame.columns)
    missing = [column for column in expected if column not in present]
    if missing:
        raise ValueError(
            f"{artifact_name} is missing required columns: {', '.join(missing)}"
        )

    if require_exact_columns:
        extras = sorted(present - set(expected.keys()))
        if extras:
            raise ValueError(
                f"{artifact_name} contains unexpected columns: {', '.join(extras)}"
            )


def validate_artifact_schema(
    frame: pl.DataFrame,
    *,
    artifact_name: str,
    require_exact_columns: bool = False,
) -> None:
    validate_schema_against_registry(
        frame,
        artifact_name=artifact_name,
        schemas=ARTIFACT_SCHEMAS,
        require_exact_columns=require_exact_columns,
    )
