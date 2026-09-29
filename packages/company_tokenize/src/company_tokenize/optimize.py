"""How an optimize run judges tokenizer candidates, and what it records about them.

A candidate is one `(vocab_size_requested, min_frequency)` pair trained on one seed.
It is judged in two steps:

1. Safety gates reject it when its unknown-token rate is above the threshold or its
   fertility is outside `[fertility_min, fertility_max]`. Token-count median and p95
   are reported for every candidate but gate nothing (`classify_candidate_rejection`).
2. A candidate that passes gets one normalised selection score, lower being better,
   combining fertility distance to target, unknown-token rate, absolute fertility
   drift between train and validation, and any rise in unknown-token rate between
   them.

`select_best_pair` picks the pair with the lowest median score among those meeting the
eligibility pass rate, and prefers the smaller effective vocabulary among pairs
within the vocab-size distance tolerance of the best.

A run log accumulates every candidate's result. `build_optimize_canary_payload()`
fingerprints a run's configuration, and a candidate already in the log under a
matching fingerprint is skipped rather than trained again (`candidate_key`,
`is_candidate_excluded`). `compute_optimize_grid_hash()` hashes the fields that tell
sibling runs over one corpus apart: the vocab schedule, minimum frequencies, seeds,
policy and tolerances, and SentencePiece's character coverage and byte fallback.

The refining helpers (`derive_refine_step_vocab`, `interpolate_fertility`,
`predict_fertility_crossing`, `evaluate_refined_points_against_fit`) model fertility
across the winning bracket, linearly in log vocab, to place refined vocab points
where fertility crosses its target.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

from .contracts import OptimizeCandidatePolicy
from .paths import to_portable_path_str
from .training import TrainerOptions, load_tokenizer_encoder, trainer_params_payload


def classify_candidate_rejection(
    *,
    metrics: dict[str, float],
    unk_rate_threshold: float,
    fertility_tolerance: float,
    fertility_min: float,
    fertility_max: float,
    token_count_median_max: float,
    token_count_p95_max: float,
) -> tuple[bool, str]:
    """Return safety-gate rejection decision and reason string for optimize metrics.

    `token_count_median_max` and `token_count_p95_max` are accepted for
    call-site compatibility (they are still fields on the policy callers pass
    through) but are not checked. `token_count_p95` is a step function over
    the vocabulary parameter it would gate, not a continuous quality signal,
    so what a cap on it costs depends on where its step happens to fall
    relative to the fertility elbow rather than on any candidate property.
    `token_count_median` is tokens per name, so its floor is the corpus's
    words per name: `offeneregister`'s names have a median of 4 words, and a
    cap of 4 rejected every candidate near the fertility target. Both are
    still reported per candidate (see `evaluate_tokenizer_metrics`).
    """
    rejection_reasons: list[str] = []
    if metrics["unk_rate"] > unk_rate_threshold:
        rejection_reasons.append("unk_rate")
    if metrics["fertility"] < fertility_min:
        rejection_reasons.append("fertility_too_low")
    if metrics["fertility"] > fertility_max:
        rejection_reasons.append("fertility_too_high")

    rejected = len(rejection_reasons) > 0
    return rejected, ";".join(rejection_reasons)


def compute_candidate_selection_score(
    *,
    metrics: dict[str, float],
    fertility_distance_for_selection: float,
    fertility_tolerance: float,
    unk_rate_threshold: float,
    token_count_median_max: float,
    token_count_p95_max: float,
    fertility_generalization_delta: float,
    unk_rate_generalization_delta: float,
) -> float:
    """Compute a normalized, human-readable selection score for candidate ranking.

    Lower is better. The score combines quality and stability terms:
    - fertility distance to target (see `fertility_distance_for_selection` below)
    - unknown-token rate (held-out/validation -- a real OOV risk signal)
    - absolute fertility train/validation drift
    - positive unknown-rate drift (validation minus train)

    Neither token count contributes a term, for the reasons
    `classify_candidate_rejection` gives, so `token_count_median_max` and
    `token_count_p95_max` are accepted here only for call-site compatibility.

    `fertility_distance_for_selection` is passed separately from `metrics`
    (rather than read off `metrics["fertility_distance"]`) because the two
    fertility_distance readings a candidate produces -- one from its train
    split, one from its held-out validation split -- measure genuinely
    different things, and only one of them matches what `--finalize` later
    promotes. TRAIN's is the right one: see
    `src/training/optimize_search_methodology.md` Part 5. Callers pass
    `train_metrics["fertility_distance"]` here in production; `unk_rate`
    deliberately stays sourced from held-out `metrics` (validation), since
    OOV risk and generalization stability are genuinely about how the
    candidate handles data its vocabulary didn't see, unlike the fertility
    target crossing point.
    """
    fert_scale = max(float(fertility_tolerance), 1e-9)
    unk_scale = max(float(unk_rate_threshold), 1e-9)

    return (
        (float(fertility_distance_for_selection) / fert_scale)
        + (float(metrics["unk_rate"]) / unk_scale)
        + (abs(float(fertility_generalization_delta)) / fert_scale)
        + (max(0.0, float(unk_rate_generalization_delta)) / unk_scale)
    )


def split_system_from_uri(system_uri: str | None) -> str:
    text = (system_uri or "").strip().lower()
    if not text:
        return "unknown"
    if "://" in text:
        return text.split("://", 1)[0]
    if "/" in text:
        return text.split("/", 1)[0]
    return text


def write_seed_split_artifacts(
    *,
    corpus_path: Path,
    optimize_dir: Path,
    seed: int,
    validation_fraction: float,
) -> tuple[Path, Path, int, int]:
    # Polars' built-in hash properly mixes seed and row index; a hand-rolled
    # linear formula here previously used the classic weak ANSI-C rand() LCG
    # constants (a=1103515245, c=12345, m=2^31) with the seed folded in as a
    # pure additive offset. Nearby seeds under that scheme produced nearly
    # identical splits (measured: seeds 42/43 on a real corpus overlapped
    # 76% in validation-set membership, vs. an expected ~20% under
    # independence) while other seed gaps landed in disjoint regions (0%
    # overlap) -- neither is genuine random sampling. That silently
    # undermined the whole point of evaluating multiple seeds: they weren't
    # independent draws, so seed-averaged fertility-distance estimates were
    # far noisier/less trustworthy than the multi-seed design assumes.
    # hash()-based split resolution below matches independence within noise
    # (measured ~20.0% overlap across several seed pairs after this fix).
    split_resolution = 1_000_000
    base_expr = (pl.col("row_nr").hash(seed=seed) % split_resolution).cast(
        pl.Float64
    ) / float(split_resolution)

    base = (
        pl.scan_parquet(corpus_path)
        .select(["system_uri", "name"])
        .with_row_index("row_nr")
        .with_columns(base_expr.alias("_split_rand"))
    )

    train_path = optimize_dir / f"train_seed_{seed}.parquet"
    validation_path = optimize_dir / f"validation_seed_{seed}.parquet"

    train_df = (
        base.filter(pl.col("_split_rand") >= validation_fraction)
        .select(["system_uri", "name"])
        .collect()
    )
    validation_df = (
        base.filter(pl.col("_split_rand") < validation_fraction)
        .select(["system_uri", "name"])
        .collect()
    )

    if validation_df.height == 0:
        raise ValueError(
            "Validation split is empty; adjust seed or validation fraction."
        )
    if train_df.height == 0:
        raise ValueError("Train split is empty; adjust seed or validation fraction.")

    train_df.write_parquet(train_path)
    validation_df.write_parquet(validation_path)
    return train_path, validation_path, train_df.height, validation_df.height


def evaluate_tokenizer_metrics(
    *,
    corpus_path: Path,
    tokenizer_path: Path,
    trainer: str,
    fertility_target: float,
    diagnostic_top_n: int,
) -> tuple[
    dict[str, float],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    def _token_character_length(token: str) -> int:
        # Normalize common subword markers so length reflects actual character payload.
        if token.startswith("##"):
            return len(token[2:])
        if token.startswith("▁"):
            return len(token[1:])
        return len(token)

    def _fertility_of(row: dict[str, object]) -> float:
        value = row["fertility"]
        return float(value) if isinstance(value, (int, float)) else 0.0

    encode_tokens, unk_token = load_tokenizer_encoder(
        tokenizer_path=tokenizer_path, trainer=trainer
    )
    df = pl.read_parquet(corpus_path, columns=["system_uri", "name"])

    total_tokens = 0
    total_unk_tokens = 0
    total_single_char_tokens = 0
    fertility_values: list[float] = []
    token_counts: list[int] = []

    per_system: dict[str, dict[str, float]] = {}
    high_fertility_rows: list[dict[str, object]] = []
    unk_rows: list[dict[str, object]] = []

    for row in df.iter_rows(named=True):
        name_raw = row.get("name")
        name = "" if name_raw is None else str(name_raw).strip()
        if not name:
            continue

        system_uri = "" if row.get("system_uri") is None else str(row.get("system_uri"))
        system_key = split_system_from_uri(system_uri)

        tokens = encode_tokens(name)
        token_count = len(tokens)
        if token_count == 0:
            continue

        word_count = max(len(name.split()), 1)
        unk_count = sum(1 for token in tokens if token == unk_token)
        single_char_token_count = sum(
            1 for token in tokens if _token_character_length(token) == 1
        )
        fertility = token_count / word_count

        fertility_values.append(fertility)
        token_counts.append(token_count)
        total_tokens += token_count
        total_unk_tokens += unk_count
        total_single_char_tokens += single_char_token_count

        bucket = per_system.setdefault(
            system_key,
            {"rows": 0, "tokens": 0, "unk_tokens": 0, "fertility_sum": 0.0},
        )
        bucket["rows"] += 1
        bucket["tokens"] += token_count
        bucket["unk_tokens"] += unk_count
        bucket["fertility_sum"] += fertility

        if len(high_fertility_rows) < diagnostic_top_n:
            high_fertility_rows.append(
                {
                    "system_uri": system_uri,
                    "name": name,
                    "fertility": fertility,
                    "token_count": token_count,
                }
            )
            high_fertility_rows.sort(key=_fertility_of, reverse=True)
        elif fertility > _fertility_of(high_fertility_rows[-1]):
            high_fertility_rows[-1] = {
                "system_uri": system_uri,
                "name": name,
                "fertility": fertility,
                "token_count": token_count,
            }
            high_fertility_rows.sort(key=_fertility_of, reverse=True)

        if unk_count > 0 and len(unk_rows) < diagnostic_top_n:
            unk_rows.append(
                {
                    "system_uri": system_uri,
                    "name": name,
                    "unk_tokens": unk_count,
                    "token_count": token_count,
                }
            )

    if not fertility_values:
        raise ValueError(
            "No evaluable validation rows with non-empty names were found."
        )

    unk_rate = (total_unk_tokens / total_tokens) if total_tokens else 1.0
    single_char_token_pct = (
        ((total_single_char_tokens / total_tokens) * 100.0) if total_tokens else 0.0
    )
    fertility_mean = sum(fertility_values) / len(fertility_values)
    fertility_distance = abs(fertility_mean - fertility_target)
    token_mean = sum(token_counts) / len(token_counts)
    sorted_tokens = sorted(token_counts)
    token_median = statistics.median(sorted_tokens)
    p95_index = min(
        len(sorted_tokens) - 1, max(0, math.ceil(len(sorted_tokens) * 0.95) - 1)
    )
    token_p95 = sorted_tokens[p95_index]

    per_system_rows: list[dict[str, object]] = []
    for system, bucket in sorted(per_system.items()):
        rows = int(bucket["rows"])
        system_token_count = int(bucket["tokens"])
        unk_tokens = int(bucket["unk_tokens"])
        fertility_sum = float(bucket["fertility_sum"])
        per_system_rows.append(
            {
                "system": system,
                "rows": rows,
                "unk_rate": (unk_tokens / system_token_count)
                if system_token_count
                else 0.0,
                "fertility": (fertility_sum / rows) if rows else 0.0,
            }
        )

    metrics = {
        "rows": float(len(fertility_values)),
        "unk_rate": float(unk_rate),
        "single_char_token_pct": float(single_char_token_pct),
        "fertility": float(fertility_mean),
        "fertility_distance": float(fertility_distance),
        "token_count_mean": float(token_mean),
        "token_count_median": float(token_median),
        "token_count_p95": float(token_p95),
    }
    return metrics, per_system_rows, high_fertility_rows, unk_rows


def append_run_log(
    *,
    run_log_path: Path,
    run_rows: Sequence[dict[str, Any]],
    existing_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    incoming = pl.DataFrame(run_rows)
    if existing_df is not None and existing_df.width > 0 and existing_df.height > 0:
        existing = existing_df
        combined = pl.concat([existing, incoming], how="diagonal_relaxed", rechunk=True)
    elif run_log_path.exists():
        existing = pl.read_parquet(run_log_path)
        if existing.width > 0 and existing.height > 0:
            combined = pl.concat(
                [existing, incoming], how="diagonal_relaxed", rechunk=True
            )
        else:
            combined = incoming
    else:
        combined = incoming

    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(run_log_path)
    return combined


def merge_run_logs(
    *,
    run_log_dir: Path,
    merged_run_log_path: Path,
) -> pl.DataFrame:
    shards = sorted(run_log_dir.glob("*.parquet"))
    if not shards:
        merged_run_log_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pl.DataFrame()
        empty.write_parquet(merged_run_log_path)
        return empty

    frames = [
        frame
        for frame in (pl.read_parquet(shard) for shard in shards)
        if frame.width > 0 and frame.height > 0
    ]
    if not frames:
        merged_run_log_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pl.DataFrame()
        empty.write_parquet(merged_run_log_path)
        return empty

    # diagonal_relaxed (not vertical_relaxed) because shards can predate a
    # run-row schema change (e.g. new phase/expansion_round/strategy columns)
    # -- vertical_relaxed only relaxes dtypes, not column sets, and would
    # crash here the first time an old-schema shard meets a new one.
    merged = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
    merged_run_log_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(merged_run_log_path)
    return merged


def recompute_selection_scores(
    *,
    run_log_df: pl.DataFrame,
    policy: OptimizeCandidatePolicy,
) -> pl.DataFrame:
    """Recompute every row's `selection_score` from its stored raw metrics.

    Uses the *current* `compute_candidate_selection_score` formula and policy scales,
    instead of trusting whatever scalar was persisted when the row was written.

    Why this exists: the optimize canary/reuse system (`is_candidate_excluded`)
    skips retraining a candidate whose (seed, vocab, min_frequency) already
    has a row in run-log history, as long as the *config* (grid, thresholds,
    corpus) hasn't changed -- config changes are the only thing the canary
    payload can see. A change to the scoring *formula* itself (e.g. switching
    which fertility_distance drives selection, see this module's
    `compute_candidate_selection_score` docstring and
    `optimize_search_methodology.md` Part 5) leaves the config untouched, so
    the canary matches and a reused candidate's stale, pre-change
    `selection_score` would otherwise survive into `select_best_pair`
    unnoticed. Recomputing here decouples that: expensive model retraining
    stays reusable via the canary, while cheap ranking always reflects
    current code. A row from before `train_fertility_distance` existed falls
    back to its own (validation-split) `fertility_distance` rather than
    erroring, so old run logs don't become unreadable.
    """
    required_columns = {
        "fertility_distance",
        "unk_rate",
        "token_count_median",
        "token_count_p95",
        "fertility_generalization_delta",
        "unk_rate_generalization_delta",
    }
    if run_log_df.height == 0 or not required_columns.issubset(set(run_log_df.columns)):
        return run_log_df

    has_train_distance = "train_fertility_distance" in run_log_df.columns
    recomputed_scores: list[float] = []
    for row in run_log_df.iter_rows(named=True):
        train_distance = (
            row.get("train_fertility_distance") if has_train_distance else None
        )
        fertility_distance_for_selection = (
            float(train_distance)
            if train_distance is not None
            else float(row["fertility_distance"])
        )
        recomputed_scores.append(
            compute_candidate_selection_score(
                metrics={
                    "unk_rate": row["unk_rate"],
                    "token_count_median": row["token_count_median"],
                    "token_count_p95": row["token_count_p95"],
                },
                fertility_distance_for_selection=fertility_distance_for_selection,
                fertility_tolerance=policy.fertility_tolerance,
                unk_rate_threshold=policy.unk_rate_threshold,
                token_count_median_max=policy.token_count_median_max,
                token_count_p95_max=policy.token_count_p95_max,
                fertility_generalization_delta=row["fertility_generalization_delta"],
                unk_rate_generalization_delta=row["unk_rate_generalization_delta"],
            )
        )
    return run_log_df.with_columns(pl.Series("selection_score", recomputed_scores))


def reclassify_rejections(
    *,
    run_log_df: pl.DataFrame,
    policy: OptimizeCandidatePolicy,
) -> pl.DataFrame:
    """Recompute every row's `rejected`/`rejection_reason` from its stored raw metrics.

    Uses the *current* safety-gate thresholds, instead of trusting whichever gate values
    were active when the row was written.

    Companion to `recompute_selection_scores` above, and for the same reason:
    the optimize canary excludes pure gate/scoring policy fields (safety-gate
    thresholds and scoring scales don't change what a candidate *is*, only how
    it's judged), so a threshold-only rerun reuses history whose `rejected`
    flag was baked in under the old thresholds. Without this, a candidate the
    old threshold rejected would stay excluded forever even after loosening
    the gate that rejected it, since `select_best_pair`'s eligibility filter
    reads `rejected` directly.
    """
    required_columns = {
        "unk_rate",
        "fertility",
        "token_count_median",
        "token_count_p95",
    }
    if run_log_df.height == 0 or not required_columns.issubset(set(run_log_df.columns)):
        return run_log_df

    rejected_flags: list[bool] = []
    rejection_reasons: list[str] = []
    for row in run_log_df.iter_rows(named=True):
        rejected, reason = classify_candidate_rejection(
            metrics={
                "unk_rate": row["unk_rate"],
                "fertility": row["fertility"],
                "token_count_median": row["token_count_median"],
                "token_count_p95": row["token_count_p95"],
            },
            unk_rate_threshold=policy.unk_rate_threshold,
            fertility_tolerance=policy.fertility_tolerance,
            fertility_min=policy.fertility_min,
            fertility_max=policy.fertility_max,
            token_count_median_max=policy.token_count_median_max,
            token_count_p95_max=policy.token_count_p95_max,
        )
        rejected_flags.append(rejected)
        rejection_reasons.append(reason)

    return run_log_df.with_columns(
        pl.Series("rejected", rejected_flags),
        pl.Series("rejection_reason", rejection_reasons),
    )


def select_best_pair(
    *,
    run_log_df: pl.DataFrame,
    eligibility_pass_rate: float,
    vocab_size_distance_tolerance: float,
    policy: OptimizeCandidatePolicy | None = None,
) -> dict[str, object] | None:
    if vocab_size_distance_tolerance < 0:
        raise ValueError(
            "vocab_size_distance_tolerance must be greater than or equal to zero."
        )

    if policy is not None:
        run_log_df = reclassify_rejections(run_log_df=run_log_df, policy=policy)
        run_log_df = recompute_selection_scores(run_log_df=run_log_df, policy=policy)

    # `median_fertility_distance` (validation-split) is reported for visibility
    # but no longer drives selection -- `selection_score` does, and it's built
    # from train-split fertility_distance (see compute_candidate_selection_score's
    # docstring). `median_train_fertility_distance` surfaces that actual
    # selection input alongside it. Older run logs predating this change may
    # lack `train_fertility_distance`; degrade to null rather than erroring so
    # `--reuse-existing` against old shards still works.
    train_distance_expr = (
        pl.col("train_fertility_distance").filter(pl.col("rejected").not_()).median()
        if "train_fertility_distance" in run_log_df.columns
        else pl.lit(None, dtype=pl.Float64)
    ).alias("median_train_fertility_distance")

    grouped = (
        run_log_df.group_by(["vocab_size_requested", "min_frequency"])
        .agg(
            [
                pl.len().alias("total_runs"),
                pl.col("rejected").not_().sum().alias("accepted_runs"),
                pl.col("selection_score")
                .filter(pl.col("rejected").not_())
                .median()
                .alias("median_selection_score"),
                pl.col("fertility_distance")
                .filter(pl.col("rejected").not_())
                .median()
                .alias("median_fertility_distance"),
                train_distance_expr,
                pl.col("unk_rate")
                .filter(pl.col("rejected").not_())
                .median()
                .alias("median_unk_rate"),
                pl.col("vocab_size_resolved")
                .filter(pl.col("rejected").not_())
                .median()
                .alias("median_vocab_size_resolved"),
            ]
        )
        .with_columns(
            [
                (pl.col("accepted_runs") / pl.col("total_runs")).alias("pass_rate"),
                pl.when(pl.col("vocab_size_requested") == -1)
                .then(pl.col("median_vocab_size_resolved"))
                .otherwise(pl.col("vocab_size_requested"))
                .alias("selection_vocab_size"),
            ]
        )
        .filter(
            (pl.col("accepted_runs") > 0)
            & (pl.col("pass_rate") >= eligibility_pass_rate)
            & pl.col("median_selection_score").is_not_null()
        )
    )

    if grouped.height == 0:
        return None

    min_selection_score = grouped.get_column("median_selection_score").min()
    if not isinstance(min_selection_score, (int, float)):
        raise TypeError(
            "median_selection_score column produced a non-numeric minimum value."
        )
    best_score = float(min_selection_score)
    shortlist = grouped.filter(
        pl.col("median_selection_score") <= (best_score + vocab_size_distance_tolerance)
    ).sort(
        [
            "selection_vocab_size",
            "median_selection_score",
            "median_fertility_distance",
            "median_unk_rate",
            "min_frequency",
        ]
    )

    return shortlist.row(0, named=True)


class CellFertilityDistanceSummary:
    """One cell's in-sample fertility and fertility distance, seed spread, and seed count.

    `median` and `stdev` cover `(min_frequency, vocab_requested)`'s
    seed-to-seed sample of fertility distance, with `stdev` `None` when
    fewer than two seeds exist; `fertility` and `fertility_stdev` are the
    same for in-sample fertility itself, `None` for a run log that does not
    record it. Feeds the refining phase's fertility interpolation and
    noise-derived step size (`fertility_slope_at`, `derive_refine_step_vocab`
    below, `optimize_search_methodology.md` Part 8). Noise near the crossing
    is read from `fertility_stdev`, not `stdev`: seeds either side of the
    target fold onto the same distance, so the distance's spread understates
    the noise exactly where refining looks.
    """

    __slots__ = ("fertility", "fertility_stdev", "median", "seed_count", "stdev")

    def __init__(
        self,
        *,
        median: float,
        stdev: float | None,
        seed_count: int,
        fertility: float | None = None,
        fertility_stdev: float | None = None,
    ) -> None:
        self.median = median
        self.stdev = stdev
        self.seed_count = seed_count
        self.fertility = fertility
        self.fertility_stdev = fertility_stdev


def summarize_cell_train_fertility_distance(
    *,
    run_log_df: pl.DataFrame,
    min_frequency: int,
    vocab_requested: int,
) -> CellFertilityDistanceSummary | None:
    """Summarize a grid cell's in-sample readings across its recorded seeds.

    Reads `train_fertility_distance` across whatever seeds this
    `(min_frequency, vocab_requested)` cell has on record. `median` uses
    the same passing-candidates-else-all-candidates fallback
    `_score_vocab_candidates` (`optimize_expansion_strategy.py`) applies to
    pilot data, so a cell where every seed happened to be rejected still
    yields a usable point for the fertility interpolation rather than an
    empty one. `fertility` is the median of `train_fertility` (else
    `fertility`) over the same rows.
    `stdev`/`seed_count` are `None`/`0` when fewer than two seed readings
    exist -- a standard deviation needs at least two points, and the
    refining phase's step-size derivation degrades to its floor rather than
    dividing by a missing noise estimate in that case. Returns `None` only
    when the cell has no rows at all.
    """
    if run_log_df.height == 0:
        return None
    cell = run_log_df.filter(
        (pl.col("min_frequency") == min_frequency)
        & (pl.col("vocab_size_requested") == vocab_requested)
    )
    if cell.height == 0:
        return None

    passing = cell.filter(pl.col("rejected").not_())
    scored = passing if passing.height > 0 else cell

    distance_column = (
        "train_fertility_distance"
        if "train_fertility_distance" in scored.columns
        else "fertility_distance"
    )
    scored_values = scored.get_column(distance_column).drop_nulls()
    if scored_values.len() == 0:
        return None
    median = float(scored_values.median())  # type: ignore[arg-type]

    all_values = [
        float(value)
        for value in cell.get_column(distance_column).drop_nulls().to_list()
    ]
    stdev = statistics.stdev(all_values) if len(all_values) >= 2 else None

    fertility: float | None = None
    fertility_stdev: float | None = None
    fertility_column = next(
        (
            column
            for column in ("train_fertility", "fertility")
            if column in scored.columns
        ),
        None,
    )
    if fertility_column is not None:
        fertility_values = scored.get_column(fertility_column).drop_nulls()
        if fertility_values.len() > 0:
            fertility = float(fertility_values.median())  # type: ignore[arg-type]
        all_fertilities = [
            float(value)
            for value in cell.get_column(fertility_column).drop_nulls().to_list()
        ]
        if len(all_fertilities) >= 2:
            fertility_stdev = statistics.stdev(all_fertilities)

    return CellFertilityDistanceSummary(
        median=median,
        stdev=stdev,
        seed_count=len(all_values),
        fertility=fertility,
        fertility_stdev=fertility_stdev,
    )


def compute_standard_error_of_difference(*, sigma: float, seed_count: int) -> float:
    """Standard error of the difference between two seed-count-seed means.

    `SE_diff = sigma * sqrt(2/seed_count)` -- `optimize_search_methodology.md`
    Part 3's formula, reused here as the noise unit the refining phase's
    step size and fit-match check are both expressed in.
    """
    if seed_count <= 0:
        raise ValueError("seed_count must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    return sigma * math.sqrt(2.0 / seed_count)


def derive_refine_step_vocab(
    *,
    se_diff: float,
    slope: float,
    minimum_step: int = 100,
) -> int:
    """Derive the refining phase's vocab step from noise and the fertility slope.

    Fertility distance is `|fertility - target|`, and fertility falls
    smoothly with vocabulary, so near the crossing the distance is a V:
    a candidate `d` vocab from the crossing is penalised by `|slope| * d`,
    where `slope` is fertility per unit of vocab there. The finest spacing
    worth evaluating is where that penalty matches the noise floor,
    `d = SE_diff / |slope|` (`optimize_search_methodology.md` Part 8).
    Returns `minimum_step` when `slope` is zero, since a flat line gives no
    signal for how finely to space refined points.
    """
    if se_diff < 0:
        raise ValueError("se_diff must be non-negative.")
    if minimum_step <= 0:
        raise ValueError("minimum_step must be positive.")
    if slope == 0:
        return minimum_step
    step = se_diff / abs(slope)
    return max(minimum_step, round(step))


def _fertility_segment(
    vocab_points: Sequence[int], fertilities: Sequence[float], vocab: float
) -> tuple[float, float, float, float]:
    """The bracket segment `(x0, f0, x1, f1)` in log vocab that `vocab` falls in.

    A vocab outside the bracket takes its nearest end segment, so the line
    extrapolates rather than failing.
    """
    if len(vocab_points) != len(fertilities) or len(vocab_points) < 2:
        raise ValueError(
            "Interpolating fertility needs at least two vocab points and one "
            f"fertility each; got {len(vocab_points)} vocab points and "
            f"{len(fertilities)} fertilities."
        )
    pairs = sorted(zip(vocab_points, fertilities, strict=True))
    xs = [math.log(float(point)) for point, _ in pairs]
    if len(set(xs)) != len(xs):
        raise ValueError(f"Need distinct vocab points; got {list(vocab_points)}.")
    target_x = math.log(float(vocab))
    index = 0
    while index < len(xs) - 2 and target_x > xs[index + 1]:
        index += 1
    return xs[index], float(pairs[index][1]), xs[index + 1], float(pairs[index + 1][1])


def interpolate_fertility(
    *, vocab_points: Sequence[int], fertilities: Sequence[float], vocab: float
) -> float:
    """Fertility at `vocab`, linear in log vocab between the measured bracket points."""
    x0, f0, x1, f1 = _fertility_segment(vocab_points, fertilities, vocab)
    return f0 + (f1 - f0) * (math.log(float(vocab)) - x0) / (x1 - x0)


def predict_fertility_crossing(
    *,
    vocab_points: Sequence[int],
    fertilities: Sequence[float],
    fertility_target: float,
) -> float | None:
    """The vocab where interpolated fertility crosses `fertility_target`, the V's point.

    Solved on the first bracket segment, in log vocab, whose ends straddle the
    target; `None` when fertility stays on one side of it across the bracket.
    """
    pairs = sorted(zip(vocab_points, fertilities, strict=True))
    for (v0, f0), (v1, f1) in pairwise(pairs):
        f0, f1 = float(f0), float(f1)
        if f0 == f1 or (f0 - fertility_target) * (f1 - fertility_target) > 0:
            continue
        x0, x1 = math.log(float(v0)), math.log(float(v1))
        return math.exp(x0 + (fertility_target - f0) * (x1 - x0) / (f1 - f0))
    return None


def fertility_slope_at(
    *, vocab_points: Sequence[int], fertilities: Sequence[float], vocab: float
) -> float:
    """Fertility per unit of vocab at `vocab`, from its log-vocab segment."""
    x0, f0, x1, f1 = _fertility_segment(vocab_points, fertilities, vocab)
    return (f1 - f0) / (x1 - x0) / float(vocab)


def evaluate_refined_points_against_fit(
    *,
    fit_vocab_points: Sequence[int],
    fit_fertilities: Sequence[float],
    fertility_target: float,
    refined_points: Sequence[tuple[int, float]],
    noise_threshold: float,
) -> list[tuple[int, float, bool]]:
    """Compare each refined point's measured distance to the V the bracket predicts.

    The predicted distance at a refined vocab is
    `|interpolate_fertility(...) - fertility_target|`; returns
    `(vocab, residual, matched)` per refined point, `matched` True when the
    measured distance falls within `noise_threshold` (the caller passes
    `2 * SE_diff`, `compute_standard_error_of_difference` above) of it. The
    bracket points themselves are matched exactly by construction, so only
    the interior points test the model.
    """
    results: list[tuple[int, float, bool]] = []
    for vocab, distance in refined_points:
        predicted = abs(
            interpolate_fertility(
                vocab_points=fit_vocab_points, fertilities=fit_fertilities, vocab=vocab
            )
            - fertility_target
        )
        residual = float(distance) - predicted
        results.append((vocab, residual, abs(residual) <= noise_threshold))
    return results


def scoped_history_for_trainer(
    *,
    run_log_df: pl.DataFrame,
    scope: str,
    systems_key: str,
    trainer: str,
) -> pl.DataFrame:
    if run_log_df.height == 0:
        # A genuinely fresh sweep leaf's merged run log is a bare, columnless
        # DataFrame (see `merge_run_logs`'s no-shards branch) -- nothing to
        # scope, and filtering on a column-less frame raises rather than
        # returning empty.
        return run_log_df
    scoped = run_log_df.filter(
        (pl.col("scope") == scope) & (pl.col("systems_key") == systems_key)
    )
    if "trainer" not in scoped.columns:
        return scoped
    if trainer == "wordpiece":
        return scoped.filter(
            pl.col("trainer").is_null() | (pl.col("trainer") == "wordpiece")
        )
    return scoped.filter(pl.col("trainer") == trainer)


def candidate_key(
    *, seed: int, vocab_requested: int, min_frequency: int
) -> tuple[int, int, int]:
    return int(seed), int(vocab_requested), int(min_frequency)


def is_candidate_excluded(
    *,
    seed: int,
    vocab_requested: int,
    min_frequency: int,
    excluded_candidate_keys: set[tuple[int, int, int]],
) -> bool:
    return (
        candidate_key(
            seed=seed, vocab_requested=vocab_requested, min_frequency=min_frequency
        )
        in excluded_candidate_keys
    )


def build_optimize_canary_payload(
    *,
    target_scope: str,
    systems: list[str],
    trainer: str,
    trainer_options: TrainerOptions,
    profile: str,
    vocab_schedule: list[int],
    min_frequencies: list[int],
    seeds: list[int],
    elbow_extra_steps: int,
    validation_fraction: float,
    policy: OptimizeCandidatePolicy,
    vocab_size_distance_tolerance: float,
    pilot_seed_count: int,
    vocab_frontier_top_k: int,
    vocab_frontier_distance_tolerance: float,
    corpus_path: Path,
    corpus_content_hash: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    # Deliberately excludes pure safety-gate/scoring policy fields
    # (fertility_target/tolerance/min/max, unk_rate_threshold,
    # token_count_median_max, token_count_p95_max, eligibility_pass_rate,
    # vocab_size_distance_tolerance): none of them change what gets trained
    # for a given (corpus, vocab, min_frequency, seed) -- they only change how
    # an already-trained candidate's stored metrics are judged and ranked.
    # `select_best_pair` reclassifies/rescores every cached row from those
    # stored metrics under the *current* policy on every call (see
    # `reclassify_rejections`/`recompute_selection_scores`), so a threshold-only
    # rerun stays correct without invalidating the canary and forcing a full
    # re-sweep. Only fields that actually affect what gets trained belong here.
    return {
        "scope": target_scope,
        "systems": systems,
        "profile": profile,
        "trainer": trainer,
        "trainer_params": trainer_params_payload(
            trainer=trainer, options=trainer_options
        ),
        "vocab_schedule": vocab_schedule,
        "min_frequencies": min_frequencies,
        "seeds": seeds,
        "elbow_extra_steps": elbow_extra_steps,
        "validation_fraction": validation_fraction,
        "pilot_seed_count": pilot_seed_count,
        "vocab_frontier_top_k": vocab_frontier_top_k,
        "vocab_frontier_distance_tolerance": vocab_frontier_distance_tolerance,
        "corpus_path": to_portable_path_str(corpus_path, project_root=project_root),
        "corpus_content_hash": corpus_content_hash,
    }


# Fields from `build_optimize_canary_payload` that identify *what's being
# explored*, not what data it was trained on. `scope`/`systems`/`corpus_path`
# and `trainer` are excluded because the tokenizer's own `optimize_dir`
# already fixes those, and `corpus_content_hash` gets its own, coarser
# directory level (see `resolve_optimize_sweep_paths` in `training.py`) so
# two grids against the same corpus sit side by side. `trainer_params` is
# here because nothing else in the path carries SentencePiece's character
# coverage and byte fallback; WordPiece's are all null, so its hash does not
# move with options it never reads.
_OPTIMIZE_GRID_HASH_FIELDS = (
    "profile",
    "trainer_params",
    "vocab_schedule",
    "min_frequencies",
    "seeds",
    "elbow_extra_steps",
    "validation_fraction",
    "pilot_seed_count",
    "vocab_frontier_top_k",
    "vocab_frontier_distance_tolerance",
)


def compute_optimize_grid_hash(payload: dict[str, Any]) -> str:
    """Short, stable digest of an optimize canary payload's exploration scope.

    Takes the payload as returned by `build_optimize_canary_payload`. Two payloads
    agreeing on every field in `_OPTIMIZE_GRID_HASH_FIELDS` produce the same hash
    regardless of corpus or trainer, which is exactly what's needed to key the
    innermost directory level -- corpus and tokenizer identity are already
    handled by the levels above it.

    This is the tokenizer archive's one key derivation, and stays this
    package's own rather than moving to the `src`-tier shared archive
    convention: the package owns the sweep tree the hash keys. It builds the
    payload hashed here (`build_optimize_canary_payload`), lays out the path
    (`resolve_optimize_sweep_paths`) and reads a stored `grid_hash` back
    (`candidate_archive.py`), and a package here cannot depend on
    `src/workspace`. `src/training/optimize_execution.py` calls this exact
    function rather than a second, `src`-tier canonicalisation, so every grid
    hash on disk comes from one implementation.
    """
    grid_subset = {key: payload[key] for key in _OPTIMIZE_GRID_HASH_FIELDS}
    canonical = json.dumps(grid_subset, sort_keys=True, default=str)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=4).hexdigest()


def build_optimize_summary_payload(
    *,
    trainer: str,
    target_scope: str,
    systems: list[str],
    session_id: str,
    wall_seconds: float,
    effective_elbow_extra_steps: int,
    elbow_min_points: int,
    elbow_min_improvement: float,
    best_pair: dict[str, Any],
    final_resolved_vocab: int,
    best_vocab_requested: int,
    best_min_frequency: int,
    final_tokenizer_path: Path,
    easy_result_path: Path | None,
    final_metrics: dict[str, Any],
    retrain_elapsed_seconds: float,
    per_system_rows: list[dict[str, Any]],
    high_fertility_rows: list[dict[str, Any]],
    unk_rows: list[dict[str, Any]],
    run_log_path: Path,
    summary_path: Path,
    trainer_options: TrainerOptions,
    candidate_stats: dict[str, Any],
    eligibility_pass_rate: float,
    vocab_size_distance_tolerance: float,
    unk_rate_threshold: float,
    delete_rejected_models: bool,
    fertility_target: float,
    fertility_tolerance: float,
    fertility_min: float,
    fertility_max: float,
    token_count_median_max: float,
    token_count_p95_max: float,
    completed_jobs: int,
    submitted_jobs: int,
    expansion_strategy: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    winner_candidate = {
        "vocab_size_requested": best_vocab_requested,
        "vocab_size_resolved": final_resolved_vocab,
        "min_frequency": best_min_frequency,
        "median_selection_score": float(best_pair["median_selection_score"]),
        "median_fertility_distance": float(best_pair["median_fertility_distance"]),
        "median_train_fertility_distance": (
            float(best_pair["median_train_fertility_distance"])
            if best_pair.get("median_train_fertility_distance") is not None
            else None
        ),
        "pass_rate": float(best_pair["pass_rate"]),
        "accepted_runs": int(best_pair["accepted_runs"]),
        "total_runs": int(best_pair["total_runs"]),
    }
    winner_output = {
        "tokenizer_path": to_portable_path_str(
            final_tokenizer_path, project_root=project_root
        ),
        "easy_result_path": to_portable_path_str(
            easy_result_path, project_root=project_root
        )
        if easy_result_path is not None
        else None,
        "post_retrain_metrics": final_metrics,
        "retrain_elapsed_seconds": retrain_elapsed_seconds,
        "per_system": per_system_rows,
        "diagnostics": {
            "high_fertility_rows": high_fertility_rows,
            "unk_rows": unk_rows,
        },
    }
    return {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "optimize",
        "effective_elbow_extra_steps": effective_elbow_extra_steps,
        "elbow_min_points": elbow_min_points,
        "elbow_min_improvement": elbow_min_improvement,
        "trainer": trainer,
        "scope": target_scope,
        "systems": systems,
        "session_id": session_id,
        "wall_seconds": wall_seconds,
        "history_first": True,
        "eligibility_pass_rate": eligibility_pass_rate,
        "vocab_size_distance_tolerance": vocab_size_distance_tolerance,
        "unk_rate_threshold": unk_rate_threshold,
        "delete_rejected_models": delete_rejected_models,
        "fertility_target": fertility_target,
        "fertility_tolerance": fertility_tolerance,
        "fertility_min": fertility_min,
        "fertility_max": fertility_max,
        "token_count_median_max": token_count_median_max,
        "token_count_p95_max": token_count_p95_max,
        "expansion_strategy": expansion_strategy,
        "job_totals": {
            "completed_jobs": completed_jobs,
            "submitted_jobs": submitted_jobs,
        },
        "candidate_stats": candidate_stats,
        "winner_candidate": winner_candidate,
        "winner_output": winner_output,
        "artifacts": {
            "run_log_parquet": to_portable_path_str(
                run_log_path, project_root=project_root
            ),
            "summary_path": to_portable_path_str(
                summary_path, project_root=project_root
            ),
            "tokenizer_path": to_portable_path_str(
                final_tokenizer_path, project_root=project_root
            ),
        },
        "trainer_params": trainer_params_payload(
            trainer=trainer, options=trainer_options
        ),
    }
