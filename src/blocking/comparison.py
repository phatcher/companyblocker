"""Compare finished blocking runs: strategies side by side, and pair by pair.

`build_strategy_comparison()` reshapes two or more runs (`StrategyRunEntry`) into one
frame, a row per `(label, stage, country, population)`, where population is the
universe or one name-equality level, `never` being the residual that exact joins
cannot supply. Each row carries precision, recall, reduction ratio, candidate volume
and recall at k, whether the backend is exact, the target's row count, the run's own
phase timings and peak memory, and the axes a run varied on: `representation`,
`similarity_backend`, `tokenizer`, `name_transform` and `accelerator_settings`. The last
is `backend_options` plus `exact_name_filter` serialized canonically as one string,
since the option set is per backend and owned by `company_vectorize`; settings that
change what is measured rather than its cost are left out of it. Two runs that
disagree on population or truth key for a country they both scored are refused unless
`allow_mixed_population=True`, since a blended row would average over different pairs.

Across pairings, `combine_strategy_comparisons()` concatenates reports and
`summarize_runtime_scaling()` gives runtime, precision and recall per method bucketed
by target size, reducing to one row per run leg first, since `runtime_seconds` is
repeated onto every row a run emits. `check_strategy_comparison_repeatability()`
compares two reports of one configuration column by column and returns the matches
rather than asserting them.

Pair by pair:

- `diff_pair_recovery()`: the truth pairs one run found that another of the same
  pairing missed, so a recovery rate is counted from named pairs.
- `diff_name_equality_levels()` and `tally_name_equality_transitions()`: the truth
  pairs whose name-equality level differs between two cleanse profiles, and whether
  a pair one profile left at `never` was found anyway or only made free by cleansing.
- `build_pair_outcomes()` and its siblings: every run's verdict on every truth pair,
  the runs' settings, and each run's kept candidates counted by similarity and rank,
  per run and per source, so a run can be read at a tighter cutoff, at a cap per
  source or at an equal budget of candidates. A run is only ever read more tightly
  than it was made.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from validation.contracts import POPULATION_UNIVERSE
from validation.recall_curve import compute_recall_curve
from validation.runner import NAME_EQUALITY_NEVER
from workspace.identity import RunKeys, digest_settings

from .contracts import (
    BLOCKING_ARTIFACT_SCHEMAS,
    STRATEGY_COMPARISON_BACKFILLABLE_COLUMNS,
    BlockingRunConfig,
    BlockingRunResult,
    BlockingStrategyConfig,
    validate_blocking_artifact_schema,
)

_STRATEGY_COMPARISON_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["strategy_comparison"]
_COMBINED_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["strategy_comparison_combined"]
_RUNTIME_SCALING_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS[
    "strategy_comparison_runtime_scaling"
]
_PAIR_RECOVERY_ATTRIBUTION_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS[
    "pair_recovery_attribution"
]
_NAME_EQUALITY_ATTRIBUTION_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS[
    "name_equality_attribution"
]
_NAME_EQUALITY_TRANSITIONS_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS[
    "name_equality_transitions"
]
_PAIR_OUTCOME_RUNS_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["pair_outcome_runs"]
_PAIR_OUTCOMES_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["pair_outcomes"]
_PAIR_OUTCOME_CANDIDATES_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["pair_outcome_candidates"]
_PAIR_OUTCOME_SOURCE_CANDIDATES_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS[
    "pair_outcome_source_candidates"
]
_PAIR_OUTCOME_RECALL_CURVES_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS[
    "pair_outcome_recall_curves"
]

# The step of similarity a run's candidates are counted at in
# `build_pair_outcome_candidates()`.
_CANDIDATE_CUTOFF_STEPS_PER_UNIT = 1000

# `never_attribution`'s two values, for a truth pair one of two runs left in
# the `never` name-equality level: that run found it anyway, so cleanse only
# made it free, or that run missed it, so cleanse is what recovered it.
NEVER_SOLVED_BY_ALGORITHM = "solved_by_algorithm"
NEVER_RECOVERED_BY_CLEANSE = "recovered_by_cleanse"

# The key one truth pair is identified by across two labelled runs'
# `pair_truth_eval_detail` frames -- the same pairing (source_system,
# target_system, country) scored twice, so the pair identity itself is just
# (source_id, target_id).
_PAIR_RECOVERY_KEY: tuple[str, ...] = (
    "source_system",
    "target_system",
    "country",
    "source_id",
    "target_id",
)

# What identifies one measured run leg inside the combined table. `runtime_seconds`
# is repeated onto every row a run produced -- one per population per country,
# times the pruned/raw stage pair -- so aggregating runtime over rows would weight
# a run by how many rows it happened to emit. The `"pruned"` stage's universe row
# gives exactly one row per country per run, and this key deduplicates the rest.
_RUN_LEG_KEY: tuple[str, ...] = (
    "pair_label",
    "label",
    "representation",
    "similarity_backend",
    "accelerator_settings",
    "name_transform",
    "country",
)

# Static classification of `BlockingStrategyConfig.similarity_backend` values,
# sourced from `packages/company_vectorize/README.md`'s "Similarity backends": `sklearn` and
# `sparse_dot_topn` compute exact top-k cosine similarity; `svd_rerank` scans
# every target too (still exhaustive -- not a sub-linear index) but reranks
# against a reduced-dimension projection, which can permanently miss a true
# neighbour whose projected rank falls outside the candidate shortlist.
# `dense_brute` is exact too: a chunked exhaustive scan of a dense target
# matrix, scoring every source row against every target row like `sklearn`
# and `sparse_dot_topn` do, just a block at a time instead of holding the
# whole target resident. `hnsw` is the first genuine sub-linear ANN index in
# this repo (a `usearch` HNSW graph over a dense target), approximate by
# construction -- its own recall against `dense_brute` is what a caller
# prices, never gated. This is deliberately a fixed lookup, not a derived
# threshold: there is no scale-derived cutoff to compute here, only the
# documented exact-vs-approximate fact about each backend and the raw
# `target_rows` scale already carried by `pair_truth_eval` (README's own
# scale guidance: brute-force scanning is fine at `ie`'s ~820K rows, not at
# `fr`'s ~12.9M or `gb`'s ~5.7M). `lsh`/`kmeans`/`hdbscan` are not classified
# either way yet -- `_is_exact_backend` reports `None` for them, genuinely
# unknown rather than known to be either.
_EXACT_SIMILARITY_BACKENDS: frozenset[str] = frozenset(
    {"sklearn", "sparse_dot_topn", "dense_brute"}
)
_APPROXIMATE_SIMILARITY_BACKENDS: frozenset[str] = frozenset({"svd_rerank", "hnsw"})

# `BlockingStrategyConfig` fields that change how much work a run does without
# changing what it is nominally computing, and that do not already live in
# `backend_options`. `exact_name_filter` resolves direct `name` matches
# through a join instead of paying for the tokenize/text-view/clustering path,
# and its own field docstring names the case where two runs differ only in it
# (forcing every row through clustering to isolate a strategy's recall). Left
# out deliberately: `top_k`, `min_similarity`, the tfidf n-gram range and the
# tokenizer settings, which change what is being measured rather than the
# cost of measuring it, and so belong to the comparison's `label` rather
# than to this discriminator.
_WORKFLOW_ACCELERATOR_FIELDS: tuple[str, ...] = ("exact_name_filter",)


@dataclass(slots=True, frozen=True)
class StrategyRunEntry:
    """One completed blocking run to fold into a strategy-comparison report.

    `label` is the caller-assigned name for this run in the comparison (for
    example `"tfidf"`, `"wordpiece"`, `"sentencepiece"`) -- kept distinct
    from `config.strategy.representation` since a caller may want two runs
    of the same representation under different configs (for example two
    `top_k` values) side by side, which would collide on `representation`
    alone.

    `config` supplies the `BlockingRunConfig` that produced `result` --
    `representation`/`similarity_backend` scale-assumption metadata is read
    from it rather than duplicated onto this entry.

    `runtime_seconds` is optional and externally measured: the whole run,
    loading and writing included, which a caller comparing runs measures by
    wrapping each `execute_blocking_run()` call in its own
    `time.perf_counter()` (mirroring `scripts/run_blocking.py`'s single-run
    timing) and passes here. `BlockingRunResult.timings` carries the run's
    own per-phase records, but not a whole-run figure. Omitted (`None`) when
    no caller measured it -- the comparison still includes the run, just
    without a `runtime_seconds` value for it.

    `finished_at` is when the run's production record says it finished, as
    the record writes it, for a run read back from disk; `None` for a run made
    in this process. It is what orders a pairing's runs of one method in time.
    """

    label: str
    config: BlockingRunConfig
    result: BlockingRunResult
    runtime_seconds: float | None = None
    finished_at: str | None = None


def _encode_accelerator_settings(strategy: BlockingStrategyConfig) -> str:
    """Render a run's accelerator configuration as one canonical string.

    `key=<json value>` pairs, keys sorted, joined by `;` -- deterministic, so
    two runs configured identically produce byte-identical values and group
    together, while a `prefix_filter` run and its unpruned baseline do not.
    Sorted rather than dict-ordered because `backend_options` is assembled by
    whichever caller built the config, and insertion order there is not part
    of the configuration.

    Never empty and never null: `exact_name_filter` always has a value,
    so a run with no backend options still encodes to a non-empty string. A
    null in this column therefore means only one thing -- the row was written
    before the column existed, and that run's accelerator settings are
    genuinely unknown rather than known to be defaults.
    """

    settings: dict[str, object] = dict(strategy.backend_options or {})
    for field in _WORKFLOW_ACCELERATOR_FIELDS:
        if field in settings:
            raise ValueError(
                f"strategy.backend_options must not carry '{field}': it is read "
                "from the strategy itself, and a backend option of the same name "
                "would make the encoded accelerator settings ambiguous"
            )
        settings[field] = getattr(strategy, field)

    return ";".join(
        f"{key}={json.dumps(settings[key], sort_keys=True, default=str)}"
        for key in sorted(settings)
    )


def _encode_name_transform(strategy: BlockingStrategyConfig) -> str:
    """The transform a run compared both sides' names under, as one string,
    `<kind>:<cleanse_profile>;preprocess:<preprocess_profile>`. Both profiles
    are always part of it: every run derives its name forms under the cleanse
    profile, so it shapes an `identity` run's name-equality levels as much as
    a `cleanse` run's scored text, and preprocesses the scored text under the
    other, which the `preprocessed` level is read from. It is the run's own
    recorded `name_transform` in one string."""
    return (
        f"{strategy.name_transform}:{strategy.cleanse_profile}"
        f";preprocess:{strategy.preprocess_profile}"
    )


def _encode_tokenizer(strategy: BlockingStrategyConfig) -> str:
    """The tokenizer a run was scored with, as `<tokenizer id>:<profile>`,
    the id carrying a SentencePiece run's encoding (`sentencepiece_bpe`,
    `sentencepiece_unigram`): the profile is the label that picks one of the
    scope's stored tokenizers, so two runs of one trainer under different
    labels, or two encodings, are a tokenizer comparison too."""
    trainer = strategy.tokenizer
    if trainer == "sentencepiece":
        trainer = f"{trainer}_{strategy.tokenizer_encoding}"
    return f"{trainer}:{strategy.tokenizer_profile}"


def _is_exact_backend(similarity_backend: str) -> bool | None:
    if similarity_backend in _EXACT_SIMILARITY_BACKENDS:
        return True
    if similarity_backend in _APPROXIMATE_SIMILARITY_BACKENDS:
        return False
    return None


_PHASE_FIGURE_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "country": pl.Utf8,
    "target_index_seconds": pl.Float64,
    "scoring_seconds": pl.Float64,
    "peak_rss_bytes": pl.Int64,
}


def _phase_figures(result: BlockingRunResult) -> pl.DataFrame:
    """One row per country of `result.timings`: the target index build's and
    the scan's wall-clock, and the largest peak resident set size any of that
    country's phases recorded. A run recorded before phases sampled their peak
    has none, and reports it as null rather than its end-of-phase figures,
    which miss what a phase allocated and freed. Empty for a run that recorded
    no phases."""
    countries: list[str] = []
    target_index: dict[str, float | None] = {}
    scoring: dict[str, float | None] = {}
    peak_rss: dict[str, int | None] = {}
    for record in result.timings:
        country = record.get("label")
        if not isinstance(country, str):
            continue
        if country not in peak_rss:
            countries.append(country)
            target_index[country] = None
            scoring[country] = None
            peak_rss[country] = None
        routine = record.get("routine")
        elapsed = record.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            if routine == "target_index":
                target_index[country] = float(elapsed)
            elif routine == "scoring":
                scoring[country] = float(elapsed)
        rss = record.get("peak_rss_bytes")
        if isinstance(rss, int):
            current = peak_rss[country]
            peak_rss[country] = rss if current is None else max(current, rss)
    return pl.DataFrame(
        [
            {
                "country": country,
                "target_index_seconds": target_index[country],
                "scoring_seconds": scoring[country],
                "peak_rss_bytes": peak_rss[country],
            }
            for country in countries
        ],
        schema=_PHASE_FIGURE_SCHEMA,
    )


_POPULATION_TRUTH_KEY_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "country": pl.Utf8,
    "population_key": pl.Utf8,
    "truth_key": pl.Utf8,
}


def _population_key(keys: RunKeys) -> str:
    """Both sides' population keys digested together, one key for the pair of
    populations a run scored."""
    return digest_settings(
        {"source": keys.source_population, "target": keys.target_population}
    )


def _population_truth_key_figures(result: BlockingRunResult) -> pl.DataFrame:
    """One row per country `result` scored, each carrying the run's
    `population_key` (both sides' population keys digested together) and
    `truth_key` (`None` for a run with no ground truth). Joined into
    `_entry_rows` the same way `_phase_figures` is, read back from what the
    run recorded rather than recomputed here.
    """
    keys = result.keys
    if keys is None:
        return pl.DataFrame(schema=_POPULATION_TRUTH_KEY_SCHEMA)
    population_key = _population_key(keys)
    countries = sorted(set(result.pruning_summary.get_column("country").to_list()))
    rows = [
        {"country": country, "population_key": population_key, "truth_key": keys.truth}
        for country in countries
    ]
    return pl.DataFrame(rows, schema=_POPULATION_TRUTH_KEY_SCHEMA)


def _empty_metric_row(entry: StrategyRunEntry, *, stage: str) -> pl.DataFrame:
    row = dict.fromkeys(_STRATEGY_COMPARISON_SCHEMA, None)
    row.update(
        {
            "label": entry.label,
            "representation": entry.config.strategy.representation,
            "tokenizer": _encode_tokenizer(entry.config.strategy),
            "similarity_backend": entry.config.strategy.similarity_backend,
            "is_exact_backend": _is_exact_backend(
                entry.config.strategy.similarity_backend
            ),
            "accelerator_settings": _encode_accelerator_settings(entry.config.strategy),
            "name_transform": _encode_name_transform(entry.config.strategy),
            "runtime_seconds": entry.runtime_seconds,
            "stage": stage,
            "source_system": entry.config.source.system,
            "target_system": entry.config.target.system,
        }
    )
    return pl.DataFrame([row], schema=_STRATEGY_COMPARISON_SCHEMA)


def _entry_rows(
    entry: StrategyRunEntry, *, stage: str, eval_frame: pl.DataFrame | None
) -> pl.DataFrame:
    if eval_frame is None or eval_frame.height == 0:
        return _empty_metric_row(entry, stage=stage)

    is_exact_backend = _is_exact_backend(entry.config.strategy.similarity_backend)
    return (
        eval_frame.join(_phase_figures(entry.result), on="country", how="left")
        .join(_population_truth_key_figures(entry.result), on="country", how="left")
        .with_columns(
            pl.lit(entry.label).alias("label"),
            pl.lit(entry.config.strategy.representation).alias("representation"),
            pl.lit(_encode_tokenizer(entry.config.strategy)).alias("tokenizer"),
            pl.lit(entry.config.strategy.similarity_backend).alias(
                "similarity_backend"
            ),
            pl.lit(is_exact_backend, dtype=pl.Boolean).alias("is_exact_backend"),
            pl.lit(_encode_accelerator_settings(entry.config.strategy)).alias(
                "accelerator_settings"
            ),
            pl.lit(_encode_name_transform(entry.config.strategy)).alias(
                "name_transform"
            ),
            pl.lit(entry.runtime_seconds, dtype=pl.Float64).alias("runtime_seconds"),
            pl.lit(stage).alias("stage"),
        )
        .select(list(_STRATEGY_COMPARISON_SCHEMA.keys()))
        # Through `pl.Schema` rather than passing the dict straight in:
        # `cast` takes a `Mapping` keyed by `str | Selector | <dtype>`, and a
        # `Mapping`'s key type is invariant, so a plain `dict[str, ...]` is not
        # one of those.
        .cast(pl.Schema(_STRATEGY_COMPARISON_SCHEMA))
    )


def build_strategy_comparison(
    entries: list[StrategyRunEntry], *, allow_mixed_population: bool = False
) -> pl.DataFrame:
    """Compare two or more completed blocking runs on the same scenario.

    Generated entirely from what each run's `BlockingRunResult` already
    carries -- `validation.runner.compute_pair_truth_eval()`'s
    precision/recall/`reduction_ratio` output (via
    `pair_truth_eval`/`raw_pair_truth_eval`) plus scale-assumption metadata
    read from each entry's `BlockingRunConfig`. No new scoring logic: this
    is a side-by-side reshape over existing outputs, the same spirit as
    `inspection.summarize_cluster_metrics()` but across separate runs
    (different strategies on the same scenario) rather than within one run.

    One row per `(label, stage, country, population)`. `stage` is `"pruned"` (from
    `result.pair_truth_eval`, scored against the candidate set clustering
    would actually receive) or `"raw"` (from `result.raw_pair_truth_eval`,
    the unpruned scored candidates, showing pruning's precision/recall
    cost) -- mirrors `BlockingRunResult`'s own pruned/raw pairing rather
    than picking one and discarding the other. When a run's source dataset
    has no ground truth (`pair_truth_eval`/`raw_pair_truth_eval` both
    `None`), that run still contributes one row per stage with metric
    columns and `population` null, so it stays comparable on scale/runtime
    metadata even without scoring.

    `accelerator_settings` records the accelerator configuration each run was
    made under, canonically serialized from its `BlockingStrategyConfig` (see
    `_encode_accelerator_settings()`). It is the column that stops two runs
    differing only in an accelerator -- a `sklearn` run with `prefix_filter`
    and the unpruned baseline it is being measured against -- from being
    indistinguishable here and pooled by anything aggregating these rows;
    `representation`, `similarity_backend` and `is_exact_backend` are equal
    for both, and `label` is the caller's, so none of them separate the pair.

    `population` carries `compute_pair_truth_eval()`'s rows straight through
    from `eval_frame`: the universe and each `name_equality` level, each with
    its whole matrix. Read the `never` population, not the universe, when
    comparing labels meant to differ on something other than the trivial-match
    population (an encoder-checkpoint swap, for example): trivial exact-name
    matches dominate the universe (99.8% of `gleif -> gb`'s true positives), so
    they would mask whatever such a comparison actually changes. Select a
    population by name; the rows of one label, stage and country are otherwise
    indistinguishable but for it.

    `tokenizer` is the tokenizer candidate each run was
    scored with, read straight off `entry.config.strategy.tokenizer`
    -- a tokenizer comparison (`wordpiece` vs `sentencepiece`, or two
    archived candidates for the same trainer via a caller-supplied `label`)
    is one more labelled run in this same framework, not a separate
    promotion decision, so it needs its own column rather than being folded
    into `representation` or `label`.

    `is_exact_backend` and `target_rows` are the scale-assumption columns,
    carrying the brute-force-acceptable vs. LSH-required distinction:
    `is_exact_backend` is a fixed per-backend classification (`None` for any
    backend not yet classified), and `target_rows` is the row count each
    `compute_pair_truth_eval()` call already scored against -- read the two
    together against `packages/company_vectorize/README.md`'s documented
    scale guidance rather than a threshold computed here.

    `population_key`/`truth_key` are each entry's own run keys
    (`BlockingRunResult.keys`), rehydrated the same way every other figure
    here is. By default this function refuses to build a
    table where two entries disagree on either key for the same country --
    they scored, or were classified against, two different populations, and
    a blended row would silently average over that rather than say so. Pass
    `allow_mixed_population=True` for the "caller asks for the mixed view"
    case, as `scripts/compare_blocking_strategies.py --cross-cleanser` does:
    the columns are always present regardless, so the disagreement stays
    visible either way.

    Raises `ValueError` if `entries` is empty -- there is nothing to compare
    -- or, unless `allow_mixed_population` is set, if any two entries
    disagree on `population_key` or `truth_key` for a country both scored.
    """

    if not entries:
        raise ValueError("entries must be non-empty to build a strategy comparison")

    frames: list[pl.DataFrame] = []
    for entry in entries:
        frames.append(
            _entry_rows(entry, stage="pruned", eval_frame=entry.result.pair_truth_eval)
        )
        frames.append(
            _entry_rows(entry, stage="raw", eval_frame=entry.result.raw_pair_truth_eval)
        )

    comparison = pl.concat(frames, how="vertical")
    if not allow_mixed_population:
        _refuse_mixed_populations(comparison)
    return comparison


def _refuse_mixed_populations(comparison: pl.DataFrame) -> None:
    """Raise if `comparison` places two different `population_key` or
    `truth_key` values in the same country's rows.

    Checked per country, not across the whole table: a table spanning more
    than one country is expected to carry a different population per
    country, and only a disagreement *within* one country's rows means two
    entries were not comparable to begin with.
    """
    for key_column in ("population_key", "truth_key"):
        mixed = (
            comparison.filter(
                pl.col("country").is_not_null() & pl.col(key_column).is_not_null()
            )
            .group_by("country")
            .agg(pl.col(key_column).n_unique().alias("_distinct"))
            .filter(pl.col("_distinct") > 1)
        )
        if mixed.height:
            countries = sorted(mixed.get_column("country").to_list())
            raise ValueError(
                f"entries disagree on {key_column} for countries {countries!r} -- "
                "these runs scored different populations and cannot share one "
                "strategy_comparison row set; pass allow_mixed_population=True to "
                "build the mixed view anyway, which keeps population_key/truth_key "
                "as columns so the difference stays visible"
            )


def combine_strategy_comparisons(comparisons: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate per-pair `strategy_comparison` frames into one table.

    Each pair's comparison is written into its own
    `artifacts/blocking/<source>__<target>/comparison/` directory, so nothing until
    now put two countries' runs beside each other. This adds the `pair_label`
    column that separates them again once concatenated, derived from each
    row's own `source_system`/`target_system` rather than from the directory
    the frame was loaded from -- the row knows which run it describes, a
    directory only knows where someone put the file.

    Columns listed in `contracts.STRATEGY_COMPARISON_BACKFILLABLE_COLUMNS` are
    filled with nulls when a frame predates them, which is how a comparison
    written before `accelerator_settings` existed still aggregates. Its
    accelerator settings stay null rather than being defaulted: unknown and
    known-to-be-default are different facts, and only the null keeps a legacy
    run from silently grouping with a run that really was unaccelerated. Any
    *other* missing column is a broken file and raises.

    Raises `ValueError` if `comparisons` is empty.
    """

    if not comparisons:
        raise ValueError("comparisons must be non-empty to combine")

    frames: list[pl.DataFrame] = []
    for comparison in comparisons:
        backfilled = comparison.with_columns(
            pl.lit(None, dtype=_STRATEGY_COMPARISON_SCHEMA[column]).alias(column)
            for column in STRATEGY_COMPARISON_BACKFILLABLE_COLUMNS
            if column not in comparison.columns
        )
        validate_blocking_artifact_schema(
            backfilled, artifact_name="strategy_comparison"
        )
        frames.append(
            backfilled.with_columns(
                pl.concat_str(
                    pl.col("source_system"), pl.lit("__"), pl.col("target_system")
                ).alias("pair_label")
            )
            .select(list(_COMBINED_SCHEMA.keys()))
            .cast(pl.Schema(_COMBINED_SCHEMA))
        )

    return pl.concat(frames, how="vertical")


def summarize_runtime_scaling(combined: pl.DataFrame) -> pl.DataFrame:
    """Reduce a combined comparison table to a runtime-against-scale read.

    One row per `(representation, similarity_backend, accelerator_settings,
    target_rows` decade`)`, carrying min/median/max `runtime_seconds` so a
    backend's scaling with target size -- or its absence, for a
    partition-based backend -- is readable down a single column group.
    `target_rows` is bucketed by decade (`1e6-1e7`, and so on) rather than by a
    hand-picked list of edges, since the real spread runs from `ie`'s ~820K to
    `fr`'s ~12.9M and a decade boundary separates those without encoding any
    particular corpus's row counts into this function.

    `accelerator_settings` is part of the grouping key, never merely carried:
    an accelerated leg and the baseline it is measured against agree on every
    other grouping column, so leaving it out is exactly the silent pooling the
    column was added to stop. A null (a run written before the column existed)
    groups on its own for the same reason. `name_transform` is in the key for
    the same reason again: a cleansed-both-sides leg and the raw leg it is
    measured against differ in nothing else.

    Runtime is measured once per `execute_blocking_run()` call and then
    repeated onto every row that run emitted, so the rows are first reduced to
    one per run leg (`"pruned"` stage, deduplicated on `_RUN_LEG_KEY`) before
    any runtime statistic is taken. `run_legs` counts those, not rows.
    `runtime_covers_countries` is the largest number of countries any one
    runtime figure in the bucket spans: `1` means every figure is a
    single-country measurement, and anything higher means at least one is a
    whole-run total shared across that many countries and so an upper bound on
    the per-country time, not the time itself.

    `median_precision`/`median_recall` come from the same `"pruned"` legs --
    the candidate set clustering would actually receive -- so quality is
    readable against scale in the same row as runtime. The raw-stage rows are
    left in the combined table for anyone who needs pruning's cost; they are
    not a second scaling read.

    `median_recall_at_k`/`median_candidate_set_size_ratio` apply the same
    median-of-"pruned"-legs treatment to `compute_pair_truth_eval()`'s two
    ranking-quality/candidate-volume columns -- see that function's
    docstring for what each measures.
    """

    validate_blocking_artifact_schema(
        combined, artifact_name="strategy_comparison_combined"
    )

    # The universe row is selected by name before deduplicating: a leg holds
    # one row per population, and keeping whichever came first would read an
    # arbitrary population's quality figures without failing. A run with no
    # ground truth has no population at all and still contributes its runtime.
    legs = combined.filter(
        (pl.col("stage") == "pruned")
        & (
            pl.col("population").is_null()
            | (pl.col("population") == POPULATION_UNIVERSE)
        )
    ).unique(subset=list(_RUN_LEG_KEY), keep="first", maintain_order=True)

    bucket_exponent = (
        pl.when(pl.col("target_rows") > 0)
        .then(pl.col("target_rows").cast(pl.Float64).log10().floor().cast(pl.Int64))
        .otherwise(None)
    )
    legs = legs.with_columns(
        bucket_exponent.alias("_bucket_exponent"),
        pl.col("country")
        .n_unique()
        .over([key for key in _RUN_LEG_KEY if key != "country"])
        .cast(pl.Int64)
        .alias("_runtime_countries"),
    ).with_columns(
        pl.lit(10.0)
        .pow(pl.col("_bucket_exponent"))
        .cast(pl.Int64)
        .alias("target_rows_bucket_start"),
        pl.concat_str(
            pl.lit("1e"),
            pl.col("_bucket_exponent").cast(pl.Utf8),
            pl.lit("-1e"),
            (pl.col("_bucket_exponent") + 1).cast(pl.Utf8),
        ).alias("target_rows_bucket"),
    )

    summary = legs.group_by(
        [
            "representation",
            "similarity_backend",
            "accelerator_settings",
            "name_transform",
            "target_rows_bucket",
            "target_rows_bucket_start",
        ]
    ).agg(
        pl.len().cast(pl.Int64).alias("run_legs"),
        pl.col("target_rows").min().alias("min_target_rows"),
        pl.col("target_rows").max().alias("max_target_rows"),
        pl.col("runtime_seconds").min().alias("min_runtime_seconds"),
        pl.col("runtime_seconds").median().alias("median_runtime_seconds"),
        pl.col("runtime_seconds").max().alias("max_runtime_seconds"),
        # The run's own phase figures over the same legs: the index build and
        # the scan as medians, memory as the bucket's largest recorded peak.
        pl.col("target_index_seconds").median().alias("median_target_index_seconds"),
        pl.col("scoring_seconds").median().alias("median_scoring_seconds"),
        pl.col("peak_rss_bytes").max().alias("max_peak_rss_bytes"),
        pl.col("precision").median().alias("median_precision"),
        pl.col("recall").median().alias("median_recall"),
        # Same "pruned" legs, same median-across-runs treatment as
        # median_precision/median_recall above.
        pl.col("recall_at_k").median().alias("median_recall_at_k"),
        pl.col("candidate_set_size_ratio")
        .median()
        .alias("median_candidate_set_size_ratio"),
        pl.col("_runtime_countries").max().alias("runtime_covers_countries"),
    )

    return (
        summary.select(list(_RUNTIME_SCALING_SCHEMA.keys()))
        .cast(pl.Schema(_RUNTIME_SCALING_SCHEMA))
        .sort(
            [
                "representation",
                "similarity_backend",
                "accelerator_settings",
                "name_transform",
                "target_rows_bucket_start",
            ],
            nulls_last=True,
        )
    )


# The columns a repeatability check compares by default -- the two
# ranking-quality/candidate-volume metrics, `recall_at_k` and
# `candidate_set_size_ratio`, plus the blended figures they sit beside. Deliberately excludes `runtime_seconds` (expected
# to vary run to run) and every identifying column (`label`/`stage`/
# `country`/...), which are the join key rather than a measurement.
REPEATABILITY_CHECK_COLUMNS: tuple[str, ...] = (
    "precision",
    "recall",
    "reduction_ratio",
    "recall_at_k",
    "candidate_set_size_ratio",
)


def check_strategy_comparison_repeatability(
    first: pl.DataFrame,
    second: pl.DataFrame,
    *,
    columns: Sequence[str] = REPEATABILITY_CHECK_COLUMNS,
    key: Sequence[str] = ("label", "stage", "country", "population"),
) -> pl.DataFrame:
    """Compare two `strategy_comparison` frames from repeated runs of the
    same configuration and report where `columns` disagree.

    Candidate generation is verified deterministic across repeated runs, so
    two runs of an identical `BlockingRunConfig` are expected to produce
    byte-identical `columns` values for every `key` -- this is the check for
    that expectation, scoped by default to `recall_at_k`/
    `candidate_set_size_ratio` and the blended figures they sit beside,
    rather than the whole `strategy_comparison` schema (`runtime_seconds` is
    deliberately excluded from the default `columns`: wall-clock time is
    never expected to repeat exactly).

    `population` is part of the default key: a label, stage and country hold
    one row per population, and a key without it would join every population
    against every other and compare the universe with a level. A null key
    component matches a null, so a run with no ground truth, which has no
    population, still matches its own re-run.

    Not itself an assertion: returns one row per `key` combination present
    in either frame, an `f"{column}_match"` boolean per compared column, and
    an overall `matches` boolean, so a caller (a script re-running the same
    cell twice, or a test) decides what to do with a mismatch. `both null`
    counts as a match (the same "not computed" value both times); a value
    present on one side and null on the other, or a key present in only one
    frame (every column on the missing side comes back null from the `full`
    join below), counts as a mismatch on every compared column.
    """

    if not columns:
        raise ValueError("columns must be non-empty to check repeatability")

    key_columns = list(key)
    left = first.select([*key_columns, *columns]).rename(
        {column: f"{column}_first" for column in columns}
    )
    right = second.select([*key_columns, *columns]).rename(
        {column: f"{column}_second" for column in columns}
    )
    # A null key is the same fact on both sides -- a run with no ground truth has
    # no population -- so it joins to itself rather than splitting two identical
    # re-runs into two unmatched rows.
    joined = left.join(
        right, on=key_columns, how="full", coalesce=True, nulls_equal=True
    )

    match_columns = []
    for column in columns:
        first_col = pl.col(f"{column}_first")
        second_col = pl.col(f"{column}_second")
        both_null = first_col.is_null() & second_col.is_null()
        equal_values = (first_col == second_col).fill_null(False)
        match_columns.append((both_null | equal_values).alias(f"{column}_match"))

    joined = joined.with_columns(match_columns)
    joined = joined.with_columns(
        pl.all_horizontal([pl.col(f"{column}_match") for column in columns]).alias(
            "matches"
        )
    )

    return joined.select(
        [*key_columns, *[f"{column}_match" for column in columns], "matches"]
    ).sort(key_columns)


def diff_pair_recovery(
    baseline_detail: pl.DataFrame, variant_detail: pl.DataFrame
) -> pl.DataFrame:
    """Attribute recovery between two labelled runs from their own per-pair
    outcomes, rather than from a delta between their aggregate metrics.

    `baseline_detail`/`variant_detail` are two runs'
    `BlockingRunResult.pair_truth_eval_detail` frames -- the same pairing
    (same `source_system`/`target_system`/`country`) scored twice, once by
    each run. Every truth pair already carries a `found` verdict, so
    "what did the second run recover that the first missed" is read off the
    two runs' own verdicts for the same pair, not inferred from a change in
    blended precision/recall: a real gleif -> gb measurement is the reason
    this matters -- a 12x aggregate recall delta between a canonical run and
    a variant-expanded one turned out, once read pair by pair, to be the
    scored population shrinking from 3,006 pairs to 514, not the algorithm
    improving.

    Returns one row per truth pair present in both frames where `baseline_detail`
    missed it (`found=False`) and `variant_detail` found it (`found=True`) --
    `BLOCKING_ARTIFACT_SCHEMAS["pair_recovery_attribution"]`'s shape,
    carrying the pair's own ids/names plus the similarity and rank the
    variant run found it at, so "aliases recover x%" is derived from named
    pairs (`result.height`, or `result.height / <baseline truth pair count>`)
    rather than asserted from an aggregate delta.

    Only truth pairs are considered (`is_truth_pair=True` on both sides): a
    predicted-but-untrue row's `found` is `None` (see
    `compute_pair_truth_eval_detail`'s own docstring) and has no "recovered"
    reading. A pair present in only one frame (the two runs scored a
    different population, which is a caller error -- these should be two
    labelled runs *of the same pairing*) is dropped by the inner join rather
    than surfaced, since there is nothing to attribute recovery between.
    """

    key = list(_PAIR_RECOVERY_KEY)
    baseline_truth = baseline_detail.filter(pl.col("is_truth_pair")).select(
        [*key, pl.col("found").alias("baseline_found")]
    )
    variant_truth = variant_detail.filter(pl.col("is_truth_pair")).select(
        [
            *key,
            "source_name",
            "target_name",
            pl.col("found").alias("variant_found"),
            pl.col("similarity").alias("variant_similarity"),
            pl.col("rank").alias("variant_rank"),
        ]
    )

    joined = baseline_truth.join(variant_truth, on=key, how="inner")
    recovered = joined.filter(
        ~pl.col("baseline_found").fill_null(False) & pl.col("variant_found")
    )

    return (
        recovered.select(list(_PAIR_RECOVERY_ATTRIBUTION_SCHEMA.keys()))
        .cast(pl.Schema(_PAIR_RECOVERY_ATTRIBUTION_SCHEMA))
        .sort(key)
    )


def diff_name_equality_levels(
    baseline: StrategyRunEntry, variant: StrategyRunEntry
) -> pl.DataFrame:
    """Name the truth pairs whose name-equality level differs between two runs
    of one pairing, with each run's level and verdict for the pair.

    A run classifies the levels from name forms it derived under its own
    cleanse profile, so two runs under different profiles read their `never`
    recall over different pairs, and a change in that recall cannot be told
    apart from the cleanser redrawing the residual until the pairs that moved
    are named. `baseline` and `variant` are two completed runs of the same
    pairing and representation, read through their `pair_truth_eval_detail`;
    neither is re-classified under the other's profile.

    Returns one row per truth pair present in both runs whose `name_equality`
    differs, in `BLOCKING_ARTIFACT_SCHEMAS["name_equality_attribution"]`'s
    shape, each run named by its `<transform>:<profile>`. `never_attribution`
    is set for a pair one of the two runs left in `never`:
    `NEVER_SOLVED_BY_ALGORITHM` when that run found it anyway, and
    `NEVER_RECOVERED_BY_CLEANSE` when that run missed it. It is null for a
    pair moving between the other levels. A run with no detail frame
    contributes no pairs, and a pair present in only one run is dropped, as
    `diff_pair_recovery()` drops it.
    """

    baseline_detail = baseline.result.pair_truth_eval_detail
    variant_detail = variant.result.pair_truth_eval_detail
    if baseline_detail is None or variant_detail is None:
        return pl.DataFrame(schema=_NAME_EQUALITY_ATTRIBUTION_SCHEMA)

    key = list(_PAIR_RECOVERY_KEY)
    baseline_truth = baseline_detail.filter(pl.col("is_truth_pair")).select(
        [
            *key,
            "source_name",
            "target_name",
            pl.col("name_equality").alias("baseline_name_equality"),
            pl.col("found").alias("baseline_found"),
        ]
    )
    variant_truth = variant_detail.filter(pl.col("is_truth_pair")).select(
        [
            *key,
            pl.col("name_equality").alias("variant_name_equality"),
            pl.col("found").alias("variant_found"),
        ]
    )

    # The verdict of whichever run left the pair in `never`; null when neither did.
    never_run_found = (
        pl.when(pl.col("baseline_name_equality") == NAME_EQUALITY_NEVER)
        .then(pl.col("baseline_found"))
        .when(pl.col("variant_name_equality") == NAME_EQUALITY_NEVER)
        .then(pl.col("variant_found"))
    )
    moved = (
        baseline_truth.join(variant_truth, on=key, how="inner")
        .filter(
            pl.col("baseline_name_equality").ne_missing(pl.col("variant_name_equality"))
        )
        .with_columns(
            pl.lit(baseline.config.strategy.representation).alias("representation"),
            # Two backends find different pairs, so which one a verdict came
            # from is part of the verdict.
            pl.lit(baseline.config.strategy.similarity_backend).alias(
                "similarity_backend"
            ),
            pl.lit(_encode_name_transform(baseline.config.strategy)).alias(
                "baseline_name_transform"
            ),
            pl.lit(_encode_name_transform(variant.config.strategy)).alias(
                "variant_name_transform"
            ),
            pl.when(never_run_found)
            .then(pl.lit(NEVER_SOLVED_BY_ALGORITHM))
            .when(~never_run_found)
            .then(pl.lit(NEVER_RECOVERED_BY_CLEANSE))
            .alias("never_attribution"),
        )
    )

    return (
        moved.select(list(_NAME_EQUALITY_ATTRIBUTION_SCHEMA.keys()))
        .cast(pl.Schema(_NAME_EQUALITY_ATTRIBUTION_SCHEMA))
        .sort(key)
    )


def tally_name_equality_transitions(attribution: pl.DataFrame) -> pl.DataFrame:
    """Count a `diff_name_equality_levels()` frame's truth pairs by level
    transition, per representation and country, with the pairs one run left
    in `never` split by `never_attribution`.

    One row per `(representation, both runs' name transforms, pairing,
    country, baseline level, variant level, never_attribution)` present, in
    `BLOCKING_ARTIFACT_SCHEMAS["name_equality_transitions"]`'s shape,
    `truth_pairs` counting its pairs.
    """

    validate_blocking_artifact_schema(
        attribution, artifact_name="name_equality_attribution"
    )
    group = [
        column
        for column in _NAME_EQUALITY_TRANSITIONS_SCHEMA
        if column != "truth_pairs"
    ]
    return (
        attribution.group_by(group)
        .agg(pl.len().alias("truth_pairs"))
        .select(list(_NAME_EQUALITY_TRANSITIONS_SCHEMA.keys()))
        .cast(pl.Schema(_NAME_EQUALITY_TRANSITIONS_SCHEMA))
        .sort(group, nulls_last=True)
    )


def _refuse_repeated_labels(entries: Sequence[StrategyRunEntry]) -> None:
    """Raise if two entries share a `label`: it is what a pair's outcome row
    joins to its run on, so a repeated one would give a pair two verdicts from
    what reads as one run."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for entry in entries:
        (repeated if entry.label in seen else seen).add(entry.label)
    if repeated:
        raise ValueError(
            f"entries repeat the labels {sorted(repeated)!r}; a pair's outcome "
            "joins to its run on the label, so each run needs its own"
        )


def build_pair_outcome_runs(entries: Sequence[StrategyRunEntry]) -> pl.DataFrame:
    """The runs of one pairing, one row each, with the settings each was made
    under: the table `build_pair_outcomes()`'s rows join to on `label`.

    A report picks its runs here, by what they differ in, and reads those
    runs' verdicts from the outcomes; a run's settings are recorded once
    rather than on every pair's row. The settings are encoded the way
    `build_strategy_comparison()` encodes them, so a run reads the same in
    both tables. `truth_pairs` is the run's count of outcome rows, null for a
    run that scored no ground truth, `candidate_pairs` the pairs the run
    kept at its own settings, and `target_rows` the target side it scanned,
    summed over its countries, null for a run that scored no ground truth.

    Raises `ValueError` if two entries share a label.
    """

    _refuse_repeated_labels(entries)
    rows: list[dict[str, object]] = []
    for entry in entries:
        strategy = entry.config.strategy
        keys = entry.result.keys
        detail = entry.result.pair_truth_eval_detail
        rows.append(
            {
                "label": entry.label,
                "source_system": entry.config.source.system,
                "target_system": entry.config.target.system,
                "representation": strategy.representation,
                "tokenizer": _encode_tokenizer(strategy),
                "similarity_backend": strategy.similarity_backend,
                "is_exact_backend": _is_exact_backend(strategy.similarity_backend),
                "accelerator_settings": _encode_accelerator_settings(strategy),
                "name_transform": _encode_name_transform(strategy),
                "top_k": strategy.top_k,
                "min_similarity": strategy.min_similarity,
                "max_candidates_per_source": strategy.max_candidates_per_source,
                "population_key": None if keys is None else _population_key(keys),
                "truth_key": None if keys is None else keys.truth,
                "runtime_seconds": entry.runtime_seconds,
                "finished_at": entry.finished_at,
                "truth_pairs": (
                    None
                    if detail is None
                    else detail.filter(pl.col("is_truth_pair")).height
                ),
                "candidate_pairs": entry.result.candidate_pair_count,
                "target_rows": _target_rows(entry),
            }
        )
    return pl.DataFrame(rows, schema=_PAIR_OUTCOME_RUNS_SCHEMA).sort("label")


def _target_rows(entry: StrategyRunEntry) -> int | None:
    """The target rows `entry` scanned, one figure per country summed, from
    its `pair_truth_eval`; `None` for a run with no ground truth."""
    evaluation = entry.result.pair_truth_eval
    if evaluation is None or evaluation.height == 0:
        return None
    return int(
        evaluation.select("country", "target_rows")
        .unique(subset="country")
        .get_column("target_rows")
        .sum()
    )


def build_pair_outcomes(entries: Sequence[StrategyRunEntry]) -> pl.DataFrame:
    """Every run's verdict on every truth pair of one pairing, one row per
    pair per run, stacked rather than joined.

    Each run's `pair_truth_eval_detail` already holds one row per truth pair,
    and runs of one pairing over one population hold the same pairs, so the
    runs line up on the pair's key with no join between them. Every
    name-equality level is kept: a report reading the `never` pairs filters
    for them, and which runs it reads is a selection from
    `build_pair_outcome_runs()`. A predicted pair that is not a truth pair
    has no verdict and is left out, and a run with no detail frame
    contributes no rows.

    Raises `ValueError` if two entries share a label.
    """

    _refuse_repeated_labels(entries)
    columns = list(_PAIR_OUTCOMES_SCHEMA.keys())
    frames = [
        entry.result.pair_truth_eval_detail.filter(pl.col("is_truth_pair"))
        .with_columns(pl.lit(entry.label).alias("label"))
        .select(columns)
        for entry in entries
        if entry.result.pair_truth_eval_detail is not None
    ]
    if not frames:
        return pl.DataFrame(schema=_PAIR_OUTCOMES_SCHEMA)
    return (
        pl.concat(frames, how="vertical")
        .cast(pl.Schema(_PAIR_OUTCOMES_SCHEMA))
        .sort(["label", *_PAIR_RECOVERY_KEY])
    )


def build_pair_outcome_candidates(entries: Sequence[StrategyRunEntry]) -> pl.DataFrame:
    """Each run's kept candidate pairs counted by where they sit: one row per
    run per cell of similarity, to the thousandth below it, and per-source
    rank, `candidates` counting the pairs in that cell alone.

    Two runs made at one threshold do not spend the same comparisons, since
    cosine is on a different scale per representation, so a fair reading cuts
    each to a shared budget, by a cutoff, by a cap on candidates per source,
    or by both. The pairs a run keeps under any of those is a sum over these
    cells, which is what says which cutoff spends a budget. A similarity is
    counted at the thousandth below it, so a cutoff read here never keeps
    fewer pairs than the sum says. A run can only be read more tightly than
    it was made, never more loosely: the pairs below its threshold and beyond
    its `top_k` were never kept. A run holding no candidates has no rows.

    Raises `ValueError` if two entries share a label.
    """

    _refuse_repeated_labels(entries)
    steps = _CANDIDATE_CUTOFF_STEPS_PER_UNIT
    frames = []
    for entry in entries:
        edges = entry.result.matched_edges
        if not {"similarity", "rank"} <= set(edges.columns) or edges.height == 0:
            continue
        frames.append(
            edges.select(
                ((pl.col("similarity") * steps).floor() / steps).alias(
                    "min_similarity"
                ),
                pl.col("rank").cast(pl.Int64),
            )
            .group_by("min_similarity", "rank")
            .agg(pl.len().alias("candidates"))
            .select(
                pl.lit(entry.label).alias("label"),
                "min_similarity",
                "rank",
                "candidates",
            )
        )
    if not frames:
        return pl.DataFrame(schema=_PAIR_OUTCOME_CANDIDATES_SCHEMA)
    return (
        pl.concat(frames, how="vertical")
        .cast(pl.Schema(_PAIR_OUTCOME_CANDIDATES_SCHEMA))
        .sort(["label", "min_similarity", "rank"])
    )


def build_pair_outcome_source_candidates(
    entries: Sequence[StrategyRunEntry],
) -> pl.DataFrame:
    """`build_pair_outcome_candidates()`'s cells kept per source, for every
    source that holds a truth pair in the run: one row per run per source per
    cell of similarity and rank it holds a candidate in.

    A name-equality level is a set of truth pairs, and so of their sources,
    so the candidates a run spends on one level's sources, and with them its
    precision and reduction ratio on that level, are a sum over these rows.
    Sources with no truth pair are left out, since no level reads them. A run
    with no detail frame or no candidates has no rows.

    Raises `ValueError` if two entries share a label.
    """

    _refuse_repeated_labels(entries)
    steps = _CANDIDATE_CUTOFF_STEPS_PER_UNIT
    frames = []
    for entry in entries:
        edges = entry.result.matched_edges
        detail = entry.result.pair_truth_eval_detail
        if (
            detail is None
            or not {"similarity", "rank"} <= set(edges.columns)
            or edges.height == 0
        ):
            continue
        truth_sources = (
            detail.filter(pl.col("is_truth_pair"))
            .select("country", "source_id")
            .unique()
        )
        frames.append(
            edges.join(truth_sources, on=["country", "source_id"], how="semi")
            .select(
                "country",
                "source_id",
                ((pl.col("similarity") * steps).floor() / steps).alias(
                    "min_similarity"
                ),
                pl.col("rank").cast(pl.Int64),
            )
            .group_by("country", "source_id", "min_similarity", "rank")
            .agg(pl.len().alias("candidates"))
            .select(
                pl.lit(entry.label).alias("label"),
                "country",
                "source_id",
                "min_similarity",
                "rank",
                "candidates",
            )
        )
    if not frames:
        return pl.DataFrame(schema=_PAIR_OUTCOME_SOURCE_CANDIDATES_SCHEMA)
    return (
        pl.concat(frames, how="vertical")
        .cast(pl.Schema(_PAIR_OUTCOME_SOURCE_CANDIDATES_SCHEMA))
        .sort(["label", "country", "source_id", "min_similarity", "rank"])
    )


def build_pair_outcome_recall_curves(
    entries: Sequence[StrategyRunEntry],
) -> pl.DataFrame:
    """Each run's recall-against-comparisons-spent curve, one row per run per
    population per budget step, from its own `matched_edges` and
    `pair_truth_eval_detail` (`validation.recall_curve.compute_recall_curve`).

    Stacked beside the outcomes so a report reads each run's own area over
    the comparisons it spent without reopening the run. A run with no detail
    frame has no rows.

    Raises `ValueError` if two entries share a label.
    """

    _refuse_repeated_labels(entries)
    frames = [
        compute_recall_curve(
            matched_edges=entry.result.matched_edges,
            pair_truth_eval_detail=entry.result.pair_truth_eval_detail,
        ).with_columns(pl.lit(entry.label).alias("label"))
        for entry in entries
        if entry.result.pair_truth_eval_detail is not None
    ]
    if not frames:
        return pl.DataFrame(schema=_PAIR_OUTCOME_RECALL_CURVES_SCHEMA)
    return (
        pl.concat(frames, how="vertical_relaxed")
        .select(list(_PAIR_OUTCOME_RECALL_CURVES_SCHEMA.keys()))
        .cast(pl.Schema(_PAIR_OUTCOME_RECALL_CURVES_SCHEMA))
        .sort(["label", "population", "budget_step"])
    )
