"""What a blocking run is configured with, what it returns, and the shape of every frame it writes.

`BlockingRunConfig` names the two systems, the countries, the truth rule and a
`BlockingStrategyConfig`, which wraps `company_vectorize`'s strategy and build settings
with this area's own: the name transform and cleanse profile, the exact-name filter,
pruning, target-neighbour linking and the backend options. `validate_blocking_run_config()`
rejects a bad strategy or numeric field, a tokenizer that does not match its
representation, and a target that is not among the source's ground truth.
`blocking_run_settings()` walks the resolved configuration field by field, so the
settings key covers a field added later with no edit.

`BlockingRunResult` holds a run's frames and keys. `BLOCKING_ARTIFACT_SCHEMAS` holds the
shapes of the frames this area owns; the frames shared with validation are checked
against its schemas. `EmptySourceLoadError`, a `ValueError`, marks a country whose
source loaded no rows, which a script reports as a failed run; a real source with no
candidates is a result and writes a full artefact set.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from company_tokenize.name_preprocessing import (
    DEFAULT_NAME_PREPROCESSING_PROFILE,
    parse_name_preprocessing_profile,
)
from company_vectorize.clustering_contract import TFIDF_ANALYZERS
from company_vectorize.clustering_factory import resolve_clustering_strategy

from validation.contracts import (
    ARTIFACT_SCHEMAS,
    PolarsDType,
    validate_schema_against_registry,
)
from validation.runner import PAIR_TRUTH_EVAL_DETAIL_COLUMNS
from workspace.identity import RunKeys, digest_settings
from workspace.roots import WorkspaceRoots

from .name_transform import (
    DEFAULT_CLEANSE_PROFILE,
    DEFAULT_NAME_TRANSFORM,
    IDENTITY_NAME_TRANSFORM,
    NAME_TRANSFORMS,
)
from .truth import MatchedLayerTruth, SourceTruthResolver

_VALID_SIMILARITY_BACKENDS = {
    "sklearn",
    "sparse_dot_topn",
    "svd_rerank",
    "kmeans",
    "hdbscan",
    "lsh",
    "dense_brute",
    "hnsw",
}

# `lsh` hashes the columns of a sparse matrix, which a dense representation
# does not have.
_DENSE_REPRESENTATIONS = {"sbert", "encoder"}

_VALID_NAME_SOURCES = {"name", "short_name"}


class EmptySourceLoadError(ValueError):
    """A pairing's source side resolved zero rows for a country it was asked
    to score.

    Distinct from a legitimately empty candidate set (real source rows, no
    matches found), which still completes and writes a full, valid artefact
    set with zero-count fields -- see `BlockingRunResult`'s docstring. This
    error means the source *load itself* produced nothing, which is never a
    legitimate outcome for a country `resolve_blocking_countries()` included
    in the run: a subclass of `ValueError` so `scripts/run_blocking.py`'s
    existing `except (ValueError, FileNotFoundError)` handling in `main()`
    already reports it as a failed run (non-zero exit, nothing written to
    the run's directory) with no separate wiring needed, and so
    `scripts/compare_blocking_strategies.py`'s subprocess-based cells already
    record it as a `status="error"` row distinct from a `"completed"` one
    that merely found nothing to match.
    """


@dataclass(slots=True, frozen=True)
class BlockingDatasetDescriptor:
    """One side (source or target) of a two-dataset blocking run.

    Built by `loader.load_dataset_descriptor()`; not normally constructed by
    hand outside of tests.
    """

    system: str
    system_dir: Path
    layer: str
    has_ground_truth: bool
    matched_target_systems: tuple[str, ...]
    available_countries: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class BlockingStrategyConfig:
    """Thin wrapper over `company_vectorize`'s clustering-strategy contract.

    Field set mirrors exactly what `resolve_clustering_strategy()` /
    `resolve_target_index_build_settings()` / `ClusteringStrategy.score_source_chunk()`
    need, so callers can go from this straight into those calls with no
    translation layer -- except `exact_name_filter`, which is
    `_score_country`'s own concern: whether source rows
    with a direct `name` match on the target side are resolved via a direct
    join instead of paying for the tokenize/text-view/clustering path.
    Defaults on since the win applies unconditionally to every
    representation; set `False` to force every row through clustering (for
    example to isolate a strategy's own recall when comparing against an
    LSH candidate generator). `name_source` is the same kind of
    `_score_country`-only concern, scoped to the *source* side only: which
    column `_score_country` reads as the source's text-view/tokenization
    input, `"name"` (default) or `"short_name"` (`company_cleanse`'s
    existing suffix/noise-word stem). There is deliberately no target-side
    equivalent -- the target's text view always stays on its existing column
    regardless of `name_source`, since the field serves the asymmetric
    terse-source/full-fidelity-target case rather than offering a general
    per-side representation choice. `sbert_model_name` is the same kind of
    thin pass-through as `text_view`: `None` (default) means "resolve it",
    not "use the checkpoint literally named None" -- `workflow.py` resolves
    it from `BlockingRunConfig.countries` via
    `company_vectorize.clustering_policy.resolve_sbert_model_for_jurisdictions()`
    when unset, falling back to the pretrained English default. Only
    meaningful for `representation="sbert"`; ignored otherwise. Accepts a
    `company_vectorize` sbert model registry slug, a hub checkpoint
    identifier, or a local checkpoint path -- the same three forms
    `resolve_target_index_build_settings()` already accepts.

    `name_transform` names the one function applied to *both* sides' names
    before the exact-name fast path, tokenization and the text view see
    either (`name_transform.py`): `identity` (default, each side scores the
    column it always read), `cleanse`, `acronym` or `short_name`, the last
    three cleansing both sides live under `cleanse_profile`
    (`company_cleanse`'s normalization-profile syntax, `default` unless
    set) and scoring the derived column. Any transform but `identity`
    scores its own column on both sides, so `name_source` must stay `name`
    with it; the two compose only through `identity`.

    `target_neighbor_min_similarity`/`target_neighbor_max_per_target`
    turn on target-to-target linking: `None` (default, both must be `None`
    together) leaves clustering exactly as it always was, source-to-target
    edges only. Set both to probe every target row against the run's own
    target index, on `similarity_backend`, for its near neighbours --
    `target_neighbor_min_similarity` is deliberately its own threshold, above
    `min_similarity`, since target-side edges chain more readily than
    source-side ones; `target_neighbor_max_per_target`
    bounds how many neighbours one target row keeps after its own near match.
    The resulting target-neighbour edges are unioned into the edge set
    `workflow.py` builds `clusters`/`cluster_shape` from, so a cluster spans a
    name family once its members are close enough to each other, not only to
    whichever source happened to reach one of them; `clusters_before_union`/
    `cluster_shape_before_union` keep the source-to-target-only clustering
    beside it for comparison. Turning this on also lets a source some name
    form already matched skip the scan (see `_score_country`), since its
    other near targets are now reached through the matched target's own
    neighbour edges rather than only through the source's own scan.
    """

    representation: str
    top_k: int
    min_similarity: float
    max_candidates_per_source: int | None
    similarity_backend: str = "sklearn"
    backend_options: dict[str, object] | None = None
    tfidf_ngram_min: int = 2
    tfidf_ngram_max: int = 3
    tfidf_analyzer: str = "char_wb"
    text_view: str | None = None
    name_source: str = "name"
    preprocess_profile: str = DEFAULT_NAME_PREPROCESSING_PROFILE
    tokenizer: str = "wordpiece"
    # SentencePiece's `bpe` or `unigram`, promoted separately; unread for
    # wordpiece (`company_tokenize.paths.tokenizer_id`).
    tokenizer_encoding: str = "bpe"
    tokenizer_scope: str = "country"
    tokenizer_profile: str = "promoted"
    tokenizer_path: str | None = None
    max_candidates_per_target: int | None = None
    candidate_similarity_ratio: float | None = None
    exact_name_filter: bool = True
    sbert_model_name: str | None = None
    name_transform: str = DEFAULT_NAME_TRANSFORM
    cleanse_profile: str = DEFAULT_CLEANSE_PROFILE
    target_neighbor_min_similarity: float | None = None
    target_neighbor_max_per_target: int | None = None


@dataclass(slots=True, frozen=True)
class BlockingRunConfig:
    """Configuration for a two-dataset blocking workflow run.

    `emit_diagnostics` is off by default -- populating it costs one
    extra per-country dataframe snapshot of every operator-transformed value
    the run actually produced, which is wasted work at full corpus scale for
    the common case where nothing looks surprising. Set `True` to have
    `execute_blocking_run()` populate `BlockingRunResult.source_diagnostics`/
    `target_diagnostics` for post-hoc EDA; see those fields' docstring for
    the join key connecting them.

    `truth` is the rule that reads each source row's ground truth (see
    `truth.py`). The default, `MatchedLayerTruth`, is the source system's own
    recorded cross-system match; `ColumnTruth("source_uri")` is the rule for
    a perturbed dataset validated against the system it was made from, or a
    recorded name variant scored against its own system, whose truth is the
    identity each row was derived from. Part of the run identity.
    """

    roots: WorkspaceRoots
    prepared_base_dir: Path | None
    source: BlockingDatasetDescriptor
    target: BlockingDatasetDescriptor
    countries: tuple[str, ...] | None
    strategy: BlockingStrategyConfig
    emit_diagnostics: bool = False
    truth: SourceTruthResolver = field(default_factory=MatchedLayerTruth)


@dataclass(slots=True, frozen=True)
class BlockingRunResult:
    """Result of a completed blocking workflow run.

    `matched_edges`/`pair_truth_eval` reflect the pruned candidate set --
    what would actually be handed to clustering. `raw_matched_edges`/
    `raw_pair_truth_eval` reflect the unpruned scored candidates, so the
    pruning's precision/recall cost is directly visible. Both
    `pair_truth_eval` fields are `None` when the source dataset was loaded
    without ground truth (`require_ground_truth=False`); otherwise each
    carries one `compute_pair_truth_eval()` row per scored country.
    `pair_truth_eval_detail` is `pair_truth_eval`'s per-pair companion:
    `None` when the source dataset lacks ground truth (same as
    `pair_truth_eval`), and also `None` -- unlike `pair_truth_eval`, whose
    bucket columns merely go null -- when every scored country's `name`/
    `name_cleansed` lookup was unavailable, since a per-pair row with no
    names to show answers nothing. Otherwise one row per truth pair plus one
    row per predicted-but-untrue pair over the *pruned* candidate set
    (`compute_pair_truth_eval_detail()`, scored against `matched_edges`, not
    `raw_matched_edges` -- there is no raw variant of this field). Carries
    both sides' raw/cleansed names, the difficulty bucket each pair's
    raw-vs-cleansed name comparison places it in, and whether a truth pair
    was found or missed, so a false negative -- which leaves no trace in
    `matched_edges` -- is directly inspectable.

    This is the *whole* per-pair evaluation set, every difficulty
    bucket bundled together -- not, despite its former name
    (`residual_pairs`), only the still-unequal remainder. That remainder is
    one level within it (`name_equality == "never"`), reached via
    `inspection.filter_residual_pairs()`'s default. Do not read
    `pair_truth_eval_detail.height` as a residual count: most of this
    frame's rows are trivially-resolvable pairs (the `raw`, `basic` and
    `cleansed` levels) that the exact-name fast path, or cleansing, already
    closed -- see that function's docstring for where the actual remainder
    figure lives.

    `clusters` is connected-components clustering of `matched_edges` (the
    pruned set) unioned with `target_neighbor_edges` when the strategy's
    `target_neighbor_min_similarity`/`target_neighbor_max_per_target`
    are set, always populated regardless of ground-truth availability.
    `cluster_shape`, `directional_coverage`, and `similarity_distribution`
    are likewise always populated, derived from `matched_edges`/`clusters`
    alone with no dependency on ground truth. `target_neighbor_edges` is
    the target-to-target edge frame itself (`target_id_a`, `target_id_b`,
    `similarity`, `country`), empty when the strategy leaves the feature
    off. `clusters_before_union`/`cluster_shape_before_union` are the same
    clustering computed from `matched_edges` alone, with no target-neighbour
    edges unioned in -- identical to `clusters`/`cluster_shape` when the
    feature is off, and kept beside them so a run with the feature on can
    still show what changed. `target_neighbor_summary` is one row per
    scored country: the threshold and cap the edges were probed under, how
    many edges were found, how many targets hit the cap (a calibration
    reading over how readily near-name families chain together), and
    whether the probe reused a stored edge frame rather than recomputing it.
    `exact_match_summary` is the
    exact-join cascade's per-country counts: the sources some derived name
    form made equal to a target, split by form, and those no form matched --
    a source row count, not a pair count, and not the remainder either (see
    above); every source is scanned regardless of a match, unless the
    strategy's target-neighbour linking is on, in which case a matched
    source's own near targets are reached through the matched target's
    neighbour edges instead and the source is left out of the scan (see
    `_score_country`) -- always populated (zero matches when
    `strategy.exact_name_filter` is `False`), mirroring `pruning_summary`'s
    one-row-per-scored-country shape.

    `source_diagnostics`/`target_diagnostics` are `None` unless
    `BlockingRunConfig.emit_diagnostics` is `True`, in which case each is
    populated with one row per scored row on its side, carrying whichever
    operator-transformed columns this run actually produced (the raw input
    column the run read -- `strategy.name_source` on the source side,
    always `"name"` on the target side -- `"name_cleansed"` when the loaded
    layer carries it, and the feature column actually scored on,
    `cluster_text` or `cluster_tokens` depending on the run's effective text
    view). `source_diagnostics` is scoped to the unmatched population that
    actually went through the tokenize/text-view pipeline -- rows resolved
    via the exact-name fast path never reach it, since no operator ran
    on them (they're still visible via `exact_match_summary`/`matched_edges`
    with `similarity == 1.0`). `target_diagnostics` covers every scored
    target row, one row per canonical `target_id` regardless of whether
    name-variant expansion added extra index keys for it internally.
    Both carry a `cluster_id` column (null when that row's `source_id`/
    `target_id` never appears in `clusters`, i.e. it was never matched) --
    **this is the documented join key**: look up a source row's `cluster_id`
    in `source_diagnostics`, then filter `target_diagnostics` to that same
    `cluster_id` to see exactly which target row(s) it landed with, and
    compare the transformed-value columns to see why.
    """

    matched_edges: pl.DataFrame
    raw_matched_edges: pl.DataFrame
    pair_truth_eval: pl.DataFrame | None
    raw_pair_truth_eval: pl.DataFrame | None
    pair_truth_eval_detail: pl.DataFrame | None
    pruning_summary: pl.DataFrame
    exact_match_summary: pl.DataFrame
    clusters: pl.DataFrame
    cluster_shape: pl.DataFrame
    directional_coverage: pl.DataFrame
    similarity_distribution: pl.DataFrame
    candidate_pair_count: int
    artefact_paths: tuple[Path, ...] = ()
    summary: dict[str, object] | None = None
    source_diagnostics: pl.DataFrame | None = None
    target_diagnostics: pl.DataFrame | None = None
    # The target-to-target edge frame, and the source-to-target-only
    # clustering kept beside `clusters`/`cluster_shape` for comparison -- see
    # this dataclass's own docstring for what each carries and when it is
    # empty/identical to its `clusters`/`cluster_shape` counterpart.
    target_neighbor_edges: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "target_id_a": pl.Utf8,
                "target_id_b": pl.Utf8,
                "similarity": pl.Float64,
                "country": pl.Utf8,
            }
        )
    )
    clusters_before_union: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "cluster_id": pl.Utf8,
                "node_id": pl.Utf8,
                "node_name": pl.Utf8,
                "node_role": pl.Utf8,
            }
        )
    )
    cluster_shape_before_union: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "country": pl.Utf8,
                "total_clusters": pl.Int64,
                "singleton_clusters": pl.Int64,
                "singleton_ratio": pl.Float64,
                "mean_cluster_size": pl.Float64,
                "p95_cluster_size": pl.Float64,
                "max_cluster_size": pl.Int64,
            }
        )
    )
    target_neighbor_summary: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "country": pl.Utf8,
                "min_similarity": pl.Float64,
                "max_per_target": pl.Int64,
                "edge_count": pl.Int64,
                "targets_at_cap": pl.Int64,
                "cache_hit": pl.Boolean,
            }
        )
    )
    # The keys this run is identified by (`workspace.identity.RunKeys`),
    # each hashed from the rows it consumed; `None` only on a result built
    # by hand in a test.
    keys: RunKeys | None = None
    # The systems the run scored, recorded so a run made without ground
    # truth still says what it was.
    source_system: str | None = None
    target_system: str | None = None
    # What the run applied to both sides' names before comparing them
    # (`name_transform.NameTransform.describe()`), recorded on the run's own
    # output so a reader knows what was compared without inferring it from
    # the configuration that produced it.
    name_transform: dict[str, object] = field(default_factory=dict)
    # One `workspace.telemetry.PhaseRecord` mapping per phase the run went
    # through, per country, in the order they ended: wall-clock, resident set
    # size and CPU share, so an encode or a scan's cost is read off the run
    # rather than off a terminal.
    timings: tuple[dict[str, object], ...] = ()


def _validate_strategy(strategy: BlockingStrategyConfig) -> None:
    try:
        resolve_clustering_strategy(strategy.representation)
    except ValueError as exc:
        raise ValueError(f"strategy.representation: {exc}") from exc

    if strategy.top_k <= 0:
        raise ValueError("strategy.top_k must be greater than zero")
    if not (0.0 <= strategy.min_similarity <= 1.0):
        raise ValueError("strategy.min_similarity must be between 0 and 1")
    if (
        strategy.max_candidates_per_source is not None
        and strategy.max_candidates_per_source <= 0
    ):
        raise ValueError(
            "strategy.max_candidates_per_source must be greater than zero when provided"
        )
    if strategy.similarity_backend not in _VALID_SIMILARITY_BACKENDS:
        allowed = ", ".join(sorted(_VALID_SIMILARITY_BACKENDS))
        raise ValueError(f"strategy.similarity_backend must be one of: {allowed}")
    if (
        strategy.similarity_backend == "lsh"
        and strategy.representation in _DENSE_REPRESENTATIONS
    ):
        raise ValueError(
            f"strategy.similarity_backend 'lsh' hashes a sparse matrix and cannot "
            f"take the dense representation '{strategy.representation}'"
        )
    if strategy.name_source not in _VALID_NAME_SOURCES:
        allowed = ", ".join(sorted(_VALID_NAME_SOURCES))
        raise ValueError(f"strategy.name_source must be one of: {allowed}")
    if strategy.name_transform not in NAME_TRANSFORMS:
        allowed = ", ".join(sorted(NAME_TRANSFORMS))
        raise ValueError(f"strategy.name_transform must be one of: {allowed}")
    if not strategy.cleanse_profile.strip():
        raise ValueError("strategy.cleanse_profile must be non-empty")
    if (
        strategy.name_transform != IDENTITY_NAME_TRANSFORM
        and strategy.name_source != "name"
    ):
        raise ValueError(
            f"strategy.name_source={strategy.name_source!r} is a source-only column "
            f"choice and cannot combine with name_transform="
            f"{strategy.name_transform!r}, which scores its own column on both sides"
        )
    if not strategy.tokenizer_profile.strip():
        raise ValueError("strategy.tokenizer_profile must be non-empty")
    if strategy.tokenizer_path is not None and not strategy.tokenizer_path.strip():
        raise ValueError("strategy.tokenizer_path must be non-empty when provided")
    try:
        parse_name_preprocessing_profile(strategy.preprocess_profile)
    except ValueError as error:
        raise ValueError(f"strategy.preprocess_profile: {error}") from error
    if strategy.sbert_model_name is not None and not strategy.sbert_model_name.strip():
        raise ValueError(
            "strategy.sbert_model_name must be non-empty when provided (omit it, "
            "i.e. leave it None, to resolve one from the run's countries)"
        )
    if strategy.candidate_similarity_ratio is not None and not (
        0.0 < strategy.candidate_similarity_ratio <= 1.0
    ):
        raise ValueError(
            "strategy.candidate_similarity_ratio must be between 0 (exclusive) and 1 "
            "(inclusive) when provided"
        )
    if (
        strategy.max_candidates_per_target is not None
        and strategy.max_candidates_per_target <= 0
    ):
        raise ValueError(
            "strategy.max_candidates_per_target must be greater than zero when provided"
        )
    if (
        strategy.representation in ("wordpiece", "sentencepiece")
        and strategy.tokenizer != strategy.representation
    ):
        raise ValueError(
            f"strategy.representation '{strategy.representation}' requires "
            f"strategy.tokenizer='{strategy.representation}'"
        )
    if (strategy.target_neighbor_min_similarity is None) != (
        strategy.target_neighbor_max_per_target is None
    ):
        raise ValueError(
            "strategy.target_neighbor_min_similarity and "
            "strategy.target_neighbor_max_per_target must be set together, "
            "or both left None to leave target-to-target linking off"
        )
    if strategy.target_neighbor_min_similarity is not None and not (
        0.0 <= strategy.target_neighbor_min_similarity <= 1.0
    ):
        raise ValueError(
            "strategy.target_neighbor_min_similarity must be between 0 and 1"
        )
    if (
        strategy.target_neighbor_max_per_target is not None
        and strategy.target_neighbor_max_per_target <= 0
    ):
        raise ValueError(
            "strategy.target_neighbor_max_per_target must be greater than zero "
            "when provided"
        )
    if strategy.representation == "tfidf":
        if strategy.tfidf_ngram_min <= 0:
            raise ValueError("strategy.tfidf_ngram_min must be greater than zero")
        if strategy.tfidf_ngram_max < strategy.tfidf_ngram_min:
            raise ValueError(
                "strategy.tfidf_ngram_max must be greater than or equal to "
                "strategy.tfidf_ngram_min"
            )
        if strategy.tfidf_analyzer not in TFIDF_ANALYZERS:
            raise ValueError(
                "strategy.tfidf_analyzer must be one of "
                f"{sorted(TFIDF_ANALYZERS)}, got {strategy.tfidf_analyzer!r}"
            )


def _validate_ground_truth_target(config: BlockingRunConfig) -> None:
    config.truth.validate_pairing(
        source_system=config.source.system,
        target_system=config.target.system,
        source_has_ground_truth=config.source.has_ground_truth,
        matched_target_systems=config.source.matched_target_systems,
    )


def validate_blocking_run_config(config: BlockingRunConfig) -> None:
    if config.countries is not None and len(config.countries) == 0:
        raise ValueError("countries cannot be empty when provided")
    _validate_strategy(config.strategy)
    _validate_ground_truth_target(config)


def blocking_run_settings(config: BlockingRunConfig) -> dict[str, object]:
    """Everything a run was configured with, as one flat mapping.

    Derived from the whole resolved configuration rather than an enumerated
    list of fields, so a field added to `BlockingStrategyConfig` or
    `BlockingRunConfig` later extends the settings key without anyone having
    to remember to: the dataclasses are walked, and only the members that are
    *locations* rather than identity are dropped (`roots`,
    `prepared_base_dir` and each descriptor's `system_dir` -- absolute paths
    that differ between two checkouts of the same run). What the run read is
    not here: that is the population, truth and index keys.
    """
    strategy = {
        f"strategy.{field}": value
        for field, value in dataclasses.asdict(config.strategy).items()
    }
    sides = {
        f"{role}.{field}": value
        for role, descriptor in (("source", config.source), ("target", config.target))
        for field, value in dataclasses.asdict(descriptor).items()
        if field != "system_dir"
    }
    # The truth rule is walked the same way: its kind plus whatever fields
    # the rule carries, so a column-reading rule keys on the column it reads.
    truth = {"truth.kind": config.truth.kind}
    if dataclasses.is_dataclass(config.truth):
        truth.update(
            {
                f"truth.{field}": value
                for field, value in dataclasses.asdict(config.truth).items()
            }
        )
    return {
        **strategy,
        **sides,
        **truth,
        "countries": sorted(config.countries) if config.countries else None,
        "emit_diagnostics": config.emit_diagnostics,
    }


def blocking_settings_key(config: BlockingRunConfig) -> str:
    """The settings key of a run over `config`: `blocking_run_settings` digested
    through the workspace identity contract."""
    return digest_settings(blocking_run_settings(config))


def build_blocking_run_identity(
    config: BlockingRunConfig, *, keys: RunKeys
) -> dict[str, object]:
    """What a run's `manifest.json` records as its identity: its four keys,
    which name its directory, and the settings its settings key digests, so a
    caller who knows a run's parameters can recognise it by them."""
    return {**keys.as_identity(), "settings": blocking_run_settings(config)}


_MATCHED_EDGES_ARTIFACT_SCHEMA: dict[str, PolarsDType] = {
    "source_id": pl.Utf8,
    "target_id": pl.Utf8,
    "similarity": pl.Float64,
    "rank": pl.Int64,
    "country": pl.Utf8,
}

# `accelerator_settings` is the discriminator that keeps an accelerated run
# from merging into its own baseline. `representation`, `similarity_backend`
# and `is_exact_backend` are identical for a `sklearn` run made with
# `prefix_filter` and one made without, so aggregating on those alone pools a
# pruned run with the unpruned run it was meant to be measured against. It is
# one serialised column rather than a named column per setting because the
# settings it records come from `BlockingStrategyConfig.backend_options`, an
# open `dict[str, object]` whose key set is per-backend and owned by
# `company_vectorize` -- naming them here would either add a mostly-null
# column per backend option or make this schema track that package's option
# surface. See `comparison._encode_accelerator_settings()` for the encoding.
_STRATEGY_COMPARISON_ARTIFACT_SCHEMA: dict[str, PolarsDType] = {
    "label": pl.Utf8,
    "representation": pl.Utf8,
    # The tokenizer candidate a run was scored with, carried as its
    # own axis alongside representation/similarity_backend/cleanse level --
    # `wordpiece`/`sentencepiece` artifacts are per `(system, trainer)` and
    # both live simultaneously (`company_tokenize`'s archived-candidate
    # override makes any of them addressable), so two runs differing only in
    # `tokenizer` are a tokenizer comparison, not a representation
    # change, and need their own column rather than being folded into
    # `label`. Written as `<trainer>:<profile>`, the label that picks one of
    # the scope's stored tokenizers.
    "tokenizer": pl.Utf8,
    "similarity_backend": pl.Utf8,
    "is_exact_backend": pl.Boolean,
    "accelerator_settings": pl.Utf8,
    # The function both sides' names passed through before comparison, as
    # `comparison._encode_name_transform` writes it:
    # `<kind>:<cleanse_profile>;preprocess:<preprocess_profile>`. Two runs
    # differing only in it are a
    # transform comparison, and it is in the grouping key of every
    # aggregate for the same reason `accelerator_settings` is.
    "name_transform": pl.Utf8,
    "runtime_seconds": pl.Float64,
    # The run's own phase figures for this row's country, read from
    # `BlockingRunResult.timings`: the target index build (the encode for a
    # dense representation, the vectoriser fit for a sparse one), the scan,
    # and the largest resident set size any phase recorded. These put cost
    # beside recall on one row, where `runtime_seconds` is a whole-run figure
    # measured outside the workflow. Null for a run made before the workflow
    # recorded phases.
    "target_index_seconds": pl.Float64,
    "scoring_seconds": pl.Float64,
    "peak_rss_bytes": pl.Int64,
    "stage": pl.Utf8,
    "source_system": pl.Utf8,
    "target_system": pl.Utf8,
    "country": pl.Utf8,
    # Which population of `pair_truth_eval` the row's figures are over:
    # `universe` or a `name_equality` level. Part of the row's key, so the
    # universe and each level are read as their own rows rather than as a
    # blended figure beside prefixed columns.
    "population": pl.Utf8,
    # Not the same concept as `population` above -- this is the run's source
    # and target population keys digested together, the rows the run
    # actually scored, `null` when the row carries no country (a run with no
    # ground truth). `build_strategy_comparison()` groups on this and
    # `truth_key` by default and refuses a table that mixes either within one
    # country's rows, so two runs that scored different populations are never
    # silently pooled into one comparison.
    "population_key": pl.Utf8,
    # The run's truth key, the digest of the resolved truth map it was scored
    # against. `null` for a run with no ground truth, matching
    # `population_key`.
    "truth_key": pl.Utf8,
    "target_rows": pl.Int64,
    "labelled_sources": pl.Int64,
    "truth_pairs": pl.Int64,
    "predicted_pairs": pl.Int64,
    "tp": pl.Int64,
    "fp": pl.Int64,
    "fn": pl.Int64,
    "precision": pl.Float64,
    "recall": pl.Float64,
    "reduction_ratio": pl.Float64,
    # candidate_pair_count/source_rows/candidate_set_size_ratio/recall_at_k,
    # produced by `validation.runner.compute_pair_truth_eval()` alongside the
    # blended figures above -- see its docstring for what each measures.
    # Backfillable (STRATEGY_COMPARISON_BACKFILLABLE_COLUMNS below) since an
    # older comparison file has none of them.
    "candidate_pair_count": pl.Int64,
    "source_rows": pl.Int64,
    "candidate_set_size_ratio": pl.Float64,
    "recall_at_k": pl.Float64,
}

BLOCKING_ARTIFACT_SCHEMAS: dict[str, dict[str, PolarsDType]] = {
    "matched_edges": _MATCHED_EDGES_ARTIFACT_SCHEMA,
    "raw_matched_edges": _MATCHED_EDGES_ARTIFACT_SCHEMA,
    "pruning_summary": {
        "country": pl.Utf8,
        "raw_edge_count": pl.Int64,
        "dropped_by_similarity_ratio": pl.Int64,
        "dropped_by_target_cap": pl.Int64,
        "pruned_edge_count": pl.Int64,
    },
    "exact_match_summary": {
        "country": pl.Utf8,
        "exact_match_count": pl.Int64,
        "resolved_by_raw": pl.Int64,
        "resolved_by_basic": pl.Int64,
        "resolved_by_cleansed": pl.Int64,
        "resolved_by_transform": pl.Int64,
        "multi_target_source_count": pl.Int64,
        "unmatched_source_row_count": pl.Int64,
    },
    "directional_coverage": {
        "country": pl.Utf8,
        "source_records": pl.Int64,
        "source_records_with_cluster": pl.Int64,
        "source_records_clustered_with_target": pl.Int64,
        "directional_coverage_ratio": pl.Float64,
    },
    # One row per target-to-target near-neighbour edge a run's own
    # target index was probed for; empty when the strategy leaves the
    # feature off. Undirected for clustering purposes -- `graph_cluster
    # .build_connected_components()` does not care which side is `_a`/`_b`
    # -- but not symmetrised on disk: each row is one probed target's own
    # kept candidate, so a mutual pair can appear as two rows.
    "target_neighbor_edges": {
        "target_id_a": pl.Utf8,
        "target_id_b": pl.Utf8,
        "similarity": pl.Float64,
        "country": pl.Utf8,
    },
    # One row per scored country, the target-neighbour probe's own
    # settings and outcome -- see `BlockingRunResult.target_neighbor_summary`'s
    # docstring reference on that field.
    "target_neighbor_summary": {
        "country": pl.Utf8,
        "min_similarity": pl.Float64,
        "max_per_target": pl.Int64,
        "edge_count": pl.Int64,
        "targets_at_cap": pl.Int64,
        "cache_hit": pl.Boolean,
    },
    "similarity_distribution": {
        "bucket_start": pl.Float64,
        "bucket_end": pl.Float64,
        "edge_count": pl.Int64,
    },
    # Diagnostic frames carry a variable set of transformed-value
    # columns depending on the run's representation/text-view, so only the
    # columns guaranteed present in every run are checked here (matches
    # validate_blocking_artifact_schema()'s default require_exact_columns=False,
    # which only rejects *missing* required columns, never extra ones).
    "source_diagnostics": {
        "source_id": pl.Utf8,
        "country": pl.Utf8,
        "cluster_id": pl.Utf8,
    },
    "target_diagnostics": {
        "target_id": pl.Utf8,
        "country": pl.Utf8,
        "cluster_id": pl.Utf8,
    },
    # One row per truth pair plus one row per predicted-but-untrue
    # pair, the per-pair companion to `pair_truth_eval`'s population rows --
    # every name-equality level bundled together (
    # named `pair_truth_eval_detail`, not `residual_pairs`, since it is not
    # pre-filtered to the still-unequal remainder). Columns are owned by
    # `validation.runner` (the module that actually computes them, via
    # `compute_pair_truth_eval_detail`) and imported here rather than
    # restated, matching the drift-avoidance `_EMPTY_PAIR_TRUTH_EVAL_SCHEMA`
    # already applies to `pair_truth_eval`. Registered in `blocking`'s own
    # schema registry rather than `validation.contracts.ARTIFACT_SCHEMAS`
    # since, unlike `pair_truth_eval`, `pair_truth_eval_detail` has no
    # consumer in `validation`'s own run output.
    "pair_truth_eval_detail": PAIR_TRUTH_EVAL_DETAIL_COLUMNS,
    # `comparison.diff_pair_recovery()`'s own artifact -- one row per
    # truth pair a `variant` labelled run found that its `baseline` missed,
    # named by the pair's own ids rather than expressed as a delta between
    # two runs' aggregate metrics. See that function's docstring for the
    # join this is built from.
    "pair_recovery_attribution": {
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "source_id": pl.Utf8,
        "target_id": pl.Utf8,
        "source_name": pl.Utf8,
        "target_name": pl.Utf8,
        "variant_similarity": pl.Float64,
        "variant_rank": pl.Int64,
    },
    # `comparison.diff_name_equality_levels()`'s own artifact: one row per
    # truth pair whose name-equality level differs between two runs of one
    # pairing, which a change of cleanse profile redraws, with both runs'
    # level and `found` verdict. `never_attribution` says, for a pair one run
    # left in `never`, whether that run found it anyway or missed it.
    "name_equality_attribution": {
        "representation": pl.Utf8,
        "similarity_backend": pl.Utf8,
        "baseline_name_transform": pl.Utf8,
        "variant_name_transform": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "source_id": pl.Utf8,
        "target_id": pl.Utf8,
        "source_name": pl.Utf8,
        "target_name": pl.Utf8,
        "baseline_name_equality": pl.Utf8,
        "variant_name_equality": pl.Utf8,
        "baseline_found": pl.Boolean,
        "variant_found": pl.Boolean,
        "never_attribution": pl.Utf8,
    },
    # `comparison.tally_name_equality_transitions()`: that attribution counted
    # by level transition, per representation and country.
    "name_equality_transitions": {
        "representation": pl.Utf8,
        "similarity_backend": pl.Utf8,
        "baseline_name_transform": pl.Utf8,
        "variant_name_transform": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "baseline_name_equality": pl.Utf8,
        "variant_name_equality": pl.Utf8,
        "never_attribution": pl.Utf8,
        "truth_pairs": pl.Int64,
    },
    # `comparison.build_pair_outcome_runs()`: one row per run of a pairing,
    # carrying the settings it was made under, which `pair_outcomes` joins to
    # on `label` rather than repeating on every pair's row. `truth_pairs` is
    # null for a run that scored no ground truth, which has no outcome rows.
    # `target_rows` is the target side the run scanned, summed over its
    # countries, which a reduction ratio is read against.
    "pair_outcome_runs": {
        "label": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "representation": pl.Utf8,
        "tokenizer": pl.Utf8,
        "similarity_backend": pl.Utf8,
        "is_exact_backend": pl.Boolean,
        "accelerator_settings": pl.Utf8,
        "name_transform": pl.Utf8,
        "top_k": pl.Int64,
        "min_similarity": pl.Float64,
        "max_candidates_per_source": pl.Int64,
        "population_key": pl.Utf8,
        "truth_key": pl.Utf8,
        "runtime_seconds": pl.Float64,
        "finished_at": pl.Utf8,
        "truth_pairs": pl.Int64,
        "candidate_pairs": pl.Int64,
        "target_rows": pl.Int64,
    },
    # `comparison.build_pair_outcome_source_candidates()`: the same cells as
    # `pair_outcome_candidates` below, kept per source for every source that
    # holds a truth pair, so a population's own candidates, and with them its
    # precision and reduction ratio, are a sum over its sources' cells.
    "pair_outcome_source_candidates": {
        "label": pl.Utf8,
        "country": pl.Utf8,
        "source_id": pl.Utf8,
        "min_similarity": pl.Float64,
        "rank": pl.Int64,
        "candidates": pl.Int64,
    },
    # `comparison.build_pair_outcome_candidates()`: a run's kept candidate
    # pairs counted by the thousandth of similarity and the per-source rank
    # each sits at, one row per run per cell it holds a candidate in.
    # `candidates` is the count in that cell alone, so the pairs a run keeps
    # at a cutoff, under a cap per source, or both, is a sum over cells, and
    # that sum is what turns a budget of candidates into the cutoff that
    # spends it.
    "pair_outcome_candidates": {
        "label": pl.Utf8,
        "min_similarity": pl.Float64,
        "rank": pl.Int64,
        "candidates": pl.Int64,
    },
    # `comparison.build_pair_outcomes()`: one row per truth pair per run, the
    # run's verdict on that pair. A pair is `(source_system, target_system,
    # country, source_id, target_id)` and is the same pair in every run of the
    # pairing, so runs line up on it; `name_equality` is the level the run gave
    # the pair under its own cleanse profile, and the two cleansed names are
    # the forms that level was read from. `similarity` and `rank` are null for
    # a pair the run never predicted.
    "pair_outcomes": {
        "label": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "source_id": pl.Utf8,
        "target_id": pl.Utf8,
        "source_name": pl.Utf8,
        "target_name": pl.Utf8,
        "source_name_cleansed": pl.Utf8,
        "target_name_cleansed": pl.Utf8,
        "name_equality": pl.Utf8,
        "found": pl.Boolean,
        "similarity": pl.Float64,
        "rank": pl.Int64,
    },
    # `comparison.build_pair_outcome_recall_curves()`: each run's
    # `validation.recall_curve.compute_recall_curve()` curve, stacked and
    # labelled, so each run's own area is read without reopening the run.
    "pair_outcome_recall_curves": {
        "label": pl.Utf8,
        **ARTIFACT_SCHEMAS["recall_curve"],
    },
    # `audit.compute_pair_audit()`: one row per truth pair of one
    # run, the run's own verdict beside what an exact scan at the run's own
    # `top_k`/`min_similarity`/`max_candidates_per_source` would have done.
    # `exact_similarity`/`exact_rank` are the pair's true cosine similarity
    # and rank against the run's own cached target index -- read from the
    # run's `raw_matched_edges` when its backend already scored exactly and
    # the pair is present there, rescored through the same scoring seam
    # otherwise (`workflow.rescore_missed_pairs`) to the run's own `top_k`, so
    # a rescored pair ranked beyond it has both null. `exact_kept` is whether
    # that similarity/rank clears the run's own cut; `distance_to_threshold`
    # is `exact_similarity - min_similarity`, negative below the cut.
    # `verdict` is one of `audit.VERDICT_FOUND` (the run found it and the
    # exact scan agrees), `audit.VERDICT_LOST_TO_BACKEND` (the run missed it,
    # the exact scan would have kept it -- a backend recall failure),
    # `audit.VERDICT_NOT_KEPT_BY_EXACT_SCAN` (the run missed it and the exact
    # scan would not have kept it either) or `audit.VERDICT_BACKEND_ONLY`
    # (the run found it but the exact scan's own top-k would drop it).
    # `rescored` says whether `exact_similarity`/`exact_rank` came from a
    # fresh scan rather than the run's own recorded score.
    "pair_audit": {
        "label": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "source_id": pl.Utf8,
        "target_id": pl.Utf8,
        "name_equality": pl.Utf8,
        "run_found": pl.Boolean,
        "exact_similarity": pl.Float64,
        "exact_rank": pl.Int64,
        "exact_kept": pl.Boolean,
        "distance_to_threshold": pl.Float64,
        "verdict": pl.Utf8,
        "rescored": pl.Boolean,
    },
    # `audit.summarize_pair_audit()`: a `pair_audit` frame reduced to one row
    # per `(label, source_system, target_system, country, name_equality)`,
    # `exact_scan_recall` being `exact_kept / truth_pairs` -- the exact
    # scan's own recall at the run's settings, to compare against the run's
    # `recall` on `pair_truth_eval`.
    "pair_audit_summary": {
        "label": pl.Utf8,
        "source_system": pl.Utf8,
        "target_system": pl.Utf8,
        "country": pl.Utf8,
        "name_equality": pl.Utf8,
        "truth_pairs": pl.Int64,
        "run_found": pl.Int64,
        "exact_kept": pl.Int64,
        "lost_to_backend": pl.Int64,
        "not_kept_by_exact_scan": pl.Int64,
        "backend_only": pl.Int64,
        "exact_scan_recall": pl.Float64,
    },
    "strategy_comparison": _STRATEGY_COMPARISON_ARTIFACT_SCHEMA,
    "strategy_comparison_combined": {
        # Which `artifacts/blocking/<pair>/` a row belongs to, derived from the
        # row's own `source_system`/`target_system` rather than from the
        # directory it was read out of, so a file that has been moved or
        # copied still aggregates under the pair it actually describes.
        "pair_label": pl.Utf8,
        **_STRATEGY_COMPARISON_ARTIFACT_SCHEMA,
    },
    # One row per `(representation, similarity_backend, accelerator_settings,
    # target_rows` decade`)`. `accelerator_settings` is in the grouping key,
    # not just carried along: dropping it would pool an accelerated leg with
    # its own baseline, which is what the column was added to prevent.
    "strategy_comparison_runtime_scaling": {
        "representation": pl.Utf8,
        "similarity_backend": pl.Utf8,
        "accelerator_settings": pl.Utf8,
        "name_transform": pl.Utf8,
        "target_rows_bucket": pl.Utf8,
        "target_rows_bucket_start": pl.Int64,
        "run_legs": pl.Int64,
        "min_target_rows": pl.Int64,
        "max_target_rows": pl.Int64,
        "min_runtime_seconds": pl.Float64,
        "median_runtime_seconds": pl.Float64,
        "max_runtime_seconds": pl.Float64,
        # The phase figures over the same run legs: the index build and the
        # scan as medians, memory as the bucket's largest peak.
        "median_target_index_seconds": pl.Float64,
        "median_scoring_seconds": pl.Float64,
        "max_peak_rss_bytes": pl.Int64,
        "median_precision": pl.Float64,
        "median_recall": pl.Float64,
        # Same-name median of the recall_at_k/candidate_set_size_ratio per-cell
        # metrics, read off the same "pruned" run legs as
        # median_precision/median_recall above.
        "median_recall_at_k": pl.Float64,
        "median_candidate_set_size_ratio": pl.Float64,
        "runtime_covers_countries": pl.Int64,
    },
}

# Columns added to `strategy_comparison` after files had already been written
# under it. `comparison.combine_strategy_comparisons()` back-fills exactly
# these with nulls when reading an older file, and nothing else -- any other
# missing column is a broken file, and still fails validation rather than
# being silently nulled into the aggregate. The candidate_pair_count/
# source_rows/candidate_set_size_ratio/recall_at_k quartet is null for a file
# written before those columns existed. `population` is deliberately not
# backfillable: a file without it predates the population rows, and its level
# figures cannot be reconstructed from it, so it is re-run rather than read.
STRATEGY_COMPARISON_BACKFILLABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "accelerator_settings",
        "tokenizer",
        "name_transform",
        "candidate_pair_count",
        "source_rows",
        "candidate_set_size_ratio",
        "recall_at_k",
        "target_index_seconds",
        "scoring_seconds",
        "peak_rss_bytes",
        # A file written before the run keys existed carries none at all,
        # so its rows backfill to null and group on their own -- the
        # same "unknown, not known-to-be-default" reasoning
        # `accelerator_settings` already documents above.
        "population_key",
        "truth_key",
    }
)


def validate_blocking_artifact_schema(
    frame: pl.DataFrame,
    *,
    artifact_name: str,
    require_exact_columns: bool = False,
) -> None:
    validate_schema_against_registry(
        frame,
        artifact_name=artifact_name,
        schemas=BLOCKING_ARTIFACT_SCHEMAS,
        require_exact_columns=require_exact_columns,
    )
