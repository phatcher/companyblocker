"""Audit a finished run's truth pairs against the exact scan at the run's own
settings.

Precision and recall (`pair_truth_eval`) say a run's overall hit rate, not
why a miss was missed: whether the exact scan at the run's own `top_k`,
`min_similarity` and `max_candidates_per_source` would have kept the pair
too (lost to the backend's own routing) or would not have kept it either
(lost to the threshold or the representation). `compute_pair_audit` answers
that per pair, from the run's own cached target index, without an exhaustive
sources x targets scan: only the truth pairs need settling, and both
vectors of a missed pair exist whether or not the backend reached it.

`produce_pair_audit` wraps the computation as its own production
(`workspace.records.produce`), keyed on the run it audits, so a second audit
of the same run reuses the first's location instead of rescoring anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from workspace.identity import ConsumedReference
from workspace.records import produce, read_record
from workspace.reference import locate, reference_at
from workspace.roots import WorkspaceRoots

from .comparison import _PAIR_RECOVERY_KEY, StrategyRunEntry, _is_exact_backend
from .contracts import (
    _DENSE_REPRESENTATIONS,
    BLOCKING_ARTIFACT_SCHEMAS,
    BlockingStrategyConfig,
    validate_blocking_artifact_schema,
)
from .loader import load_country_frame
from .run_layout import (
    AuditArtefact,
    BlockingAuditLocation,
    BlockingRunLocation,
    audit_reference_for,
    resolve_audit_location,
)
from .stored_runs import load_strategy_run_entry
from .workflow import rescore_missed_pairs

_PAIR_AUDIT_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["pair_audit"]
_PAIR_AUDIT_SUMMARY_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["pair_audit_summary"]

# `pair_audit.verdict`'s four values. `FOUND` and `BACKEND_ONLY` both describe
# a pair the run found (`run_found=True`); `LOST_TO_BACKEND` and
# `NOT_KEPT_BY_EXACT_SCAN` both describe one it missed.
VERDICT_FOUND = "found"
VERDICT_BACKEND_ONLY = "backend_only"
VERDICT_LOST_TO_BACKEND = "lost_to_backend"
VERDICT_NOT_KEPT_BY_EXACT_SCAN = "not_kept_by_exact_scan"


def _exact_rescore_backend(representation: str) -> str:
    """The exact backend `rescore_missed_pairs` scores an unsettled pair
    with, chosen so the rescore never fits an index the run's own memory gate
    would have refused: `"dense_brute"`, a chunked exhaustive scan, for a
    dense representation (`contracts._DENSE_REPRESENTATIONS`), since
    `"sklearn"` would fit the same whole-target index
    `company_vectorize.dense_target_memory_gate` may have refused at scale;
    `"sklearn"` (brute-force cosine, always available) for a sparse one,
    where that gate does not apply. Both are exact
    (`comparison._EXACT_SIMILARITY_BACKENDS`), so either is what this
    audit's own "exact scan" is defined against.
    """
    return "dense_brute" if representation in _DENSE_REPRESENTATIONS else "sklearn"


def _effective_top_k(strategy: BlockingStrategyConfig) -> int:
    """The run's own rank boundary: `top_k`, tightened by
    `max_candidates_per_source` when the run set it."""
    if strategy.max_candidates_per_source is None:
        return strategy.top_k
    return min(strategy.top_k, strategy.max_candidates_per_source)


def _rescore_lookup(
    entry: StrategyRunEntry, needs_rescore: pl.DataFrame, *, backend: str
) -> pl.DataFrame:
    """Every `(source_id, target_id, country)` in `needs_rescore` rescored
    against `entry`'s own cached target index, one `rescore_missed_pairs`
    call per country so the country's source rows are loaded and its target
    index resolved once rather than per pair. Each source keeps only the
    run's own `_effective_top_k`, all a verdict reads, so a pair ranked
    beyond it has no rescored row and is not kept by the exact scan."""
    schema = {
        "source_id": pl.Utf8,
        "target_id": pl.Utf8,
        "country": pl.Utf8,
        "rescored_similarity": pl.Float64,
        "rescored_rank": pl.Int64,
    }
    if needs_rescore.height == 0:
        return pl.DataFrame(schema=schema)

    parts: list[pl.DataFrame] = []
    for country in sorted(needs_rescore.get_column("country").unique().to_list()):
        country_rows = needs_rescore.filter(pl.col("country") == country)
        source_ids = sorted(country_rows.get_column("source_id").unique().to_list())
        full_source_frame = load_country_frame(entry.config.source, country=country)
        subset = full_source_frame.filter(
            pl.col("system_uri").cast(pl.Utf8).is_in(source_ids)
        )
        raw_edges = rescore_missed_pairs(
            entry.config,
            country=country,
            source_rows=subset,
            backend=backend,
            top_k=_effective_top_k(entry.config.strategy),
        )
        parts.append(
            raw_edges.select(
                "source_id",
                "target_id",
                "country",
                pl.col("similarity").alias("rescored_similarity"),
                pl.col("rank").alias("rescored_rank"),
            )
        )
    return pl.concat(parts, how="vertical_relaxed")


def compute_pair_audit(entry: StrategyRunEntry) -> pl.DataFrame:
    """Every truth pair of `entry`'s run, the run's own verdict beside what
    an exact scan at the run's own settings would have done.

    Raises `ValueError` naming `max_candidates_per_target` when the run was
    made with one set: whether a pair survives that cap depends on every
    other source competing for the same target, which this per-pair audit
    cannot settle from one pair alone.

    A pair already scored by an exact backend (`comparison._is_exact_backend`)
    and present in the run's own `raw_matched_edges` is read from there --
    its recorded similarity and rank already are the exact ones. Every other
    truth pair (missed outright, or scored by a backend that is not exact) is
    rescored through `workflow.rescore_missed_pairs`, one call per country
    covering every pair that country needs.
    """
    strategy = entry.config.strategy
    if strategy.max_candidates_per_target is not None:
        raise ValueError(
            "strategy.max_candidates_per_target is set "
            f"({strategy.max_candidates_per_target!r}); whether a pair "
            "survives that cap depends on every other source competing for "
            "the same target, so a run made with it cannot be audited pair "
            "by pair."
        )

    detail = entry.result.pair_truth_eval_detail
    if detail is None:
        return pl.DataFrame(schema=_PAIR_AUDIT_SCHEMA)

    truth = detail.filter(pl.col("is_truth_pair"))
    if truth.height == 0:
        return pl.DataFrame(schema=_PAIR_AUDIT_SCHEMA)

    key = list(_PAIR_RECOVERY_KEY)
    raw_lookup = entry.result.raw_matched_edges.select(
        "source_id",
        "target_id",
        "country",
        pl.col("similarity").alias("raw_similarity"),
        pl.col("rank").alias("raw_rank"),
    )
    joined = truth.join(
        raw_lookup, on=["source_id", "target_id", "country"], how="left"
    )

    is_exact = bool(_is_exact_backend(strategy.similarity_backend))
    needs_rescore = joined.filter(
        pl.col("raw_similarity").is_null() | pl.lit(not is_exact)
    )
    rescore_backend = _exact_rescore_backend(strategy.representation)
    rescored_lookup = _rescore_lookup(entry, needs_rescore, backend=rescore_backend)

    effective_top_k = _effective_top_k(strategy)
    joined = (
        joined.join(
            rescored_lookup, on=["source_id", "target_id", "country"], how="left"
        )
        .with_columns(
            (pl.col("raw_similarity").is_null() | pl.lit(not is_exact)).alias(
                "rescored"
            )
        )
        # A rescored pair takes the rescore's values alone, null when it
        # ranked beyond the cut: the run's own score is not exact there.
        .with_columns(
            pl.when(pl.col("rescored"))
            .then(pl.col("rescored_similarity"))
            .otherwise(pl.col("raw_similarity"))
            .alias("exact_similarity"),
            pl.when(pl.col("rescored"))
            .then(pl.col("rescored_rank"))
            .otherwise(pl.col("raw_rank"))
            .alias("exact_rank"),
        )
        .with_columns(
            (
                pl.col("exact_rank").is_not_null()
                & (pl.col("exact_rank") <= effective_top_k)
                & (pl.col("exact_similarity") >= strategy.min_similarity)
            ).alias("exact_kept"),
            (pl.col("exact_similarity") - strategy.min_similarity).alias(
                "distance_to_threshold"
            ),
            pl.col("found").alias("run_found"),
            pl.lit(entry.label).alias("label"),
        )
    )
    joined = joined.with_columns(
        pl.when(pl.col("run_found") & pl.col("exact_kept"))
        .then(pl.lit(VERDICT_FOUND))
        .when(pl.col("run_found") & ~pl.col("exact_kept"))
        .then(pl.lit(VERDICT_BACKEND_ONLY))
        .when(~pl.col("run_found") & pl.col("exact_kept"))
        .then(pl.lit(VERDICT_LOST_TO_BACKEND))
        .otherwise(pl.lit(VERDICT_NOT_KEPT_BY_EXACT_SCAN))
        .alias("verdict")
    )

    return (
        joined.select(list(_PAIR_AUDIT_SCHEMA.keys()))
        .cast(pl.Schema(_PAIR_AUDIT_SCHEMA))
        .sort(key)
    )


def summarize_pair_audit(pair_audit: pl.DataFrame) -> pl.DataFrame:
    """`pair_audit` (a `compute_pair_audit()` frame) reduced to one row per
    `(label, source_system, target_system, country, name_equality)`, with
    `exact_scan_recall` -- `exact_kept / truth_pairs` -- the exact scan's own
    recall at the run's settings, to read beside `pair_truth_eval`'s
    `recall` for the same population."""
    validate_blocking_artifact_schema(pair_audit, artifact_name="pair_audit")
    group = ["label", "source_system", "target_system", "country", "name_equality"]

    summary = (
        pair_audit.group_by(group)
        .agg(
            pl.len().cast(pl.Int64).alias("truth_pairs"),
            pl.col("run_found").sum().cast(pl.Int64).alias("run_found"),
            pl.col("exact_kept").sum().cast(pl.Int64).alias("exact_kept"),
            (pl.col("verdict") == VERDICT_LOST_TO_BACKEND)
            .sum()
            .cast(pl.Int64)
            .alias("lost_to_backend"),
            (pl.col("verdict") == VERDICT_NOT_KEPT_BY_EXACT_SCAN)
            .sum()
            .cast(pl.Int64)
            .alias("not_kept_by_exact_scan"),
            (pl.col("verdict") == VERDICT_BACKEND_ONLY)
            .sum()
            .cast(pl.Int64)
            .alias("backend_only"),
        )
        .with_columns(
            (pl.col("exact_kept").cast(pl.Float64) / pl.col("truth_pairs")).alias(
                "exact_scan_recall"
            )
        )
    )
    return (
        summary.select(list(_PAIR_AUDIT_SUMMARY_SCHEMA.keys()))
        .cast(pl.Schema(_PAIR_AUDIT_SUMMARY_SCHEMA))
        .sort(group)
    )


def produce_pair_audit(
    roots: WorkspaceRoots,
    location: BlockingRunLocation,
    *,
    invocation: Sequence[str] = (),
) -> tuple[BlockingAuditLocation, bool]:
    """Audit `location`'s finished run, reusing a previous audit of it.

    Returns the audit's location and whether it was reused (`True`) rather
    than freshly computed (`False`). The audit is its own production
    (`workspace.records.produce`), consuming the run itself as a recorded
    input -- immutable, so it takes no content digest -- with the run's own
    four identity keys and the rescore backend recorded as parameters, so a
    reader can see which cached target index (the run's own `index` key) an
    audit's rescoring read without a second, formally-registered reference to
    it: the run's own `RunKeys.index` already names which one it is, since
    the audit never uses another.

    Reused rather than made again exactly when the run's own inputs have not
    changed: the audit's location is keyed on the run's own key
    (`run_layout.audit_reference_for`), which a changed row, setting or
    target index moves. Checked, and returned early, before the run is even
    loaded back from disk, so a repeated audit costs one record read.
    """
    output = audit_reference_for(location)
    output_dir = locate(roots, output)
    existing = read_record(roots, output)
    if existing is not None:
        return (
            BlockingAuditLocation(
                directory=output_dir.resolve(),
                pairing=location.pairing,
                representation=location.representation,
            ),
            True,
        )

    entry = load_strategy_run_entry(roots, location)
    pair_audit = compute_pair_audit(entry)
    summary = summarize_pair_audit(pair_audit)
    validate_blocking_artifact_schema(pair_audit, artifact_name="pair_audit")
    validate_blocking_artifact_schema(summary, artifact_name="pair_audit_summary")

    run_reference = reference_at(roots, location.directory)
    with produce(
        roots,
        output,
        inputs={"run": ConsumedReference(run_reference)},
        parameters={
            "run_keys": (
                entry.result.keys.as_identity()
                if entry.result.keys is not None
                else None
            ),
            "rescore_backend": _exact_rescore_backend(
                entry.config.strategy.representation
            ),
        },
        invocation=invocation,
        key=location.directory.name,
    ) as production:
        produced_location = BlockingAuditLocation(
            directory=production.directory,
            pairing=location.pairing,
            representation=location.representation,
        )
        pair_audit.write_parquet(produced_location.path(AuditArtefact.PAIR_AUDIT))
        summary.write_parquet(produced_location.path(AuditArtefact.PAIR_AUDIT_SUMMARY))

    return resolve_audit_location(roots, location), False


def read_pair_audit(
    roots: WorkspaceRoots, location: BlockingRunLocation
) -> pl.DataFrame | None:
    """`location`'s run's `pair_audit` frame as its audit wrote it, or `None`
    when the run has not been audited. Found as the audit filed under the run's
    own key (`run_layout.audit_reference_for`), never audited here."""
    if read_record(roots, audit_reference_for(location)) is None:
        return None
    audit_location = resolve_audit_location(roots, location)
    return pl.read_parquet(audit_location.path(AuditArtefact.PAIR_AUDIT))


__all__ = [
    "VERDICT_BACKEND_ONLY",
    "VERDICT_FOUND",
    "VERDICT_LOST_TO_BACKEND",
    "VERDICT_NOT_KEPT_BY_EXACT_SCAN",
    "compute_pair_audit",
    "produce_pair_audit",
    "read_pair_audit",
    "summarize_pair_audit",
]
