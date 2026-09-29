"""Write a blocking run's artefacts, and make a run a production that is finished or absent.

`write_blocking_report()` validates every frame against its schema and writes each as
parquet beside `summary.json`, which `build_blocking_summary()` builds: per-country
results, the run's keys, its name transform, its phase timings, and
`evaluation_skipped` when nothing was measured. Writing is a separate step;
`execute_blocking_run()` never calls it.

`produce_blocking_run()` stages a run through `workspace.records.produce`. It is entered
before scoring, so a run naming a missing input is refused before any work; artefacts
are written into a staging directory and the record written last, so a run that
fails or is interrupted leaves no location and no record, and a run is finished
exactly when it holds one. The record's inputs (`blocking_run_inputs()`) are the source
and target datasets with their population keys, so the catalog lists every run that
consumed a dataset. `write_run_manifest()` records beside the artefacts the keys, the
settings they digest, the commit, and each setting's value and whether it was given
or defaulted.

The reports spanning runs (pair recovery, name-equality attribution, pair outcomes,
recall curves, strategy comparisons and their aggregate) are written by their own
functions, never by `write_blocking_report()`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from validation.contracts import POPULATION_UNIVERSE, validate_artifact_schema
from validation.runner import NAME_EQUALITY_LEVELS, NAME_EQUALITY_UNKNOWN
from workspace.data_layout import dataset_reference_at, perturbed_dataset_reference
from workspace.identity import ConsumedReference, RunKeys, digest_directory
from workspace.kind_layout import Kind
from workspace.records import produce
from workspace.reference import (
    InvalidReferenceError,
    Reference,
    Side,
    locate,
    reference,
)
from workspace.roots import WorkspaceRoots
from workspace.run_inputs import is_perturbed_source, perturbed_parts

from .contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingRunResult,
    blocking_run_settings,
    validate_blocking_artifact_schema,
)
from .run_layout import (
    BlockingComparisonLocation,
    BlockingRunLocation,
    ComparisonArtefact,
    RunArtefact,
    resolve_run_location_for,
    run_key,
    run_reference,
)

# The figures every population carries, in one shape, so the universe and each
# level are compared like with like. `confusion` nests `tp`/`fp`/`fn`.
_POPULATION_FIGURES: tuple[str, ...] = (
    "labelled_sources",
    "target_rows",
    "pair_universe",
    "truth_pairs",
    "predicted_pairs",
    "precision",
    "recall",
    "reduction_ratio",
    "recall_at_k",
    "candidate_pair_count",
    "source_rows",
    "candidate_set_size_ratio",
)
_CONFUSION_CELLS: tuple[str, ...] = ("tp", "fp", "fn")
# Derived on the levels for a person reading them in order, never stored.
_CASCADE_FIELDS: tuple[str, ...] = (
    "unresolved_before",
    "removed_here",
    "unresolved_after",
)


def _population_block(row: Mapping[str, object]) -> dict[str, object]:
    """One `pair_truth_eval` population row as a block: its figures, with the
    three matrix cells nested so a reader sees the matrix as one thing."""
    block: dict[str, object] = {"population": row.get("population")}
    for figure in _POPULATION_FIGURES:
        block[figure] = row.get(figure)
    block["confusion"] = {cell: row.get(cell) for cell in _CONFUSION_CELLS}
    return block


def _difficulty_levels(
    level_rows: Mapping[str, Mapping[str, object]],
    universe_truth_pairs: object,
) -> list[dict[str, object]]:
    """The cleansing levels in order, each a full population block.

    The levels are a cascade: each holds the pairs the level before it did not
    resolve. Listing them in order restores that, and each carries the drop-off
    beside its own matrix -- how many truth pairs were still unresolved when it
    ran, how many it took out, and how many remained -- so a reader sees how
    much each level of cleansing closed and how blocking did on what was left.
    """
    levels: list[dict[str, object]] = []
    remaining = universe_truth_pairs
    for name in NAME_EQUALITY_LEVELS:
        row = level_rows.get(name)
        if row is None:
            continue
        level = _population_block(row)
        truth_pairs = row.get("truth_pairs")
        if isinstance(remaining, int) and isinstance(truth_pairs, int):
            level["unresolved_before"] = remaining
            level["removed_here"] = truth_pairs
            remaining -= truth_pairs
            level["unresolved_after"] = remaining
        levels.append(level)
    return levels


def build_country_blocks(
    pair_truth_eval: pl.DataFrame | None,
) -> list[dict[str, object]] | None:
    """One block per scored country: its universe, then its cleansing levels in
    order, then the pairs no verdict was possible for, each in one shape.

    `None` when the run had no ground truth, matching the frame it reads. A run
    given no name forms has a universe and no levels.
    """
    if pair_truth_eval is None:
        return None
    blocks: list[dict[str, object]] = []
    countries = pair_truth_eval.get_column("country").unique(maintain_order=True)
    for country in countries.to_list():
        rows = {
            str(row["population"]): row
            for row in pair_truth_eval.filter(pl.col("country") == country).iter_rows(
                named=True
            )
        }
        universe = rows.get(POPULATION_UNIVERSE)
        block: dict[str, object] = {
            "country": country,
            "whole_population": (
                _population_block(universe) if universe is not None else None
            ),
            "name_equality_levels": _difficulty_levels(
                rows, universe.get("truth_pairs") if universe is not None else None
            ),
        }
        unknown = rows.get(NAME_EQUALITY_UNKNOWN)
        if unknown is not None:
            block["unknown"] = _population_block(unknown)
        blocks.append(block)
    return blocks


def flatten_country_block(block: Mapping[str, object]) -> list[dict[str, object]]:
    """The inverse of `build_country_blocks`: one country block back to its
    `pair_truth_eval` rows, one per population.

    Kept beside the function that nests, so the two cannot drift. A consumer
    that reads the flat rows -- the strategy comparison, which aggregates
    across runs -- rehydrates through this rather than learning the nested
    shape, and a consumer reading a run on its own reads the nesting directly.
    """
    blocks: list[object] = [block.get("whole_population")]
    levels = block.get("name_equality_levels")
    if isinstance(levels, list):
        blocks.extend(levels)
    blocks.append(block.get("unknown"))

    rows: list[dict[str, object]] = []
    for population in blocks:
        if not isinstance(population, Mapping):
            continue
        row: dict[str, object] = {"country": block.get("country")}
        for key, value in population.items():
            if key == "confusion" and isinstance(value, Mapping):
                row.update(dict(value))
            elif key not in _CASCADE_FIELDS:
                row[key] = value
        rows.append(row)
    return rows


def build_blocking_summary(result: BlockingRunResult) -> dict[str, object]:
    """Build a serialisable summary for a blocking run.

    Nested rather than flat: a per-run summary is read by a person, so the
    difficulty cascade is a list of levels (see `_difficulty_levels`) instead
    of the `<level>_<field>` columns a parquet row is forced into.
    """

    return {
        # Recorded on the result rather than read from a metrics frame, so a
        # run made without ground truth still says what it was.
        "source_system": result.source_system,
        "target_system": result.target_system,
        # Explicit, so a run made with --ground-truth false is
        # readable afterwards as one that was asked to skip evaluation,
        # rather than only implied by an absent countries block -- the same
        # condition build_country_blocks reads to return None.
        "evaluation_skipped": result.pair_truth_eval is None,
        "candidate_pair_count": result.candidate_pair_count,
        "raw_candidate_pair_count": result.raw_matched_edges.height,
        # "cluster_shape" (unqualified) is the clustering that
        # actually ran -- source-to-target edges unioned with any
        # target-neighbour edges the strategy's linking settings produced --
        # and is identical to "cluster_shape_before_union" whenever that
        # union was empty (the feature off, the default). The largest
        # component this item's own Done-when clause names is
        # "max_cluster_size" on each of the two blocks; the threshold and
        # cap that produced the union are "target_neighbor_summary" below.
        "cluster_shape": (
            result.cluster_shape.row(0, named=True)
            if result.cluster_shape.height
            else None
        ),
        "cluster_shape_before_union": (
            result.cluster_shape_before_union.row(0, named=True)
            if result.cluster_shape_before_union.height
            else None
        ),
        "target_neighbor_summary": result.target_neighbor_summary.to_dicts(),
        "directional_coverage": (
            result.directional_coverage.row(0, named=True)
            if result.directional_coverage.height
            else None
        ),
        "pruning_summary": result.pruning_summary.to_dicts(),
        "exact_match_summary": result.exact_match_summary.to_dicts(),
        "similarity_distribution": result.similarity_distribution.to_dicts(),
        # Each scored country as one block: the whole-population matrix, then
        # the difficulty cascade beneath it. The headline "what blocking had
        # to earn" figure is the last level's, the pairs still unequal after
        # cleansing -- neither `exact_match_summary`'s row-count split nor
        # `pair_truth_eval_detail`'s all-buckets pair count is that figure.
        "countries": build_country_blocks(result.pair_truth_eval),
        "raw_countries": build_country_blocks(result.raw_pair_truth_eval),
        # The four keys this run is identified by (`workspace.identity`).
        "keys": result.keys.as_identity() if result.keys is not None else None,
        # The function both sides' names passed through before comparison,
        # so a reader knows what was compared without the configuration.
        "name_transform": dict(result.name_transform),
        # Each phase's wall-clock, resident set size and CPU share, per
        # country, in the order they ended (`workspace.telemetry`).
        "timings": [dict(record) for record in result.timings],
    }


def write_blocking_report(
    location: BlockingRunLocation, result: BlockingRunResult
) -> list[Path]:
    """Write notebook-friendly blocking artefacts to disk.

    `source_diagnostics.parquet`/`target_diagnostics.parquet` are
    only written when `result.source_diagnostics`/`target_diagnostics` are
    populated, i.e. the run was executed with
    `BlockingRunConfig.emit_diagnostics=True` -- omitted entirely otherwise,
    matching every other conditionally-populated artefact this function
    writes.
    """

    location.directory.mkdir(parents=True, exist_ok=True)

    validate_blocking_artifact_schema(
        result.matched_edges, artifact_name="matched_edges"
    )
    validate_blocking_artifact_schema(
        result.raw_matched_edges, artifact_name="raw_matched_edges"
    )
    validate_blocking_artifact_schema(
        result.pruning_summary, artifact_name="pruning_summary"
    )
    validate_blocking_artifact_schema(
        result.exact_match_summary, artifact_name="exact_match_summary"
    )
    validate_blocking_artifact_schema(
        result.similarity_distribution, artifact_name="similarity_distribution"
    )
    validate_blocking_artifact_schema(
        result.directional_coverage, artifact_name="directional_coverage"
    )
    validate_artifact_schema(result.clusters, artifact_name="clusters")
    validate_artifact_schema(result.cluster_shape, artifact_name="cluster_shape")
    # `clusters_before_union`/`cluster_shape_before_union` are the
    # same shapes `clusters`/`cluster_shape` already validate against --
    # source-to-target-only clustering is a `clusters` frame like any
    # other, it is only the edges feeding it that differ.
    validate_artifact_schema(result.clusters_before_union, artifact_name="clusters")
    validate_artifact_schema(
        result.cluster_shape_before_union, artifact_name="cluster_shape"
    )
    validate_blocking_artifact_schema(
        result.target_neighbor_edges, artifact_name="target_neighbor_edges"
    )
    validate_blocking_artifact_schema(
        result.target_neighbor_summary, artifact_name="target_neighbor_summary"
    )
    if result.pair_truth_eval is not None:
        validate_artifact_schema(
            result.pair_truth_eval, artifact_name="pair_truth_eval"
        )
    if result.raw_pair_truth_eval is not None:
        validate_artifact_schema(
            result.raw_pair_truth_eval, artifact_name="pair_truth_eval"
        )
    if result.pair_truth_eval_detail is not None:
        validate_blocking_artifact_schema(
            result.pair_truth_eval_detail, artifact_name="pair_truth_eval_detail"
        )
    if result.source_diagnostics is not None:
        validate_blocking_artifact_schema(
            result.source_diagnostics, artifact_name="source_diagnostics"
        )
    if result.target_diagnostics is not None:
        validate_blocking_artifact_schema(
            result.target_diagnostics, artifact_name="target_diagnostics"
        )

    written: list[Path] = []

    def _write(frame: pl.DataFrame, artefact: RunArtefact) -> None:
        path = location.path(artefact)
        frame.write_parquet(path)
        written.append(path)

    # Row-level data only. Every summary this run produced -- the pruning and
    # exact-match splits, the cluster shape, the directional coverage, and both
    # pair-truth evaluations with their difficulty cascades -- is a section of
    # `summary.json` instead of a one-row parquet of its own. A single-row
    # parquet is the worst of both: a person cannot read it, and aggregating it
    # across runs means opening every run directory.
    _write(result.matched_edges, RunArtefact.MATCHED_EDGES)
    _write(result.raw_matched_edges, RunArtefact.RAW_MATCHED_EDGES)
    _write(result.clusters, RunArtefact.CLUSTERS)
    # Written unconditionally, like `clusters` itself -- empty
    # (matching `clusters`/`clusters` exactly) rather than omitted whenever
    # the strategy leaves target-neighbour linking off, so a reader never has
    # to branch on whether the feature was on to find these files.
    _write(result.clusters_before_union, RunArtefact.CLUSTERS_BEFORE_UNION)
    _write(result.target_neighbor_edges, RunArtefact.TARGET_NEIGHBOR_EDGES)
    if result.pair_truth_eval_detail is not None:
        _write(result.pair_truth_eval_detail, RunArtefact.PAIR_TRUTH_EVAL_DETAIL)
    if result.source_diagnostics is not None:
        _write(result.source_diagnostics, RunArtefact.SOURCE_DIAGNOSTICS)
    if result.target_diagnostics is not None:
        _write(result.target_diagnostics, RunArtefact.TARGET_DIAGNOSTICS)

    summary_path = location.path(RunArtefact.SUMMARY)
    summary_path.write_text(
        json.dumps(build_blocking_summary(result), indent=2, default=str),
        encoding="utf-8",
    )
    written.append(summary_path)

    return written


def write_run_manifest(
    location: BlockingRunLocation,
    *,
    identity: Mapping[str, object],
    configuration: Mapping[str, object],
    roots: WorkspaceRoots,
    commit: str | None,
) -> Path:
    """Write `manifest.json`, what a run directory says about how it was made.

    `identity` is the mapping the directory's key is digested from
    (`contracts.build_blocking_run_identity`), so a caller who knows a run's
    parameters can find the run by them rather than by its key. `configuration`
    is the settings that applied to the run, each with the value it took and
    whether that value was given or defaulted. `roots` is written as
    `workspace_roots`, each root's path and source, so the run says where it
    read from and wrote to. `commit` is the checkout's commit, provenance
    that no key reads.
    """
    location.directory.mkdir(parents=True, exist_ok=True)
    path = location.path(RunArtefact.MANIFEST)
    path.write_text(
        json.dumps(
            {
                "identity": dict(identity),
                "configuration": dict(configuration),
                "workspace_roots": roots.to_manifest(),
                "commit": commit,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _dataset_reference(
    roots: WorkspaceRoots, descriptor: BlockingDatasetDescriptor
) -> Reference:
    """The dataset a descriptor's rows were read from, as the reference that
    locates it."""
    try:
        return dataset_reference_at(roots, descriptor.system_dir)
    except InvalidReferenceError as error:
        raise ValueError(
            f"{descriptor.system_dir} holds no registered dataset, so a run "
            f"reading it cannot record what it consumed: {error}"
        ) from error


def _profile_input(
    roots: WorkspaceRoots, *, name: str, version: str
) -> ConsumedReference:
    """The profile a perturbed dataset was generated from: the promoted
    version where it exists, otherwise its draft, which is mutable and so
    brings a digest of its content."""
    promoted = reference(Kind.PERTURBATION, Side.PROFILE, name=name, version=version)
    if locate(roots, promoted).is_dir():
        return ConsumedReference(promoted)
    draft = reference(
        Kind.PERTURBATION, Side.PROFILE, name=name, version=version, stage="draft"
    )
    draft_dir = locate(roots, draft)
    if draft_dir.is_dir():
        return ConsumedReference(draft, digest_directory(draft_dir))
    return ConsumedReference(promoted)


def blocking_run_inputs(
    config: BlockingRunConfig, *, keys: RunKeys
) -> dict[str, ConsumedReference]:
    """What a run over `config` consumes, by role, for its production record.

    Each side is the layer snapshot its rows were read from, with the side's
    population key as the content digest: the key is hashed from the rows the
    run consumed, which is what a mutable layer's digest has to say. A
    perturbed source is its dataset's reference instead, together with the
    profile that dataset was generated from.
    """
    roots = config.roots
    inputs = {
        "target": ConsumedReference(
            _dataset_reference(roots, config.target), keys.target_population
        )
    }
    if is_perturbed_source(config.source.system):
        source_system, profile_id, version, seed = perturbed_parts(config.source.system)
        inputs["source"] = ConsumedReference(
            perturbed_dataset_reference(
                source_system=source_system,
                profile_id=profile_id,
                version=version,
                seed=seed,
            ),
            keys.source_population,
        )
        inputs["source_profile"] = _profile_input(
            roots, name=profile_id, version=version
        )
    else:
        inputs["source"] = ConsumedReference(
            _dataset_reference(roots, config.source), keys.source_population
        )
    return inputs


@contextmanager
def produce_blocking_run(
    config: BlockingRunConfig, *, keys: RunKeys, invocation: Sequence[str]
) -> Iterator[BlockingRunLocation]:
    """Stage one run through `workspace.records.produce`, yielding the staged
    location to write its artefacts into.

    Entered before scoring, so a run naming an input that does not exist is
    refused before any work, and a run that fails or is interrupted leaves no
    location and no record. On success the record is written last, under the
    run's own key, naming what `blocking_run_inputs` gives and the run's
    settings as its parameters, and the staged directory becomes the run's.
    """
    final = resolve_run_location_for(config, keys=keys)
    output = run_reference(
        source_system=config.source.system,
        target_system=config.target.system,
        representation=config.strategy.representation,
        keys=keys,
    )
    with produce(
        config.roots,
        output,
        inputs=blocking_run_inputs(config, keys=keys),
        parameters=blocking_run_settings(config),
        invocation=invocation,
        key=run_key(keys),
    ) as production:
        yield BlockingRunLocation(
            directory=production.directory,
            pairing=final.pairing,
            representation=final.representation,
        )


def write_strategy_comparison_report(
    location: BlockingComparisonLocation, comparison: pl.DataFrame
) -> Path:
    """Write a `comparison.build_strategy_comparison()` frame to disk.

    Same schema-validate-then-write convention as `write_blocking_report()`,
    for the one additional artefact a strategy comparison produces
    (`BLOCKING_ARTIFACT_SCHEMAS["strategy_comparison"]`). Kept as a separate
    entry point rather than folded into `write_blocking_report()` since a
    comparison spans multiple runs/output directories, not the single run
    `write_blocking_report()` writes for.
    """

    location.directory.mkdir(parents=True, exist_ok=True)
    validate_blocking_artifact_schema(comparison, artifact_name="strategy_comparison")

    path = location.path(ComparisonArtefact.REPORT)
    comparison.write_parquet(path)
    return path


def write_pair_recovery_report(
    location: BlockingComparisonLocation, recovery: pl.DataFrame
) -> Path:
    """Write a `comparison.diff_pair_recovery()` frame to disk.

    Same schema-validate-then-write convention as
    `write_strategy_comparison_report()`, kept as its own entry point for the
    same reason: a recovery report spans two runs' output directories, not
    the single run `write_blocking_report()` writes for.
    """

    location.directory.mkdir(parents=True, exist_ok=True)
    validate_blocking_artifact_schema(
        recovery, artifact_name="pair_recovery_attribution"
    )

    path = location.path(ComparisonArtefact.PAIR_RECOVERY_ATTRIBUTION)
    recovery.write_parquet(path)
    return path


def write_name_equality_attribution_report(
    location: BlockingComparisonLocation,
    *,
    attribution: pl.DataFrame,
    transitions: pl.DataFrame,
) -> list[Path]:
    """Write a `comparison.diff_name_equality_levels()` frame and its
    `tally_name_equality_transitions()` tally, schema-validated, as
    `name_equality_attribution.parquet` and `name_equality_transitions.parquet`.

    Its own entry point for the reason `write_pair_recovery_report()` is: the
    attribution spans two runs' output directories, and belongs beside the
    comparison it explains.
    """

    location.directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for artifact_name, artefact, frame in (
        (
            "name_equality_attribution",
            ComparisonArtefact.NAME_EQUALITY_ATTRIBUTION,
            attribution,
        ),
        (
            "name_equality_transitions",
            ComparisonArtefact.NAME_EQUALITY_TRANSITIONS,
            transitions,
        ),
    ):
        validate_blocking_artifact_schema(frame, artifact_name=artifact_name)
        path = location.path(artefact)
        frame.write_parquet(path)
        written.append(path)
    return written


def write_pair_outcomes_report(
    location: BlockingComparisonLocation,
    *,
    runs: pl.DataFrame,
    outcomes: pl.DataFrame,
    candidates: pl.DataFrame,
    source_candidates: pl.DataFrame,
    recall_curves: pl.DataFrame,
    audits: pl.DataFrame,
) -> list[Path]:
    """Write a `comparison.build_pair_outcome_runs()` frame and the
    `build_pair_outcomes()`, `build_pair_outcome_candidates()`,
    `build_pair_outcome_source_candidates()` and
    `build_pair_outcome_recall_curves()` frames that join to it, plus the
    `pair_audit` rows of whichever runs have been audited, schema-validated,
    as `pair_outcome_runs.parquet`, `pair_outcomes.parquet`,
    `pair_outcome_candidates.parquet`,
    `pair_outcome_source_candidates.parquet`,
    `pair_outcome_recall_curves.parquet` and `pair_outcome_audits.parquet`.

    One entry point for all six, and all are validated before any is
    written, since the others are read through the runs: a file of one
    from another moment than the others would join a pair's verdict, or a
    run's cutoffs, to the wrong run's settings, or to none.

    Raises `ValueError` if an outcome, candidate, curve or audit row names a
    run the runs table lacks.
    """

    validate_blocking_artifact_schema(runs, artifact_name="pair_outcome_runs")
    validate_blocking_artifact_schema(outcomes, artifact_name="pair_outcomes")
    validate_blocking_artifact_schema(
        candidates, artifact_name="pair_outcome_candidates"
    )
    validate_blocking_artifact_schema(
        source_candidates, artifact_name="pair_outcome_source_candidates"
    )
    validate_blocking_artifact_schema(
        recall_curves, artifact_name="pair_outcome_recall_curves"
    )
    validate_blocking_artifact_schema(audits, artifact_name="pair_audit")
    for name, frame in (
        ("pair outcomes", outcomes),
        ("pair outcome candidates", candidates),
        ("pair outcome source candidates", source_candidates),
        ("pair outcome recall curves", recall_curves),
        ("pair outcome audits", audits),
    ):
        unknown = frame.select("label").unique().join(runs, on="label", how="anti")
        if unknown.height:
            raise ValueError(
                f"{name} name runs the runs table does not hold: "
                f"{sorted(unknown.get_column('label').to_list())!r}"
            )

    location.directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for artefact, frame in (
        (ComparisonArtefact.PAIR_OUTCOME_RUNS, runs),
        (ComparisonArtefact.PAIR_OUTCOMES, outcomes),
        (ComparisonArtefact.PAIR_OUTCOME_CANDIDATES, candidates),
        (ComparisonArtefact.PAIR_OUTCOME_SOURCE_CANDIDATES, source_candidates),
        (ComparisonArtefact.PAIR_OUTCOME_RECALL_CURVES, recall_curves),
        (ComparisonArtefact.PAIR_OUTCOME_AUDITS, audits),
    ):
        path = location.path(artefact)
        frame.write_parquet(path)
        written.append(path)
    return written


def write_recall_curve_report(
    location: BlockingRunLocation, *, curve: pl.DataFrame, summary: pl.DataFrame
) -> list[Path]:
    """Write `validation.recall_curve.compute_recall_curve()`'s per-run
    artefacts: `recall_curve.parquet`, one row per population per budget
    step, and `recall_curve_summary.parquet`, the one-row normalised area
    per name-equality level.

    Validated against `validation.contracts.ARTIFACT_SCHEMAS` rather than
    `blocking`'s own registry: a recall curve is a second reading of a
    completed run's own `matched_edges`/`pair_truth_eval_detail`, not a new
    candidate artefact blocking produces, and its schema lives beside
    `pair_truth_eval`'s in the area that owns the populations it reports
    over.
    """
    location.directory.mkdir(parents=True, exist_ok=True)
    validate_artifact_schema(curve, artifact_name="recall_curve")
    validate_artifact_schema(summary, artifact_name="recall_curve_summary")

    written: list[Path] = []
    for frame, artefact in (
        (curve, RunArtefact.RECALL_CURVE),
        (summary, RunArtefact.RECALL_CURVE_SUMMARY),
    ):
        path = location.path(artefact)
        frame.write_parquet(path)
        written.append(path)
    return written


def write_strategy_comparison_aggregate(
    location: BlockingComparisonLocation,
    *,
    combined: pl.DataFrame,
    runtime_scaling: pl.DataFrame,
) -> list[Path]:
    """Write a cross-pair comparison aggregate and its scaling summary.

    Both frames get a `.parquet` and a `.csv`, unlike the single-pair
    `write_strategy_comparison_report()` above, which writes only parquet and
    leaves the CSV to its caller. The CSV is the point here rather than a
    convenience: this aggregate exists to be read across countries and
    backends at a glance, and the frames are small enough (one row per run
    leg, and one per grouping bucket) that a spreadsheet is a reasonable
    reader.
    """

    location.directory.mkdir(parents=True, exist_ok=True)
    validate_blocking_artifact_schema(
        combined, artifact_name="strategy_comparison_combined"
    )
    validate_blocking_artifact_schema(
        runtime_scaling, artifact_name="strategy_comparison_runtime_scaling"
    )

    written: list[Path] = []
    for frame, parquet, csv in (
        (combined, ComparisonArtefact.COMBINED, ComparisonArtefact.COMBINED_CSV),
        (
            runtime_scaling,
            ComparisonArtefact.RUNTIME_SCALING,
            ComparisonArtefact.RUNTIME_SCALING_CSV,
        ),
    ):
        parquet_path = location.path(parquet)
        frame.write_parquet(parquet_path)
        written.append(parquet_path)

        csv_path = location.path(csv)
        frame.write_csv(csv_path)
        written.append(csv_path)

    return written
