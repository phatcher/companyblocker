"""Read finished blocking runs back from disk, as the comparison reads them.

A finished run is a location holding its production record
(`workspace.records.produce` writes it last), and everything a strategy
comparison needs is in that location: the record carries the settings the run
was configured with, the systems it read and when it started and finished,
and the run's own files carry what it scored. So a report over the runs a
pairing holds is built from disk alone, and making a run stays
`scripts/run_blocking.py`'s job.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from validation.contracts import ARTIFACT_SCHEMAS
from workspace.identity import RunKeys
from workspace.records import Record, read_record
from workspace.reference import locate, parse_reference, reference_at
from workspace.roots import WorkspaceRoots

from .comparison import StrategyRunEntry
from .contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingRunResult,
    BlockingStrategyConfig,
)
from .reporting import flatten_country_block
from .run_layout import (
    BlockingRunLocation,
    RunArtefact,
    iter_blocking_runs,
    read_run_location,
    select_blocking_runs,
)
from .truth import ColumnTruth, MatchedLayerTruth, SourceTruthResolver


class UnreadableRunError(ValueError):
    """A finished run whose record does not fit the configuration this code
    reads: it was made under another shape, and is made again rather than
    read through a fallback."""


def load_blocking_run_result(roots: WorkspaceRoots, run_dir: Path) -> BlockingRunResult:
    """Reconstruct the `BlockingRunResult` a finished run wrote at `run_dir`.

    `scripts/run_blocking.py` calls `write_blocking_report()` inside its own
    process before exiting, so a finished run's artefacts are all on disk.
    `source_diagnostics`/`target_diagnostics` are left `None`: no comparison
    reads them.
    """

    location = read_run_location(roots, run_dir)

    def _read_required(artefact: RunArtefact) -> pl.DataFrame:
        return pl.read_parquet(location.path(artefact))

    def _read_optional(artefact: RunArtefact) -> pl.DataFrame | None:
        path = location.path(artefact)
        return pl.read_parquet(path) if path.exists() else None

    # The summary-shaped frames are sections of `summary.json` rather than
    # one-row parquets of their own, so they are rehydrated from there. The
    # per-country evaluations come back from the nested country blocks the
    # summary records, flattened to the row shape `comparison` reads.
    summary = json.loads(location.path(RunArtefact.SUMMARY).read_text(encoding="utf-8"))

    def _frame(section: object) -> pl.DataFrame:
        rows = section if isinstance(section, list) else []
        return pl.DataFrame(rows)

    def _eval_frame(blocks: object) -> pl.DataFrame | None:
        if not isinstance(blocks, list) or not blocks:
            return None
        # The systems are recorded once at the summary's top level rather than
        # repeated on every country block; the flat row shape carries them per
        # row, so they are restored here.
        # A country block holds every population, so it flattens to several
        # rows; selecting the contract's columns keeps a rehydrated frame the
        # same shape as one `compute_pair_truth_eval` wrote.
        return pl.DataFrame(
            [
                {
                    "source_system": summary.get("source_system"),
                    "target_system": summary.get("target_system"),
                    **row,
                }
                for block in blocks
                for row in flatten_country_block(block)
            ],
            schema=ARTIFACT_SCHEMAS["pair_truth_eval"],
        )

    matched_edges = _read_required(RunArtefact.MATCHED_EDGES)
    return BlockingRunResult(
        matched_edges=matched_edges,
        raw_matched_edges=_read_required(RunArtefact.RAW_MATCHED_EDGES),
        pair_truth_eval=_eval_frame(summary.get("countries")),
        raw_pair_truth_eval=_eval_frame(summary.get("raw_countries")),
        pair_truth_eval_detail=_read_optional(RunArtefact.PAIR_TRUTH_EVAL_DETAIL),
        pruning_summary=_frame(summary.get("pruning_summary")),
        exact_match_summary=_frame(summary.get("exact_match_summary")),
        clusters=_read_required(RunArtefact.CLUSTERS),
        cluster_shape=_frame(
            [summary["cluster_shape"]] if summary.get("cluster_shape") else []
        ),
        directional_coverage=_frame(
            [summary["directional_coverage"]]
            if summary.get("directional_coverage")
            else []
        ),
        similarity_distribution=_frame(summary.get("similarity_distribution")),
        candidate_pair_count=matched_edges.height,
        # The run's keys and systems, rehydrated from the summary the run
        # wrote, so `build_strategy_comparison()` can group and refuse on them.
        keys=(RunKeys.from_identity(summary["keys"]) if summary.get("keys") else None),
        source_system=summary.get("source_system"),
        target_system=summary.get("target_system"),
        timings=tuple(
            record
            for record in (summary.get("timings") or [])
            if isinstance(record, dict)
        ),
    )


def _section(parameters: Mapping[str, object], prefix: str) -> dict[str, Any]:
    """The parameters `blocking_run_settings` filed under `<prefix>.`, by field."""
    return {
        name.removeprefix(f"{prefix}."): value
        for name, value in parameters.items()
        if name.startswith(f"{prefix}.")
    }


def _tuple_fields(dataclass_type: type, values: Mapping[str, Any]) -> dict[str, Any]:
    """`values` with each field a record holds as a JSON list given back as
    the tuple the dataclass declares."""
    tuples = {
        field.name
        for field in dataclasses.fields(dataclass_type)
        if str(field.type).startswith("tuple")
    }
    return {
        name: tuple(value) if name in tuples and isinstance(value, list) else value
        for name, value in values.items()
    }


def _truth_from(parameters: Mapping[str, object]) -> SourceTruthResolver:
    fields = _section(parameters, "truth")
    kind = fields.pop("kind", None)
    if kind == ColumnTruth(column="_").kind:
        return ColumnTruth(**fields)
    if kind == MatchedLayerTruth().kind:
        return MatchedLayerTruth(**fields)
    raise TypeError(f"unknown truth kind {kind!r}")


def blocking_run_config_from_record(
    roots: WorkspaceRoots, record: Record
) -> BlockingRunConfig:
    """The configuration a finished run records, the reverse of
    `contracts.blocking_run_settings`.

    The settings leave out locations, so each side's directory is where the
    reference the run consumed sits beneath `roots` today. A record naming a
    field the configuration no longer has, or missing one it requires, raises
    `UnreadableRunError`; a field added since the run takes its default, the
    rule the settings key keeps.
    """
    parameters = record.parameters

    def _descriptor(role: str) -> BlockingDatasetDescriptor:
        return BlockingDatasetDescriptor(
            system_dir=locate(roots, parse_reference(record.inputs[role].uri)),
            **_tuple_fields(BlockingDatasetDescriptor, _section(parameters, role)),
        )

    countries = parameters.get("countries")
    try:
        return BlockingRunConfig(
            roots=roots,
            prepared_base_dir=None,
            source=_descriptor("source"),
            target=_descriptor("target"),
            countries=tuple(countries) if isinstance(countries, list) else None,
            strategy=BlockingStrategyConfig(**_section(parameters, "strategy")),
            emit_diagnostics=bool(parameters.get("emit_diagnostics", False)),
            truth=_truth_from(parameters),
        )
    except (KeyError, TypeError) as error:
        raise UnreadableRunError(
            f"{record.uri} records settings this configuration does not read "
            f"({error}); make the run again"
        ) from error


def _record_seconds(record: Record) -> float:
    """The time between a record's start and finish: the run from the point it
    was staged, before scoring, to its record being written."""
    started = datetime.fromisoformat(record.started_at)
    finished = datetime.fromisoformat(record.finished_at)
    return (finished - started).total_seconds()


def load_strategy_run_entry(
    roots: WorkspaceRoots, location: BlockingRunLocation
) -> StrategyRunEntry:
    """One finished run as a strategy comparison reads it, from disk alone.

    Labelled `<representation>/<run key>`, the way `report_recall_curve.py`
    names a run, since a pairing holds several runs of one representation.
    `runtime_seconds` is the record's own start to finish.
    """
    record = read_record(roots, reference_at(roots, location.directory))
    if record is None:
        raise UnreadableRunError(f"{location.directory} holds no finished run")
    return StrategyRunEntry(
        label=f"{location.representation}/{location.directory.name}",
        config=blocking_run_config_from_record(roots, record),
        result=load_blocking_run_result(roots, location.directory),
        runtime_seconds=_record_seconds(record),
        finished_at=record.finished_at,
    )


def stored_pairings(roots: WorkspaceRoots) -> list[tuple[str, str]]:
    """Every source and target pair a finished run is on disk for, as each
    run's record names them, sorted."""
    pairings: set[tuple[str, str]] = set()
    for location in iter_blocking_runs(roots):
        record = read_record(roots, reference_at(roots, location.directory))
        if record is None:
            continue
        parameters = record.parameters
        if "source.system" not in parameters or "target.system" not in parameters:
            raise UnreadableRunError(
                f"{record.uri} records no source or target system; make the run again"
            )
        pairings.add(
            (str(parameters["source.system"]), str(parameters["target.system"]))
        )
    return sorted(pairings)


def load_stored_strategy_entries(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    target_system: str,
    representation: str | None = None,
) -> tuple[list[StrategyRunEntry], list[UnreadableRunError]]:
    """Every finished run one pairing holds, as comparison entries, and the
    runs that could not be read, each as the error that says why."""
    entries: list[StrategyRunEntry] = []
    unreadable: list[UnreadableRunError] = []
    for location in select_blocking_runs(
        roots,
        source_system=source_system,
        target_system=target_system,
        representation=representation,
    ):
        try:
            entries.append(load_strategy_run_entry(roots, location))
        except UnreadableRunError as error:
            unreadable.append(error)
    return entries, unreadable


__all__ = [
    "UnreadableRunError",
    "blocking_run_config_from_record",
    "load_blocking_run_result",
    "load_stored_strategy_entries",
    "load_strategy_run_entry",
    "stored_pairings",
]
