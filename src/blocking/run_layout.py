"""Blocking's view of where a run's output sits and what a run holds.

A run is a `blocking://<source>/<target>/<representation>/.../<key>` reference,
one layout per kind of source, which `workspace.reference` registers and
locates; this
module builds that reference from a run's systems, representation and identity
(`workspace.identity`), and names every artefact inside a run directory. A
caller asks for a location and reads its members; nothing outside this module
spells an artefact's file name, and nothing here or beyond composes a run's
directory, so renaming an output or adding a level to the tree moves every
reader with the writer instead of leaving readers asking for files that no
longer exist.

The tree orders those segments its own way, which the URI does not say.
Target first because every run scores against one system's primary records and
the point of the tree is to read everything scored against one target together;
kind next so the resilience ladder's tiers sit beside each other; the source's
own single segment after it; then the representation, and last the digest of the
run's four identity keys. One pairing's cross-method comparison report sits
beside that pairing's representations, and the report combining every pairing
sits beside blocking's data directory rather than inside it, as does every
measurement script's report.

Reading, not only writing, goes through here: `iter_blocking_runs` and
`iter_blocking_comparisons` hand a caller the runs themselves, each with its
target, source kind, source and representation already separated out, so no
caller holds a glob or counts directory levels of its own.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from workspace.artifact_archive import compute_artifact_signature, resolve_candidate_dir
from workspace.artifact_layout import blocking_artifact_root
from workspace.identity import RunKeys
from workspace.kind_layout import Kind
from workspace.records import read_record
from workspace.reference import (
    BLOCKING_NAMES,
    BLOCKING_PERTURBED,
    BLOCKING_PLAIN,
    InvalidReferenceError,
    Layout,
    Reference,
    Side,
    locate,
    reference,
    reference_at,
    select_references,
    selection_directory,
)
from workspace.roots import WorkspaceRoots
from workspace.run_inputs import is_perturbed_source, perturbed_parts

from .contracts import BlockingRunConfig

RUN_KEY_DIGEST_SIZE = 6
"""Bytes of the run key's digest -- 12 hex characters in the directory name.

A display choice, not a collision policy: the full signature covers the run's
four keys, and this only decides how much of it names the directory. Twelve
characters is what this repo's run directories carried before the key was
derived here, and it is short enough to compare two runs by eye, which a
32-character digest is not.
"""

COMPARISON_DIR_NAME = "comparison"
"""The directory holding a comparison report: beside a pairing's runs for the
report comparing them, and beside blocking's data directory for the report
combining every pairing."""

MEASUREMENTS_DIR_NAME = "measurements"
"""The directory holding every measurement script's reports, beside blocking's
data directory: a measurement scores a sampled corpus in memory and is no run
of the tree."""

MEASUREMENT_REPORT_NAME = "report.json"
"""The file one measurement's report is written to, inside its keyed directory."""


class BlockingSourceKind(StrEnum):
    """Which tier of the resilience ladder a run's source rows are.

    Every run scores against one target system's primary records, so the
    kind names only the source: the system's own entity rows (`DATA`), its
    recorded name variants (`NAMES`), or a perturbed set made from it
    (`PERTURBED`). It is a level of the run tree, so every run against one
    target sits under that target and the three tiers sit beside each other,
    which is what lets one glob read every method's result for one dataset
    or every dataset's result for one method.
    """

    DATA = "data"
    NAMES = "names"
    PERTURBED = "perturbed"
    PERTURBED_NAMES = "perturbed-names"
    BENCHMARK = "benchmark"


class RunArtefact(StrEnum):
    """Every file a run directory holds, and the name it holds it under.

    The one list of a run's contents. A writer asks a location for the
    artefact it is about to write and a reader for the one it wants, so a
    rename here reaches both at once and an artefact this area does not write
    cannot be asked for at all.
    """

    SUMMARY = "summary.json"
    MANIFEST = "manifest.json"
    MATCHED_EDGES = "matched_edges.parquet"
    RAW_MATCHED_EDGES = "raw_matched_edges.parquet"
    CLUSTERS = "clusters.parquet"
    CLUSTERS_BEFORE_UNION = "clusters_before_union.parquet"
    TARGET_NEIGHBOR_EDGES = "target_neighbor_edges.parquet"
    PAIR_TRUTH_EVAL = "pair_truth_eval.parquet"
    PAIR_TRUTH_EVAL_DETAIL = "pair_truth_eval_detail.parquet"
    SOURCE_DIAGNOSTICS = "source_diagnostics.parquet"
    TARGET_DIAGNOSTICS = "target_diagnostics.parquet"
    RECALL_CURVE = "recall_curve.parquet"
    RECALL_CURVE_SUMMARY = "recall_curve_summary.parquet"


class AuditArtefact(StrEnum):
    """Every file a run's audit directory holds."""

    PAIR_AUDIT = "pair_audit.parquet"
    PAIR_AUDIT_SUMMARY = "pair_audit_summary.parquet"


class ComparisonArtefact(StrEnum):
    """Every file a comparison report directory holds.

    One pairing's directory and the combined one across pairings hold the same
    artefacts, which is what lets the combined report be read the same way as
    the reports it concatenates.
    """

    REPORT = "strategy_comparison.parquet"
    REPORT_CSV = "strategy_comparison.csv"
    FAILURES = "strategy_comparison_failures.jsonl"
    SUITE_STATUS = "baseline_suite_status.json"
    PAIR_RECOVERY_ATTRIBUTION = "pair_recovery_attribution.parquet"
    NAME_EQUALITY_ATTRIBUTION = "name_equality_attribution.parquet"
    NAME_EQUALITY_TRANSITIONS = "name_equality_transitions.parquet"
    PAIR_OUTCOME_RUNS = "pair_outcome_runs.parquet"
    PAIR_OUTCOME_RUNS_CSV = "pair_outcome_runs.csv"
    PAIR_OUTCOMES = "pair_outcomes.parquet"
    PAIR_OUTCOME_CANDIDATES = "pair_outcome_candidates.parquet"
    PAIR_OUTCOME_SOURCE_CANDIDATES = "pair_outcome_source_candidates.parquet"
    PAIR_OUTCOME_RECALL_CURVES = "pair_outcome_recall_curves.parquet"
    PAIR_OUTCOME_AUDITS = "pair_outcome_audits.parquet"
    COMBINED = "strategy_comparison_combined.parquet"
    COMBINED_CSV = "strategy_comparison_combined.csv"
    RUNTIME_SCALING = "strategy_comparison_runtime_scaling.parquet"
    RUNTIME_SCALING_CSV = "strategy_comparison_runtime_scaling.csv"


@dataclass(frozen=True, slots=True)
class BlockingPairing:
    """One source system scored against one target, as the tree files it."""

    target_system: str
    source_kind: BlockingSourceKind
    source_segment: str

    @property
    def label(self) -> str:
        """The `<target>/<kind>/<source>` a pairing's output is filed under,
        the one form a report groups or reports pairings by."""
        return f"{self.target_system}/{self.source_kind.value}/{self.source_segment}"


@dataclass(frozen=True, slots=True)
class BlockingRunLocation:
    """Where one run's artefacts are, and which run they belong to.

    Handed out by `resolve_run_location` to a writer and by
    `iter_blocking_runs` to a reader, so both name a file the same way: through
    `path()`, never by joining one.
    """

    directory: Path
    pairing: BlockingPairing
    representation: str

    def path(self, artefact: RunArtefact) -> Path:
        """Where one of this run's artefacts is."""
        return self.directory / artefact.value


@dataclass(frozen=True, slots=True)
class BlockingAuditLocation:
    """Where one run's audit is, beside the run rather than inside it.

    Handed out by `resolve_audit_location` to `audit.produce_pair_audit`, the
    location's only writer, so a reader names one the same way, through
    `path()`.
    """

    directory: Path
    pairing: BlockingPairing
    representation: str

    def path(self, artefact: AuditArtefact) -> Path:
        """Where one of this audit's artefacts is."""
        return self.directory / artefact.value


@dataclass(frozen=True, slots=True)
class BlockingComparisonLocation:
    """Where a comparison report's artefacts are.

    `pairing` is the pairing whose methods the report compares, and `None` for
    the report combining every pairing, which describes no single one.
    """

    directory: Path
    pairing: BlockingPairing | None

    def path(self, artefact: ComparisonArtefact) -> Path:
        """Where one of this report's artefacts is."""
        return self.directory / artefact.value


PERTURBED_SOURCE_SEPARATOR = "_"
"""What joins a perturbed source's parts into its one segment. No system code
or profile name holds it, so the segment splits back into exactly its parts."""


def blocking_source_placement(source_system: str) -> tuple[BlockingSourceKind, str]:
    """The kind a source system string names, and the single segment that
    stands for it in a run's reference.

    A perturbed dataset's URI (`perturbed://<system>/<profile>/<version>/<seed>`) is the
    perturbed kind with its parts joined by `PERTURBED_SOURCE_SEPARATOR`,
    `ie_en-lite_v1_42`, one segment so every kind sits at the same depth and a
    glob across kinds lines up; anything else is the system's own data. The
    identity keeps the source as it is, this is only where the run is filed.
    """
    if is_perturbed_source(source_system):
        system, profile_id, version, seed = perturbed_parts(source_system)
        return BlockingSourceKind.PERTURBED, PERTURBED_SOURCE_SEPARATOR.join(
            (system, profile_id, version, str(seed))
        )
    return BlockingSourceKind.DATA, source_system


def blocking_pairing(*, source_system: str, target_system: str) -> BlockingPairing:
    """The pairing one run belongs to, from the two systems it scores."""
    kind, source_segment = blocking_source_placement(source_system)
    return BlockingPairing(
        target_system=target_system, source_kind=kind, source_segment=source_segment
    )


_PAIRING_LAYOUTS: dict[BlockingSourceKind, Layout] = {
    BlockingSourceKind.DATA: BLOCKING_PLAIN,
    BlockingSourceKind.NAMES: BLOCKING_NAMES,
    BlockingSourceKind.PERTURBED: BLOCKING_PERTURBED,
}
"""The layout each kind of source a run is made over today is filed by."""


def _pairing_values(*, source_system: str, target_system: str) -> dict[str, str]:
    """The reference segments one pairing fixes, from the source it was given."""
    values = {"target": target_system}
    if not is_perturbed_source(source_system):
        return {**values, "source": source_system}
    source, profile, version, seed = perturbed_parts(source_system)
    return {
        **values,
        "source": source,
        "derivation": BlockingSourceKind.PERTURBED.value,
        "profile": profile,
        "version": version,
        "seed": str(seed),
    }


def pairing_layout(pairing: BlockingPairing) -> Layout:
    """The layout one pairing's runs are filed by."""
    try:
        return _PAIRING_LAYOUTS[pairing.source_kind]
    except KeyError:
        raise ValueError(
            f"No run is made over a {pairing.source_kind.value} source."
        ) from None


def pairing_reference(*, source_system: str, target_system: str) -> Reference:
    """The selection naming every run of one pairing, to be read with
    `pairing_layout`, since a plain pairing's segments are held by every other
    kind's layout too."""
    return reference(
        Kind.BLOCKING,
        Side.DATA,
        **_pairing_values(source_system=source_system, target_system=target_system),
    )


def run_key(keys: RunKeys) -> str:
    """The key segment a run is filed under: a digest of its four identity keys,
    `RUN_KEY_DIGEST_SIZE` bytes wide."""
    return compute_artifact_signature(
        keys.as_identity(), digest_size=RUN_KEY_DIGEST_SIZE
    )


def _location(roots: WorkspaceRoots, ref: Reference) -> BlockingRunLocation:
    fields = ref.fields
    if "derivation" in fields:
        kind = (
            BlockingSourceKind.PERTURBED_NAMES
            if "dataset" in fields
            else BlockingSourceKind.PERTURBED
        )
        source_segment = PERTURBED_SOURCE_SEPARATOR.join(
            fields[name] for name in ("source", "profile", "version", "seed")
        )
    elif "dataset" in fields:
        kind, source_segment = BlockingSourceKind.NAMES, fields["source"]
    elif fields["source"] == BlockingSourceKind.BENCHMARK.value:
        kind, source_segment = BlockingSourceKind.BENCHMARK, fields["target"]
    else:
        kind, source_segment = BlockingSourceKind.DATA, fields["source"]
    return BlockingRunLocation(
        directory=locate(roots, ref).resolve(),
        pairing=BlockingPairing(
            target_system=fields["target"],
            source_kind=kind,
            source_segment=source_segment,
        ),
        representation=fields["representation"],
    )


def blocking_runs_root(roots: WorkspaceRoots) -> Path:
    """The directory every run sits beneath: blocking's data directory under
    `artifacts/`."""
    return selection_directory(roots, reference(Kind.BLOCKING, Side.DATA))


def resolve_pairing_dir(
    roots: WorkspaceRoots, *, source_system: str, target_system: str
) -> Path:
    """The directory one pairing's runs and their comparison report sit under."""
    pairing = blocking_pairing(source_system=source_system, target_system=target_system)
    return selection_directory(
        roots,
        pairing_reference(source_system=source_system, target_system=target_system),
        layout=pairing_layout(pairing),
    )


def resolve_comparison_location(
    roots: WorkspaceRoots, *, source_system: str, target_system: str
) -> BlockingComparisonLocation:
    """Where one pairing's cross-method comparison report is written, beside
    the runs it compares."""
    pairing_dir = resolve_pairing_dir(
        roots, source_system=source_system, target_system=target_system
    )
    return BlockingComparisonLocation(
        directory=pairing_dir / COMPARISON_DIR_NAME,
        pairing=blocking_pairing(
            source_system=source_system, target_system=target_system
        ),
    )


def resolve_combined_comparison_location(
    roots: WorkspaceRoots,
) -> BlockingComparisonLocation:
    """Where the report combining every pairing's comparison is written: beside
    blocking's data directory, never inside it, since it describes no single
    pairing and must not be found by a scan of the runs."""
    return BlockingComparisonLocation(
        directory=blocking_artifact_root(roots) / COMPARISON_DIR_NAME, pairing=None
    )


def run_reference(
    *, source_system: str, target_system: str, representation: str, keys: RunKeys
) -> Reference:
    """The reference one run is filed under, from the run's identity."""
    return reference(
        Kind.BLOCKING,
        Side.DATA,
        **_pairing_values(source_system=source_system, target_system=target_system),
        representation=representation,
        key=run_key(keys),
    )


def resolve_run_location(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    target_system: str,
    representation: str,
    keys: RunKeys,
) -> BlockingRunLocation:
    """Where one run's output belongs, from the run's identity."""
    return _location(
        roots,
        run_reference(
            source_system=source_system,
            target_system=target_system,
            representation=representation,
            keys=keys,
        ),
    )


def resolve_run_location_for(
    config: BlockingRunConfig, *, keys: RunKeys
) -> BlockingRunLocation:
    """Where the run `config` describes belongs, once its keys are known, which
    `workflow.resolve_blocking_run_keys` gives before scoring."""
    return resolve_run_location(
        config.roots,
        source_system=config.source.system,
        target_system=config.target.system,
        representation=config.strategy.representation,
        keys=keys,
    )


def audit_reference_for(location: BlockingRunLocation) -> Reference:
    """The reference `location`'s run's audit is filed under.

    Keyed on the same `key` segment as the run itself -- `location.directory`'s
    own name, the run's key -- so a second audit of one run resolves to the
    location the first wrote, and never on a settings digest of its own, which
    an audit has none of beyond the run it reads. Only a plain
    (`BlockingSourceKind.DATA`) run is wired today; a run over any other
    source kind raises, rather than guess at a location this module has never
    checked stays apart from another kind's.
    """
    if location.pairing.source_kind is not BlockingSourceKind.DATA:
        raise ValueError(
            f"auditing a {location.pairing.source_kind.value!r} run is not "
            "supported yet; only a plain run (BlockingSourceKind.DATA) is."
        )
    return reference(
        Kind.BLOCKING_AUDIT,
        Side.DATA,
        source=location.pairing.source_segment,
        target=location.pairing.target_system,
        representation=location.representation,
        key=location.directory.name,
    )


def resolve_audit_location(
    roots: WorkspaceRoots, location: BlockingRunLocation
) -> BlockingAuditLocation:
    """Where `location`'s run's audit belongs, whether or not it exists yet."""
    ref = audit_reference_for(location)
    return BlockingAuditLocation(
        directory=locate(roots, ref).resolve(),
        pairing=location.pairing,
        representation=location.representation,
    )


def read_run_location(roots: WorkspaceRoots, directory: Path) -> BlockingRunLocation:
    """The location a run directory names, read back from where it sits.

    The pairing and representation are the reference's own segments, so a
    directory that is not a blocking run's location raises rather than
    reporting a pairing made of whatever the path happened to hold.
    """
    try:
        ref = reference_at(roots, directory)
    except InvalidReferenceError as error:
        raise ValueError(
            f"{Path(directory)} is not a blocking run directory: {error}"
        ) from error
    if (ref.kind, ref.side) != (Kind.BLOCKING, Side.DATA):
        raise ValueError(
            f"{Path(directory)} is {ref.uri}, not a blocking run directory."
        )
    return _location(roots, ref)


def is_finished_run(roots: WorkspaceRoots, location: BlockingRunLocation) -> bool:
    """Whether `location` holds a finished run: its production record, which
    `workspace.records.produce` writes last and only on success. The run's
    key is its reference's last segment, so a record there is a record of
    that identity and nothing further is compared."""
    return read_record(roots, reference_at(roots, location.directory)) is not None


def iter_blocking_runs(roots: WorkspaceRoots) -> Iterator[BlockingRunLocation]:
    """Every finished run in the tree, each with its pairing and representation.

    A run is a location holding its production record, so a directory left
    by anything other than a finished production is not yielded. The
    comparison report beside a pairing's representations holds no run
    beneath it and is never mistaken for one.
    """
    for ref in select_references(roots, reference(Kind.BLOCKING, Side.DATA)):
        if read_record(roots, ref) is not None:
            yield _location(roots, ref)


def select_blocking_runs(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    target_system: str,
    representation: str | None = None,
    run_key: str | None = None,
) -> list[BlockingRunLocation]:
    """The finished runs of one pairing, narrowed to a representation and a
    run key when given, in the tree's own order.

    The way a caller names a run it did not just write: a run's directory is
    its identity's digest, which nobody composes by hand, so a run is picked
    by the facts the tree is filed under and, where those leave several, by
    the key `run_blocking.py` printed.
    """
    pairing = blocking_pairing(source_system=source_system, target_system=target_system)
    values = _pairing_values(source_system=source_system, target_system=target_system)
    if representation is not None:
        values["representation"] = representation
    return [
        _location(roots, ref)
        for ref in select_references(
            roots,
            reference(Kind.BLOCKING, Side.DATA, **values),
            layout=pairing_layout(pairing),
        )
        if (run_key is None or ref.fields["key"] == run_key)
        and read_record(roots, ref) is not None
    ]


def resolve_measurement_report_path(
    roots: WorkspaceRoots, *, measurement: str, settings: Mapping[str, object]
) -> Path:
    """Where one measurement's report is written, keyed by the settings that
    decided it, so two measurements sampled differently never overwrite each
    other and the same measurement repeated lands where it did before."""
    return (
        resolve_candidate_dir(
            blocking_artifact_root(roots) / MEASUREMENTS_DIR_NAME,
            measurement,
            settings=settings,
            digest_size=RUN_KEY_DIGEST_SIZE,
        )
        / MEASUREMENT_REPORT_NAME
    )


def iter_blocking_comparisons(
    roots: WorkspaceRoots,
) -> Iterator[BlockingComparisonLocation]:
    """Every pairing's comparison report in the tree, each with its pairing.

    The combined report is not among them: it sits outside the tree by
    `resolve_combined_comparison_location`, so concatenating these can never
    read a previous combination of themselves.
    """
    root = blocking_runs_root(roots)
    if not root.exists():
        return
    for target_dir in _subdirectories(root):
        for kind_dir in _subdirectories(target_dir):
            try:
                source_kind = BlockingSourceKind(kind_dir.name)
            except ValueError:
                continue
            for source_dir in _subdirectories(kind_dir):
                directory = source_dir / COMPARISON_DIR_NAME
                location = BlockingComparisonLocation(
                    directory=directory,
                    pairing=BlockingPairing(
                        target_system=target_dir.name,
                        source_kind=source_kind,
                        source_segment=source_dir.name,
                    ),
                )
                if location.path(ComparisonArtefact.REPORT).is_file():
                    yield location


def _subdirectories(directory: Path) -> list[Path]:
    """Every subdirectory of `directory`, in name order, or none if it is
    missing."""
    if not directory.is_dir():
        return []
    return sorted(
        (child for child in directory.iterdir() if child.is_dir()),
        key=lambda child: child.name,
    )


__all__ = [
    "COMPARISON_DIR_NAME",
    "MEASUREMENTS_DIR_NAME",
    "MEASUREMENT_REPORT_NAME",
    "PERTURBED_SOURCE_SEPARATOR",
    "RUN_KEY_DIGEST_SIZE",
    "AuditArtefact",
    "BlockingAuditLocation",
    "BlockingComparisonLocation",
    "BlockingPairing",
    "BlockingRunLocation",
    "BlockingSourceKind",
    "ComparisonArtefact",
    "RunArtefact",
    "audit_reference_for",
    "blocking_pairing",
    "blocking_runs_root",
    "blocking_source_placement",
    "is_finished_run",
    "iter_blocking_comparisons",
    "iter_blocking_runs",
    "pairing_layout",
    "pairing_reference",
    "read_run_location",
    "resolve_audit_location",
    "resolve_combined_comparison_location",
    "resolve_comparison_location",
    "resolve_measurement_report_path",
    "resolve_pairing_dir",
    "resolve_run_location",
    "resolve_run_location_for",
    "run_key",
    "run_reference",
    "select_blocking_runs",
]
