"""Build a pairing's strategy comparison from the finished runs on disk,
without running anything.

Every finished run of the pairing is read back from its own location: its
production record gives the settings it was made under and how long it took,
and its files give what it scored. A strategy with no run on disk is absent
from the table; making a run is `run_blocking.py`'s job. The report is written
where `compare_blocking_strategies.py` writes its own, so
`aggregate_strategy_comparisons.py` combines it like any other. Beside it go
the pairing's per-pair outcomes: every run's verdict on every truth pair, and
the table of runs those rows join to.

With `--source` and `--target` omitted, every pairing holding a finished run
is reported. Runs on disk may have read different populations, so the table
keeps them all, each row carrying its `population_key` and `truth_key`, and a
country holding more than one is named on the way out.

Usage:
    .venv/Scripts/python.exe scripts/report_strategy_comparison.py \
        --source gleif --target ie
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_dry_run_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from blocking.audit import read_pair_audit
from blocking.comparison import (
    build_pair_outcome_candidates,
    build_pair_outcome_recall_curves,
    build_pair_outcome_runs,
    build_pair_outcome_source_candidates,
    build_pair_outcomes,
    build_strategy_comparison,
)
from blocking.contracts import BLOCKING_ARTIFACT_SCHEMAS
from blocking.reporting import (
    write_pair_outcomes_report,
    write_strategy_comparison_report,
)
from blocking.run_layout import (
    ComparisonArtefact,
    resolve_comparison_location,
    select_blocking_runs,
)
from blocking.stored_runs import load_stored_strategy_entries, stored_pairings
from workspace.roots import WorkspaceRoots

_REPORT_ARTEFACTS = (
    ComparisonArtefact.REPORT,
    ComparisonArtefact.REPORT_CSV,
    ComparisonArtefact.PAIR_OUTCOME_RUNS,
    ComparisonArtefact.PAIR_OUTCOME_RUNS_CSV,
    ComparisonArtefact.PAIR_OUTCOMES,
    ComparisonArtefact.PAIR_OUTCOME_CANDIDATES,
    ComparisonArtefact.PAIR_OUTCOME_SOURCE_CANDIDATES,
    ComparisonArtefact.PAIR_OUTCOME_RECALL_CURVES,
    ComparisonArtefact.PAIR_OUTCOME_AUDITS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pairing's strategy comparison from the finished blocking "
            "runs on disk, running nothing. Omit --source and --target together "
            "to report every pairing that holds a finished run."
        ),
    )
    add_workspace_roots_args(parser)
    parser.add_argument("--source", dest="source_system", default=None)
    parser.add_argument("--target", dest="target_system", default=None)
    parser.add_argument(
        "--representation",
        default=None,
        help="Report only runs made with this representation; omit for every "
        "representation the pairing holds.",
    )
    add_dry_run_arg(parser)
    return parser


def _mixed_countries(comparison: pl.DataFrame) -> list[str]:
    """The countries whose rows carry more than one population or truth key."""
    keyed = comparison.filter(pl.col("country").is_not_null())
    mixed = (
        keyed.group_by("country")
        .agg(
            pl.col("population_key").drop_nulls().n_unique().alias("populations"),
            pl.col("truth_key").drop_nulls().n_unique().alias("truths"),
        )
        .filter((pl.col("populations") > 1) | (pl.col("truths") > 1))
    )
    return sorted(mixed["country"].to_list())


def _stored_audits(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    target_system: str,
    representation: str | None,
    labels: set[str],
) -> pl.DataFrame:
    """The `pair_audit` rows of every reported run that has been audited;
    a run with no audit contributes none. Nothing is audited here."""
    frames = []
    for location in select_blocking_runs(
        roots,
        source_system=source_system,
        target_system=target_system,
        representation=representation,
    ):
        if f"{location.representation}/{location.directory.name}" not in labels:
            continue
        audit = read_pair_audit(roots, location)
        if audit is not None:
            frames.append(audit)
    schema = pl.Schema(BLOCKING_ARTIFACT_SCHEMAS["pair_audit"])
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames, how="vertical").cast(schema)


def _report_pairing(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    target_system: str,
    representation: str | None,
) -> int:
    """Write one pairing's report and return how many runs it holds."""
    entries, unreadable = load_stored_strategy_entries(
        roots,
        source_system=source_system,
        target_system=target_system,
        representation=representation,
    )
    for error in unreadable:
        print(f"[strategy-comparison] skipped: {error}", file=sys.stderr)
    if not entries:
        print(
            f"[strategy-comparison] {source_system} -> {target_system}: no "
            "finished run to report",
            file=sys.stderr,
        )
        return 0

    comparison = build_strategy_comparison(entries, allow_mixed_population=True)
    location = resolve_comparison_location(
        roots, source_system=source_system, target_system=target_system
    )
    report_path = write_strategy_comparison_report(location, comparison)
    csv_path = location.path(ComparisonArtefact.REPORT_CSV)
    csv_path.write_text(comparison.write_csv(), encoding="utf-8")
    print(
        f"[strategy-comparison] {source_system} -> {target_system}: "
        f"{len(entries)} run(s): {report_path}"
    )
    print(f"[strategy-comparison] human-readable csv: {csv_path}")

    runs = build_pair_outcome_runs(entries)
    outcomes = build_pair_outcomes(entries)
    audits = _stored_audits(
        roots,
        source_system=source_system,
        target_system=target_system,
        representation=representation,
        labels={entry.label for entry in entries},
    )
    runs_path, outcomes_path, *_ = write_pair_outcomes_report(
        location,
        runs=runs,
        outcomes=outcomes,
        candidates=build_pair_outcome_candidates(entries),
        source_candidates=build_pair_outcome_source_candidates(entries),
        recall_curves=build_pair_outcome_recall_curves(entries),
        audits=audits,
    )
    runs_csv_path = location.path(ComparisonArtefact.PAIR_OUTCOME_RUNS_CSV)
    runs_csv_path.write_text(runs.write_csv(), encoding="utf-8")
    print(
        f"[strategy-comparison] {source_system} -> {target_system}: "
        f"{outcomes.height} pair outcome(s): {outcomes_path}"
    )
    print(f"[strategy-comparison] their runs: {runs_path} and {runs_csv_path}")
    audited = audits.get_column("label").n_unique()
    print(
        f"[strategy-comparison] {source_system} -> {target_system}: "
        f"{audited} of {len(entries)} run(s) audited"
    )
    for country in _mixed_countries(comparison):
        print(
            f"[strategy-comparison] {source_system} -> {target_system}: country="
            f"{country} holds runs over more than one population; compare rows "
            "sharing a population_key and truth_key"
        )
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.source_system is None) != (args.target_system is None):
        parser.error("--source and --target are given together or omitted together")
    roots = resolve_workspace_roots_from_args(args)

    pairings = (
        [(args.source_system, args.target_system)]
        if args.source_system is not None
        else stored_pairings(roots)
    )

    if args.dry_run:
        report_dry_run(
            "report_strategy_comparison",
            pairings=pairings,
            representation=args.representation,
        )
        report_output_plan(
            "[dry-run]  ",
            [
                PlannedOutput(
                    resolve_comparison_location(
                        roots, source_system=source, target_system=target
                    ).path(artefact),
                    OUTPUT_CLEAR,
                )
                for source, target in pairings
                for artefact in _REPORT_ARTEFACTS
            ],
        )
        return 0

    reported = sum(
        _report_pairing(
            roots,
            source_system=source,
            target_system=target,
            representation=args.representation,
        )
        for source, target in pairings
    )
    if reported == 0:
        print("[strategy-comparison] error: no finished run on disk", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
