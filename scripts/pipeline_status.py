"""Report how fresh each pipeline stage's output is, and the command that would refresh each stale one.

Per system, the latest artefact timestamp for acquire, shard, canonical, cleanse and tokenize, with `!` on a stage older than its upstream's latest output; then match-analysis freshness per scenario and blocking-run freshness per run. Match is left out of the per-system table, since it is a per-scenario concern. Each stale cell is followed by the command that refreshes it, derived from `process_companies.py`'s own stage choices so a renamed stage cannot leave a dead command behind.

The report is advisory and runs nothing: `process_companies.py`'s own freshness checks decide what runs. A timestamp cannot say whether a layer was regenerated into the right shape; `check_name_layer_identity.py` answers that for the name-variant sidecars.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

# `design_artifact_status` is a repo gate and lives in `tooling/`, put on
# `sys.path` by `_bootstrap` above rather than beside this script.
import design_artifact_status
import polars as pl
import process_companies
from cli_common import add_root_arg, run_reporting_argument_errors

from acquisition.constants_pipeline import PIPELINE_STAGE_NAMES
from acquisition.pipeline_selection import normalize_systems
from blocking.run_layout import BlockingSourceKind, RunArtefact, iter_blocking_runs
from workspace.artifact_layout import analysis_report_runs_root
from workspace.data_layout import data_root, system_layer_dir
from workspace.roots import WorkspaceRoots, default_workspace_roots

PROCESS_CHOICES_WITH_ACQUIRE = process_companies.PROCESS_CHOICES_WITH_ACQUIRE

# Ordered (stage, predecessor) pairs mirroring the process_companies.py pipeline.
# "match" is deliberately excluded from the per-system table below: it's really a
# per-(source, target) scenario concern (and also depends on the target system's
# cleanse output, which this single-system view can't see - process_companies.py's
# own staleness check gives up on that for the same reason and always reruns match).
# The match-analysis section reports scenario freshness directly, and a scenario's
# absence there already tells you it hasn't been run. STAGE_DIRS still maps "match"
# so build_match_analysis_status() can resolve matched/ mtimes per system.
STAGE_DIRS: dict[str, tuple[str, ...]] = {
    "acquire": ("acquire",),
    "shard": ("source",),
    "canonical": ("canonical",),
    "cleanse": ("cleansed",),
    "match": ("matched",),
    "tokenize": ("tokenized",),
}
# Derived from the pipeline's own stage-name constant, minus "match" (see
# above), rather than a separately hand-kept tuple, so a renamed or added
# stage shows up here without a second edit.
STAGE_ORDER = tuple(stage for stage in PIPELINE_STAGE_NAMES if stage != "match")
STAGE_PREDECESSOR: dict[str, str | None] = {
    "acquire": None,
    "shard": "acquire",
    "canonical": "shard",
    "cleanse": "canonical",
    "tokenize": "cleanse",
}


def build_stage_refresh_command(*, system: str, stage: str) -> str:
    """Return the `process_companies.py` invocation that refreshes `stage` for `system`.

    Validated against `PROCESS_CHOICES_WITH_ACQUIRE`, the actual argparse
    `--processes` choices `process_companies.py` accepts, so a stage renamed
    or removed there raises here instead of silently printing a command that
    no longer runs.
    """

    if stage not in PROCESS_CHOICES_WITH_ACQUIRE:
        raise ValueError(
            f"Unknown pipeline stage {stage!r}; process_companies.py accepts "
            f"one of {PROCESS_CHOICES_WITH_ACQUIRE}"
        )
    return (
        "uv run python scripts/process_companies.py "
        f"--systems {system} --processes {stage}"
    )


def find_stale_stage_cells(
    roots: WorkspaceRoots, systems: list[str]
) -> list[tuple[str, str]]:
    """Return `(system, stage)` pairs the per-system stage table marks stale."""

    stale: list[tuple[str, str]] = []
    for system in systems:
        statuses = {
            stage: _resolve_stage_status(roots, system, stage) for stage in STAGE_ORDER
        }
        for stage in STAGE_ORDER:
            status = statuses[stage]
            predecessor = STAGE_PREDECESSOR[stage]
            predecessor_mtime = statuses[predecessor].mtime if predecessor else None
            is_stale = (
                status.mtime is not None
                and predecessor_mtime is not None
                and status.mtime < predecessor_mtime
            )
            if is_stale:
                stale.append((system, stage))
    return stale


REPORT_TABLE_CONFIG: dict[str, Any] = {
    "tbl_cols": -1,
    "tbl_width_chars": 10000,
    "tbl_hide_column_data_types": True,
    "tbl_hide_dtype_separator": True,
    "tbl_hide_dataframe_shape": True,
}


@dataclass(frozen=True)
class StageStatus:
    mtime: datetime | None
    file_count: int


def _latest_mtime_and_count(directory: Path) -> StageStatus:
    if not directory.exists():
        return StageStatus(mtime=None, file_count=0)
    files = [path for path in directory.rglob("*") if path.is_file()]
    if not files:
        return StageStatus(mtime=None, file_count=0)
    latest = max(path.stat().st_mtime for path in files)
    return StageStatus(
        mtime=datetime.fromtimestamp(latest).astimezone(), file_count=len(files)
    )


def _resolve_stage_status(
    roots: WorkspaceRoots, system: str, stage: str
) -> StageStatus:
    for candidate_name in STAGE_DIRS[stage]:
        directory = system_layer_dir(roots, system, layer=candidate_name)
        if directory.exists():
            return _latest_mtime_and_count(directory)
    return StageStatus(mtime=None, file_count=0)


def _discover_systems(roots: WorkspaceRoots) -> list[str]:
    root_data_dir = data_root(roots)
    if not root_data_dir.exists():
        return []
    return sorted(path.name for path in root_data_dir.iterdir() if path.is_dir())


def _format_cell(status: StageStatus, *, is_stale: bool) -> str:
    if status.mtime is None:
        return "-"
    marker = " !" if is_stale else ""
    return f"{status.mtime:%Y-%m-%d %H:%M}{marker}"


def build_pipeline_status_table(
    roots: WorkspaceRoots, systems: list[str]
) -> pl.DataFrame:
    rows: list[dict[str, str]] = []
    for system in systems:
        statuses = {
            stage: _resolve_stage_status(roots, system, stage) for stage in STAGE_ORDER
        }
        row: dict[str, str] = {"system": system}
        for stage in STAGE_ORDER:
            status = statuses[stage]
            predecessor = STAGE_PREDECESSOR[stage]
            predecessor_mtime = statuses[predecessor].mtime if predecessor else None
            is_stale = (
                status.mtime is not None
                and predecessor_mtime is not None
                and status.mtime < predecessor_mtime
            )
            row[stage] = _format_cell(status, is_stale=is_stale)
        rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"system": pl.Utf8})


@dataclass(frozen=True)
class ScenarioStatus:
    scenario: str
    latest_run_date: str | None
    run_mtime: datetime | None
    source_input_mtime: datetime | None
    is_stale: bool


def _parse_scenario_name(scenario: str) -> tuple[str, str, str] | None:
    if "_to_" not in scenario:
        return None
    source_system, remainder = scenario.split("_to_", 1)
    if "_" not in remainder:
        return None
    target_system, country = remainder.rsplit("_", 1)
    if not source_system or not target_system or not country:
        return None
    return source_system, target_system, country


def build_match_analysis_status(roots: WorkspaceRoots) -> list[ScenarioStatus]:
    runs_root = analysis_report_runs_root(roots, "match_analysis")
    if not runs_root.exists():
        return []

    latest_run_by_scenario: dict[str, tuple[str, Path]] = {}
    for run_date_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for scenario_dir in sorted(p for p in run_date_dir.iterdir() if p.is_dir()):
            latest_run_by_scenario[scenario_dir.name] = (
                run_date_dir.name,
                scenario_dir,
            )

    results: list[ScenarioStatus] = []
    for scenario, (run_date, scenario_dir) in sorted(latest_run_by_scenario.items()):
        run_status = _latest_mtime_and_count(scenario_dir)
        parsed = _parse_scenario_name(scenario)
        source_input_mtime: datetime | None = None
        if parsed is not None:
            source_system, target_system, _country = parsed
            candidate_mtimes = [
                _resolve_stage_status(roots, source_system, "match").mtime,
                _resolve_stage_status(roots, source_system, "cleanse").mtime,
                _resolve_stage_status(roots, target_system, "cleanse").mtime,
            ]
            present = [m for m in candidate_mtimes if m is not None]
            source_input_mtime = max(present) if present else None

        is_stale = (
            run_status.mtime is not None
            and source_input_mtime is not None
            and run_status.mtime < source_input_mtime
        )
        results.append(
            ScenarioStatus(
                scenario=scenario,
                latest_run_date=run_date,
                run_mtime=run_status.mtime,
                source_input_mtime=source_input_mtime,
                is_stale=is_stale,
            )
        )
    return results


def build_match_analysis_refresh_command(scenario: str) -> str | None:
    """Return the `analyze_matches.py` invocation that refreshes `scenario`.

    Built from the `(source_system, target_system, country)` the scenario
    name already encodes -- the same flags `analyze_matches.py`'s parser
    takes -- rather than a separately maintained template. Returns `None`
    when the scenario name doesn't parse, matching `source_input_mtime`'s
    own "can't tell" case.
    """

    parsed = _parse_scenario_name(scenario)
    if parsed is None:
        return None
    source_system, target_system, country = parsed
    return (
        "uv run python scripts/analyze_matches.py "
        f"--source {source_system} --target {target_system} "
        f"--country {country}"
    )


@dataclass(frozen=True)
class BlockingRunStatus:
    pair: str
    source_system: str
    target_system: str
    source_kind: str
    representation: str
    run_id: str
    run_mtime: datetime | None
    input_mtime: datetime | None
    is_stale: bool


def _resolve_prepared_input_mtime(
    roots: WorkspaceRoots, system: str
) -> datetime | None:
    """Latest mtime of the layer a blocking run would actually read for `system`.

    Mirrors `blocking.loader.load_dataset_descriptor`'s own preference order
    (`matched` before `cleansed`) since a blocking run records neither which
    layer it read nor when -- there is no provenance to prefer here, only the
    same fallback the loader itself applies.
    """

    matched_status = _resolve_stage_status(roots, system, "match")
    if matched_status.mtime is not None:
        return matched_status.mtime
    return _resolve_stage_status(roots, system, "cleanse").mtime


def _summary_source_system(summary_path: Path) -> str | None:
    """The source system a run's own `summary.json` records, or None when the
    file does not say. A perturbed source is filed under a hyphenated
    segment the selector cannot be recovered from, so the refresh command
    reads the run's own record of it."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    source_system = summary.get("source_system") if isinstance(summary, dict) else None
    return str(source_system) if source_system else None


def build_blocking_run_status(roots: WorkspaceRoots) -> list[BlockingRunStatus]:
    """Report freshness for every run under blocking's data directory.

    The runs come from `blocking.run_layout.iter_blocking_runs`, which owns
    where a run sits and what it holds, so this report finds what the writers
    wrote without knowing the tree's shape itself.

    A run's own key covers the layers it read, so a run made over re-cleansed
    inputs resolves to a different directory rather than overwriting the
    earlier one. The mtime comparison below therefore reports a run that has
    been *superseded*, not one that was silently overwritten. Only a source of
    the `data` kind reads a system's own prepared layer; a perturbed or names
    source is derived and has no layer here to date it against, so its
    verdict rests on the target side alone.
    """

    input_mtimes: dict[str, datetime | None] = {}

    def _input_mtime(system: str) -> datetime | None:
        if system not in input_mtimes:
            input_mtimes[system] = _resolve_prepared_input_mtime(roots, system)
        return input_mtimes[system]

    results: list[BlockingRunStatus] = []
    for location in iter_blocking_runs(roots):
        pairing = location.pairing
        target_input_mtime = _input_mtime(pairing.target_system)
        # Only a source of the `data` kind reads a system's own prepared layer;
        # a perturbed or names source is derived and has none to date against.
        source_input_mtime = (
            _input_mtime(pairing.source_segment)
            if pairing.source_kind == BlockingSourceKind.DATA
            else None
        )
        candidate_mtimes = [
            m for m in (source_input_mtime, target_input_mtime) if m is not None
        ]
        input_mtime = max(candidate_mtimes) if candidate_mtimes else None

        summary_path = location.path(RunArtefact.SUMMARY)
        run_status = _latest_mtime_and_count(location.directory)
        is_stale = (
            run_status.mtime is not None
            and input_mtime is not None
            and run_status.mtime < input_mtime
        )
        results.append(
            BlockingRunStatus(
                pair=pairing.label,
                source_system=_summary_source_system(summary_path)
                or pairing.source_segment,
                target_system=pairing.target_system,
                source_kind=pairing.source_kind.value,
                representation=location.representation,
                run_id=location.directory.name,
                run_mtime=run_status.mtime,
                input_mtime=input_mtime,
                is_stale=is_stale,
            )
        )
    return results


def build_blocking_refresh_command(
    *, source_system: str, target_system: str, representation: str
) -> str:
    """Return the `run_blocking.py` invocation that refreshes this run.

    Uses the same `--source`/`--target`/`--representation`
    flags `run_blocking.py`'s own parser takes; the run's config-hash
    directory name is not reproducible from this alone (it also depends on
    strategy options this report has no record of), so the command
    regenerates a run for the pairing and representation, not byte-identical
    output at the same path.
    """

    return (
        "uv run python scripts/run_blocking.py "
        f"--source {source_system} --target {target_system} "
        f"--representation {representation}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report the latest artifact timestamp for each pipeline stage "
            "(acquire/shard/canonical/cleanse/tokenize) per system, "
            "match-analysis run freshness per (source, target, country) scenario, "
            "and blocking run freshness per (source, target, representation) run "
            "under artifacts/blocking/, so it's clear at a glance what needs re-running. "
            "Every stale cell is printed with the command that would refresh it -- "
            "printed only, never executed."
        )
    )
    add_root_arg(parser, help_text="Project root containing data/ and artifacts/.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=(
            "Systems to report on. Space or comma separated; 'all' expands to all "
            "live systems. Defaults to every system directory found under data/."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    systems = (
        normalize_systems(args.systems) if args.systems else _discover_systems(roots)
    )
    if not systems:
        print("No systems found under data/ and none were specified with --systems.")
        return 0

    print(f"Pipeline stage status ({root})")
    print(
        "Each cell: <latest mtime>; '!' means older than its upstream stage's latest output."
    )
    print()
    with pl.Config(**REPORT_TABLE_CONFIG):
        print(build_pipeline_status_table(roots, systems))

    stale_stage_cells = find_stale_stage_cells(roots, systems)
    if stale_stage_cells:
        print("Refresh commands for stale cells above:")
        for stale_system, stale_stage in stale_stage_cells:
            stage_command = build_stage_refresh_command(
                system=stale_system, stage=stale_stage
            )
            print(f"  [{stale_system}] {stale_stage}: {stage_command}")

    scenario_statuses = build_match_analysis_status(roots)
    if scenario_statuses:
        print()
        print("Match-analysis run freshness (artifacts/analysis/match_analysis/runs/):")
        scenario_rows = [
            {
                "scenario": status.scenario,
                "latest_run_date": status.latest_run_date or "-",
                "run_mtime": status.run_mtime.strftime("%Y-%m-%d %H:%M")
                if status.run_mtime
                else "-",
                "stale": "!" if status.is_stale else "",
            }
            for status in scenario_statuses
        ]
        with pl.Config(**REPORT_TABLE_CONFIG):
            print(pl.DataFrame(scenario_rows))

        stale_scenarios = [status for status in scenario_statuses if status.is_stale]
        if stale_scenarios:
            print("Refresh commands for stale scenarios above:")
            for scenario_status in stale_scenarios:
                scenario_command = build_match_analysis_refresh_command(
                    scenario_status.scenario
                )
                if scenario_command is not None:
                    print(f"  [{scenario_status.scenario}]: {scenario_command}")

    blocking_statuses = build_blocking_run_status(roots)
    if blocking_statuses:
        print()
        print(
            "Blocking run freshness (artifacts/blocking/data/, <target>/<kind>/<source>):"
        )
        blocking_rows = [
            {
                "pair": status.pair,
                "representation": status.representation,
                "run": status.run_id,
                "run_mtime": status.run_mtime.strftime("%Y-%m-%d %H:%M")
                if status.run_mtime
                else "-",
                "stale": "!" if status.is_stale else "",
            }
            for status in blocking_statuses
        ]
        with pl.Config(**REPORT_TABLE_CONFIG):
            print(pl.DataFrame(blocking_rows))

        stale_blocking_runs = [
            status for status in blocking_statuses if status.is_stale
        ]
        if stale_blocking_runs:
            print("Refresh commands for stale blocking runs above:")
            for blocking_status in stale_blocking_runs:
                blocking_command = build_blocking_refresh_command(
                    source_system=blocking_status.source_system,
                    target_system=blocking_status.target_system,
                    representation=blocking_status.representation,
                )
                print(
                    f"  [{blocking_status.pair}/{blocking_status.representation}/"
                    f"{blocking_status.run_id}]: {blocking_command}"
                )

    print()
    print("Design artifact freshness (pyscn / graphify):")
    for status in design_artifact_status.build_design_artifact_statuses(root):
        print(f"  {design_artifact_status.format_status_line(status)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
