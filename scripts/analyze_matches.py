"""Analyse Match-stage outcomes for one or more source, target and country scenarios.

KPI tables, no-match triage and comparison charts, from the `match_uri` artefacts Match wrote.
`--country` defaults to `--target`; `--scenarios-file` takes a JSON list of
`{"source_system", "target_system", "country"}` objects to run several at once. Output is written
under `artifacts/analysis/match_analysis/runs/<date>/<scenario>/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_declared_arguments,
    add_dry_run_arg,
    add_workspace_roots_args,
    declared_settings,
    report_dry_run,
    report_output_plan,
    resolve_declared_settings,
    resolve_workspace_roots_from_args,
    resolved_setting_values,
    run_reporting_argument_errors,
)

from analysis._cli_helper import SETTINGS as ANALYSIS_SETTINGS
from analysis.match_metrics import (
    MatchAnalysisPaths,
    resolve_match_analysis_run_root,
    run_match_analysis,
    run_match_analysis_batch,
)
from analysis.report_layout import metrics_dir, viz_dir
from workspace.roots import WorkspaceRoots

_DECLARATIONS = declared_settings(ANALYSIS_SETTINGS)
_SURFACE: dict[str, dict[str, object]] = {
    "max_rows": {"flag": "--max-rows"},
}


def _load_scenarios_file(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Scenarios file must contain a JSON list: {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse match outcomes for one or more source/target/country scenarios: "
            "KPI tables, no-match triage, and comparison charts."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--date",
        dest="run_date",
        default=datetime.now(UTC).date().isoformat(),
        help="Run date in YYYY-MM-DD format. Scenarios sharing a run date land under the same runs/<date>/ folder.",
    )
    parser.add_argument(
        "--source",
        dest="source_system",
        default=None,
        help="Source system code, e.g. gleif, wikidata.",
    )
    parser.add_argument(
        "--target",
        dest="target_system",
        default=None,
        help="Target system code, e.g. ie, fr, gb.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Country / hive-partition filter. Defaults to --target.",
    )
    parser.add_argument(
        "--target-display",
        default="",
        help="Optional display label for the target, e.g. 'OpenCorporates (IE)'.",
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--scenarios-file",
        default=None,
        help=(
            "Path to a JSON file containing a list of scenario objects, for example: "
            '[{"source_system": "gleif", "target_system": "gb", "country": "gb"}, '
            '{"source_system": "gleif", "target_system": "fr"}]. '
            "Runs every scenario in one invocation and overrides "
            "--source/--target/--country/--target-display/--max-rows."
        ),
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve every scenario's run directory and report the settings "
            "and output paths that would apply, without reading or writing "
            "anything."
        ),
    )
    return parser


def _report(paths: MatchAnalysisPaths) -> None:
    print(f"[{paths.scenario}] summary: {paths.summary_path}")
    print(f"[{paths.scenario}] no-match reasons: {paths.reasons_path}")
    print(f"[{paths.scenario}] match outcome chart: {paths.match_outcome_chart_path}")
    print(f"[{paths.scenario}] equality rates chart: {paths.equality_rates_chart_path}")
    print(
        f"[{paths.scenario}] equality overlap chart: {paths.equality_overlap_venn_path}"
    )
    print(
        f"[{paths.scenario}] overlap states chart: {paths.matched_overlap_states_chart_path}"
    )
    print(
        f"[{paths.scenario}] no-match reasons chart: {paths.no_match_reasons_chart_path}"
    )
    print(
        f"[{paths.scenario}] recoverability chart: {paths.no_match_recoverability_chart_path}"
    )


def _scenario_kwargs(
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    if args.scenarios_file:
        return _load_scenarios_file(Path(args.scenarios_file))
    return [
        {
            "source_system": args.source_system,
            "target_system": args.target_system,
            "country": args.country,
            "target_display": args.target_display,
        }
    ]


def _report_dry_run(
    roots: WorkspaceRoots,
    *,
    run_date: str,
    max_rows: int | None,
    scenarios: list[dict[str, object]],
) -> None:
    for scenario in scenarios:
        source_system = str(scenario["source_system"])
        target_system = str(scenario["target_system"])
        country = str(scenario.get("country") or target_system)
        run_root = resolve_match_analysis_run_root(
            roots,
            run_date=run_date,
            source_system=source_system,
            target_system=target_system,
            country=country,
        )
        report_dry_run(f"analyze_matches[{run_root.name}]", max_rows=max_rows)
        report_output_plan(
            f"[dry-run] analyze_matches[{run_root.name}]:",
            [
                PlannedOutput(metrics_dir(run_root), OUTPUT_CLEAR),
                PlannedOutput(viz_dir(run_root), OUTPUT_CLEAR),
            ],
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    roots = resolve_workspace_roots_from_args(args)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    max_rows = resolved_setting_values(resolved)["max_rows"]

    if not args.scenarios_file and not (args.source_system and args.target_system):
        parser.error(
            "--source and --target are required unless --scenarios-file is given."
        )

    scenarios = _scenario_kwargs(args)

    if args.dry_run:
        _report_dry_run(
            roots, run_date=args.run_date, max_rows=max_rows, scenarios=scenarios
        )
        return 0

    if args.scenarios_file:
        results = run_match_analysis_batch(
            roots,
            scenarios,
            run_date=args.run_date,
            max_rows=max_rows,
            progress=lambda message: print(f"[info] {message}"),
        )
        for paths in results:
            _report(paths)
        return 0

    paths = run_match_analysis(
        roots,
        source_system=args.source_system,
        target_system=args.target_system,
        country=args.country,
        target_display=args.target_display,
        max_rows=max_rows,
        run_date=args.run_date,
        progress=lambda message: print(f"[info] {message}"),
    )
    _report(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
