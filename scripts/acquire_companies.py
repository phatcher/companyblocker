from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    add_root_arg,
    add_run_date_arg,
    add_systems_arg,
    run_reporting_argument_errors,
    supported_codes_epilog,
)

from acquisition.pipeline import run_acquisition
from workspace.roots import default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire bulk company data for one or more systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=supported_codes_epilog(),
    )
    add_systems_arg(parser)
    add_run_date_arg(parser, subject="Run date folder", default_behavior="today")
    add_root_arg(parser, help_text="Project root containing the data/ directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading files or writing the manifest.",
    )
    parser.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="Fail instead of skipping systems that are blocked or still need research.",
    )
    parser.add_argument(
        "--allow-research",
        action="store_true",
        help="Allow execution for systems marked research_required. Use only for discovery or validation runs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    results = run_acquisition(
        systems=args.systems,
        roots=roots,
        run_date=args.run_date,
        dry_run=args.dry_run,
        skip_unsupported=not args.fail_on_unsupported,
        allow_research=args.allow_research,
        progress=lambda message: print(f"[info] {message}"),
    )

    has_problem = False
    for result in results:
        print(f"[{result.status}] {result.message}")
        if result.status not in {"downloaded", "dry_run", "freshness_skipped"}:
            has_problem = True

    if args.fail_on_unsupported and has_problem:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
