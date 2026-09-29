"""Materialize a named perturbation profile's output into a loader-readable dataset.

Loads a named profile (`company_perturbation.profile_schema`) and runs
`perturbation_materializer.py`'s `materialize_perturbations()` against a real source system's
cleansed data, producing one perturbed row per source record. Prints the resulting
flags a run reads it back with, `--source <system> --perturbed <profile>`, the version and
seed being typed only where several exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    add_declared_arguments,
    add_dry_run_arg,
    add_perturb_args,
    add_root_arg,
    declared_settings,
    profile_from_args,
    report_resolved_settings,
    resolve_declared_settings,
    resolved_setting_values,
    run_reporting_argument_errors,
)

from validation._cli_helper import SETTINGS as VALIDATION_SETTINGS
from validation.perturbation_materializer import (
    MaterializationConfig,
    materialize_perturbations,
)
from validation.perturbation_profiles import (
    SeedNotResolvedError,
    load_perturbation_profile,
    resolve_seed,
)
from workspace.data_layout import perturbed_dataset_dir
from workspace.reference import (
    AmbiguousReferenceError,
    InvalidReferenceError,
    ReferenceNotFoundError,
)
from workspace.roots import default_workspace_roots

_DECLARATIONS = declared_settings(VALIDATION_SETTINGS)

# Where this script takes each declared setting. The source system, the
# profile reference and the seed identify which run this is, not how it
# behaves, so they stay ordinary arguments below.
_SURFACE: dict[str, dict[str, object]] = {
    "force_rebuild": {"flag": "--force-rebuild"},
    "prepared_rows_per_file": {"flag": "--rows-per-file"},
}


def _parse_csv_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    return items or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a perturbation profile's output into a loader-readable dataset."
        ),
    )
    add_perturb_args(parser)
    parser.add_argument(
        "--source",
        dest="source_system",
        required=True,
        help="Real source system code to perturb (e.g. 'gb').",
    )
    add_root_arg(
        parser,
        help_text="Repository root containing data/ (default: current directory).",
    )
    parser.add_argument(
        "--countries",
        default=None,
        help="Comma-separated country codes to scope materialization to (default: all).",
    )
    parser.add_argument(
        "--name-col",
        default="name",
        help="Column in the source system's cleansed data to perturb (default: 'name').",
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run seed. The same source, profile and seed always give the same dataset; "
            "a different seed gives an independent one of the same size. Defaults to "
            "the profile's own 'default_seed', so a profile that names one is "
            "reproducible without this flag; pass it to resample deliberately. "
            "Required for a profile that declares none."
        ),
    )
    add_dry_run_arg(parser)
    return parser


def _print_progress(event: dict[str, object]) -> None:
    """Presentation lives here, not in the materializer -- the same split
    `run_blocking.py` uses, so the library emits facts and the script decides how
    they read."""
    phase = event.get("phase")
    at = f"[{event.get('elapsed_s', 0):>6.1f}s]"

    if phase == "start":
        print(
            f"{at} profile={event['profile_id']} source={event['source_system']} "
            f"seed={event['seed']}"
        )
    elif phase == "source_opened":
        print(f"{at} source has {event['rows']:,} rows")
    elif phase == "written":
        rows, total = int(event["rows"]), int(event["of"])  # type: ignore[call-overload]
        print(f"{at} written {rows:,}/{total:,} ({rows / total:.0%})")
    elif phase == "generated":
        rows, changed = int(event["rows"]), int(event["changed"])  # type: ignore[call-overload]
        share = f"{changed / rows:.1%}" if rows else "-"
        print(f"{at} {rows:,} rows, {changed:,} changed ({share})")
    elif phase == "validated":
        print(f"{at} schema and quality checks passed ({event['rows']:,} rows)")
    elif phase == "complete":
        print(f"{at} manifest written to {event['manifest']}")


def run_materialization(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    authored = load_perturbation_profile(
        roots,
        profile_from_args(args, roots),
    )
    profile = authored.profile

    seed = resolve_seed(profile, requested=args.seed)

    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    values = resolved_setting_values(resolved)

    config = MaterializationConfig(
        roots=roots,
        source_system=str(args.source_system).strip().lower(),
        profile=profile,
        profile_version=authored.version,
        seed=seed,
        countries=_parse_csv_list(args.countries),
        name_col=str(args.name_col),
        rows_per_file=int(values["prepared_rows_per_file"]),
        force_rebuild=bool(values["force_rebuild"]),
    )

    if args.dry_run:
        report_resolved_settings(
            "materialize_perturbations",
            resolved,
            profile=authored.reference.uri,
            scenarios=[scenario.scenario_id for scenario in profile.scenarios],
            seed=seed,
            seed_from="profile" if args.seed is None else "argument",
            source_system=config.source_system,
            countries=config.countries,
            name_col=config.name_col,
            output_dir=str(
                perturbed_dataset_dir(
                    roots=roots,
                    source_system=config.source_system,
                    profile_id=profile.profile_id,
                    version=authored.version,
                    seed=seed,
                )
            ),
        )
        return 0

    result = materialize_perturbations(config, progress_callback=_print_progress)

    print(
        f"[materialize] profile_id={result.profile_id} rows_emitted={result.rows_emitted}"
    )
    print(f"[materialize] output_dir={result.output_dir}")
    print(f"[materialize] manifest={result.manifest_path}")
    print(
        "[materialize] read this output with "
        f"--source {config.source_system} --perturbed {result.profile_id}"
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_materialization(args)
    except (
        SeedNotResolvedError,
        InvalidReferenceError,
        ReferenceNotFoundError,
        AmbiguousReferenceError,
    ) as unusable:
        # An unusable combination of arguments, not a failed run: reported the way
        # argparse reports a missing one, rather than as a traceback.
        parser.error(str(unusable))


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
