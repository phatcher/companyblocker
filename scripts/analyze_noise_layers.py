from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    OUTPUT_EXTEND,
    PlannedOutput,
    add_declared_arguments,
    add_dry_run_arg,
    add_workspace_roots_args,
    declared_settings,
    report_output_plan,
    report_resolved_settings,
    resolve_declared_settings,
    resolve_workspace_roots_from_args,
    resolved_setting_values,
    run_reporting_argument_errors,
)

from analysis._cli_helper import SETTINGS as ANALYSIS_SETTINGS
from analysis.noise_layers import run_noise_layer_analysis
from analysis.report_layout import metrics_dir, viz_dir
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

_DECLARATIONS = declared_settings(ANALYSIS_SETTINGS)
_SURFACE: dict[str, dict[str, object]] = {
    "cleanse_tier": {"flag": "--cleanse-tier"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-corpus token table and noise-layer pictures. Projects "
            "an existing analysis.token_zipf run's per-tier token stats and each "
            "named system's tokenizer token_tfidf_stats.parquet into one "
            "token/system/tier/language/document-frequency/IDF table, computes "
            "what each noise layer (cleanse-stage tier diff, packaged default "
            "noise-word list, per-corpus profiles, and any --extra-noise-words "
            "file) removes relative to the untouched raw-tier baseline, and "
            "renders per-corpus bar charts plus a Venn/UpSet overlap diagram per "
            "layer, alongside a membership parquet of every removed token."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--systems",
        required=True,
        help="Comma-separated system codes to compare (not auto-discovered).",
    )
    parser.add_argument(
        "--date",
        dest="run_date",
        default=datetime.now(UTC).date().isoformat(),
        help="Run date in YYYY-MM-DD format for this module's own run tree.",
    )
    parser.add_argument(
        "--zipf-run-date",
        default=None,
        help=(
            "Run date of the existing analysis.token_zipf run to read per-tier "
            "token stats from. Defaults to --date."
        ),
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--extra-noise-words",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "An additional noise-word file to compare as one more layer "
            "(e.g. a pooled noise-word candidate file), named LABEL=PATH, PATH "
            "absolute or relative to the checkout. Repeatable."
        ),
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run directory and report the settings and output "
            "paths that would apply, without reading any run's data or "
            "rendering anything."
        ),
    )
    return parser


def _parse_extra_noise_words(
    roots: WorkspaceRoots, values: list[str]
) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--extra-noise-words expects LABEL=PATH, got: {value!r}")
        label, _, path_str = value.partition("=")
        path = Path(path_str.strip())
        if not path.is_absolute():
            path = (roots.checkout / path).resolve()
        parsed.append((label.strip(), path))
    return parsed


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    if not systems:
        parser.error("--systems must name at least one system.")
    roots = resolve_workspace_roots_from_args(args)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    cleanse_tier = resolved_setting_values(resolved)["cleanse_tier"]
    extra_noise_words = _parse_extra_noise_words(roots, args.extra_noise_words)

    if args.dry_run:
        run_root = analysis_report_run_dir(roots, "noise_layers", args.run_date)
        report_resolved_settings(
            "analyze_noise_layers",
            resolved,
            systems=systems,
            zipf_run_date=args.zipf_run_date or args.run_date,
            extra_noise_words=extra_noise_words,
        )
        report_output_plan(
            "[dry-run] analyze_noise_layers:",
            [
                PlannedOutput(
                    metrics_dir(run_root),
                    OUTPUT_EXTEND,
                    note="each named system/layer file is overwritten; a prior "
                    "run's file for a system dropped from --systems is kept",
                ),
                PlannedOutput(
                    viz_dir(run_root),
                    OUTPUT_EXTEND,
                    note="each named system/layer chart is overwritten; a prior "
                    "run's chart for a system dropped from --systems is kept",
                ),
            ],
        )
        return 0

    paths = run_noise_layer_analysis(
        roots,
        systems=systems,
        run_date=args.run_date,
        zipf_run_date=args.zipf_run_date,
        cleanse_tier=cleanse_tier,
        extra_noise_word_paths=extra_noise_words,
        progress=lambda message: print(f"[info] {message}"),
    )

    print(f"Metrics: {paths.metrics_dir}")
    print(f"Plots: {paths.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
