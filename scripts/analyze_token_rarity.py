"""The rare end of each system's corpus, set against general-language word frequency.

Reads the `token_tfidf_stats.parquet` artefacts `generate_noise_words.py` writes and, for a system
with a mapped reference language, compares its rare tokens with `wordfreq` to measure how much of
the vocabulary a general-text tokenizer would fail to represent. "Rare" defaults to hapax
legomena (`--max-document-frequency 1`). `global` is left out unless `--include-global`, since its
corpus has no single reference language.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
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
from company_tokenize.paths import scope_directory_files
from company_tokenize.tfidf import select_rare_token_candidates

from analysis._cli_helper import SETTINGS as ANALYSIS_SETTINGS
from analysis.report_layout import metrics_dir, reports_dir
from analysis.token_rarity import (
    attach_language_frequency,
    resolve_system_language,
    resolve_token_rarity_paths,
    summarize_language_rarity,
)
from workspace.artifact_layout import (
    analysis_report_run_dir,
    tokenizer_artifact_root,
    tokenizer_scope_dir,
)
from workspace.roots import WorkspaceRoots

_DECLARATIONS = declared_settings(ANALYSIS_SETTINGS)
_SURFACE: dict[str, dict[str, object]] = {
    "max_document_frequency": {"flag": "--max-document-frequency", "default": 1},
    "max_document_frequency_pct": {"flag": "--max-document-frequency-pct"},
    "min_token_length": {"flag": "--min-token-length"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract rare/unusual tokens from generate_noise_words.py TF-IDF stats and, "
            "for systems with a mapped reference language, compare them against general-"
            "language word frequency (wordfreq) to quantify how much of the corpus "
            "vocabulary a standard tokenizer would fail to represent."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--date",
        dest="run_date",
        default=datetime.now(UTC).date().isoformat(),
        help="Run date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help=(
            "Optional comma-separated system codes. If omitted, discovered from "
            "artifacts/tokenizers/*/token_tfidf_stats.parquet ('global' excluded unless "
            "--include-global is set)."
        ),
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help=(
            "Include the 'global' tokenizer corpus (rare-token extraction only; no "
            "language-frequency comparison since it spans multiple languages)."
        ),
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run directory and report the settings and output "
            "paths that would apply, without reading any stats or writing "
            "anything."
        ),
    )
    return parser


def _discover_systems(roots: WorkspaceRoots) -> list[str]:
    tokenizers_dir = tokenizer_artifact_root(roots)
    if not tokenizers_dir.exists():
        return []
    return sorted(
        child.name
        for child in tokenizers_dir.iterdir()
        if child.is_dir() and scope_directory_files(child).tfidf_stats.exists()
    )


def _fmt_pct(value: object) -> str:
    return f"{float(value):.1%}" if isinstance(value, (int, float)) else "n/a"


def _fmt_zipf(value: object) -> str:
    return f"{float(value):.2f}" if isinstance(value, (int, float)) else "n/a"


def _write_report(report_path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Token Rarity Report",
        "",
        (
            "Rare/unusual tokens extracted from corpus TF-IDF stats, compared against "
            "general-language word frequency (wordfreq) where a reference language is "
            "mapped for the system."
        ),
        "",
        "| system | language | total tokens | rare tokens | rare % unseen in wordfreq | all % unseen in wordfreq | rare mean zipf | all mean zipf |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['language'] or 'n/a'} | {row['total_tokens']:,} | "
            f"{row['rare_tokens']:,} | {_fmt_pct(row['rare_unseen_in_wordfreq_pct'])} | "
            f"{_fmt_pct(row['unseen_in_wordfreq_pct'])} | "
            f"{_fmt_zipf(row['rare_mean_zipf_frequency'])} | "
            f"{_fmt_zipf(row['mean_zipf_frequency'])} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    roots: WorkspaceRoots,
    run_date: str,
    systems: list[str] | None,
    include_global: bool,
    max_document_frequency: int | None,
    max_document_frequency_pct: float | None,
    min_token_length: int,
) -> Path:
    paths = resolve_token_rarity_paths(roots, run_date)

    selected_systems = systems or _discover_systems(roots)
    if not include_global:
        selected_systems = [system for system in selected_systems if system != "global"]
    if not selected_systems:
        raise ValueError(
            "No systems with artifacts/tokenizers/work/<system>/token_tfidf_stats.parquet "
            "found. Run generate_noise_words.py first."
        )

    summary_rows: list[dict[str, object]] = []
    for system in selected_systems:
        stats_path = scope_directory_files(
            tokenizer_scope_dir(roots, system=system)
        ).tfidf_stats
        if not stats_path.exists():
            print(
                f"[warn] Skipping '{system}': {stats_path} not found. "
                "Run generate_noise_words.py first."
            )
            continue

        stats = pl.read_parquet(stats_path)
        language = resolve_system_language(system)

        if language is None:
            rare = select_rare_token_candidates(
                stats,
                max_document_frequency=max_document_frequency,
                max_document_frequency_pct=max_document_frequency_pct,
                min_token_length=min_token_length,
            )
            rare.write_csv(paths.stats_dir / f"{system}_rare_tokens.csv")
            summary_rows.append(
                {
                    "system": system,
                    "language": None,
                    "total_tokens": stats.height,
                    "rare_tokens": rare.height,
                    "unseen_in_wordfreq_pct": None,
                    "mean_zipf_frequency": None,
                    "rare_unseen_in_wordfreq_pct": None,
                    "rare_mean_zipf_frequency": None,
                }
            )
            print(
                f"[info] {system}: {rare.height:,}/{stats.height:,} rare tokens "
                "(no mapped language; wordfreq comparison skipped)"
            )
            continue

        stats_with_zipf = attach_language_frequency(stats, language=language)
        stats_with_zipf.write_parquet(
            paths.stats_dir / f"{system}_token_language_stats.parquet"
        )

        rare_with_zipf = select_rare_token_candidates(
            stats_with_zipf,
            max_document_frequency=max_document_frequency,
            max_document_frequency_pct=max_document_frequency_pct,
            min_token_length=min_token_length,
        )
        rare_with_zipf.write_csv(paths.stats_dir / f"{system}_rare_tokens.csv")

        summary = summarize_language_rarity(
            stats_with_zipf,
            system=system,
            language=language,
            rare_max_document_frequency=max_document_frequency,
            rare_max_document_frequency_pct=max_document_frequency_pct,
        )
        summary_rows.append(summary)
        print(
            f"[info] {system} ({language}): {summary['rare_tokens']:,}/"
            f"{summary['total_tokens']:,} rare tokens, "
            f"{_fmt_pct(summary['rare_unseen_in_wordfreq_pct'])} of rare tokens unseen "
            f"in wordfreq (vs {_fmt_pct(summary['unseen_in_wordfreq_pct'])} across the "
            "full vocabulary)"
        )

    summary_df = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()
    summary_df.write_parquet(paths.summary_path)
    _write_report(paths.report_path, summary_rows)

    print(f"Summary: {paths.summary_path}")
    print(f"Report: {paths.report_path}")
    return paths.report_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    roots = resolve_workspace_roots_from_args(args)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    values = resolved_setting_values(resolved)
    systems = (
        [item.strip() for item in args.systems.split(",") if item.strip()]
        if args.systems
        else None
    )

    if args.dry_run:
        run_root = analysis_report_run_dir(roots, "token_rarity", args.run_date)
        report_resolved_settings(
            "analyze_token_rarity",
            resolved,
            systems=systems or "discovered",
            include_global=args.include_global,
        )
        report_output_plan(
            "[dry-run] analyze_token_rarity:",
            [
                PlannedOutput(
                    metrics_dir(run_root),
                    OUTPUT_EXTEND,
                    note="each named system's rare-token/language-stats files "
                    "are overwritten; a prior run's file for a system dropped "
                    "from --systems is kept",
                ),
                PlannedOutput(reports_dir(run_root), OUTPUT_CLEAR),
            ],
        )
        return 0

    run_analysis(
        roots=roots,
        run_date=args.run_date,
        systems=systems,
        include_global=args.include_global,
        max_document_frequency=values["max_document_frequency"],
        max_document_frequency_pct=values["max_document_frequency_pct"],
        min_token_length=values["min_token_length"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
