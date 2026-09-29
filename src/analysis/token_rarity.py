"""The rare end of each system's corpus TF-IDF distribution, set against general-language word frequency.

Compares the low-frequency corpus vocabulary with `wordfreq` to measure how much of it a standard tokenizer would fail to represent. Run through `scripts/analyze_token_rarity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
from company_tokenize.tfidf import select_rare_token_candidates
from wordfreq import zipf_frequency

from analysis.report_layout import metrics_dir as _metrics_dir
from analysis.report_layout import reports_dir as _reports_dir
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

# System codes not listed here (for example 'global', which spans every
# system's language at once) have no single reference language and are
# excluded from the wordfreq comparison.
# Single-jurisdiction systems only. `gleif` and `global` are cross-jurisdiction and
# are deliberately absent: there is no one language to score them against, so
# `resolve_system_language()` returns None and the wordfreq columns come back empty
# rather than wrong. Do not "fix" that by mapping them to `en` -- it would populate
# every OOV figure with a number computed against the wrong reference, which reads
# as a result instead of an absence. Scoring a cross-jurisdiction corpus means
# resolving a language per row from `jurisdiction_code`, which is separate work.
SYSTEM_LANGUAGES: dict[str, str] = {
    "gb": "en",
    "ie": "en",
    "fr": "fr",
    "dk": "da",
    "ee": "et",
    "fi": "fi",
    "offeneregister": "de",
}


@dataclass(frozen=True)
class TokenRarityPaths:
    run_root: Path
    stats_dir: Path
    reports_dir: Path
    report_path: Path
    summary_path: Path


def resolve_token_rarity_paths(
    roots: WorkspaceRoots, run_date: str
) -> TokenRarityPaths:
    run_root = analysis_report_run_dir(roots, "token_rarity", run_date)
    stats_dir = _metrics_dir(run_root)
    reports_dir = _reports_dir(run_root)
    stats_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return TokenRarityPaths(
        run_root=run_root,
        stats_dir=stats_dir,
        reports_dir=reports_dir,
        report_path=reports_dir / "token_rarity_summary.md",
        summary_path=stats_dir / "token_rarity_summary.parquet",
    )


def resolve_system_language(system: str) -> str | None:
    return SYSTEM_LANGUAGES.get(system.strip().lower())


def _mean_or_zero(series: pl.Series) -> float:
    return float(cast("float | None", series.mean()) or 0.0)


def attach_language_frequency(
    token_stats: pl.DataFrame,
    *,
    language: str,
) -> pl.DataFrame:
    """Attach wordfreq Zipf-frequency stats to each token for one language.

    ``zipf_frequency`` is 0.0 for tokens wordfreq has never observed for the
    language -- itself further evidence a token is unusual, not just a missing
    lookup, so it deliberately isn't special-cased out of the distribution.
    """
    if "token" not in token_stats.columns:
        raise ValueError("token_stats is missing required column: token")

    tokens = token_stats.get_column("token").to_list()
    zipf_values = [zipf_frequency(str(token).lower(), language) for token in tokens]
    return token_stats.with_columns(
        [
            pl.Series("zipf_frequency", zipf_values, dtype=pl.Float64),
            pl.Series(
                "unseen_in_wordfreq",
                [value == 0.0 for value in zipf_values],
                dtype=pl.Boolean,
            ),
        ]
    )


def summarize_language_rarity(
    stats_with_zipf: pl.DataFrame,
    *,
    system: str,
    language: str,
    rare_max_document_frequency: int | None,
    rare_max_document_frequency_pct: float | None,
) -> dict[str, object]:
    required_columns = {
        "token",
        "document_frequency",
        "document_frequency_pct",
        "zipf_frequency",
        "unseen_in_wordfreq",
    }
    missing = sorted(required_columns - set(stats_with_zipf.columns))
    if missing:
        raise ValueError(
            f"stats_with_zipf is missing required columns: {', '.join(missing)}"
        )

    total_tokens = stats_with_zipf.height
    if total_tokens == 0:
        return {
            "system": system,
            "language": language,
            "total_tokens": 0,
            "unseen_in_wordfreq_pct": 0.0,
            "mean_zipf_frequency": 0.0,
            "rare_tokens": 0,
            "rare_unseen_in_wordfreq_pct": 0.0,
            "rare_mean_zipf_frequency": 0.0,
        }

    unseen_pct = float(stats_with_zipf.get_column("unseen_in_wordfreq").sum()) / float(
        total_tokens
    )
    mean_zipf = _mean_or_zero(stats_with_zipf.get_column("zipf_frequency"))

    rare_frame = select_rare_token_candidates(
        stats_with_zipf,
        max_document_frequency=rare_max_document_frequency,
        max_document_frequency_pct=rare_max_document_frequency_pct,
    )
    rare_total = rare_frame.height
    if rare_total == 0:
        rare_unseen_pct = 0.0
        rare_mean_zipf = 0.0
    else:
        rare_unseen_pct = float(
            rare_frame.get_column("unseen_in_wordfreq").sum()
        ) / float(rare_total)
        rare_mean_zipf = _mean_or_zero(rare_frame.get_column("zipf_frequency"))

    return {
        "system": system,
        "language": language,
        "total_tokens": total_tokens,
        "unseen_in_wordfreq_pct": unseen_pct,
        "mean_zipf_frequency": mean_zipf,
        "rare_tokens": rare_total,
        "rare_unseen_in_wordfreq_pct": rare_unseen_pct,
        "rare_mean_zipf_frequency": rare_mean_zipf,
    }
