from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import add_root_arg, run_reporting_argument_errors
from company_tokenize import (
    DEFAULT_MAX_NOISE_WORD_CANDIDATES,
    build_noise_word_candidates_payload,
    resolve_tokenizer_paths,
    scope_directory_files,
    validate_noise_word_candidates_payload,
)

from workspace.artifact_layout import tokenizer_artifact_root
from workspace.roots import WorkspaceRoots, default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a bounded, scored (not pre-thresholded) per-system noise-word "
            "candidate artifact from the persisted corpus-token TF-IDF stats "
            "(token_tfidf_stats.parquet, produced by scripts/generate_noise_words.py). "
            "Keeps the continuous document_frequency_pct/idf score for the N lowest-IDF "
            "candidates so a downstream consumer (for example company_cleanse's "
            "short-name derivation) can apply its own cutoff, rather than "
            "inheriting one of generate_noise_words.py's fixed strict/balanced/aggressive "
            "thresholds."
        )
    )
    parser.add_argument(
        "--system",
        required=True,
        help="System code to target (for example 'ie' or 'global').",
    )
    parser.add_argument(
        "--profile",
        default="auto",
        help="Global tokenizer profile when --system global (default: auto).",
    )
    add_root_arg(parser, help_text="Project root containing artifacts/tokenizers.")
    parser.add_argument(
        "--stats-path",
        default=None,
        help=(
            "Optional explicit token_tfidf_stats.parquet path. Defaults to "
            "artifacts/tokenizers/work/<system>/token_tfidf_stats.parquet -- the same "
            "location generate_noise_words.py writes to. This script reads that "
            "already-computed artifact rather than recomputing TF-IDF from the raw "
            "corpus; run generate_noise_words.py first if it does not exist yet."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Optional explicit output path for the candidate JSON artifact. Defaults to "
            "artifacts/tokenizers/work/<system>/noise_word_candidates.json."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_NOISE_WORD_CANDIDATES,
        help=(
            "Maximum number of lowest-IDF candidates to export (default "
            f"{DEFAULT_MAX_NOISE_WORD_CANDIDATES}, chosen from where the measured "
            "idf-per-rank slope flattens across ie/gb/fr/global/offeneregister -- see "
            "company_tokenize.noise_export's module docstring for the analysis)."
        ),
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=2,
        help="Minimum token length for candidates (default 2).",
    )
    return parser


def _resolve_stats_path(
    *, roots: WorkspaceRoots, system: str, profile: str, stats_path: str | None
) -> Path:
    # Absolute without following links, so a path through the `artifacts/`
    # junction stays under the project root and is recorded relative to it.
    if stats_path:
        return Path(os.path.abspath(stats_path))
    tokenizer_root = tokenizer_artifact_root(roots)
    if system == "global":
        base = resolve_tokenizer_paths(
            tokenizer_root=tokenizer_root, scope="global", profile=profile
        ).directory
    else:
        base = resolve_tokenizer_paths(
            tokenizer_root=tokenizer_root, scope="country", system=system
        ).directory
    return scope_directory_files(base).tfidf_stats


def _resolve_out_path(
    *, roots: WorkspaceRoots, system: str, profile: str, out: str | None
) -> Path:
    if out:
        return Path(out).resolve()
    tokenizer_root = tokenizer_artifact_root(roots)
    if system == "global":
        base = resolve_tokenizer_paths(
            tokenizer_root=tokenizer_root, scope="global", profile=profile
        ).directory
    else:
        base = resolve_tokenizer_paths(
            tokenizer_root=tokenizer_root, scope="country", system=system
        ).directory
    return scope_directory_files(base).noise_word_candidates


def _recorded_path(path: Path, *, project_root: Path) -> str:
    """The path as the artifact records it: relative to the project root when under it."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def run_export(
    *,
    system: str,
    profile: str,
    root: str | Path,
    stats_path: str | None,
    out: str | None,
    max_candidates: int,
    min_token_length: int,
) -> int:
    normalized_system = system.strip().lower()
    project_root = Path(root).resolve()
    roots = default_workspace_roots(project_root)

    resolved_stats_path = _resolve_stats_path(
        roots=roots,
        system=normalized_system,
        profile=profile,
        stats_path=stats_path,
    )
    resolved_out_path = _resolve_out_path(
        roots=roots,
        system=normalized_system,
        profile=profile,
        out=out,
    )

    if not resolved_stats_path.exists():
        raise FileNotFoundError(
            f"Token TF-IDF stats parquet not found: {resolved_stats_path}. "
            "Run scripts/generate_noise_words.py for this system first."
        )

    stats = pl.read_parquet(resolved_stats_path)
    payload = build_noise_word_candidates_payload(
        stats,
        system=normalized_system,
        source_stats_path=_recorded_path(
            resolved_stats_path, project_root=project_root
        ),
        generated_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_candidates=max_candidates,
        min_token_length=min_token_length,
    )
    validate_noise_word_candidates_payload(payload, source_label=str(resolved_out_path))

    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    selection = payload["selection"]
    print(f"[info] source stats: {resolved_stats_path}")
    print(f"[info] noise word candidates: {resolved_out_path}")
    print(
        f"[info] candidates={selection['candidate_count']:,} "
        f"(max={selection['max_candidates']:,}, "
        f"source_token_count={selection['source_token_count']:,}, "
        f"idf_cutoff={selection['idf_cutoff']}, "
        f"document_frequency_pct_cutoff={selection['document_frequency_pct_cutoff']})"
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_export(
        system=args.system,
        profile=args.profile,
        root=args.root,
        stats_path=args.stats_path,
        out=args.out,
        max_candidates=args.max_candidates,
        min_token_length=args.min_token_length,
    )


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
