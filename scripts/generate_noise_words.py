from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import add_root_arg, run_reporting_argument_errors
from company_tokenize import (
    SUPPORTED_POOLING_RULES,
    compute_corpus_token_tfidf_stats,
    cumulative_document_frequency_mass_cutoff,
    expected_tfidf_namespace,
    normalize_pooling_rule,
    pool_token_tfidf_stats,
    resolve_tokenizer_paths,
    scope_directory_files,
    to_portable_path_str,
    validate_tfidf_namespace,
)

from workspace.artifact_layout import tokenizer_artifact_root
from workspace.roots import WorkspaceRoots, default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TF-IDF token artifacts from a tokenizer training corpus and persist both "
            "the full token set and selected noise-word set."
        )
    )
    parser.add_argument(
        "--system",
        default=None,
        help=(
            "System code to target (for example 'ie' or 'global'). Required "
            "unless --systems is given."
        ),
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=(
            "Pool these systems' already-persisted token_tfidf_stats.parquet "
            "files under --pooling instead of computing stats from a single "
            "system's corpus. Mutually exclusive with --system."
        ),
    )
    parser.add_argument(
        "--pooling",
        default=None,
        choices=list(SUPPORTED_POOLING_RULES),
        help="Pooling rule for --systems: 'count' (row-weighted) or 'equal'.",
    )
    parser.add_argument(
        "--profile",
        default="auto",
        help="Global tokenizer profile when --system global (default: auto).",
    )
    add_root_arg(parser, help_text="Project root containing artifacts/tokenizers.")
    parser.add_argument(
        "--corpus-path",
        default=None,
        help=(
            "Optional explicit prepared corpus parquet artifact path (single file). "
            "Overrides scope/profile resolution. Raw shard directories/globs are not accepted."
        ),
    )
    parser.add_argument(
        "--name-col",
        default="name",
        help="Text column in the prepared corpus parquet used for TF-IDF computation.",
    )
    parser.add_argument(
        "--min-document-frequency-pct",
        type=float,
        default=0.001,
        help="Optional prefilter minimum document-frequency percentage before bucket selection.",
    )
    parser.add_argument(
        "--max-idf",
        type=float,
        default=7.0,
        help="Optional prefilter maximum IDF before bucket selection.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=2,
        help="Minimum token length for noise-word candidates.",
    )
    parser.add_argument(
        "--token-stats-out",
        default=None,
        help="Optional parquet output path for full token TF-IDF stats.",
    )
    parser.add_argument(
        "--token-set-out",
        default=None,
        help="Optional JSON output path for the full token set.",
    )
    parser.add_argument(
        "--noise-words-out",
        default=None,
        help="Optional JSON output path for selected noise words.",
    )
    parser.add_argument(
        "--seed-noise-words-path",
        default=None,
        help=(
            "Optional JSON path containing external seed noise words (for example legal-form tokens) "
            "to union with TF-IDF-selected candidates."
        ),
    )
    parser.add_argument(
        "--strict-token-count",
        type=int,
        default=4,
        help="Number of TF-IDF descriptor tokens to keep for strict profile.",
    )
    parser.add_argument(
        "--balanced-token-count",
        type=int,
        default=10,
        help="Number of TF-IDF descriptor tokens to keep for balanced profile.",
    )
    parser.add_argument(
        "--aggressive-token-count",
        type=int,
        default=25,
        help="Number of TF-IDF descriptor tokens to keep for aggressive profile.",
    )
    parser.add_argument(
        "--strict-cumulative-df-mass",
        type=float,
        default=0.25,
        help=(
            "Target cumulative DF mass for strict profile selection. "
            "Threshold-matching tokens are included until this mass is reached."
        ),
    )
    parser.add_argument(
        "--balanced-cumulative-df-mass",
        type=float,
        default=0.50,
        help=(
            "Target cumulative DF mass for balanced profile selection. "
            "Threshold-matching tokens are included until this mass is reached."
        ),
    )
    parser.add_argument(
        "--aggressive-cumulative-df-mass",
        type=float,
        default=1.00,
        help=(
            "Target cumulative DF mass for aggressive profile selection. "
            "Threshold-matching tokens are included until this mass is reached."
        ),
    )
    parser.add_argument(
        "--strict-min-document-frequency-pct",
        type=float,
        default=None,
        help="Optional strict-bucket DF%% override; defaults to a stricter derived value.",
    )
    parser.add_argument(
        "--strict-max-idf",
        type=float,
        default=None,
        help="Optional strict-bucket max IDF override; defaults to a stricter derived value.",
    )
    parser.add_argument(
        "--balanced-min-document-frequency-pct",
        type=float,
        default=None,
        help="Optional balanced-bucket DF%% override; defaults to a moderately strict derived value.",
    )
    parser.add_argument(
        "--balanced-max-idf",
        type=float,
        default=None,
        help="Optional balanced-bucket max IDF override; defaults to a moderately strict derived value.",
    )
    parser.add_argument(
        "--aggressive-min-document-frequency-pct",
        type=float,
        default=None,
        help="Optional aggressive-bucket DF%% override; defaults to global prefilter DF%%.",
    )
    parser.add_argument(
        "--aggressive-max-idf",
        type=float,
        default=None,
        help="Optional aggressive-bucket max IDF override; defaults to global prefilter max IDF.",
    )
    return parser


def _load_seed_noise_words(seed_noise_words_path: str | None) -> set[str]:
    if not seed_noise_words_path:
        return set()

    path = Path(seed_noise_words_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("seed noise words JSON must be an array of strings.")

    normalized: set[str] = set()
    for value in payload:
        if not isinstance(value, str):
            raise TypeError("seed noise words JSON must contain only strings.")
        token = value.strip().lower()
        if token:
            normalized.add(token)
    return normalized


def _resolve_output_paths(
    *,
    roots: WorkspaceRoots,
    system: str,
    profile: str,
    token_stats_out: str | None,
    token_set_out: str | None,
    noise_words_out: str | None,
) -> tuple[Path, Path, Path, Path]:
    tokenizer_root = tokenizer_artifact_root(roots)
    if system == "global":
        base = resolve_tokenizer_paths(
            tokenizer_root=tokenizer_root, scope="global", profile=profile
        ).directory
    else:
        base = resolve_tokenizer_paths(
            tokenizer_root=tokenizer_root, scope="country", system=system
        ).directory
    scope_files = scope_directory_files(base)

    token_stats_path = (
        Path(token_stats_out).resolve() if token_stats_out else scope_files.tfidf_stats
    )
    token_stats_csv_path = token_stats_path.with_suffix(".csv")
    token_set_path = (
        Path(token_set_out).resolve() if token_set_out else scope_files.token_set
    )
    noise_words_path = (
        Path(noise_words_out).resolve() if noise_words_out else scope_files.noise_words
    )
    return token_stats_path, token_stats_csv_path, token_set_path, noise_words_path


def _resolve_corpus_path(
    *, roots: WorkspaceRoots, system: str, profile: str, corpus_path: str | None
) -> Path:
    if corpus_path:
        return Path(corpus_path).resolve()
    tokenizer_root = tokenizer_artifact_root(roots)
    if system == "global":
        return resolve_tokenizer_paths(
            tokenizer_root=tokenizer_root, scope="global", profile=profile
        ).corpus_path
    return resolve_tokenizer_paths(
        tokenizer_root=tokenizer_root, scope="country", system=system
    ).corpus_path


def _profile_token_limits(
    *,
    strict_token_count: int,
    balanced_token_count: int,
    aggressive_token_count: int,
) -> dict[str, int]:
    return {
        "strict": max(0, int(strict_token_count)),
        "balanced": max(0, int(balanced_token_count)),
        "aggressive": max(0, int(aggressive_token_count)),
    }


def _profile_cumulative_df_mass_limits(
    *,
    strict_cumulative_df_mass: float,
    balanced_cumulative_df_mass: float,
    aggressive_cumulative_df_mass: float,
) -> dict[str, float]:
    def clamp(value: float) -> float:
        return max(0.0, float(value))

    return {
        "strict": clamp(strict_cumulative_df_mass),
        "balanced": clamp(balanced_cumulative_df_mass),
        "aggressive": clamp(aggressive_cumulative_df_mass),
    }


def _select_until_cumulative_df_mass(
    *,
    tokens: list[str],
    token_metrics: dict[str, tuple[float, float]],
    cumulative_df_mass_limit: float,
) -> tuple[list[str], float]:
    kept_tokens = [token for token in tokens if token in token_metrics]
    pct_by_rank = [token_metrics[token][0] for token in kept_tokens]
    cutoff = cumulative_document_frequency_mass_cutoff(
        pct_by_rank, cumulative_df_mass_limit=cumulative_df_mass_limit
    )
    selected = kept_tokens[:cutoff]
    cumulative_df_mass = sum(token_metrics[token][0] for token in selected)
    return selected, cumulative_df_mass


def _resolve_profile_thresholds(
    *,
    min_document_frequency_pct: float,
    max_idf: float,
    strict_min_document_frequency_pct: float | None,
    strict_max_idf: float | None,
    balanced_min_document_frequency_pct: float | None,
    balanced_max_idf: float | None,
    aggressive_min_document_frequency_pct: float | None,
    aggressive_max_idf: float | None,
) -> dict[str, dict[str, float]]:
    base_df = max(0.0, min(1.0, float(min_document_frequency_pct)))
    base_idf = float(max_idf)

    strict_df = (
        float(strict_min_document_frequency_pct)
        if strict_min_document_frequency_pct is not None
        else max(base_df, 0.005)
    )
    strict_idf = (
        float(strict_max_idf) if strict_max_idf is not None else min(base_idf, 4.5)
    )

    balanced_df = (
        float(balanced_min_document_frequency_pct)
        if balanced_min_document_frequency_pct is not None
        else max(base_df * 0.5, 0.002)
    )
    balanced_idf = (
        float(balanced_max_idf) if balanced_max_idf is not None else min(base_idf, 6.0)
    )

    aggressive_df = (
        float(aggressive_min_document_frequency_pct)
        if aggressive_min_document_frequency_pct is not None
        else max(base_df * 0.25, 0.001)
    )
    aggressive_idf = (
        float(aggressive_max_idf) if aggressive_max_idf is not None else base_idf
    )

    strict_df = max(0.0, min(1.0, strict_df))
    balanced_df = max(0.0, min(1.0, balanced_df))
    aggressive_df = max(0.0, min(1.0, aggressive_df))

    return {
        "strict": {
            "min_document_frequency_pct": strict_df,
            "max_idf": strict_idf,
        },
        "balanced": {
            "min_document_frequency_pct": balanced_df,
            "max_idf": balanced_idf,
        },
        "aggressive": {
            "min_document_frequency_pct": aggressive_df,
            "max_idf": aggressive_idf,
        },
    }


def _select_ranked_descriptor_tokens(
    *,
    stats: pl.DataFrame,
    min_document_frequency_pct: float,
    max_idf: float,
    min_token_length: int,
) -> list[str]:
    if stats.height == 0:
        return []

    filtered = stats.filter(
        (pl.col("token").str.len_chars() >= pl.lit(min_token_length))
        & (pl.col("document_frequency_pct") >= pl.lit(min_document_frequency_pct))
        & (pl.col("idf") <= pl.lit(max_idf))
    ).sort(["document_frequency_pct", "idf", "token"], descending=[True, False, False])

    return filtered.get_column("token").to_list() if filtered.height > 0 else []


def _passes_thresholds(
    *,
    token: str,
    token_metrics: dict[str, tuple[float, float]],
    min_document_frequency_pct: float,
    max_idf: float,
) -> bool:
    metrics = token_metrics.get(token)
    if metrics is None:
        return False
    df_pct, idf = metrics
    return float(df_pct) >= float(min_document_frequency_pct) and float(idf) <= float(
        max_idf
    )


class ProfileSelectionMetrics(TypedDict):
    token_count_target: int
    token_count_actual: int
    token_count_is_minimum: bool
    cumulative_df_mass_target: float
    cumulative_df_mass_actual: float
    threshold_match_count: int
    prefilter_min_document_frequency_pct: float
    prefilter_max_idf: float
    min_token_length: int
    bucket_doc_frequency_pct_cutoff: float | None
    bucket_doc_frequency_pct_effective_cutoff: float
    bucket_idf_cutoff: float | None
    cumulative_doc_frequency_pct_cutoff: float | None
    cumulative_doc_frequency_pct_effective_cutoff: float
    cumulative_idf_cutoff: float | None


class ProfilePayload(TypedDict):
    selection: ProfileSelectionMetrics
    seed_legal: list[str]
    tokens: list[str]


def _compute_cutoffs(
    *,
    token_sequence: list[str],
    token_metrics: dict[str, tuple[float, float]],
) -> dict[str, float | None]:
    if not token_sequence:
        return {
            "doc_frequency_pct_cutoff": None,
            "idf_cutoff": None,
        }

    df_values = [
        token_metrics[token][0] for token in token_sequence if token in token_metrics
    ]
    idf_values = [
        token_metrics[token][1] for token in token_sequence if token in token_metrics
    ]
    if not df_values or not idf_values:
        return {
            "doc_frequency_pct_cutoff": None,
            "idf_cutoff": None,
        }

    return {
        "doc_frequency_pct_cutoff": float(min(df_values)),
        "idf_cutoff": float(max(idf_values)),
    }


def _build_noise_word_profile_payloads(
    stats: pl.DataFrame,
    *,
    seed_noise_words: set[str],
    min_document_frequency_pct: float,
    max_idf: float,
    min_token_length: int,
    strict_token_count: int,
    balanced_token_count: int,
    aggressive_token_count: int,
    strict_cumulative_df_mass: float,
    balanced_cumulative_df_mass: float,
    aggressive_cumulative_df_mass: float,
    strict_min_document_frequency_pct: float | None,
    strict_max_idf: float | None,
    balanced_min_document_frequency_pct: float | None,
    balanced_max_idf: float | None,
    aggressive_min_document_frequency_pct: float | None,
    aggressive_max_idf: float | None,
) -> dict[str, ProfilePayload]:
    """Run the threshold and cumulative-mass profile selection over `stats`.

    Shared by `run_generation` (stats freshly computed from one system's own
    corpus) and `run_pooled_generation` (stats pooled from several systems'
    persisted TF-IDF stats): both hand this the same four-column
    `token`/`document_frequency`/`document_frequency_pct`/`idf` frame, and
    the selection rule itself does not care which produced it.
    """
    profile_limits = _profile_token_limits(
        strict_token_count=strict_token_count,
        balanced_token_count=balanced_token_count,
        aggressive_token_count=aggressive_token_count,
    )
    profile_mass_limits = _profile_cumulative_df_mass_limits(
        strict_cumulative_df_mass=strict_cumulative_df_mass,
        balanced_cumulative_df_mass=balanced_cumulative_df_mass,
        aggressive_cumulative_df_mass=aggressive_cumulative_df_mass,
    )
    profile_thresholds = _resolve_profile_thresholds(
        min_document_frequency_pct=min_document_frequency_pct,
        max_idf=max_idf,
        strict_min_document_frequency_pct=strict_min_document_frequency_pct,
        strict_max_idf=strict_max_idf,
        balanced_min_document_frequency_pct=balanced_min_document_frequency_pct,
        balanced_max_idf=balanced_max_idf,
        aggressive_min_document_frequency_pct=aggressive_min_document_frequency_pct,
        aggressive_max_idf=aggressive_max_idf,
    )
    token_metrics = {
        row["token"]: (float(row["document_frequency_pct"]), float(row["idf"]))
        for row in stats.select(["token", "document_frequency_pct", "idf"]).iter_rows(
            named=True
        )
    }
    ranked_tokens = _select_ranked_descriptor_tokens(
        stats=stats,
        min_document_frequency_pct=float(min_document_frequency_pct),
        max_idf=float(max_idf),
        min_token_length=min_token_length,
    )

    profile_layer_tokens: dict[str, list[str]] = {}
    profile_selection_metrics: dict[str, dict[str, float | int]] = {}
    assigned_tokens: set[str] = set()
    for profile_name in ("strict", "balanced", "aggressive"):
        thresholds = profile_thresholds[profile_name]
        threshold_tokens = [
            token
            for token in ranked_tokens
            if token not in assigned_tokens
            and _passes_thresholds(
                token=token,
                token_metrics=token_metrics,
                min_document_frequency_pct=float(
                    thresholds["min_document_frequency_pct"]
                ),
                max_idf=float(thresholds["max_idf"]),
            )
        ]

        mass_limit = profile_mass_limits[profile_name]
        band_tokens, _ = _select_until_cumulative_df_mass(
            tokens=threshold_tokens,
            token_metrics=token_metrics,
            cumulative_df_mass_limit=mass_limit,
        )
        min_count = profile_limits[profile_name]
        if len(band_tokens) < min_count:
            for token in ranked_tokens:
                if token in assigned_tokens or token in band_tokens:
                    continue
                band_tokens.append(token)
                if len(band_tokens) >= min_count:
                    break

        cumulative_df_mass = float(
            sum(
                float(token_metrics[token][0])
                for token in band_tokens
                if token in token_metrics
            )
        )

        profile_layer_tokens[profile_name] = band_tokens
        profile_selection_metrics[profile_name] = {
            "cumulative_df_mass_target": float(mass_limit),
            "cumulative_df_mass_actual": float(cumulative_df_mass),
            "threshold_match_count": len(threshold_tokens),
        }
        assigned_tokens.update(band_tokens)

    profile_payloads: dict[str, ProfilePayload] = {}
    cumulative_tokens: list[str] = []
    for profile_name in ("strict", "balanced", "aggressive"):
        token_limit = profile_limits[profile_name]
        tokens = profile_layer_tokens[profile_name]
        cumulative_tokens.extend(tokens)
        bucket_cutoffs = _compute_cutoffs(
            token_sequence=tokens, token_metrics=token_metrics
        )
        cumulative_cutoffs = _compute_cutoffs(
            token_sequence=cumulative_tokens, token_metrics=token_metrics
        )
        profile_min_df = float(
            profile_thresholds[profile_name]["min_document_frequency_pct"]
        )
        bucket_df_cutoff = bucket_cutoffs["doc_frequency_pct_cutoff"]
        cumulative_df_cutoff = cumulative_cutoffs["doc_frequency_pct_cutoff"]
        effective_bucket_df_cutoff = (
            min(profile_min_df, float(bucket_df_cutoff))
            if bucket_df_cutoff is not None
            else profile_min_df
        )
        effective_cumulative_df_cutoff = (
            min(profile_min_df, float(cumulative_df_cutoff))
            if cumulative_df_cutoff is not None
            else profile_min_df
        )
        profile_payloads[profile_name] = {
            "selection": {
                "token_count_target": int(token_limit),
                "token_count_actual": len(tokens),
                "token_count_is_minimum": True,
                "cumulative_df_mass_target": float(
                    profile_selection_metrics[profile_name]["cumulative_df_mass_target"]
                ),
                "cumulative_df_mass_actual": float(
                    profile_selection_metrics[profile_name]["cumulative_df_mass_actual"]
                ),
                "threshold_match_count": int(
                    profile_selection_metrics[profile_name]["threshold_match_count"]
                ),
                "prefilter_min_document_frequency_pct": float(
                    profile_thresholds[profile_name]["min_document_frequency_pct"]
                ),
                "prefilter_max_idf": float(profile_thresholds[profile_name]["max_idf"]),
                "min_token_length": int(min_token_length),
                "bucket_doc_frequency_pct_cutoff": bucket_df_cutoff,
                "bucket_doc_frequency_pct_effective_cutoff": effective_bucket_df_cutoff,
                "bucket_idf_cutoff": bucket_cutoffs["idf_cutoff"],
                "cumulative_doc_frequency_pct_cutoff": cumulative_df_cutoff,
                "cumulative_doc_frequency_pct_effective_cutoff": effective_cumulative_df_cutoff,
                "cumulative_idf_cutoff": cumulative_cutoffs["idf_cutoff"],
            },
            "seed_legal": sorted(seed_noise_words),
            "tokens": tokens,
        }

    return profile_payloads


def _print_profile_summary(
    profile_payloads: dict[str, ProfilePayload], seed_noise_words: set[str]
) -> None:
    """One line per profile, each counting the tokens of every profile up to it
    plus the seed words, then the seed count."""
    running_tokens: list[str] = []
    for profile_name in ("strict", "balanced", "aggressive"):
        tokens = profile_payloads[profile_name]["tokens"]
        running_tokens.extend(tokens)
        combined_count = len(seed_noise_words | set(running_tokens))
        selection = profile_payloads[profile_name]["selection"]
        print(
            f"[info] noise words profile={profile_name} target={selection['token_count_target']} combined="
            f"{combined_count:,} "
            f"(seed={len(seed_noise_words):,}, tokens={len(tokens):,}, "
            f"df_cutoff={selection['cumulative_doc_frequency_pct_cutoff']}, "
            f"idf_cutoff={selection['cumulative_idf_cutoff']})"
        )
    if seed_noise_words:
        print(f"[info] seed noise words merged: {len(seed_noise_words):,}")


def run_generation(
    *,
    system: str,
    profile: str,
    root: str | Path,
    corpus_path: str | None,
    name_col: str,
    min_document_frequency_pct: float,
    max_idf: float,
    min_token_length: int,
    token_stats_out: str | None,
    token_set_out: str | None,
    noise_words_out: str | None,
    seed_noise_words_path: str | None,
    strict_token_count: int,
    balanced_token_count: int,
    aggressive_token_count: int,
    strict_cumulative_df_mass: float = 0.25,
    balanced_cumulative_df_mass: float = 0.50,
    aggressive_cumulative_df_mass: float = 1.00,
    strict_min_document_frequency_pct: float | None,
    strict_max_idf: float | None,
    balanced_min_document_frequency_pct: float | None,
    balanced_max_idf: float | None,
    aggressive_min_document_frequency_pct: float | None,
    aggressive_max_idf: float | None,
) -> int:
    normalized_system = system.strip().lower()
    project_root = Path(root).resolve()
    roots = default_workspace_roots(project_root)

    resolved_corpus_path = _resolve_corpus_path(
        roots=roots,
        system=normalized_system,
        profile=profile,
        corpus_path=corpus_path,
    )
    stats_path, stats_csv_path, token_set_path, noise_words_path = (
        _resolve_output_paths(
            roots=roots,
            system=normalized_system,
            profile=profile,
            token_stats_out=token_stats_out,
            token_set_out=token_set_out,
            noise_words_out=noise_words_out,
        )
    )

    stats = compute_corpus_token_tfidf_stats(resolved_corpus_path, name_col=name_col)
    seed_noise_words = _load_seed_noise_words(seed_noise_words_path)
    token_set = sorted(stats.get_column("token").to_list())
    profile_payloads = _build_noise_word_profile_payloads(
        stats,
        seed_noise_words=seed_noise_words,
        min_document_frequency_pct=min_document_frequency_pct,
        max_idf=max_idf,
        min_token_length=min_token_length,
        strict_token_count=strict_token_count,
        balanced_token_count=balanced_token_count,
        aggressive_token_count=aggressive_token_count,
        strict_cumulative_df_mass=strict_cumulative_df_mass,
        balanced_cumulative_df_mass=balanced_cumulative_df_mass,
        aggressive_cumulative_df_mass=aggressive_cumulative_df_mass,
        strict_min_document_frequency_pct=strict_min_document_frequency_pct,
        strict_max_idf=strict_max_idf,
        balanced_min_document_frequency_pct=balanced_min_document_frequency_pct,
        balanced_max_idf=balanced_max_idf,
        aggressive_min_document_frequency_pct=aggressive_min_document_frequency_pct,
        aggressive_max_idf=aggressive_max_idf,
    )

    noise_words_payload = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": normalized_system,
        "source_corpus": str(resolved_corpus_path),
        "tfidf_use_case": "corpus",
        "tfidf_namespace": expected_tfidf_namespace(use_case="corpus"),
        "profiles": profile_payloads,
    }
    validate_tfidf_namespace(
        use_case="corpus",
        namespace=str(noise_words_payload["tfidf_namespace"]),
    )

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    token_set_path.parent.mkdir(parents=True, exist_ok=True)
    noise_words_path.parent.mkdir(parents=True, exist_ok=True)

    stats.write_parquet(stats_path)
    stats.write_csv(stats_csv_path)
    token_set_path.write_text(json.dumps(token_set, indent=2), encoding="utf-8")
    noise_words_path.write_text(
        json.dumps(noise_words_payload, indent=2), encoding="utf-8"
    )

    print(f"[info] corpus: {resolved_corpus_path}")
    print(f"[info] token stats parquet: {stats_path}")
    print(f"[info] token stats csv: {stats_csv_path}")
    print(f"[info] token set: {token_set_path} ({len(token_set):,} tokens)")
    print(f"[info] noise words (profiled): {noise_words_path}")
    _print_profile_summary(profile_payloads, seed_noise_words)
    return 0


def _resolve_pooled_noise_words_out(
    *, roots: WorkspaceRoots, pooling: str, noise_words_out: str | None
) -> Path:
    if noise_words_out:
        return Path(noise_words_out).resolve()
    global_base = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope="global",
        profile="default",
    ).directory
    return scope_directory_files(global_base).pooled_noise_words(pooling)


def _resolve_system_tfidf_stats_path(*, roots: WorkspaceRoots, system: str) -> Path:
    country_base = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope="country",
        system=system,
    ).directory
    return scope_directory_files(country_base).tfidf_stats


def run_pooled_generation(
    *,
    systems: list[str],
    pooling: str,
    root: str | Path,
    min_document_frequency_pct: float,
    max_idf: float,
    min_token_length: int,
    noise_words_out: str | None,
    seed_noise_words_path: str | None,
    strict_token_count: int,
    balanced_token_count: int,
    aggressive_token_count: int,
    strict_cumulative_df_mass: float = 0.25,
    balanced_cumulative_df_mass: float = 0.50,
    aggressive_cumulative_df_mass: float = 1.00,
    strict_min_document_frequency_pct: float | None,
    strict_max_idf: float | None,
    balanced_min_document_frequency_pct: float | None,
    balanced_max_idf: float | None,
    aggressive_min_document_frequency_pct: float | None,
    aggressive_max_idf: float | None,
) -> int:
    """Build a pooled noise-word candidate file from several systems' own stats.

    Unlike `run_generation`, this never rescans a corpus: each system's
    `token_tfidf_stats.parquet` is already persisted, so pooling is a join
    over those files (`pool_token_tfidf_stats`), and the result flows
    through the same `_build_noise_word_profile_payloads` selection
    `run_generation` uses. Written as a rule-named sibling of the global
    directory's `noise_words.json` (`noise_words.<rule>.json`) -- never the
    global default itself, so nothing an existing run loads by default
    changes.
    """
    project_root = Path(root).resolve()
    roots = default_workspace_roots(project_root)
    normalized_pooling = normalize_pooling_rule(pooling)
    normalized_systems = sorted({s.strip().lower() for s in systems if s.strip()})
    if not normalized_systems:
        raise ValueError("--systems requires at least one non-empty system code.")

    stats_paths = {
        system: _resolve_system_tfidf_stats_path(roots=roots, system=system)
        for system in normalized_systems
    }
    stats_by_system: dict[str, pl.DataFrame] = {}
    for system, stats_path in stats_paths.items():
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Persisted TF-IDF stats not found for system '{system}': {stats_path}"
            )
        stats_by_system[system] = pl.read_parquet(stats_path)

    pooled_stats, provenance = pool_token_tfidf_stats(
        stats_by_system, pooling=normalized_pooling
    )
    seed_noise_words = _load_seed_noise_words(seed_noise_words_path)
    profile_payloads = _build_noise_word_profile_payloads(
        pooled_stats,
        seed_noise_words=seed_noise_words,
        min_document_frequency_pct=min_document_frequency_pct,
        max_idf=max_idf,
        min_token_length=min_token_length,
        strict_token_count=strict_token_count,
        balanced_token_count=balanced_token_count,
        aggressive_token_count=aggressive_token_count,
        strict_cumulative_df_mass=strict_cumulative_df_mass,
        balanced_cumulative_df_mass=balanced_cumulative_df_mass,
        aggressive_cumulative_df_mass=aggressive_cumulative_df_mass,
        strict_min_document_frequency_pct=strict_min_document_frequency_pct,
        strict_max_idf=strict_max_idf,
        balanced_min_document_frequency_pct=balanced_min_document_frequency_pct,
        balanced_max_idf=balanced_max_idf,
        aggressive_min_document_frequency_pct=aggressive_min_document_frequency_pct,
        aggressive_max_idf=aggressive_max_idf,
    )

    noise_words_path = _resolve_pooled_noise_words_out(
        roots=roots,
        pooling=normalized_pooling,
        noise_words_out=noise_words_out,
    )
    noise_words_payload = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": "global",
        "pooling_rule": normalized_pooling,
        "pooling_systems": normalized_systems,
        "pooling_weights": provenance["weights"],
        "pooling_document_counts": provenance["document_counts"],
        "pooling_stats_paths": {
            system: to_portable_path_str(path, project_root=project_root)
            for system, path in stats_paths.items()
        },
        "tfidf_use_case": "corpus",
        "tfidf_namespace": expected_tfidf_namespace(use_case="corpus"),
        "profiles": profile_payloads,
    }
    validate_tfidf_namespace(
        use_case="corpus",
        namespace=str(noise_words_payload["tfidf_namespace"]),
    )

    noise_words_path.parent.mkdir(parents=True, exist_ok=True)
    noise_words_path.write_text(
        json.dumps(noise_words_payload, indent=2), encoding="utf-8"
    )

    print(f"[info] pooling rule: {normalized_pooling}")
    print(f"[info] pooling systems: {', '.join(normalized_systems)}")
    print(f"[info] pooling weights: {provenance['weights']}")
    print(f"[info] noise words (pooled, profiled): {noise_words_path}")
    _print_profile_summary(profile_payloads, seed_noise_words)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.systems:
        if args.pooling is None:
            parser.error("--pooling is required when --systems is given.")
        if args.system is not None:
            parser.error("--system and --systems are mutually exclusive.")
        return run_pooled_generation(
            systems=args.systems,
            pooling=args.pooling,
            root=args.root,
            min_document_frequency_pct=args.min_document_frequency_pct,
            max_idf=args.max_idf,
            min_token_length=args.min_token_length,
            noise_words_out=args.noise_words_out,
            seed_noise_words_path=args.seed_noise_words_path,
            strict_token_count=args.strict_token_count,
            balanced_token_count=args.balanced_token_count,
            aggressive_token_count=args.aggressive_token_count,
            strict_cumulative_df_mass=args.strict_cumulative_df_mass,
            balanced_cumulative_df_mass=args.balanced_cumulative_df_mass,
            aggressive_cumulative_df_mass=args.aggressive_cumulative_df_mass,
            strict_min_document_frequency_pct=args.strict_min_document_frequency_pct,
            strict_max_idf=args.strict_max_idf,
            balanced_min_document_frequency_pct=args.balanced_min_document_frequency_pct,
            balanced_max_idf=args.balanced_max_idf,
            aggressive_min_document_frequency_pct=args.aggressive_min_document_frequency_pct,
            aggressive_max_idf=args.aggressive_max_idf,
        )

    if args.system is None:
        parser.error("--system is required unless --systems is given.")

    return run_generation(
        system=args.system,
        profile=args.profile,
        root=args.root,
        corpus_path=args.corpus_path,
        name_col=args.name_col,
        min_document_frequency_pct=args.min_document_frequency_pct,
        max_idf=args.max_idf,
        min_token_length=args.min_token_length,
        token_stats_out=args.token_stats_out,
        token_set_out=args.token_set_out,
        noise_words_out=args.noise_words_out,
        seed_noise_words_path=args.seed_noise_words_path,
        strict_token_count=args.strict_token_count,
        balanced_token_count=args.balanced_token_count,
        aggressive_token_count=args.aggressive_token_count,
        strict_cumulative_df_mass=args.strict_cumulative_df_mass,
        balanced_cumulative_df_mass=args.balanced_cumulative_df_mass,
        aggressive_cumulative_df_mass=args.aggressive_cumulative_df_mass,
        strict_min_document_frequency_pct=args.strict_min_document_frequency_pct,
        strict_max_idf=args.strict_max_idf,
        balanced_min_document_frequency_pct=args.balanced_min_document_frequency_pct,
        balanced_max_idf=args.balanced_max_idf,
        aggressive_min_document_frequency_pct=args.aggressive_min_document_frequency_pct,
        aggressive_max_idf=args.aggressive_max_idf,
    )


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
