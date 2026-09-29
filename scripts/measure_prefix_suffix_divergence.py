"""Measure how often real matched pairs are prefix/suffix divergent.

Read-only analysis script, not a build task. Sibling to
`scripts/measure_initialism_recall.py`, the initialism-shape measurement --
reuses that script's tokenization and matched-pair-loading helpers directly
rather than reimplementing them. Measures, against the same real `matched/`
ground-truth layer (produced by `src/acquisition/match_ops.py`) and the same
real GLEIF matched-pair population that script uses, how often a matched
source/target pair's cleansed names differ only by a *bounded run of
leading or trailing tokens* -- e.g. "cisco systems inc" (as
`cisco systems`, post company-type stripping) vs. "cisco": one side's
token sequence is a strict prefix or suffix of the other's, differing by
1-2 tokens. This is `docs/architecture/packages.md`'s short-name/
divergent-name "category 2" pattern, distinct from the zero-overlap
initialism pattern and from the structurally-unsolvable unrelated-brand
pattern (category 3). This script does not build, index, or wire any
candidate-generation code; it only measures.

Usage:
    .venv/Scripts/python.exe scripts/measure_prefix_suffix_divergence.py \
        --source gleif
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
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
from measure_initialism_recall import (
    _discover_jurisdictions,
    _resolve_target_system,
    _tokenize,
)

from analysis._cli_helper import SETTINGS as ANALYSIS_SETTINGS
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import layer_partition_dir, resolve_partition_dir
from workspace.roots import WorkspaceRoots

DEFAULT_MAX_DIVERGENCE = 2

_DECLARATIONS = declared_settings(ANALYSIS_SETTINGS)
_SURFACE: dict[str, dict[str, object]] = {
    "max_divergence": {
        "flag": "--max-divergence",
        "default": DEFAULT_MAX_DIVERGENCE,
    },
}


def _prefix_suffix_divergence_kind(
    source_tokens: tuple[str, ...],
    target_tokens: tuple[str, ...],
    *,
    max_divergence: int = DEFAULT_MAX_DIVERGENCE,
) -> str | None:
    """Classify a pair as "prefix", "suffix", or not divergent (`None`).

    A pair is prefix/suffix divergent when the shorter token sequence is a
    contiguous run at the start (prefix) or end (suffix) of the longer
    one, and the two sequences differ in length by at most
    `max_divergence` tokens -- the "bounded run" this measurement counts
    (one side has 1-2 extra leading/trailing tokens, e.g. a dropped/added
    legal suffix or qualifier, not an unrelated name). Identical sequences
    are not divergent by this definition; token-count divergence beyond the
    bound is treated as unrelated-name noise, not this pattern, following
    the initialism measurement's precedent of a bounded structural check
    rather than fuzzy similarity.
    """
    if not source_tokens or not target_tokens:
        return None
    if source_tokens == target_tokens:
        return None
    if len(source_tokens) <= len(target_tokens):
        shorter, longer = source_tokens, target_tokens
    else:
        shorter, longer = target_tokens, source_tokens
    diff = len(longer) - len(shorter)
    if diff == 0 or diff > max_divergence:
        return None
    if not shorter:
        return None
    if longer[: len(shorter)] == shorter:
        return "prefix"
    if longer[len(longer) - len(shorter) :] == shorter:
        return "suffix"
    return None


@dataclass
class JurisdictionMeasurement:
    jurisdiction: str
    target_system: str | None
    matched_pairs: int
    pairs_with_both_names: int
    exact_match_pairs: int
    prefix_shaped_pairs: int
    suffix_shaped_pairs: int
    sample_pairs: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def divergent_pairs(self) -> int:
        return self.prefix_shaped_pairs + self.suffix_shaped_pairs

    @property
    def divergent_share_of_matched(self) -> float:
        if self.pairs_with_both_names == 0:
            return 0.0
        return self.divergent_pairs / self.pairs_with_both_names

    def as_dict(self) -> dict[str, object]:
        return {
            "jurisdiction": self.jurisdiction,
            "target_system": self.target_system,
            "matched_pairs": self.matched_pairs,
            "pairs_with_both_names": self.pairs_with_both_names,
            "exact_match_pairs": self.exact_match_pairs,
            "prefix_shaped_pairs": self.prefix_shaped_pairs,
            "suffix_shaped_pairs": self.suffix_shaped_pairs,
            "divergent_pairs": self.divergent_pairs,
            "divergent_share_of_matched": round(self.divergent_share_of_matched, 6),
            "sample_pairs": self.sample_pairs,
        }


def measure_jurisdiction(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    jurisdiction: str,
    max_divergence: int = DEFAULT_MAX_DIVERGENCE,
    sample_size: int = 10,
) -> JurisdictionMeasurement:
    matched_layer_dir = system_layer_dir(roots, source_system, layer=MATCHED_LAYER_NAME)
    matched_partition = resolve_partition_dir(
        matched_layer_dir, value=jurisdiction
    ) or layer_partition_dir(matched_layer_dir, value=jurisdiction)
    target_system = _resolve_target_system(matched_partition)
    if target_system is None:
        return JurisdictionMeasurement(
            jurisdiction=jurisdiction,
            target_system=None,
            matched_pairs=0,
            pairs_with_both_names=0,
            exact_match_pairs=0,
            prefix_shaped_pairs=0,
            suffix_shaped_pairs=0,
        )

    target_layer_dir = system_layer_dir(roots, target_system, layer=CLEANSED_LAYER_NAME)
    target_partition = resolve_partition_dir(
        target_layer_dir, value=jurisdiction
    ) or layer_partition_dir(target_layer_dir, value=jurisdiction)
    target_files = sorted(target_partition.glob("*.parquet"))
    if not target_files:
        raise FileNotFoundError(
            f"Resolved target system '{target_system}' for jurisdiction "
            f"'{jurisdiction}' but found no cleansed parquet files at "
            f"{target_partition}."
        )

    # `matched/` is canonical's row with `match_uri` filled in and carries no
    # cleansed name, so the source's forms are read from its own cleansed
    # layer by `system_uri`.
    source_cleansed_dir = system_layer_dir(
        roots, source_system, layer=CLEANSED_LAYER_NAME
    )
    source_cleansed_partition = resolve_partition_dir(
        source_cleansed_dir, value=jurisdiction
    ) or layer_partition_dir(source_cleansed_dir, value=jurisdiction)
    source_cleansed_files = sorted(source_cleansed_partition.glob("*.parquet"))
    if not source_cleansed_files:
        raise FileNotFoundError(
            f"Source system '{source_system}' has no cleansed parquet files for "
            f"jurisdiction '{jurisdiction}' at {source_cleansed_partition}."
        )
    source_names_lf = pl.scan_parquet([str(f) for f in source_cleansed_files]).select(
        [
            pl.col("system_uri"),
            pl.coalesce([pl.col("short_name"), pl.col("name_cleansed")]).alias(
                "source_name"
            ),
        ]
    )
    source_files = sorted(matched_partition.glob("*.parquet"))
    source_lf = (
        pl.scan_parquet([str(f) for f in source_files])
        .filter(pl.col("match_uri").is_not_null())
        .select(["system_uri", "match_uri"])
        .join(source_names_lf, on="system_uri", how="left")
        .select(["match_uri", "source_name"])
    )
    target_lf = pl.scan_parquet([str(f) for f in target_files]).select(
        [
            pl.col("system_uri"),
            pl.coalesce([pl.col("short_name"), pl.col("name_cleansed")]).alias(
                "target_name"
            ),
        ]
    )

    joined = (
        source_lf.join(
            target_lf, left_on="match_uri", right_on="system_uri", how="inner"
        )
        .select(["source_name", "target_name"])
        .collect()
    )

    matched_pairs = joined.height
    pairs_with_both_names = 0
    exact_match_pairs = 0
    prefix_shaped_pairs = 0
    suffix_shaped_pairs = 0
    sample_pairs: list[tuple[str, str, str]] = []

    for source_name, target_name in zip(
        joined["source_name"].to_list(), joined["target_name"].to_list()
    ):
        source_tokens = _tokenize(source_name)
        target_tokens = _tokenize(target_name)
        if not source_tokens or not target_tokens:
            continue
        pairs_with_both_names += 1
        if source_tokens == target_tokens:
            exact_match_pairs += 1
            continue
        kind = _prefix_suffix_divergence_kind(
            source_tokens, target_tokens, max_divergence=max_divergence
        )
        if kind == "prefix":
            prefix_shaped_pairs += 1
        elif kind == "suffix":
            suffix_shaped_pairs += 1
        else:
            continue
        if len(sample_pairs) < sample_size:
            sample_pairs.append((source_name, target_name, kind))

    return JurisdictionMeasurement(
        jurisdiction=jurisdiction,
        target_system=target_system,
        matched_pairs=matched_pairs,
        pairs_with_both_names=pairs_with_both_names,
        exact_match_pairs=exact_match_pairs,
        prefix_shaped_pairs=prefix_shaped_pairs,
        suffix_shaped_pairs=suffix_shaped_pairs,
        sample_pairs=sample_pairs,
    )


def measure_source_system(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    jurisdictions: list[str] | None = None,
    max_divergence: int = DEFAULT_MAX_DIVERGENCE,
) -> list[JurisdictionMeasurement]:
    matched_dir = system_layer_dir(roots, source_system, layer=MATCHED_LAYER_NAME)
    if not matched_dir.exists():
        raise FileNotFoundError(f"No matched/ directory found at {matched_dir}.")

    if jurisdictions is None:
        jurisdictions = _discover_jurisdictions(matched_dir)

    return [
        measure_jurisdiction(
            roots=roots,
            source_system=source_system,
            jurisdiction=jurisdiction,
            max_divergence=max_divergence,
        )
        for jurisdiction in jurisdictions
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how often real matched/ ground-truth pairs are "
            "prefix/suffix divergent (one side's token sequence is a "
            "strict prefix or suffix of the other's, differing by a "
            "bounded 1-2 token run). Read-only measurement; "
            "does not build any candidate-generation code."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--source",
        dest="source_system",
        default="gleif",
        help="Source system whose matched/ layer to measure (default: gleif).",
    )
    parser.add_argument(
        "--jurisdictions",
        default=None,
        help=(
            "Optional comma-separated jurisdiction codes to restrict to. "
            "Defaults to every jurisdiction_code=* partition under "
            "data/<source-system>/matched/."
        ),
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Optional path to write the full measurement as JSON, absolute "
            "or relative to the checkout."
        ),
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Report the resolved source system, jurisdictions, settings and "
            "output path, without reading any matched/cleansed data or "
            "writing anything."
        ),
    )
    return parser


def _resolve_json_out(roots: WorkspaceRoots, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (roots.checkout / path).resolve()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    roots = resolve_workspace_roots_from_args(args)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    max_divergence = resolved_setting_values(resolved)["max_divergence"]
    json_out = _resolve_json_out(roots, args.json_out)

    jurisdictions = (
        [j.strip() for j in args.jurisdictions.split(",") if j.strip()]
        if args.jurisdictions
        else None
    )

    if args.dry_run:
        report_resolved_settings(
            "measure_prefix_suffix_divergence",
            resolved,
            source_system=args.source_system,
            jurisdictions=jurisdictions or "discovered",
        )
        if json_out is not None:
            report_output_plan(
                "[dry-run] measure_prefix_suffix_divergence:",
                [PlannedOutput(json_out, OUTPUT_CLEAR)],
            )
        return 0

    results = measure_source_system(
        roots=roots,
        source_system=args.source_system,
        jurisdictions=jurisdictions,
        max_divergence=max_divergence,
    )

    total_matched = sum(r.matched_pairs for r in results)
    total_with_names = sum(r.pairs_with_both_names for r in results)
    total_exact = sum(r.exact_match_pairs for r in results)
    total_prefix = sum(r.prefix_shaped_pairs for r in results)
    total_suffix = sum(r.suffix_shaped_pairs for r in results)
    total_divergent = total_prefix + total_suffix

    print(f"source_system={args.source_system} max_divergence={max_divergence}")
    print(
        f"{'jurisdiction':<14}{'target':<18}{'matched':>10}{'named':>10}"
        f"{'exact':>10}{'prefix':>10}{'suffix':>10}{'%_of_named':>12}"
    )
    for r in results:
        pct = r.divergent_share_of_matched * 100
        print(
            f"{r.jurisdiction:<14}{r.target_system!s:<18}{r.matched_pairs:>10}"
            f"{r.pairs_with_both_names:>10}{r.exact_match_pairs:>10}"
            f"{r.prefix_shaped_pairs:>10}{r.suffix_shaped_pairs:>10}"
            f"{pct:>11.4f}%"
        )

    overall_pct = (
        (total_divergent / total_with_names * 100) if total_with_names else 0.0
    )
    print("-" * 96)
    print(
        f"TOTAL matched={total_matched} named_pairs={total_with_names} "
        f"exact_match={total_exact} prefix_shaped={total_prefix} "
        f"suffix_shaped={total_suffix} divergent={total_divergent} "
        f"({overall_pct:.4f}% of named)"
    )

    sample_shown = 0
    for r in results:
        for source_name, target_name, kind in r.sample_pairs:
            if sample_shown >= 15:
                break
            print(
                f"  sample [{r.jurisdiction}/{kind}]: {source_name!r} <-> "
                f"{target_name!r}"
            )
            sample_shown += 1

    if json_out is not None:
        payload = {
            "source_system": args.source_system,
            "max_divergence": max_divergence,
            "jurisdictions": [r.as_dict() for r in results],
            "total_matched_pairs": total_matched,
            "total_pairs_with_both_names": total_with_names,
            "total_exact_match_pairs": total_exact,
            "total_prefix_shaped_pairs": total_prefix,
            "total_suffix_shaped_pairs": total_suffix,
            "total_divergent_pairs": total_divergent,
            "divergent_share_of_named_pct": round(overall_pct, 6),
        }
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote full measurement to {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
